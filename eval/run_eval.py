"""Golden-set evaluation harness.

Runs every question in ``eval/golden/questions.jsonl`` through the REAL
``/api/chats/ask`` pipeline (in-process via httpx ASGITransport -- no server
needed) and scores the answers with a local LLM judge.

Isolation: the app is pointed at ``eval/.data`` and Redis DB 1, so a run never
touches the user's documents, chat sessions, or semantic cache. The semantic
cache is flushed at the start of every run, otherwise a second run would just
measure cache hits instead of the pipeline.

Usage:
    python eval/run_eval.py                 # full run
    python eval/run_eval.py --limit 5       # smoke test
    python eval/run_eval.py --label baseline
"""
import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent

# These must be set BEFORE any backend module imports config.
os.environ.setdefault("RAG_DATA_DIR", str(EVAL_DIR / ".data"))
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6379/1?protocol=2")
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "0")  # <=0 disables the middleware

sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

import config  # noqa: E402

GOLDEN_DIR = EVAL_DIR / "golden"
RESULTS_DIR = EVAL_DIR / "results"
HISTORY_PATH = EVAL_DIR / "history.csv"

# The judge should be materially stronger than the model under test; a 0.5b judge
# produces scores close to noise. Override with EVAL_JUDGE_MODEL.
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "llama3.2:latest")

JUDGE_PROMPT = """You are a strict evaluator of a retrieval-augmented answer.

Question:
{question}

Reference answer (ground truth):
{reference}

Retrieved context the system was given:
{context}

Generated answer to evaluate:
{answer}

Score the generated answer with integers from 1 to 5:
- faithfulness: 5 = every claim is supported by the retrieved context; 1 = mostly invented.
- relevance: 5 = directly and completely answers the question; 1 = does not address it.

Also report:
- refuses: true if the answer states the documents do not contain the information, otherwise false.

Reply with ONLY a JSON object of the form:
{{"faithfulness": <1-5>, "relevance": <1-5>, "refuses": <true|false>}}"""


def git_commit() -> str:
    try:
        import subprocess
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, check=True, cwd=ROOT)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def load_questions(limit: int | None) -> list[dict]:
    rows = []
    with open(GOLDEN_DIR / "questions.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def flush_semantic_cache() -> None:
    """Drop cached answers so the run measures generation, not the cache."""
    from redis_store import get_redis
    r = get_redis()
    keys = r.smembers("semantic:cache:keys")
    if keys:
        r.delete(*keys)
    r.delete("semantic:cache:keys")


def ingest_documents() -> list[str]:
    """Ingest the golden corpus. add_source() hash-skips unchanged docs, so
    repeat runs are cheap."""
    from backend.rag import add_source
    source_ids = []
    for path in sorted((GOLDEN_DIR / "documents").iterdir()):
        if not path.is_file():
            continue
        res = add_source(path.name, path.read_text(encoding="utf-8"))
        state = "cached" if res.get("skipped") else f"{res['chunks']} chunks"
        print(f"  {path.name}: {state}")
        source_ids.append(res["id"])
    return source_ids


async def ask(client: httpx.AsyncClient, question: str, source_ids: list[str]) -> dict:
    """Drive one question through /api/chats/ask and parse the NDJSON stream."""
    t0 = time.perf_counter()
    ttft = None
    streamed: list[str] = []
    retrieved: list[dict] = []
    final: dict = {}

    request_id = None
    async with client.stream(
        "POST", "/api/chats/ask",
        json={"question": question, "session_id": None, "source_ids": source_ids},
    ) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.strip():
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if evt.get("request_id"):
                request_id = evt["request_id"]
            # First event carries every retrieved chunk; the final one only the cited.
            if "citations" in evt and not retrieved:
                retrieved = evt["citations"]
            if "text" in evt:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                streamed.append(evt["text"])
            if "answer_text" in evt:
                final = evt

    total = time.perf_counter() - t0
    return {
        "answer": (final.get("answer_text") or "".join(streamed)).strip(),
        "retrieved": retrieved,
        "cited": final.get("citations", []),
        "confidence": final.get("confidence"),
        "cached": final.get("cached", False),
        "request_id": request_id,
        # Wall time only. httpx's ASGITransport runs the app to completion and
        # buffers the body, so client-side "time to first token" is meaningless
        # here -- real TTFT comes from the server-side metrics DB instead.
        "wall_ms": round(total * 1000, 2),
    }


def latency_breakdown(request_ids: list[str]) -> dict:
    """Server-side stage timings the pipeline already records, joined by request id.

    This is where real TTFT comes from -- see the note in ask().
    """
    import sqlite3
    db = Path(config.DATA_DIR) / "metrics.db"
    ids = [r for r in request_ids if r]
    if not db.exists() or not ids:
        return {}
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(ids))
    rows = [dict(r) for r in conn.execute(
        f"SELECT * FROM request_logs WHERE request_id IN ({placeholders})", ids)]
    conn.close()
    if not rows:
        return {}

    stages = ("embed_latency", "retrieve_latency", "rerank_latency",
              "generate_latency", "ttft_latency", "cache_check_latency", "total_latency")
    out = {"n": len(rows)}
    for stage in stages:
        vals = sorted(v for v in (r.get(stage) for r in rows) if v is not None)
        out[stage.replace("_latency", "_p50_ms")] = round(
            vals[len(vals) // 2], 2) if vals else None
    out["rewrite_skipped_rate"] = round(
        sum(1 for r in rows if r.get("rewrite_skipped")) / len(rows), 4)
    return out


async def judge(client: httpx.AsyncClient, entry: dict, result: dict) -> dict:
    context = "\n\n".join(
        f"[{i + 1}] ({c.get('source_name', '?')}) {c.get('snippet', '')}"
        for i, c in enumerate(result["retrieved"])
    ) or "(no context retrieved)"

    prompt = JUDGE_PROMPT.format(
        question=entry["question"],
        reference=entry["reference_answer"],
        context=context[:8000],
        answer=result["answer"] or "(empty)",
    )
    try:
        r = await client.post(
            f"{config.OLLAMA_BASE}/api/generate",
            json={"model": JUDGE_MODEL, "prompt": prompt, "stream": False,
                  "format": "json", "options": {"temperature": 0.0}},
            timeout=180,
        )
        r.raise_for_status()
        data = json.loads(r.json().get("response", "{}"))
        return {
            "faithfulness": max(1, min(5, int(data.get("faithfulness", 3)))),
            "relevance": max(1, min(5, int(data.get("relevance", 3)))),
            "refuses": bool(data.get("refuses", False)),
        }
    except Exception as e:
        print(f"    judge failed ({e}); recording neutral 3/3")
        return {"faithfulness": 3, "relevance": 3, "refuses": False}


def retrieval_metrics(entry: dict, retrieved: list[dict]) -> dict:
    """Rank of the first chunk from the expected source document.

    Only meaningful where the golden entry names a single source doc; multihop
    and global entries span documents and are skipped (None).
    """
    expected = entry.get("source_doc")
    if not expected or expected == "multiple":
        return {"rank": None, "recall_5": None, "recall_10": None, "mrr": None}
    rank = 0
    for i, c in enumerate(retrieved, start=1):
        if c.get("source_name") == expected:
            rank = i
            break
    return {
        "rank": rank,
        "recall_5": 1.0 if 0 < rank <= 5 else 0.0,
        "recall_10": 1.0 if 0 < rank <= 10 else 0.0,
        "mrr": 1.0 / rank if rank else 0.0,
    }


def mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct), len(ordered) - 1)
    return ordered[idx]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    ap.add_argument("--label", default=None, help="also write results/<label>.json")
    args = ap.parse_args()

    questions = load_questions(args.limit)
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Model under test: {config.LLM_MODEL}")
    print(f"Data dir: {config.DATA_DIR}")
    print(f"\nIngesting golden corpus ({len(list((GOLDEN_DIR / 'documents').iterdir()))} documents)...")

    source_ids = ingest_documents()
    flush_semantic_cache()
    print(f"Flushed semantic cache. Running {len(questions)} questions...\n")

    from main import app
    transport = httpx.ASGITransport(app=app)

    results = []
    async with httpx.AsyncClient(transport=transport, base_url="http://eval",
                                 timeout=600) as app_client, \
               httpx.AsyncClient() as ollama_client:
        for i, entry in enumerate(questions, start=1):
            result = await ask(app_client, entry["question"], source_ids)
            scores = await judge(ollama_client, entry, result)
            retr = retrieval_metrics(entry, result["retrieved"])

            oos = entry["category"] == "out_of_scope"
            row = {
                "id": entry["id"],
                "category": entry["category"],
                "question": entry["question"],
                "answer": result["answer"],
                "faithfulness": scores["faithfulness"],
                "relevance": scores["relevance"],
                "refuses": scores["refuses"],
                "refusal_correct": scores["refuses"] if oos else None,
                "has_citation": bool(result["cited"]) and "[" in result["answer"],
                "confidence": result["confidence"],
                "cached": result["cached"],
                "retrieved_count": len(result["retrieved"]),
                **retr,
                "request_id": result["request_id"],
                "wall_ms": result["wall_ms"],
            }
            results.append(row)
            flag = "REFUSED" if scores["refuses"] else f"f={scores['faithfulness']} r={scores['relevance']}"
            print(f"[{i}/{len(questions)}] {entry['id']} {entry['category']:<12} "
                  f"{flag} rank={retr['rank']} {result['wall_ms']:.0f}ms")

    # ---- aggregate -------------------------------------------------------
    scoped = [r for r in results if r["category"] != "out_of_scope"]
    oos_rows = [r for r in results if r["category"] == "out_of_scope"]
    latencies = [r["wall_ms"] for r in results]
    stages = latency_breakdown([r["request_id"] for r in results])

    faith_5 = mean([r["faithfulness"] for r in scoped])
    rel_5 = mean([r["relevance"] for r in scoped])

    by_category = {}
    for cat in ("factoid", "multihop", "global", "out_of_scope"):
        rows = [r for r in results if r["category"] == cat]
        if not rows:
            continue
        # MRR needs a single expected source document; multihop/global span
        # several, so there is nothing to rank against -- report None, not 0.0.
        ranked = [r for r in rows if r["mrr"] is not None]
        oos = cat == "out_of_scope"
        by_category[cat] = {
            "n": len(rows),
            # Faithfulness/relevance do not apply to a correct refusal: the
            # rubric scores answering, and the right answer is not to answer.
            "faithfulness_norm": None if oos else round((mean([r["faithfulness"] for r in rows]) - 1) / 4, 4),
            "relevance_norm": None if oos else round((mean([r["relevance"] for r in rows]) - 1) / 4, 4),
            "refusal_rate": round(mean([1.0 if r["refusal_correct"] else 0.0 for r in rows]), 4) if oos else None,
            "mrr": round(mean([r["mrr"] for r in ranked]), 4) if ranked else None,
            "p50_ms": round(percentile([r["wall_ms"] for r in rows], 0.5), 2),
        }

    summary = {
        "timestamp": datetime.now().isoformat() + "Z",
        "commit": git_commit(),
        "model": config.LLM_MODEL,
        "judge_model": JUDGE_MODEL,
        "questions": len(results),
        # normalized 0-1 scores -- these are what plan.md section 2 targets
        "faithfulness": round((faith_5 - 1) / 4, 4),
        "relevance": round((rel_5 - 1) / 4, 4),
        # raw 1-5 scores -- what the metrics dashboard renders
        "avg_faithfulness": round(faith_5, 2),
        "avg_relevance": round(rel_5, 2),
        "avg_completeness": round(rel_5, 2),  # not scored separately yet
        "citation_rate": round(mean([1.0 if r["has_citation"] else 0.0 for r in scoped]), 4),
        # None (not 0.0) when the run contained no out-of-scope questions, so a
        # partial run cannot be misread as a total refusal failure.
        "refusal_rate": round(mean([1.0 if r["refusal_correct"] else 0.0 for r in oos_rows]), 4) if oos_rows else None,
        "false_refusal_rate": round(mean([1.0 if r["refuses"] else 0.0 for r in scoped]), 4),
        "recall_5": round(mean([r["recall_5"] for r in results]), 4),
        "recall_10": round(mean([r["recall_10"] for r in results]), 4),
        "mrr": round(mean([r["mrr"] for r in results]), 4),
        "p50_latency_ms": round(percentile(latencies, 0.5), 2),
        "p95_latency_ms": round(percentile(latencies, 0.95), 2),
        # Server-side, measured inside the pipeline. The client cannot measure
        # TTFT through ASGITransport, which buffers the whole response.
        "avg_ttft_ms": stages.get("ttft_p50_ms"),
        "latency_breakdown": stages,
        "by_category": by_category,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "results": results}
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (RESULTS_DIR / f"{stamp}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (RESULTS_DIR / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.label:
        (RESULTS_DIR / f"{args.label}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # history.csv drives the Retrieval tab of the metrics dashboard -- keep these
    # headers. Partial runs are excluded so they cannot distort the trend.
    if args.limit:
        print("(partial run: not appended to history.csv)")
    else:
        headers = ["timestamp", "commit", "recall_5", "recall_10", "mrr",
                   "avg_faithfulness", "avg_relevance", "avg_completeness",
                   "citation_accuracy", "refusal_rate",
                   "p50_latency_ms", "p95_latency_ms", "avg_ttft_ms"]
        is_new = not HISTORY_PATH.is_file()
        with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(headers)
            row = dict(summary, citation_accuracy=summary["citation_rate"])
            w.writerow([row[h] for h in headers])

    # ---- report ----------------------------------------------------------
    print("\n" + "=" * 72)
    print(f" EVAL SUMMARY  model={summary['model']}  judge={summary['judge_model']}")
    print("=" * 72)
    print(f" {'Faithfulness (0-1)':<34} {summary['faithfulness']:.3f}   target >= 0.90")
    print(f" {'Relevance (0-1)':<34} {summary['relevance']:.3f}   target >= 0.85")
    print(f" {'Citation rate':<34} {summary['citation_rate']:.3f}")
    refusal = summary["refusal_rate"]
    print(f" {'Out-of-scope refusal rate':<34} "
          + (f"{refusal:.3f}   target >= 0.90" if refusal is not None else "n/a (none in run)"))
    print(f" {'False refusals (in-scope)':<34} {summary['false_refusal_rate']:.3f}")
    print(f" {'Recall@5 / Recall@10':<34} {summary['recall_5']:.3f} / {summary['recall_10']:.3f}")
    print(f" {'MRR':<34} {summary['mrr']:.3f}")
    print(f" {'p50 / p95 total (ms)':<34} {summary['p50_latency_ms']:.0f} / {summary['p95_latency_ms']:.0f}")
    if stages:
        print("-" * 72)
        print(" server-side stage timings (p50 ms)")
        for key, label in (("embed_p50_ms", "embed"), ("retrieve_p50_ms", "retrieve"),
                           ("rerank_p50_ms", "rerank"), ("generate_p50_ms", "generate"),
                           ("ttft_p50_ms", "TTFT (real)")):
            if stages.get(key) is not None:
                print(f"   {label:<30} {stages[key]:>9.0f}")
        print(f"   {'query rewrites skipped':<30} {stages['rewrite_skipped_rate'] * 100:>8.0f}%")
    print("-" * 72)

    def cell(value: float | None, width: int = 8) -> str:
        return f"{value:>{width}.3f}" if value is not None else f"{'n/a':>{width}}"

    print(f" {'category':<14}{'n':>4}{'faith':>8}{'relev':>8}{'mrr':>8}{'p50 ms':>10}")
    for cat, m in by_category.items():
        print(f" {cat:<14}{m['n']:>4}{cell(m['faithfulness_norm'])}"
              f"{cell(m['relevance_norm'])}{cell(m['mrr'])}{m['p50_ms']:>10.0f}")
    print("=" * 72)
    print(f"Wrote {RESULTS_DIR / f'{stamp}.json'} and latest.json")


if __name__ == "__main__":
    asyncio.run(main())
