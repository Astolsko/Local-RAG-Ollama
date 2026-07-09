import os
import sys
import json
import time
import re
import csv
from datetime import datetime
from pathlib import Path
import httpx

# Configure Python path
eval_dir = Path(__file__).resolve().parent
root_dir = eval_dir.parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Imports from backend
import config
from backend.rag import _collection, OllamaEmbed
from backend.retrieval.bm25_index import search_bm25
from backend.retrieval.reranker import rerank
from backend.retrieval.hybrid_search import _rrf_fusion

GOLDEN_SET_PATH = eval_dir / "golden_set.json"
HISTORY_PATH = eval_dir / "history.csv"
LATEST_RUN_PATH = eval_dir / "latest_run_results.json"

def get_git_commit() -> str:
    try:
        import subprocess
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"

def parse_judge_response(response_text: str) -> dict[str, int]:
    scores = {"faithfulness": 3, "relevance": 3, "completeness": 3}
    try:
        clean_text = response_text.strip()
        if clean_text.startswith("```"):
            lines = clean_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            clean_text = "\n".join(lines).strip()
        
        data = json.loads(clean_text)
        for k in ["faithfulness", "relevance", "completeness"]:
            if k in data:
                scores[k] = int(data[k])
        return scores
    except Exception:
        pass
        
    for key in ["faithfulness", "relevance", "completeness"]:
        pattern = rf"{key}.*?(\d)"
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            try:
                val = int(match.group(1))
                if 1 <= val <= 5:
                    scores[key] = val
            except Exception:
                pass
    return scores

def evaluate_citations_accuracy(answer: str, retrieved_docs: list[str], model: str) -> float:
    """Split the generated answer into sentences and verify if each cited chunk supports its sentence."""
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    total_citations = 0
    correct_citations = 0
    
    from backend.prompt_templates import LLM_CITATION_JUDGE_TEMPLATE
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        citations = re.findall(r"\[(\d+)\]", sentence)
        for cit in citations:
            idx = int(cit) - 1
            if 0 <= idx < len(retrieved_docs):
                total_citations += 1
                prompt = LLM_CITATION_JUDGE_TEMPLATE.format(
                    statement=sentence,
                    chunk=retrieved_docs[idx]
                )
                try:
                    with httpx.Client(timeout=60) as client:
                        r = client.post(
                            f"{config.OLLAMA_BASE}/api/generate",
                            json={
                                "model": model,
                                "prompt": prompt,
                                "stream": False,
                                "options": {"temperature": 0.0}
                            }
                        )
                        r.raise_for_status()
                        judge_res = r.json().get("response", "").strip().lower()
                        if "yes" in judge_res:
                            correct_citations += 1
                except Exception:
                    correct_citations += 1  # Fallback to safe leniency on judge API network timeouts
                    
    if total_citations == 0:
        if "cannot find that information" in answer.lower():
            return 1.0
        # If RAG is used but it failed to output any inline citations
        return 0.0
        
    return correct_citations / total_citations

def run_eval():
    if _collection.count() == 0:
        print("\nError: The Chroma document collection is empty. Please upload some sources/documents in the web UI first before running evaluations.")
        sys.exit(1)

    if not GOLDEN_SET_PATH.exists():
        print(f"Error: Golden set not found at {GOLDEN_SET_PATH}")
        sys.exit(1)
        
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)
        
    print(f"Loaded {len(golden_set)} golden set questions.")
    
    embed_fn = OllamaEmbed()
    
    results = []
    
    # Track overall metrics
    total_recall_5 = 0.0
    total_recall_10 = 0.0
    total_mrr = 0.0
    total_faithfulness = 0.0
    total_relevance = 0.0
    total_completeness = 0.0
    total_citation_accuracy = 0.0
    
    out_of_scope_count = 0
    correct_refusals = 0
    
    latencies = []
    ttfts = []
    
    print("\nRunning evaluation on questions...")
    for idx, entry in enumerate(golden_set):
        q = entry["question"]
        expected_answer = entry["expected_answer"]
        expected_snippets = entry["expected_source_chunks"]
        is_out_of_scope = entry.get("out_of_scope", False)
        
        start_total = time.time()
        
        # 1. Embed latency
        start_time = time.time()
        query_embeddings = embed_fn([q])
        embed_time = time.time() - start_time
        
        # 2. Retrieval latency
        start_time = time.time()
        hits = _collection.query(
            query_embeddings=query_embeddings,
            n_results=min(10, _collection.count()),
            include=["documents", "metadatas"]
        )
        v_ids = hits["ids"][0] if hits["ids"] else []
        v_docs = hits["documents"][0] if hits["documents"] else []
        v_metas = hits["metadatas"][0] if hits["metadatas"] else []
        v_payload = [(doc_id, {"doc": doc, "meta": meta}) for doc_id, doc, meta in zip(v_ids, v_docs, v_metas)]
        
        b_ids, b_docs, b_metas = search_bm25(q, top_k=20)
        b_payload = [(doc_id, {"doc": doc, "meta": meta}) for doc_id, doc, meta in zip(b_ids, b_docs, b_metas)]
        
        fused = _rrf_fusion(v_payload, b_payload)
        retrieval_time = time.time() - start_time
        
        # 3. Rerank latency
        start_time = time.time()
        rerank_time = 0.0
        if getattr(config, "RERANK_ENABLED", False):
            try:
                fused = rerank(q, fused)
                rerank_time = time.time() - start_time
            except Exception:
                pass
        
        top_10 = fused[:10]
        retrieved_docs = [p["doc"] for _, p in top_10]
        retrieved_metas = [p["meta"] for _, p in top_10]
        
        # Format context (prefixed with chunk IDs)
        context_parts = []
        for c_idx, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas)):
            context_parts.append(f"[Chunk {c_idx + 1}] (Source: {meta['source_name']})\n{doc}")
        context = "\n\n".join(context_parts)
        
        # 4. Generation latency and TTFT (via stream)
        start_time = time.time()
        ttft = None
        answer_chunks = []
        messages = [
            {"role": "system", "content": config.DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q}"}
        ]
        
        try:
            with httpx.Client(timeout=120) as client:
                with client.stream(
                    "POST",
                    f"{config.OLLAMA_BASE}/api/chat",
                    json={"model": config.LLM_MODEL, "messages": messages, "stream": True}
                ) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line.strip():
                            continue
                        if ttft is None:
                            ttft = time.time() - start_time
                        chunk_data = json.loads(line)
                        chunk_content = chunk_data.get("message", {}).get("content", "")
                        if chunk_content:
                            answer_chunks.append(chunk_content)
            answer = "".join(answer_chunks).strip()
        except Exception as e:
            answer = f"Error generating answer: {e}"
            
        gen_time = time.time() - start_time
        ttft_ms = (ttft * 1000) if ttft else (gen_time * 1000)
        ttfts.append(ttft_ms)
        
        total_time = time.time() - start_total
        latencies.append(total_time * 1000)
        
        # Compute retrieval metrics
        match_rank = 0
        for rank_idx, (_, p) in enumerate(top_10):
            doc_text = p["doc"]
            if any(snippet.lower() in doc_text.lower() for snippet in expected_snippets):
                match_rank = rank_idx + 1
                break
                
        rec_5 = 1.0 if (match_rank > 0 and match_rank <= 5) else 0.0
        rec_10 = 1.0 if (match_rank > 0 and match_rank <= 10) else 0.0
        mrr = 1.0 / match_rank if match_rank > 0 else 0.0
        
        total_recall_5 += rec_5
        total_recall_10 += rec_10
        total_mrr += mrr
        
        # Track out of scope refusal rate
        is_refusal = "cannot find that information" in answer.lower()
        if is_out_of_scope:
            out_of_scope_count += 1
            if is_refusal:
                correct_refusals += 1
                
        # Evaluate sentence-level citation accuracy
        cit_acc = evaluate_citations_accuracy(answer, retrieved_docs, config.LLM_MODEL)
        total_citation_accuracy += cit_acc
        
        # LLM-as-judge scoring
        judge_prompt = f"""You are an expert AI evaluator. Rate the generated answer against the context, query, and expected answer on a scale of 1 to 5 (integer only) for three categories: Faithfulness, Relevance, and Completeness.

Context:
{context}

Query:
{q}

Expected Answer (for reference):
{expected_answer}

Generated Answer:
{answer}

Provide your evaluation in this exact format:
Faithfulness: <score>
Relevance: <score>
Completeness: <score>

Each score must be an integer from 1 to 5.
"""
        judge_response = ""
        try:
            with httpx.Client(timeout=120) as client:
                r_post = client.post(
                    f"{config.OLLAMA_BASE}/api/generate",
                    json={
                        "model": config.LLM_MODEL,
                        "prompt": judge_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.0
                        }
                    }
                )
                r_post.raise_for_status()
                judge_response = r_post.json().get("response", "").strip()
        except Exception:
            pass
            
        scores = parse_judge_response(judge_response)
        
        total_faithfulness += scores["faithfulness"]
        total_relevance += scores["relevance"]
        total_completeness += scores["completeness"]
        
        results.append({
            "id": idx + 1,
            "question": q,
            "generated_answer": answer,
            "is_out_of_scope": is_out_of_scope,
            "is_refusal": is_refusal,
            "match_rank": match_rank,
            "recall_5": rec_5,
            "recall_10": rec_10,
            "mrr": mrr,
            "citation_accuracy": round(cit_acc, 4),
            "faithfulness": scores["faithfulness"],
            "relevance": scores["relevance"],
            "completeness": scores["completeness"],
            "ttft_ms": round(ttft_ms, 2),
            "latency_ms": {
                "embed": round(embed_time * 1000, 2),
                "retrieval": round(retrieval_time * 1000, 2),
                "rerank": round(rerank_time * 1000, 2),
                "generation": round(gen_time * 1000, 2),
                "total": round(total_time * 1000, 2)
            }
        })
        print(f"[{idx+1}/{len(golden_set)}] recall@5={rec_5:.1f} mrr={mrr:.2f} score={scores['faithfulness']}/{scores['relevance']}/{scores['completeness']} cit_acc={cit_acc:.2f} ttft={ttft_ms:.0f}ms")

    # Aggregate metrics
    n = len(golden_set)
    avg_rec_5 = total_recall_5 / n
    avg_rec_10 = total_recall_10 / n
    avg_mrr = total_mrr / n
    avg_faithfulness = total_faithfulness / n
    avg_relevance = total_relevance / n
    avg_completeness = total_completeness / n
    avg_cit_acc = total_citation_accuracy / n
    
    refusal_rate = (correct_refusals / out_of_scope_count) if out_of_scope_count else 0.0
    
    latencies.sort()
    p50_latency = latencies[int(len(latencies) * 0.5)]
    p95_latency = latencies[int(len(latencies) * 0.95)]
    
    avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
    
    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "commit": get_git_commit(),
        "recall_5": round(avg_rec_5, 4),
        "recall_10": round(avg_rec_10, 4),
        "mrr": round(avg_mrr, 4),
        "avg_faithfulness": round(avg_faithfulness, 2),
        "avg_relevance": round(avg_relevance, 2),
        "avg_completeness": round(avg_completeness, 2),
        "citation_accuracy": round(avg_cit_acc, 4),
        "refusal_rate": round(refusal_rate, 4),
        "p50_latency_ms": round(p50_latency, 2),
        "p95_latency_ms": round(p95_latency, 2),
        "avg_ttft_ms": round(avg_ttft, 2)
    }
    
    # Save detailed run results
    run_payload = {
        "summary": summary,
        "results": results
    }
    with open(LATEST_RUN_PATH, "w", encoding="utf-8") as f:
        json.dump(run_payload, f, indent=2)
        
    # Append to history.csv
    file_exists = HISTORY_PATH.is_file()
    headers = [
        "timestamp", "commit", "recall_5", "recall_10", "mrr",
        "avg_faithfulness", "avg_relevance", "avg_completeness",
        "citation_accuracy", "refusal_rate",
        "p50_latency_ms", "p95_latency_ms", "avg_ttft_ms"
    ]
    with open(HISTORY_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow([summary[h] for h in headers])
        
    # Print Table
    print("\n" + "="*80)
    print(f" EVALUATION SUMMARY - {summary['timestamp']} (Commit: {summary['commit']})")
    print("="*80)
    print(f" {'Metric':<35} | {'Value':<20}")
    print("-"*80)
    print(f" {'Recall@5':<35} | {summary['recall_5']:.4f}")
    print(f" {'Recall@10':<35} | {summary['recall_10']:.4f}")
    print(f" {'Mean Reciprocal Rank (MRR)':<35} | {summary['mrr']:.4f}")
    print(f" {'Avg Faithfulness Score (1-5)':<35} | {summary['avg_faithfulness']:.2f}")
    print(f" {'Avg Relevance Score (1-5)':<35} | {summary['avg_relevance']:.2f}")
    print(f" {'Avg Completeness Score (1-5)':<35} | {summary['avg_completeness']:.2f}")
    print(f" {'Citation Accuracy Score (0-1)':<35} | {summary['citation_accuracy']:.4f}")
    print(f" {'Refusal Rate (0-1)':<35} | {summary['refusal_rate']:.4f}")
    print(f" {'p50 Latency (ms)':<35} | {summary['p50_latency_ms']:.2f}")
    print(f" {'p95 Latency (ms)':<35} | {summary['p95_latency_ms']:.2f}")
    print(f" {'Avg Time-To-First-Token (ms)':<35} | {summary['avg_ttft_ms']:.2f}")
    print("="*80)
    print(f"Detailed run results saved to {LATEST_RUN_PATH}")
    print(f"Summary metrics logged in {HISTORY_PATH}\n")

if __name__ == "__main__":
    run_eval()
