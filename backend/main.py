import sys
from pathlib import Path
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(backend_dir.parent))

import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import config
from chat_history import delete_chat, get_chat, list_chats, save_chat
from backend.rag import add_source, ask, ask_stream, get_source, delete_source, get_system_prompt, list_sources, set_system_prompt
from redis_store import clear_session, get_session_messages, ping as redis_ping, require_redis

config.DATA_DIR.mkdir(parents=True, exist_ok=True)
config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

from backend.observability.metrics_db import init_db
init_db()

app = FastAPI(title="RAG LLM", version="1.1.0")
from rate_limit import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _redis_or_503():
    try:
        require_redis()
    except Exception as e:
        raise HTTPException(503, f"Redis unavailable: {e}") from e


class SourceIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)


class AskIn(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None
    source_ids: list[str] | None = None


class EndChatIn(BaseModel):
    session_id: str = Field(min_length=1)
    title: str | None = Field(default=None, max_length=200)


class SystemPromptIn(BaseModel):
    text: str = Field(min_length=1)


class SettingsUpdate(BaseModel):
    OLLAMA_BASE: str | None = None
    EMBED_MODEL: str | None = None
    LLM_MODEL: str | None = None
    REDIS_URL: str | None = None
    CHUNK_SIZE: int | None = None
    CHUNK_OVERLAP: int | None = None
    TOP_K: int | None = None


@app.get("/api/settings")
def get_settings():
    return config.load_settings()


@app.put("/api/settings")
def update_settings(body: SettingsUpdate):
    current = config.load_settings()
    for k, v in body.model_dump().items():
        if v is not None:
            current[k] = v
    config.save_settings(current)
    return current


@app.get("/api/settings/ollama-models")
def get_ollama_models():
    try:
        import httpx
        r = httpx.get(f"{config.OLLAMA_BASE}/api/tags", timeout=5.0)
        r.raise_for_status()
        models = r.json().get("models", [])
        return {"models": [m["name"] for m in models]}
    except Exception:
        return {"models": []}


@app.get("/api/health")
def health():
    chroma_ok = False
    try:
        from backend.rag import _client
        _client.heartbeat()
        chroma_ok = True
    except Exception:
        pass

    ollama_ok = False
    try:
        import httpx
        r = httpx.get(f"{config.OLLAMA_BASE}/", timeout=2.0)
        if r.status_code == 200 or "Ollama is running" in r.text:
            ollama_ok = True
    except Exception:
        pass

    redis_ok = False
    try:
        redis_ok = bool(redis_ping())
    except Exception:
        pass

    all_ok = redis_ok and chroma_ok and ollama_ok

    return {
        "status": "ok" if all_ok else "degraded",
        "phase": 3,
        "ollama": config.OLLAMA_BASE,
        "embed_model": config.EMBED_MODEL,
        "llm_model": config.LLM_MODEL,
        "sources": len(list_sources()),
        "redis": redis_ok,
        "saved_chats": len(list_chats()),
        "uptime_checks": {
            "redis": redis_ok,
            "chroma": chroma_ok,
            "ollama": ollama_ok
        }
    }


@app.get("/api/settings/system-prompt")
def read_system_prompt():
    return {"text": get_system_prompt()}


@app.put("/api/settings/system-prompt")
def update_system_prompt(body: SystemPromptIn):
    _redis_or_503()
    try:
        return {"text": set_system_prompt(body.text)}
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/chats/start")
def start_chat():
    _redis_or_503()
    return {"session_id": str(uuid.uuid4())}


@app.post("/api/chats/ask")
def chat_ask(body: AskIn):
    _redis_or_503()
    try:
        if not body.question.strip():
            raise HTTPException(400, "Question is empty")
        from chat_stream import chat_stream
        generator = chat_stream(body.question.strip(), body.session_id, body.source_ids)
        return StreamingResponse(generator, media_type="text/event-stream")
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/sources/{source_id}")
def get_source_detail(source_id: str):
    source = get_source(source_id)
    if not source:
        raise HTTPException(404, "Source not found")
    return source


@app.post("/api/chats/end")
def end_chat(body: EndChatIn):
    _redis_or_503()
    messages = get_session_messages(body.session_id)
    if not messages:
        clear_session(body.session_id)
        raise HTTPException(400, "Nothing to save — chat is empty")

    saved = save_chat(body.title, messages, get_system_prompt())
    clear_session(body.session_id)
    return saved


@app.post("/api/chats/clear/{session_id}")
def clear_chat_session(session_id: str):
    _redis_or_503()
    clear_session(session_id)
    return {"cleared": session_id}


@app.get("/api/chats/history")
def chat_history():
    return list_chats()


@app.get("/api/chats/history/{chat_id}")
def chat_detail(chat_id: str):
    chat = get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    return chat


@app.delete("/api/chats/history/{chat_id}")
def remove_chat(chat_id: str):
    if not delete_chat(chat_id):
        raise HTTPException(404, "Chat not found")
    return {"deleted": chat_id}


@app.get("/api/sources")
def get_sources():
    return list_sources()


@app.post("/api/sources")
def post_source(body: SourceIn):
    try:
        return add_source(body.name.strip(), body.text)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/sources/upload")
async def upload_source(
    file: UploadFile = File(...),
    name: str | None = Form(None),
):
    filename = file.filename or "uploaded"
    source_name = (name or filename).strip()
    raw = await file.read()
    
    if filename.lower().endswith(".pdf"):
        import io
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(raw))
            text_pages = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text)
            text = "\f".join(text_pages).strip()
            if not text:
                raise ValueError("Could not extract any readable text from the uploaded PDF.")
        except Exception as e:
            raise HTTPException(400, f"Failed to parse PDF file: {e}") from e
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            raise HTTPException(
                400, 
                "Unsupported file type. Please upload a standard UTF-8 text/markdown file, or a PDF."
            ) from e

    try:
        return add_source(source_name, text)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.delete("/api/sources/{source_id}")
def remove_source(source_id: str):
    if not delete_source(source_id):
        raise HTTPException(404, "Source not found")
    return {"deleted": source_id}


@app.post("/api/ask")
def post_ask(body: AskIn):
    """Legacy endpoint — prefer /api/chats/ask with session_id."""
    _redis_or_503()
    try:
        return ask(body.question.strip(), body.session_id, body.source_ids)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


class FeedbackIn(BaseModel):
    request_id: str
    feedback: int  # 1 for thumbs-up, -1 for thumbs-down


@app.post("/api/observability/feedback")
def submit_feedback(body: FeedbackIn):
    try:
        from backend.observability.metrics_db import update_feedback
        update_feedback(body.request_id, body.feedback)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(500, f"Failed to update feedback: {e}")


@app.get("/api/observability/eval-history")
def get_eval_history():
    import csv
    # eval directory is sibling to backend
    eval_history_path = Path("eval/history.csv")
    if not eval_history_path.exists():
        eval_history_path = Path("../eval/history.csv")
        
    if not eval_history_path.exists():
        return []
        
    rows = []
    try:
        with open(eval_history_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed_row = {}
                for k, v in row.items():
                    try:
                        if "." in v:
                            parsed_row[k] = float(v)
                        else:
                            parsed_row[k] = int(v)
                    except ValueError:
                        parsed_row[k] = v
                rows.append(parsed_row)
    except Exception as e:
        raise HTTPException(500, f"Error reading evaluation history: {e}")
    return rows


@app.get("/api/observability/metrics")
def get_aggregated_metrics():
    import sqlite3
    from backend.observability.metrics_db import DB_PATH
    
    empty_summary = {
        "total_requests": 0,
        "cache_hit_rate": 0.0,
        "p50_total_latency": 0.0,
        "p95_total_latency": 0.0,
        "avg_faithfulness": 0.0,
        "refusal_rate": 0.0,
        "thumbs_up": 0,
        "thumbs_down": 0,
        "avg_embed_latency": 0.0,
        "avg_bm25_latency": 0.0,
        "avg_vector_latency": 0.0,
        "avg_rrf_latency": 0.0,
        "avg_rerank_latency": 0.0,
        "avg_generate_latency": 0.0,
        "avg_ttft_latency": 0.0,
        "avg_cache_check_latency": 0.0
    }

    if not DB_PATH.exists():
        return {
            "summary": empty_summary,
            "daily": []
        }
        
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM request_logs ORDER BY timestamp ASC")
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
    except Exception as e:
        raise HTTPException(500, f"Database access failed: {e}")
        
    if not rows:
        return {
            "summary": empty_summary,
            "daily": []
        }
        
    total_requests = len(rows)
    cache_hits = sum(1 for r in rows if r["cache_hit"] == 1)
    cache_hit_rate = cache_hits / total_requests
    
    total_latencies = [r["total_latency"] for r in rows if r["total_latency"] is not None]
    if total_latencies:
        total_latencies.sort()
        p50_total = total_latencies[int(len(total_latencies) * 0.5)]
        p95_total = total_latencies[int(len(total_latencies) * 0.95)]
    else:
        p50_total, p95_total = 0.0, 0.0
        
    faithfulness_scores = [r["faithfulness_score"] for r in rows if r["faithfulness_score"] is not None]
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
    
    refusals = sum(1 for r in rows if r.get("refusal") == 1)
    refusal_rate = refusals / total_requests
    
    thumbs_up = sum(1 for r in rows if r["user_feedback"] == 1)
    thumbs_down = sum(1 for r in rows if r["user_feedback"] == -1)

    def _avg(lst):
        vals = [v for v in lst if v is not None]
        return sum(vals) / len(vals) if vals else 0.0
    
    from collections import defaultdict
    daily_groups = defaultdict(list)
    for r in rows:
        date_str = r["timestamp"][:10]
        daily_groups[date_str].append(r)
        
    daily_data = []
    for date_str, group_rows in sorted(daily_groups.items()):
        g_total = len(group_rows)
        g_cache_hits = sum(1 for r in group_rows if r["cache_hit"] == 1)
        g_tokens = sum(r["tokens_used"] for r in group_rows if r["tokens_used"] is not None)
        g_latencies = sorted([r["total_latency"] for r in group_rows if r["total_latency"] is not None])
        g_p50 = g_latencies[int(len(g_latencies) * 0.5)] if g_latencies else 0.0
        g_p95 = g_latencies[int(len(g_latencies) * 0.95)] if g_latencies else 0.0
        
        g_thumbs_up = sum(1 for r in group_rows if r["user_feedback"] == 1)
        g_thumbs_down = sum(1 for r in group_rows if r["user_feedback"] == -1)

        g_faithfulness_scores = [r["faithfulness_score"] for r in group_rows if r["faithfulness_score"] is not None]
        g_avg_faithfulness = sum(g_faithfulness_scores) / len(g_faithfulness_scores) if g_faithfulness_scores else 0.0
        
        daily_data.append({
            "date": date_str,
            "requests": g_total,
            "tokens": g_tokens,
            "cache_hit_rate": g_cache_hits / g_total,
            "p50_latency": g_p50,
            "p95_latency": g_p95,
            "thumbs_up": g_thumbs_up,
            "thumbs_down": g_thumbs_down,
            "embed_latency": _avg([r.get("embed_latency") for r in group_rows]),
            "bm25_latency": _avg([r.get("bm25_latency") for r in group_rows]),
            "vector_latency": _avg([r.get("vector_latency") for r in group_rows]),
            "rrf_latency": _avg([r.get("rrf_latency") for r in group_rows]),
            "rerank_latency": _avg([r.get("rerank_latency") for r in group_rows]),
            "generate_latency": _avg([r.get("generate_latency") for r in group_rows]),
            "ttft_latency": _avg([r.get("ttft_latency") for r in group_rows]),
            "cache_check_latency": _avg([r.get("cache_check_latency") for r in group_rows]),
            "avg_faithfulness": g_avg_faithfulness
        })
        
    return {
        "summary": {
            "total_requests": total_requests,
            "cache_hit_rate": cache_hit_rate,
            "p50_total_latency": p50_total,
            "p95_total_latency": p95_total,
            "avg_faithfulness": avg_faithfulness,
            "refusal_rate": refusal_rate,
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
            "avg_embed_latency": _avg([r.get("embed_latency") for r in rows]),
            "avg_bm25_latency": _avg([r.get("bm25_latency") for r in rows]),
            "avg_vector_latency": _avg([r.get("vector_latency") for r in rows]),
            "avg_rrf_latency": _avg([r.get("rrf_latency") for r in rows]),
            "avg_rerank_latency": _avg([r.get("rerank_latency") for r in rows]),
            "avg_generate_latency": _avg([r.get("generate_latency") for r in rows]),
            "avg_ttft_latency": _avg([r.get("ttft_latency") for r in rows]),
            "avg_cache_check_latency": _avg([r.get("cache_check_latency") for r in rows])
        },
        "daily": daily_data
    }
