import json
import logging
from pathlib import Path
from datetime import datetime
import config
from backend.observability.metrics_db import log_request

JSONL_PATH = config.DATA_DIR / "request_logs.jsonl"
logger = logging.getLogger(__name__)

def log_query(
    request_id: str,
    query: str,
    rewritten_query: str | None,
    retrieved_chunk_ids: list[str],
    rerank_scores: list[float],
    cache_hit: bool,
    embed_latency: float,
    retrieve_latency: float,
    rerank_latency: float,
    generate_latency: float,
    total_latency: float,
    tokens_used: int,
    faithfulness_score: int | None = None,
    user_feedback: int | None = None,
    refusal: bool = False
):
    log_entry = {
        "request_id": request_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query": query,
        "rewritten_query": rewritten_query,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "rerank_scores": rerank_scores,
        "cache_hit": cache_hit,
        "latency_breakdown": {
            "embed": round(embed_latency * 1000, 2),
            "retrieve": round(retrieve_latency * 1000, 2),
            "rerank": round(rerank_latency * 1000, 2),
            "generate": round(generate_latency * 1000, 2),
            "total": round(total_latency * 1000, 2)
        },
        "tokens_used": tokens_used,
        "faithfulness_score": faithfulness_score,
        "user_feedback": user_feedback,
        "refusal": refusal
    }
    
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(JSONL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to write to request_logs.jsonl: {e}")
        
    # Also log to SQLite database (using latency values converted to milliseconds)
    log_request(
        request_id=request_id,
        query=query,
        rewritten_query=rewritten_query,
        retrieved_chunk_ids=retrieved_chunk_ids,
        rerank_scores=rerank_scores,
        cache_hit=cache_hit,
        embed_latency=round(embed_latency * 1000, 2),
        retrieve_latency=round(retrieve_latency * 1000, 2),
        rerank_latency=round(rerank_latency * 1000, 2),
        generate_latency=round(generate_latency * 1000, 2),
        total_latency=round(total_latency * 1000, 2),
        tokens_used=tokens_used,
        faithfulness_score=faithfulness_score,
        user_feedback=user_feedback,
        refusal=refusal
    )
