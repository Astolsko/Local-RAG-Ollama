import sqlite3
import json
from pathlib import Path
from datetime import datetime
import config

DB_PATH = config.DATA_DIR / "metrics.db"

def init_db():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS request_logs (
            request_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            query TEXT NOT NULL,
            rewritten_query TEXT,
            retrieved_chunk_ids TEXT,
            rerank_scores TEXT,
            cache_hit INTEGER,
            embed_latency REAL,
            retrieve_latency REAL,
            rerank_latency REAL,
            generate_latency REAL,
            total_latency REAL,
            tokens_used INTEGER,
            faithfulness_score INTEGER,
            user_feedback INTEGER,
            refusal INTEGER DEFAULT 0
        )
    """)
    # Migration helper for existing DBs
    try:
        cursor.execute("ALTER TABLE request_logs ADD COLUMN refusal INTEGER DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

    # Clear system prompt cache in Redis on startup to force reloading the new updated DEFAULT_SYSTEM_PROMPT
    try:
        from backend.redis_store import get_redis
        r = get_redis()
        if r:
            r.delete("prompt:system")
    except Exception:
        pass

    # Rebuild BM25 index to apply new tokenization
    try:
        from backend.retrieval.bm25_index import build_bm25_index
        build_bm25_index()
    except Exception:
        pass

def log_request(
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
    try:
        init_db()  # Ensure database and table are initialized
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO request_logs (
                request_id, timestamp, query, rewritten_query, retrieved_chunk_ids, rerank_scores,
                cache_hit, embed_latency, retrieve_latency, rerank_latency, generate_latency,
                total_latency, tokens_used, faithfulness_score, user_feedback, refusal
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id) DO UPDATE SET
                rewritten_query = excluded.rewritten_query,
                retrieved_chunk_ids = excluded.retrieved_chunk_ids,
                rerank_scores = excluded.rerank_scores,
                cache_hit = excluded.cache_hit,
                embed_latency = excluded.embed_latency,
                retrieve_latency = excluded.retrieve_latency,
                rerank_latency = excluded.rerank_latency,
                generate_latency = excluded.generate_latency,
                total_latency = excluded.total_latency,
                tokens_used = excluded.tokens_used,
                faithfulness_score = COALESCE(excluded.faithfulness_score, request_logs.faithfulness_score),
                user_feedback = COALESCE(excluded.user_feedback, request_logs.user_feedback),
                refusal = excluded.refusal
        """, (
            request_id,
            datetime.utcnow().isoformat() + "Z",
            query,
            rewritten_query,
            json.dumps(retrieved_chunk_ids),
            json.dumps(rerank_scores),
            1 if cache_hit else 0,
            embed_latency,
            retrieve_latency,
            rerank_latency,
            generate_latency,
            total_latency,
            tokens_used,
            faithfulness_score,
            user_feedback,
            1 if refusal else 0
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error logging request to SQLite: {e}")

def update_faithfulness(request_id: str, faithfulness_score: int):
    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE request_logs
            SET faithfulness_score = ?
            WHERE request_id = ?
        """, (faithfulness_score, request_id))
        conn.commit()
        conn.close()
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error updating faithfulness in SQLite: {e}")

def update_feedback(request_id: str, feedback: int):
    try:
        init_db()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE request_logs
            SET user_feedback = ?
            WHERE request_id = ?
        """, (feedback, request_id))
        conn.commit()
        conn.close()
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error updating feedback in SQLite: {e}")
