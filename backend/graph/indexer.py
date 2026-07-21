"""Background graph indexing: extract a document's chunks into the graph store.

Job state lives in Redis (`graph:index:{doc_id}`) so the UI can poll progress. Runs off
the request path (FastAPI BackgroundTasks) — extraction is one LLM call per chunk and must
never block an answer. Idempotent: re-indexing a doc first drops its old graph rows.
"""
import json
import logging

from backend.graph import store, extractor

logger = logging.getLogger(__name__)


def _status_key(doc_id: str) -> str:
    return f"graph:index:{doc_id}"


def _set_status(doc_id: str, **fields) -> None:
    try:
        from redis_store import get_redis
        r = get_redis()
        if r:
            r.set(_status_key(doc_id), json.dumps(fields))
    except Exception:
        pass


def get_status(doc_id: str) -> dict | None:
    try:
        from redis_store import get_redis
        r = get_redis()
        raw = r.get(_status_key(doc_id)) if r else None
        return json.loads(raw) if raw else None
    except Exception:
        return None


def all_status() -> dict:
    out: dict = {}
    try:
        from redis_store import get_redis
        r = get_redis()
        if not r:
            return out
        for k in r.scan_iter("graph:index:*"):
            raw = r.get(k)
            if raw:
                out[k.split(":")[-1]] = json.loads(raw)
    except Exception:
        pass
    return out


def index_document(doc_id: str, rebuild_communities: bool = True) -> None:
    """Extract every chunk of `doc_id` into the graph store. Never raises."""
    store.init_db()
    try:
        from backend import rag
        src = rag.get_source(doc_id)
    except Exception as e:
        _set_status(doc_id, state="error", error=str(e))
        return
    if not src:
        _set_status(doc_id, state="error", error="source not found")
        return

    chunks = src.get("chunks", [])
    total = len(chunks)
    _set_status(doc_id, state="running", done_chunks=0, total_chunks=total)
    store.delete_doc(doc_id)  # clean re-index

    for i, ch in enumerate(chunks):
        chunk_id = f"{doc_id}:{ch['chunk_index']}"
        try:
            extractor.persist(extractor.extract_from_chunk(ch["text"]), chunk_id, doc_id)
        except Exception as e:
            logger.warning(f"graph index chunk {chunk_id} failed: {e}")
        _set_status(doc_id, state="running", done_chunks=i + 1, total_chunks=total)

    if rebuild_communities:
        try:
            from backend.graph import communities
            communities.rebuild()
        except Exception as e:
            logger.warning(f"community rebuild failed: {e}")

    _set_status(doc_id, state="done", done_chunks=total, total_chunks=total)
