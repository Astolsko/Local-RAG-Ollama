"""Redis-backed cache for text embeddings.

Keyed by sha256(model + text) so query and ingestion embeddings share one cache.
Hits skip the Ollama call entirely. Individual keys (not one hash) so each entry
carries its own EMBED_CACHE_TTL — Redis hashes can't expire per field.
Degrades to no-op when Redis is down.
"""
import hashlib
import json
import logging

import config
from redis_store import get_redis

logger = logging.getLogger(__name__)

DEFAULT_TTL = 604800  # 7 days


def _key(model: str, text: str) -> str:
    h = hashlib.sha256(f"{model}\x00{text}".encode("utf-8")).hexdigest()
    return f"embed:{h}"


def get_cached(model: str, texts: list[str]) -> list[list[float] | None]:
    """Cached embeddings aligned to `texts`; None where absent/unreadable."""
    r = get_redis()
    if not r:
        return [None] * len(texts)
    try:
        pipe = r.pipeline()
        for t in texts:
            pipe.get(_key(model, t))
        raw = pipe.execute()
    except Exception as e:
        logger.warning(f"embed cache read failed: {e}")
        return [None] * len(texts)
    out: list[list[float] | None] = []
    for v in raw:
        try:
            out.append(json.loads(v) if v else None)
        except Exception:
            out.append(None)
    return out


def set_cached(model: str, pairs: list[tuple[str, list[float]]]) -> None:
    r = get_redis()
    if not r or not pairs:
        return
    ttl = getattr(config, "EMBED_CACHE_TTL", DEFAULT_TTL)
    try:
        pipe = r.pipeline()
        for text, emb in pairs:
            pipe.setex(_key(model, text), ttl, json.dumps(emb))
        pipe.execute()
    except Exception as e:
        logger.warning(f"embed cache write failed: {e}")


def record(hits: int, misses: int) -> None:
    """Bump Redis hit/miss counters (surfaced as embed cache hit rate in metrics)."""
    r = get_redis()
    if not r:
        return
    try:
        if hits:
            r.incrby("metrics:embed_cache_hits", hits)
        if misses:
            r.incrby("metrics:embed_cache_misses", misses)
    except Exception:
        pass
