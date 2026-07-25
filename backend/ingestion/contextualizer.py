import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor

import httpx
from backend import config

logger = logging.getLogger(__name__)

# Blurbs are one LLM call per chunk, so a long document serialises hundreds of them.
# Ollama queues past its own parallelism limit, but overlapping the request round-trips
# still removes the dead time between calls.
BLURB_WORKERS = 4


def _blurb_prompt(doc_text: str, chunk_text: str) -> str:
    # ponytail: the whole document goes into every chunk's prompt, so prompt tokens grow
    # as O(doc_size * chunk_count) at ingestion. Passing the enclosing section (or a
    # cached one-shot document summary) instead would make it linear — but it changes
    # blurb text, hence the embeddings, so it needs a re-ingest and an eval.
    return (
        "Here is a document section and a chunk from it. Write one sentence situating this chunk within the document.\n\n"
        f"Document Section:\n{doc_text}\n\n"
        f"Chunk:\n{chunk_text}"
    )


def _generate_blurb(client: httpx.Client, doc_text: str, chunk_text: str) -> str:
    """One Ollama call for one chunk's blurb. Returns "" on any failure."""
    try:
        r = client.post(
            f"{config.OLLAMA_BASE}/api/generate",
            json={
                "model": config.LLM_MODEL,
                "prompt": _blurb_prompt(doc_text, chunk_text),
                "stream": False,
                "options": {"temperature": 0.0},
                "keep_alive": getattr(config, "OLLAMA_KEEP_ALIVE", "10m"),
            },
        )
        r.raise_for_status()
        blurb = r.json().get("response", "").strip()
        if blurb.startswith('"') and blurb.endswith('"'):
            blurb = blurb[1:-1].strip()
        return blurb
    except Exception as e:
        logger.error(f"Failed to generate context blurb via Ollama: {e}")
        return ""


class Contextualizer:
    def __init__(self):
        # Fallback local in-memory cache if Redis is down/unavailable
        self.fallback_cache = {}

    def _get_redis_client(self):
        try:
            from redis_store import get_redis
            r = get_redis()
            if r.ping():
                return r
        except Exception:
            pass
        return None

    def get_blurb(self, doc_text: str, chunk_text: str) -> str:
        """Get a single blurb, generating it and caching if not present."""
        hash_input = f"{doc_text}|||{chunk_text}"
        h = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
        redis_key = f"context:blurb:{h}"
        
        # Try Redis cache first
        r = self._get_redis_client()
        if r:
            try:
                cached = r.get(redis_key)
                if cached:
                    return cached
            except Exception as e:
                logger.warning(f"Failed to read from Redis cache: {e}")
        else:
            if h in self.fallback_cache:
                return self.fallback_cache[h]
            
        with httpx.Client(timeout=60) as client:
            blurb = _generate_blurb(client, doc_text, chunk_text)
        if not blurb:
            return ""

        if r:
            try:
                r.set(redis_key, blurb)
            except Exception as e:
                logger.warning(f"Failed to write to Redis cache: {e}")
        else:
            self.fallback_cache[h] = blurb
        return blurb

    def get_blurbs_batch(self, doc_text: str, chunk_texts: list[str]) -> list[str]:
        """Fetch blurbs for a batch of chunks, calling Ollama only for cache misses."""
        blurbs = []
        needed_generations = []
        needed_indices = []
        
        hashes = []
        redis_keys = []
        for chunk in chunk_texts:
            hash_input = f"{doc_text}|||{chunk}"
            h = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
            hashes.append(h)
            redis_keys.append(f"context:blurb:{h}")
            
        # Try bulk lookup in Redis
        r = self._get_redis_client()
        cached_values = []
        if r:
            try:
                cached_values = r.mget(redis_keys)
            except Exception as e:
                logger.warning(f"Failed to bulk read from Redis cache: {e}")
                cached_values = [None] * len(chunk_texts)
        else:
            cached_values = [self.fallback_cache.get(h) for h in hashes]
            
        for idx, val in enumerate(cached_values):
            if val is not None:
                blurbs.append(val)
            else:
                blurbs.append("")
                needed_generations.append((hashes[idx], redis_keys[idx], chunk_texts[idx]))
                needed_indices.append(idx)
                
        if needed_generations:
            # One shared client, requests overlapped: this loop used to open a fresh
            # connection per chunk and wait for each generation before starting the next.
            with httpx.Client(timeout=60) as client:
                with ThreadPoolExecutor(max_workers=BLURB_WORKERS) as pool:
                    generated = list(pool.map(
                        lambda g: _generate_blurb(client, doc_text, g[2]),
                        needed_generations,
                    ))

            for (h, rk, _chunk), idx, blurb in zip(needed_generations, needed_indices, generated):
                blurbs[idx] = blurb
                if not blurb:
                    continue
                if r:
                    try:
                        r.set(rk, blurb)
                    except Exception as e:
                        logger.warning(f"Failed to write to Redis cache: {e}")
                else:
                    self.fallback_cache[h] = blurb

        return blurbs
