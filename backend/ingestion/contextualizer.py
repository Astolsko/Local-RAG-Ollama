import hashlib
import logging
import httpx
from backend import config

logger = logging.getLogger(__name__)

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
            
        prompt = (
            "Here is a document section and a chunk from it. Write one sentence situating this chunk within the document.\n\n"
            f"Document Section:\n{doc_text}\n\n"
            f"Chunk:\n{chunk_text}"
        )
        
        try:
            with httpx.Client(timeout=60) as client:
                r_post = client.post(
                    f"{config.OLLAMA_BASE}/api/generate",
                    json={
                        "model": config.LLM_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.0
                        }
                    }
                )
                r_post.raise_for_status()
                blurb = r_post.json().get("response", "").strip()
                if blurb.startswith('"') and blurb.endswith('"'):
                    blurb = blurb[1:-1].strip()
                
                # Update cache
                if r:
                    try:
                        r.set(redis_key, blurb)
                    except Exception as e:
                        logger.warning(f"Failed to write to Redis cache: {e}")
                else:
                    self.fallback_cache[h] = blurb
                    
                return blurb
        except Exception as e:
            logger.error(f"Failed to generate context blurb via Ollama: {e}")
            return ""

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
            for (h, rk, chunk), idx in zip(needed_generations, needed_indices):
                prompt = (
                    "Here is a document section and a chunk from it. Write one sentence situating this chunk within the document.\n\n"
                    f"Document Section:\n{doc_text}\n\n"
                    f"Chunk:\n{chunk}"
                )
                try:
                    with httpx.Client(timeout=60) as client:
                        r_post = client.post(
                            f"{config.OLLAMA_BASE}/api/generate",
                            json={
                                "model": config.LLM_MODEL,
                                "prompt": prompt,
                                "stream": False,
                                "options": {
                                    "temperature": 0.0
                                }
                            }
                        )
                        r_post.raise_for_status()
                        blurb = r_post.json().get("response", "").strip()
                        if blurb.startswith('"') and blurb.endswith('"'):
                            blurb = blurb[1:-1].strip()
                        
                        blurbs[idx] = blurb
                        if r:
                            try:
                                r.set(rk, blurb)
                            except Exception as e:
                                logger.warning(f"Failed to write to Redis cache: {e}")
                        else:
                            self.fallback_cache[h] = blurb
                except Exception as e:
                    logger.error(f"Failed to generate context blurb for chunk {idx}: {e}")
                    blurbs[idx] = ""
            
        return blurbs
