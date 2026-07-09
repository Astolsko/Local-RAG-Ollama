import json
import math
import hashlib
import logging
import config
from redis_store import get_redis

logger = logging.getLogger(__name__)

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if len(v1) != len(v2) or not v1:
        return 0.0
    dot_prod = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_prod / (norm_a * norm_b)

def check_semantic_cache(query: str, query_embedding: list[float]) -> dict | None:
    """Check Redis for a semantically similar query and return cached response if match is above threshold."""
    if hasattr(query_embedding, "tolist"):
        query_embedding = query_embedding.tolist()
    redis_client = get_redis()
    if not redis_client:
        return None
        
    try:
        current_version_raw = redis_client.get("corpus:version")
        if isinstance(current_version_raw, bytes):
            current_version = current_version_raw.decode("utf-8")
        elif isinstance(current_version_raw, str):
            current_version = current_version_raw
        else:
            current_version = "0"
        
        # Get all cache keys
        keys_set = redis_client.smembers("semantic:cache:keys")
        if not keys_set:
            return None
            
        keys = [k.decode("utf-8") if isinstance(k, bytes) else k for k in keys_set]
        
        # Fetch entries using a pipeline for performance
        pipe = redis_client.pipeline()
        for key in keys:
            pipe.get(key)
        values = pipe.execute()
        
        best_sim = -1.0
        best_entry = None
        stale_keys = []
        
        threshold = getattr(config, "CACHE_SIMILARITY_THRESHOLD", 0.75)
        
        for key, val_bytes in zip(keys, values):
            if not val_bytes:
                # Cache entry expired via TTL but still in keys set
                stale_keys.append(key)
                continue
                
            try:
                entry = json.loads(val_bytes)
                # Check for invalidation due to corpus version mismatch
                if str(entry.get("corpus_version", "")) != current_version:
                    stale_keys.append(key)
                    # Delete stale key
                    redis_client.delete(key)
                    continue
                    
                cached_embedding = entry.get("embedding")
                if cached_embedding:
                    sim = cosine_similarity(query_embedding, cached_embedding)
                    if sim > best_sim:
                        best_sim = sim
                        best_entry = entry
            except Exception as e:
                logger.error(f"Error parsing cache entry {key}: {e}")
                
        # Clean up stale keys from set in background
        if stale_keys:
            redis_client.srem("semantic:cache:keys", *stale_keys)
            
        if best_entry and best_sim >= threshold:
            logger.info(f"Semantic cache HIT: similarity {best_sim:.4f} >= threshold {threshold}")
            # Increment cache hit rate metric
            redis_client.incr("metrics:semantic_cache_hits")
            return {
                "answer_text": best_entry["answer_text"],
                "answer": best_entry["answer_text"],  # legacy compatibility
                "citations": best_entry["citations"],
                "confidence": best_entry.get("confidence", 0.85),
                "cached": True
            }
            
    except Exception as e:
        logger.error(f"Error checking semantic cache: {e}")
        
    logger.info("Semantic cache MISS")
    redis_client.incr("metrics:semantic_cache_misses")
    return None

def set_semantic_cache(query: str, query_embedding: list[float], answer_text: str, citations: list[dict], confidence: float):
    """Store the query, embedding, and response details in Redis with a 24-hour TTL."""
    if hasattr(query_embedding, "tolist"):
        query_embedding = query_embedding.tolist()
    redis_client = get_redis()
    if not redis_client:
        return
        
    try:
        current_version_raw = redis_client.get("corpus:version")
        if isinstance(current_version_raw, bytes):
            current_version = current_version_raw.decode("utf-8")
        elif isinstance(current_version_raw, str):
            current_version = current_version_raw
        else:
            current_version = "0"
        
        # Calculate SHA256 of the query to create a unique, deterministic cache key
        query_hash = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()
        key = f"semantic:cache:entry:{query_hash}"
        
        entry = {
            "query": query,
            "embedding": query_embedding,
            "answer_text": answer_text,
            "citations": citations,
            "confidence": confidence,
            "corpus_version": current_version,
            "timestamp": datetime_string()
        }
        
        # Save entry with a 24-hour TTL (86400 seconds)
        redis_client.setex(key, 86400, json.dumps(entry))
        # Track the key in our semantic cache set
        redis_client.sadd("semantic:cache:keys", key)
        logger.info(f"Saved query '{query[:30]}...' to semantic cache under key {key}")
        
    except Exception as e:
        logger.error(f"Error writing to semantic cache: {e}")

def datetime_string() -> str:
    from datetime import datetime
    return datetime.utcnow().isoformat() + "Z"
