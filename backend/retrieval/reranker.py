import os
from typing import List, Tuple, Dict, Any

import config

# Lazy-load the cross‑encoder model; keep a module‑level cache
_MODEL = None

def _load_model():
    """Load the cross-encoder model defined in config.RERANK_MODEL.
    The function is called once at first use and caches the model in the module.
    Returns None if reranking is disabled.
    """
    global _MODEL
    if not getattr(config, "RERANK_ENABLED", False):
        return None
    if _MODEL is None:
        try:
            from sentence_transformers import CrossEncoder
            _MODEL = CrossEncoder(config.RERANK_MODEL)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Reranker model '{config.RERANK_MODEL}' could not be loaded (falling back to RRF fusion): {e}"
            )
            _MODEL = None
    return _MODEL

def rerank(query: str, candidates: List[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
    """Rerank a list of candidates using the cross‑encoder.

    Args:
        query: The user query string.
        candidates: List of (doc_id, payload) where payload contains at least a ``doc`` key.

    Returns:
        A list of (doc_id, payload) sorted by relevance score, truncated to
        ``config.RERANK_TOP_K`` (defaults to 5). If reranking is disabled or the model
        fails to load, the original ordering is returned.
    """
    model = _load_model()
    if model is None:
        return candidates
    texts = [(query, payload.get("doc", "")) for _, payload in candidates]
    scores = model.predict(texts)
    scored = [(cand, score) for cand, score in zip(candidates, scores)]
    for cand, score in scored:
        cand[1]["rerank_score"] = float(score)
    scored.sort(key=lambda x: x[1], reverse=True)
    top_k = getattr(config, "RERANK_TOP_K", 5)
    return [cand for cand, _ in scored[:top_k]]
