"""Candidate reranking — FlashRank (default) or a SentenceTransformers cross-encoder.

`RERANKER=flashrank|cross-encoder|off` picks the backend. Both write a **normalized 0..1**
relevance score to `rerank_score`, so confidence scoring and the metrics dashboard never
have to know which one ran: FlashRank already sigmoids its ONNX logits, and the
cross-encoder's raw logits get squashed here to match.

Models load lazily on first use and are cached until the backend setting changes.
"""
import logging
import math
from typing import Any, Dict, List, Tuple

import config

logger = logging.getLogger(__name__)

_MODEL = None
_LOADED_BACKEND: str | None = None


def _selected_backend() -> str:
    if not getattr(config, "RERANK_ENABLED", False):
        return "off"
    return (getattr(config, "RERANKER", "flashrank") or "flashrank").strip().lower()


def _load_model() -> Tuple[Any, str]:
    """Return (model, backend). Model is None when reranking is off or unavailable."""
    global _MODEL, _LOADED_BACKEND
    backend = _selected_backend()
    if backend == "off":
        return None, "off"
    if _MODEL is not None and _LOADED_BACKEND == backend:
        return _MODEL, backend

    try:
        if backend == "flashrank":
            from flashrank import Ranker
            # Default cache_dir is /tmp, which is not a real path on Windows.
            _MODEL = Ranker(
                model_name=getattr(config, "FLASHRANK_MODEL", "ms-marco-MiniLM-L-12-v2"),
                cache_dir=str(config.DATA_DIR / "flashrank"),
                log_level="WARNING",
            )
        else:
            from sentence_transformers import CrossEncoder
            _MODEL = CrossEncoder(config.RERANK_MODEL)
        _LOADED_BACKEND = backend
    except Exception as e:
        logger.warning(f"Reranker backend '{backend}' could not be loaded "
                       f"(falling back to RRF fusion order): {e}")
        _MODEL, _LOADED_BACKEND = None, None
        return None, backend

    return _MODEL, backend


def _flashrank_scores(model, query: str, candidates) -> List[float]:
    from flashrank import RerankRequest
    passages = [{"id": i, "text": payload.get("doc", "")}
                for i, (_, payload) in enumerate(candidates)]
    ranked = model.rerank(RerankRequest(query=query, passages=passages))
    scores = [0.0] * len(candidates)
    for row in ranked:
        scores[int(row["id"])] = float(row["score"])  # already 0..1
    return scores


def _cross_encoder_scores(model, query: str, candidates) -> List[float]:
    raw = model.predict([(query, payload.get("doc", "")) for _, payload in candidates])
    # bge-reranker emits unbounded logits; squash to share FlashRank's 0..1 scale.
    return [1.0 / (1.0 + math.exp(-float(s))) for s in raw]


def rerank(query: str, candidates: List[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
    """Rerank candidates by relevance to the query.

    Args:
        query: The user query string.
        candidates: List of (doc_id, payload) where payload contains at least a ``doc`` key.

    Returns:
        (doc_id, payload) sorted by score, truncated to ``config.RERANK_TOP_K``. If
        reranking is disabled or the model fails to load, the original ordering is
        returned untouched.
    """
    model, backend = _load_model()
    if model is None or not candidates:
        return candidates

    try:
        scores = (_flashrank_scores(model, query, candidates) if backend == "flashrank"
                  else _cross_encoder_scores(model, query, candidates))
    except Exception as e:
        logger.warning(f"Reranking failed on backend '{backend}', keeping fusion order: {e}")
        return candidates

    for (_, payload), score in zip(candidates, scores):
        payload["rerank_score"] = score
    ranked = sorted(candidates, key=lambda c: c[1].get("rerank_score", 0.0), reverse=True)
    return ranked[:getattr(config, "RERANK_TOP_K", 5)]
