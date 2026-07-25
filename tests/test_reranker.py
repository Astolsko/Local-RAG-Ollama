"""Task 2.3 — reranker backend selection, score normalization, and the skip shortcut."""
import math
from unittest.mock import MagicMock

import pytest


def _candidates():
    return [
        ("c1", {"doc": "least relevant", "meta": {}}),
        ("c2", {"doc": "most relevant", "meta": {}}),
    ]


@pytest.fixture
def fresh_reranker(monkeypatch):
    """Reranker with its module-level model cache cleared."""
    from backend.retrieval import reranker
    monkeypatch.setattr(reranker, "_MODEL", None)
    monkeypatch.setattr(reranker, "_LOADED_BACKEND", None)
    import config
    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    monkeypatch.setattr(config, "RERANK_TOP_K", 5)
    return reranker


def test_flashrank_scores_map_back_to_the_right_candidates(fresh_reranker, monkeypatch):
    """FlashRank returns results sorted by score, so its rows must be matched by id —
    zipping them against the input order would attach scores to the wrong chunks."""
    import config
    monkeypatch.setattr(config, "RERANKER", "flashrank")

    fake = MagicMock()
    fake.rerank.return_value = [{"id": 1, "score": 0.92}, {"id": 0, "score": 0.11}]
    monkeypatch.setattr(fresh_reranker, "_load_model", lambda: (fake, "flashrank"))

    out = fresh_reranker.rerank("query", _candidates())
    assert [cid for cid, _ in out] == ["c2", "c1"]
    assert out[0][1]["rerank_score"] == 0.92
    assert out[1][1]["rerank_score"] == 0.11


def test_cross_encoder_logits_are_squashed_onto_the_shared_scale(fresh_reranker, monkeypatch):
    """bge-reranker emits unbounded logits; both backends must report 0..1 so that
    confidence scoring does not need to know which one ran."""
    import config
    monkeypatch.setattr(config, "RERANKER", "cross-encoder")

    fake = MagicMock()
    fake.predict.return_value = [-2.0, 3.0]
    monkeypatch.setattr(fresh_reranker, "_load_model", lambda: (fake, "cross-encoder"))

    out = fresh_reranker.rerank("query", _candidates())
    assert [cid for cid, _ in out] == ["c2", "c1"]
    assert math.isclose(out[0][1]["rerank_score"], 1 / (1 + math.exp(-3.0)))
    assert all(0.0 <= p["rerank_score"] <= 1.0 for _, p in out)


def test_rerank_off_returns_candidates_untouched(fresh_reranker, monkeypatch):
    import config
    monkeypatch.setattr(config, "RERANKER", "off")
    cands = _candidates()
    assert fresh_reranker.rerank("query", cands) is cands


def test_backend_failure_falls_back_to_fusion_order(fresh_reranker, monkeypatch):
    import config
    monkeypatch.setattr(config, "RERANKER", "flashrank")

    fake = MagicMock()
    fake.rerank.side_effect = RuntimeError("onnx exploded")
    monkeypatch.setattr(fresh_reranker, "_load_model", lambda: (fake, "flashrank"))

    cands = _candidates()
    assert [cid for cid, _ in fresh_reranker.rerank("query", cands)] == ["c1", "c2"]


def test_rerank_is_skipped_when_the_top_dense_hit_is_confident(monkeypatch):
    """RERANK_SKIP_THRESHOLD: a near-exact dense match leaves the reranker nothing to
    reorder, so it should not be paid for."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 1
    mock_collection.query.return_value = {
        "ids": [["c1"]],
        "documents": [["an exact match"]],
        "metadatas": [[{"source_name": "s1", "source_id": "id-s1", "chunk_index": 0}]],
        "distances": [[0.05]],  # cosine similarity 0.95
    }
    monkeypatch.setattr("backend.rag._collection", mock_collection)
    monkeypatch.setattr("backend.embeddings.embed_texts", lambda ts: [[0.1, 0.2] for _ in ts])
    monkeypatch.setattr("backend.retrieval.bm25_index._load_index", lambda: None)

    import config
    monkeypatch.setattr(config, "ENABLE_QUERY_REWRITE", False)
    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    monkeypatch.setattr(config, "RERANK_SKIP_THRESHOLD", 0.85)

    called = []
    monkeypatch.setattr("backend.retrieval.hybrid_search.rerank",
                        lambda q, c: called.append(1) or c)

    from backend.retrieval.hybrid_search import hybrid_search_sync
    docs, metas, ids, metrics = hybrid_search_sync("exact match", top_k=5)

    assert not called, "reranker ran despite a 0.95 dense similarity"
    assert metrics["rerank_skipped"] is True
    assert metrics["rerank_latency"] == 0.0
    assert metas[0]["distance"] == 0.05  # surfaced for the confidence fallback


def test_rerank_runs_when_dense_similarity_is_weak(monkeypatch):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 1
    mock_collection.query.return_value = {
        "ids": [["c1"]],
        "documents": [["a loose match"]],
        "metadatas": [[{"source_name": "s1", "source_id": "id-s1", "chunk_index": 0}]],
        "distances": [[0.6]],  # cosine similarity 0.4
    }
    monkeypatch.setattr("backend.rag._collection", mock_collection)
    monkeypatch.setattr("backend.embeddings.embed_texts", lambda ts: [[0.1, 0.2] for _ in ts])
    monkeypatch.setattr("backend.retrieval.bm25_index._load_index", lambda: None)

    import config
    monkeypatch.setattr(config, "ENABLE_QUERY_REWRITE", False)
    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    monkeypatch.setattr(config, "RERANK_SKIP_THRESHOLD", 0.85)

    def fake_rerank(query, candidates):
        for _, payload in candidates:
            payload["rerank_score"] = 0.77
        return candidates

    monkeypatch.setattr("backend.retrieval.hybrid_search.rerank", fake_rerank)

    from backend.retrieval.hybrid_search import hybrid_search_sync
    docs, metas, ids, metrics = hybrid_search_sync("loose match", top_k=5)

    assert metrics["rerank_skipped"] is False
    assert metas[0]["rerank_score"] == 0.77
