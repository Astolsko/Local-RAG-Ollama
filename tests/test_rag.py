import pytest
from unittest.mock import MagicMock

def test_chunk_text(monkeypatch):
    import config
    monkeypatch.setattr(config, "CHUNK_SIZE", 500)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 50)
    
    from backend.rag import chunk_text
    assert chunk_text("") == []
    assert chunk_text("hello world") == ["hello world"]
    
    # Check boundaries
    long_text = "a" * 1200
    chunks = chunk_text(long_text)
    assert len(chunks) >= 3

def test_retrieve_context_no_sources(monkeypatch):
    # Mock collection count to be 0
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    monkeypatch.setattr("backend.rag._collection", mock_collection)

    from backend.rag import _retrieve_context

    with pytest.raises(ValueError, match="No sources indexed yet"):
        _retrieve_context("query", "sys")

def test_retrieve_context_with_sources(monkeypatch):
    mock_collection = MagicMock()
    mock_collection.count.return_value = 2
    mock_collection.query.return_value = {
        "documents": [["doc1", "doc2"]],
        "metadatas": [[{"source_name": "s1", "chunk_index": 0, "source_id": "id-s1"}, {"source_name": "s2", "chunk_index": 0, "source_id": "id-s2"}]],
        "ids": [["id1", "id2"]]
    }
    monkeypatch.setattr("backend.rag._collection", mock_collection)
    monkeypatch.setattr("backend.rag.get_cached_rag", lambda *a: None)
    monkeypatch.setattr("backend.rag.set_cached_rag", lambda *a: None)
    import config
    monkeypatch.setattr(config, "RERANK_ENABLED", False)
    monkeypatch.setattr(config, "ENABLE_QUERY_REWRITE", False)
    monkeypatch.setattr("backend.retrieval.hybrid_search.search_bm25", lambda *a, **k: ([], [], []))

    from backend.rag import _retrieve_context
    context, citations, from_cache = _retrieve_context("query", "sys", ["id-s1", "id-s2"])
    assert "[s1]" in context
    assert "[s2]" in context
    assert len(citations) == 2
    assert from_cache is False
