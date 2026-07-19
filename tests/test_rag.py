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

def test_should_rewrite_skips_short_self_contained_queries():
    """Rewriting costs an LLM call; short standalone queries must not pay it."""
    from backend.retrieval.query_rewrite import should_rewrite

    # short and self-contained -> skip
    assert not should_rewrite("What is the total connected load?")
    assert not should_rewrite("Kestrel Terminal budget")

    # long enough to be worth expanding -> rewrite
    assert should_rewrite("What is the total connected electrical load for the Kestrel automated cargo system?")

    # short but refers back to something -> rewrite
    assert should_rewrite("what does it cost")
    assert should_rewrite("who runs that?")

    # a follow-up turn may depend on earlier context -> always rewrite
    assert should_rewrite("and the draft?", has_history=True)

    assert not should_rewrite("   ")


def test_hybrid_search_filters_both_paths_by_source_ids(monkeypatch):
    """Selecting a source must restrict dense AND sparse retrieval, not just gate RAG on/off."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 2
    mock_collection.query.return_value = {
        "ids": [["id-s1:0"]],
        "documents": [["text from s1"]],
        "metadatas": [[{"source_name": "s1", "source_id": "id-s1", "chunk_index": 0}]],
    }
    monkeypatch.setattr("backend.rag._collection", mock_collection)

    import config
    monkeypatch.setattr(config, "RERANK_ENABLED", False)
    monkeypatch.setattr(config, "ENABLE_QUERY_REWRITE", False)

    # s2 scores higher on BM25, so an unfiltered sparse path would surface it
    monkeypatch.setattr(
        "backend.retrieval.bm25_index._load_index",
        lambda: {
            "bm25": MagicMock(get_scores=lambda q: [9.0, 1.0]),
            "ids": ["id-s2:0", "id-s1:0"],
            "docs": ["text from s2", "text from s1"],
            "metas": [
                {"source_name": "s2", "source_id": "id-s2", "chunk_index": 0},
                {"source_name": "s1", "source_id": "id-s1", "chunk_index": 0},
            ],
        },
    )

    from backend.retrieval.hybrid_search import hybrid_search_sync
    docs, metas, ids, _ = hybrid_search_sync("query", top_k=5, source_ids=["id-s1"])

    # dense path is filtered by Chroma itself
    assert mock_collection.query.call_args.kwargs["where"] == {"source_id": "id-s1"}
    # sparse path drops the higher-scoring foreign chunk
    assert docs, "expected at least one chunk from the selected source"
    assert all(m["source_id"] == "id-s1" for m in metas)
    assert "text from s2" not in docs
