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
    monkeypatch.setattr("backend.embeddings.embed_texts", lambda ts: [[0.1, 0.2] for _ in ts])

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


def test_queries_embedded_once_and_rerank_score_reaches_metadata(monkeypatch):
    """Two regressions guarded here.

    1. Every query is embedded in a single batched call and handed to Chroma as vectors.
       Passing `query_texts` made Chroma embed each rewritten query itself, so an expanded
       query paid N extra Ollama round-trips on top of the one the caller already made.
    2. The rerank score must land on `meta` — chat_stream/rag build citations from the
       metadata dicts, so reading it off the payload left confidence pinned at its 0.85
       fallback for every answer.
    """
    mock_collection = MagicMock()
    mock_collection.count.return_value = 2
    mock_collection.query.return_value = {
        "ids": [["c1"]],
        "documents": [["doc one"]],
        "metadatas": [[{"source_name": "s1", "source_id": "id-s1", "chunk_index": 0}]],
    }
    monkeypatch.setattr("backend.rag._collection", mock_collection)

    embed_calls = []

    def fake_embed(texts):
        embed_calls.append(list(texts))
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr("backend.embeddings.embed_texts", fake_embed)
    monkeypatch.setattr(
        "backend.retrieval.bm25_index._load_index",
        lambda: {
            "bm25": MagicMock(get_scores=lambda q: [1.0]),
            "ids": ["c1"],
            "docs": ["doc one"],
            "metas": [{"source_name": "s1", "source_id": "id-s1", "chunk_index": 0}],
        },
    )

    import config
    monkeypatch.setattr(config, "ENABLE_QUERY_REWRITE", True)
    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    monkeypatch.setattr("backend.retrieval.hybrid_search.should_rewrite", lambda q, h: True)
    monkeypatch.setattr("backend.retrieval.hybrid_search.rewrite_query",
                        lambda q, h: [q, "alt one", "alt two"])

    def fake_rerank(query, candidates):
        for _, payload in candidates:
            payload["rerank_score"] = 2.5
        return candidates

    monkeypatch.setattr("backend.retrieval.hybrid_search.rerank", fake_rerank)

    from backend.retrieval.hybrid_search import hybrid_search_sync
    question = "a long question about the terminal cargo handling system"
    docs, metas, ids, _ = hybrid_search_sync(question, top_k=5)

    # one batched embed covering every rewritten query, not one call per query
    assert len(embed_calls) == 1
    assert embed_calls[0] == [question, "alt one", "alt two"]
    # Chroma searched the vectors we supplied and never re-embedded the text itself
    assert mock_collection.query.call_args.kwargs["query_embeddings"] == [[0.1, 0.2]]
    assert "query_texts" not in mock_collection.query.call_args.kwargs
    # the score the citation/confidence path actually reads
    assert metas[0]["rerank_score"] == 2.5


def test_bm25_index_is_not_reloaded_from_disk_every_search(monkeypatch, tmp_path):
    """The pickle was deserialized per search, so one rewritten query paid four full
    corpus loads. Cache is keyed on mtime, so a rebuild still invalidates it."""
    import pickle
    from backend.retrieval import bm25_index

    index_file = tmp_path / "bm25_index.pkl"

    def write_index(docs):
        from rank_bm25 import BM25Okapi
        payload = {
            "bm25": BM25Okapi([bm25_index.tokenize(d) for d in docs]),
            "ids": [f"c{i}" for i in range(len(docs))],
            "docs": docs,
            "metas": [{"source_name": "s", "source_id": "id-s", "chunk_index": i}
                      for i in range(len(docs))],
        }
        with open(index_file, "wb") as f:
            pickle.dump(payload, f)

    write_index(["harbor cargo terminal", "annual risk report"])
    monkeypatch.setattr(bm25_index, "INDEX_PATH", index_file)
    monkeypatch.setattr(bm25_index, "_CACHE", None)

    loads = []
    real_load = pickle.load
    monkeypatch.setattr(pickle, "load", lambda f: (loads.append(1), real_load(f))[1])

    for _ in range(4):
        ids, _docs, _metas = bm25_index.search_bm25("cargo", top_k=2)
    assert ids, "expected a hit from the index"
    assert len(loads) == 1, f"index deserialized {len(loads)} times, expected 1"

    # a rebuild bumps mtime -> the stale cache must be dropped
    import os
    write_index(["harbor cargo terminal", "annual risk report", "third doc"])
    os.utime(index_file, (index_file.stat().st_atime, index_file.stat().st_mtime + 10))
    bm25_index.search_bm25("cargo", top_k=2)
    assert len(loads) == 2, "rebuilt index was not reloaded"
