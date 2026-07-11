import pytest
from unittest.mock import MagicMock, patch

def test_split_into_sentences():
    from backend.ingestion.chunker import split_into_sentences
    text = "Hello world! This is sentence two. Dr. John is here."
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "Hello world!"
    assert sentences[1] == "This is sentence two."
    assert sentences[2] == "Dr. John is here."

def test_cosine_distance():
    from backend.ingestion.chunker import cosine_distance
    v1 = [1.0, 0.0]
    v2 = [1.0, 0.0]
    # Cosine distance between identical vectors is 0
    assert abs(cosine_distance(v1, v2)) < 1e-6

    v3 = [0.0, 1.0]
    # Cosine distance between orthogonal vectors is 1
    assert abs(cosine_distance(v1, v3) - 1.0) < 1e-6

def test_semantic_chunk_text_fallback():
    # If embedding fails, it should fallback to recursive character splitter
    from backend.ingestion.chunker import semantic_chunk_text
    
    with patch("backend.ingestion.chunker.embed_texts", side_effect=Exception("API Error")):
        text = "Hello. World. This is a fallback test."
        chunks = semantic_chunk_text(text, chunk_size=20, chunk_overlap=0)
        assert len(chunks) > 1

def test_add_source_caching_check(monkeypatch):
    mock_collection = MagicMock()
    # Mocking that the source with name "doc_1" already exists with matching hash
    mock_collection.get.return_value = {
        "ids": ["id-1", "id-2"],
        "metadatas": [{"doc_hash": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824", "source_id": "cached-id-123"}]
    }
    monkeypatch.setattr("backend.rag._collection", mock_collection)
    monkeypatch.setattr("backend.retrieval.bm25_index.build_bm25_index", lambda: None)
    
    from backend.rag import add_source
    res = add_source("doc_1", "hello") # SHA-256 for "hello" is 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
    
    assert res["skipped"] is True
    assert res["id"] == "cached-id-123"
    assert res["chunks"] == 2
    # Ensure contextualizer and embedding are not called (skipped)
    mock_collection.add.assert_not_called()
