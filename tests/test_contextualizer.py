import pytest
import hashlib
from unittest.mock import MagicMock, patch
from backend.ingestion.contextualizer import Contextualizer

def test_contextualizer_fallback_cache_hit():
    doc_text = "Main document body text."
    chunk_text = "Individual chunk."
    hash_input = f"{doc_text}|||{chunk_text}"
    key = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    contextualizer = Contextualizer()
    contextualizer.fallback_cache[key] = "This is a cached blurb."

    # Force redis client to return None to test fallback cache pathway
    with patch.object(contextualizer, "_get_redis_client", return_value=None):
        with patch("httpx.Client.post") as mock_post:
            blurb = contextualizer.get_blurb(doc_text, chunk_text)
            assert blurb == "This is a cached blurb."
            mock_post.assert_not_called()

def test_contextualizer_fallback_cache_miss_generation():
    doc_text = "Main document body text."
    chunk_text = "Individual chunk."
    hash_input = f"{doc_text}|||{chunk_text}"
    key = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    contextualizer = Contextualizer()

    # Mock success response from Ollama generate API
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": '"Generated blurb via Ollama."'}
    
    with patch.object(contextualizer, "_get_redis_client", return_value=None):
        with patch("httpx.Client.post", return_value=mock_response) as mock_post:
            blurb = contextualizer.get_blurb(doc_text, chunk_text)
            assert blurb == "Generated blurb via Ollama."
            
            # Verify fallback cache is updated
            assert contextualizer.fallback_cache[key] == "Generated blurb via Ollama."

def test_contextualizer_redis_cache_hit():
    doc_text = "Main document body text."
    chunk_text = "Individual chunk."
    hash_input = f"{doc_text}|||{chunk_text}"
    key = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
    redis_key = f"context:blurb:{key}"

    contextualizer = Contextualizer()
    mock_redis = MagicMock()
    mock_redis.get.return_value = "Cached blurb from Redis"

    with patch.object(contextualizer, "_get_redis_client", return_value=mock_redis):
        with patch("httpx.Client.post") as mock_post:
            blurb = contextualizer.get_blurb(doc_text, chunk_text)
            assert blurb == "Cached blurb from Redis"
            mock_redis.get.assert_called_once_with(redis_key)
            mock_post.assert_not_called()

def test_contextualizer_redis_cache_miss_generation():
    doc_text = "Main document body text."
    chunk_text = "Individual chunk."
    hash_input = f"{doc_text}|||{chunk_text}"
    key = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
    redis_key = f"context:blurb:{key}"

    contextualizer = Contextualizer()
    mock_redis = MagicMock()
    mock_redis.get.return_value = None

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "New generated blurb"}

    with patch.object(contextualizer, "_get_redis_client", return_value=mock_redis):
        with patch("httpx.Client.post", return_value=mock_response):
            blurb = contextualizer.get_blurb(doc_text, chunk_text)
            assert blurb == "New generated blurb"
            mock_redis.set.assert_called_once_with(redis_key, "New generated blurb")

def test_contextualizer_redis_batch_lookup():
    doc_text = "Whole document text."
    chunks = ["chunk 1", "chunk 2", "chunk 3"]
    
    contextualizer = Contextualizer()
    mock_redis = MagicMock()
    # Mock mget to return one cached value and two cache misses
    mock_redis.mget.return_value = ["Cached blurb 1", None, None]
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"response": "Generated blurb"}
    
    with patch.object(contextualizer, "_get_redis_client", return_value=mock_redis):
        with patch("httpx.Client.post", return_value=mock_response) as mock_post:
            blurbs = contextualizer.get_blurbs_batch(doc_text, chunks)
            
            assert len(blurbs) == 3
            assert blurbs[0] == "Cached blurb 1"
            assert blurbs[1] == "Generated blurb"
            assert blurbs[2] == "Generated blurb"
            
            # httpx should only be called twice
            assert mock_post.call_count == 2
            
            # set should be called twice to store the generated blurbs in Redis
            assert mock_redis.set.call_count == 2
