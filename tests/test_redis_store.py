import pytest
from unittest.mock import MagicMock
import json

def test_rag_cache_key(monkeypatch):
    # Mock redis client
    mock_redis = MagicMock()
    monkeypatch.setattr("backend.redis_store.get_redis", lambda: mock_redis)
    monkeypatch.setattr("backend.redis_store.require_redis", lambda: mock_redis)

    from backend.redis_store import _rag_cache_key, get_cached_rag, set_cached_rag

    key1 = _rag_cache_key("hello", 5, "prompt1", ["id1", "id2"])
    key2 = _rag_cache_key("hello", 5, "prompt1", ["id2", "id1"])
    assert key1 == key2  # Sorted ids should produce same key

    # Test set/get cache
    mock_redis.get.return_value = '{"answer": "cached!"}'
    data = get_cached_rag("hello", 5, "prompt1", ["id1"])
    assert data == {"answer": "cached!"}
    mock_redis.get.assert_called_once()
