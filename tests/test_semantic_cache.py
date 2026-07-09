import sys
import math
from pathlib import Path

# Add workspace root and backend directories to path
root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from backend.cache.semantic_cache import cosine_similarity

def test_cosine_similarity():
    # Exact match
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert math.isclose(cosine_similarity(v1, v2), 1.0)
    
    # Orthogonal (no similarity)
    v3 = [0.0, 1.0, 0.0]
    assert math.isclose(cosine_similarity(v1, v3), 0.0)
    
    # Opposite
    v4 = [-1.0, 0.0, 0.0]
    assert math.isclose(cosine_similarity(v1, v4), -1.0)
    
    # Normal vector similarity
    v5 = [1.0, 1.0, 0.0]
    expected = 1.0 / math.sqrt(2.0)
    assert math.isclose(cosine_similarity(v1, v5), expected)
    
    print("Unit test: test_cosine_similarity passed successfully.")

def test_semantic_cache_redis_string_decoding(monkeypatch):
    from unittest.mock import MagicMock
    from backend.cache.semantic_cache import check_semantic_cache

    mock_redis = MagicMock()
    # Mock get("corpus:version") to return a string (as when decode_responses=True is active)
    mock_redis.get.return_value = "5"
    mock_redis.smembers.return_value = {b"key1"}
    
    # Pipeline execute returns serialized cached entry
    mock_redis.pipeline.return_value.execute.return_value = [
        b'{"query": "test", "embedding": [1.0], "answer_text": "ans", "citations": [], "corpus_version": "5"}'
    ]
    
    monkeypatch.setattr("backend.cache.semantic_cache.get_redis", lambda: mock_redis)
    
    # Calling check_semantic_cache should now succeed without AttributeError
    res = check_semantic_cache("test", [1.0])
    assert res is not None
    assert res["answer_text"] == "ans"

if __name__ == "__main__":
    test_cosine_similarity()

