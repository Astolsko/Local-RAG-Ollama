import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

def test_health_endpoint(monkeypatch):
    # Mock main.py's list_sources, list_chats, require_redis, redis_ping
    monkeypatch.setattr("backend.main.list_sources", lambda: [])
    monkeypatch.setattr("backend.main.list_chats", lambda: [])
    monkeypatch.setattr("backend.main.redis_ping", lambda: True)

    from backend.main import app
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["redis"] is True

def test_system_prompt_endpoints(tmp_path, monkeypatch):
    # Re-route system prompt file to a temp file
    monkeypatch.setattr("config.SYSTEM_PROMPT_FILE", tmp_path / "system_prompt.txt")
    monkeypatch.setattr("backend.rag.config.SYSTEM_PROMPT_FILE", tmp_path / "system_prompt.txt")
    
    # Mock redis
    mock_redis = MagicMock()
    for prefix in ["redis_store", "backend.redis_store"]:
        try:
            monkeypatch.setattr(f"{prefix}.get_redis", lambda: mock_redis)
            monkeypatch.setattr(f"{prefix}.require_redis", lambda: mock_redis)
        except Exception:
            pass
    monkeypatch.setattr("backend.main.require_redis", lambda: mock_redis)

    from backend.main import app
    client = TestClient(app)
    
    # GET
    response = client.get("/api/settings/system-prompt")
    assert response.status_code == 200
    assert "study and Q&A assistant" in response.json()["text"]

    # PUT (should now succeed and return 200)
    response = client.put("/api/settings/system-prompt", json={"text": "This is a new system prompt"})
    print("DEBUG RESPONSE JSON:", response.json())
    assert response.status_code == 200
    assert response.json()["text"] == "This is a new system prompt"

    # GET again to verify persistence
    response = client.get("/api/settings/system-prompt")
    assert response.status_code == 200
    assert response.json()["text"] == "This is a new system prompt"
