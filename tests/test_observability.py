import os
import pytest
import sqlite3
import json
from pathlib import Path
from fastapi.testclient import TestClient
from backend.main import app

def test_database_logging_and_api(tmp_path, monkeypatch):
    # Setup temporary database file and temp data directory
    temp_db_path = tmp_path / "metrics.db"
    temp_jsonl_path = tmp_path / "request_logs.jsonl"
    
    monkeypatch.setattr("backend.observability.metrics_db.DB_PATH", temp_db_path)
    monkeypatch.setattr("backend.observability.logger.JSONL_PATH", temp_jsonl_path)
    monkeypatch.setattr("config.DATA_DIR", tmp_path)
    
    from backend.observability.logger import log_query
    from backend.observability.metrics_db import init_db
    
    # 1. Initialize DB
    init_db()
    assert temp_db_path.exists()
    
    # 2. Log request
    log_query(
        request_id="req-123",
        query="what is the price of oil?",
        rewritten_query="rewritten query",
        retrieved_chunk_ids=["chunk1", "chunk2"],
        rerank_scores=[0.8, 0.6],
        cache_hit=False,
        embed_latency=0.015,
        retrieve_latency=0.045,
        rerank_latency=0.020,
        generate_latency=1.2,
        total_latency=1.28,
        tokens_used=150,
        refusal=False
    )
    
    # Check JSONL log file
    assert temp_jsonl_path.exists()
    with open(temp_jsonl_path, "r", encoding="utf-8") as f:
        log_lines = f.readlines()
    assert len(log_lines) == 1
    log_data = json.loads(log_lines[0])
    assert log_data["request_id"] == "req-123"
    assert log_data["latency_breakdown"]["embed"] == 15.0  # 0.015 * 1000
    
    # Check SQLite DB
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM request_logs WHERE request_id = 'req-123'")
    row = cursor.fetchone()
    conn.close()
    
    assert row is not None
    assert row[0] == "req-123"
    assert row[2] == "what is the price of oil?"
    assert row[3] == "rewritten query"
    
    # 3. Test API Endpoint - submit feedback
    client = TestClient(app)
    
    response = client.post("/api/observability/feedback", json={"request_id": "req-123", "feedback": 1})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    
    # Verify feedback was updated in DB
    conn = sqlite3.connect(temp_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT user_feedback FROM request_logs WHERE request_id = 'req-123'")
    fb = cursor.fetchone()[0]
    conn.close()
    assert fb == 1
    
    # 4. Test API Endpoint - fetch metrics with new columns
    response = client.get("/api/observability/metrics")
    assert response.status_code == 200
    metrics_data = response.json()
    
    assert metrics_data["summary"]["total_requests"] == 1
    assert metrics_data["summary"]["thumbs_up"] == 1
    assert metrics_data["summary"]["thumbs_down"] == 0
    assert "avg_embed_latency" in metrics_data["summary"]
    assert "avg_bm25_latency" in metrics_data["summary"]
    assert "avg_vector_latency" in metrics_data["summary"]
    assert "avg_rrf_latency" in metrics_data["summary"]
    assert "avg_ttft_latency" in metrics_data["summary"]
    assert len(metrics_data["daily"]) == 1
    assert metrics_data["daily"][0]["requests"] == 1
    assert "bm25_latency" in metrics_data["daily"][0]
    assert "vector_latency" in metrics_data["daily"][0]
    assert "rrf_latency" in metrics_data["daily"][0]
    assert "ttft_latency" in metrics_data["daily"][0]
    assert "cache_check_latency" in metrics_data["daily"][0]
    assert "avg_faithfulness" in metrics_data["daily"][0]
