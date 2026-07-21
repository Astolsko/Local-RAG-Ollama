import sqlite3

import backend.retrieval.bm25_index as bm25
from backend.observability import metrics_db


def test_logging_path_does_not_rebuild_bm25(tmp_path, monkeypatch):
    """Task 2.0: answering/logging a request must not rebuild the whole BM25 index."""
    monkeypatch.setattr(metrics_db, "DB_PATH", tmp_path / "metrics.db")
    monkeypatch.setattr("config.DATA_DIR", tmp_path)

    calls = {"n": 0}
    monkeypatch.setattr(bm25, "build_bm25_index", lambda: calls.__setitem__("n", calls["n"] + 1))

    metrics_db.init_db()
    metrics_db.log_request(
        request_id="r1", query="q", rewritten_query=None,
        retrieved_chunk_ids=["c1"], rerank_scores=[0.5], cache_hit=False,
        embed_latency=0.0, retrieve_latency=0.0, rerank_latency=0.0,
        generate_latency=0.0, total_latency=0.0, tokens_used=1,
    )
    metrics_db.update_faithfulness("r1", 4)
    metrics_db.update_feedback("r1", 1)

    assert calls["n"] == 0, "BM25 index rebuilt on the request path"

    # sanity: the row was still written
    conn = sqlite3.connect(tmp_path / "metrics.db")
    assert conn.execute("SELECT COUNT(*) FROM request_logs").fetchone()[0] == 1
    conn.close()
