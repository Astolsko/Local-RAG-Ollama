"""Full original text of every ingested document, keyed by source id.

Chunks carry ``char_start``/``char_end`` offsets into this text, which is what lets the UI
highlight a cited span inside the document it came from. Chroma stores chunks, not the
source, so the text has to live somewhere — here.

Its own SQLite file (``data/documents.db``, honoring RAG_DATA_DIR), matching how the graph
store keeps ``data/graph.db`` separate from the metrics DB.
"""
import sqlite3
from datetime import datetime, timezone

import config

DB_PATH = config.DATA_DIR / "documents.db"


def _conn() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS documents(
                id TEXT PRIMARY KEY, name TEXT, text TEXT, created_at TEXT)
            """
        )


def save_document(doc_id: str, name: str, text: str) -> None:
    """Store (or replace) a document's full text."""
    init_db()
    with _conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO documents(id, name, text, created_at) VALUES (?,?,?,?)",
            (doc_id, name, text, datetime.now(timezone.utc).isoformat()),
        )


def get_document(doc_id: str) -> dict | None:
    init_db()
    with _conn() as c:
        row = c.execute(
            "SELECT id, name, text, created_at FROM documents WHERE id=?", (doc_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_document(doc_id: str) -> None:
    init_db()
    with _conn() as c:
        c.execute("DELETE FROM documents WHERE id=?", (doc_id,))
