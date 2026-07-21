"""GraphRAG-lite store: entities, relations, entity↔chunk links, communities.

Lives in its own SQLite file (``data/graph.db``, honoring RAG_DATA_DIR) so it never
contends with the metrics DB. A networkx view is loaded on demand and cached until the
next write. Everything here is behind ``GRAPH_ENABLED`` at the call sites — the store
itself is inert until something writes to it.
"""
import re
import sqlite3
import threading

import config

DB_PATH = config.DATA_DIR / "graph.db"

_lock = threading.Lock()  # ponytail: global lock; graph writes are rare (background indexing)
_nx_cache = None


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _conn() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS entities(
                id INTEGER PRIMARY KEY, name TEXT, type TEXT, norm_name TEXT,
                UNIQUE(norm_name, type));
            CREATE TABLE IF NOT EXISTS relations(
                id INTEGER PRIMARY KEY, source_id INT, target_id INT,
                predicate TEXT, chunk_id TEXT, confidence REAL);
            CREATE TABLE IF NOT EXISTS entity_chunks(
                entity_id INT, chunk_id TEXT, doc_id TEXT,
                PRIMARY KEY(entity_id, chunk_id));
            CREATE TABLE IF NOT EXISTS communities(
                id INTEGER PRIMARY KEY, level INT, summary TEXT, entity_ids TEXT);
            """
        )


def _invalidate():
    global _nx_cache
    _nx_cache = None


def upsert_entity(name: str, etype: str) -> int:
    """Insert-or-get an entity by (norm_name, type). Returns its id."""
    norm = _norm(name)
    with _lock, _conn() as c:
        row = c.execute(
            "SELECT id FROM entities WHERE norm_name=? AND type=?", (norm, etype)
        ).fetchone()
        if row:
            return row["id"]
        cur = c.execute(
            "INSERT INTO entities(name, type, norm_name) VALUES(?,?,?)", (name, etype, norm)
        )
        _invalidate()
        return cur.lastrowid


def add_relation(source_id: int, target_id: int, predicate: str, chunk_id: str,
                 confidence: float = 1.0) -> None:
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO relations(source_id, target_id, predicate, chunk_id, confidence) "
            "VALUES(?,?,?,?,?)", (source_id, target_id, predicate, chunk_id, confidence)
        )
        _invalidate()


def link_entity_chunk(entity_id: int, chunk_id: str, doc_id: str) -> None:
    with _lock, _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO entity_chunks(entity_id, chunk_id, doc_id) VALUES(?,?,?)",
            (entity_id, chunk_id, doc_id),
        )


def delete_doc(doc_id: str) -> None:
    """Remove a document's entity links + relations, then drop orphaned entities."""
    with _lock, _conn() as c:
        chunk_rows = c.execute(
            "SELECT DISTINCT chunk_id FROM entity_chunks WHERE doc_id=?", (doc_id,)
        ).fetchall()
        chunk_ids = [r["chunk_id"] for r in chunk_rows]
        c.execute("DELETE FROM entity_chunks WHERE doc_id=?", (doc_id,))
        for cid in chunk_ids:
            c.execute("DELETE FROM relations WHERE chunk_id=?", (cid,))
        # drop entities no longer linked to any chunk
        c.execute(
            "DELETE FROM entities WHERE id NOT IN (SELECT entity_id FROM entity_chunks)"
        )
        _invalidate()


def counts() -> dict:
    with _conn() as c:
        return {
            "entities": c.execute("SELECT COUNT(*) FROM entities").fetchone()[0],
            "relations": c.execute("SELECT COUNT(*) FROM relations").fetchone()[0],
            "communities": c.execute("SELECT COUNT(*) FROM communities").fetchone()[0],
        }


def load_networkx():
    """Return an in-memory networkx.Graph of entities+relations (cached until next write)."""
    global _nx_cache
    if _nx_cache is not None:
        return _nx_cache
    import networkx as nx

    g = nx.Graph()
    with _conn() as c:
        for r in c.execute("SELECT id, name, type FROM entities"):
            g.add_node(r["id"], name=r["name"], type=r["type"])
        for r in c.execute("SELECT source_id, target_id, predicate FROM relations"):
            if r["source_id"] is None or r["target_id"] is None:
                continue
            g.add_edge(r["source_id"], r["target_id"], predicate=r["predicate"])
    _nx_cache = g
    return g
