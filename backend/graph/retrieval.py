"""Graph retrieval paths: relational (entity-neighborhood) and global (community summaries).

Relational returns extra (chunk_id, doc, meta) candidates the caller merges with the vector
path and reranks together. Global returns community summaries as pseudo-chunks. Both degrade
to empty when the graph is sparse, so a misroute costs quality, never correctness.
"""
import json
import logging

import config
from backend.graph import store

logger = logging.getLogger(__name__)


def _link_query_entities(query: str) -> list[int]:
    """Entity ids whose norm_name is a substring of the query (fuzzy-ish, cheap)."""
    q = query.lower()
    with store._conn() as c:
        rows = c.execute("SELECT id, norm_name FROM entities").fetchall()
    return [r["id"] for r in rows if r["norm_name"] and len(r["norm_name"]) >= 3 and r["norm_name"] in q]


def _neighborhood(entity_ids: list[int], hops: int) -> set[int]:
    g = store.load_networkx()
    seen = set(e for e in entity_ids if e in g)
    frontier = set(seen)
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            nxt.update(g.neighbors(n))
        frontier = nxt - seen
        seen |= nxt
        if not frontier:
            break
    return seen


def relational_candidates(query: str) -> list[tuple[str, dict]]:
    """(chunk_id, {doc, meta}) candidates from the query entities' k-hop neighborhood."""
    entity_ids = _link_query_entities(query)
    if not entity_ids:
        return []
    hops = getattr(config, "GRAPH_MAX_HOPS", 2)
    nodes = _neighborhood(entity_ids, hops)
    if not nodes:
        return []

    with store._conn() as c:
        placeholders = ",".join("?" * len(nodes))
        rows = c.execute(
            f"SELECT DISTINCT chunk_id FROM entity_chunks WHERE entity_id IN ({placeholders})",
            tuple(nodes),
        ).fetchall()
    chunk_ids = [r["chunk_id"] for r in rows]
    if not chunk_ids:
        return []

    try:
        from backend import rag
        got = rag._collection.get(ids=chunk_ids, include=["documents", "metadatas"])
    except Exception as e:
        logger.warning(f"relational chunk fetch failed: {e}")
        return []

    out = []
    for cid, doc, meta in zip(got.get("ids", []), got.get("documents", []), got.get("metadatas", [])):
        out.append((cid, {"doc": doc, "meta": meta}))
    return out


def community_summaries() -> list[dict]:
    """All stored community summaries: [{id, summary, entity_ids}]."""
    store.init_db()
    with store._conn() as c:
        rows = c.execute("SELECT id, summary, entity_ids FROM communities WHERE summary != ''").fetchall()
    return [{"id": r["id"], "summary": r["summary"], "entity_ids": json.loads(r["entity_ids"])} for r in rows]
