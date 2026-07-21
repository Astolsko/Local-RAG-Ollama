"""Community detection + LLM summaries over the entity graph (GraphRAG global layer).

Uses networkx's built-in Louvain (no python-louvain dependency). One summary per
community of >=3 entities, from its relations + a sample of linked chunk texts. Runs only
in the background indexing job. The LLM call is isolated in ``_summarize`` for testing.
"""
import json
import logging

import httpx

import config
from backend.graph import store

logger = logging.getLogger(__name__)

MIN_COMMUNITY_SIZE = 3


def detect() -> list[list[int]]:
    """Return communities as lists of entity ids (largest first)."""
    g = store.load_networkx()
    if g.number_of_nodes() == 0:
        return []
    from networkx.algorithms.community import louvain_communities
    comms = louvain_communities(g, seed=42)
    return sorted((sorted(int(n) for n in c) for c in comms), key=len, reverse=True)


def _relation_lines(entity_ids: list[int]) -> list[str]:
    ids = set(entity_ids)
    lines = []
    with store._conn() as c:
        rows = c.execute(
            "SELECT s.name AS s, r.predicate AS p, t.name AS t "
            "FROM relations r JOIN entities s ON r.source_id=s.id "
            "JOIN entities t ON r.target_id=t.id"
        ).fetchall()
    for row in rows:
        lines.append(f"{row['s']} —{row['p']}→ {row['t']}")
    return lines[:40]


def _chunk_sample(entity_ids: list[int], limit: int = 3) -> list[str]:
    if not entity_ids:
        return []
    with store._conn() as c:
        placeholders = ",".join("?" * len(entity_ids))
        rows = c.execute(
            f"SELECT DISTINCT chunk_id FROM entity_chunks WHERE entity_id IN ({placeholders}) LIMIT ?",
            (*entity_ids, limit),
        ).fetchall()
    chunk_ids = [r["chunk_id"] for r in rows]
    if not chunk_ids:
        return []
    # chunk_id is "{source_id}:{index}" — pull the text from Chroma
    try:
        from backend import rag
        got = rag._collection.get(ids=chunk_ids, include=["documents"])
        return got.get("documents", []) or []
    except Exception:
        return []


def _summarize(prompt: str, model: str) -> str:
    r = httpx.post(
        f"{config.OLLAMA_BASE}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": 0.2},
              "keep_alive": getattr(config, "OLLAMA_KEEP_ALIVE", "10m")},
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("response", "").strip()


def _gen_model() -> str:
    try:
        from backend.model_tiers import model_for
        return model_for("LLM_MODEL")
    except Exception:
        return config.LLM_MODEL


def rebuild() -> int:
    """Recompute communities and their summaries; replace the communities table. Returns count."""
    store.init_db()
    comms = [c for c in detect() if len(c) >= MIN_COMMUNITY_SIZE]
    model = _gen_model()
    with store._conn() as conn:
        conn.execute("DELETE FROM communities")
        for comm in comms:
            rels = _relation_lines(comm)
            sample = _chunk_sample(comm)
            prompt = (
                "Summarize this cluster of related entities in <=150 words. "
                "Base it only on the relations and text below.\n\n"
                "Relations:\n" + "\n".join(rels) +
                "\n\nSample text:\n" + "\n---\n".join(sample)
            )
            try:
                summary = _summarize(prompt, model)
            except Exception as e:
                logger.warning(f"community summary failed: {e}")
                summary = ""
            conn.execute(
                "INSERT INTO communities(level, summary, entity_ids) VALUES(?,?,?)",
                (0, summary, json.dumps(comm)),
            )
    return len(comms)
