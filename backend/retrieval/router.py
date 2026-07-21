"""Query router: pick factoid | relational | global per query.

Heuristics first (cheap); only an ambiguous relational case (relational keywords but
<2 known entity mentions) spends one small LLM classify call. `ROUTER_MODE` can pin a mode.
"""
import logging

import config

logger = logging.getLogger(__name__)

GLOBAL_KW = ("summarize", "summary", "overview", "themes", "theme", "across", "all documents",
             "all the documents", "recurring", "overall", "main points")
RELATIONAL_KW = ("relationship", "related", "relate", "connect", "connection", "between",
                 "compare", "link", "how is", "how are", "how does")


def _known_entity_hits(query: str) -> int:
    """Count distinct known entities whose norm_name appears in the query."""
    q = query.lower()
    try:
        from backend.graph import store
        with store._conn() as c:
            rows = c.execute("SELECT DISTINCT norm_name FROM entities").fetchall()
    except Exception:
        return 0
    hits = 0
    for r in rows:
        nn = r["norm_name"]
        if nn and len(nn) >= 3 and nn in q:
            hits += 1
    return hits


def _llm_classify(query: str) -> str:
    import httpx
    try:
        from backend.model_tiers import model_for
        model = model_for("REWRITE_MODEL")
    except Exception:
        model = getattr(config, "REWRITE_MODEL", "") or config.LLM_MODEL
    prompt = (
        "Classify the query as exactly one word: factoid, relational, or global.\n"
        "- factoid: a single fact lookup.\n"
        "- relational: how two or more things connect/compare.\n"
        "- global: summary/themes across the whole corpus.\n\n"
        f"Query: {query}\nAnswer:"
    )
    try:
        r = httpx.post(
            f"{config.OLLAMA_BASE}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.0},
                  "keep_alive": getattr(config, "OLLAMA_KEEP_ALIVE", "10m")},
            timeout=30,
        )
        r.raise_for_status()
        ans = r.json().get("response", "").strip().lower()
        for label in ("relational", "global", "factoid"):
            if label in ans:
                return label
    except Exception as e:
        logger.warning(f"router LLM classify failed: {e}")
    return "factoid"


def route(query: str) -> str:
    """Return 'factoid' | 'relational' | 'global'."""
    mode = getattr(config, "ROUTER_MODE", "auto")
    if mode in ("factoid", "relational", "global"):
        return mode
    if mode == "vector":  # explicit vector pin maps to factoid path
        return "factoid"

    q = query.lower()
    if any(kw in q for kw in GLOBAL_KW):
        return "global"

    relational = any(kw in q for kw in RELATIONAL_KW)
    entity_hits = _known_entity_hits(query)
    if entity_hits >= 2:
        return "relational"
    if relational:
        # relational keywords but <2 known entities → ambiguous, ask the LLM
        return _llm_classify(query)
    return "factoid"
