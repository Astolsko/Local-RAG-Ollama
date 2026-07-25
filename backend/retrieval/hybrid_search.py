import os
import asyncio
import logging
import time
from typing import List, Tuple, Any

import config

logger = logging.getLogger(__name__)
from backend.retrieval.query_rewrite import should_rewrite, rewrite_query
# import backend.rag as rag_mod  # moved inside function
from backend.retrieval.bm25_index import search_bm25
from backend.retrieval.reranker import rerank

RRF_K = 60

def _source_filter(source_ids: list[str] | None) -> dict | None:
    """Build a Chroma `where` clause restricting results to the given sources."""
    if not source_ids:
        return None
    if len(source_ids) == 1:
        return {"source_id": source_ids[0]}
    return {"source_id": {"$in": source_ids}}

def _vector_search(embedding: List[float], top_k: int, source_ids: list[str] | None = None) -> Tuple[List[str], List[str], List[Any]]:
    """Chroma vector search on a *precomputed* query embedding, returning (ids, docs, metas).

    Takes the embedding rather than the text: passing `query_texts` makes Chroma call the
    collection's embedding function itself, i.e. one Ollama round-trip per rewritten query
    on top of the one the caller already paid. Blocking (Chroma is sync).
    """
    # Lazy import to avoid circular dependency
    from backend import rag as rag_mod
    _collection = rag_mod._collection
    hits = _collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, _collection.count()),
        where=_source_filter(source_ids),
        include=["documents", "metadatas", "distances"],
    )
    ids = hits["ids"][0] if hits["ids"] else []
    docs = hits["documents"][0] if hits["documents"] else []
    metas = hits["metadatas"][0] if hits["metadatas"] else []
    # Distances drive the rerank-skip shortcut and the confidence fallback when no
    # reranker ran. Older mocks/collections may omit them.
    raw_dists = hits.get("distances") or []
    dists = raw_dists[0] if raw_dists else [None] * len(ids)
    return ids, docs, metas, dists


async def _timed_fan_out(fn, items, top_k, source_ids):
    """Run blocking `fn(item, top_k, source_ids)` across items concurrently; time the batch."""
    t = time.perf_counter()
    out = await asyncio.gather(*[asyncio.to_thread(fn, it, top_k, source_ids) for it in items])
    return out, time.perf_counter() - t


async def _search_both(embeddings, queries, top_k, source_ids):
    """Dense and sparse retrieval, concurrently, each fanned out across the queries."""
    return await asyncio.gather(
        _timed_fan_out(_vector_search, embeddings, top_k, source_ids),
        _timed_fan_out(search_bm25, queries, top_k, source_ids),
    )

def _rrf_fusion(vector: List[Tuple[str, Any]], bm25: List[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
    """Combine rankings using Reciprocal Rank Fusion.
    `vector` and `bm25` are lists of (id, payload) where payload includes doc and meta.
    Returns list of (id, payload) sorted by fused score.
    """
    scores = {}
    # vector ranks
    for rank, (doc_id, payload) in enumerate(vector, start=1):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (RRF_K + rank)
    # bm25 ranks
    for rank, (doc_id, payload) in enumerate(bm25, start=1):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (RRF_K + rank)
    # sort ids by score desc
    sorted_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
    # build merged list preserving payload from whichever source appears first
    merged = []
    payload_map = {pid: pl for pid, pl in vector + bm25}
    for pid in sorted_ids:
        merged.append((pid, payload_map[pid]))
    return merged

def hybrid_search_sync(question: str, top_k: int = None, source_ids: list[str] | None = None,
                       has_history: bool = False) -> Tuple[List[str], List[str], List[Any], dict]:
    """Synchronously perform hybrid search (with optional query rewrite) and return documents, metadatas, and metrics.
    Returns four elements: docs, metas, ids, metrics_info.
    When ``source_ids`` is given, both the dense and sparse paths are restricted to those sources.
    ``has_history`` marks a follow-up turn, which always needs query rewriting.
    """
    if top_k is None:
        top_k = config.TOP_K

    # Determine which queries to run. Rewriting costs an LLM call, so short
    # self-contained queries skip it (see should_rewrite).
    queries: List[str] = [question]
    rewrite_skipped = True
    if getattr(config, "ENABLE_QUERY_REWRITE", False) and should_rewrite(question, has_history):
        # rewrite_query returns list: original + rewrites
        queries = rewrite_query(question, has_history)
        rewrite_skipped = False

    all_vector_payload: List[Tuple[str, Any]] = []
    all_bm25_payload: List[Tuple[str, Any]] = []

    t_start_retrieve = time.perf_counter()
    # Embed every query in one batched /api/embed call. Chroma would otherwise embed each
    # query separately inside its own search thread, so an expanded query cost 4 Ollama
    # round-trips contending on one model instead of a single batch.
    from backend.embeddings import embed_texts
    q_embeddings = embed_texts(queries)

    # Both backends run concurrently, each fanned out across the (rewritten) queries.
    # Results come back in query order, so the fused ranking is identical to the serial
    # version. asyncio.run (not get_event_loop) because this runs in a worker thread with
    # no loop of its own — the old path built a fresh loop per request and never closed it.
    (vector_results, vector_latency_total), (bm25_results, bm25_latency_total) = asyncio.run(
        _search_both(q_embeddings, queries, top_k * 2, source_ids)
    )

    best_dense_sim = 0.0
    for v_ids, v_docs, v_metas, v_dists in vector_results:
        for doc_id, doc, meta, dist in zip(v_ids, v_docs, v_metas, v_dists):
            if dist is not None and isinstance(meta, dict):
                # Chroma cosine distance: 0 is identical, 2 is opposite.
                meta["distance"] = dist
                best_dense_sim = max(best_dense_sim, 1.0 - float(dist))
            all_vector_payload.append((doc_id, {"doc": doc, "meta": meta}))
    for b_ids, b_docs, b_metas in bm25_results:
        all_bm25_payload.extend(
            [(doc_id, {"doc": doc, "meta": meta}) for doc_id, doc, meta in zip(b_ids, b_docs, b_metas)]
        )

    # Fuse across all queries
    t_rrf_start = time.perf_counter()
    fused = _rrf_fusion(all_vector_payload, all_bm25_payload)
    rrf_latency = time.perf_counter() - t_rrf_start

    # GraphRAG-lite augmentation — no-op unless GRAPH_ENABLED. Merges graph candidates into
    # the fused list so they get reranked alongside vector results (a misroute costs quality,
    # never correctness). route stays "vector" when the flag is off.
    route = "vector"
    if getattr(config, "GRAPH_ENABLED", False):
        route = _augment_with_graph(question, fused, source_ids)

    retrieve_latency = time.perf_counter() - t_start_retrieve

    # Optional reranking (still based on the original question).
    # Confidence shortcut (Task 2.3): when the best dense hit is already this similar, the
    # reranker has nothing left to reorder, so skip it and keep the fusion order.
    rerank_latency = 0.0
    skip_threshold = getattr(config, "RERANK_SKIP_THRESHOLD", 0.85)
    rerank_skipped_confident = best_dense_sim >= skip_threshold
    if getattr(config, "RERANK_ENABLED", False) and not rerank_skipped_confident:
        # Cap candidates before reranking — rerank cost is linear in candidate count, and
        # RRF has already surfaced the best few. See Task 2.2.
        fused = fused[:getattr(config, "RRF_TOP_N", 12)]
        t_start_rerank = time.perf_counter()
        fused = rerank(question, fused)
        rerank_latency = time.perf_counter() - t_start_rerank
        fused = fused[:config.RERANK_TOP_K]
        # rerank() records the score on the payload, but every caller builds its citations
        # out of `metas` and reads confidence from there — so it was always missing and
        # calculate_confidence fell through to its flat 0.85 default. Mirror it onto meta.
        for _, p in fused:
            p["meta"]["rerank_score"] = p.get("rerank_score")
    elif rerank_skipped_confident:
        logger.info(f"rerank skipped: top dense similarity {best_dense_sim:.3f} >= {skip_threshold}")
        fused = fused[:config.RERANK_TOP_K]
    else:
        fused = fused[:top_k]

    # Unpack results
    docs = [p["doc"] for _, p in fused]
    metas = [p["meta"] for _, p in fused]
    ids = [i for i, _ in fused]
    
    # Extract rerank scores if present
    rerank_scores = [p.get("rerank_score", 0.0) for _, p in fused]

    metrics_info = {
        "rewritten_queries": queries,
        "rewrite_skipped": rewrite_skipped,
        "retrieve_latency": retrieve_latency,
        "rerank_latency": rerank_latency,
        "rerank_scores": rerank_scores,
        "vector_latency": vector_latency_total,
        "bm25_latency": bm25_latency_total,
        "rrf_latency": rrf_latency,
        "route": route,
        "rerank_skipped": rerank_skipped_confident,
        "best_dense_sim": best_dense_sim,
    }

    return docs, metas, ids, metrics_info


def _augment_with_graph(question: str, fused: List[Tuple[str, Any]],
                        source_ids: list[str] | None) -> str:
    """Merge graph candidates into `fused` in place. Returns the chosen route label."""
    try:
        from backend.retrieval.router import route as route_query
        from backend.graph import retrieval as graph_retrieval
    except Exception as e:
        logger.warning(f"graph augmentation unavailable: {e}")
        return "vector"

    try:
        route = route_query(question)
    except Exception as e:
        logger.warning(f"router failed, staying on vector path: {e}")
        return "vector"

    existing = {i for i, _ in fused}
    allowed = set(source_ids) if source_ids else None

    def _add(cid, payload):
        if cid in existing:
            return
        if allowed is not None and (payload.get("meta") or {}).get("source_id") not in allowed:
            return
        fused.append((cid, payload))
        existing.add(cid)

    try:
        if route == "relational":
            for cid, payload in graph_retrieval.relational_candidates(question):
                _add(cid, payload)
        elif route == "global":
            for cs in graph_retrieval.community_summaries():
                _add(f"community:{cs['id']}", {
                    "doc": cs["summary"],
                    "meta": {"source_name": "Community Summary", "source_id": "", "chunk_index": 0},
                })
    except Exception as e:
        logger.warning(f"graph retrieval failed on route={route}: {e}")

    return route
