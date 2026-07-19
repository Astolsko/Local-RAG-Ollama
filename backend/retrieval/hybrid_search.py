import os
import asyncio
from typing import List, Tuple, Any

import config
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

async def _vector_search(question: str, top_k: int, source_ids: list[str] | None = None) -> Tuple[List[str], List[str], List[Any]]:
    """Perform Chroma vector search returning (ids, docs, metas)."""
    # Lazy import to avoid circular dependency
    from backend import rag as rag_mod
    _collection = rag_mod._collection
    hits = _collection.query(
        query_texts=[question],
        n_results=min(top_k, _collection.count()),
        where=_source_filter(source_ids),
        include=["documents", "metadatas"],
    )
    ids = hits["ids"][0] if hits["ids"] else []
    docs = hits["documents"][0] if hits["documents"] else []
    metas = hits["metadatas"][0] if hits["metadatas"] else []
    return ids, docs, metas

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
    import time
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
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    t_start_retrieve = time.perf_counter()
    vector_latency_total = 0.0
    bm25_latency_total = 0.0
    for q in queries:
        # Run both searches for each query
        t_v_start = time.perf_counter()
        v_ids, v_docs, v_metas = loop.run_until_complete(_vector_search(q, top_k * 2, source_ids))
        vector_latency_total += time.perf_counter() - t_v_start

        t_b_start = time.perf_counter()
        b_ids, b_docs, b_metas = search_bm25(q, top_k * 2, source_ids)
        bm25_latency_total += time.perf_counter() - t_b_start
        
        all_vector_payload.extend(
            [(doc_id, {"doc": doc, "meta": meta}) for doc_id, doc, meta in zip(v_ids, v_docs, v_metas)]
        )
        all_bm25_payload.extend(
            [(doc_id, {"doc": doc, "meta": meta}) for doc_id, doc, meta in zip(b_ids, b_docs, b_metas)]
        )

    # Fuse across all queries
    t_rrf_start = time.perf_counter()
    fused = _rrf_fusion(all_vector_payload, all_bm25_payload)
    rrf_latency = time.perf_counter() - t_rrf_start
    retrieve_latency = time.perf_counter() - t_start_retrieve

    # Optional reranking (still based on the original question)
    rerank_latency = 0.0
    if getattr(config, "RERANK_ENABLED", False):
        t_start_rerank = time.perf_counter()
        fused = rerank(question, fused)
        rerank_latency = time.perf_counter() - t_start_rerank
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
        "rrf_latency": rrf_latency
    }

    return docs, metas, ids, metrics_info
