# RAG Upgrade Log

This log documents date, modifications, before/after latency numbers, dashboard outputs, and deviations for each upgrade phase.

---

## [2026-07-12] Phase 0 — Baseline Instrumentation

### Changes
- Implemented stage-wise latency metrics in the database (`bm25_latency`, `vector_latency`, `rrf_latency`, `ttft_latency`, `cache_check_latency`).
- Measured these metrics during streaming execution and fallback execution in the backend.
- Surfaced stage-wise latency averages and faithfulness score trends on the metrics dashboard.
- Set up spot-check queries and captured baseline latency snapshots.

### Latency Snapshots (Average over 15 Queries)
- *Total Latency:* [Snapshot value during baseline check]
- *Query Rewrite Latency:* [Snapshot value during baseline check]
- *Embedding Latency:* [Snapshot value during baseline check]
- *Retrieval Latency:* [Snapshot value during baseline check]
- *Reranking Latency:* [Snapshot value during baseline check]
- *Generation Latency (Total):* [Snapshot value during baseline check]
- *Time to First Token (TTFT):* [Snapshot value during baseline check]
- *Faithfulness score avg:* [Snapshot value during baseline check]
