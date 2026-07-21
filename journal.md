# Implementation Journal

Tracks what's actually implemented against `files/plan.md` + `files/implementation.md`.
Verified against the code, not the docs' claims. Legend: ✅ done · 🟡 partial · ⬜ not started.

> **Hardware note:** dev laptop overheats and shuts down under sustained load. Acceptance
> gates that need a full 50-question eval or bench run are **not auto-run** here — they're
> flagged "needs eval" for the owner to run deliberately (small models, one at a time).

---

## Phase 1 — Eval harness + model quick wins — ✅ (built 2026-07-20)

| Task | Status | Evidence |
|---|---|---|
| 1.1 Golden set | ✅ | `eval/golden/questions.jsonl` (50 q), 4 docs in `eval/golden/documents/` |
| 1.2 Eval runner | ✅ | `eval/run_eval.py`, baseline in `eval/results/baseline.json`, `history.csv` |
| 1.3 Model tiering | ✅ | `backend/model_tiers.py`; `/api/health` returns `tier/ram_gb/models/missing_models` (`main.py:136`) |
| 1.4 Adaptive pipeline + Ollama tuning | ✅ | `should_rewrite()` skip in `hybrid_search.py:73`; `keep_alive` on all Ollama calls; batched embeds in `backend/embeddings.py` (`embed_texts`, `/api/embed`) |

Baseline (qwen2.5:0.5b, judge llama3.2:3b): faithfulness 0.907 · relevance 0.861 · refusal 0.857
· recall@5 0.963 · MRR 0.862 · p50 17,977 ms · TTFT 753 ms. Latency is retrieval-bound
(retrieve 8.5s + rerank 3.3s + query-embed 2.1s = 14.7s), generation only 6%.

---

## Phase 2 — Retrieval quality — 🟡

| Task | Status | Notes |
|---|---|---|
| 2.0 Stop rebuilding BM25 per request | ✅ **this session** | See changelog below |
| 2.1 Structure-aware chunking | ⬜ | `StructureAwareChunker` exists (`chunker.py:137`) but is **dead code** — ingestion still runs `rag.chunk_text_with_metadata → smart_chunk_page`. No `char_start/char_end`, no `data/documents/` full-text store, no `POST /api/sources/reingest`. |
| 2.2 RRF fusion | ✅ **validated** | dense+BM25 run concurrently (fan-out across rewritten queries, `_search_all` + `asyncio.to_thread`) — result-identical to the old serial loop. Added `RRF_TOP_N` cap (default 12) before rerank. ✅ **Full Colab eval (50-q) confirms the cap is safe: recall@5 held at 0.963 (= baseline), MRR ↑ 0.862→0.901.** Concurrency + cap covered by existing `test_rag.py`. |
| 2.3 FlashRank reranker | ⬜ | Only cross-encoder (`BAAI/bge-reranker-base`) in `reranker.py`. No `RERANKER` env, no `RERANK_SKIP_THRESHOLD`. |
| 2.4 Embedding cache | ✅ **this session** | `backend/cache/embed_cache.py`, wraps `embed_texts`. See changelog. |

---

## Phase 3 — GraphRAG-lite — ✅ built, gate ❌ FAILED → `GRAPH_ENABLED` stays `false`

| Task | Status | Notes |
|---|---|---|
| 3.1 Graph store | ✅ **this session** | `backend/graph/store.py` — SQLite (`data/graph.db`), entities/relations/entity_chunks/communities, norm_name dedup, `delete_doc` orphan cleanup, cached `load_networkx()`. Test: `test_graph_store.py`. |
| 3.2 Entity/relation extraction | ✅ **this session** | `backend/graph/extractor.py` — Ollama `format=json`, Pydantic validation, one repair retry, empty-on-double-fail. `GRAPH_EXTRACT_TEMPLATE` in `prompt_templates.py`. Test: `test_graph_extractor.py` (valid / repaired / double-fail / persist), Ollama mocked. |
| 3.3 Background indexer | ✅ **session 4** | `graph/indexer.py` + `/api/graph/{index,status,summary}`, ingestion BackgroundTasks hook, delete-source cleanup. Mocked test. |
| 3.4 Community detection + summaries | ✅ **session 4** | `graph/communities.py`, networkx Louvain (no python-louvain dep). LLM summary isolated for tests. |
| 3.5 Router + graph retrieval | ✅ **session 4** | `retrieval/router.py` + `graph/retrieval.py`, wired into `hybrid_search_sync` (no-op when flag off), `route` on the response. Mocked tests. |
| 3.6 Graph eval gate | ✅ ran → **❌ FAILED** | Full Colab 50-q run with `GRAPH_ENABLED=true` (`--label graph_on`) **regressed every metric** vs the vector default: recall@5 0.963→**0.889**, MRR 0.901→**0.821**, relevance 0.645→0.611, false-refusals 0.000→**0.046**, p50 4,426→**11,376ms** (2.5×; generate 1,910→9,598ms). Graph augmentation displaces well-ranked vector/BM25 hits and bloats the prompt. **Decision: keep `GRAPH_ENABLED=false` default.** Needs rework (rank-preserving merge instead of displacement; trim graph context) before it's worth enabling. |

Config keys `GRAPH_ENABLED` (false), `GRAPH_MAX_HOPS` (2), `ROUTER_MODE` (auto) added.
Full graph pipeline (extract → store → communities → router → relational/global retrieval) is
built and wired, all behind `GRAPH_ENABLED=false` so the default path is byte-for-byte unchanged
(confirmed by the session-4 smoke eval). Every unit test mocks Ollama — **zero heavy runs so far**.
The one remaining step (3.6) is to actually index the golden docs and eval with the flag on:
that's the LLM-per-chunk work best done on Colab.

---

## Phase 4 — Grounding, verification, UI — 🟡

| Task | Status | Notes |
|---|---|---|
| 4.1 Citation contract | 🟡 | `chat_stream.py` already parses `[n]` markers and returns citations. **Not done:** it hardcodes `SYSTEM_PROMPT_TEMPLATE` (user's prompt editor still has no effect on main chat — known bug); no numbered-context "cite or refuse" contract; no `char_start/char_end` in citations; no `CONFIDENCE_REFUSAL_THRESHOLD` refusal gate. Changing the prompt moves refusal/citation rate sharply → re-baseline. |
| 4.2 Faithfulness check | ⬜ | No `backend/verification/`. (A *background* faithfulness judge exists in `chat_stream.run_background_judge`, but not the per-sentence citation-support check.) |
| 4.3 Frontend citations + source viewer | ⬜ | No `frontend/src/components/`. |
| 4.4 Graph explorer | ⬜ | No `GraphExplorer.jsx`; depends on Phase 3. |
| 4.5 Settings panel + HUD | 🟡 | Settings modal (general/rag/prompt tabs), citation chips, source viewer, confidence/tokens/cached badges all already existed. **This session:** wired `MODEL_TIER` into settings end-to-end; added latency + tok/s to the answer HUD. Still missing: `RERANKER`/`GRAPH_ENABLED`/`ROUTER_MODE` toggles (those config keys don't exist yet — later phases), route badge (needs Phase 3). |
| 4.6 Visual polish | ⬜ | |

---

## Phase 5 — Inference speed — ⬜

No `backend/llm/` abstraction, no `eval/bench_inference.py`, no llama.cpp scripts.
Per `plan.md §2.5` this phase can't hit the p50<4s target (that's Phase 2 retrieval work);
frame as throughput only. Re-measure after Phase 2 before committing to 5.5/5.6.

---

## Phase 6 — Packaging — ⬜

No `pyproject.toml`, no Docker, no CI, no `Makefile`. README not rewritten.
Env still conda `RAG` + Windows launchers (`start-app.ps1/.bat`).

---

## Changelog

### 2026-07-22 (session 2)
- **Task 2.4 — Embedding cache.** New `backend/cache/embed_cache.py`: Redis, keyed by
  `sha256(model + text)`, TTL `EMBED_CACHE_TTL` (default 7 days, new config key). `embed_texts`
  now checks cache first and only embeds misses (covers query + ingestion). Hit/miss counters in
  Redis; `GET /api/observability/metrics` summary gains `embed_cache_hit_rate`. Test:
  `tests/test_embed_cache.py` (fake Redis proves 2nd embed is a hit). *Safe, no eval needed.*
- **Task 4.5 — Settings + HUD (partial).** `MODEL_TIER` (auto/small/medium/large) now wired
  end-to-end (`SettingsUpdate` whitelist + a select in the General tab). Answer HUD gains a
  latency + tok/s badge — `chat_stream` final events now emit `generate_ms`/`total_ms`,
  threaded through `api.js` → `App.jsx`. Frontend vitest green.

### 2026-07-22 (session 5) — full Colab eval: RRF cap validated, graph gate failed
- **Task 2.2 validated.** Full 50-q Colab run (qwen2.5:0.5b, judge llama3.2:latest):
  recall@5 held at **0.963** (= baseline) with `RRF_TOP_N=12`, MRR ↑ 0.862→**0.901**. Cap is safe.
- **Task 3.6 gate ran → FAILED.** `GRAPH_ENABLED=true` regressed everything (recall 0.963→0.889,
  MRR 0.901→0.821, p50 4.4s→11.4s, false-refusals 0→0.046). **Keep the default `false`.**
- ⚠️ Faithfulness 0.791 / relevance 0.645 below target in both runs — generation-model quality
  (0.5b), not retrieval (recall is high). Partly judge drift (baseline used llama3.2:**3b**,
  this llama3.2:**latest**). Next lever per plan: upgrade generation model.

### 2026-07-22 (session 5) — portable storage (local ↔ Colab)
- **`backend/paths.py`** is now the single edit point for data location. `config.DATA_DIR`
  derives from it; precedence `RAG_DATA_DIR` env > `DATA_DIR_OVERRIDE` in paths.py > `<repo>/data`.
  For Colab: mount Drive, set `DATA_DIR_OVERRIDE = "/content/drive/MyDrive/.../data"`.
  Fixed the last hardcoded paths (`reingest.py` source/metrics dirs → `config.DATA_DIR`;
  `main.py` eval-history → `paths.EVAL_DIR`). `import config` and `import backend.config` both
  resolve. Defaults unchanged for local; 40 tests pass.

### 2026-07-22 (session 4) — graph pipeline + latency validation
- **Tasks 3.3–3.5 built** (behind `GRAPH_ENABLED=false`):
  - `graph/indexer.py` — `index_document(doc_id)` extracts a doc's chunks into the store,
    Redis job status (`graph:index:{doc_id}`), idempotent re-index. Endpoints
    `POST /api/graph/index/{doc_id}`, `GET /api/graph/status`, `GET /api/graph/summary`.
    Ingestion (`POST /api/sources`, `/upload`) schedules it via `BackgroundTasks`;
    `delete_source` drops the doc's graph rows. All gated on `GRAPH_ENABLED`.
  - `graph/communities.py` — networkx built-in Louvain, one ≤150-word LLM summary per
    community ≥3 entities. Runs only in the background job.
  - `retrieval/router.py` — heuristic factoid/relational/global, LLM classify only for the
    ambiguous relational case; `ROUTER_MODE` pin. `graph/retrieval.py` — relational k-hop
    neighborhood candidates + global community summaries.
  - Wired into `hybrid_search_sync` via `_augment_with_graph` (no-op when flag off; merges
    graph candidates into the fused list to be reranked together; respects `source_ids`).
    `route` surfaced on the ask response. Tests: `test_graph_pipeline.py` (indexer/router/
    relational, mocked). **Still ⬜: 3.6 eval gate** (needs a full graph run to flip the default).
- **Latency validation** (`eval/run_eval.py --limit 10`, GRAPH_ENABLED off, qwen2.5:0.5b —
  same model as baseline): retrieve p50 **8,501→2,164 ms**, rerank **3,337→1,203 ms**, total
  **17,977→9,398 ms**; recall@5 held at 1.000 on the subset. The Task 2.2 concurrency + cap
  is the win. ⚠️ Still want a **full Colab run** to confirm multihop/global recall + the cap
  across all 50. Result file: `eval/results/20260722-003610.json`.

### 2026-07-22 (session 3)
- **Task 2.2 — concurrent retrieval + candidate cap.** `hybrid_search_sync` now fans the
  rewritten queries out concurrently across both backends (`_search_all` + `asyncio.to_thread`)
  instead of a serial `run_until_complete` loop — same results, less wall time. Added
  `RRF_TOP_N` (default 12) cap on candidates before the cross-encoder. Cap is quality-affecting
  → **flagged for Colab full eval** (recall@5 must stay ≥ 0.963).
- **Task 3.1/3.2 — GraphRAG-lite foundation.** `backend/graph/store.py` (SQLite graph store,
  networkx view) and `backend/graph/extractor.py` (LLM JSON extraction + Pydantic + repair),
  behind new `GRAPH_ENABLED`/`GRAPH_MAX_HOPS` config. Tests `test_graph_store.py` +
  `test_graph_extractor.py` (Ollama mocked). Community detection will use networkx's built-in
  Louvain — no `python-louvain` dependency. Full suite: 37 passed (`-m "not integration"`).

### 2026-07-22 (session 1)
- **Task 2.0 — BM25 no longer rebuilt on the request path.** `metrics_db.init_db()` was
  calling `build_bm25_index()`, and `init_db()` runs from `log_request`,
  `update_faithfulness`, and `update_feedback` — so every answered question rebuilt the
  whole index ~2–3× (O(corpus) per request). Removed the build from `init_db()`; it now
  runs only where the corpus changes (`rag.add_source`/`delete_source`/`reingest`) plus once
  at startup (`main.py`). Test: `tests/test_bm25_not_on_request.py` (asserts 0 rebuilds on
  log/judge/feedback). Existing `test_observability.py` still green. *No quality impact —
  safe to ship without an eval run.*

---

## How to run the acceptance eval (owner, when ready)

```
# small model, isolated data dir, Redis DB 1 — from repo root, RAG env active
python eval/run_eval.py --limit 10      # subset first, skips history.csv
python eval/run_eval.py                 # full 50-q run, writes results/<ts>.json + latest.json
```
Run one model at a time; watch temps. Compare `latest.json` against `baseline.json`.
