# Local-RAG-Ollama — Improvement Plan

**Repo:** https://github.com/Astolsko/Local-RAG-Ollama
**Companion doc:** `implementation.md` (step-by-step build instructions for Claude Code)

> **Status: Phase 1 built, measured on 2026-07-20.** Sections marked
> **[VERIFIED]** were checked against the code or measured on the golden set;
> they replace the original assumptions, several of which were wrong. See §1.5
> for what the codebase actually contained, and §8 for measured baselines.

---

## 1. Current State Assessment

### What already works (do not break)
- Hybrid retrieval: ChromaDB dense vectors + BM25 sparse, fused before generation
- Query rewriting via local LLM
- Cross-encoder (SentenceTransformer) reranking
- Contextual chunking at ingestion
- Redis: semantic query cache, token-bucket rate limiting, multi-turn session memory
- Observability: SQLite metrics DB + React metrics dashboard, thumbs up/down feedback
- Streaming answers, chat history persistence, system-prompt editing
- Backend/frontend test suites, one-click Windows launchers

### Gaps holding the project back
1. **Generation ceiling** — `qwen2.5:0.5b` is too weak to synthesize, cite, or refuse reliably. Most "inaccurate replies" trace to this, not retrieval.
2. **Chunk-local retrieval only** — no entity/relationship awareness. Fails on multi-hop questions ("How is X related to Y?") and global questions ("Summarize the main themes across all docs").
3. **No grounding enforcement** — answers don't carry citations; nothing verifies claims against retrieved chunks; the model never says "not in the documents."
4. **No measurement loop** — `eval/` exists but there is no golden Q&A set, no automated faithfulness/relevance scoring, so improvements can't be proven.
5. **Fixed pipeline** — every query pays for rewrite + hybrid + rerank even when a cache-warm factoid needs none of it.
6. **Windows-only, conda-based setup** — no Docker, no cross-platform scripts, high friction for anyone cloning the repo.
7. **UI is functional, not impressive** — no source highlighting, no inline citations, no graph view, no settings panel (Phases 4/5 of the original roadmap are still open).

---

## 1.5 [VERIFIED] What the codebase actually contained

Read before planning any phase. Several assumptions above and in
`implementation.md` did not match the code.

### Already implemented (do not rebuild)
| Assumed missing | Reality |
|---|---|
| RRF fusion (Task 2.2) | Already there: `retrieval/hybrid_search.py` `_rrf_fusion`, `RRF_K = 60` |
| BM25 disk persistence (Task 2.2) | Already there: `data/chroma/bm25_index.pkl`, rebuilt on ingest/delete |
| Structure-aware chunking (Task 2.1) | `StructureAwareChunker` **exists but is dead code**. Ingestion runs `rag.add_source` → `smart_chunk_page` → `semantic_chunk_text`. Task 2.1 is *wiring it up*, not writing it |

### Architecture corrections
- **`backend/chat_stream.py` is the real answer path.** `POST /api/chats/ask` →
  `chat_stream.chat_stream()`. `implementation.md` never mentions this file, yet
  every citation, verification and routing change lands there. `rag.py:ask()`
  serves only the legacy `POST /api/ask`.
- **`config.py` is not Pydantic settings.** It is a `DEFAULT_SETTINGS` dict plus a
  module `__getattr__` resolving env var → `data/settings.json` → default. New
  tunables are added as dict entries. Ground rule 3 was wrong.
- Dead code removed during Phase 1: `rag.ask_stream()` and `rag._retrieve_context()`
  had no reachable callers.

### Live bugs found (fixed unless noted)
| Bug | Impact | Status |
|---|---|---|
| `source_ids` never filtered retrieval | Selecting one source still returned and cited chunks from every other source. `hybrid_search_sync` was called without any filter on all three paths | **Fixed** — filter applied in `hybrid_search_sync`, covering dense (Chroma `where`) and sparse (pre-truncation filter) |
| `QUERY_REWRITE_MODEL` default was the literal string `"{OLLAMA_MODEL}"` | Unresolved placeholder; read nowhere | **Fixed** (deleted) |
| **System-prompt editor has no effect on the main chat** | `chat_stream.py` hardcodes `SYSTEM_PROMPT_TEMPLATE`; `PUT /api/settings/system-prompt` writes `data/system_prompt.txt`, which only the legacy `/api/ask` reads. There are **two** system prompts (`config.DEFAULT_SYSTEM_PROMPT` and `prompt_templates.SYSTEM_PROMPT_TEMPLATE`) | **Open** — fix with Task 4.1, which rewrites the prompt anyway |
| **Every logged request rebuilds the entire BM25 index** | `metrics_db.log_request()` → `init_db()` → `build_bm25_index()`. Also via `update_faithfulness` and `update_feedback`, so each question triggers ~2 full rebuilds plus one per thumbs-up/down. O(corpus) work per request | **Open** — fix before Task 5.2's load test, which will serialize on it |

### Environment notes
- `make` is **not installed** on the dev machine; `Makefile` deferred to Phase 6 (CI).
  Run `python eval/run_eval.py` directly.
- Dev machine: 15.8 GB RAM, 10 physical cores, RTX 4060 Laptop → **medium** tier.
- Models pulled: `qwen2.5:0.5b`, `llama3.2:latest` (= llama3.2:3b, a medium-tier
  generation model), `nomic-embed-text`.
- **`data/settings.json` diverges from shipped defaults**: `LLM_MODEL=llama3.2:latest`,
  `CACHE_SIMILARITY_THRESHOLD=0.95`, and **`CHUNK_SIZE=50`** — 50 *characters*, with
  `CHUNK_OVERLAP=50` equal to it. That is degenerate and is the most likely cause of
  poor answers in the live app. Baselines were measured on shipped defaults, not this.

---

## 2. Goals — What "10x" Means, Measurably

| Dimension | Target |
|---|---|
| Groundedness | Every claim carries an inline citation `[n]` resolvable to an exact chunk; unsupported-claim rate < 5% on eval set |
| Accuracy | Faithfulness ≥ 0.90 and answer relevance ≥ 0.85 on a 50-question golden set (LLM-judge scored) |
| Honesty | Correctly answers "I don't have that in the documents" on out-of-scope eval questions ≥ 90% of the time |
| Latency | First streamed token < 1.5s; p50 full answer < 4s on cache miss (CPU-only, 8 GB RAM, 3B model); cache hit < 300ms — **[VERIFIED] TTFT already met at 753ms; p50 is 17,977ms and the gap is retrieval, not inference. See §2.5** |
| Inference throughput | ≥ 1.3x decode tokens/sec over stock Ollama defaults on the same hardware + model, proven by the benchmark harness (quant policy + runtime flags + prefix reuse + optional speculative decoding) |
| Hardware | Runs on 8 GB RAM CPU-only laptop (small tier); auto-selects bigger models when RAM/GPU allows |
| Deployment | `docker compose up` works on Windows/macOS/Linux; alternative native path is one script; CI green on every push |
| UI | Inline citation chips → highlighted source viewer, interactive knowledge-graph explorer, live settings panel, polished dark/light theme |

---

## 2.5 [VERIFIED] Where the latency actually goes — this reorders the plan

Measured p50 per stage over the 50-question golden set (`qwen2.5:0.5b`, shipped
defaults, medium tier). Stage timings come from the pipeline's own metrics DB:

| Stage | p50 | Share |
|---|---|---|
| embed (query) | 2,102 ms | 12% |
| **retrieve** | **8,501 ms** | **47%** |
| **rerank** | **3,337 ms** | **19%** |
| generate | 998 ms | 6% |
| *TTFT (real)* | *753 ms* | — |
| **total** | **17,977 ms** | |

**Generation is 6% of total latency.** Workstream F2 / Phase 5 — quantization,
speculative decoding, a `-march=native` llama.cpp build, KV-cache tuning — all
target that 998 ms slice. Even a perfect 2x inference win removes ~500 ms from an
18-second answer. **Phase 5 as written cannot reach the p50 < 4s target**, and its
≥1.3x decode-throughput goal, while achievable, is close to irrelevant to
user-perceived latency here.

The 14.7 s actually sits in:
1. **Query rewrite** — one LLM call expanding to 4 queries, each then run through
   *both* vector and BM25 (`retrieve` = 8.5 s). Phase 1's adaptive skip targets this.
2. **Reranking** — cross-encoder (`BAAI/bge-reranker-base`) over ~80 candidates
   (3.3 s). Task 2.3's FlashRank swap targets this.
3. **Query embedding** — 2.1 s for a single short string, worth profiling on its own.

**Recommended resequencing:** treat Workstream B (retrieval) and the adaptive
pipeline as the latency work, and demote Phase 5 to a throughput/efficiency phase
with honest framing. Do not promise the p50 target from inference tuning.

---

## 3. The GraphRAG Decision

**Verdict: augment, don't replace.** Keep hybrid vector+BM25 as the fast default path and add a lightweight knowledge-graph layer on top ("GraphRAG-lite"), with a router deciding per query which path to use.

### Why not a full switch to Microsoft-style GraphRAG
- Indexing requires an LLM pass over every chunk for entity/relation extraction plus community summarization — on local hardware with a small model this makes ingestion 10–50x slower and quality-fragile.
- Vector search is *better* than graph traversal for the majority of queries (simple factoids), which is what most users ask most of the time.
- Ripping out a working hybrid pipeline to bet everything on graph quality from a 3B extractor is high risk, low reward.

### Why add a graph layer at all
- It is the only way to answer **multi-hop** ("what connects A to B?") and **global/thematic** ("what are the recurring risks across these reports?") questions well.
- A visible, interactive knowledge graph is the single most impressive UI feature this project can ship.
- Entity-level retrieval catches answers that chunk similarity misses when wording differs.

### Chosen architecture: GraphRAG-lite (custom, in-repo)
- **Extraction:** at ingestion, the local LLM extracts entities + typed relations per chunk using Ollama's JSON mode (`format=json`) with a strict schema, retry-and-repair on malformed output. Runs as a background job so ingestion stays responsive.
- **Storage:** SQLite tables (`entities`, `relations`, `entity_chunks`, `communities`) in the existing `data/` dir; loaded into a `networkx` graph in memory. No new database server — stays deploy-light.
- **Global layer:** Leiden/Louvain community detection over the graph; one LLM-written summary per community, precomputed after ingestion.
- **Retrieval paths:**
  - *Vector path* (default): existing hybrid search + rerank.
  - *Graph path:* link query entities → expand k-hop neighborhood → collect linked chunks + relation triples → rerank alongside vector results.
  - *Global path:* map-reduce over community summaries for corpus-wide questions.
- **Router:** a fast classifier (heuristics + one small LLM call) labels each query `factoid | relational | global` and picks the path. Misroutes are cheap: graph and vector results merge through the same reranker.

**Alternative considered:** adopting the LightRAG library. Faster to ship, but adds a heavy dependency, hides the mechanics, and gives less portfolio/learning value. Custom-lite chosen; keep LightRAG as fallback if custom extraction quality disappoints.

---

## 4. Workstreams

### A. Model & pipeline quick wins
- **Hardware-aware model tiering:** detect RAM/VRAM at startup; recommend/pull tier: `small` (qwen2.5:1.5b / llama3.2:1b), `medium` (qwen2.5:3b / llama3.2:3b — new default), `large` (qwen2.5:7b / llama3.1:8b). Separate (smaller) model for rewriting/routing vs. generation.
- **Adaptive pipeline:** skip query rewrite for short unambiguous queries; skip rerank when top-1 dense score clears a confidence threshold; always check semantic cache first (already done).
- **Ollama tuning:** `keep_alive` so models stay warm; `num_ctx` sized to fit context, not maxed; concurrent embedding batching at ingestion.

### B. Retrieval quality
- **Structure-aware chunking:** split on headings/sections (Markdown, PDF layout via `pymupdf`) before size-based fallback; store section title + doc metadata on every chunk.
- **Reciprocal Rank Fusion (RRF)** to merge dense + BM25 rankings instead of ad-hoc score mixing.
- **Reranker swap:** FlashRank (tiny, CPU-fast) as default; current SentenceTransformer cross-encoder as configurable upgrade.
- **Embedding option:** keep `nomic-embed-text` default; add `bge-m3` as configurable higher-quality option; embedding cache in Redis keyed by content hash.

### C. GraphRAG-lite (as specified in §3)
Extraction job → graph store → community summaries → router → graph/global retrieval → fusion with vector path.

### D. Grounding, verification & evaluation
- **Citation contract:** generator is prompted to cite `[n]` per claim; chunks are numbered in the prompt; the API returns a `citations[]` array mapping `[n]` → chunk id + char offsets.
- **Faithfulness gate:** post-generation, each answer sentence is checked for support against cited chunks (embedding-similarity check with a small NLI-style threshold); unsupported sentences get flagged in the UI.
- **Refusal behavior:** if best retrieval confidence < threshold, answer states the documents don't cover the question instead of guessing.
- **Eval harness (build FIRST):** golden set of ~50 Q/A pairs over sample docs (factoid, multi-hop, global, out-of-scope categories); automated scoring of faithfulness, relevance, citation precision, latency; `make eval` command; regression run in CI. Thumbs-down feedback from the metrics DB feeds new eval cases.

### E. UI overhaul
- Inline **citation chips** in streamed answers; clicking opens a **source panel** with the exact chunk highlighted inside the full document text (original Phase 4).
- **Knowledge-graph explorer** page: interactive force-directed graph (`react-force-graph-2d`), node click → entity's chunks and mentions, visual path highlighting for multi-hop answers.
- **Settings panel** (original Phase 5): model tier, chunk size, top-k, toggle graph mode — live via API, no restart.
- **Answer HUD:** per-message latency, tokens/sec, cache-hit badge, route taken (vector/graph/global).
- Visual polish: consistent design system, dark/light theme, keyboard shortcuts, mobile-responsive layout.

### F. Performance & inference speed

**F1. App-level performance**
- Run dense and BM25 searches concurrently (`asyncio.gather`).
- Lazy-load reranker/extractor models on first use; free on idle.
- Precompute BM25 index at ingestion, persist to disk (avoid rebuild per query).
- Cache layers: semantic answer cache (exists) → embedding cache (new) → rendered community summaries (new).

**F2. Inference-level speed (Ollama and below)**

Guiding physics: token generation is memory-bandwidth-bound — every token requires reading (nearly) all model weights, so `tokens/sec ≈ usable memory bandwidth ÷ model bytes`. Every item below attacks one of four levers: *read fewer bytes per token*, *read from faster memory*, *verify several tokens per pass*, or *skip work entirely*. OS/kernel-driver work is explicitly out of scope — the bottleneck is userspace compute kernels and memory bandwidth, both already handled by GGML and vendor drivers.

- **Quantization policy (fewer bytes/token):** model tiers pin explicit quant tags — `q4_K_M` as the default generation quant, `q8_0` as a configurable quality-first option. KV cache quantized via `OLLAMA_KV_CACHE_TYPE=q8_0`; flash attention on via `OLLAMA_FLASH_ATTENTION=1`.
- **Context & output budgets (skip work):** per-role `num_ctx` (generation 8192, rewrite 2048, extraction 4096 — not model max) and `num_predict` caps per role; `num_thread` = physical cores on CPU-only machines; `keep_alive` to avoid model reloads (already in Workstream A).
- **Prefix-cache-friendly prompting (skip work):** prompt assembly guarantees a byte-stable prefix — static system prompt + instructions first, retrieved context in deterministic order, volatile parts (history, question) last — so the runtime's prefill KV cache is reused across requests. Verified by logging the runtime's reported prompt-eval token counts.
- **Inference backend abstraction (enables everything below):** all LLM calls go through one `InferenceClient` interface with two implementations — `ollama` (default) and `llamacpp` (llama.cpp's `llama-server`, OpenAI-compatible API). Switching is a single env var; no call-site changes.
- **Speculative decoding (multi-token verification):** a small same-family draft model (e.g. qwen2.5 0.5b) proposes tokens; the target model verifies them in one pass. Delivered via `llama-server --model-draft`; if the installed Ollama version exposes native speculative decoding, a runtime capability check prefers it. Enabled as recommended config only if the benchmark harness measures ≥1.3x decode throughput.
- **Self-built llama.cpp (faster kernels for *this* machine):** build script compiling llama.cpp with `-march=native` / correct CUDA architecture, with a prebuilt-release fallback. Stock generic binaries routinely leave 20–40% single-stream throughput on the table.
- **GPU offload guidance:** detect GPU, document partial-offload behavior; even partial VRAM residency multiplies effective bandwidth.
- **Benchmark harness (proves all of it):** standardized TTFT + decode tok/s measurement across backend × config matrix, published as a README table.

### G. Packaging & deployment
- **Docker Compose:** services for backend, frontend (nginx), Redis; Ollama either as a service (Linux) or host-networked (Mac/Win where GPU passthrough is limited). One command up.
- **Native path:** replace conda with `uv` (fast, no env activation pain); cross-platform `start.sh` alongside existing `.ps1/.bat`; `Makefile` with `make dev / make eval / make test`.
- **CI:** GitHub Actions — lint (ruff), backend pytest, frontend vitest, eval smoke run on PRs.
- **Docs:** rewrite README quickstart around the two install paths; architecture diagram updated with graph layer; `.env.example`.

---

## 5. Phasing & Sequencing

Order chosen so every improvement after Phase 1 is *provable* with numbers, and each phase ships something usable on its own.

| Phase | Contents | Why this order |
|---|---|---|
| **1. Measure & quick wins** | Workstream D-eval harness + A (model tiering, adaptive pipeline) | You cannot claim 10x without a baseline. Model upgrade alone will produce the single biggest accuracy jump. |
| **2. Retrieval quality** | Workstream B | Better chunks + RRF + fast rerank lift the floor for everything downstream, verified against Phase 1 baseline. |
| **3. Graph layer** | Workstream C | Built on stable retrieval; router keeps vector path as safe default while graph quality matures. |
| **4. Grounding + UI** | Workstream D (citations, faithfulness gate, refusal) + E | Citations need the retrieval/graph plumbing final; UI showcases everything built so far. |
| **5. Inference speed + performance** | Workstream F | Optimize what is now feature-complete; backend abstraction + benchmark harness make every speed claim measurable (TTFT, tok/s) before packaging freezes the config. |
| **6. Packaging + shipping** | Workstream G | Package the tuned system; Docker/CI lock in both eval scores and benchmark numbers as regressions. |

Each phase ends with: full test suite green, `make eval` run, metrics recorded in `eval/results/` for before/after comparison.

---

## 6. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Small local model produces malformed extraction JSON | Ollama `format=json`, strict Pydantic validation, 2x retry with repair prompt, drop-and-log on final failure (graph degrades gracefully — vector path unaffected) |
| Graph indexing too slow on big corpora | Background task queue with progress endpoint; incremental per-document indexing; extraction model = smallest tier |
| RAM pressure from reranker + extractor + LLM together | Lazy loading, model tiering, sequential (not parallel) model use on small tier |
| Router misclassifies queries | Merge graph+vector candidates through one reranker so a misroute degrades quality, never correctness; log route + feedback for tuning |
| Docker + Ollama GPU friction on Mac/Windows | Document host-Ollama mode as default for Mac/Win; containerized Ollama only on Linux |
| Scope creep | Phases are independently shippable; graph layer is feature-flagged (`GRAPH_ENABLED=true`) |
| Draft/target model mismatch breaks speculative decoding | Pin same-family, same-tokenizer pairs in the tier table (qwen2.5 0.5b → qwen2.5 3b/7b); log draft acceptance rate; benchmark gate before recommending |
| llama.cpp build friction across OSes | `INFERENCE_BACKEND=ollama` stays the zero-effort default; build script has a prebuilt-release download fallback; llamacpp path is opt-in |
| Speculative decoding gains vary by workload | Harness measures on RAG-shaped prompts specifically; enable in docs/defaults only at ≥1.3x measured decode speedup |
| Aggressive quantization (weights or KV cache) hurts answer quality | Every speed config change must hold `make eval` faithfulness within ±0.02; q8/f16 remain configurable |

---

## 7. New Configuration Surface (summary)

```
MODEL_TIER=auto|small|medium|large
LLM_MODEL / REWRITE_MODEL / EXTRACT_MODEL   # per-role models
RERANKER=flashrank|cross-encoder|off
GRAPH_ENABLED=true
GRAPH_MAX_HOPS=2
ROUTER_MODE=auto|vector|graph|global
CONFIDENCE_REFUSAL_THRESHOLD=0.35
EMBED_CACHE_TTL=604800

# Inference speed (Workstream F2)
INFERENCE_BACKEND=ollama|llamacpp
LLAMACPP_BASE=http://localhost:8080
DRAFT_MODEL=qwen2.5:0.5b-instruct-q4_k_m   # speculative decoding draft; empty = off
NUM_CTX_GENERATE=8192 / NUM_CTX_REWRITE=2048 / NUM_CTX_EXTRACT=4096
NUM_PREDICT_GENERATE=1024 / NUM_PREDICT_REWRITE=128 / NUM_PREDICT_EXTRACT=512

# Ollama server env (set by start scripts, documented in README)
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0
OLLAMA_KEEP_ALIVE=10m
```

See `implementation.md` for exact file-by-file build steps, schemas, prompts, and acceptance criteria.

Config surface actually added in Phase 1 (as `DEFAULT_SETTINGS` entries, not Pydantic):
```
MODEL_TIER=auto|small|medium|large
REWRITE_MODEL= / EXTRACT_MODEL=          # empty = use the tier's choice
OLLAMA_KEEP_ALIVE=10m
RAG_DATA_DIR=                            # eval isolation; overrides data/
```

---

## 8. [VERIFIED] Measured baseline — 2026-07-20

`eval/results/baseline.json`. Model `qwen2.5:0.5b`, shipped defaults, 50-question
golden set, judged by `llama3.2:3b`. This is the "before" every later phase compares to.

| Metric | Baseline | Target | Met |
|---|---|---|---|
| Faithfulness | 0.907 | ≥ 0.90 | yes |
| Relevance | 0.861 | ≥ 0.85 | yes |
| Citation rate | 0.628 | — | — |
| Out-of-scope refusal | 0.857 (6/7) | ≥ 0.90 | no |
| False refusals (in-scope) | 0.000 | — | — |
| Recall@5 | 0.963 | — | — |
| MRR | 0.862 | — | — |
| p50 total latency | 17,977 ms | < 4,000 ms | no |
| TTFT (server-side) | 753 ms | < 1,500 ms | yes |

By category, faithfulness degrades exactly where the graph layer is meant to help:
**factoid 0.950 → multihop 0.850 → global 0.844.** That is the gap Phase 3 must close,
and it is now measured rather than assumed.

### Caveats on these numbers
- Measured on **shipped defaults**, not `data/settings.json` (see §1.5) — so not a
  measurement of the live app.
- Citation rate 0.628 and refusal 0.857 are both explained by the current system
  prompt, which *encourages* answering from general knowledge when context is
  missing. Task 4.1 changes that contract; expect both to move sharply.
- Faithfulness is judged by a 3B model. Treat ±0.02 movements as noise, which is
  exactly the width of several acceptance gates — prefer category-level deltas and
  latency for go/no-go decisions.

### Eval harness contract (`eval/run_eval.py`)
- Drives the **real** `/api/chats/ask` in-process via `httpx.ASGITransport`.
- Isolated by env var alone: `RAG_DATA_DIR=eval/.data`, Redis **DB 1**,
  `RATE_LIMIT_PER_MINUTE=0`. Never touches user data. `eval/.data/` is gitignored.
- **Flushes the semantic cache each run** — otherwise a second run measures cache
  hits, not the pipeline.
- Judge must be materially stronger than the model under test (`EVAL_JUDGE_MODEL`,
  default `llama3.2:latest`). A 0.5b judge yields noise.
- **Client-side TTFT is not measurable through `ASGITransport`**: it runs the app to
  completion and buffers the body, so first-token time equals total time by
  construction. Real TTFT and all stage timings are read back from the metrics DB by
  `request_id`. **Task 5.1's bench harness must not use ASGITransport** — it needs a
  real server over HTTP to measure TTFT.
