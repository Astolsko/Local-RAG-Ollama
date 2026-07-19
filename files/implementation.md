# Local-RAG-Ollama — Implementation Guide (for Claude Code)

This document translates `plan.md` into ordered, verifiable engineering tasks.
Execute phases in order. Within a phase, execute tasks in order unless marked parallel-safe.

---

## 0. Ground Rules

1. **Never break existing behavior.** All current API endpoints in `backend/main.py` keep their paths and response shapes; add fields, don't remove them. Run `pytest tests/` and `npm test` (in `frontend/`) after every task.
2. **Feature-flag everything new.** New behavior defaults must be safe: `GRAPH_ENABLED` defaults `false` until Phase 3 acceptance passes, then flips to `true`.
3. **Config via `backend/config.py`** — **[CORRECTED]** this is **not** Pydantic settings. It is a `DEFAULT_SETTINGS` dict plus a module-level `__getattr__` resolving **env var → `data/settings.json` → default**. Add a tunable by adding a dict entry (type is inferred from the default's type), plus an `.env.example` entry and a README config-table row. Do not rewrite this to Pydantic mid-project: `GET/PUT /api/settings` and the settings UI depend on the dict shape.
4. **Every new module gets tests** in `tests/` mirroring existing test style (see `tests/conftest.py` for fixtures). Mock Ollama and Redis in unit tests; mark integration tests that need live services with `@pytest.mark.integration`.
5. **Commit per task** with message format `phase-N: <task summary>`.
6. **Before starting:** read `backend/rag.py`, `backend/main.py`, `backend/config.py`, `backend/retrieval/*.py`, `frontend/src/App.jsx`, `frontend/src/api.js` in full. These are the integration points for everything below.
7. **[ADDED] `backend/chat_stream.py` is the real answer path** and is missing from the original list above. `POST /api/chats/ask` → `chat_stream.chat_stream()`. Every citation, verification, routing and prompt change lands **there**, not in `rag.py`. `rag.py:ask()` serves only the legacy `POST /api/ask`. Read it before Phases 3–5.
8. **[ADDED] Run commands directly — `make` is not installed** on the dev machine. Use `python eval/run_eval.py`. The `Makefile` is deferred to Phase 6 with CI.
9. **[ADDED] Read `plan.md` §1.5 first.** It records what the codebase already contained; several tasks below assume work that is already done.

---

## Phase 1 — Evaluation Harness + Model Quick Wins

> **STATUS: built 2026-07-20.** Baseline recorded in `eval/results/baseline.json`
> (see `plan.md` §8). Deviations from the text below, all deliberate:
> - **No `Makefile`** — `make` is not installed; deferred to Phase 6.
> - **Golden corpus has no real PDF.** Four docs: 3 markdown + 1 `.txt` spec.
>   The `pypdf` extraction path is therefore **not covered by the eval** — relevant
>   to Task 2.1, which swaps it for `pymupdf`.
> - **Eval isolation is env-var based, not a separate collection.** `rag.py` and
>   `bm25_index.py` both hardcode the Chroma collection name `"sources"`, so a
>   separate `eval_golden` collection was not reachable without touching production
>   code. Instead `config.DATA_DIR` gained a `RAG_DATA_DIR` override, and the runner
>   points the whole app (Chroma + BM25 + metrics DB + settings) at `eval/.data`,
>   with Redis on **DB 1** and `RATE_LIMIT_PER_MINUTE=0`.
> - **Golden schema** is `{id, category, question, reference_answer, source_doc, must_cite}`;
>   `source_doc` is `"multiple"` for multihop/global and `null` for out_of_scope.

### Task 1.1 — Golden evaluation set
- Create `eval/golden/documents/` with 3–5 sample documents (public-domain text, a technical PDF converted to text, a markdown doc with headings).
- Create `eval/golden/questions.jsonl`, ~50 entries:
  ```json
  {"id": "q001", "category": "factoid|multihop|global|out_of_scope", "question": "...", "reference_answer": "...", "source_doc": "...", "must_cite": true}
  ```
  Distribution: ~25 factoid, ~10 multihop, ~8 global, ~7 out_of_scope (answer not present in any doc; reference_answer = refusal).
- Write the questions yourself from the sample docs; multihop questions must require combining facts from 2+ chunks/documents.

### Task 1.2 — Eval runner
- Create `eval/run_eval.py`:
  - Ingests `eval/golden/documents/` into a **separate** Chroma collection (never pollute user data; use collection name `eval_golden`).
  - Runs every question through the full `/api/chats/ask` pipeline (call the FastAPI app in-process with `httpx.ASGITransport` — no server needed).
  - Scores with the local LLM as judge (use the medium-tier model):
    - `faithfulness` (0–1): are all claims supported by retrieved chunks?
    - `relevance` (0–1): does the answer address the question?
    - `citation_precision` (0–1, Phase 4+): do `[n]` markers point at supporting chunks?
    - `refusal_correct` (bool, out_of_scope only)
  - Records latency per question. **[CORRECTED] Client-side "first token" is not
    measurable** through `ASGITransport` (it buffers the whole body). The runner
    records wall time per question and joins the pipeline's own metrics DB by
    `request_id` for real TTFT and the per-stage breakdown.
  - Writes `eval/results/<timestamp>.json` + prints a summary table; also writes/updates `eval/results/latest.json`. `--label <name>` additionally writes `results/<name>.json`; `--limit N` runs a subset and skips `history.csv`.
  - **Keeps appending `eval/history.csv`** with the original headers — the metrics
    dashboard's Retrieval tab reads it via `GET /api/observability/eval-history`.
    Changing those headers breaks the UI.
  - **Flushes the semantic cache** (`semantic:cache:keys`) at the start of every run.
    Without this, run 2 measures cache hits, not the pipeline.
  - **Judge model must be stronger than the model under test** (`EVAL_JUDGE_MODEL`,
    default `llama3.2:latest`). A 0.5b judge produces near-noise scores.
  - Report `None`/`n/a`, never `0.0`, for metrics with no applicable rows (MRR for
    multihop/global, which span documents; faithfulness for out_of_scope, where the
    correct behaviour is not answering). An empty average printed as 0.000 reads as a
    total failure.
- ~~Create `Makefile`~~ — **deferred to Phase 6**; `make` is not installed. Run
  `python eval/run_eval.py`.
- **Acceptance:** the runner completes on the current pipeline and produces a baseline results file. Baseline numbers recorded in `eval/results/baseline.json`.

### Task 1.3 — Hardware-aware model tiering
- Create `backend/model_tiers.py`:
  - `detect_tier() -> str`: use `psutil` for total RAM (+ optional GPU detect via `ollama ps`/`nvidia-smi` if trivially available, else RAM only). `<10 GB → small`, `10–20 GB → medium`, `>20 GB → large`.
  - Tier table:
    | tier | LLM_MODEL | REWRITE_MODEL | EXTRACT_MODEL |
    |---|---|---|---|
    | small | `qwen2.5:1.5b` | `qwen2.5:0.5b` | `qwen2.5:1.5b` |
    | medium | `qwen2.5:3b` | `qwen2.5:0.5b` | `qwen2.5:1.5b` |
    | large | `qwen2.5:7b` | `qwen2.5:1.5b` | `qwen2.5:3b` |
- Config: `MODEL_TIER=auto|small|medium|large` (default `auto`); explicit `LLM_MODEL`/`REWRITE_MODEL`/`EXTRACT_MODEL` env vars override tier choices.
- Extend `/api/health` response with `{tier, models, ram_gb}`.
- On startup, log which models are missing from Ollama (`GET /api/tags` on Ollama) and expose `missing_models` in `/api/health` so the UI can prompt the user to `ollama pull`.
- **Acceptance:** unit tests for tier selection boundaries; health endpoint shows tier info; app still works with only the old model pulled (tier falls back with a logged warning to whatever `LLM_MODEL` is available).

### Task 1.4 — Adaptive pipeline + Ollama tuning
- In `backend/rag.py`:
  - Skip query rewrite when the (trimmed) query is < 8 words AND contains no pronouns/anaphora (`it, that, they, this, he, she, those`) AND session has no prior turns. Log `rewrite_skipped=true` to metrics.
  - Add `keep_alive="10m"` to all Ollama generate/embed calls.
- Batch embeddings at ingestion: embed chunks in batches (Ollama `/api/embed` accepts a list) instead of one call per chunk.
- **Acceptance:** re-run `make eval`; latency p50 must drop vs. baseline with faithfulness within ±0.02. Record as `eval/results/phase1.json`.

---

## Phase 2 — Retrieval Quality

> **[VERIFIED] Read before starting — parts of this phase already exist.**
> Phase 1 measurement also makes this the **latency-critical phase**: retrieve
> (8.5 s) + rerank (3.3 s) + query embed (2.1 s) = 14.7 s of a 17,977 ms p50,
> versus 998 ms in generation (`plan.md` §2.5). The p50 < 4 s target is won or
> lost here, **not** in Phase 5.
>
> - **RRF is already implemented** — `hybrid_search.py` `_rrf_fusion`, `RRF_K = 60`.
> - **BM25 is already persisted** to `data/chroma/bm25_index.pkl` and rebuilt on
>   ingest/delete. No per-query rebuild.
> - **`StructureAwareChunker` already exists** in `chunker.py` — but is dead code.
> - **`hybrid_search_sync` now takes `source_ids` and `has_history`.** Preserve both
>   when refactoring; `source_ids` filtering is a correctness fix (see `plan.md` §1.5).

### Task 2.0 — [NEW, do first] Stop rebuilding BM25 on every request
- `metrics_db.log_request()` calls `init_db()`, which calls `build_bm25_index()`.
  `update_faithfulness()` and `update_feedback()` do the same. Every answered
  question therefore rebuilds the entire BM25 index ~2x (request + background
  judge), plus once per thumbs-up/down — O(corpus) work per request.
- Move index-building out of `init_db()`; it belongs only in ingest/delete paths
  (`rag.add_source`, `rag.delete_source`) where it already is. Keep schema
  creation/migration in `init_db()`.
- **Acceptance:** ingest a doc, ask a question, assert `build_bm25_index` is not
  called on the request path (mock/counter). This blocks Task 5.2's load test,
  which will otherwise serialize on it.

### Task 2.1 — Structure-aware chunking
- **[CORRECTED]** `StructureAwareChunker` already exists in `chunker.py` with heading
  splitting and `section_title` extraction, but **nothing calls it** — ingestion goes
  `rag.add_source` → `smart_chunk_page` → `semantic_chunk_text`. The task is to route
  ingestion through it and add offsets, not to write it from scratch.
- **Note:** `semantic_chunk_text` embeds every sentence to find split points. With
  batched embedding (`backend/embeddings.py`, added in Phase 1) that is one request
  per document rather than per sentence — keep using `embed_texts`.
- Modify `backend/ingestion/chunker.py`:
  - Markdown/text: split on heading boundaries (`#`, `##`, `###`) first, then apply existing size/overlap splitting inside oversized sections.
  - PDF: switch extraction to `pymupdf` (add to `requirements.txt`); use block/heading info to find section boundaries; fall back to size-based when structure is absent.
  - Every chunk's metadata gains: `{doc_id, doc_name, section_title, chunk_index, char_start, char_end}` — `char_start/char_end` are offsets into the stored full document text (needed for UI highlighting in Phase 4).
- Store full original document text: create `data/documents/` persistence (SQLite table `documents(id, name, text, created_at)` — add to metrics DB or a new `data/appdata.db`; choose one DB file and use it for all Phase 2–3 tables).
- Add `POST /api/sources/reingest` (wraps existing `reingest.py`) to rebuild chunks for existing docs.
- **Acceptance:** unit tests: markdown with headings produces section-aligned chunks with correct offsets; existing ingestion tests still pass.

### Task 2.2 — RRF fusion — **mostly DONE already**
- ~~replace current score mixing with RRF~~ — **already implemented**: `_rrf_fusion`
  with `RRF_K = 60`. Only remaining piece: cap candidates at `RRF_TOP_N` (default 12)
  before reranking. This is also a **latency fix** — reranking currently sees ~80
  candidates (`top_k * 2` per query × up to 4 rewritten queries), costing 3.3 s.
- ~~Persist the BM25 index to disk~~ — **already done**, at `data/chroma/bm25_index.pkl`.
  (But see Task 2.0: it is being rebuilt on the *request* path by the metrics logger.)
- **Still to do — the real win here:** run dense and BM25 concurrently. Currently
  `hybrid_search_sync` loops queries **serially**, running vector then BM25 for each,
  via `loop.run_until_complete`. With rewriting active that is up to 8 sequential
  searches. Parallelise across queries *and* across the two backends.
- **Acceptance:** unit test for RRF math with a known fixture; eval retrieval metrics
  ≥ baseline (recall@5 0.963, MRR 0.862) **and** `retrieve_p50_ms` well below 8,501 ms.

### Task 2.3 — FlashRank reranker option
- Add `flashrank` to requirements. In `backend/retrieval/reranker.py`, support `RERANKER=flashrank|cross-encoder|off` (default `flashrank`, model `ms-marco-MiniLM-L-12-v2` or FlashRank's default nano model). Lazy-load on first use.
- Confidence shortcut: if the top dense-retrieval similarity ≥ `RERANK_SKIP_THRESHOLD` (default 0.85), skip reranking and log it.
- **Acceptance:** eval faithfulness/relevance within ±0.02 of cross-encoder while rerank latency drops; both backends selectable via env.

### Task 2.4 — Embedding cache
- New `backend/cache/embed_cache.py`: Redis hash keyed by `sha256(model + text)`, TTL `EMBED_CACHE_TTL` (default 7 days). Wrap all embedding calls (queries and ingestion).
- **Acceptance:** unit test proves second embed of identical text hits cache (mock Redis); metrics DB logs embed cache hit rate.
- Record `eval/results/phase2.json`.

---

## Phase 3 — GraphRAG-lite

All graph code lives in `backend/graph/`. Everything is behind `GRAPH_ENABLED` (default `false` until Task 3.6 passes).

### Task 3.1 — Graph store
- Create `backend/graph/store.py` using the app SQLite DB:
  ```sql
  CREATE TABLE IF NOT EXISTS entities(
    id INTEGER PRIMARY KEY, name TEXT, type TEXT, norm_name TEXT,
    UNIQUE(norm_name, type));
  CREATE TABLE IF NOT EXISTS relations(
    id INTEGER PRIMARY KEY, source_id INT, target_id INT,
    predicate TEXT, chunk_id TEXT, confidence REAL);
  CREATE TABLE IF NOT EXISTS entity_chunks(
    entity_id INT, chunk_id TEXT, doc_id TEXT, PRIMARY KEY(entity_id, chunk_id));
  CREATE TABLE IF NOT EXISTS communities(
    id INTEGER PRIMARY KEY, level INT, summary TEXT, entity_ids TEXT);
  ```
  `norm_name` = lowercased, whitespace-collapsed name for dedup. Provide `load_networkx()` returning an in-memory `networkx.Graph` (cache it; invalidate on writes).

### Task 3.2 — Entity/relation extraction
- Create `backend/graph/extractor.py`:
  - Per chunk, call Ollama with `EXTRACT_MODEL`, `format="json"`, and a prompt (add to `prompt_templates.py`) requesting:
    ```json
    {"entities":[{"name":"...","type":"person|org|location|concept|event|other"}],
     "relations":[{"source":"...","target":"...","predicate":"<verb phrase>"}]}
    ```
    Prompt rules: entities must appear verbatim-or-near-verbatim in the chunk; ≤10 entities, ≤10 relations per chunk; predicate ≤5 words.
  - Validate with Pydantic. On validation failure: one retry appending the validation error to the prompt ("repair"). On second failure: log, skip chunk, continue.
  - Dedup entities via `norm_name` on insert; relations link entity ids and record the source `chunk_id` with a default confidence 1.0.
- **Acceptance:** unit tests with mocked Ollama covering: valid JSON, malformed-then-repaired, double failure (chunk skipped, no exception).

### Task 3.3 — Background indexing job
- Create `backend/graph/indexer.py`:
  - `index_document(doc_id)` runs extraction over that doc's chunks; triggered automatically after each successful ingestion (FastAPI `BackgroundTasks`) when `GRAPH_ENABLED`.
  - Job status in Redis: `graph:index:{doc_id} = {state: queued|running|done|error, done_chunks, total_chunks}`.
  - Endpoints: `POST /api/graph/index/{doc_id}` (manual trigger/re-run), `GET /api/graph/status` (all jobs), `GET /api/graph/summary` (entity/relation/community counts).
  - Deleting a source (`DELETE /api/sources/{id}`) also removes its entities' links/relations (drop orphaned entities).
- **Acceptance:** integration test: ingest a small doc → poll status to `done` → store has entities.

### Task 3.4 — Community detection + summaries
- Create `backend/graph/communities.py`: run Louvain (`python-louvain`) over the networkx graph after indexing completes; for each community with ≥3 entities, generate a ≤150-word summary with the LLM from its top relations + a sample of linked chunk texts; store in `communities`.
- Recompute incrementally-cheap: full recompute is acceptable but must run in the background job, never in a request path.
- **Acceptance:** unit test with a synthetic graph asserts ≥1 community found and summaries stored (LLM mocked).

### Task 3.5 — Router + graph retrieval
- Create `backend/retrieval/router.py`:
  - Heuristics first: query mentions "summarize/overview/themes/across/all documents" → `global`; contains "relationship/related/connect/between/compare X and Y" or ≥2 known entity mentions (check against `entities.norm_name`) → `relational`; else → `factoid`.
  - If heuristics are ambiguous (define: relational keywords but <2 entity matches), one call to `REWRITE_MODEL` to classify. `ROUTER_MODE` env can pin a mode.
- Create `backend/graph/retrieval.py`:
  - *Relational:* link query entities (substring/fuzzy match on `norm_name`) → expand `GRAPH_MAX_HOPS` (default 2) neighborhood → collect linked chunks + relation triples rendered as text lines ("A —predicate→ B"). Merge these candidates with vector-path candidates, dedupe by chunk id, rerank together, keep TOP_K.
  - *Global:* retrieve all community summaries → map: ask LLM which are relevant to the query → reduce: answer from selected summaries; cite community ids. Skip chunk retrieval.
- Wire into `backend/rag.py` behind `GRAPH_ENABLED`. `/api/chats/ask` response gains `"route": "vector|relational|global"` and metrics DB logs the route.
- **Acceptance:** unit tests for router labels on a fixture query list; integration test: a multihop question over two ingested docs returns chunks from both.

### Task 3.6 — Graph eval gate
- Extend eval judge to score multihop/global categories separately.
- **Acceptance to flip `GRAPH_ENABLED` default to `true`:** multihop + global scores improve over `phase2.json` with factoid scores within ±0.02 and factoid latency unchanged (router must not slow the fast path). Record `eval/results/phase3.json`.

---

## Phase 4 — Grounding, Verification, UI

### Task 4.1 — Citation contract

> **[VERIFIED] There are TWO system prompts, and the user-editable one is ignored
> by the main chat path.**
> - `chat_stream.py` hardcodes `prompt_templates.SYSTEM_PROMPT_TEMPLATE`.
> - `config.DEFAULT_SYSTEM_PROMPT` + `data/system_prompt.txt` (written by
>   `PUT /api/settings/system-prompt`, read via `rag.get_system_prompt()`) are used
>   **only** by the legacy `POST /api/ask`.
> - So editing the system prompt in the UI currently changes nothing for
>   `/api/chats/ask`. Fix this here: make `chat_stream` use
>   `rag._ensure_system_prompt_cached()` and collapse the two prompts into one source
>   of truth, or the citation contract below will silently not apply.
>
> **Both current prompts actively work against the refusal goal** — they instruct the
> model to answer from general knowledge when context is missing, merely flagging that
> it did so. Baseline: refusal 0.857, citation rate 0.628. Expect both to move sharply
> once the contract changes; re-baseline rather than comparing across the prompt change.

- Prompt change (`prompt_templates.py`): context chunks are numbered `[1]..[k]`; instruct: "Every factual sentence must end with its supporting citation(s) like [2]. If the context does not contain the answer, say so and cite nothing."
- Parse citations from the streamed answer server-side; `/api/chats/ask` response (and stream-final event) gains:
  ```json
  "citations": [{"n": 1, "chunk_id": "...", "doc_id": "...", "doc_name": "...", "section_title": "...", "char_start": 0, "char_end": 512}]
  ```
- **Refusal gate:** if the best post-rerank score < `CONFIDENCE_REFUSAL_THRESHOLD` (default 0.35), bypass generation with a fixed grounded-refusal template (still shows nearest sources as "possibly related").

### Task 4.2 — Faithfulness check
- Create `backend/verification/faithfulness.py`: after generation, split the answer into sentences; for each sentence with citation `[n]`, compute embedding similarity to the cited chunk; sentences below `FAITHFULNESS_MIN_SIM` (default 0.45) or citing nothing while making factual claims get `"flagged": true` in a `verification` array on the response. Runs async after streaming completes; delivered as a final SSE event so it never delays first token.
- Log flagged-sentence rate to metrics DB.
- **Acceptance:** citation_precision now scored in `make eval`; unsupported-claim rate < 5% on golden set; out_of_scope refusal ≥ 90%.

### Task 4.3 — Frontend: citations + source viewer
- `App.jsx` answer rendering: convert `[n]` into clickable chips.
- New `frontend/src/components/SourceViewer.jsx`: side panel showing the full document text (`GET /api/documents/{doc_id}` — add this endpoint returning stored full text) with the cited span highlighted via `char_start/char_end`, auto-scrolled into view. Chip click opens it.
- Flagged sentences (from `verification`) get a subtle warning underline with tooltip "not fully supported by sources".

### Task 4.4 — Frontend: knowledge-graph explorer
- Add route/page `frontend/src/pages/GraphExplorer.jsx` using `react-force-graph-2d`.
- Backend `GET /api/graph/export?limit=500` returns `{nodes:[{id,name,type,degree}], links:[{source,target,predicate}]}` (top-degree nodes first).
- Interactions: node click → side card with entity's relations + linked chunks (fetch `GET /api/graph/entity/{id}`); search box to focus a node; color by entity type; community hulls optional (stretch).
- When an answer used the relational route, response includes the entity path used; "view in graph" button on the message deep-links to the explorer with those nodes highlighted.

### Task 4.5 — Frontend: settings panel + answer HUD
- `frontend/src/components/SettingsPanel.jsx` backed by new `GET/PUT /api/settings` (model tier, TOP_K, chunk size [applies on next ingest], `RERANKER`, `GRAPH_ENABLED`, `ROUTER_MODE`). Persist to a small settings table; config precedence: env < settings DB < request overrides.
- Per-message HUD: route badge, cache-hit badge, latency, tokens/sec (backend already tracks these in metrics — surface them on the ask response).

### Task 4.6 — Visual polish pass
- Introduce CSS design tokens (spacing, radii, palette) in `index.css`; dark/light theme toggle persisted in `localStorage` (this is a Vite app, not a Claude artifact — localStorage is fine here); loading skeletons; empty states; keyboard shortcut (Ctrl/⌘+K focus input); responsive layout ≤ 768px.
- **Phase acceptance:** vitest coverage for chips, SourceViewer highlight math, settings panel round-trip; `make eval` recorded as `eval/results/phase4.json`.

---

## Phase 5 — Inference Speed & Performance

> ## [VERIFIED] Read this before planning Phase 5 — its premise does not hold here
>
> Measured on the golden set (`plan.md` §2.5): **generation is 998 ms of a 17,977 ms
> p50 — about 6%.** Real TTFT is already **753 ms**, inside the < 1.5 s target. The
> remaining 14.7 s is retrieval: query rewrite expansion (8.5 s), cross-encoder
> reranking (3.3 s), query embedding (2.1 s).
>
> Consequences:
> - **Phase 5 cannot deliver the p50 < 4 s target.** A perfect 2x decode win removes
>   ~500 ms from an 18-second answer. That target belongs to Phase 2 (and Task 2.0).
> - The ≥1.3x decode-throughput goal is still *achievable and worth measuring* — just
>   do not sell it as user-perceived latency. Frame this phase as throughput and
>   efficiency, not responsiveness.
> - **Re-measure the breakdown after Phase 2** before committing to 5.5/5.6 (backend
>   abstraction, llama.cpp build, speculative decoding). If retrieval drops to ~2 s,
>   generation becomes ~33% of total and this phase's value rises sharply. Decide then.
> - **Task 5.1's harness must not use `httpx.ASGITransport`.** It runs the app to
>   completion and buffers the response body, so client-side TTFT equals total latency
>   by construction. Measure TTFT against a real `uvicorn` server over HTTP, or read it
>   from the metrics DB by `request_id` as `eval/run_eval.py` does.

**Context for this phase.** Token generation is memory-bandwidth-bound: each token requires reading essentially all model weights, so decode speed ≈ bandwidth ÷ model bytes. Work below targets four levers: fewer bytes per token (quantization), faster memory (GPU offload), multiple tokens per pass (speculative decoding), and skipped work (prefix caching, output caps). Do NOT pursue OS/kernel-driver approaches — the bottleneck is userspace GGML compute kernels and memory bandwidth, not the OS.

**Rule for this phase:** Task 5.1's benchmark harness is built FIRST. Every later task in this phase must show its effect in a bench run, and `make eval` faithfulness must stay within ±0.02 of `phase4.json` after every change (speed never buys quality regressions).

### Task 5.1 — Inference benchmark harness (build first)
- Create `eval/bench_inference.py` measuring, per configuration cell:
  - **TTFT** (request sent → first streamed token), **decode tok/s** (tokens after the first ÷ elapsed), **total latency**, and — when the backend reports them — `prompt_eval_count` / `eval_count` from the final stream stats (Ollama includes these in its terminal streaming response; llama-server exposes timings).
  - Protocol: 1 priming run + 5 warm runs per cell; report median.
- Fixed prompt set (checked into `eval/bench_prompts.json`):
  1. `short_factoid` — ~50-token prompt, `num_predict=128`.
  2. `rag_shaped` — realistic prompt: system prompt + 4 numbered context chunks (~2,500 tokens total) + question, `num_predict=512`.
  3. `repeated_prefix` — two sequential requests sharing an identical system+context prefix but different questions (measures prefill/prefix-cache reuse via the second request's `prompt_eval_count` and TTFT).
- Configuration matrix via CLI flags: `--backend ollama|llamacpp|llamacpp-draft`, `--model <tag>`, plus option overrides. Output: `eval/results/bench_<timestamp>.json` + a printed markdown table.
- Add `make bench` target.
- **Acceptance:** run against the CURRENT stock configuration before any other Phase 5 task and commit `eval/results/bench_baseline.json`.

### Task 5.2 — Concurrency + memory audit
- Verify dense/BM25 run concurrently (Task 2.2), extraction never blocks requests, reranker/extractor lazy-load. Add an idle unloader: models unused for 15 min release memory (FlashRank/cross-encoder objects deleted; rely on Ollama keep_alive expiry for LLMs).
- Load-test script `eval/load_test.py` (10 concurrent asks) — no errors, p95 recorded.

### Task 5.3 — Ollama runtime tuning + quantization defaults
- **Explicit quant tags in `backend/model_tiers.py`:** replace bare tags with pinned quants, e.g. medium tier → `qwen2.5:3b-instruct-q4_K_M` (generation), `qwen2.5:0.5b-instruct-q4_K_M` (rewrite), `qwen2.5:1.5b-instruct-q4_K_M` (extract); analogous for small/large. Verify each tag exists in the Ollama library (`ollama pull` dry-run in a setup check) — if a pinned quant tag is unavailable, fall back to the bare tag and log it. Document `q8_0` variants as the quality-first upgrade path in README.
- **Per-role context/output budgets** in `config.py`, passed as options on every generate call:
  - `NUM_CTX_GENERATE=8192`, `NUM_CTX_REWRITE=2048`, `NUM_CTX_EXTRACT=4096`
  - `NUM_PREDICT_GENERATE=1024`, `NUM_PREDICT_REWRITE=128`, `NUM_PREDICT_EXTRACT=512`
- **Thread pinning on CPU-only:** when no GPU detected, set `num_thread = psutil.cpu_count(logical=False)`.
- **Server-side flags:** update `start-app.ps1`/`.bat` (and later `scripts/start.sh`) to export before starting/checking Ollama: `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_KEEP_ALIVE=10m`. README must state clearly these are env vars for the *Ollama server process*, not the backend. `/api/health` echoes the intended flag configuration.
- **Acceptance:** `make bench` vs `bench_baseline.json`: decode tok/s ≥ baseline, TTFT not worse; `make eval` within ±0.02 faithfulness. Record `bench_phase5_tuning.json`.

### Task 5.4 — Prefix-cache-friendly prompt assembly
- Audit prompt construction in `backend/rag.py` + `prompt_templates.py`. Enforce this exact assembly order, byte-stable across requests: `[static system prompt] [static instruction block] [context chunks — ordered by (rerank score desc, chunk_id asc) for a deterministic tiebreak] [conversation history] [current question]`.
- Forbidden inside the prefix: timestamps, request/session ids, random ordering, any per-request variation before the history section.
- Unit test: build the prompt twice with identical inputs → assert byte-identical output; build with same context but different question → assert the shared prefix (up through context) is byte-identical.
- **Acceptance:** in the `repeated_prefix` bench case, the second request shows reduced `prompt_eval_count` and lower TTFT than the first (prefill reuse). If the runtime does not exhibit reuse, investigate (model reload between calls? keep_alive? context shift?) and document findings in the bench report — do not silently pass.

### Task 5.5 — Inference backend abstraction
- Create `backend/llm/client.py`:
  - `class InferenceClient(Protocol)` with `async def generate_stream(messages, options) -> AsyncIterator[TokenEvent]` (yields tokens, then one final `StatsEvent` with token counts/timings) and `async def generate_json(messages, schema, options)`.
  - `OllamaClient`: move ALL existing Ollama generation HTTP logic here, behavior-identical.
  - `LlamaCppClient`: targets llama.cpp `llama-server`'s OpenAI-compatible `/v1/chat/completions` with `stream=true`; maps options (`num_predict`→`max_tokens`, temperature, stop) and parses SSE chunks; reads server timings for the StatsEvent when available.
  - **Embeddings are NOT abstracted:** they stay on Ollama (`nomic-embed-text`) regardless of `INFERENCE_BACKEND`.
- Refactor every generation call site to use a factory: `rag.py`, `retrieval/query_rewrite.py`, `ingestion/contextualizer.py`, `graph/extractor.py`, `graph/communities.py`, the eval judge.
- Config: `INFERENCE_BACKEND=ollama` (default) | `llamacpp`; `LLAMACPP_BASE=http://localhost:8080`.
- **Acceptance:** unit tests with mocked HTTP for both clients (stream parsing, option mapping, error/timeout paths); full `pytest` green with backend=ollama; `make eval` unchanged within noise; switching backends requires only the env var.

### Task 5.6 — llama.cpp build + speculative decoding
- `scripts/build_llamacpp.sh` (+ `.ps1` equivalent):
  - Clone llama.cpp; `cmake -B build -DGGML_NATIVE=ON`; if `nvcc` present add `-DGGML_CUDA=ON` (auto arch); macOS gets Metal by default. Build `llama-server`.
  - `--prebuilt` flag: skip compiling, download the latest official release binary for the platform instead (the friction fallback).
- `scripts/run_llamacpp.sh` (+ `.ps1`):
  - Downloads GGUFs if absent (document exact `huggingface-cli download` commands in the script header for the tier's Qwen2.5-instruct q4_K_M target GGUF and the `qwen2.5-0.5b-instruct` q4_K_M draft GGUF).
  - Launch: `llama-server -m <target.gguf> --ctx-size $NUM_CTX_GENERATE --port 8080 --flash-attn` and, when `DRAFT_MODEL` is set, append `--model-draft <draft.gguf>` (llama.cpp's draft-model speculative decoding).
  - Comment block: the draft MUST share the target's tokenizer family — pin qwen2.5-0.5b for qwen2.5 3b/7b targets; never mix families.
- **Ollama-native speculation check:** at startup with backend=ollama, detect the installed Ollama version; consult its `--help`/docs output for native speculative-decoding support rather than assuming any env var name (support has been landing in recent versions and the interface may differ). If present, log the enablement instructions and note it in `/api/health`; do not hard-fail or hardcode.
- **Acceptance:** with `INFERENCE_BACKEND=llamacpp`, full ingest + ask works end-to-end; `make bench` matrix run covering `{ollama-tuned, llamacpp, llamacpp-draft}` on the medium-tier model recorded as `bench_phase5_backends.json`; draft acceptance rate captured from llama-server timings/logs where exposed.

### Task 5.7 — Recommended-config gate
- Compare `bench_phase5_backends.json`: if `llamacpp-draft` achieves **≥1.3x decode tok/s** vs tuned Ollama at equal quant AND `make eval` quality holds within ±0.02 → mark it the "recommended fast path" in README and surface it as a hint in the Settings panel. Otherwise, tuned Ollama stays recommended and the README publishes the honest numbers either way.
- Settings panel gains an inference-backend selector (ollama / llamacpp) showing the latest bench numbers per backend when available.

---

## Phase 6 — Packaging + Deployment

### Task 6.1 — uv + cross-platform scripts
- Add `pyproject.toml` (uv-compatible; keep `requirements.txt` generated from it for compatibility).
- `scripts/start.sh` + keep existing `.ps1/.bat` (update them if paths changed; all launchers export the Ollama server flags from Task 5.3); `make dev` starts Redis check, backend, frontend.

### Task 6.2 — Docker Compose
- `Dockerfile.backend` (python slim + uv install), `Dockerfile.frontend` (node build → nginx serve, proxy `/api` to backend).
- `docker-compose.yml`: `backend`, `frontend`, `redis`; Ollama:
  - Default: host Ollama, backend uses `OLLAMA_BASE=http://host.docker.internal:11434` (with `extra_hosts` mapping for Linux).
  - Optional profile `--profile bundled-ollama` adding an `ollama/ollama` service for Linux users — set `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_KEEP_ALIVE=10m` in that service's environment.
- The `llamacpp` backend is documented as a **native-only** path in v1 (containerizing a `-march=native` build defeats its purpose); compose users run llama-server on the host and set `LLAMACPP_BASE` accordingly.
- Healthchecks on all services; named volume for `data/`.
- **Acceptance:** `docker compose up` → UI reachable, ingest + ask works end-to-end.

### Task 6.3 — CI
- `.github/workflows/ci.yml`: ruff lint, `pytest -m "not integration"`, frontend `npm test` + `npm run build`. (Full eval and bench need Ollama — keep them local via `make eval` / `make bench`, not CI.)

### Task 6.4 — Docs
- Rewrite README: 2-path quickstart (Docker vs native/uv), updated architecture mermaid diagram including graph layer + router + inference backend abstraction, new config table, screenshots/GIF placeholders, updated roadmap marking Phases 4–5 of the old roadmap complete.
- Record final `eval/results/final.json`; README "Benchmarks" section publishes BOTH tables: quality (`baseline.json` → `final.json`) and inference speed (`bench_baseline.json` → `bench_phase5_backends.json`), with the hardware used clearly stated.

---

## Definition of Done (whole project)

- [ ] All phases' acceptance criteria met; test suites green; CI green
- [ ] `make eval`: faithfulness ≥ 0.90, relevance ≥ 0.85, citation precision ≥ 0.90, refusal ≥ 90%, unsupported-claims < 5%
- [ ] Latency targets from `plan.md` §2 met on the medium tier
- [ ] `docker compose up` works from a clean clone
- [ ] README reflects reality; before/after benchmark table published
