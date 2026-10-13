# PROGRESS

Source of truth for resuming. Each milestone gets one entry ending in `COMPLETE` once `make gate` is green and `make freeze` has run.

## Decisions

- **Eval repo = pallets/click 8.5.0, indexing only `src/`.** Library code is what users ask about; click's `tests/` has many same-named test functions that would make `definition` questions ambiguous.
- **Span convention:** symbol spans start at the first decorator line when decorated (matches tree-sitter's `decorated_definition`), end at the last body line.
- **Qualified names:** dotted module path from the indexed root (`pkg/__init__.py` -> `pkg`) plus enclosing class/function names (no `<locals>`).
- **Kinds:** `class`, `method` (def directly in a class body), `function` (everything else, including nested defs).
- **Calls convention:** final identifier of each call whose callee is a Name or attribute chain, inside the body, excluding nested def/class nodes and their decorators.
- **Resolved edges (oracle + system):** only scope-resolvable calls count as `resolved` (nested defs, same-module defs, imports incl. aliases/relative/re-exports, `self.`/`cls.` methods of the enclosing class, attribute access on imported modules). Anything else is ambiguous/unresolved, never guessed.
- **Freeze marker:** `scripts/frozen.py freeze` writes `tests/FROZEN_MARKER` so `check` can distinguish "never frozen" (warning) from "manifest deleted" (error).
- **`make claims`** runs `scripts/claims.py`, which runs `pytest -m claim` with junit XML and writes `reports/claims.json`; claim tests may write their measured numbers into `reports/claims_details.json`.
- **Offline env:** `tests/conftest.py` and the Makefile set `HF_HUB_OFFLINE=1`; the offline smoke test also points HTTP(S)_PROXY at a closed port so any download attempt fails.

## Milestones

### M0 — Bootstrap, scaffold, guardrails — COMPLETE
- Env: Python 3.12.12 (pyenv), git 2.50.1. `.venv` with the M0 dependency list; `requirements.lock` from `pip freeze`.
- Eval repo: pallets/click tag 8.5.0 (highest non-prerelease from `git ls-remote --tags`), SHA in `data/EVAL_REPO_SHA.txt`. Counts in `reports/m0_eval_repo.json`: 17 files, 579 functions/methods, 214 with docstrings (passes 300/150 check).
- Embedding model `BAAI/bge-small-en-v1.5` downloaded to `data/models/` by `scripts/fetch_model.py`; loads offline (dim 384 at fetch time; the eval-marked smoke test re-verifies offline).
- Fixture repo: 13 files in 3 packages (`shopcore`, `storage`, `service`), 60 functions/methods, 9 classes (8 with methods), async defs, nested defs, decorators, a unicode identifier, two `validate` functions in different modules (for M6 ambiguity), `import x as y` and `from x import f as g` aliases, and a prompt-injection string in `service/api_handlers.py` (for M7).
- `tests/fixtures/expected.json` hand-transcribed from numbered listings (69 symbols, 33 resolved edges). The oracle cross-check caught one hand error (missed a `range` call in `retry.decorator.wrapper`), fixed in the JSON.
- `tests/helpers/oracle.py`: ast oracle (symbols, calls, scope-resolved edges, name-callers).
- `scripts/frozen.py` (freeze/check with marker file), `scripts/claims.py`, Makefile, `thresholds.toml` (empty floors).
- Tests: 20 gating unit tests + 1 eval test, all passing (`reports/m0_gate.txt`). Frozen: 23 files, 0 floors.
- Note: `scripts/frozen.py` was written before its tests in M0 (bootstrap); all later milestones are test-first.
- Known gaps: none.

### M1 — Ingest and symbol extraction — COMPLETE
- `ingest.py`: `walk_repo(root, ignore, max_bytes)` -> sorted `SourceFile(path, text)`; skips `.git/venv/.venv/node_modules/__pycache__`, files > 1 MB, non-UTF-8, symlinks, non-`.py`; glob ignore list plus optional `.coderagignore`.
- `symbols.py`: tree-sitter `Symbol(kind, name, qualified_name, path, start_line, end_line, docstring, decorators, is_async, parent)`; `parse_source` returns `ParseResult(symbols, warnings)` and never raises on syntax errors; `iter_definitions` returns (Symbol, node) pairs for reuse by the call graph. Spans include decorators; end line ignores trailing comments that tree-sitter folds into blocks (a non-obvious mismatch with `ast`). Docstrings follow `ast.get_docstring` semantics (concatenated strings, escapes, raw/f/bytes prefixes, `inspect.cleandoc`).
- Tests: 33 gating + 3 eval passing (`reports/m1_gate.txt`). Symbol match with the oracle is exact on the fixture (69 symbols) and on click `src/` (667 symbols, docstrings included).
- Known gaps: `match` statement bodies are not searched for defs (the oracle doesn't either).

### M2 — AST-aware chunking (plus baseline) — COMPLETE
- `chunking.py`: `ast_chunks(file, max_tokens=400)` emits one chunk per top-level function, class (header + docstring + field code, methods carved out as holes), and method, plus module-level chunk(s). Functions nested inside functions stay in the parent chunk. Each `Chunk` has `chunk_id, path, symbol, kind, start_line, end_line, segments, code, text` (text = `# path :: qualified_name` header + code), `part/parts` for over-budget splits, and `partial` (tree-sitter says the text does not parse standalone).
- Splits happen only at statement starts (recursing into over-budget compound statements and their else/except/... clauses), greedy-packed; pieces are contiguous and lossless.
- `fixed_chunks(file, size=40, overlap=10)` line-window baseline.
- `symbols.split_lines` splits on `\n` only: `str.splitlines` also splits on form feed / U+2028 and would shift line numbers versus tree-sitter and ast.
- Tests: 47 gating + 5 eval passing (`reports/m2_gate.txt`), incl. a 50-function random property test and a click-wide coverage/no-overlap/lossless test. Chunk stats on click in `reports/m2_chunk_stats.json` (budget 400: 694 chunks, 110 split pieces, 0 partial).
- Decision: token budget is a regex token count (`\w+|[^\w\s]`), default 400, roughly under bge-small's 512 wordpiece limit for typical code (not measured).

### M3 — SQLite store and lexical (BM25) search — COMPLETE
- `store.py`: tables `meta, files, chunks, symbols, edges, chunks_fts (FTS5, unicode61 with '_' as a token char), embeddings`. `index_repo(root, db, chunker="ast"|"fixed", ...)` keys files by sha256 and replaces only changed files' rows; deleted files are purged from every table; changing chunker settings or root triggers a full rebuild (the embedding cache is kept). `Store` exposes chunk/symbol lookups and `hits_for`.
- `lexical.py`: `split_identifier` (snake/camel/Pascal/acronym/digit aware, unicode-safe via `str.isupper/islower`), `tokenize` (original lowercased identifier + parts; query mode drops a generic English stopword list and dedupes), `search_lexical(store, query, k)` using FTS5 `bm25()` (score = -bm25).
- Tests: 72 gating + 5 eval passing (`reports/m3_gate.txt`), including 11 hand-written fixture queries (expected symbol in top 3).

### M4 — Eval question generator and retrieval metrics — COMPLETE
- `evalgen.py`: own stdlib-`ast` pass (independent of tree-sitter and of the frozen test oracle). Types: `definition` (unique, non-dunder names; class truth = header region up to its first direct method), `docstring_masked` (first sentence, words sharing a sub-word with the name removed, >= 4 words left), `callers` (unique target names with 1-10 name-based callers). Deterministic per seed; default mix 40/40/20 %. Click pools: 327 / 186 / 186 candidates; sampled 80 / 80 / 40 = 200 (`reports/questions_eval.json`).
- `metrics.py`: `overlap_fraction`, `hit` (same path and >= 50 % of the truth span's lines covered by the chunk's segments), `recall_at_k`, `mrr` (both None when there is no truth), `evaluate_retrieval`.
- `benchmark.py`: BM25 search adapter, `floor_value`, read-only `git_head`.
- BM25 baseline (`reports/baseline_bm25.json`), click recall@5: definition 0.9125, docstring_masked 0.9375, callers 0.7358 (recall@1: 0.70 / 0.8375 / 0.1675). Fixture recall@5: 0.9818 / 1.0 / 1.0.
- Floors added (measured - 0.03, rounded down): `bm25_definition_recall_at_5 = 0.88`, `bm25_docstring_masked_recall_at_5 = 0.90`, `bm25_callers_recall_at_5 = 0.70`, `fixture_bm25_* = 0.95 / 0.97 / 0.97`. The fixture floor test is gating (deterministic, offline); the real-repo floor test is `eval`.
- Tests: 89 gating + 9 eval passing (`reports/m4_gate.txt`).
- Known gaps: name-based `callers` truth counts any call with that final name (e.g. a parameter named `func`), so some truth sets are noisy; definition recall@1 is low (0.70) because callers of a name outrank its definition under BM25.

### M5 — Embeddings and hybrid retrieval — COMPLETE (claim FAILED, see BLOCKERS.md)
- `embeddings.py`: `Embedder` protocol (`name`, `dim`, `embed_documents`, `embed_query`), `HashEmbedder` (blake2b signed feature hashing over identifier-aware tokens), `LocalEmbedder` (fastembed `BAAI/bge-small-en-v1.5`, `local_files_only=True`, cache dir `data/models/`), `embed_chunks` (SQLite cache keyed by (content_hash, model); returns an empty index if the embedder yields nothing).
- `hybrid.py`: `rrf` (weighted, k=60, deterministic tie-break), `VectorIndex` (numpy brute-force cosine), `vector_search`, `HybridSearcher` (BM25 + vector via RRF, falls back to BM25 order when there are no vectors). Default w_lex=1.0, w_vec=0.5.
- `benchmark.compare_retrieval` / `comparison_markdown`; reports `reports/retrieval_comparison.json/.md`.
- Click test set recall@5 (bm25 / vector / hybrid): definition 0.9125 / 0.9375 / 0.9250; docstring_masked 0.9375 / 0.7750 / 0.9000; callers 0.7358 / 0.4308 / 0.5658.
- Floors added: `hybrid_definition_recall_at_5 = 0.89`, `hybrid_docstring_masked_recall_at_5 = 0.87`, `hybrid_callers_recall_at_5 = 0.53`, `fixture_hybrid_* = 0.97` each.
- `make claims` run: `reports/claims.json` shows the M5 claim failed (hybrid 0.9000 vs BM25 0.9375). Details and both tuning approaches in BLOCKERS.md.
- Tests: 100 gating + 15 eval passing (`reports/m5_gate.txt`).
- Decision: tuning was done on a disjoint dev split (seed 1) so the seed-0 test numbers are not tuned on.
- [Unverified] First embedding of click's 694 chunks took roughly 3 minutes on this Mac (seen in console, not saved to reports). The cache makes later runs fast.

### M6 — Call graph and reference lookup — COMPLETE
- `callgraph.py`: tree-sitter call sites per function (nested def/class bodies excluded), import maps (plain, aliased, relative, from-package, re-exports), resolution with `resolved` / `ambiguous` (one row per same-named candidate, never a silent pick) / `unresolved` (builtins, external modules, no candidates). `find_callers` / `find_callees` accept a qualified or simple name. Edges are rebuilt for the whole index whenever any file changes (`store._after_change`).
- Fixture floors (spec values): `callgraph_fixture_precision = 0.95`, `callgraph_fixture_recall = 0.95`; measured resolved edges equal the 33 hand-written edges exactly.
- Click (`reports/callgraph.json`): resolved-edge precision 1.0, recall 1.0 vs the ast oracle (402 edges); edge rows: 487 resolved, 1599 ambiguous, 1202 unresolved. This is agreement between two independent implementations of the same static rules, not runtime call accuracy.
- Bug found and fixed: py-tree-sitter returns a fresh `Node` wrapper per access, so `child is src_node` never matched; `from types import X` then also bound `types -> types.types`, shadowing `from . import types`. Compare by `start_byte` instead. Recall on click rose to 1.0 (`reports/callgraph.json`); the pre-fix value was only seen in console [Unverified: 0.9876].
- Known limits (documented in the module docstring): no type inference (`obj.m()` on params/attributes is ambiguous at best), inherited methods via `self` are ambiguous, no `getattr`/dynamic dispatch, star imports not followed.
- Tests: 109 gating + 16 eval passing (`reports/m6_gate.txt`).

### M7 — Tools, LLM client, and agent loop — COMPLETE
- `llm.py`: `LLMClient` protocol; `AnthropicClient` (Messages API via the `anthropic` 1.x SDK; model id only from `CODERAG_MODEL`, raises if unset; returns text, tool calls, usage, stop_reason, and the raw content blocks so thinking/tool_use blocks are appended back unchanged); `ScriptedLLM` (replays scripted turns, records every request).
- `tools.py`: `search_code` (hybrid searcher if given, else BM25; locations only), `search_symbols`, `read_file` (rejects absolute/Windows/`~`/`..`/NUL paths and anything whose resolved path leaves the root, incl. symlinks; caps lines and file size), `find_callers`. Arguments validated against each tool's JSON schema (unknown tool, non-object args, missing/extra/mistyped args -> `ToolError`); output capped.
- `agent.py`: loop with `max_steps=8`, all tool results of a turn returned in one user message, outputs wrapped in `<tool_output>` with an UNTRUSTED label and closing-tag neutralization, truncation, JSON answer parsing (fences tolerated) with a format-reminder retry, structured failures (`max_steps_exhausted`, `invalid_final_answer`, `refusal`, `llm_error: ...`), token usage summed across steps.
- `tests/test_m7_live.py` is `live` (skips without `ANTHROPIC_API_KEY`, and also without `CODERAG_MODEL`). Not run: no key in this environment.
- Tests: 136 gating + 16 eval passing, 1 live skipped (`reports/m7_gate.txt`).
- Decision: no server-side refusal fallback is configured because the model is chosen by env var; a `refusal` stop reason is surfaced as a structured failure.

### M8 — Citation verification — COMPLETE
- `citations.py`: `verify(citation, repo_root, max_lines=200)` -> `Verification(verified, reason)` with reasons `ok`, `invalid_path`, `file_not_found`, `invalid_range`, `reversed_range`, `lines_out_of_range`, `range_too_large`, `missing_symbol`, `symbol_not_in_range`, `unreadable_file`. The symbol (simple, qualified, or dotted suffix such as `Class.method`) must be defined in that file per tree-sitter, with its definition starting inside the range or the range lying inside its span. `verify_all` annotates citations.
- Path confinement is shared with `read_file` via `tools.confine_path`.
- `agent.py`: each final answer's citations get `verified` and `reason`; if any fail (or there are none) one repair round feeds the reasons back to the model (`max_repairs=1`); `unsupported=True` when no citation verifies. Unsupported answers are still returned (`ok=True`) but flagged.
- Tests: gating + eval passing (`reports/m8_gate.txt`).

### M9 — CLI, API, and eval harness — COMPLETE
- `cli.py` (typer): `coderag index <repo> [--db] [--chunker]`, `coderag ask <repo> "<q>" [--retriever hybrid|bm25] [--scripted script.json]` (prints answer plus each citation as verified / UNVERIFIED(reason), and an UNSUPPORTED banner), `coderag eval <repo> ...`. Default DB: `.coderag/<name>-<pathhash>.db` (gitignored). `ask` without `--scripted` needs `ANTHROPIC_API_KEY` and `CODERAG_MODEL`.
- `api.py` (FastAPI): `GET /health`, `POST /index`, `POST /ask`; pydantic-validated bodies (422 on bad JSON/fields), repo paths resolved and confined to `CODERAG_ALLOWED_ROOTS` (403 otherwise), 409 if not indexed, 503 without an LLM.
- `evaluation.py` + `make eval`: retrieval (bm25/vector/hybrid), agent metrics over 30 sampled questions (citation accuracy, grounded rate, unsupported/failure rate, latency p50/p95, tokens, estimated cost from a price table or `CODERAG_PRICE_IN/OUT`), and the AST-vs-fixed ablation (same retrievers, plus lines per retrieved chunk). Agent mode was **scripted** (no API key here); the report labels these as pipeline checks with synthetic token counts and cost n/a.
- `make eval` results (`reports/eval.json`, `reports/eval.md`, `reports/ablation_chunking.*`): scripted agent citation accuracy 1.0, grounded rate 0.70, failure 0.0 (pipeline checks only). Ablation overall recall@5, BM25: AST 0.8872 vs fixed 0.8232; hybrid: AST 0.8432 vs fixed 0.7823; mean lines per top-5 chunk 22.5 (AST) vs 39.4 (fixed).
- Claim added post hoc (after seeing the ablation): `tests/test_m9_claims.py` AST >= fixed + 0.02 on BM25 overall recall@5. `make claims`: passed (0.8872 vs 0.8232); the M5 hybrid claim still fails. Both in `reports/claims.json`.
- Tests: 172 gating + 16 eval passing (`reports/m9_gate.txt`).

### M10 — README, limitations, and final audit — COMPLETE
- `README.md`: overview, Mermaid architecture diagram, quickstart, evaluation method, results tables copied from `reports/eval.md` and `reports/ablation_chunking.md`, call graph and agent sections, Limitations. States plainly that hybrid did not beat BM25 (failed claim) and that the AST-vs-fixed claim was post hoc.
- `scripts/audit.py`: every README number outside code/URLs (decimals and percentages at written precision, integers >= 10) must appear in `reports/`; a sentence asserting an improvement for a failed claim fails the audit. Tests in `tests/test_m10_audit.py`, including the real README.
- Cleanup: removed unused `Store.chunks_for_path/symbols_for_path/all_symbols`, `evalgen.load_questions`, `Chunk.meta`; docstrings added to Protocol methods and API endpoints; ruff clean.
- `reports/fixture_stats.json` added so the README's fixture size is backed by a report (the audit caught it).
- Final: 182 gating + 16 eval passing, frozen check OK (46 files, 14 floors), audit OK (`reports/m10_gate.txt`).
- After the final `make freeze`: 47 frozen files, 14 floors (`reports/final_frozen_check.txt`).

### Bring-your-own-LLM — COMPLETE (offline-verified only; no provider verified live)
- **Adapters** (`src/coderag/providers/`):
  - `openai_compat.py`: Chat Completions; serves OpenAI, Groq, Together, OpenRouter, Ollama and any compatible server.
  - `gemini.py`: `generateContent` REST, with the key in the `x-goog-api-key` header.
  - `registry.py`: `build_client`, `list_providers`, `key_from_env`.
  - `base.py`: error hierarchy, bounded retries on 429/5xx only, history helpers.
  - All adapters translate the agent's Anthropic-shaped history both ways and normalize text, tool calls (raw string kept when JSON is malformed), usage and stop reason. `AnthropicClient` gained a `provider` attribute (additive).
- **SSRF guard** (`src/coderag/netguard.py`):
  - https only, no userinfo, public addresses only, including decoded numeric and IPv6-embedded IPv4 forms.
  - Re-validated at send time and pinned to the validated IP, keeping Host and TLS SNI via httpcore's `sni_hostname` extension.
  - No redirects, `trust_env=False`, response-size and total-time caps.
  - Optional `CODERAG_PROVIDER_HOSTS` allowlist; `CODERAG_ALLOW_INSECURE_LOCAL=1` for local Ollama only.
- **Key scrubbing** (`src/coderag/redact.py`): masks known key shapes always, and exact values registered per request with `secret_scope`. Installed process-wide by wrapping the log record factory; keeps uvicorn's positional access-log args intact and replaces tracebacks with scrubbed text.
- **API** (`api.py`, `apiguard.py`), backward compatible:
  - `X-LLM-Provider/-Model/-Key/-Base-URL` headers build a per-request client. Body credentials are ignored. `CODERAG_BYO_LLM_ONLY=1` returns 401 without headers.
  - Optional bearer token (`CODERAG_API_TOKEN`, constant-time compare), in-memory per-client rate limits, a 413 body cap (declared or streamed), a `max_steps` cap, repo file/byte caps, and an indexing concurrency cap (429 + Retry-After).
  - `CODERAG_REQUIRE_HTTPS` with an opt-in trusted proxy header. 422 responses no longer echo input values. `GET /providers` added.
- **CLI and eval:** `coderag ask --provider --model --base-url` (key only from env, never argv); `make eval` honors `CODERAG_PROVIDER`. Reports record `provider` and `model`; mode is `real` or `scripted`.
- **Docs:** README "Bring your own LLM" section (providers all marked [Unverified live]), Limitations updated, `.env.example` covering all 29 environment variables the code reads (`reports/byo_env_vars.txt`), `.env` gitignored.
- **Tests** (new files only: `tests/test_byo_*.py`): 328 gating + 16 eval passing, 7 live skipped (`reports/byo_gate.txt`), up from 182 gating (`reports/m10_gate.txt`). Frozen check OK and audit OK before the BYO freeze.
- **Decisions:**
  - `httpx` added to runtime dependencies because adapters and the SSRF guard use it directly. It was already installed; `anthropic` 1.x uses `httpx2`, not `httpx`.
  - Only `openai_compatible` accepts a caller-supplied base URL; preset hosts are fixed.
  - The Anthropic path keeps the official SDK with its fixed host, so it is not routed through the guard (the SDK uses `httpx2`).
  - The rate-limit identity is the hashed server token when auth is on, else the client IP (or X-Forwarded-For only with `CODERAG_TRUST_PROXY=1`).
  - The Gemini `thoughtSignature` round-trip is implemented from the API's documented shape [Unverified live].
- **Known gaps:**
  - No provider has been run live; no keys exist in this environment.
  - The rate limiter is per process (multiple workers each get their own budget).
  - Exact-value key scrubbing only applies to log records created inside the request scope; records logged elsewhere get pattern-based scrubbing only.
  - The API still retrieves with BM25 unless a `searcher_factory` is passed.
  - The frozen conftest live-skip quirk (BLOCKERS.md).
- After the BYO `make freeze`: 52 frozen files, 14 floors (`reports/byo_final_frozen_check.txt`).

### Web app (Streamlit) — COMPLETE (offline AppTest + headless launch + one manual browser run in demo mode; no real provider)
- `src/coderag/app_logic.py` (no Streamlit imports): repo validation (directory, `CODERAG_ALLOWED_ROOTS`, Python files present, size caps from `ApiSettings`), incremental indexing summary, example questions generated from real symbols in the index, cited-line snippets with real line numbers via `tools.confine_path` (capped), friendly (message, suggestion) mapping for every provider error class and agent failure, clean trace summaries, and `run_question` (secret scope, demo mode via the eval's scripted policy, hybrid-to-BM25 fallback warning). The client factory is a module-level hook so tests inject fakes.
- `src/coderag/streamlit_app.py`: a thin UI.
  - Sidebar: provider, model, password key field, base URL only for `openai_compatible`, Ollama note, demo toggle (on by default without a key), retriever (BM25 default), max steps.
  - Main: repo path with bundled-example button, Index, example-question buttons, question box, Ask, answer with unsupported banner, per-citation expanders with verified/unverified badges and code, a "How it got there" trace, session-only history with Clear, and "About safety".
  - No `st.cache_*` anywhere.
- `.streamlit/config.toml` (127.0.0.1 bind, no usage stats, no tracebacks in page), `coderag app` launcher (no secrets on argv), `make app`, optional extra `app = ["streamlit"]`, README "Web app" section.
- Tests (new files): `tests/test_app_logic.py`, `tests/test_streamlit_app.py` (AppTest, skips if Streamlit is absent), `tests/test_app_cli.py`. Gate: 386 gating + 16 eval passing, 7 live skipped (`reports/app_gate.txt`), up from 328 (`reports/byo_gate.txt`).
- Headless launch: health endpoint returned `ok` and the page was served on 127.0.0.1 (`reports/app_headless_launch.txt`). Manual browser run in demo mode with the bundled repo: index, examples, answer, verified badge and cited code all rendered (`reports/app_browser_check.txt`).
- Decisions:
  - `streamlit` added as an optional extra only, for the requested web UI; installed version recorded in `reports/app_versions.txt` (1.65.0).
  - The app calls the library in-process, not the HTTP API.
  - `evaluation._scripted_policy` was renamed to the public `scripted_policy` (only used internally) so demo mode reuses it.
  - `run_question` installs log scrubbing itself (idempotent): a test caught that `app_logic` used outside the Streamlit entry point could otherwise log a pattern-less key.
  - Index location reuses `CODERAG_DB_DIR` (no new environment variable).
- Known gaps:
  - Never run against a real model provider.
  - Only demo mode was exercised in a real browser.
  - Hybrid retrieval in the app embeds every chunk on first use, which is slow on a large repo.
  - The demo policy cites the top BM25 hit, which can be a caller rather than the asked-about function (seen in the browser run).
- After the web-app `make freeze`: see `reports/app_final_frozen_check.txt`.

### Deployment prep for Streamlit Community Cloud — COMPLETE (verified locally on a clean install; not yet deployed)
- **Security fixes:**
  - **Markdown injection / image exfiltration:** model answers, citation paths and symbols, trace fields, messages, questions and paths now pass through `app_logic.md_escape` before `st.markdown`, so a prompt-injected `![](https://attacker/...)` renders as inert text.
  - **Hosted mode** (`CODERAG_HOSTED=1`, set as a root-level secret, which Streamlit exposes as an env var): only the bundled example repo is allowed (`CODERAG_ALLOWED_ROOTS` default), `CODERAG_ALLOW_INSECURE_LOCAL` is dropped, the index goes to a temp dir, Ollama is hidden, the page starts in demo mode, and keys come only from what the visitor types, never from the server environment. A hosted notice explains this.
  - **Privacy:** `reports/claims_junit.xml` embedded the machine hostname. It was deleted, and `scripts/claims.py` now writes junit XML to a temp dir. Saved deploy reports had a scratch path containing the username, which was redacted. No real keys, personal paths or emails remain in committable files; the only key-shaped strings are fake values in tests.
- **Deployment bug found by a clean install:** with a normal (non-editable) install, `app_logic` located the bundled repo relative to `site-packages`. It now falls back to the working directory (Streamlit Cloud runs from the repo root).
- **Packaging:**
  - `fastembed` moved to an optional extra `embed` (hybrid retrieval, eval suite, `scripts/fetch_model.py`).
  - `requirements.txt` (for Streamlit Cloud) installs `.[app]` only, and the app hides the hybrid option when `fastembed` is absent.
  - `requirements.lock` was regenerated from the current venv (it was stale from M0).
  - `.gitignore` gains `.streamlit/secrets.toml`, `.DS_Store` and `*.db`.
  - Empty `logs/`, `__pycache__` and `.ruff_cache` were removed.
- **Docs:** README install lines use `.[embed,app]`, plus a "Deploying on Streamlit Community Cloud" subsection; `CODERAG_HOSTED` added to `.env.example`; BLOCKERS entry on the frozen `server.address` line.
- **Tests:** new `tests/test_hosted_deploy.py`. Gate: 406 gating + 16 eval passing, 7 live skipped (`reports/deploy_gate.txt`). A clean copy installed only from `requirements.txt` passed 406 unit tests without `fastembed` (`reports/deploy_clean_install_check.txt`), and its hosted-mode run indexed the bundled repo, answered in demo mode with a verified citation, rejected `/`, and passed the health check (`reports/deploy_hosted_mode_check.txt`).
- **Kept on purpose** (not needed to run the app, but part of the project's record): `CODERAG_OVERNIGHT.md` (the build spec), `PROGRESS.md`, `MORNING_REPORT.md`, `BLOCKERS.md`, `reports/` (the README audit depends on it), `.frozen/` (the gate depends on it).
- **Known gaps:**
  - Not yet deployed to Streamlit Cloud.
  - `server.address = "127.0.0.1"` may conflict with Cloud's health check [Unverified].
  - The eval tests need `data/` (gitignored) plus `.[embed]`, so a fresh clone runs unit tests only until bootstrapped.
