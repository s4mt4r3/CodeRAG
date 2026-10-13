# Morning report: coderag overnight run

**Status:** all milestones M0-M10 are COMPLETE. No STOPPED.md was written. The final gate is green: 182 gating tests and 16 eval tests pass, the frozen check is OK (46 files, 14 floors, before the final freeze), and `scripts/audit.py` is OK (`reports/m10_gate.txt`). After the final `make freeze`, the manifest holds 47 files and 14 floors (`reports/final_frozen_check.txt`). No git commands that change a repository were run; the only git use was `ls-remote`, `clone`, and `rev-parse`.

## Milestones

| | What | Notes |
|---|---|---|
| M0 | env, click eval repo (8.5.0), offline embedding model, fixture repo + hand-written `expected.json`, `ast` oracle, freeze guard | The oracle cross-check caught one of my own hand-transcription errors in `expected.json` |
| M1 | ingest + tree-sitter symbols | Exact match with the `ast` oracle on the fixture and on click (`src/`) |
| M2 | AST chunker + fixed baseline | No coverage gaps, overlaps, or lossy splits on click (eval test) |
| M3 | SQLite + FTS5 BM25, incremental indexing | |
| M4 | question generator, metrics, BM25 floors | |
| M5 | embeddings, RRF hybrid | **Claim failed:** hybrid did not beat BM25 (see below) |
| M6 | call graph | Fixed a tree-sitter Node identity bug; agrees with the oracle on click |
| M7 | tools, LLM client, agent loop | Live test not run (no API key) |
| M8 | citation verification + one repair round | |
| M9 | CLI, FastAPI, `make eval`, chunking ablation | Agent metrics are scripted pipeline checks |
| M10 | README, audit script, cleanup | |

## Floors set (`thresholds.toml`, frozen)

- **BM25, click (recall@5):** definition 0.88, docstring_masked 0.90, callers 0.70.
- **BM25, fixture:** 0.95 / 0.97 / 0.97.
- **Hybrid, click (recall@5):** definition 0.89, docstring_masked 0.87, callers 0.53.
- **Hybrid, fixture:** 0.97 each.
- **Call graph, fixture:** precision 0.95 and recall 0.95 (values from the spec).

## Key metrics (from `reports/`)

- **Overall recall@5 on click** (200 questions, `reports/eval.json`): BM25 0.8872, vector 0.7712, hybrid 0.8432.
- **Chunking ablation, BM25 overall recall@5:** AST 0.8872 vs fixed 0.8232.
- **Chunking ablation, hybrid overall recall@5:** AST 0.8432 vs fixed 0.7823.
- **Call graph vs the `ast` oracle on click:** precision 1.0, recall 1.0 (402 edges). This shows agreement on the same static rules, not runtime accuracy.
- **Agent:** scripted pipeline checks only. Citation accuracy 1.0, grounded rate 0.70, failure rate 0.0. Token counts are synthetic and cost is n/a.

## Claims (`reports/claims.json`)

- **FAILED: hybrid beats BM25 by at least 0.02 recall@5 on `docstring_masked`.**
  - Result: hybrid 0.9000 vs BM25 0.9375.
  - I tried two approaches on a disjoint dev split: an RRF weight grid, and doc-view embeddings / query embedding modes. Neither closed the gap (`reports/m5_dev_tuning.txt`).
  - The claim test is unchanged, and the README says hybrid did not win.
- **PASSED: AST chunking beats fixed-size by at least 0.02 BM25 overall recall@5** (0.8872 vs 0.8232).
  - Caveat: I wrote this claim after seeing the ablation, so it is post hoc and labeled that way.

## BLOCKERS.md

There is one entry: the failed M5 hybrid claim, with numbers and both tuning attempts. Nothing else blocked.

## Decisions to review

- **Indexed scope:** only click's `src/` is indexed. Its tests have many duplicate test names, which would make `definition` questions ambiguous.
- **Nested functions:** functions nested inside functions stay in their parent's chunk.
- **Class chunks:** a class chunk is its header, docstring and fields, with the methods carved out.
- **Class `definition` truth:** the truth span for a class is its header region.
- **Hit rule:** a retrieved chunk hits when it covers at least half of the truth span's lines. This favors big chunks, which is why the ablation also reports lines per chunk.
- **Hybrid weights:** shipped with w_vec=0.5, chosen on the dev split.
- **Edge confidence:** call edges are `resolved` only by scope rules. Any name match outside those rules is `ambiguous`, with one row per candidate.

## Five places to read first

1. **`src/coderag/tools.py` (`confine_path`, `Toolbox.read_file`).** This is the security boundary. It resolves symlinks and then reads, which leaves a check-then-use race if someone swaps a symlink in between. That's acceptable for a local read-only tool but not for a hostile multi-user server.
2. **`src/coderag/agent.py` (`Agent.run`, `wrap_tool_output`).**
   - Failure classification and the repair flow.
   - Injection defense: delimiter wrapping plus a system-prompt rule. It is a mitigation, not a guarantee.
   - "unsupported" answers are returned with `ok=True` but flagged.
3. **`src/coderag/chunking.py` (`_FileChunker`).** The hole/segment bookkeeping and statement-boundary packing are the most intricate code. The invariants are checked by property tests and a click-wide eval test.
4. **`src/coderag/callgraph.py` (`_imports`, `_Resolver.resolve`).**
   - Re-export following and relative imports.
   - The Node identity bug fixed in M6 lived here.
   - Precision and recall of 1.0 only prove the two implementations agree.
5. **`src/coderag/hybrid.py` + `embeddings.embed_chunks`.** If the embedder returns the wrong number of vectors, `embed_chunks` returns an empty index and hybrid silently becomes BM25. The CLI prints a note only when the model fails to load.

Also worth checking: in `evalgen.py`, how `docstring_masked` masks words and how `callers` truth is built (name-based and noisy).

## Not done / unverified

- **Live API:** the live smoke test and a real-model agent eval were not run (no `ANTHROPIC_API_KEY`). The agent numbers are pipeline checks only.
- **Embedding time:** [Unverified] the first embedding of click took about 3 minutes on this Mac. That was seen in the console and not saved to a report.
- **Sandbox:** the run happened directly on this Mac inside `~/Desktop/CodeRAG`, not in a sandbox as the spec's Section 0 suggests. It used pip installs, one model download and one `git clone`.
- **Git:** `CodeRAG/` sits inside the Desktop-level git repo and shows as untracked there. Nothing was added or committed.

## Addendum: bring-your-own-LLM (same day, later session)

- **What was built:**
  - Provider adapters for OpenAI-compatible servers (OpenAI, Groq, Together, OpenRouter, Ollama, custom) and Gemini, plus the existing Anthropic client.
  - Per-request credentials on `/ask` via `X-LLM-*` headers.
  - An SSRF guard with IP pinning, and process-wide key scrubbing in logs and errors.
  - Optional API token, rate, size and concurrency limits, and an HTTPS requirement.
  - CLI `--provider` / `--model` / `--base-url` (keys only from env).
  - `.env.example` and a README section.
- **Verified:** offline only. 328 gating + 16 eval tests pass, 7 live tests skipped (`reports/byo_gate.txt`). **No provider has been run live.** All are marked [Unverified live] in the README.
- **Read first:**
  1. `netguard.py` (`validate_url`, `GuardedTransport`).
  2. `redact.py` (record factory, `secret_scope`).
  3. `api.py` (`ask`, `resolve_llm`, where the secret scope begins and ends).
  4. `providers/gemini.py` (synthesized ids, `thoughtSignature`, schema stripping).
  5. `apiguard.py` (body buffering, per-process rate limiter, proxy trust).
- **New blocker note:** the frozen conftest skips all live tests unless `ANTHROPIC_API_KEY` is set (see BLOCKERS.md).

## Addendum: Streamlit web app (later session)

- Launch: `pip install -e '.[app]'`, then `make app` (or `coderag app`), and open http://127.0.0.1:8501.
- Verified:
  - 386 gating + 16 eval tests pass (`reports/app_gate.txt`).
  - Headless launch health check passes (`reports/app_headless_launch.txt`).
  - One manual browser run in demo mode works end to end (`reports/app_browser_check.txt`).
- Not verified: any real model provider through the app; the error panels in a real browser (covered by AppTest only).
- Read first:
  1. `app_logic.run_question` (secret scope, error mapping).
  2. `streamlit_app.py` (where the key is read; confirm nothing is cached).
  3. `app_logic.read_snippet` (path confinement for displayed code).
  4. `.streamlit/config.toml` (bind address, tracebacks off).
  5. `cli.app_cmd` (launcher argv).

## Addendum: Streamlit Cloud deployment prep (later session)

- **Fixed:**
  - Markdown image exfiltration in the app.
  - Server keys being usable by visitors and unrestricted repo paths (new `CODERAG_HOSTED=1` mode).
  - Hostname leak in `reports/claims_junit.xml`.
  - The bundled-repo path breaking under non-editable installs.
- **Packaging:** `fastembed` is now the optional `.[embed]` extra; `requirements.txt` installs `.[app]`.
- **Verified:** a clean install from `requirements.txt` passes 406 unit tests and runs in hosted mode (`reports/deploy_*`).
- **Not verified:** an actual Streamlit Cloud deploy. The `server.address` line is the main risk (see BLOCKERS.md).
