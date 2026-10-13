# BLOCKERS

Problems that could not be fixed without breaking a rule (frozen tests, floors, claims), with the real numbers.

## M5 claim failed: hybrid does not beat BM25 on `docstring_masked`

- Claim (`tests/test_m5_claims.py`, unchanged): hybrid recall@5 >= BM25 recall@5 + 0.02 on click `docstring_masked` test questions (seed 0, n=80).
- Result (`reports/claims.json`): BM25 0.9375, hybrid 0.9000. **Failed.** Hybrid is lower, not just short of the margin.
- Tuned on a dev split (seed 1, disjoint from test ids; `reports/m5_dev_tuning.txt`, scripts in `reports/m5_tuning_scripts/`):
  1. RRF vector weight grid 0.2..1.0: dev `docstring_masked` hybrid 0.9438-0.9551 vs BM25 0.9663.
  2. Different embedding text (signature + docstring "doc view") and symmetric vs query-instruction query embedding: no improvement (vector-only 0.7978 doc view vs 0.8315 full chunk; query modes identical).
- Likely cause: masked docstring sentences still share many exact words with the docstring inside the chunk, so BM25 is near ceiling (0.94-0.97), and bge-small vector-only recall on long code chunks is much lower (0.775 test), so fusion mostly injects noise.
- Shipped config: w_lex=1.0, w_vec=0.5 (chosen on dev). The README does not claim hybrid improves retrieval.

## BYO-LLM: frozen conftest skips every live test without ANTHROPIC_API_KEY

- `tests/conftest.py` (frozen in M0) marks every `@pytest.mark.live` test as skipped unless `ANTHROPIC_API_KEY` is set, regardless of which provider the test targets.
- Workaround (no frozen file touched): each BYO live test in `tests/test_byo_live.py` also checks its own key and model variables. To run a non-Anthropic live test, set `ANTHROPIC_API_KEY` to any non-empty value as well. Documented in README Limitations and `.env.example`.

## Streamlit Cloud: frozen test pins `server.address = "127.0.0.1"` in `.streamlit/config.toml`

- `tests/test_streamlit_app.py::test_streamlit_config_is_safe` (frozen) requires that line, which is the right default for local use.
- On Streamlit Community Cloud the config file must sit at the repo root and is shared by the deployment. Their docs say nothing about `server.address`. A forum report shows a different `[server]` option (the port) breaking the Cloud health check. Whether a 127.0.0.1 bind works there is [Unverified].
- Workaround: none possible without editing a frozen test. If the deployed app fails its health check, the human owner can remove the line (the local launchers `make app` and `coderag app` already pass `--server.address 127.0.0.1` themselves), accepting that this frozen test then fails, and record the decision here.
