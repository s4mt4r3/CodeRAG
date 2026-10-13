PY := .venv/bin/python
export HF_HUB_OFFLINE := 1
export TRANSFORMERS_OFFLINE := 1

.PHONY: test eval-tests gate freeze claims eval app

test:
	$(PY) -m ruff check .
	$(PY) -m pytest -m "not live and not eval and not claim" -q

# pytest exits 5 when no tests are collected; that counts as success until eval tests exist.
eval-tests:
	$(PY) -m pytest -m eval -q; rc=$$?; if [ $$rc -eq 5 ]; then echo "no eval tests collected"; exit 0; fi; exit $$rc

gate:
	$(MAKE) test && $(MAKE) eval-tests && $(PY) scripts/frozen.py check

freeze:
	$(PY) scripts/frozen.py freeze

claims:
	$(PY) scripts/claims.py

eval:
	$(PY) -m coderag.cli eval data/eval_repo/src --out reports --db-dir data/index \
		--db-prefix eval --repo-name pallets/click --sha $$(cat data/EVAL_REPO_SHA.txt)

# Local web app (needs: pip install -e '.[app]')
app:
	.venv/bin/streamlit run src/coderag/streamlit_app.py
