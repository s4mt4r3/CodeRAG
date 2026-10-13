"""M9 claim (non-gating, post hoc): AST chunking beats fixed-size chunking under BM25.

Written after the first ablation run was observed; the 0.02 margin mirrors the M5 claim.
"""

from __future__ import annotations

import json

import pytest

from coderag import benchmark
from coderag.evalgen import generate_questions
from coderag.store import Store, index_repo
from conftest import EVAL_SRC, PROJECT_ROOT, REPORTS_DIR

pytestmark = pytest.mark.claim


def test_ast_chunking_beats_fixed_bm25_recall_at_5():
    idx = PROJECT_ROOT / "data" / "index"
    idx.mkdir(parents=True, exist_ok=True)
    index_repo(EVAL_SRC, idx / "eval_ast.db")
    index_repo(EVAL_SRC, idx / "eval_fixed.db", chunker="fixed")
    qs = generate_questions(EVAL_SRC, n=200, seed=0)
    ast = benchmark.run(qs, benchmark.bm25_search(Store(idx / "eval_ast.db")))
    fixed = benchmark.run(qs, benchmark.bm25_search(Store(idx / "eval_fixed.db")))
    a, f = ast["overall"]["recall_at_5"], fixed["overall"]["recall_at_5"]
    details_path = REPORTS_DIR / "claims_details.json"
    details = json.loads(details_path.read_text()) if details_path.exists() else {}
    details["ast_beats_fixed_bm25_overall"] = {
        "required_margin": 0.02, "ast_recall_at_5": a, "fixed_recall_at_5": f,
        "n": len(qs), "passed": a >= f + 0.02, "post_hoc": True,
    }
    details_path.write_text(json.dumps(details, indent=2) + "\n")
    assert a >= f + 0.02, f"ast {a:.4f} vs fixed {f:.4f}"
