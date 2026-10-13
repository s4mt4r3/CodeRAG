"""M5 claim (non-gating): hybrid improves docstring_masked recall@5 over BM25 by >= 0.02."""

from __future__ import annotations

import json

import pytest

from coderag import benchmark
from coderag.embeddings import LocalEmbedder
from coderag.evalgen import generate_questions
from coderag.store import Store, index_repo
from conftest import EVAL_SRC, PROJECT_ROOT, REPORTS_DIR

pytestmark = pytest.mark.claim


def test_hybrid_beats_bm25_on_docstring_masked():
    db = PROJECT_ROOT / "data" / "index" / "eval_ast.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    index_repo(EVAL_SRC, db)
    qs = [q for q in generate_questions(EVAL_SRC, n=200, seed=0) if q.type == "docstring_masked"]
    res = benchmark.compare_retrieval(Store(db), qs, LocalEmbedder())
    bm25 = res["bm25"]["by_type"]["docstring_masked"]["recall_at_5"]
    hybrid = res["hybrid"]["by_type"]["docstring_masked"]["recall_at_5"]
    details_path = REPORTS_DIR / "claims_details.json"
    details = json.loads(details_path.read_text()) if details_path.exists() else {}
    details["hybrid_beats_bm25_docstring_masked"] = {
        "required_margin": 0.02, "bm25_recall_at_5": bm25, "hybrid_recall_at_5": hybrid,
        "n": len(qs), "passed": hybrid >= bm25 + 0.02,
    }
    details_path.write_text(json.dumps(details, indent=2) + "\n")
    assert hybrid >= bm25 + 0.02, f"hybrid {hybrid:.4f} vs bm25 {bm25:.4f}"
