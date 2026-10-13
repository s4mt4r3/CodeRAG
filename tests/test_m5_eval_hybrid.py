"""M5 (eval): BM25 vs vector vs hybrid with the real local embedder.

Writes reports/retrieval_comparison.json and .md, and enforces hybrid floors.
"""

from __future__ import annotations

import json

import pytest

from coderag import benchmark
from coderag.embeddings import LocalEmbedder
from coderag.evalgen import generate_questions
from coderag.store import Store, index_repo
from conftest import EVAL_SRC, FIXTURE_ROOT, PROJECT_ROOT, REPORTS_DIR

pytestmark = pytest.mark.eval
TYPES = ("definition", "docstring_masked", "callers")


@pytest.fixture(scope="module")
def comparison():
    emb = LocalEmbedder()
    idx = PROJECT_ROOT / "data" / "index"
    idx.mkdir(parents=True, exist_ok=True)
    index_repo(EVAL_SRC, idx / "eval_ast.db")
    index_repo(FIXTURE_ROOT, idx / "fixture_ast.db")
    real = benchmark.compare_retrieval(
        Store(idx / "eval_ast.db"), generate_questions(EVAL_SRC, n=200, seed=0), emb)
    fx = benchmark.compare_retrieval(
        Store(idx / "fixture_ast.db"), generate_questions(FIXTURE_ROOT, n=200, seed=0), emb)
    report = {"real_repo": real, "fixture": fx, "seed": 0}
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "retrieval_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    md = "# Retrieval comparison (recall@k / MRR)\n\n"
    md += f"Embedder: `{real['config']['embedder']}`, RRF k={real['config']['rrf_k']}, "
    md += f"w_lex={real['config']['w_lex']}, w_vec={real['config']['w_vec']}.\n\n"
    md += benchmark.comparison_markdown("Real repo (pallets/click src, seed 0)", real) + "\n"
    md += benchmark.comparison_markdown("Fixture repo", fx)
    (REPORTS_DIR / "retrieval_comparison.md").write_text(md)
    return report


@pytest.mark.parametrize("qtype", TYPES)
def test_hybrid_meets_real_repo_floor(comparison, floors, qtype):
    name = f"hybrid_{qtype}_recall_at_5"
    assert name in floors, f"missing floor {name}"
    assert comparison["real_repo"]["hybrid"]["by_type"][qtype]["recall_at_5"] >= floors[name]


@pytest.mark.parametrize("qtype", TYPES)
def test_hybrid_meets_fixture_floor(comparison, floors, qtype):
    name = f"fixture_hybrid_{qtype}_recall_at_5"
    assert name in floors, f"missing floor {name}"
    assert comparison["fixture"]["hybrid"]["by_type"][qtype]["recall_at_5"] >= floors[name]
