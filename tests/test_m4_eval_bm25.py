"""M4 (eval): BM25 baseline on the real repo; writes reports/baseline_bm25.json."""

from __future__ import annotations

import json

import pytest

from coderag import benchmark
from coderag.evalgen import generate_questions, save_questions
from coderag.store import Store, index_repo
from conftest import EVAL_REPO, EVAL_SRC, FIXTURE_ROOT, PROJECT_ROOT, REPORTS_DIR

pytestmark = pytest.mark.eval
TYPES = ("definition", "docstring_masked", "callers")


def _summary(res: dict) -> dict:
    return {"overall": res["overall"], "by_type": res["by_type"]}


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    sha = (PROJECT_ROOT / "data" / "EVAL_REPO_SHA.txt").read_text().strip()
    assert benchmark.git_head(EVAL_REPO) == sha, "eval repo is not at the recorded SHA"
    db = PROJECT_ROOT / "data" / "index" / "eval_ast.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    index_repo(EVAL_SRC, db)
    qs = generate_questions(EVAL_SRC, n=200, seed=0)
    REPORTS_DIR.mkdir(exist_ok=True)
    save_questions(qs, REPORTS_DIR / "questions_eval.json")
    real = benchmark.run(qs, benchmark.bm25_search(Store(db)))

    fx_db = tmp_path_factory.mktemp("fx") / "fx.db"
    index_repo(FIXTURE_ROOT, fx_db)
    fx = benchmark.run(generate_questions(FIXTURE_ROOT, n=200, seed=0),
                       benchmark.bm25_search(Store(fx_db)))
    report = {
        "system": "bm25",
        "eval_repo": {"name": "pallets/click", "sha": sha, "indexed_root": "src"},
        "seed": 0,
        "n_questions": len(qs),
        "real_repo": _summary(real),
        "fixture": _summary(fx),
    }
    (REPORTS_DIR / "baseline_bm25.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


@pytest.mark.parametrize("qtype", TYPES)
def test_real_repo_bm25_meets_floor(results, floors, qtype):
    name = f"bm25_{qtype}_recall_at_5"
    assert name in floors, f"missing floor {name}"
    assert results["real_repo"]["by_type"][qtype]["recall_at_5"] >= floors[name]


def test_question_count(results):
    assert 150 <= results["n_questions"] <= 210
