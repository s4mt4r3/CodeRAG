"""M4: BM25 meets its frozen floors on the fixture (gating, deterministic, offline)."""

from __future__ import annotations

import pytest

from coderag import benchmark
from coderag.evalgen import generate_questions
from coderag.store import Store, index_repo
from conftest import FIXTURE_ROOT

TYPES = ("definition", "docstring_masked", "callers")


@pytest.fixture(scope="module")
def fixture_result(tmp_path_factory):
    db = tmp_path_factory.mktemp("bm25") / "fx.db"
    index_repo(FIXTURE_ROOT, db)
    store = Store(db)
    qs = generate_questions(FIXTURE_ROOT, n=200, seed=0)
    return benchmark.run(qs, benchmark.bm25_search(store))


@pytest.mark.parametrize("qtype", TYPES)
def test_fixture_bm25_meets_floor(fixture_result, floors, qtype):
    name = f"fixture_bm25_{qtype}_recall_at_5"
    assert name in floors, f"missing floor {name}"
    assert fixture_result["by_type"][qtype]["recall_at_5"] >= floors[name]


def test_floor_value_rounding():
    assert benchmark.floor_value(0.8749) == 0.84
    assert benchmark.floor_value(0.90) == 0.87
    assert benchmark.floor_value(0.01) == 0.0
