"""M6: name-based call graph from tree-sitter, scored against the ast oracle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from coderag.callgraph import build_edges, find_callees, find_callers
from coderag.ingest import walk_repo
from coderag.store import Store, index_repo
from conftest import EXPECTED_JSON, FIXTURE_ROOT
from helpers import oracle


def _resolved(edges) -> set[tuple[str, str]]:
    return {(e.caller, e.callee) for e in edges if e.confidence == "resolved"}


def _pr(got: set, exp: set) -> tuple[float, float]:
    tp = len(got & exp)
    precision = tp / len(got) if got else 1.0
    recall = tp / len(exp) if exp else 1.0
    return precision, recall


@pytest.fixture(scope="module")
def fixture_edges():
    return build_edges(walk_repo(FIXTURE_ROOT))


def test_fixture_resolved_edge_precision_and_recall(fixture_edges, floors):
    precision, recall = _pr(_resolved(fixture_edges), oracle.resolved_edges(FIXTURE_ROOT))
    assert "callgraph_fixture_precision" in floors and "callgraph_fixture_recall" in floors
    assert precision >= floors["callgraph_fixture_precision"]
    assert recall >= floors["callgraph_fixture_recall"]


def test_fixture_matches_hand_written_edges(fixture_edges):
    exp = {tuple(e) for e in json.loads(EXPECTED_JSON.read_text())["resolved_edges"]}
    assert _resolved(fixture_edges) == exp


def test_alias_cases_resolve(fixture_edges):
    got = _resolved(fixture_edges)
    # import storage.serializers as ser
    assert ("storage.repository.OrderRepository.save",
            "storage.serializers.order_to_dict") in got
    # from .utils import clamp as clamp_value
    assert ("shopcore.pricing.apply_discount", "shopcore.utils.clamp") in got
    # from shopcore import pricing  (module imported from a package)
    assert ("service.orders.OrderService.quote", "shopcore.pricing.compute_total") in got


def test_parameter_named_like_a_package_is_not_resolved(fixture_edges):
    # handle_create(service, ...) calls service.create_order(); `service` is a parameter,
    # not the `service` package, so this must not resolve to a module attribute.
    rows = [e for e in fixture_edges if e.caller == "service.api_handlers.handle_create"
            and e.callee_name == "create_order"]
    assert rows and all(e.confidence != "resolved" for e in rows)


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def test_same_name_in_two_modules_is_ambiguous(tmp_path: Path):
    _write(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/a.py": "def helper(x):\n    return x\n",
        "pkg/b.py": "def helper(x):\n    return -x\n",
        "pkg/c.py": (
            "def run(obj):\n    return obj.helper(1)\n\n\n"
            "def bare():\n    return helper(2)\n"
        ),
    })
    edges = build_edges(walk_repo(tmp_path))
    for caller in ("pkg.c.run", "pkg.c.bare"):
        rows = [e for e in edges if e.caller == caller and e.callee_name == "helper"]
        assert rows, caller
        assert {e.confidence for e in rows} == {"ambiguous"}, caller
        assert {e.callee for e in rows} == {"pkg.a.helper", "pkg.b.helper"}


def test_import_alias_variants(tmp_path: Path):
    _write(tmp_path, {
        "lib/__init__.py": "from .core import engine as public_engine\n",
        "lib/core.py": "def engine():\n    return 1\n\n\ndef other():\n    return 2\n",
        "app.py": (
            "import lib.core as lc\n"
            "import lib.core\n"
            "from lib.core import other as o\n"
            "from lib import public_engine\n"
            "import json\n\n\n"
            "def main():\n"
            "    lc.engine()\n"
            "    lib.core.other()\n"
            "    o()\n"
            "    public_engine()\n"
            "    json.dumps({})\n"
        ),
    })
    edges = build_edges(walk_repo(tmp_path))
    got = _resolved(edges)
    assert got == oracle.resolved_edges(tmp_path)
    assert got == {
        ("app.main", "lib.core.engine"),
        ("app.main", "lib.core.other"),
    }
    calls = {e.expr: e for e in edges if e.caller == "app.main"}
    assert calls["json.dumps"].confidence == "unresolved"
    assert calls["public_engine"].callee == "lib.core.engine"
    assert calls["o"].callee == "lib.core.other"


def test_self_method_and_nested_function(tmp_path: Path):
    _write(tmp_path, {
        "m.py": (
            "class K:\n"
            "    def a(self):\n"
            "        return self.b()\n\n"
            "    def b(self):\n"
            "        def inner():\n"
            "            return 3\n"
            "        return inner()\n"
        ),
    })
    got = _resolved(build_edges(walk_repo(tmp_path)))
    assert got == {("m.K.a", "m.K.b"), ("m.K.b", "m.K.b.inner")}


def test_store_integration_find_callers_and_callees(tmp_path: Path):
    db = tmp_path / "i.db"
    index_repo(FIXTURE_ROOT, db)
    store = Store(db)
    callers = find_callers(store, "compute_total")
    resolved = {e.caller for e in callers if e.confidence == "resolved"}
    assert resolved == {"service.orders.OrderService.quote"}
    callees = find_callees(store, "service.orders.OrderService.checkout")
    names = {e.callee_name for e in callees}
    assert {"quote", "mark_paid", "add_points", "render_receipt"} <= names
    quote = [e for e in callees if e.callee_name == "quote"]
    assert quote[0].confidence == "resolved"
    assert quote[0].callee == "service.orders.OrderService.quote"
    by_name = find_callers(store, "validate")
    assert {e.confidence for e in by_name} >= {"resolved"}
    assert all(e.path and e.line > 0 for e in callers)


def test_edges_refresh_on_reindex(tmp_path: Path):
    import shutil

    repo = tmp_path / "r"
    shutil.copytree(FIXTURE_ROOT, repo, ignore=shutil.ignore_patterns("__pycache__"))
    db = tmp_path / "i.db"
    index_repo(repo, db)
    (repo / "service" / "orders.py").unlink()
    index_repo(repo, db)
    store = Store(db)
    assert not [e for e in find_callers(store, "compute_total") if e.confidence == "resolved"]
