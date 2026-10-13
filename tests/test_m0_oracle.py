"""M0: the hand-written expected.json and the ast oracle agree on the fixture."""

from __future__ import annotations

import ast
import json

from conftest import EXPECTED_JSON, FIXTURE_ROOT
from helpers import oracle


def _expected() -> dict:
    return json.loads(EXPECTED_JSON.read_text(encoding="utf-8"))


def test_fixture_parses_with_ast():
    files = oracle.iter_python_files(FIXTURE_ROOT)
    assert len(files) >= 12
    for rel in files:
        ast.parse((FIXTURE_ROOT / rel).read_text(encoding="utf-8"), filename=rel)


def test_symbols_agree_with_expected():
    exp = {s["qualified_name"]: s for s in _expected()["symbols"]}
    got = {s.qualified_name: s for s in oracle.extract_repo(FIXTURE_ROOT)}
    assert set(got) == set(exp)
    for q, e in exp.items():
        g = got[q]
        assert (g.kind, g.path, g.start_line, g.end_line) == (
            e["kind"],
            e["path"],
            e["start_line"],
            e["end_line"],
        ), q


def test_calls_agree_with_expected():
    exp = {s["qualified_name"]: s for s in _expected()["symbols"] if s["kind"] != "class"}
    got = {s.qualified_name: s for s in oracle.extract_repo(FIXTURE_ROOT)}
    for q, e in exp.items():
        assert got[q].calls == sorted(e["calls"]), q


def test_resolved_edges_agree_with_expected():
    exp = {tuple(e) for e in _expected()["resolved_edges"]}
    assert oracle.resolved_edges(FIXTURE_ROOT) == exp


def test_fixture_shape_requirements():
    syms = oracle.extract_repo(FIXTURE_ROOT)
    funcs = [s for s in syms if s.kind != "class"]
    classes_with_methods = {s.parent for s in syms if s.kind == "method"}
    assert len(funcs) >= 30
    assert len(classes_with_methods) >= 6
    documented = sum(1 for s in funcs if s.docstring) / len(funcs)
    assert 0.6 <= documented <= 0.8
    assert any(s.is_async for s in funcs)
    assert any(s.decorators for s in funcs)
    by_q = {s.qualified_name: s for s in syms}
    assert any(s.parent and by_q[s.parent].kind != "class" for s in funcs)
    assert len(oracle.resolved_edges(FIXTURE_ROOT)) >= 15


def test_one_method_called_from_two_files():
    files_by_name: dict[str, set[str]] = {}
    for s in oracle.extract_repo(FIXTURE_ROOT):
        for name in s.calls:
            files_by_name.setdefault(name, set()).add(s.path)
    methods = {s.name for s in oracle.extract_repo(FIXTURE_ROOT) if s.kind == "method"}
    assert any(len(files_by_name.get(m, ())) >= 2 for m in methods)


def test_module_name_convention():
    assert oracle.module_name("a/b/c.py") == "a.b.c"
    assert oracle.module_name("a/b/__init__.py") == "a.b"
    assert oracle.module_name("top.py") == "top"
