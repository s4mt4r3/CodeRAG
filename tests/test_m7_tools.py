"""M7: read-only, path-confined agent tools."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from coderag.store import Store, index_repo
from coderag.tools import Toolbox, ToolError
from conftest import FIXTURE_ROOT


@pytest.fixture
def toolbox(tmp_path: Path) -> Toolbox:
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURE_ROOT, repo, ignore=shutil.ignore_patterns("__pycache__"))
    secret = tmp_path / "outside_secret.txt"
    secret.write_text("TOP SECRET\n")
    os.symlink(secret, repo / "leak.py")
    os.symlink(tmp_path, repo / "escape_dir")
    db = tmp_path / "i.db"
    index_repo(repo, db)
    return Toolbox(Store(db), max_read_lines=50)


def test_tool_set_is_read_only(toolbox: Toolbox):
    names = {s["name"] for s in toolbox.specs()}
    assert names == {"search_code", "search_symbols", "read_file", "find_callers"}
    for spec in toolbox.specs():
        assert spec["description"] and spec["input_schema"]["type"] == "object"


def test_read_file_happy_path(toolbox: Toolbox):
    out = toolbox.read_file("shopcore/utils.py", 9, 13)
    assert out["path"] == "shopcore/utils.py"
    assert (out["start_line"], out["end_line"]) == (9, 13)
    assert "def slugify" in out["text"]
    assert out["text"].splitlines()[0].startswith("9")
    assert out["truncated"] is False


@pytest.mark.parametrize("bad", [
    "/etc/passwd", "../outside_secret.txt", "shopcore/../../outside_secret.txt",
    "C:\\Windows\\win.ini", "~/x.py", "", "shopcore/../../../etc/passwd",
])
def test_read_file_rejects_absolute_and_traversal(toolbox: Toolbox, bad: str):
    with pytest.raises(ToolError):
        toolbox.read_file(bad, 1, 5)


def test_read_file_rejects_symlink_escaping_root(toolbox: Toolbox):
    with pytest.raises(ToolError):
        toolbox.read_file("leak.py", 1, 5)
    with pytest.raises(ToolError):
        toolbox.read_file("escape_dir/outside_secret.txt", 1, 5)


def test_read_file_caps_huge_range(toolbox: Toolbox):
    out = toolbox.read_file("shopcore/models.py", 1, 100_000)
    assert out["truncated"] is True
    assert out["end_line"] - out["start_line"] + 1 <= 50
    assert len(out["text"].splitlines()) <= 50


def test_read_file_rejects_bad_ranges_and_missing(toolbox: Toolbox):
    with pytest.raises(ToolError):
        toolbox.read_file("shopcore/models.py", 0, 5)
    with pytest.raises(ToolError):
        toolbox.read_file("shopcore/models.py", 10, 5)
    with pytest.raises(ToolError):
        toolbox.read_file("shopcore/models.py", 10_000, 10_005)
    with pytest.raises(ToolError):
        toolbox.read_file("shopcore/nope.py", 1, 2)
    with pytest.raises(ToolError):
        toolbox.read_file("shopcore", 1, 2)


def test_search_and_symbol_tools(toolbox: Toolbox):
    hits = toolbox.search_code("normalize sku", k=3)
    assert hits and hits[0]["symbol"] == "shopcore.utils.normalize_sku"
    assert {"path", "start_line", "end_line", "symbol", "kind", "score"} <= set(hits[0])
    syms = toolbox.search_symbols("validate")
    assert {s["qualified_name"] for s in syms} == {
        "storage.serializers.validate", "service.api_handlers.validate"}
    callers = toolbox.find_callers("compute_total")
    assert any(c["caller"] == "service.orders.OrderService.quote"
               and c["confidence"] == "resolved" for c in callers)


def test_dispatch_validates_arguments(toolbox: Toolbox):
    assert json.loads(toolbox.call("search_symbols", {"name": "Order"}))
    with pytest.raises(ToolError):
        toolbox.call("delete_file", {"path": "x"})
    with pytest.raises(ToolError):
        toolbox.call("read_file", "not a dict")
    with pytest.raises(ToolError):
        toolbox.call("read_file", {"start": 1})
    with pytest.raises(ToolError):
        toolbox.call("read_file", {"path": "shopcore/utils.py", "start": "one"})
    with pytest.raises(ToolError):
        toolbox.call("search_code", {"query": "x", "k": 10_000, "extra": 1})


def test_output_is_size_capped(toolbox: Toolbox):
    out = toolbox.call("search_code", {"query": "order", "k": 50})
    assert len(out) <= toolbox.max_output_chars + 200
