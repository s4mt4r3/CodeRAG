"""M1: repository walking and filtering."""

from __future__ import annotations

from pathlib import Path

from coderag.ingest import walk_repo
from conftest import FIXTURE_ROOT


def _make(root: Path, rel: str, data: bytes | str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        p.write_text(data, encoding="utf-8")
    else:
        p.write_bytes(data)


def test_walk_fixture_returns_relative_posix_paths():
    files = walk_repo(FIXTURE_ROOT)
    paths = [f.path for f in files]
    assert paths == sorted(paths)
    assert "shopcore/models.py" in paths
    assert all(not p.startswith("/") for p in paths)
    assert len(paths) == 13
    models = next(f for f in files if f.path == "shopcore/models.py")
    assert "class Customer" in models.text


def test_skips_ignored_dirs_oversized_non_utf8_and_non_python(tmp_path: Path):
    _make(tmp_path, "pkg/a.py", "x = 1\n")
    for d in [".git", "venv", ".venv", "node_modules", "__pycache__", "pkg/__pycache__"]:
        _make(tmp_path, f"{d}/hidden.py", "y = 2\n")
    _make(tmp_path, "pkg/huge.py", "#" * (1_000_001))
    _make(tmp_path, "pkg/latin1.py", "s = 'caf\xe9'\n".encode("latin-1"))
    _make(tmp_path, "README.md", "# hi\n")
    _make(tmp_path, "pkg/data.json", "{}")
    paths = [f.path for f in walk_repo(tmp_path)]
    assert paths == ["pkg/a.py"]


def test_respects_ignore_list(tmp_path: Path):
    _make(tmp_path, "pkg/a.py", "x = 1\n")
    _make(tmp_path, "pkg/gen/b.py", "x = 2\n")
    _make(tmp_path, "docs/conf.py", "x = 3\n")
    _make(tmp_path, "pkg/test_c.py", "x = 4\n")
    paths = [f.path for f in walk_repo(tmp_path, ignore=["docs", "pkg/gen/*", "test_*.py"])]
    assert paths == ["pkg/a.py"]


def test_coderagignore_file(tmp_path: Path):
    _make(tmp_path, "keep.py", "x = 1\n")
    _make(tmp_path, "drop/x.py", "x = 2\n")
    _make(tmp_path, ".coderagignore", "# comment\ndrop\n")
    assert [f.path for f in walk_repo(tmp_path)] == ["keep.py"]


def test_size_limit_is_configurable(tmp_path: Path):
    _make(tmp_path, "a.py", "x = 1\n" * 100)
    assert walk_repo(tmp_path, max_bytes=10) == []
