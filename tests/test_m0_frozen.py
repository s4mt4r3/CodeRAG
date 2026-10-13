"""M0: scripts/frozen.py freeze/check semantics, exercised on a temp project."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import PROJECT_ROOT


def _run(root: Path, cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(root / "scripts" / "frozen.py"), "--root", str(root), cmd],
        capture_output=True, text=True, timeout=60,
    )


@pytest.fixture
def proj(tmp_path: Path) -> Path:
    (tmp_path / "scripts").mkdir()
    shutil.copy(PROJECT_ROOT / "scripts" / "frozen.py", tmp_path / "scripts" / "frozen.py")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("def test_a():\n    assert True\n")
    (tmp_path / "thresholds.toml").write_text("[floors]\nrecall_at_5 = 0.80\n")
    return tmp_path


def _set_floors(root: Path, body: str) -> None:
    (root / "thresholds.toml").write_text("[floors]\n" + body)


def test_check_before_first_freeze_passes_with_warning(proj: Path):
    r = _run(proj, "check")
    assert r.returncode == 0
    assert "warning" in r.stdout.lower()


def test_freeze_records_files_and_floors(proj: Path):
    assert _run(proj, "freeze").returncode == 0
    m = json.loads((proj / ".frozen" / "manifest.json").read_text())
    assert "tests/test_a.py" in m["files"]
    assert "scripts/frozen.py" in m["files"]
    assert m["floors"]["recall_at_5"]["value"] == 0.80
    assert _run(proj, "check").returncode == 0


def test_modified_file_fails_check(proj: Path):
    _run(proj, "freeze")
    (proj / "tests" / "test_a.py").write_text("def test_a():\n    pass\n")
    r = _run(proj, "check")
    assert r.returncode != 0
    assert "changed" in r.stderr


def test_deleted_file_fails_check(proj: Path):
    _run(proj, "freeze")
    (proj / "tests" / "test_a.py").unlink()
    r = _run(proj, "check")
    assert r.returncode != 0
    assert "missing" in r.stderr


def test_lowered_floor_fails_check(proj: Path):
    _run(proj, "freeze")
    _set_floors(proj, "recall_at_5 = 0.79\n")
    r = _run(proj, "check")
    assert r.returncode != 0
    assert "lowered" in r.stderr


def test_removed_floor_fails_check(proj: Path):
    _run(proj, "freeze")
    _set_floors(proj, "")
    r = _run(proj, "check")
    assert r.returncode != 0
    assert "removed" in r.stderr


def test_new_test_file_passes(proj: Path):
    _run(proj, "freeze")
    (proj / "tests" / "test_b.py").write_text("def test_b():\n    pass\n")
    assert _run(proj, "check").returncode == 0
    assert _run(proj, "freeze").returncode == 0
    m = json.loads((proj / ".frozen" / "manifest.json").read_text())
    assert "tests/test_b.py" in m["files"]


def test_new_floor_passes(proj: Path):
    _run(proj, "freeze")
    _set_floors(proj, "recall_at_5 = 0.80\nmrr = 0.5\n")
    assert _run(proj, "check").returncode == 0
    assert _run(proj, "freeze").returncode == 0
    m = json.loads((proj / ".frozen" / "manifest.json").read_text())
    assert m["floors"]["mrr"]["value"] == 0.5


def test_freeze_refuses_to_overwrite_changed_entry(proj: Path):
    _run(proj, "freeze")
    before = (proj / ".frozen" / "manifest.json").read_text()
    (proj / "tests" / "test_a.py").write_text("# tampered\n")
    (proj / "tests" / "test_c.py").write_text("def test_c():\n    pass\n")
    r = _run(proj, "freeze")
    assert r.returncode != 0
    assert (proj / ".frozen" / "manifest.json").read_text() == before


def test_missing_manifest_after_freeze_is_error(proj: Path):
    _run(proj, "freeze")
    (proj / ".frozen" / "manifest.json").unlink()
    assert _run(proj, "check").returncode != 0


def test_editing_frozen_script_fails_check(proj: Path):
    _run(proj, "freeze")
    script = proj / "scripts" / "frozen.py"
    script.write_text(script.read_text() + "\n# edit\n")
    assert _run(proj, "check").returncode != 0
