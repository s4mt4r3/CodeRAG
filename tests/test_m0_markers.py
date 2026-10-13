"""M0: pytest markers are registered and live tests skip without a key."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import PROJECT_ROOT


@pytest.mark.live
def test_live_marker_requires_key():
    assert os.environ.get("ANTHROPIC_API_KEY"), "live tests must only run with a key"


def test_live_test_is_skipped_without_key(tmp_path: Path):
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs",
         "tests/test_m0_markers.py::test_live_marker_requires_key"],
        cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 skipped" in proc.stdout


def test_unknown_marker_is_rejected(tmp_path: Path):
    test_file = tmp_path / "test_bad_marker.py"
    test_file.write_text("import pytest\n\n@pytest.mark.nosuchmarker\ndef test_x():\n    pass\n")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--strict-markers", "-c",
         str(PROJECT_ROOT / "pyproject.toml"), "--rootdir", str(tmp_path), str(test_file)],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode != 0
