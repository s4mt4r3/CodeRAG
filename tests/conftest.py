"""Shared pytest configuration: offline env, live-test skipping, common paths."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Tests must never reach the network; the embedding model is loaded from data/models/.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "fixture_repo"
EXPECTED_JSON = PROJECT_ROOT / "tests" / "fixtures" / "expected.json"
EVAL_REPO = PROJECT_ROOT / "data" / "eval_repo"
EVAL_SRC = EVAL_REPO / "src"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
THRESHOLDS = PROJECT_ROOT / "thresholds.toml"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    skip_live = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture
def fixture_root() -> Path:
    return FIXTURE_ROOT


@pytest.fixture
def floors() -> dict[str, float]:
    import tomllib

    data = tomllib.loads(THRESHOLDS.read_text(encoding="utf-8"))
    return {k: float(v) for k, v in data.get("floors", {}).items()}
