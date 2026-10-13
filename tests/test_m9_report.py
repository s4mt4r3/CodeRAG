"""M9: eval report structure, latency percentiles, and cost estimation."""

from __future__ import annotations

import json
import math

import pytest

from coderag.evaluation import render_markdown, run_eval
from coderag.metrics import estimate_cost, percentile
from conftest import FIXTURE_ROOT


def test_percentile_synthetic():
    vals = list(range(1, 101))
    assert percentile(vals, 50) == pytest.approx(50.5)
    assert percentile(vals, 95) == pytest.approx(95.05)
    assert percentile([7.0], 95) == 7.0
    assert percentile([3, 1, 2], 0) == 1 and percentile([3, 1, 2], 100) == 3
    assert percentile([], 50) is None


def test_cost_synthetic():
    assert estimate_cost(1_000_000, 500_000, 4.0, 20.0) == pytest.approx(14.0)
    assert estimate_cost(0, 0, 4.0, 20.0) == 0.0
    assert estimate_cost(1234, 567, None, 20.0) is None


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    return run_eval(FIXTURE_ROOT, db_dir=tmp_path_factory.mktemp("idx"), n_questions=40,
                    agent_n=10, embedder=None, seed=0)


def test_report_has_expected_keys(report):
    assert {"generated_at", "eval_repo", "retrieval", "agent", "ablation_chunking"} <= set(report)
    agent = report["agent"]
    for key in ["mode", "label", "model", "n_questions", "citation_accuracy",
                "answer_grounded_rate", "unsupported_rate", "failure_rate", "latency_ms",
                "tokens", "estimated_cost_usd"]:
        assert key in agent, key
    assert agent["mode"] == "scripted"
    assert "pipeline check" in agent["label"].lower()
    assert set(agent["latency_ms"]) == {"p50", "p95"}
    assert {"input", "output", "mean_per_question"} <= set(agent["tokens"])
    assert "bm25" in report["retrieval"]
    abl = report["ablation_chunking"]
    assert {"ast", "fixed"} <= set(abl["results"])
    json.dumps(report)  # serializable


def test_report_numbers_are_finite(report):
    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, float):
            assert math.isfinite(x)
    walk(report)


def test_markdown_renders_without_missing_values(report):
    md = render_markdown(report)
    assert "# coderag evaluation" in md.lower()
    assert "None" not in md and "nan" not in md.lower() and "{" not in md
    assert "pipeline check" in md.lower()
    assert "| ast |" in md and "| fixed |" in md
