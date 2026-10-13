"""M10: scripts/audit.py rejects fabricated numbers and claims of failed claim tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from conftest import PROJECT_ROOT

_spec = importlib.util.spec_from_file_location("audit", PROJECT_ROOT / "scripts" / "audit.py")
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


@pytest.fixture
def reports(tmp_path: Path) -> Path:
    d = tmp_path / "reports"
    d.mkdir()
    (d / "eval.json").write_text(json.dumps({
        "retrieval": {"bm25": {"recall_at_5": 0.8871666666666667, "n": 200}},
        "agent": {"n_questions": 30, "rate": 0.7},
        "config": {"rrf_k": 60},
    }))
    (d / "claims.json").write_text(json.dumps({"details": {
        "hybrid_beats_bm25_docstring_masked": {"passed": False},
        "ast_beats_fixed_bm25_overall": {"passed": True},
    }}))
    (d / "notes.md").write_text("| x | 0.1234 |\n")
    return d


def run(tmp_path: Path, reports: Path, text: str) -> list[str]:
    readme = tmp_path / "README.md"
    readme.write_text(text)
    return audit.audit(readme, reports)


def test_numbers_from_reports_pass(tmp_path, reports):
    text = ("BM25 recall@5 is 0.8872 over 200 questions (30 agent questions, RRF k=60). "
            "Grounded rate 0.70, i.e. 70%. Another: 0.1234.\n")
    assert run(tmp_path, reports, text) == []


def test_fabricated_number_fails(tmp_path, reports):
    problems = run(tmp_path, reports, "Hybrid recall@5 is 0.9512 on 200 questions.\n")
    assert problems and any("0.9512" in p for p in problems)


def test_fabricated_percentage_fails(tmp_path, reports):
    assert run(tmp_path, reports, "Recall improved by 37%.\n")


def test_numbers_in_code_are_ignored(tmp_path, reports):
    text = "Run:\n\n```bash\npip install x==12.34\nmake eval --n 999\n```\n\nUse `--k 4242`.\n"
    assert run(tmp_path, reports, text) == []


def test_small_integers_and_identifiers_are_ignored(tmp_path, reports):
    assert run(tmp_path, reports, "Milestones M0 to M10 in 3 steps with bge-small-en-v1.5.\n") == []


def test_claiming_a_failed_claim_fails(tmp_path, reports):
    problems = run(tmp_path, reports, "Hybrid retrieval outperforms BM25 on masked questions.\n")
    assert problems and any("hybrid" in p.lower() for p in problems)


def test_negated_mention_of_failed_claim_passes(tmp_path, reports):
    text = "Hybrid did not beat BM25 on masked docstring questions; it was worse.\n"
    assert run(tmp_path, reports, text) == []


def test_passed_claim_may_be_stated(tmp_path, reports):
    assert run(tmp_path, reports, "AST chunking beats fixed-size chunking under BM25.\n") == []


def test_cli_exit_codes(tmp_path, reports):
    good = tmp_path / "good.md"
    good.write_text("Recall 0.8872.\n")
    bad = tmp_path / "bad.md"
    bad.write_text("Recall 0.9999.\n")
    assert audit.main(["--readme", str(good), "--reports", str(reports)]) == 0
    assert audit.main(["--readme", str(bad), "--reports", str(reports)]) == 1


def test_project_readme_passes_audit():
    readme = PROJECT_ROOT / "README.md"
    assert readme.exists()
    assert audit.audit(readme, PROJECT_ROOT / "reports") == []
