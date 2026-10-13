"""M6 (eval): call-graph precision/recall vs the ast oracle on the real repo."""

from __future__ import annotations

import json

import pytest

from coderag.callgraph import build_edges
from coderag.ingest import walk_repo
from conftest import EVAL_SRC, REPORTS_DIR
from helpers import oracle

pytestmark = pytest.mark.eval


def test_eval_repo_callgraph_report():
    edges = build_edges(walk_repo(EVAL_SRC))
    got = {(e.caller, e.callee) for e in edges if e.confidence == "resolved"}
    exp = oracle.resolved_edges(EVAL_SRC)
    tp = len(got & exp)
    counts: dict[str, int] = {}
    for e in edges:
        counts[e.confidence] = counts.get(e.confidence, 0) + 1
    report = {
        "repo": "pallets/click (src)",
        "resolved_edges_system": len(got),
        "resolved_edges_oracle": len(exp),
        "true_positives": tp,
        "precision": tp / len(got) if got else None,
        "recall": tp / len(exp) if exp else None,
        "edge_rows_by_confidence": counts,
        "false_positives_sample": sorted(got - exp)[:20],
        "false_negatives_sample": sorted(exp - got)[:20],
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "callgraph.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["precision"] is not None and report["recall"] is not None
