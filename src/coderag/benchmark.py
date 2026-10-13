"""Glue for running retrieval benchmarks over an index (used by tests, CLI, reports)."""

from __future__ import annotations

import math
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from coderag.lexical import search_lexical
from coderag.metrics import Retrieved, evaluate_retrieval
from coderag.store import Hit, Store

SearchFn = Callable[[str, int], Sequence[Retrieved]]


def as_retrieved(hits: Sequence[Hit]) -> list[Retrieved]:
    """Convert hits to the ``(path, segments)`` form the metrics expect."""
    return [(h.path, h.segments) for h in hits]


def bm25_search(store: Store) -> SearchFn:
    """A metrics-compatible search function backed by BM25."""
    return lambda q, k: as_retrieved(search_lexical(store, q, k))


def run(questions: Sequence, search: SearchFn, ks: Sequence[int] = (1, 5, 10)) -> dict:
    """Evaluate ``search`` on ``questions``."""
    return evaluate_retrieval(questions, search, ks)


def floor_value(measured: float, margin: float = 0.03) -> float:
    """Measured minus margin, rounded down to 2 decimals (never below 0)."""
    return max(0.0, math.floor((measured - margin) * 100 + 1e-9) / 100)


def git_head(repo: str | Path) -> str:
    """HEAD SHA of a git checkout (read-only)."""
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def compare_retrieval(store: Store, questions: Sequence, embedder, ks=(1, 5, 10)) -> dict:
    """BM25 vs vector-only vs hybrid on the same questions and index."""
    from coderag.hybrid import HybridSearcher

    hs = HybridSearcher(store, embedder)
    systems = {
        "bm25": bm25_search(store),
        "vector": lambda q, k: as_retrieved(hs.search_vector(q, k)),
        "hybrid": lambda q, k: as_retrieved(hs.search(q, k)),
    }
    out = {}
    for name, fn in systems.items():
        res = run(questions, fn, ks)
        out[name] = {"overall": res["overall"], "by_type": res["by_type"]}
    out["config"] = {
        "embedder": embedder.name, "rrf_k": hs.rrf_k, "w_lex": hs.w_lex, "w_vec": hs.w_vec,
        "depth": hs.depth,
    }
    return out


def comparison_markdown(title: str, results: dict, metrics=("recall_at_1", "recall_at_5",
                                                             "recall_at_10", "mrr")) -> str:
    """Render a comparison dict (from ``compare_retrieval``) as a markdown table."""
    systems = [s for s in ("bm25", "vector", "hybrid") if s in results]
    types = sorted(results[systems[0]]["by_type"])
    lines = [f"## {title}", "", "| type | n | system | " + " | ".join(metrics) + " |",
             "|---|---|---|" + "---|" * len(metrics)]
    for t in [*types, "overall"]:
        for s in systems:
            row = results[s]["overall"] if t == "overall" else results[s]["by_type"][t]
            vals = " | ".join(f"{row[m]:.4f}" for m in metrics)
            lines.append(f"| {t} | {row['n']} | {s} | {vals} |")
    return "\n".join(lines) + "\n"
