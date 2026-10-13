"""Retrieval metrics with span-overlap matching.

A retrieved chunk *hits* a truth span when it is in the same file and covers at
least half of the truth span's lines. Chunks are given as ``(path, segments)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

Span = tuple[str, int, int]
Retrieved = tuple[str, Sequence[tuple[int, int]]]

MIN_OVERLAP = 0.5


def overlap_fraction(segments: Sequence[tuple[int, int]], truth: tuple[int, int] | Span) -> float:
    """Fraction of the truth span's lines covered by ``segments``."""
    start, end = truth[-2], truth[-1]
    size = end - start + 1
    if size <= 0:
        return 0.0
    covered = 0
    for a, b in segments:
        lo, hi = max(a, start), min(b, end)
        if lo <= hi:
            covered += hi - lo + 1
    return min(covered, size) / size


def hit(retrieved: Retrieved, truth: Span, min_overlap: float = MIN_OVERLAP) -> bool:
    """Whether one retrieved chunk hits one truth span."""
    path, segments = retrieved
    return path == truth[0] and overlap_fraction(segments, truth) >= min_overlap


def recall_at_k(retrieved: Sequence[Retrieved], truths: Sequence[Span], k: int) -> float | None:
    """Fraction of truth spans hit by any of the top ``k`` results; None without truth."""
    if not truths:
        return None
    top = list(retrieved)[:k]
    found = sum(1 for t in truths if any(hit(r, t) for r in top))
    return found / len(truths)


def mrr(retrieved: Sequence[Retrieved], truths: Sequence[Span]) -> float | None:
    """Reciprocal rank of the first result hitting any truth span; None without truth."""
    if not truths:
        return None
    for rank, r in enumerate(retrieved, start=1):
        if any(hit(r, t) for t in truths):
            return 1.0 / rank
    return 0.0


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def evaluate_retrieval(
    questions: Sequence,
    search: Callable[[str, int], Sequence[Retrieved]],
    ks: Sequence[int] = (1, 5, 10),
) -> dict:
    """Per-type and overall recall@k and MRR for ``search(question_text, k)``."""
    depth = max(ks)
    per_q: list[dict] = []
    for q in questions:
        retrieved = list(search(q.question, depth))
        truths = [tuple(t) for t in q.truth]
        row = {"id": q.id, "type": q.type, "mrr": mrr(retrieved, truths)}
        for k in ks:
            row[f"recall_at_{k}"] = recall_at_k(retrieved, truths, k)
        per_q.append(row)
    metric_names = [f"recall_at_{k}" for k in ks] + ["mrr"]

    def summarize(rows: list[dict]) -> dict:
        out: dict = {"n": len(rows)}
        for m in metric_names:
            out[m] = _mean([r[m] for r in rows])
        return out

    types = sorted({r["type"] for r in per_q})
    return {
        "overall": summarize(per_q),
        "by_type": {t: summarize([r for r in per_q if r["type"] == t]) for t in types},
        "per_question": per_q,
    }


def percentile(values: Sequence[float], p: float) -> float | None:
    """Linear-interpolated percentile (numpy's default method); None for no values."""
    if not values:
        return None
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def estimate_cost(input_tokens: int, output_tokens: int, price_in_per_mtok: float | None,
                  price_out_per_mtok: float | None) -> float | None:
    """USD cost from token counts and per-million-token prices; None if a price is unknown."""
    if price_in_per_mtok is None or price_out_per_mtok is None:
        return None
    return (input_tokens * price_in_per_mtok + output_tokens * price_out_per_mtok) / 1_000_000
