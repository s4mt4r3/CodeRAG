"""M4: retrieval metrics against hand-computed examples."""

from __future__ import annotations

import pytest

from coderag.metrics import hit, mrr, overlap_fraction, recall_at_k

A = ("a.py", 10, 19)  # 10 lines
B = ("b.py", 1, 4)


def r(path, *segs):
    return (path, list(segs))


def test_overlap_fraction():
    assert overlap_fraction([(10, 19)], (10, 19)) == 1.0
    assert overlap_fraction([(15, 30)], (10, 19)) == 0.5
    assert overlap_fraction([(10, 12), (17, 18)], (10, 19)) == 0.5
    assert overlap_fraction([(20, 30)], (10, 19)) == 0.0


def test_hit_requires_same_path_and_half_overlap():
    assert hit(r("a.py", (15, 40)), A)
    assert not hit(r("a.py", (16, 40)), A)  # 4/10 lines
    assert not hit(r("x.py", (10, 19)), A)


def test_recall_at_k_hand_computed():
    retrieved = [r("z.py", (1, 5)), r("a.py", (10, 19)), r("b.py", (1, 4))]
    assert recall_at_k(retrieved, [A, B], 1) == 0.0
    assert recall_at_k(retrieved, [A, B], 2) == 0.5
    assert recall_at_k(retrieved, [A, B], 3) == 1.0


def test_k_larger_than_results():
    assert recall_at_k([r("a.py", (10, 19))], [A, B], 50) == 0.5
    assert recall_at_k([], [A], 5) == 0.0


def test_no_truth_is_undefined():
    assert recall_at_k([r("a.py", (1, 2))], [], 5) is None
    assert mrr([r("a.py", (1, 2))], []) is None


def test_mrr_hand_computed():
    retrieved = [r("z.py", (1, 5)), r("q.py", (1, 5)), r("b.py", (1, 4))]
    assert mrr(retrieved, [A, B]) == pytest.approx(1 / 3)
    assert mrr([r("a.py", (10, 19))], [A]) == 1.0
    assert mrr([r("z.py", (1, 5))], [A]) == 0.0


def test_ties_and_duplicates_count_once():
    # Two retrieved chunks hitting the same truth span do not double count.
    retrieved = [r("a.py", (10, 13)), r("a.py", (14, 19)), r("a.py", (10, 19))]
    assert recall_at_k(retrieved, [A], 3) == 1.0
    assert recall_at_k(retrieved, [A, B], 3) == 0.5
    assert mrr(retrieved, [A]) == pytest.approx(1 / 2)
