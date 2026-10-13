"""M4: oracle-backed question generation."""

from __future__ import annotations

from coderag.evalgen import generate_questions, mask_docstring_sentence
from coderag.lexical import split_identifier
from conftest import FIXTURE_ROOT
from helpers import oracle


def test_deterministic_given_seed():
    a = generate_questions(FIXTURE_ROOT, n=40, seed=7)
    b = generate_questions(FIXTURE_ROOT, n=40, seed=7)
    c = generate_questions(FIXTURE_ROOT, n=40, seed=8)
    assert [q.to_dict() for q in a] == [q.to_dict() for q in b]
    assert [q.to_dict() for q in a] != [q.to_dict() for q in c]


def test_all_types_present_with_unique_ids():
    qs = generate_questions(FIXTURE_ROOT, n=60, seed=0)
    assert {q.type for q in qs} == {"definition", "docstring_masked", "callers"}
    assert len({q.id for q in qs}) == len(qs)
    assert all(q.truth for q in qs)


def test_truth_spans_match_the_oracle():
    syms = oracle.extract_repo(FIXTURE_ROOT)
    by_q = {s.qualified_name: s for s in syms}
    callers = oracle.name_callers(FIXTURE_ROOT)
    for q in generate_questions(FIXTURE_ROOT, n=200, seed=0):
        target = by_q[q.symbol]
        if q.type == "callers":
            exp = {(by_q[c].path, by_q[c].start_line, by_q[c].end_line)
                   for c in callers[target.name]}
            assert set(map(tuple, q.truth)) == exp, q.id
        elif q.type == "docstring_masked" or target.kind != "class":
            assert [tuple(t) for t in q.truth] == [
                (target.path, target.start_line, target.end_line)
            ], q.id
        else:  # class definition: header region (up to its first direct method)
            (path, start, end), = q.truth
            assert path == target.path and start == target.start_line
            assert start <= end <= target.end_line
            children = [s for s in syms if s.parent == target.qualified_name]
            if children:
                assert end < min(c.start_line for c in children)


def test_masked_questions_never_contain_name_tokens():
    for q in generate_questions(FIXTURE_ROOT, n=200, seed=0):
        if q.type != "docstring_masked":
            continue
        name_parts = set(split_identifier(q.symbol.rsplit(".", 1)[-1]))
        words = q.question.replace("`", " ").split()
        for w in words:
            parts = set(split_identifier("".join(ch for ch in w if ch.isalnum() or ch == "_")))
            assert not (parts & name_parts), (q.id, q.question)


def test_mask_examples():
    assert mask_docstring_sentence(
        "Normalize the SKU string before storage.", "normalize_sku"
    ) == "the string before storage."
    assert mask_docstring_sentence("Look up the sales tax rate.", "tax_rate") == (
        "Look up the sales."
    )


def test_question_templates():
    qs = generate_questions(FIXTURE_ROOT, n=200, seed=0)
    d = next(q for q in qs if q.type == "definition")
    assert d.question.startswith("Where is `") and d.question.endswith("` defined?")
    c = next(q for q in qs if q.type == "callers")
    assert c.question.startswith("Which functions call `")
