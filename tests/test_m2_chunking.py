"""M2: AST-aware chunking and the fixed-size baseline."""

from __future__ import annotations

import ast
import random
import textwrap

import pytest

from coderag.chunking import ast_chunks, fixed_chunks
from coderag.ingest import SourceFile, walk_repo
from conftest import FIXTURE_ROOT
from helpers import oracle


def _parses_alone(code: str) -> bool:
    body = textwrap.dedent(code)
    try:
        ast.parse(body)
        return True
    except SyntaxError:
        pass
    try:  # indented code whose dedent is blocked by flush-left string content
        ast.parse("class _Wrap:\n" + code)
        return True
    except SyntaxError:
        return False


def _lines(chunk) -> set[int]:
    return {n for a, b in chunk.segments for n in range(a, b + 1)}


@pytest.fixture(scope="module")
def fixture_files() -> list[SourceFile]:
    return walk_repo(FIXTURE_ROOT)


def test_chunks_carry_metadata_and_header(fixture_files):
    for f in fixture_files:
        for c in ast_chunks(f):
            assert c.path == f.path
            assert c.kind in {"function", "method", "class", "module"}
            assert c.start_line <= c.end_line
            assert c.text.splitlines()[0] == f"# {c.path} :: {c.symbol}"
            assert c.text.endswith(c.code)
            assert c.chunk_id


def test_every_chunk_parses_alone_or_is_flagged(fixture_files):
    for f in fixture_files:
        for c in ast_chunks(f):
            if c.parts == 1:
                assert not c.partial, (c.path, c.symbol)
                assert _parses_alone(c.code), (c.path, c.symbol)
            elif not _parses_alone(c.code):
                assert c.partial, (c.path, c.symbol, c.part)


def test_full_coverage_of_non_blank_lines_without_overlap(fixture_files):
    for f in fixture_files:
        owner: dict[int, str] = {}
        for c in ast_chunks(f):
            for n in _lines(c):
                assert n not in owner, f"{f.path}:{n} in {owner.get(n)} and {c.chunk_id}"
                owner[n] = c.chunk_id
        for n, line in enumerate(f.text.splitlines(), start=1):
            if line.strip():
                assert n in owner, f"{f.path}:{n} not covered"


def test_chunk_code_matches_segment_lines(fixture_files):
    for f in fixture_files:
        src = f.text.splitlines(keepends=True)
        for c in ast_chunks(f):
            assert c.code == "".join("".join(src[a - 1 : b]) for a, b in c.segments)


def test_line_ranges_match_oracle(fixture_files):
    syms = {s.qualified_name: s for s in oracle.extract_repo(FIXTURE_ROOT)}
    unit_names = {
        q for q, s in syms.items() if s.parent is None or syms[s.parent].kind == "class"
    }
    seen = set()
    for f in fixture_files:
        for c in ast_chunks(f):
            if c.kind == "module":
                continue
            o = syms[c.symbol]
            seen.add(c.symbol)
            assert c.kind == o.kind
            if c.kind == "class":
                assert c.start_line == o.start_line
                assert _lines(c) <= set(range(o.start_line, o.end_line + 1))
            else:
                assert (c.start_line, c.end_line) == (o.start_line, o.end_line), c.symbol
    assert seen == unit_names


def test_nested_functions_stay_inside_parent_chunk(fixture_files):
    f = next(x for x in fixture_files if x.path == "storage/cache.py")
    chunks = {c.symbol: c for c in ast_chunks(f)}
    assert "storage.cache.memoize.decorator" not in chunks
    assert "def wrapper(arg):" in chunks["storage.cache.memoize"].code


def test_class_chunk_excludes_methods(fixture_files):
    f = next(x for x in fixture_files if x.path == "shopcore/models.py")
    chunks = {c.symbol: c for c in ast_chunks(f)}
    cls = chunks["shopcore.models.Customer"]
    assert "loyalty_points: int = 0" in cls.code
    assert "def display_name" not in cls.code
    assert '"""A registered customer who can place orders."""' in cls.code


def test_no_split_under_budget(fixture_files):
    for f in fixture_files:
        assert all(c.parts == 1 for c in ast_chunks(f, max_tokens=10_000))


def _big_function() -> str:
    body = "".join(
        f"    if x > {i}:\n        y = compute_value(x, {i}, 'label_{i}')\n"
        f"        total += y\n\n" for i in range(60)
    )
    return (
        "import math\n\n\n"
        "@decorate(level=3)\n"
        "def big(x):\n"
        '    """A long function that must be split."""\n'
        "    total = 0\n" + body + "    return total\n"
    )


def test_large_function_splits_on_statement_boundaries_losslessly():
    src = _big_function()
    f = SourceFile("big.py", src)
    chunks = [c for c in ast_chunks(f, max_tokens=120) if c.symbol == "big.big"]
    assert len(chunks) > 1
    assert [c.part for c in chunks] == list(range(len(chunks)))
    assert all(c.parts == len(chunks) for c in chunks)
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.end_line + 1 == b.start_line
    lines = src.splitlines(keepends=True)
    assert "".join(c.code for c in chunks) == "".join(lines[3:])
    stmt_starts = {n.lineno for n in ast.walk(ast.parse(src)) if isinstance(n, ast.stmt)}
    for c in chunks[1:]:
        assert c.start_line in stmt_starts, "piece must start at a statement boundary"


def test_module_chunk_holds_top_level_code(fixture_files):
    f = next(x for x in fixture_files if x.path == "shopcore/pricing.py")
    mods = [c for c in ast_chunks(f) if c.kind == "module"]
    assert len(mods) == 1
    assert "TAX_RATES = " in mods[0].code
    assert "def " not in mods[0].code
    assert mods[0].symbol == "shopcore.pricing"


def test_empty_and_blank_files():
    assert ast_chunks(SourceFile("e.py", "")) == []
    assert ast_chunks(SourceFile("b.py", "\n\n   \n")) == []


# ---------------------------------------------------------------- property test

_NAMES = ["alpha", "beta", "gamma", "delta", "eps", "zeta", "theta"]


def _rand_block(rng: random.Random, depth: int, indent: str) -> list[str]:
    out = []
    for _ in range(rng.randint(1, 6)):
        r = rng.random()
        name = rng.choice(_NAMES)
        if r < 0.15 and depth < 3:
            out.append(f"{indent}if {name} > {rng.randint(0, 9)}:")
            out += _rand_block(rng, depth + 1, indent + "    ")
            if rng.random() < 0.5:
                out.append(f"{indent}else:")
                out += _rand_block(rng, depth + 1, indent + "    ")
        elif r < 0.25 and depth < 3:
            out.append(f"{indent}for i in range({rng.randint(1, 5)}):")
            out += _rand_block(rng, depth + 1, indent + "    ")
        elif r < 0.32 and depth < 2:
            out.append(f"{indent}def inner_{rng.randint(0, 99)}(v):")
            out += _rand_block(rng, depth + 1, indent + "    ")
            out.append(f"{indent}    return v")
        elif r < 0.40:
            out.append(f"{indent}# note about {name}")
        elif r < 0.45:
            out.append("")
        elif r < 0.55:
            out.append(f"{indent}{name} = call_it(")
            out.append(f"{indent}    {rng.randint(0, 99)},")
            out.append(f"{indent})")
        else:
            out.append(f"{indent}{name} = {name} + {rng.randint(0, 99)}")
    out.append(f"{indent}pass")
    return out


def _rand_function(rng: random.Random, idx: int) -> str:
    lines = []
    if rng.random() < 0.3:
        lines.append("@some.decorator(1)")
    prefix = "async def" if rng.random() < 0.2 else "def"
    lines.append(f"{prefix} fn_{idx}(alpha, beta=1):")
    if rng.random() < 0.6:
        lines.append(f'    """Docstring for fn_{idx}."""')
    lines += _rand_block(rng, 1, "    ")
    lines.append("    return alpha")
    return "\n".join(lines) + "\n"


def test_property_chunk_then_reassemble_equals_original():
    rng = random.Random(1234)
    for idx in range(50):
        fn = _rand_function(rng, idx)
        src = f"X = {idx}\n\n\n" + fn
        ast.parse(src)
        f = SourceFile(f"rand_{idx}.py", src)
        budget = rng.choice([20, 40, 80, 10_000])
        pieces = [c for c in ast_chunks(f, max_tokens=budget) if c.symbol == f"rand_{idx}.fn_{idx}"]
        assert pieces, idx
        assert "".join(c.code for c in pieces) == fn, (idx, budget)
        for a, b in zip(pieces, pieces[1:], strict=False):
            assert a.end_line + 1 == b.start_line


# ---------------------------------------------------------------- fixed baseline


def test_fixed_chunks_windows_and_overlap():
    src = "".join(f"line_{i} = {i}\n" for i in range(1, 26))
    f = SourceFile("w.py", src)
    chunks = fixed_chunks(f, size=10, overlap=3)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 10), (8, 17), (15, 24), (22, 25)]
    assert all(c.kind == "fixed" for c in chunks)
    lines = src.splitlines(keepends=True)
    for c in chunks:
        assert c.code == "".join(lines[c.start_line - 1 : c.end_line])
        assert c.text.splitlines()[0].startswith("# w.py")


def test_fixed_chunks_validates_arguments():
    f = SourceFile("w.py", "x = 1\n")
    with pytest.raises(ValueError):
        fixed_chunks(f, size=5, overlap=5)
    assert len(fixed_chunks(f, size=5, overlap=0)) == 1
    assert fixed_chunks(SourceFile("e.py", ""), size=5, overlap=1) == []
