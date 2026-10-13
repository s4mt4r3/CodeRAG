"""M1: tree-sitter symbol extraction matches the ast oracle."""

from __future__ import annotations

import time

from coderag.symbols import extract_repo_symbols, parse_source
from conftest import FIXTURE_ROOT
from helpers import oracle


def _key(s):
    return (s.qualified_name, s.kind, s.path, s.start_line, s.end_line)


def test_fixture_symbols_match_oracle_exactly():
    got = sorted(_key(s) for s in extract_repo_symbols(FIXTURE_ROOT))
    exp = sorted(_key(s) for s in oracle.extract_repo(FIXTURE_ROOT))
    assert got == exp


def test_fixture_symbol_attributes_match_oracle():
    got = {s.qualified_name: s for s in extract_repo_symbols(FIXTURE_ROOT)}
    for o in oracle.extract_repo(FIXTURE_ROOT):
        g = got[o.qualified_name]
        assert g.name == o.name
        assert g.docstring == o.docstring, o.qualified_name
        assert g.is_async == o.is_async, o.qualified_name
        assert g.parent == o.parent, o.qualified_name
        assert [d.replace(" ", "") for d in g.decorators] == [
            d.replace(" ", "") for d in o.decorators
        ], o.qualified_name


def test_empty_file():
    res = parse_source("empty.py", "")
    assert res.symbols == []
    assert res.warnings == []


def test_syntax_error_does_not_crash_and_recovers():
    src = (
        "def ok():\n"
        "    return 1\n"
        "\n"
        "\n"
        "def broken(:\n"
        "    pass\n"
        "\n"
        "\n"
        "class Fine:\n"
        "    def m(self):\n"
        "        return 2\n"
    )
    res = parse_source("bad.py", src)
    names = {s.qualified_name for s in res.symbols}
    assert "bad.ok" in names
    assert res.warnings, "a parse warning must be reported"
    assert any("bad.py" in w for w in res.warnings)


def test_unicode_identifiers_and_text():
    src = (
        '"""Módulo con acentos — and an em dash."""\n'
        "\n"
        "\n"
        "def größe(x):\n"
        '    """Größe berechnen."""\n'
        "    return x\n"
        "\n"
        "\n"
        "class Ñandú:\n"
        "    def café(self):\n"
        "        return '☕'\n"
    )
    res = parse_source("uni.py", src)
    got = {(s.qualified_name, s.kind, s.start_line, s.end_line) for s in res.symbols}
    assert got == {
        ("uni.größe", "function", 4, 6),
        ("uni.Ñandú", "class", 9, 11),
        ("uni.Ñandú.café", "method", 10, 11),
    }
    doc = {s.qualified_name: s.docstring for s in res.symbols}
    assert doc["uni.größe"] == "Größe berechnen."


def test_very_long_file(tmp_path):
    parts = []
    for i in range(3000):
        parts.append(f"def f{i}(a, b):\n    \"\"\"Doc {i}.\"\"\"\n    return a + b + {i}\n\n")
    src = "".join(parts)
    (tmp_path / "big.py").write_text(src)
    t0 = time.perf_counter()
    res = parse_source("big.py", src)
    elapsed = time.perf_counter() - t0
    assert len(res.symbols) == 3000
    exp = sorted(_key(s) for s in oracle.extract_repo(tmp_path))
    assert sorted(_key(s) for s in res.symbols) == exp
    assert elapsed < 10


def test_defs_inside_compound_statements_and_async():
    src = (
        "import sys\n"
        "\n"
        "if sys.platform == 'win32':\n"
        "    def plat():\n"
        "        return 'w'\n"
        "else:\n"
        "    def plat():\n"
        "        return 'p'\n"
        "\n"
        "try:\n"
        "    import fast\n"
        "except ImportError:\n"
        "    def fallback():\n"
        "        pass\n"
        "\n"
        "\n"
        "class K:\n"
        "    @staticmethod\n"
        "    async def run():\n"
        "        x = 1  # trailing comment\n"
        "        # comment at end of block\n"
        "\n"
        "    class Inner:\n"
        "        pass\n"
    )
    res = parse_source("m.py", src)
    got = sorted((s.qualified_name, s.kind, s.start_line, s.end_line, s.is_async)
                 for s in res.symbols)
    assert got == [
        ("m.K", "class", 17, 24, False),
        ("m.K.Inner", "class", 23, 24, False),
        ("m.K.run", "method", 18, 20, True),
        ("m.fallback", "function", 13, 14, False),
        ("m.plat", "function", 4, 5, False),
        ("m.plat", "function", 7, 8, False),
    ]


def test_eval_style_module_name():
    res = parse_source("pkg/sub/__init__.py", "def f():\n    pass\n")
    assert [s.qualified_name for s in res.symbols] == ["pkg.sub.f"]
