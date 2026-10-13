"""M1 (eval): tree-sitter symbols match the ast oracle on the real eval repo."""

from __future__ import annotations

import pytest

from coderag.symbols import extract_repo_symbols
from conftest import EVAL_SRC
from helpers import oracle

pytestmark = pytest.mark.eval


def test_eval_repo_symbols_match_oracle():
    key = lambda s: (s.qualified_name, s.kind, s.path, s.start_line, s.end_line)  # noqa: E731
    got = sorted(map(key, extract_repo_symbols(EVAL_SRC)))
    exp = sorted(map(key, oracle.extract_repo(EVAL_SRC)))
    assert len(exp) >= 300
    assert got == exp


def test_eval_repo_docstrings_match_oracle():
    got = {(s.path, s.start_line): s.docstring for s in extract_repo_symbols(EVAL_SRC)}
    exp = {(s.path, s.start_line): s.docstring for s in oracle.extract_repo(EVAL_SRC)}
    assert got == exp
