"""M2 (eval): chunking invariants hold on the real eval repo."""

from __future__ import annotations

from collections import defaultdict

import pytest

from coderag.chunking import ast_chunks
from coderag.ingest import walk_repo
from conftest import EVAL_SRC

pytestmark = pytest.mark.eval


@pytest.mark.parametrize("budget", [150, 400])
def test_eval_repo_coverage_no_overlap_and_lossless_splits(budget):
    for f in walk_repo(EVAL_SRC):
        owner: dict[int, str] = {}
        by_unit = defaultdict(list)
        for c in ast_chunks(f, max_tokens=budget):
            for a, b in c.segments:
                for n in range(a, b + 1):
                    assert n not in owner, (f.path, n)
                    owner[n] = c.chunk_id
            if c.kind in ("function", "method"):
                by_unit[c.symbol].append(c)
        for n, line in enumerate(f.text.split("\n"), start=1):
            if line.strip():
                assert n in owner, (f.path, n)
        lines = f.text.split("\n")
        for pieces in by_unit.values():
            if len(pieces) > 1 and all(p.parts == len(pieces) for p in pieces):
                start, end = pieces[0].start_line, pieces[-1].end_line
                original = "\n".join(lines[start - 1 : end]) + "\n"
                assert "".join(p.code for p in pieces) == original
