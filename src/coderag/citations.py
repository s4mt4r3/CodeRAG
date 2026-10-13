"""Citation schema checks against the repository on disk.

A citation ``{"path", "start_line", "end_line", "symbol"}`` is verified when the
path is a real file inside the repo, the range is valid and not absurdly large,
and the cited symbol (simple, qualified, or dotted suffix like ``Class.method``)
is defined there: its definition starts inside the range, or the range lies
inside the symbol's span (citing part of a body). Symbols come from tree-sitter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from coderag.symbols import Symbol, parse_source, split_lines
from coderag.tools import confine_path

DEFAULT_MAX_LINES = 200


@dataclass(frozen=True)
class Verification:
    """Outcome of verifying one citation; ``reason`` is ``"ok"`` when verified."""

    verified: bool
    reason: str


def _matches(sym: Symbol, cited: str) -> bool:
    return cited in (sym.name, sym.qualified_name) or sym.qualified_name.endswith("." + cited)


def verify(citation: dict, repo_root: str | Path, max_lines: int = DEFAULT_MAX_LINES
           ) -> Verification:
    """Check one citation; failures carry a specific machine-readable reason."""
    root = Path(repo_root).resolve()
    full, problem = confine_path(root, citation.get("path"))
    if full is None:
        return Verification(False, "file_not_found" if problem.startswith("no such file")
                            else "invalid_path")
    start, end = citation.get("start_line"), citation.get("end_line")
    if not isinstance(start, int) or not isinstance(end, int):
        return Verification(False, "invalid_range")
    if end < start:
        return Verification(False, "reversed_range")
    try:
        text = full.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return Verification(False, "unreadable_file")
    n_lines = len(split_lines(text))
    if start < 1 or end > n_lines:
        return Verification(False, "lines_out_of_range")
    if end - start + 1 > max_lines:
        return Verification(False, "range_too_large")
    cited = (citation.get("symbol") or "").strip()
    if not cited:
        return Verification(False, "missing_symbol")
    rel = full.relative_to(root).as_posix()
    for sym in parse_source(rel, text).symbols:
        if not _matches(sym, cited):
            continue
        if start <= sym.start_line <= end or (sym.start_line <= start and end <= sym.end_line):
            return Verification(True, "ok")
    return Verification(False, "symbol_not_in_range")


def verify_all(citations: list[dict], repo_root: str | Path,
               max_lines: int = DEFAULT_MAX_LINES) -> list[dict]:
    """Copies of ``citations`` annotated with ``verified`` and ``reason``."""
    out = []
    for c in citations:
        v = verify(c, repo_root, max_lines)
        out.append({**c, "verified": v.verified, "reason": v.reason})
    return out
