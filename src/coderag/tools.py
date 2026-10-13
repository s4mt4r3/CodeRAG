"""Read-only agent tools confined to the indexed repository root.

Nothing here writes, executes, or touches the network. ``read_file`` rejects
absolute paths, ``..`` components, and any path whose resolved location (after
following symlinks) leaves the root, and caps how much it returns.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from coderag.callgraph import find_callers
from coderag.lexical import search_lexical
from coderag.store import Store

MAX_FILE_BYTES = 1_000_000


class ToolError(Exception):
    """A tool call that was rejected or failed; the message is shown to the model."""


def confine_path(root: Path, path: object) -> tuple[Path | None, str]:
    """Resolve ``path`` (relative, model-supplied) to a file inside ``root``.

    Returns ``(path, "")`` or ``(None, problem)``; problems starting with
    ``"no such file"`` mean the path was well-formed but absent.
    """
    if not isinstance(path, str) or not path.strip():
        return None, "path must be a non-empty relative path"
    if "\x00" in path or path.startswith("~"):
        return None, "invalid path"
    if PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute() \
            or PureWindowsPath(path).drive:
        return None, "absolute paths are not allowed; use a path relative to the repo"
    parts = PurePosixPath(path.replace("\\", "/")).parts
    if ".." in parts:
        return None, "'..' is not allowed in paths"
    full = (root / Path(*parts)).resolve()
    if not full.is_relative_to(root):
        return None, "path resolves outside the repository"
    if not full.is_file():
        return None, f"no such file: {path}"
    return full, ""


_SPECS = [
    {
        "name": "search_code",
        "description": "Search the indexed repository for code chunks relevant to a query "
                       "(hybrid keyword + semantic). Returns locations, not full code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for."},
                "k": {"type": "integer", "description": "Max results (1-20).", "default": 5},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_symbols",
        "description": "Find functions, classes, or methods by simple or qualified name.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": "Read lines [start, end] (1-based, inclusive) of a repository file "
                       "given as a path relative to the repository root.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start": {"type": "integer", "default": 1},
                "end": {"type": "integer"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "find_callers",
        "description": "List call sites of a function/method from the static call graph, "
                       "each with confidence resolved / ambiguous / unresolved.",
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
            "additionalProperties": False,
        },
    },
]

_TYPES = {"string": str, "integer": int}


class Toolbox:
    """The agent's tools over one index."""

    def __init__(self, store: Store, searcher: Any = None, max_read_lines: int = 200,
                 max_output_chars: int = 8000, max_k: int = 20) -> None:
        if not store.root:
            raise ValueError("index has no recorded root")
        self.store = store
        self.root = Path(store.root).resolve()
        self.searcher = searcher
        self.max_read_lines = max_read_lines
        self.max_output_chars = max_output_chars
        self.max_k = max_k

    def specs(self) -> list[dict]:
        """Tool definitions in Anthropic ``tools`` format."""
        return [dict(s) for s in _SPECS]

    # ------------------------------------------------------------ tools

    def search_code(self, query: str, k: int = 5) -> list[dict]:
        """Top-``k`` chunk locations for ``query``."""
        k = max(1, min(int(k), self.max_k))
        hits = (self.searcher.search(query, k) if self.searcher is not None
                else search_lexical(self.store, query, k))
        return [{"chunk_id": h.chunk_id, "path": h.path, "start_line": h.start_line,
                 "end_line": h.end_line, "symbol": h.symbol, "kind": h.kind,
                 "score": round(h.score, 6)} for h in hits]

    def search_symbols(self, name: str) -> list[dict]:
        """Definitions matching ``name``."""
        return [{"qualified_name": s.qualified_name, "kind": s.kind, "path": s.path,
                 "start_line": s.start_line, "end_line": s.end_line,
                 "docstring": (s.docstring or "").split("\n\n")[0][:300]}
                for s in self.store.find_symbols(name, limit=25)]

    def find_callers(self, symbol: str) -> list[dict]:
        """Call sites of ``symbol`` with confidence flags."""
        return [{"caller": e.caller, "callee": e.callee, "expr": e.expr, "path": e.path,
                 "line": e.line, "confidence": e.confidence}
                for e in find_callers(self.store, symbol, limit=50)]

    def resolve_path(self, path: str) -> Path:
        """Map a model-supplied relative path to a real file inside the root, or raise."""
        full, problem = confine_path(self.root, path)
        if full is None:
            raise ToolError(problem)
        return full

    def read_file(self, path: str, start: int = 1, end: int | None = None) -> dict:
        """Numbered lines ``start..end`` of a repo file, capped at ``max_read_lines``."""
        full = self.resolve_path(path)
        if full.stat().st_size > MAX_FILE_BYTES:
            raise ToolError("file too large to read")
        try:
            lines = full.read_text(encoding="utf-8").split("\n")
        except UnicodeDecodeError as exc:
            raise ToolError("file is not UTF-8 text") from exc
        if lines and lines[-1] == "":
            lines.pop()
        end = len(lines) if end is None else end
        if start < 1 or end < start:
            raise ToolError("need 1 <= start <= end")
        if start > len(lines):
            raise ToolError(f"start beyond end of file ({len(lines)} lines)")
        end = min(end, len(lines))
        truncated = end - start + 1 > self.max_read_lines
        if truncated:
            end = start + self.max_read_lines - 1
        text = "\n".join(f"{n}: {lines[n - 1]}" for n in range(start, end + 1))
        rel = full.relative_to(self.root).as_posix()
        return {"path": rel, "start_line": start, "end_line": end, "text": text,
                "truncated": truncated}

    # ------------------------------------------------------------ dispatch

    def _validate(self, name: str, args: Any) -> dict:
        spec = next((s for s in _SPECS if s["name"] == name), None)
        if spec is None:
            raise ToolError(f"unknown tool {name!r}")
        if not isinstance(args, dict):
            raise ToolError("tool arguments must be a JSON object")
        schema = spec["input_schema"]
        props = schema["properties"]
        for key in schema["required"]:
            if key not in args:
                raise ToolError(f"missing required argument {key!r}")
        for key, value in args.items():
            if key not in props:
                raise ToolError(f"unexpected argument {key!r}")
            want = _TYPES[props[key]["type"]]
            if not isinstance(value, want) or isinstance(value, bool):
                raise ToolError(f"argument {key!r} must be {props[key]['type']}")
        return args

    def call(self, name: str, args: Any) -> str:
        """Validate and run a tool; returns JSON text capped at ``max_output_chars``."""
        args = self._validate(name, args)
        result = getattr(self, name)(**args)
        text = json.dumps(result, ensure_ascii=False)
        if len(text) > self.max_output_chars:
            text = text[: self.max_output_chars] + f" ...[truncated {len(text)} chars]"
        return text
