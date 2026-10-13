"""AST-aware chunking (one chunk per function/method/class plus module code) and a
fixed-size line-window baseline used for ablation.

Chunk units are top-level functions, classes, and methods (recursively for nested
classes). Functions nested inside functions stay inside their parent's chunk. A
class chunk holds the class header, docstring, and field-level code; its methods
are separate chunks, so a class chunk may consist of several line ``segments``.
Units over the token budget are split only at statement boundaries.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass

from tree_sitter import Node

from coderag.ingest import SourceFile
from coderag.symbols import last_line, module_name, parse_tree, split_lines, text_of

DEFAULT_MAX_TOKENS = 400
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_CLAUSES = frozenset({
    "elif_clause", "else_clause", "except_clause", "except_group_clause",
    "finally_clause", "case_clause",
})

Segment = tuple[int, int]


@dataclass
class Chunk:
    """A retrievable unit of code with its location and provenance."""

    chunk_id: str
    path: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    segments: list[Segment]
    code: str
    text: str
    part: int = 0
    parts: int = 1
    partial: bool = False


def count_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (identifiers and punctuation)."""
    return len(_TOKEN_RE.findall(text))


def _direct_defs(container: Node) -> Iterator[tuple[Node, Node]]:
    for child in container.children:
        if child.type == "decorated_definition":
            defn = child.child_by_field_name("definition")
            if defn is not None:
                yield child, defn
        elif child.type in ("function_definition", "class_definition"):
            yield child, child


def _statements(block: Node) -> list[Node]:
    return [c for c in block.named_children if c.type != "comment"]


def _blocks_of(node: Node) -> list[Node]:
    if node.type == "decorated_definition":
        defn = node.child_by_field_name("definition")
        return _blocks_of(defn) if defn is not None else []
    out = []
    for child in node.children:
        if child.type == "block":
            out.append(child)
        elif child.type in _CLAUSES:
            out.extend(_blocks_of(child))
    return out


def _start(node: Node) -> int:
    return node.start_point[0] + 1


class _FileChunker:
    def __init__(self, file: SourceFile, max_tokens: int) -> None:
        self.file = file
        self.max_tokens = max_tokens
        self.lines = split_lines(file.text)
        self.tok = [0] + [count_tokens(ln) for ln in self.lines]
        self.chunks: list[Chunk] = []

    def tokens(self, a: int, b: int) -> int:
        return sum(self.tok[a : b + 1])

    def code_of(self, segments: list[Segment]) -> str:
        return "".join("".join(self.lines[a - 1 : b]) for a, b in segments)

    # ------------------------------------------------------------ boundaries

    def _inner_boundaries(self, stmt: Node, out: set[int]) -> None:
        """Statement starts inside an over-budget compound statement."""
        for child in stmt.children:
            if child.type in _CLAUSES:
                out.add(_start(child))
        for block in _blocks_of(stmt):
            for inner in _statements(block):
                out.add(_start(inner))
                if self.tokens(_start(inner), last_line(inner)) > self.max_tokens:
                    self._inner_boundaries(inner, out)

    def _block_boundaries(self, block: Node | None, skip_first: bool) -> set[int]:
        out: set[int] = set()
        if block is None:
            return out
        stmts = _statements(block)
        for i, stmt in enumerate(stmts):
            if i > 0 or not skip_first:
                out.add(_start(stmt))
            if self.tokens(_start(stmt), last_line(stmt)) > self.max_tokens:
                self._inner_boundaries(stmt, out)
        return out

    # ------------------------------------------------------------ packing

    def _pack(self, atoms: list[Segment]) -> list[list[Segment]]:
        """Greedily merge consecutive atoms into pieces within the token budget."""
        pieces: list[list[Segment]] = []
        cur: list[Segment] = []
        cur_tok = 0
        for a, b in atoms:
            t = self.tokens(a, b)
            if cur and cur_tok + t > self.max_tokens:
                pieces.append(cur)
                cur, cur_tok = [], 0
            if cur and cur[-1][1] + 1 == a:
                cur[-1] = (cur[-1][0], b)
            else:
                cur.append((a, b))
            cur_tok += t
        if cur:
            pieces.append(cur)
        return pieces

    @staticmethod
    def _split_runs(runs: list[Segment], cuts: set[int]) -> list[Segment]:
        atoms: list[Segment] = []
        for a, b in runs:
            start = a
            for c in sorted(x for x in cuts if a < x <= b):
                atoms.append((start, c - 1))
                start = c
            atoms.append((start, b))
        return atoms

    # ------------------------------------------------------------ emitting

    def _standalone_ok(self, code: str) -> bool:
        if not parse_tree(textwrap.dedent(code)).root_node.has_error:
            return True
        return not parse_tree("class _Wrap:\n" + code).root_node.has_error

    def _emit(self, symbol: str, kind: str, pieces: list[list[Segment]]) -> None:
        total = len(pieces)
        for i, segs in enumerate(pieces):
            code = self.code_of(segs)
            start, end = segs[0][0], segs[-1][1]
            self.chunks.append(Chunk(
                chunk_id=f"{self.file.path}:{start}:{symbol}#{i}",
                path=self.file.path,
                symbol=symbol,
                kind=kind,
                start_line=start,
                end_line=end,
                segments=segs,
                code=code,
                text=f"# {self.file.path} :: {symbol}\n{code}",
                part=i,
                parts=total,
                partial=not self._standalone_ok(code),
            ))

    def _owned_runs(self, owned: list[int]) -> list[Segment]:
        """Maximal runs of owned lines, trimmed of leading/trailing blank lines."""
        runs: list[Segment] = []
        for n in sorted(owned):
            if runs and runs[-1][1] + 1 == n:
                runs[-1] = (runs[-1][0], n)
            else:
                runs.append((n, n))
        trimmed = []
        for a, b in runs:
            while a <= b and not self.lines[a - 1].strip():
                a += 1
            while b >= a and not self.lines[b - 1].strip():
                b -= 1
            if a <= b:
                trimmed.append((a, b))
        return trimmed

    def _emit_container(self, symbol: str, kind: str, owned: list[int], cuts: set[int]) -> None:
        runs = self._owned_runs(owned)
        if not runs:
            return
        if sum(self.tokens(a, b) for a, b in runs) <= self.max_tokens:
            self._emit(symbol, kind, [runs])
        else:
            self._emit(symbol, kind, self._pack(self._split_runs(runs, cuts)))

    def _function(self, qname: str, kind: str, outer: Node, defn: Node) -> None:
        start, end = _start(outer), last_line(defn)
        if self.tokens(start, end) <= self.max_tokens:
            self._emit(qname, kind, [[(start, end)]])
            return
        cuts = self._block_boundaries(defn.child_by_field_name("body"), skip_first=True)
        atoms = self._split_runs([(start, end)], cuts)
        self._emit(qname, kind, self._pack(atoms))

    def _class(self, qname: str, outer: Node, defn: Node) -> None:
        start, end = _start(outer), last_line(defn)
        body = defn.child_by_field_name("body")
        children = list(_direct_defs(body)) if body is not None else []
        holes = {n for o, d in children for n in range(_start(o), last_line(d) + 1)}
        owned = [n for n in range(start, end + 1) if n not in holes]
        cuts = self._block_boundaries(body, skip_first=True)
        cuts |= {last_line(d) + 1 for _, d in children}
        self._emit_container(qname, "class", owned, cuts)
        for o, d in children:
            self._unit(o, d, qname, in_class=True)

    def _unit(self, outer: Node, defn: Node, prefix: str, in_class: bool) -> None:
        name_node = defn.child_by_field_name("name")
        if name_node is None:
            return
        qname = f"{prefix}.{text_of(name_node)}" if prefix else text_of(name_node)
        if defn.type == "class_definition":
            self._class(qname, outer, defn)
        else:
            self._function(qname, "method" if in_class else "function", outer, defn)

    def run(self) -> list[Chunk]:
        if not self.file.text.strip():
            return []
        tree = parse_tree(self.file.text)
        root = tree.root_node
        mod = module_name(self.file.path)
        units = list(_direct_defs(root))
        holes = {n for o, d in units for n in range(_start(o), last_line(d) + 1)}
        owned = [n for n in range(1, len(self.lines) + 1) if n not in holes]
        cuts = {_start(s) for s in root.named_children if s.type != "comment"}
        cuts |= {last_line(d) + 1 for _, d in units}
        self._emit_container(mod, "module", owned, cuts)
        for o, d in units:
            self._unit(o, d, mod, in_class=False)
        self.chunks.sort(key=lambda c: (c.start_line, c.part))
        return self.chunks


def ast_chunks(file: SourceFile, max_tokens: int = DEFAULT_MAX_TOKENS) -> list[Chunk]:
    """Chunk one file by semantic unit; see the module docstring for the rules."""
    return _FileChunker(file, max_tokens).run()


def fixed_chunks(file: SourceFile, size: int = 40, overlap: int = 10) -> list[Chunk]:
    """Naive baseline: windows of ``size`` lines, consecutive windows share ``overlap``."""
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    lines = split_lines(file.text)
    n = len(lines)
    mod = module_name(file.path)
    out: list[Chunk] = []
    start = 1
    while start <= n:
        end = min(start + size - 1, n)
        code = "".join(lines[start - 1 : end])
        if code.strip():
            out.append(Chunk(
                chunk_id=f"{file.path}:{start}:fixed#0",
                path=file.path,
                symbol=mod,
                kind="fixed",
                start_line=start,
                end_line=end,
                segments=[(start, end)],
                code=code,
                text=f"# {file.path} :: lines {start}-{end}\n{code}",
            ))
        if end == n:
            break
        start += size - overlap
    return out
