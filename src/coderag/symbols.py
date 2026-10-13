"""Tree-sitter extraction of functions, classes, and methods from Python source.

Span convention: ``start_line`` is the first decorator line when the definition is
decorated (tree-sitter's ``decorated_definition``), otherwise the ``def``/``class``
line. ``end_line`` is the last line of the last non-comment statement in the body.
Lines are 1-based and inclusive. This matches the ``ast`` convention used by the
test oracle (``decorator_list[0].lineno`` .. ``end_lineno``).
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser, Tree

from coderag.ingest import walk_repo

# Statements whose nested blocks share the enclosing scope (defs inside them count).
_SCOPE_TRANSPARENT = frozenset({
    "if_statement", "elif_clause", "else_clause", "for_statement", "while_statement",
    "try_statement", "except_clause", "except_group_clause", "finally_clause",
    "with_statement", "block", "ERROR",
})
# Nodes whose last child determines where they end.
_COMPOUND = _SCOPE_TRANSPARENT - {"block", "ERROR"} | {
    "function_definition", "class_definition", "decorated_definition",
    "match_statement", "case_clause",
}
_STRING_RE = re.compile(r"([A-Za-z]*)(\"\"\"|'''|\"|')", re.S)


@dataclass
class Symbol:
    """A function, method, or class definition."""

    kind: str
    name: str
    qualified_name: str
    path: str
    start_line: int
    end_line: int
    docstring: str | None = None
    decorators: list[str] = field(default_factory=list)
    is_async: bool = False
    parent: str | None = None


@dataclass
class ParseResult:
    """Symbols from one file plus any recoverable parse warnings."""

    symbols: list[Symbol]
    warnings: list[str]


@lru_cache(maxsize=1)
def _parser() -> Parser:
    return Parser(Language(tspython.language()))


def parse_tree(source: str) -> Tree:
    """Parse Python source into a tree-sitter tree."""
    return _parser().parse(source.encode("utf-8"))


def module_name(rel_path: str) -> str:
    """Map ``a/b/c.py`` to ``a.b.c`` and ``a/b/__init__.py`` to ``a.b``."""
    parts = list(Path(rel_path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def split_lines(text: str) -> list[str]:
    """Split on ``\\n`` only, keeping line ends.

    ``str.splitlines`` also breaks on form feeds and U+2028, which would shift line
    numbers relative to tree-sitter and ``ast``.
    """
    parts = text.split("\n")
    lines = [p + "\n" for p in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


def text_of(node: Node) -> str:
    """Decoded source text of a node."""
    return (node.text or b"").decode("utf-8", errors="replace")


def last_line(node: Node) -> int:
    """1-based last line of a node, ignoring trailing comments inside blocks."""
    if node.type == "block":
        stmts = [c for c in node.named_children if c.type != "comment"]
        if stmts:
            return last_line(stmts[-1])
    elif node.type in _COMPOUND and node.children:
        return last_line(node.children[-1])
    row, col = node.end_point
    if col == 0 and row > node.start_point[0]:
        row -= 1
    return row + 1


def _decode_string_piece(raw: str) -> str | None:
    m = _STRING_RE.match(raw)
    if not m:
        return None
    prefix, quote = m.group(1).lower(), m.group(2)
    if "f" in prefix or "b" in prefix or "t" in prefix:
        return None
    if not raw.endswith(quote) or len(raw) < len(m.group(0)) + len(quote):
        return None
    body = raw[len(m.group(0)) : len(raw) - len(quote)]
    if "r" not in prefix and "\\" in body:
        body = body.encode("latin-1", "backslashreplace").decode("unicode_escape")
    return body


def docstring_of(body: Node | None) -> str | None:
    """The cleaned docstring of a block, following ``ast.get_docstring`` semantics."""
    if body is None:
        return None
    stmts = [c for c in body.named_children if c.type != "comment"]
    if not stmts or stmts[0].type != "expression_statement":
        return None
    expr = stmts[0].named_children
    if len(expr) != 1:
        return None
    node = expr[0]
    pieces = [node] if node.type == "string" else (
        [c for c in node.named_children if c.type == "string"]
        if node.type == "concatenated_string" else []
    )
    if not pieces:
        return None
    decoded = [_decode_string_piece(text_of(p)) for p in pieces]
    if any(d is None for d in decoded):
        return None
    return inspect.cleandoc("".join(d for d in decoded if d is not None))


def iter_definitions(tree: Tree, rel_path: str) -> list[tuple[Symbol, Node]]:
    """Every definition in source order, paired with its def/class node."""
    mod = module_name(rel_path)
    out: list[tuple[Symbol, Node]] = []

    def visit(container: Node, prefix: str, parent: str | None, in_class: bool) -> None:
        for child in container.children:
            outer, defn, decorators = child, child, []
            if child.type == "decorated_definition":
                defn = child.child_by_field_name("definition")
                decorators = [
                    text_of(d).lstrip()[1:].strip()
                    for d in child.children
                    if d.type == "decorator"
                ]
                if defn is None:
                    continue
            if defn.type not in ("function_definition", "class_definition"):
                if child.type in _SCOPE_TRANSPARENT:
                    visit(child, prefix, parent, in_class)
                continue
            name_node = defn.child_by_field_name("name")
            if name_node is None:
                continue
            name = text_of(name_node)
            qname = f"{prefix}.{name}" if prefix else name
            is_class = defn.type == "class_definition"
            body = defn.child_by_field_name("body")
            sym = Symbol(
                kind="class" if is_class else ("method" if in_class else "function"),
                name=name,
                qualified_name=qname,
                path=rel_path,
                start_line=outer.start_point[0] + 1,
                end_line=last_line(defn),
                docstring=docstring_of(body),
                decorators=decorators,
                is_async=(not is_class) and any(c.type == "async" for c in defn.children),
                parent=parent,
            )
            out.append((sym, defn))
            if body is not None:
                visit(body, qname, qname, is_class)

    visit(tree.root_node, mod, None, False)
    out.sort(key=lambda pair: (pair[0].start_line, pair[0].qualified_name))
    return out


def _parse_warnings(tree: Tree, rel_path: str, limit: int = 5) -> list[str]:
    if not tree.root_node.has_error:
        return []
    found: list[str] = []
    stack = [tree.root_node]
    while stack and len(found) < limit:
        node = stack.pop()
        if node.type == "ERROR" or node.is_missing:
            what = f"missing {node.type!r}" if node.is_missing else "unparseable code"
            found.append(f"{rel_path}:{node.start_point[0] + 1}: syntax error ({what})")
            continue
        if node.has_error:
            stack.extend(reversed(node.children))
    return found or [f"{rel_path}: syntax error (parse recovered)"]


def parse_source(rel_path: str, source: str) -> ParseResult:
    """Extract symbols from one file; never raises on syntax errors."""
    tree = parse_tree(source)
    symbols = [sym for sym, _ in iter_definitions(tree, rel_path)]
    return ParseResult(symbols=symbols, warnings=_parse_warnings(tree, rel_path))


def extract_repo_symbols(root: str | Path, ignore: tuple[str, ...] = ()) -> list[Symbol]:
    """Symbols for every indexable file under ``root``, sorted by (path, start_line)."""
    out: list[Symbol] = []
    for f in walk_repo(root, ignore=ignore):
        out.extend(parse_source(f.path, f.text).symbols)
    return out
