"""Name-based call graph extracted with tree-sitter.

Each call inside a function body (nested def/class bodies excluded) becomes one or
more edges with a confidence flag:

* ``resolved``: the target follows from static scope rules alone: enclosing nested
  defs, same-module top-level defs, imports (aliases, relative imports, package
  re-exports), ``self``/``cls`` methods of the enclosing class, and attribute
  access on imported modules.
* ``ambiguous``: not resolvable by scope rules, but the final name matches one or
  more definitions in the repo; one row per candidate, never a silent pick.
* ``unresolved``: builtins, external libraries, or names with no candidate.

Known limits: no type inference, so ``obj.method()`` on a parameter or attribute is
at best ambiguous; inherited methods via ``self`` are ambiguous; dynamic dispatch,
``getattr``, and star imports are not followed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tree_sitter import Node

from coderag.ingest import SourceFile
from coderag.store import Store
from coderag.symbols import Symbol, iter_definitions, module_name, parse_tree, text_of

_SKIP = frozenset({"function_definition", "class_definition", "decorated_definition"})


@dataclass
class CallEdge:
    """One call site, optionally attributed to a callee definition."""

    caller: str
    callee: str | None
    callee_name: str
    expr: str
    path: str
    line: int
    confidence: str


def _chain(node: Node) -> list[str] | None:
    parts: list[str] = []
    while node.type == "attribute":
        attr = node.child_by_field_name("attribute")
        obj = node.child_by_field_name("object")
        if attr is None or obj is None:
            return None
        parts.append(text_of(attr))
        node = obj
    if node.type == "identifier":
        parts.append(text_of(node))
        return list(reversed(parts))
    return None


def _call_sites(defn: Node) -> list[tuple[list[str] | None, str, int]]:
    """(chain or None, final name, line) for calls in a def body, skipping nested defs."""
    body = defn.child_by_field_name("body")
    out: list[tuple[list[str] | None, str, int]] = []
    stack = [body] if body is not None else []
    while stack:
        node = stack.pop()
        if node.type in _SKIP:
            continue
        if node.type == "call":
            fn = node.child_by_field_name("function")
            if fn is not None:
                chain = _chain(fn)
                if chain:
                    out.append((chain, chain[-1], node.start_point[0] + 1))
                elif fn.type == "attribute" and fn.child_by_field_name("attribute") is not None:
                    out.append((None, text_of(fn.child_by_field_name("attribute")),
                                node.start_point[0] + 1))
        stack.extend(node.children)
    out.sort(key=lambda t: t[2])
    return out


def _relative_base(mod: str, is_pkg: bool, dots: int) -> list[str]:
    base = mod.split(".") if mod else []
    if not is_pkg:
        base = base[:-1]
    if dots > 1:
        base = base[: max(0, len(base) - (dots - 1))]
    return base


def _imports(root: Node, rel_path: str) -> dict[str, str]:
    """Local name -> dotted target for every import anywhere in the module."""
    mod = module_name(rel_path)
    is_pkg = rel_path.endswith("__init__.py")
    out: dict[str, str] = {}
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "import_statement":
            for child in node.named_children:
                if child.type == "aliased_import":
                    name, alias = child.child_by_field_name("name"), child.child_by_field_name(
                        "alias")
                    if name is not None and alias is not None:
                        out[text_of(alias)] = text_of(name)
                elif child.type == "dotted_name":
                    head = text_of(child).split(".")[0]
                    out[head] = head
        elif node.type == "import_from_statement":
            src_node = node.child_by_field_name("module_name")
            if src_node is None:
                continue
            if src_node.type == "relative_import":
                prefix = next((c for c in src_node.children if c.type == "import_prefix"), None)
                dots = text_of(prefix).count(".") if prefix is not None else 0
                rest = next((c for c in src_node.children if c.type == "dotted_name"), None)
                parts = _relative_base(mod, is_pkg, dots)
                if rest is not None:
                    parts += text_of(rest).split(".")
                src = ".".join(parts)
            else:
                src = text_of(src_node)
            for child in node.named_children:
                # py-tree-sitter hands out fresh Node wrappers, so identity checks fail.
                if child.start_byte == src_node.start_byte or child.type == "wildcard_import":
                    continue
                if child.type == "aliased_import":
                    name, alias = child.child_by_field_name("name"), child.child_by_field_name(
                        "alias")
                    if name is not None and alias is not None:
                        out[text_of(alias)] = f"{src}.{text_of(name)}" if src else text_of(name)
                elif child.type == "dotted_name":
                    out[text_of(child)] = f"{src}.{text_of(child)}" if src else text_of(child)
        else:
            stack.extend(node.children)
    return out


class _Resolver:
    def __init__(self, files: Sequence[SourceFile]) -> None:
        self.modules: set[str] = set()
        self.imports: dict[str, dict[str, str]] = {}
        self.defs: list[tuple[Symbol, Node]] = []
        for f in files:
            tree = parse_tree(f.text)
            mod = module_name(f.path)
            self.modules.add(mod)
            self.imports[mod] = _imports(tree.root_node, f.path)
            self.defs.extend(iter_definitions(tree, f.path))
        self.symbols: dict[str, Symbol] = {}
        self.by_name: dict[str, list[str]] = {}
        for sym, _ in self.defs:
            if sym.qualified_name not in self.symbols:
                self.symbols[sym.qualified_name] = sym
                self.by_name.setdefault(sym.name, []).append(sym.qualified_name)

    def resolve_dotted(self, dotted: str, depth: int = 0) -> str | None:
        if depth > 8:
            return None
        if dotted in self.symbols:
            return dotted
        parts = dotted.split(".")
        for i in range(len(parts) - 1, 0, -1):
            mod = ".".join(parts[:i])
            if mod in self.modules:
                target = self.imports.get(mod, {}).get(parts[i])
                if target is None:
                    return None
                return self.resolve_dotted(".".join([target, *parts[i + 1 :]]), depth + 1)
        return None

    def _is_external(self, dotted: str) -> bool:
        head = dotted.split(".")[0]
        return not any(m == head or m.startswith(head + ".") for m in self.modules)

    def resolve(self, sym: Symbol, chain: list[str]) -> tuple[str, str | None]:
        """("resolved", target) | ("external", None) | ("unknown", None)."""
        mod = module_name(sym.path)
        root, rest = chain[0], chain[1:]
        if root in ("self", "cls") and sym.kind == "method" and sym.parent:
            if len(rest) == 1:
                cand = f"{sym.parent}.{rest[0]}"
                if cand in self.symbols and self.symbols[cand].kind != "class":
                    return "resolved", cand
            return "unknown", None
        if not rest:
            scope: str | None = sym.qualified_name
            while scope and scope in self.symbols:
                s = self.symbols[scope]
                if s.kind != "class" and f"{scope}.{root}" in self.symbols:
                    return "resolved", f"{scope}.{root}"
                scope = s.parent
        top = f"{mod}.{root}"
        if top in self.symbols:
            target = self.resolve_dotted(".".join([top, *rest]))
            return ("resolved", target) if target else ("unknown", None)
        imported = self.imports.get(mod, {}).get(root)
        if imported is not None:
            target = self.resolve_dotted(".".join([imported, *rest]))
            if target:
                return "resolved", target
            if self._is_external(imported):
                return "external", None
        return "unknown", None


def build_edges(files: Sequence[SourceFile]) -> list[CallEdge]:
    """Call edges for every function and method in ``files``."""
    res = _Resolver(files)
    edges: list[CallEdge] = []
    for sym, node in res.defs:
        if sym.kind == "class":
            continue
        for chain, name, line in _call_sites(node):
            expr = ".".join(chain) if chain else f"<expr>.{name}"
            status, target = res.resolve(sym, chain) if chain else ("unknown", None)
            base = dict(caller=sym.qualified_name, callee_name=name, expr=expr,
                        path=sym.path, line=line)
            if status == "resolved":
                edges.append(CallEdge(callee=target, confidence="resolved", **base))
                continue
            candidates = res.by_name.get(name, []) if status == "unknown" else []
            if candidates:
                edges.extend(CallEdge(callee=c, confidence="ambiguous", **base)
                             for c in candidates)
            else:
                edges.append(CallEdge(callee=None, confidence="unresolved", **base))
    return edges


def write_edges(store: Store, files: Sequence[SourceFile]) -> int:
    """Replace every edge in the store; returns the number of rows written."""
    edges = build_edges(files)
    store.conn.execute("DELETE FROM edges")
    store.conn.executemany(
        "INSERT INTO edges(caller, callee, callee_name, expr, path, line, confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(e.caller, e.callee, e.callee_name, e.expr, e.path, e.line, e.confidence)
         for e in edges],
    )
    return len(edges)


_ORDER = "CASE confidence WHEN 'resolved' THEN 0 WHEN 'ambiguous' THEN 1 ELSE 2 END"
_EDGE_COLS = "caller, callee, callee_name, expr, path, line, confidence"


def _targets(store: Store, symbol: str) -> list[str]:
    exact = store.conn.execute(
        "SELECT DISTINCT qualified_name FROM symbols WHERE qualified_name = ?", (symbol,)
    ).fetchall()
    if exact:
        return [r[0] for r in exact]
    return sorted({s.qualified_name for s in store.find_symbols(symbol)})


def _query(store: Store, column: str, targets: list[str], limit: int) -> list[CallEdge]:
    if not targets:
        return []
    marks = ",".join("?" * len(targets))
    rows = store.conn.execute(
        f"SELECT {_EDGE_COLS} FROM edges WHERE {column} IN ({marks}) "
        f"ORDER BY {_ORDER}, path, line LIMIT ?",
        (*targets, limit),
    ).fetchall()
    return [CallEdge(*r) for r in rows]


def find_callers(store: Store, symbol: str, limit: int = 200) -> list[CallEdge]:
    """Edges whose callee is ``symbol`` (qualified, or a simple name matching several)."""
    return _query(store, "callee", _targets(store, symbol), limit)


def find_callees(store: Store, symbol: str, limit: int = 200) -> list[CallEdge]:
    """Edges whose caller is ``symbol``."""
    return _query(store, "caller", _targets(store, symbol), limit)
