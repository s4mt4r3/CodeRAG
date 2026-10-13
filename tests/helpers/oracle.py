"""Ground-truth oracle built on the stdlib ``ast`` module.

The system under test parses with tree-sitter; this oracle is deliberately
independent of it. Conventions are documented in tests/fixtures/expected.json.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

SKIP_DIRS = {".git", "venv", ".venv", "node_modules", "__pycache__"}

_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass
class OracleSymbol:
    """One function, method, or class as seen by ``ast``."""

    kind: str
    name: str
    qualified_name: str
    path: str
    start_line: int
    end_line: int
    docstring: str | None
    decorators: list[str]
    is_async: bool
    parent: str | None
    calls: list[str] = field(default_factory=list)
    call_exprs: list[list[str]] = field(default_factory=list)


def module_name(rel_path: str) -> str:
    """Map ``a/b/c.py`` to ``a.b.c`` and ``a/b/__init__.py`` to ``a.b``."""
    parts = list(Path(rel_path).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def iter_python_files(root: Path) -> list[str]:
    """Relative POSIX paths of every ``.py`` file under ``root``, sorted."""
    out = []
    for p in root.rglob("*.py"):
        rel = p.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        out.append(rel.as_posix())
    return sorted(out)


def _callee_chain(func: ast.expr) -> list[str] | None:
    """``a.b.c`` -> ["a", "b", "c"]; None when the chain is not rooted at a Name."""
    parts: list[str] = []
    node = func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return None


def _final_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _body_calls(fn: ast.AST) -> list[ast.Call]:
    """Calls in a function body, not descending into nested def/class nodes."""
    calls: list[ast.Call] = []
    stack: list[ast.AST] = list(getattr(fn, "body", []))
    while stack:
        node = stack.pop()
        if isinstance(node, _DEF_NODES):
            continue
        if isinstance(node, ast.Call):
            calls.append(node)
        stack.extend(ast.iter_child_nodes(node))
    return calls


def _start_line(node: ast.AST) -> int:
    decos = getattr(node, "decorator_list", [])
    return min([d.lineno for d in decos] + [node.lineno])


def extract_file(rel_path: str, source: str) -> list[OracleSymbol]:
    """All symbols in one file, in source order."""
    tree = ast.parse(source)
    mod = module_name(rel_path)
    out: list[OracleSymbol] = []

    def visit(body: list[ast.stmt], prefix: str, parent: str | None, in_class: bool) -> None:
        for node in body:
            if not isinstance(node, _DEF_NODES):
                # Defs nested in if/try/with/for blocks still belong to this scope.
                for child_body in _compound_bodies(node):
                    visit(child_body, prefix, parent, in_class)
                continue
            qname = f"{prefix}.{node.name}" if prefix else node.name
            if isinstance(node, ast.ClassDef):
                kind = "class"
            else:
                kind = "method" if in_class else "function"
            sym = OracleSymbol(
                kind=kind,
                name=node.name,
                qualified_name=qname,
                path=rel_path,
                start_line=_start_line(node),
                end_line=node.end_lineno or node.lineno,
                docstring=ast.get_docstring(node),
                decorators=[ast.unparse(d) for d in node.decorator_list],
                is_async=isinstance(node, ast.AsyncFunctionDef),
                parent=parent,
            )
            if kind != "class":
                calls = _body_calls(node)
                sym.calls = sorted({n for c in calls if (n := _final_name(c.func))})
                sym.call_exprs = [ch for c in calls if (ch := _callee_chain(c.func))]
            out.append(sym)
            visit(node.body, qname, qname, isinstance(node, ast.ClassDef))

    visit(tree.body, mod, None, False)
    out.sort(key=lambda s: (s.start_line, s.qualified_name))
    return out


def _compound_bodies(node: ast.AST) -> list[list[ast.stmt]]:
    bodies = []
    for attr in ("body", "orelse", "finalbody"):
        val = getattr(node, attr, None)
        if isinstance(val, list) and val and isinstance(val[0], ast.stmt):
            bodies.append(val)
    for handler in getattr(node, "handlers", []) or []:
        bodies.append(handler.body)
    return bodies


def extract_repo(root: str | Path) -> list[OracleSymbol]:
    """Symbols for every Python file under ``root`` sorted by (path, start_line)."""
    root = Path(root)
    out: list[OracleSymbol] = []
    for rel in iter_python_files(root):
        out.extend(extract_file(rel, (root / rel).read_text(encoding="utf-8")))
    return out


# --------------------------------------------------------------------------- edges


def _resolve_relative(mod: str, is_pkg: bool, level: int, target: str | None) -> str:
    base = mod.split(".") if mod else []
    if not is_pkg:
        base = base[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)]
    if target:
        base = base + target.split(".")
    return ".".join(base)


def _import_map(rel_path: str, tree: ast.Module, modules: set[str]) -> dict[str, str]:
    """Local name -> fully dotted target (a module or ``module.attr``)."""
    mod = module_name(rel_path)
    is_pkg = rel_path.endswith("__init__.py")
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    out[a.asname] = a.name
                else:
                    head = a.name.split(".")[0]
                    out[head] = head
        elif isinstance(node, ast.ImportFrom):
            src = (
                _resolve_relative(mod, is_pkg, node.level, node.module)
                if node.level
                else (node.module or "")
            )
            for a in node.names:
                if a.name == "*":
                    continue
                out[a.asname or a.name] = f"{src}.{a.name}" if src else a.name
    return out


class _Resolver:
    def __init__(self, root: Path) -> None:
        self.files = iter_python_files(root)
        self.trees = {f: ast.parse((root / f).read_text(encoding="utf-8")) for f in self.files}
        self.modules = {module_name(f): f for f in self.files}
        self.symbols = {s.qualified_name: s for s in extract_repo(root)}
        self.imports = {
            module_name(f): _import_map(f, t, set(self.modules)) for f, t in self.trees.items()
        }

    def resolve_dotted(self, dotted: str, depth: int = 0) -> str | None:
        """Follow module imports (re-exports) until a known symbol is reached."""
        if depth > 8:
            return None
        if dotted in self.symbols:
            return dotted
        parts = dotted.split(".")
        for i in range(len(parts) - 1, 0, -1):
            mod = ".".join(parts[:i])
            if mod in self.modules:
                head, rest = parts[i], parts[i + 1 :]
                target = self.imports.get(mod, {}).get(head)
                if target is None:
                    return None
                return self.resolve_dotted(".".join([target, *rest]), depth + 1)
        return None

    def resolve_call(self, caller: OracleSymbol, chain: list[str]) -> str | None:
        mod = module_name(caller.path)
        root, rest = chain[0], chain[1:]
        enclosing_class = None
        if caller.kind == "method" and caller.parent:
            enclosing_class = caller.parent
        if root in ("self", "cls") and enclosing_class and len(rest) == 1:
            cand = f"{enclosing_class}.{rest[0]}"
            sym = self.symbols.get(cand)
            return cand if sym and sym.kind != "class" else None
        if not rest:
            # Enclosing function scopes, innermost first (class scopes are skipped).
            scope: str | None = caller.qualified_name
            while scope and scope in self.symbols:
                sym = self.symbols[scope]
                if sym.kind != "class":
                    cand = f"{scope}.{root}"
                    if cand in self.symbols:
                        return cand
                scope = sym.parent
        top = f"{mod}.{root}"
        if top in self.symbols:
            return self.resolve_dotted(".".join([top, *rest]))
        target = self.imports.get(mod, {}).get(root)
        if target is None:
            return None
        return self.resolve_dotted(".".join([target, *rest]))


def resolved_edges(root: str | Path) -> set[tuple[str, str]]:
    """Caller -> callee edges resolvable by static scope rules."""
    res = _Resolver(Path(root))
    edges: set[tuple[str, str]] = set()
    for sym in res.symbols.values():
        for chain in sym.call_exprs:
            callee = res.resolve_call(sym, chain)
            if callee:
                edges.add((sym.qualified_name, callee))
    return edges


def name_callers(root: str | Path) -> dict[str, set[str]]:
    """Simple callee name -> qualified names of functions that call that name."""
    out: dict[str, set[str]] = {}
    for sym in extract_repo(root):
        for name in sym.calls:
            out.setdefault(name, set()).add(sym.qualified_name)
    return out
