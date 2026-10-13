"""Oracle-backed question generation.

Ground truth comes from Python's stdlib ``ast`` (never from the tree-sitter code
under test), so the benchmark needs no hand labels. Question types:

* ``definition``: "Where is `name` defined?" -> the symbol's span (for a class, its
  header region: decorators, signature, docstring and code before its first method).
* ``docstring_masked``: the first docstring sentence with the function's own name
  parts removed -> the function's span. Tests semantic rather than name matching.
* ``callers``: "Which functions call `name`?" -> spans of every function whose body
  calls that name (name-based, nested defs excluded).
"""

from __future__ import annotations

import ast
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from coderag.ingest import walk_repo
from coderag.lexical import split_identifier
from coderag.symbols import module_name

DEFAULT_MIX = {"definition": 0.4, "docstring_masked": 0.4, "callers": 0.2}
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_WORD_RE = re.compile(r"\w+")


@dataclass
class Question:
    """One benchmark question with span-level ground truth."""

    id: str
    type: str
    question: str
    symbol: str
    truth: list[tuple[str, int, int]]
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-safe representation."""
        d = asdict(self)
        d["truth"] = [list(t) for t in self.truth]
        return d


@dataclass
class _Sym:
    qname: str
    name: str
    kind: str
    path: str
    start: int
    end: int
    docstring: str | None
    parent: str | None
    calls: set[str]
    header_end: int


def _calls(fn: ast.AST) -> set[str]:
    out: set[str] = set()
    stack = list(getattr(fn, "body", []))
    while stack:
        node = stack.pop()
        if isinstance(node, _DEFS):
            continue
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                out.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                out.add(node.func.attr)
        stack.extend(ast.iter_child_nodes(node))
    return out


def _header_end(cls: ast.ClassDef, lines: list[str]) -> int:
    direct = [n for n in cls.body if isinstance(n, _DEFS)]
    if not direct:
        return cls.end_lineno or cls.lineno
    first = min(min([d.lineno for d in n.decorator_list] + [n.lineno]) for n in direct)
    end = first - 1
    while end > cls.lineno and not lines[end - 1].strip():
        end -= 1
    return end


def _extract(root: Path) -> list[_Sym]:
    out: list[_Sym] = []
    for f in walk_repo(root):
        try:
            tree = ast.parse(f.text)
        except SyntaxError:
            continue
        lines = f.text.split("\n")
        mod = module_name(f.path)

        def visit(body: list[ast.stmt], prefix: str, parent: str | None, in_class: bool,
                  _lines: list[str] = lines, _path: str = f.path) -> None:
            for node in body:
                if not isinstance(node, _DEFS):
                    for attr in ("body", "orelse", "finalbody"):
                        sub = getattr(node, attr, None)
                        if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                            visit(sub, prefix, parent, in_class)
                    for h in getattr(node, "handlers", []) or []:
                        visit(h.body, prefix, parent, in_class)
                    continue
                qname = f"{prefix}.{node.name}"
                is_class = isinstance(node, ast.ClassDef)
                start = min([d.lineno for d in node.decorator_list] + [node.lineno])
                end = node.end_lineno or node.lineno
                out.append(_Sym(
                    qname=qname, name=node.name,
                    kind="class" if is_class else ("method" if in_class else "function"),
                    path=_path, start=start, end=end, docstring=ast.get_docstring(node),
                    parent=parent, calls=set() if is_class else _calls(node),
                    header_end=_header_end(node, _lines) if is_class else end,
                ))
                visit(node.body, qname, qname, is_class)

        visit(tree.body, mod, None, False)
    return out


def first_sentence(doc: str) -> str:
    """First sentence of the first paragraph of a docstring, whitespace-collapsed."""
    para = " ".join(doc.strip().split("\n\n")[0].split())
    m = re.search(r"(.+?[.!?])(\s|$)", para)
    return (m.group(1) if m else para).strip()


def mask_docstring_sentence(sentence: str, name: str) -> str:
    """Drop every word sharing a sub-word with ``name`` (e.g. ``tax_rate`` -> tax, rate)."""
    parts = set(split_identifier(name))
    kept = []
    for word in sentence.split():
        subs = {p for w in _WORD_RE.findall(word) for p in split_identifier(w)}
        if not subs & parts:
            kept.append(word)
    out = " ".join(kept)
    if sentence.endswith(".") and out and not out.endswith("."):
        out = out.rstrip(",;:") + "."
    return out


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _candidates(syms: list[_Sym]) -> dict[str, list[Question]]:
    qcount = Counter(s.qname for s in syms)
    ncount = Counter(s.name for s in syms)
    by_q = {s.qname: s for s in syms}
    callers: dict[str, list[_Sym]] = {}
    for s in syms:
        for c in s.calls:
            callers.setdefault(c, []).append(s)
    out: dict[str, list[Question]] = {t: [] for t in DEFAULT_MIX}
    for s in syms:
        if qcount[s.qname] > 1 or _is_dunder(s.name):
            continue
        unique_name = ncount[s.name] == 1
        if unique_name:
            out["definition"].append(Question(
                id=f"definition:{s.qname}", type="definition",
                question=f"Where is `{s.name}` defined?", symbol=s.qname,
                truth=[(s.path, s.start, s.header_end if s.kind == "class" else s.end)],
            ))
        if s.kind != "class" and s.docstring:
            masked = mask_docstring_sentence(first_sentence(s.docstring), s.name)
            if len(_WORD_RE.findall(masked)) >= 4:
                out["docstring_masked"].append(Question(
                    id=f"docstring_masked:{s.qname}", type="docstring_masked",
                    question=masked, symbol=s.qname, truth=[(s.path, s.start, s.end)],
                ))
        if unique_name and s.kind != "class" and 1 <= len(callers.get(s.name, [])) <= 10:
            spans = sorted({(c.path, c.start, c.end) for c in callers[s.name]})
            out["callers"].append(Question(
                id=f"callers:{s.qname}", type="callers",
                question=f"Which functions call `{s.name}`?", symbol=s.qname, truth=spans,
                meta={"callers": sorted(c.qname for c in callers[s.name])},
            ))
    for qs in out.values():
        qs.sort(key=lambda q: q.id)
    assert all(q.symbol in by_q for qs in out.values() for q in qs)
    return out


def generate_questions(
    root: str | Path,
    n: int = 200,
    seed: int = 0,
    mix: dict[str, float] | None = None,
) -> list[Question]:
    """Deterministically sample about ``n`` questions across the three types."""
    mix = mix or DEFAULT_MIX
    pools = _candidates(_extract(Path(root)))
    rng = random.Random(seed)
    out: list[Question] = []
    for qtype in sorted(mix):
        pool = pools.get(qtype, [])
        want = min(len(pool), round(n * mix[qtype]))
        out.extend(rng.sample(pool, want))
    return out


def save_questions(questions: list[Question], path: str | Path) -> None:
    """Write questions as JSON."""
    Path(path).write_text(json.dumps([q.to_dict() for q in questions], indent=1) + "\n")

