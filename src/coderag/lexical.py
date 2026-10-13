"""Identifier-aware tokenization and BM25 search over the FTS5 index."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coderag.store import Hit, Store

_WORD_RE = re.compile(r"\w+")

# Generic English function words only; nothing tuned to a particular question template.
STOPWORDS = frozenset("""
a about above after again all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had
has have having he her here hers him his how i if in into is it its itself just me more most
my no nor not of off on once only or other our out over own same she should so some such
than that the their them then there these they this those through to too under until up
very was we were what when where which while who whom why will with would you your
""".split())


def split_identifier(word: str) -> list[str]:
    """Split snake_case, camelCase, PascalCase, ACRONYMCase and digit runs; lowercase."""
    parts: list[str] = []
    for piece in word.split("_"):
        cur = ""
        for i, ch in enumerate(piece):
            if cur:
                prev = cur[-1]
                nxt = piece[i + 1] if i + 1 < len(piece) else ""
                if (
                    ch.isdigit() != prev.isdigit()
                    or (prev.islower() and ch.isupper())
                    or (prev.isupper() and ch.isupper() and nxt.islower())
                ):
                    parts.append(cur)
                    cur = ""
            cur += ch
        if cur:
            parts.append(cur)
    return [p.lower() for p in parts]


def tokenize(text: str, query: bool = False) -> list[str]:
    """Tokens for indexing (with repeats) or for a query (deduplicated, no stopwords).

    Each identifier yields its lowercased original form plus its sub-words, so
    ``parseConfig``, ``parse_config`` and ``ParseConfig`` all match ``parse config``.
    """
    out: list[str] = []
    for word in _WORD_RE.findall(text):
        parts = split_identifier(word)
        low = word.lower()
        if parts != [low]:
            out.append(low)
        out.extend(parts)
    if query:
        seen: set[str] = set()
        deduped = []
        for t in out:
            if t not in STOPWORDS and t not in seen:
                seen.add(t)
                deduped.append(t)
        return deduped
    return out


def fts_query(query: str) -> str | None:
    """An FTS5 MATCH expression OR-ing the quoted query tokens, or None if empty."""
    toks = tokenize(query, query=True)
    if not toks:
        return None
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in toks)


def search_lexical(store: Store, query: str, k: int = 10) -> list[Hit]:
    """Top-``k`` chunks by BM25 (higher score is better)."""
    expr = fts_query(query)
    if expr is None or k <= 0:
        return []
    rows = store.conn.execute(
        "SELECT rowid, bm25(chunks_fts) AS r FROM chunks_fts WHERE chunks_fts MATCH ? "
        "ORDER BY r LIMIT ?",
        (expr, k),
    ).fetchall()
    return store.hits_for([(rowid, -rank) for rowid, rank in rows])
