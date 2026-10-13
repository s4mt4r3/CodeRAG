"""SQLite index: files, chunks, symbols, call edges, FTS5, and an embedding cache.

``index_repo`` is idempotent and incremental: files are keyed by content hash, and
only files whose hash changed have their rows replaced.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from coderag.chunking import DEFAULT_MAX_TOKENS, Chunk, ast_chunks, fixed_chunks
from coderag.ingest import SourceFile, walk_repo
from coderag.lexical import tokenize
from coderag.symbols import Symbol, parse_source

SCHEMA_VERSION = "1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY, content_hash TEXT NOT NULL, size INTEGER NOT NULL,
    n_lines INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    rowid INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE NOT NULL, path TEXT NOT NULL,
    symbol TEXT NOT NULL, kind TEXT NOT NULL, start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL, segments TEXT NOT NULL, part INTEGER NOT NULL,
    parts INTEGER NOT NULL, partial INTEGER NOT NULL, code TEXT NOT NULL,
    text TEXT NOT NULL, content_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY, path TEXT NOT NULL, qualified_name TEXT NOT NULL,
    name TEXT NOT NULL, kind TEXT NOT NULL, start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL, docstring TEXT, decorators TEXT NOT NULL,
    is_async INTEGER NOT NULL, parent TEXT
);
CREATE INDEX IF NOT EXISTS symbols_path ON symbols(path);
CREATE INDEX IF NOT EXISTS symbols_name ON symbols(name);
CREATE TABLE IF NOT EXISTS edges (
    id INTEGER PRIMARY KEY, caller TEXT NOT NULL, callee TEXT, callee_name TEXT NOT NULL,
    expr TEXT NOT NULL, path TEXT NOT NULL, line INTEGER NOT NULL, confidence TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS edges_caller ON edges(caller);
CREATE INDEX IF NOT EXISTS edges_callee ON edges(callee);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED, body, tokenize = "unicode61 tokenchars '_'"
);
CREATE TABLE IF NOT EXISTS embeddings (
    content_hash TEXT NOT NULL, model TEXT NOT NULL, dim INTEGER NOT NULL,
    vector BLOB NOT NULL, PRIMARY KEY (content_hash, model)
);
"""

_CHUNK_COLS = (
    "rowid, chunk_id, path, symbol, kind, start_line, end_line, segments, part, parts, "
    "partial, code, text, content_hash"
)


def sha256_text(text: str) -> str:
    """Hex sha256 of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class ChunkRow:
    """A chunk as stored in SQLite."""

    rowid: int
    chunk_id: str
    path: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    segments: list[tuple[int, int]]
    part: int
    parts: int
    partial: bool
    code: str
    text: str
    content_hash: str


@dataclass
class Hit:
    """A ranked retrieval result."""

    chunk_id: str
    path: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    segments: list[tuple[int, int]]
    score: float
    rowid: int = 0


@dataclass
class IndexStats:
    """What an ``index_repo`` call changed."""

    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    chunks: int = 0
    warnings: list[str] = field(default_factory=list)


def _row_to_chunk(r: tuple) -> ChunkRow:
    return ChunkRow(
        rowid=r[0], chunk_id=r[1], path=r[2], symbol=r[3], kind=r[4], start_line=r[5],
        end_line=r[6], segments=[tuple(s) for s in json.loads(r[7])], part=r[8], parts=r[9],
        partial=bool(r[10]), code=r[11], text=r[12], content_hash=r[13],
    )


def _row_to_symbol(r: tuple) -> Symbol:
    return Symbol(
        kind=r[3], name=r[2], qualified_name=r[1], path=r[0], start_line=r[4], end_line=r[5],
        docstring=r[6], decorators=json.loads(r[7]), is_async=bool(r[8]), parent=r[9],
    )


_SYMBOL_COLS = (
    "path, qualified_name, name, kind, start_line, end_line, docstring, decorators, "
    "is_async, parent"
)


class Store:
    """Read access to an index database (plus the writes ``index_repo`` needs)."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        """Close the connection."""
        self.conn.close()

    def meta(self, key: str) -> str | None:
        """A value from the meta table."""
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        """Upsert a value in the meta table."""
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    @property
    def root(self) -> str | None:
        """Absolute path of the indexed repository root."""
        return self.meta("root")

    def get_chunk(self, chunk_id: str) -> ChunkRow | None:
        """One chunk by id."""
        r = self.conn.execute(
            f"SELECT {_CHUNK_COLS} FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return _row_to_chunk(r) if r else None

    def all_chunks(self) -> list[ChunkRow]:
        """Every chunk ordered by rowid."""
        rows = self.conn.execute(f"SELECT {_CHUNK_COLS} FROM chunks ORDER BY rowid").fetchall()
        return [_row_to_chunk(r) for r in rows]

    def hits_for(self, scored: list[tuple[int, float]]) -> list[Hit]:
        """Materialize (rowid, score) pairs into hits, preserving order."""
        if not scored:
            return []
        ids = [rid for rid, _ in scored]
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT {_CHUNK_COLS} FROM chunks WHERE rowid IN ({marks})", ids
        ).fetchall()
        by_id = {r[0]: _row_to_chunk(r) for r in rows}
        out = []
        for rid, score in scored:
            c = by_id.get(rid)
            if c is not None:
                out.append(Hit(c.chunk_id, c.path, c.symbol, c.kind, c.start_line,
                               c.end_line, c.segments, float(score), c.rowid))
        return out

    def find_symbols(self, name: str, limit: int = 50) -> list[Symbol]:
        """Symbols whose simple or qualified name equals ``name`` or ends with ``.name``."""
        rows = self.conn.execute(
            f"SELECT {_SYMBOL_COLS} FROM symbols WHERE name = ? OR qualified_name = ? "
            "OR qualified_name LIKE ? ESCAPE '\\' ORDER BY path, start_line LIMIT ?",
            (name, name, "%." + name.replace("\\", "\\\\").replace("%", "\\%")
             .replace("_", "\\_"), limit),
        ).fetchall()
        return [_row_to_symbol(r) for r in rows]

    # ------------------------------------------------------------------ writes

    def _delete_path(self, path: str) -> None:
        self.conn.execute(
            "DELETE FROM chunks_fts WHERE rowid IN (SELECT rowid FROM chunks WHERE path = ?)",
            (path,),
        )
        for table in ("chunks", "symbols", "edges", "files"):
            self.conn.execute(f"DELETE FROM {table} WHERE path = ?", (path,))

    def _wipe(self) -> None:
        for table in ("chunks_fts", "chunks", "symbols", "edges", "files"):
            self.conn.execute(f"DELETE FROM {table}")

    def _insert_chunk(self, c: Chunk) -> None:
        cur = self.conn.execute(
            "INSERT INTO chunks(chunk_id, path, symbol, kind, start_line, end_line, segments, "
            "part, parts, partial, code, text, content_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (c.chunk_id, c.path, c.symbol, c.kind, c.start_line, c.end_line,
             json.dumps(c.segments), c.part, c.parts, int(c.partial), c.code, c.text,
             sha256_text(c.text)),
        )
        self.conn.execute(
            "INSERT INTO chunks_fts(rowid, chunk_id, body) VALUES (?, ?, ?)",
            (cur.lastrowid, c.chunk_id, " ".join(tokenize(c.text))),
        )

    def _insert_symbol(self, s: Symbol) -> None:
        self.conn.execute(
            f"INSERT INTO symbols({_SYMBOL_COLS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (s.path, s.qualified_name, s.name, s.kind, s.start_line, s.end_line, s.docstring,
             json.dumps(s.decorators), int(s.is_async), s.parent),
        )


def index_repo(
    root: str | Path,
    db_path: str | Path,
    chunker: str = "ast",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    fixed_size: int = 40,
    fixed_overlap: int = 10,
    ignore: tuple[str, ...] = (),
) -> IndexStats:
    """Index ``root`` into ``db_path``; unchanged files are left untouched."""
    if chunker not in ("ast", "fixed"):
        raise ValueError(f"unknown chunker {chunker!r}")
    root = Path(root).resolve()
    settings = json.dumps({
        "schema": SCHEMA_VERSION, "chunker": chunker, "max_tokens": max_tokens,
        "fixed_size": fixed_size, "fixed_overlap": fixed_overlap, "ignore": list(ignore),
    }, sort_keys=True)
    store = Store(db_path)
    stats = IndexStats()
    try:
        with store.conn:
            if store.meta("settings") != settings or store.meta("root") != str(root):
                store._wipe()
                store.set_meta("settings", settings)
                store.set_meta("root", str(root))
            known = dict(store.conn.execute("SELECT path, content_hash FROM files").fetchall())
            seen: set[str] = set()
            changed = False
            files = walk_repo(root, ignore=ignore)
            for f in files:
                seen.add(f.path)
                digest = sha256_text(f.text)
                if known.get(f.path) == digest:
                    stats.unchanged += 1
                    continue
                if f.path in known:
                    stats.updated += 1
                    store._delete_path(f.path)
                else:
                    stats.added += 1
                changed = True
                parsed = parse_source(f.path, f.text)
                stats.warnings.extend(parsed.warnings)
                for s in parsed.symbols:
                    store._insert_symbol(s)
                chunks = (
                    ast_chunks(f, max_tokens=max_tokens) if chunker == "ast"
                    else fixed_chunks(f, size=fixed_size, overlap=fixed_overlap)
                )
                for c in chunks:
                    store._insert_chunk(c)
                store.conn.execute(
                    "INSERT INTO files(path, content_hash, size, n_lines) VALUES (?, ?, ?, ?)",
                    (f.path, digest, len(f.text.encode("utf-8")), f.text.count("\n") + 1),
                )
            for path in sorted(set(known) - seen):
                store._delete_path(path)
                stats.removed += 1
                changed = True
            if changed:
                _after_change(store, files)
        stats.chunks = store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        store.close()
    return stats


def _after_change(store: Store, files: list[SourceFile]) -> None:
    """Recompute whole-index derived data: call edges depend on every file."""
    from coderag.callgraph import write_edges

    write_edges(store, files)
