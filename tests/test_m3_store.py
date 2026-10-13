"""M3: SQLite store, idempotent and incremental indexing."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from coderag.lexical import search_lexical
from coderag.store import Store, index_repo
from conftest import FIXTURE_ROOT


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    dst = tmp_path / "repo"
    shutil.copytree(FIXTURE_ROOT, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def _snapshot(db: Path) -> dict:
    con = sqlite3.connect(db)
    snap = {
        "files": con.execute("SELECT path, content_hash FROM files ORDER BY path").fetchall(),
        "chunks": con.execute(
            "SELECT rowid, chunk_id, content_hash FROM chunks ORDER BY rowid"
        ).fetchall(),
        "symbols": con.execute("SELECT COUNT(*) FROM symbols").fetchone(),
        "fts": con.execute("SELECT COUNT(*) FROM chunks_fts").fetchone(),
    }
    con.close()
    return snap


def test_schema_has_required_tables(repo: Path, tmp_path: Path):
    db = tmp_path / "i.db"
    index_repo(repo, db)
    con = sqlite3.connect(db)
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master")}
    assert {"files", "chunks", "symbols", "edges", "chunks_fts"} <= names
    assert con.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 13
    assert con.execute("SELECT COUNT(*) FROM symbols").fetchone()[0] == 69


def test_reindex_unchanged_repo_changes_nothing(repo: Path, tmp_path: Path):
    db = tmp_path / "i.db"
    first = index_repo(repo, db)
    snap = _snapshot(db)
    second = index_repo(repo, db)
    assert _snapshot(db) == snap
    assert first.added == 13 and second.added == 0
    assert second.updated == 0 and second.removed == 0 and second.unchanged == 13


def test_modified_file_replaces_only_its_chunks(repo: Path, tmp_path: Path):
    db = tmp_path / "i.db"
    index_repo(repo, db)
    before = _snapshot(db)
    target = repo / "shopcore" / "utils.py"
    target.write_text(
        target.read_text() + "\n\ndef shout(text: str) -> str:\n    return text.upper()\n"
    )
    stats = index_repo(repo, db)
    assert stats.updated == 1 and stats.unchanged == 12
    after = _snapshot(db)
    keep_before = [r for r in before["chunks"] if not r[1].startswith("shopcore/utils.py:")]
    keep_after = [r for r in after["chunks"] if not r[1].startswith("shopcore/utils.py:")]
    assert keep_before == keep_after, "other files' chunk rows must be untouched"
    assert any("shopcore.utils.shout" in r[1] for r in after["chunks"])
    store = Store(db)
    assert "shopcore.utils.shout" in [h.symbol for h in search_lexical(store, "shout", k=3)]


def test_deleted_file_is_removed_and_never_returned(repo: Path, tmp_path: Path):
    db = tmp_path / "i.db"
    index_repo(repo, db)
    store = Store(db)
    assert any(h.path == "service/workers.py" for h in search_lexical(store, "JobRunner", k=5))
    (repo / "service" / "workers.py").unlink()
    stats = index_repo(repo, db)
    assert stats.removed == 1
    con = sqlite3.connect(db)
    for table in ("files", "chunks", "symbols"):
        n = con.execute(f"SELECT COUNT(*) FROM {table} WHERE path = ?",
                        ("service/workers.py",)).fetchone()[0]
        assert n == 0, table
    store = Store(db)
    for q in ["JobRunner", "retry async", "cached tax rate", "send daily digest"]:
        assert all(h.path != "service/workers.py" for h in search_lexical(store, q, k=50))


def test_store_lookups(repo: Path, tmp_path: Path):
    db = tmp_path / "i.db"
    index_repo(repo, db)
    store = Store(db)
    syms = store.find_symbols("validate")
    assert {s.qualified_name for s in syms} == {
        "storage.serializers.validate", "service.api_handlers.validate",
    }
    c = store.get_chunk("shopcore/utils.py:9:shopcore.utils.slugify#0")
    assert c is not None and c.start_line == 9 and "def slugify" in c.code
    assert store.root == str(repo.resolve())


def test_changing_chunker_settings_rebuilds(repo: Path, tmp_path: Path):
    db = tmp_path / "i.db"
    index_repo(repo, db, chunker="ast")
    stats = index_repo(repo, db, chunker="fixed")
    assert stats.added == 13
    con = sqlite3.connect(db)
    kinds = {r[0] for r in con.execute("SELECT DISTINCT kind FROM chunks")}
    assert kinds == {"fixed"}
