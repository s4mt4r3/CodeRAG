"""Walk a repository and return the Python source files worth indexing."""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SKIP_DIRS = frozenset({".git", "venv", ".venv", "node_modules", "__pycache__"})
DEFAULT_MAX_BYTES = 1_000_000
IGNORE_FILE = ".coderagignore"
LANGUAGE_SUFFIXES = (".py",)


@dataclass(frozen=True)
class SourceFile:
    """A source file: POSIX path relative to the repo root, and its text."""

    path: str
    text: str


def _read_ignore_file(root: Path) -> list[str]:
    path = root / IGNORE_FILE
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def is_ignored(rel_path: str, patterns: Sequence[str]) -> bool:
    """True if ``rel_path`` or any of its parent directories matches a glob pattern."""
    parts = rel_path.split("/")
    prefixes = ["/".join(parts[: i + 1]) for i in range(len(parts))]
    for pat in patterns:
        pat = pat.rstrip("/")
        for candidate in prefixes + parts:
            if fnmatch.fnmatchcase(candidate, pat):
                return True
    return False


def walk_repo(
    root: str | Path,
    ignore: Sequence[str] = (),
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[SourceFile]:
    """Return indexable Python files under ``root``, sorted by relative path.

    Skips well-known vendor/VCS dirs, files over ``max_bytes``, files that are not
    valid UTF-8, and anything matching ``ignore`` or the root's ``.coderagignore``.
    """
    root = Path(root).resolve()
    patterns = [*ignore, *_read_ignore_file(root)]
    out: list[SourceFile] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in DEFAULT_SKIP_DIRS
            and not is_ignored(f"{rel_dir}/{d}" if rel_dir else d, patterns)
        )
        for name in sorted(filenames):
            if not name.endswith(LANGUAGE_SUFFIXES):
                continue
            rel = f"{rel_dir}/{name}" if rel_dir else name
            if is_ignored(rel, patterns):
                continue
            full = Path(dirpath) / name
            if full.is_symlink() or not full.is_file():
                continue
            if full.stat().st_size > max_bytes:
                continue
            try:
                text = full.read_bytes().decode("utf-8")
            except UnicodeDecodeError:
                continue
            out.append(SourceFile(path=rel, text=text))
    out.sort(key=lambda f: f.path)
    return out
