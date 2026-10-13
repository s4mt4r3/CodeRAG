"""Freeze guard for test files and metric floors.

    python scripts/frozen.py [--root DIR] freeze   # record new entries, never change old ones
    python scripts/frozen.py [--root DIR] check    # fail on any drift from the manifest

The manifest lives at ``.frozen/manifest.json``. ``freeze`` also writes
``tests/FROZEN_MARKER`` so that ``check`` can tell a never-frozen project
(missing manifest is a warning) from one whose manifest was deleted (an error).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

MANIFEST = Path(".frozen") / "manifest.json"
MARKER = Path("tests") / "FROZEN_MARKER"
SELF = Path("scripts") / "frozen.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracked_files(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    tests = root / "tests"
    if tests.is_dir():
        for p in sorted(tests.rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            out[p.relative_to(root).as_posix()] = _sha256(p)
    if (root / SELF).is_file():
        out[SELF.as_posix()] = _sha256(root / SELF)
    return out


def _floors(root: Path) -> dict[str, float]:
    path = root / "thresholds.toml"
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return {k: float(v) for k, v in data.get("floors", {}).items()}


def _load(root: Path) -> dict | None:
    path = root / MANIFEST
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _problems(root: Path, manifest: dict, strict_floors: bool) -> list[str]:
    """Differences between the manifest and disk.

    With ``strict_floors`` any floor change is a problem (used by freeze); otherwise
    only a missing or lowered floor is (used by check, so floors may be raised).
    """
    problems: list[str] = []
    files = _tracked_files(root)
    for rel, entry in manifest.get("files", {}).items():
        if rel not in files:
            problems.append(f"frozen file missing: {rel}")
        elif files[rel] != entry["sha256"]:
            problems.append(f"frozen file changed: {rel}")
    floors = _floors(root)
    for name, entry in manifest.get("floors", {}).items():
        if name not in floors:
            problems.append(f"frozen floor removed: {name}")
        elif floors[name] < entry["value"]:
            problems.append(f"frozen floor lowered: {name} {entry['value']} -> {floors[name]}")
        elif strict_floors and floors[name] != entry["value"]:
            problems.append(f"frozen floor changed: {name} {entry['value']} -> {floors[name]}")
    return problems


def freeze(root: Path) -> int:
    """Record every new test file and floor; refuse if a recorded entry drifted."""
    manifest = _load(root) or {"files": {}, "floors": {}}
    problems = _problems(root, manifest, strict_floors=True)
    if problems:
        for p in problems:
            print(f"freeze refused: {p}", file=sys.stderr)
        return 1
    marker = root / MARKER
    if not marker.exists():
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("Written by scripts/frozen.py freeze. Do not delete.\n", encoding="utf-8")
    now = datetime.now(UTC).isoformat(timespec="seconds")
    added = 0
    for rel, digest in _tracked_files(root).items():
        if rel not in manifest["files"]:
            manifest["files"][rel] = {"sha256": digest, "frozen_at": now}
            added += 1
    for name, value in _floors(root).items():
        if name not in manifest["floors"]:
            manifest["floors"][name] = {"value": value, "frozen_at": now}
            added += 1
    path = root / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"froze {added} new entries "
        f"({len(manifest['files'])} files, {len(manifest['floors'])} floors total)"
    )
    return 0


def check(root: Path) -> int:
    """Exit non-zero if any frozen file or floor is missing, changed, or lowered."""
    manifest = _load(root)
    if manifest is None:
        if (root / MARKER).exists():
            print("frozen check FAILED: manifest missing after a freeze", file=sys.stderr)
            return 1
        print("warning: no frozen manifest yet (nothing frozen); passing")
        return 0
    problems = _problems(root, manifest, strict_floors=False)
    if problems:
        for p in problems:
            print(f"frozen check FAILED: {p}", file=sys.stderr)
        return 1
    print(
        f"frozen check OK ({len(manifest['files'])} files, "
        f"{len(manifest['floors'])} floors)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("command", choices=["freeze", "check"])
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    return freeze(root) if args.command == "freeze" else check(root)


if __name__ == "__main__":
    sys.exit(main())
