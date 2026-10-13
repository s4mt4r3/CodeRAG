"""README number and claim audit.

    python scripts/audit.py [--readme README.md] [--reports reports]

Fails (exit 1) when:
* a number in the README prose or tables does not appear in any ``reports/`` file
  (decimals and percentages are matched at the precision written in the README;
  integers below 10, numbers inside identifiers such as ``M10`` or ``v1.5``, code
  blocks, inline code, and URLs are not audited), or
* the README states an improvement for a claim that ``reports/claims.json`` records
  as failed (a sentence naming the claim's subject with an improvement verb and no
  negation).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_NUM = re.compile(r"(?<![\w.\-/])(\d+(?:\.\d+)?)(%?)(?![\w]|\.\d)")
_IMPROVE = re.compile(r"\b(beat|beats|outperform\w*|improv\w*|better than|boost\w*|"
                      r"superior|wins?|higher than)\b")
_NEGATE = re.compile(r"\b(not|no|never|fail\w*|worse|lower|without|n't)\b|n't\b")


def _strip_code(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    return re.sub(r"\]\([^)]*\)", "]", text)


def _collect_values(obj: object, out: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, int | float):
        out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(float(m.group(1)) for m in _NUM.finditer(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _collect_values(k, out)
            _collect_values(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_values(v, out)


def report_values(reports: Path) -> list[float]:
    """Every number found in report files (JSON values, and numbers in text files)."""
    values: list[float] = []
    for p in sorted(reports.rglob("*")):
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if p.suffix == ".json":
            try:
                _collect_values(json.loads(text), values)
                continue
            except json.JSONDecodeError:
                pass
        _collect_values(text, values)
    return values


def _matches(token: str, values: list[float]) -> bool:
    decimals = len(token.split(".")[1]) if "." in token else 0
    target = float(token)
    if decimals == 0:
        return any(v == target for v in values)
    return any(round(v, decimals) == round(target, decimals) for v in values)


def number_problems(readme_text: str, values: list[float]) -> list[str]:
    """README numbers with no matching value in the reports."""
    problems = []
    for lineno, line in enumerate(_strip_code(readme_text).splitlines(), start=1):
        for m in _NUM.finditer(line):
            token, pct = m.group(1), m.group(2)
            if "." not in token and not pct and int(token) < 10:
                continue
            if pct:
                places = (len(token.split(".")[1]) if "." in token else 0) + 2
                as_frac = f"{float(token) / 100:.{places}f}"
                ok = _matches(token, values) or _matches(as_frac, values)
            else:
                ok = _matches(token, values)
            if not ok:
                problems.append(f"README line {lineno}: number {token}{pct} not found in reports/")
    return problems


def failed_claims(reports: Path) -> list[tuple[str, str, str]]:
    """(claim key, subject, object) for each claim recorded as failed."""
    path = reports / "claims.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    out = []
    for key, d in (data.get("details") or {}).items():
        if d.get("passed") is False and "_beats_" in key:
            subject, rest = key.split("_beats_", 1)
            out.append((key, subject.replace("_", " "), rest.split("_")[0]))
    return out


def claim_problems(readme_text: str, reports: Path) -> list[str]:
    """Sentences that assert an improvement recorded as a failed claim."""
    problems = []
    sentences = re.split(r"(?<=[.!?])\s+|\n", _strip_code(readme_text))
    for key, subject, _obj in failed_claims(reports):
        subj = re.compile(r"\b" + re.escape(subject) + r"\b", re.I)
        for s in sentences:
            low = s.lower()
            if subj.search(s) and _IMPROVE.search(low) and not _NEGATE.search(low):
                problems.append(f"README asserts failed claim {key!r}: {s.strip()[:120]}")
    return problems


def audit(readme: Path, reports: Path) -> list[str]:
    """All audit problems for ``readme`` against ``reports``."""
    text = readme.read_text(encoding="utf-8")
    return number_problems(text, report_values(reports)) + claim_problems(text, reports)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exit 0 when the README passes."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    ap.add_argument("--reports", default=str(ROOT / "reports"))
    args = ap.parse_args(argv)
    problems = audit(Path(args.readme), Path(args.reports))
    for p in problems:
        print(p)
    print("audit OK" if not problems else f"audit FAILED ({len(problems)} problems)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
