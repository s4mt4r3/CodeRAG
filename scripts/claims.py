"""Run non-gating ``claim`` tests and record each outcome in reports/claims.json.

Claims never block the gate. A failing claim is recorded as failed, never hidden.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def main() -> int:
    """Run ``pytest -m claim`` and write reports/claims.json (always exits 0)."""
    REPORTS.mkdir(exist_ok=True)
    # The junit XML records the machine's hostname, so it stays out of reports/.
    xml_path = Path(tempfile.mkdtemp(prefix="coderag-claims-")) / "claims_junit.xml"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-m", "claim", "-q", f"--junitxml={xml_path}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    print(proc.stdout[-3000:])
    claims = []
    if xml_path.exists():
        for case in ET.parse(xml_path).getroot().iter("testcase"):
            failure = case.find("failure")
            if failure is None:
                failure = case.find("error")
            skipped = case.find("skipped")
            outcome = "failed" if failure is not None else "skipped" if skipped is not None \
                else "passed"
            entry = {"id": f"{case.get('classname')}::{case.get('name')}", "outcome": outcome}
            if failure is not None:
                entry["message"] = (failure.get("message") or "")[:500]
            claims.append(entry)
    details_path = REPORTS / "claims_details.json"
    details = json.loads(details_path.read_text()) if details_path.exists() else {}
    out = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pytest_exit_code": proc.returncode,
        "claims": claims,
        "details": details,
    }
    (REPORTS / "claims.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote reports/claims.json ({len(claims)} claims)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
