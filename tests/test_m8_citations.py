"""M8: citation verification and the one-shot repair round."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from coderag.agent import Agent
from coderag.citations import verify
from coderag.llm import ScriptedLLM
from coderag.store import Store, index_repo
from coderag.tools import Toolbox
from conftest import FIXTURE_ROOT

ROOT = FIXTURE_ROOT


def cite(path="shopcore/utils.py", start=9, end=13, symbol="slugify"):
    return {"path": path, "start_line": start, "end_line": end, "symbol": symbol}


def test_valid_citation_passes():
    for sym in ("slugify", "shopcore.utils.slugify"):
        v = verify(cite(symbol=sym), ROOT)
        assert v.verified and v.reason == "ok", sym


def test_citing_part_of_a_symbol_body_passes():
    v = verify(cite(path="service/orders.py", start=38, end=46, symbol="checkout"), ROOT)
    assert v.verified, v.reason


def test_class_and_method_citations():
    assert verify(cite("shopcore/models.py", 10, 31, "Customer"), ROOT).verified
    assert verify(cite("shopcore/models.py", 19, 21, "Customer.display_name"), ROOT).verified


@pytest.mark.parametrize(("citation", "reason"), [
    (cite(path="shopcore/nope.py"), "file_not_found"),
    (cite(path="/etc/passwd"), "invalid_path"),
    (cite(path="../outside.py"), "invalid_path"),
    (cite(start=30, end=900), "lines_out_of_range"),
    (cite(start=0, end=5), "lines_out_of_range"),
    (cite(symbol="normalize_sku"), "symbol_not_in_range"),
    (cite(symbol="does_not_exist_anywhere"), "symbol_not_in_range"),
    (cite(symbol=""), "missing_symbol"),
    (cite(start=13, end=9), "reversed_range"),
    (cite(path="shopcore/models.py", start=1, end=79, symbol="Order"), "range_too_large"),
])
def test_bad_citations_fail_with_specific_reason(citation, reason):
    v = verify(citation, ROOT, max_lines=60)
    assert not v.verified
    assert v.reason == reason


def test_wrong_file_for_right_symbol():
    v = verify(cite(path="shopcore/pricing.py", start=9, end=13, symbol="slugify"), ROOT)
    assert not v.verified and v.reason == "symbol_not_in_range"


# ------------------------------------------------------------------ agent repair loop


@pytest.fixture
def toolbox(tmp_path: Path) -> Toolbox:
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURE_ROOT, repo, ignore=shutil.ignore_patterns("__pycache__"))
    db = tmp_path / "i.db"
    index_repo(repo, db)
    return Toolbox(Store(db))


def answer(*cites):
    return json.dumps({"answer": "slugify makes slugs.", "citations": list(cites)})


def test_repair_round_fixes_bad_citation(toolbox):
    llm = ScriptedLLM([
        ScriptedLLM.final(answer(cite(start=16, end=18))),  # wrong lines for slugify
        ScriptedLLM.final(answer(cite())),
    ])
    res = Agent(llm, toolbox).run("What does slugify do?")
    assert res.ok
    assert res.repair_rounds == 1
    assert res.citations[0]["verified"] is True and res.citations[0]["reason"] == "ok"
    assert res.unsupported is False
    repair_msg = llm.calls[1]["messages"][-1]["content"]
    assert "symbol_not_in_range" in repair_msg


def test_repair_never_exceeds_cap(toolbox):
    bad = answer(cite(start=16, end=18))
    llm = ScriptedLLM([ScriptedLLM.final(bad) for _ in range(5)])
    res = Agent(llm, toolbox).run("q")
    assert res.repair_rounds == 1
    assert len(llm.calls) == 2
    assert res.unsupported is True
    assert res.citations[0]["verified"] is False


def test_fabricated_citation_is_flagged_unsupported(toolbox):
    fake = cite(path="shopcore/payments.py", start=1, end=20, symbol="charge_card")
    llm = ScriptedLLM([ScriptedLLM.final(answer(fake)), ScriptedLLM.final(answer(fake))])
    res = Agent(llm, toolbox).run("How are cards charged?")
    assert res.ok  # an answer was produced ...
    assert res.unsupported is True  # ... but it is flagged, never presented as fine
    assert res.citations[0] == {**fake, "verified": False, "reason": "file_not_found"}


def test_no_citations_is_unsupported(toolbox):
    llm = ScriptedLLM([ScriptedLLM.final(answer()), ScriptedLLM.final(answer())])
    res = Agent(llm, toolbox).run("q")
    assert res.unsupported is True
    assert res.repair_rounds == 1


def test_partially_verified_answer_is_supported(toolbox):
    llm = ScriptedLLM([
        ScriptedLLM.final(answer(cite(), cite(start=16, end=18))),
        ScriptedLLM.final(answer(cite(), cite(start=16, end=18))),
    ])
    res = Agent(llm, toolbox).run("q")
    assert res.unsupported is False
    assert [c["verified"] for c in res.citations] == [True, False]


def test_good_first_answer_needs_no_repair(toolbox):
    llm = ScriptedLLM([ScriptedLLM.final(answer(cite()))])
    res = Agent(llm, toolbox).run("q")
    assert res.repair_rounds == 0 and len(llm.calls) == 1 and not res.unsupported
