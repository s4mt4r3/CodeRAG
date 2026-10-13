"""M7 (live, optional): one real question against the fixture repo."""

from __future__ import annotations

import os

import pytest

from coderag.agent import Agent
from coderag.llm import AnthropicClient
from coderag.store import Store, index_repo
from coderag.tools import Toolbox
from conftest import FIXTURE_ROOT


@pytest.mark.live
def test_live_agent_returns_valid_schema(tmp_path):
    if not os.environ.get("CODERAG_MODEL"):
        pytest.skip("CODERAG_MODEL not set")
    db = tmp_path / "i.db"
    index_repo(FIXTURE_ROOT, db)
    res = Agent(AnthropicClient(), Toolbox(Store(db))).run("Where is normalize_sku defined?")
    assert res.ok, res.failure
    assert isinstance(res.answer, str) and res.answer
    for c in res.citations:
        assert {"path", "start_line", "end_line", "symbol"} <= set(c)
