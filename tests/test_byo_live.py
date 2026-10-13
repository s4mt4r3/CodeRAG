"""BYO-LLM (live, optional): one real question per provider, each gated on its own key.

Set the provider key and a model, e.g. OPENAI_API_KEY + CODERAG_LIVE_OPENAI_MODEL.
"""

from __future__ import annotations

import os

import pytest

from coderag.agent import Agent
from coderag.providers.registry import build_client
from coderag.store import Store, index_repo
from coderag.tools import Toolbox
from conftest import FIXTURE_ROOT

CASES = [
    ("anthropic", "ANTHROPIC_API_KEY", "CODERAG_LIVE_ANTHROPIC_MODEL"),
    ("openai", "OPENAI_API_KEY", "CODERAG_LIVE_OPENAI_MODEL"),
    ("gemini", "GEMINI_API_KEY", "CODERAG_LIVE_GEMINI_MODEL"),
    ("groq", "GROQ_API_KEY", "CODERAG_LIVE_GROQ_MODEL"),
    ("together", "TOGETHER_API_KEY", "CODERAG_LIVE_TOGETHER_MODEL"),
    ("openrouter", "OPENROUTER_API_KEY", "CODERAG_LIVE_OPENROUTER_MODEL"),
]


@pytest.mark.live
@pytest.mark.parametrize(("provider", "key_env", "model_env"), CASES)
def test_live_provider_answers_with_valid_schema(provider, key_env, model_env, tmp_path):
    key, model = os.environ.get(key_env), os.environ.get(model_env)
    if not key or not model:
        pytest.skip(f"{key_env} and {model_env} not both set")
    db = tmp_path / "i.db"
    index_repo(FIXTURE_ROOT, db)
    res = Agent(build_client(provider, model, key), Toolbox(Store(db))).run(
        "Where is normalize_sku defined?")
    assert res.ok, res.failure
    for c in res.citations:
        assert {"path", "start_line", "end_line", "symbol", "verified"} <= set(c)
