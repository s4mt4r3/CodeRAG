"""BYO-LLM: CLI provider options (key never on argv) and eval provider selection."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from coderag import cli as cli_module
from coderag import evaluation
from coderag.cli import app
from coderag.llm import AnthropicClient, ScriptedLLM
from conftest import FIXTURE_ROOT

runner = CliRunner()
GOOD = json.dumps({"answer": "It slugifies text.", "citations": [
    {"path": "shopcore/utils.py", "start_line": 9, "end_line": 13, "symbol": "slugify"}]})
KEY = "sk-test-cli-0123456789SECRET"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    dst = tmp_path / "repo"
    shutil.copytree(FIXTURE_ROOT, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "CODERAG_LLM_KEY",
                "CODERAG_MODEL", "CODERAG_PROVIDER", "CODERAG_BASE_URL"]:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def fake_build(monkeypatch):
    calls = []

    def build(provider, model, key, base_url=None, **kw):
        calls.append((provider, model, key, base_url))
        return ScriptedLLM([ScriptedLLM.final(GOOD)])

    monkeypatch.setattr(cli_module, "build_client", build)
    monkeypatch.setattr(evaluation, "build_client", build)
    return calls


def test_ask_with_provider_reads_key_from_provider_env(repo, tmp_path, monkeypatch, fake_build):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    r = runner.invoke(app, ["ask", str(repo), "q", "--db", str(tmp_path / "i.db"),
                            "--retriever", "bm25", "--provider", "openai", "--model", "gpt-x"])
    assert r.exit_code == 0, r.output
    assert fake_build == [("openai", "gpt-x", KEY, None)]
    assert KEY not in r.output and "verified" in r.output


def test_coderag_llm_key_takes_precedence(repo, tmp_path, monkeypatch, fake_build):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-other-0000000000")
    monkeypatch.setenv("CODERAG_LLM_KEY", KEY)
    monkeypatch.setenv("CODERAG_MODEL", "gpt-env")
    r = runner.invoke(app, ["ask", str(repo), "q", "--db", str(tmp_path / "i.db"),
                            "--retriever", "bm25", "--provider", "openai_compatible",
                            "--base-url", "https://llm.example.com/v1"])
    assert r.exit_code == 0, r.output
    assert fake_build == [("openai_compatible", "gpt-env", KEY, "https://llm.example.com/v1")]


def test_missing_key_names_the_env_vars(repo, tmp_path, fake_build):
    r = runner.invoke(app, ["ask", str(repo), "q", "--db", str(tmp_path / "i.db"),
                            "--retriever", "bm25", "--provider", "gemini", "--model", "g"])
    assert r.exit_code == 2
    assert "GEMINI_API_KEY" in r.output and "CODERAG_LLM_KEY" in r.output
    assert fake_build == []


def test_missing_model_is_an_error(repo, tmp_path, monkeypatch, fake_build):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    r = runner.invoke(app, ["ask", str(repo), "q", "--db", str(tmp_path / "i.db"),
                            "--provider", "openai"])
    assert r.exit_code == 2 and "--model" in r.output


@pytest.mark.parametrize("flag", ["--api-key", "--key", "--llm-key"])
def test_key_cannot_be_passed_on_argv(repo, tmp_path, flag):
    r = runner.invoke(app, ["ask", str(repo), "q", flag, KEY])
    assert r.exit_code != 0


def test_bad_provider_reports_error_without_key(repo, tmp_path, monkeypatch):
    monkeypatch.setenv("CODERAG_LLM_KEY", KEY)
    r = runner.invoke(app, ["ask", str(repo), "q", "--db", str(tmp_path / "i.db"),
                            "--retriever", "bm25", "--provider", "nope", "--model", "m"])
    assert r.exit_code == 2 and KEY not in r.output


def test_llm_from_env_legacy_anthropic_path_unchanged(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-0000000000")
    monkeypatch.setenv("CODERAG_MODEL", "claude-x")
    client = evaluation.llm_from_env()
    assert isinstance(client, AnthropicClient) and client.model == "claude-x"


def test_llm_from_env_none_without_config():
    assert evaluation.llm_from_env() is None


def test_llm_from_env_provider_selection(monkeypatch, fake_build):
    monkeypatch.setenv("CODERAG_PROVIDER", "groq")
    monkeypatch.setenv("CODERAG_MODEL", "llama-x")
    monkeypatch.setenv("GROQ_API_KEY", KEY)
    assert evaluation.llm_from_env() is not None
    assert fake_build == [("groq", "llama-x", KEY, None)]


def test_report_records_provider_and_model(tmp_path):
    report = evaluation.run_eval(FIXTURE_ROOT, db_dir=tmp_path / "idx", n_questions=20,
                                 agent_n=3, embedder=None)
    assert report["agent"]["provider"] == "scripted"
    assert report["agent"]["model"] == "scripted"
    assert "pipeline check" in report["agent"]["label"].lower()

    class FakeReal(ScriptedLLM):
        provider, model = "openai", "gpt-x"

        def __init__(self):
            super().__init__([ScriptedLLM.final(GOOD) for _ in range(20)])

    report = evaluation.run_eval(FIXTURE_ROOT, db_dir=tmp_path / "idx", n_questions=20,
                                 agent_n=3, embedder=None, llm=FakeReal())
    assert (report["agent"]["provider"], report["agent"]["model"]) == ("openai", "gpt-x")
    assert report["agent"]["mode"] == "real"
    md = evaluation.render_markdown(report)
    assert "openai" in md and "None" not in md
