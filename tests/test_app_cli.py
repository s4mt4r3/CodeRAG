"""Web app: the `coderag app` launcher."""

from __future__ import annotations

from typer.testing import CliRunner

from coderag import cli as cli_module
from coderag.cli import app

runner = CliRunner()


def test_app_help():
    r = runner.invoke(app, ["app", "--help"])
    assert r.exit_code == 0 and "--port" in r.output


def test_launcher_passes_no_secrets_on_argv(monkeypatch):
    secret = "sk-test-LAUNCH0123456789abcd"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setenv("CODERAG_LLM_KEY", secret)
    seen = {}

    def fake_call(argv, **kwargs):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr(cli_module.subprocess, "call", fake_call)
    r = runner.invoke(app, ["app", "--port", "8599"])
    assert r.exit_code == 0, r.output
    argv = seen["argv"]
    assert "streamlit" in " ".join(argv) and "run" in argv
    assert argv[argv.index("--server.address") + 1] == "127.0.0.1"
    assert "8599" in argv
    assert secret not in " ".join(argv) and secret not in r.output
    assert "http://127.0.0.1:8599" in r.output


def test_launcher_without_streamlit(monkeypatch):
    monkeypatch.setattr(cli_module, "_streamlit_available", lambda: False)
    r = runner.invoke(app, ["app"])
    assert r.exit_code == 2 and ".[app]" in r.output
