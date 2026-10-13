"""M9: typer CLI and FastAPI app on the fixture repo (offline, scripted LLM)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from coderag.api import create_app
from coderag.cli import app
from coderag.llm import ScriptedLLM
from conftest import FIXTURE_ROOT

GOOD = {"answer": "It slugifies text.",
        "citations": [{"path": "shopcore/utils.py", "start_line": 9, "end_line": 13,
                       "symbol": "slugify"}]}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    dst = tmp_path / "allowed" / "repo"
    shutil.copytree(FIXTURE_ROOT, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


@pytest.fixture
def no_llm_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CODERAG_MODEL", raising=False)


# ------------------------------------------------------------------ CLI

runner = CliRunner()


def test_cli_index(repo: Path, tmp_path: Path):
    db = tmp_path / "i.db"
    r = runner.invoke(app, ["index", str(repo), "--db", str(db)])
    assert r.exit_code == 0, r.output
    assert "13 files" in r.output and db.exists()
    r2 = runner.invoke(app, ["index", str(repo), "--db", str(db)])
    assert "13 unchanged" in r2.output


def test_cli_ask_with_scripted_llm(repo: Path, tmp_path: Path, no_llm_env):
    db = tmp_path / "i.db"
    script = tmp_path / "script.json"
    script.write_text(json.dumps([
        {"tool": "search_code", "arguments": {"query": "slugify"}},
        {"final": json.dumps(GOOD)},
    ]))
    r = runner.invoke(app, ["ask", str(repo), "What does slugify do?", "--db", str(db),
                            "--scripted", str(script), "--retriever", "bm25"])
    assert r.exit_code == 0, r.output
    assert "It slugifies text." in r.output
    assert "shopcore/utils.py:9-13" in r.output and "verified" in r.output


def test_cli_ask_flags_unverified_citations(repo: Path, tmp_path: Path, no_llm_env):
    bad = {"answer": "x", "citations": [{"path": "nope.py", "start_line": 1, "end_line": 2,
                                         "symbol": "f"}]}
    script = tmp_path / "s.json"
    script.write_text(json.dumps([{"final": json.dumps(bad)}, {"final": json.dumps(bad)}]))
    r = runner.invoke(app, ["ask", str(repo), "q", "--db", str(tmp_path / "i.db"),
                            "--scripted", str(script), "--retriever", "bm25"])
    assert r.exit_code == 0, r.output
    assert "UNSUPPORTED" in r.output and "file_not_found" in r.output


def test_cli_ask_without_llm_config_fails_clearly(repo: Path, tmp_path: Path, no_llm_env):
    r = runner.invoke(app, ["ask", str(repo), "q", "--db", str(tmp_path / "i.db"),
                            "--retriever", "bm25"])
    assert r.exit_code != 0
    assert "ANTHROPIC_API_KEY" in r.output


def test_cli_eval_writes_reports(repo: Path, tmp_path: Path, no_llm_env):
    out = tmp_path / "reports"
    r = runner.invoke(app, ["eval", str(repo), "--out", str(out), "--db-dir",
                            str(tmp_path / "idx"), "--n", "40", "--agent-n", "8",
                            "--no-embeddings"])
    assert r.exit_code == 0, r.output
    data = json.loads((out / "eval.json").read_text())
    assert data["agent"]["mode"] == "scripted"
    assert (out / "eval.md").exists()
    assert (out / "ablation_chunking.json").exists() and (out / "ablation_chunking.md").exists()


# ------------------------------------------------------------------ API


@pytest.fixture
def client(repo: Path, tmp_path: Path) -> TestClient:
    def llm_factory():
        return ScriptedLLM([ScriptedLLM.final(json.dumps(GOOD))])

    application = create_app(allowed_roots=[repo.parent], db_dir=tmp_path / "dbs",
                             llm_factory=llm_factory)
    return TestClient(application)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_index_and_ask(client, repo):
    r = client.post("/index", json={"repo": str(repo)})
    assert r.status_code == 200, r.text
    assert r.json()["files"] == 13
    r = client.post("/ask", json={"repo": str(repo), "question": "What does slugify do?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == GOOD["answer"]
    assert body["citations"][0]["verified"] is True
    assert body["unsupported"] is False


def test_repo_outside_allowlist_is_rejected(client, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    for path in [str(outside), "/etc", str(tmp_path / "allowed" / ".." / "elsewhere")]:
        r = client.post("/index", json={"repo": path})
        assert r.status_code == 403, (path, r.text)
    r = client.post("/ask", json={"repo": str(outside), "question": "q"})
    assert r.status_code == 403


def test_invalid_json_body(client):
    r = client.post("/index", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422
    r = client.post("/ask", json={"repo": 5})
    assert r.status_code == 422
    r = client.post("/ask", json={"repo": "x", "question": ""})
    assert r.status_code == 422


def test_ask_before_index_is_conflict(client, repo):
    r = client.post("/ask", json={"repo": str(repo), "question": "q"})
    assert r.status_code == 409


def test_ask_without_llm_is_service_unavailable(repo, tmp_path, no_llm_env):
    c = TestClient(create_app(allowed_roots=[repo.parent], db_dir=tmp_path / "d"))
    assert c.post("/index", json={"repo": str(repo)}).status_code == 200
    assert c.post("/ask", json={"repo": str(repo), "question": "q"}).status_code == 503
