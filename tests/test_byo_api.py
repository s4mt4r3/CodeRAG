"""BYO-LLM: per-request credentials, key scrubbing, auth, limits, HTTPS on the API."""

from __future__ import annotations

import json
import logging
import shutil
import threading
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from coderag import api as api_module
from coderag.api import create_app
from coderag.apiguard import ApiSettings
from coderag.llm import ScriptedLLM
from coderag.providers.openai_compat import OpenAICompatClient
from conftest import FIXTURE_ROOT

PATTERN_KEY = "sk-test-ABCDEFGH0123456789zzzz"
PLAIN_KEY = "zz9plainSECRETvalue42"  # matches no key pattern; only exact scrubbing hides it
GOOD = json.dumps({"answer": "It slugifies text.", "citations": [
    {"path": "shopcore/utils.py", "start_line": 9, "end_line": 13, "symbol": "slugify"}]})


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    dst = tmp_path / "allowed" / "repo"
    shutil.copytree(FIXTURE_ROOT, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ["ANTHROPIC_API_KEY", "CODERAG_MODEL", "CODERAG_PROVIDER", "CODERAG_API_TOKEN",
                "CODERAG_BYO_LLM_ONLY", "CODERAG_REQUIRE_HTTPS", "CODERAG_TRUST_PROXY",
                "CODERAG_ALLOW_INSECURE_LOCAL", "CODERAG_PROVIDER_HOSTS"]:
        monkeypatch.delenv(var, raising=False)


class Factory:
    """Records how the API built the client; returns a scripted or adapter client."""

    def __init__(self, make=None):
        self.calls = []
        self.make = make or (lambda *a: ScriptedLLM([ScriptedLLM.final(GOOD)]))

    def __call__(self, provider, model, key, base_url):
        self.calls.append((provider, model, key, base_url))
        return self.make(provider, model, key, base_url)


def make_client(repo, tmp_path, settings=None, factory=None, llm_factory=None, clock=None,
                raise_server_exceptions=True) -> TestClient:
    kwargs = {"allowed_roots": [repo.parent], "db_dir": tmp_path / "dbs",
              "llm_factory": llm_factory, "settings": settings or ApiSettings(),
              "client_factory": factory or Factory()}
    if clock is not None:
        kwargs["clock"] = clock
    app = create_app(**kwargs)
    c = TestClient(app, raise_server_exceptions=raise_server_exceptions)
    assert c.post("/index", json={"repo": str(repo)}).status_code == 200
    return c


def byo(key=PATTERN_KEY, provider="openai", model="gpt-test", base_url=None):
    h = {"X-LLM-Provider": provider, "X-LLM-Model": model}
    if key is not None:
        h["X-LLM-Key"] = key
    if base_url:
        h["X-LLM-Base-URL"] = base_url
    return h


def ask(c, repo, headers=None, **body):
    return c.post("/ask", json={"repo": str(repo), "question": "What does slugify do?", **body},
                  headers=headers or {})


# ------------------------------------------------------------------ credentials


def test_byo_headers_build_a_per_request_client(repo, tmp_path):
    f = Factory()
    c = make_client(repo, tmp_path, factory=f)
    r = ask(c, repo, byo())
    assert r.status_code == 200, r.text
    assert r.json()["citations"][0]["verified"] is True
    assert f.calls == [("openai", "gpt-test", PATTERN_KEY, None)]
    assert PATTERN_KEY not in r.text


def test_server_fallback_without_headers_still_works(repo, tmp_path):
    c = make_client(repo, tmp_path,
                    llm_factory=lambda: ScriptedLLM([ScriptedLLM.final(GOOD)]))
    assert ask(c, repo).status_code == 200


def test_byo_only_requires_headers(repo, tmp_path):
    c = make_client(repo, tmp_path, settings=ApiSettings(byo_llm_only=True),
                    llm_factory=lambda: ScriptedLLM([ScriptedLLM.final(GOOD)]))
    assert ask(c, repo).status_code == 401
    assert ask(c, repo, byo()).status_code == 200


def test_byo_only_from_env(monkeypatch, repo, tmp_path):
    monkeypatch.setenv("CODERAG_BYO_LLM_ONLY", "1")
    assert ApiSettings.from_env().byo_llm_only is True


def test_credentials_in_json_body_are_ignored(repo, tmp_path):
    f = Factory()
    c = make_client(repo, tmp_path, settings=ApiSettings(byo_llm_only=True), factory=f)
    r = ask(c, repo, None, provider="openai", model="m", api_key=PATTERN_KEY,
            x_llm_key=PATTERN_KEY)
    assert r.status_code == 401 and f.calls == []
    assert PATTERN_KEY not in r.text


@pytest.mark.parametrize("headers", [
    {"X-LLM-Provider": "openai"},
    {"X-LLM-Key": PATTERN_KEY},
    {"X-LLM-Provider": "openai", "X-LLM-Key": PATTERN_KEY},
    byo(provider="not-a-provider"),
    byo(key=None),
    byo(base_url="https://llm.example.com/v1"),  # base URL only for openai_compatible
    byo(provider="openai_compatible"),  # missing base URL
    byo(provider="openai_compatible", base_url="https://169.254.169.254/latest"),
    byo(provider="openai_compatible", base_url="http://localhost:11434/v1"),
    byo(model="bad model!"),
])
def test_bad_credentials_are_400_and_never_echo_key(repo, tmp_path, headers):
    from coderag.providers.registry import build_client

    f = Factory(make=lambda p, m, k, b: build_client(p, m, k, b))
    c = make_client(repo, tmp_path, factory=f)
    r = ask(c, repo, headers)
    assert r.status_code == 400, r.text
    assert PATTERN_KEY not in r.text


def test_provider_error_echoing_key_is_scrubbed_everywhere(repo, tmp_path, caplog):
    def make(provider, model, key, base_url):
        def handler(req):
            logging.getLogger("coderag.test.provider").warning("upstream said: key=%s", key)
            return httpx.Response(401, json={"error": {"message": f"Incorrect key {key}"}})

        return OpenAICompatClient(base_url="https://api.example.com/v1", model=model,
                                  api_key=key, http=httpx.Client(
                                      transport=httpx.MockTransport(handler)))

    caplog.set_level(logging.DEBUG)
    c = make_client(repo, tmp_path, factory=Factory(make))
    for key in (PATTERN_KEY, PLAIN_KEY):
        r = ask(c, repo, byo(key=key))
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False and body["failure"].startswith("llm_error")
        assert key not in r.text
        assert key not in caplog.text
        for rec in caplog.records:
            assert key not in rec.getMessage()


def test_traceback_text_is_scrubbed(repo, tmp_path, caplog):
    class Exploding:
        model = "m"

        def __init__(self, key):
            self.key = key

        def complete(self, system, messages, tools):
            try:
                raise RuntimeError(f"connection failed for key {self.key}")
            except RuntimeError:
                logging.getLogger("coderag.test.boom").exception("provider blew up")
                raise

    caplog.set_level(logging.DEBUG)
    c = make_client(repo, tmp_path, factory=Factory(lambda p, m, k, b: Exploding(k)))
    r = ask(c, repo, byo(key=PLAIN_KEY))
    assert r.status_code == 200 and PLAIN_KEY not in r.text
    assert "RuntimeError" in r.json()["failure"]
    assert PLAIN_KEY not in caplog.text
    boom = [rec for rec in caplog.records if rec.name == "coderag.test.boom"]
    assert boom and "connection failed for key [REDACTED]" in (boom[0].exc_text or "")


def test_unhandled_error_returns_generic_500_without_key(repo, tmp_path, caplog):
    def make(provider, model, key, base_url):
        raise RuntimeError(f"factory crashed with {key}")

    caplog.set_level(logging.DEBUG)
    c = make_client(repo, tmp_path, factory=Factory(make), raise_server_exceptions=False)
    r = ask(c, repo, byo(key=PLAIN_KEY))
    assert r.status_code == 500
    assert PLAIN_KEY not in r.text and PLAIN_KEY not in caplog.text


def test_uvicorn_style_access_lines_are_scrubbed_but_keep_args_shape(caplog):
    from coderag.redact import install_log_scrubbing

    install_log_scrubbing()
    caplog.set_level(logging.INFO, logger="uvicorn.access")
    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d', "127.0.0.1:5000", "POST",
        f"/ask?x-llm-key={PATTERN_KEY}", "1.1", 200)
    rec = caplog.records[-1]
    assert PATTERN_KEY not in caplog.text
    assert isinstance(rec.args, tuple) and len(rec.args) == 5 and rec.args[4] == 200


def test_validation_errors_do_not_echo_input(repo, tmp_path):
    c = make_client(repo, tmp_path)
    r = c.post("/ask", json={"repo": str(repo), "question": PATTERN_KEY * 300})
    assert r.status_code == 422 and PATTERN_KEY not in r.text


# ------------------------------------------------------------------ auth, limits, https


def test_api_token_auth(repo, tmp_path):
    settings = ApiSettings(api_token="server-token-123")
    app = create_app(allowed_roots=[repo.parent], db_dir=tmp_path / "d", settings=settings,
                     client_factory=Factory())
    c = TestClient(app)
    assert c.get("/health").status_code == 200
    assert c.get("/providers").status_code == 200
    assert c.post("/index", json={"repo": str(repo)}).status_code == 401
    bad = {"Authorization": "Bearer wrong"}
    assert c.post("/index", json={"repo": str(repo)}, headers=bad).status_code == 401
    good = {"Authorization": "Bearer server-token-123"}
    assert c.post("/index", json={"repo": str(repo)}, headers=good).status_code == 200
    assert ask(c, repo, byo()).status_code == 401
    r = ask(c, repo, {**byo(), **good})
    assert r.status_code == 200
    assert "server-token-123" not in r.text


def test_rate_limit_returns_429_with_retry_after(repo, tmp_path):
    now = [1000.0]
    c = make_client(repo, tmp_path, settings=ApiSettings(ask_per_minute=2),
                    clock=lambda: now[0])
    assert ask(c, repo, byo()).status_code == 200
    assert ask(c, repo, byo()).status_code == 200
    r = ask(c, repo, byo())
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    now[0] += 61
    assert ask(c, repo, byo()).status_code == 200


def test_index_rate_limit(repo, tmp_path):
    c = make_client(repo, tmp_path, settings=ApiSettings(index_per_minute=1))
    assert c.post("/index", json={"repo": str(repo)}).status_code == 429


def test_body_size_limit(repo, tmp_path):
    c = make_client(repo, tmp_path, settings=ApiSettings(max_body_bytes=500))
    r = c.post("/ask", json={"repo": str(repo), "question": "x" * 2000}, headers=byo())
    assert r.status_code == 413
    r = c.post("/ask", content=b"{" + b" " * 2000 + b"}",
               headers={"content-type": "application/json", "transfer-encoding": "chunked"})
    assert r.status_code == 413


def test_max_steps_is_capped(repo, tmp_path):
    seen = {}

    def make(p, m, k, b):
        return ScriptedLLM([ScriptedLLM.tool("search_code", {"query": "x"}) for _ in range(30)])

    real_agent = api_module.Agent

    class Spy(real_agent):
        def __init__(self, *a, **kw):
            seen["max_steps"] = kw.get("max_steps")
            super().__init__(*a, **kw)

    app = create_app(allowed_roots=[repo.parent], db_dir=tmp_path / "d",
                     settings=ApiSettings(max_steps=3), client_factory=Factory(make))
    c = TestClient(app)
    c.post("/index", json={"repo": str(repo)})
    api_module.Agent = Spy
    try:
        r = ask(c, repo, byo(), max_steps=20)
    finally:
        api_module.Agent = real_agent
    assert r.status_code == 200 and seen["max_steps"] == 3
    assert r.json()["failure"] == "max_steps_exhausted"


def test_repo_size_caps(repo, tmp_path):
    app = create_app(allowed_roots=[repo.parent], db_dir=tmp_path / "d",
                     settings=ApiSettings(max_repo_files=3), client_factory=Factory())
    assert TestClient(app).post("/index", json={"repo": str(repo)}).status_code == 413
    app = create_app(allowed_roots=[repo.parent], db_dir=tmp_path / "d2",
                     settings=ApiSettings(max_repo_bytes=1000), client_factory=Factory())
    assert TestClient(app).post("/index", json={"repo": str(repo)}).status_code == 413


def test_index_concurrency_cap(repo, tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()
    real = api_module.index_repo

    def slow_index(*a, **kw):
        started.set()
        release.wait(10)
        return real(*a, **kw)

    monkeypatch.setattr(api_module, "index_repo", slow_index)
    app = create_app(allowed_roots=[repo.parent], db_dir=tmp_path / "d",
                     settings=ApiSettings(index_concurrency=1), client_factory=Factory())
    c = TestClient(app)
    results = {}
    t = threading.Thread(target=lambda: results.setdefault(
        "first", c.post("/index", json={"repo": str(repo)}).status_code))
    t.start()
    assert started.wait(10)
    second = c.post("/index", json={"repo": str(repo)})
    release.set()
    t.join(10)
    assert results["first"] == 200
    assert second.status_code == 429 and "retry-after" in second.headers


def test_https_required(repo, tmp_path):
    app = create_app(allowed_roots=[repo.parent], db_dir=tmp_path / "d",
                     settings=ApiSettings(require_https=True), client_factory=Factory())
    c = TestClient(app)
    assert c.get("/health").status_code == 200
    assert c.post("/index", json={"repo": str(repo)}).status_code == 400
    fwd = {"X-Forwarded-Proto": "https"}
    assert c.post("/index", json={"repo": str(repo)}, headers=fwd).status_code == 400
    app = create_app(allowed_roots=[repo.parent], db_dir=tmp_path / "d2",
                     settings=ApiSettings(require_https=True, trust_proxy=True),
                     client_factory=Factory())
    c = TestClient(app)
    assert c.post("/index", json={"repo": str(repo)}, headers=fwd).status_code == 200
    s = TestClient(app, base_url="https://testserver")
    assert s.post("/index", json={"repo": str(repo)}).status_code == 200


def test_providers_endpoint(repo, tmp_path):
    c = make_client(repo, tmp_path)
    data = c.get("/providers").json()
    names = {p["name"] for p in data["providers"]}
    assert {"anthropic", "openai", "gemini", "openai_compatible"} <= names


def test_allowlist_confinement_unchanged(repo, tmp_path):
    c = make_client(repo, tmp_path)
    assert c.post("/index", json={"repo": "/etc"}).status_code == 403
    assert ask(c, Path("/etc"), byo()).status_code == 403
