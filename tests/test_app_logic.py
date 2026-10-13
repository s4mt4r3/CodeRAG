"""Web app: Streamlit-free logic (repo checks, indexing, examples, snippets, errors, asking)."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import httpx
import pytest

from coderag import app_logic as L
from coderag.llm import ScriptedLLM
from coderag.providers.openai_compat import OpenAICompatClient
from conftest import FIXTURE_ROOT

KEY = "zz9plainSECRETforApp77"
PATTERN_KEY = "sk-test-APPKEY0123456789abcdef"
GOOD_CITE = {"path": "shopcore/utils.py", "start_line": 9, "end_line": 13, "symbol": "slugify"}
BAD_CITE = {"path": "shopcore/utils.py", "start_line": 16, "end_line": 18, "symbol": "slugify"}


def final(*cites, answer="slugify makes slugs."):
    return ScriptedLLM.final(json.dumps({"answer": answer, "citations": list(cites)}))


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    dst = tmp_path / "repo"
    shutil.copytree(FIXTURE_ROOT, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    for var in ["CODERAG_ALLOWED_ROOTS", "CODERAG_MAX_REPO_FILES", "CODERAG_MAX_REPO_BYTES",
                "CODERAG_MODEL", "OPENAI_API_KEY", "GEMINI_API_KEY", "CODERAG_LLM_KEY",
                "ANTHROPIC_API_KEY", "CODERAG_ALLOW_INSECURE_LOCAL"]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CODERAG_DB_DIR", str(tmp_path / "dbs"))
    yield
    L.reset_client_factory()


# ------------------------------------------------------------------ repo validation


def test_valid_repo(repo):
    check = L.validate_repo(str(repo))
    assert check.ok and check.path == repo.resolve() and check.files == 13
    assert check.message


@pytest.mark.parametrize("bad", ["", "   ", "/definitely/not/here", __file__])
def test_not_a_directory(bad):
    check = L.validate_repo(bad)
    assert not check.ok and check.path is None
    assert "directory" in check.message.lower() or "enter" in check.message.lower()


def test_no_python_files(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "notes.md").write_text("# hi\n")
    check = L.validate_repo(str(d))
    assert not check.ok and "no python files" in check.message.lower()


def test_allowlist(repo, tmp_path, monkeypatch):
    outside = tmp_path / "other"
    shutil.copytree(repo, outside)
    monkeypatch.setenv("CODERAG_ALLOWED_ROOTS", str(repo))
    assert L.allowlist_active()
    assert L.validate_repo(str(repo)).ok
    check = L.validate_repo(str(outside))
    assert not check.ok and "CODERAG_ALLOWED_ROOTS" in check.message
    assert not L.validate_repo(str(repo / ".." / "other")).ok
    monkeypatch.delenv("CODERAG_ALLOWED_ROOTS")
    assert not L.allowlist_active() and L.validate_repo(str(outside)).ok


def test_size_caps(repo, monkeypatch):
    check = L.validate_repo(str(repo), max_files=3)
    assert not check.ok and "too large" in check.message.lower()
    assert not L.validate_repo(str(repo), max_bytes=1000).ok
    monkeypatch.setenv("CODERAG_MAX_REPO_FILES", "3")
    assert not L.validate_repo(str(repo)).ok


def test_default_repo_path(tmp_path):
    (tmp_path / "data" / "eval_repo" / "src").mkdir(parents=True)
    assert L.default_repo_path(tmp_path) == str(tmp_path / "data" / "eval_repo" / "src")
    other = tmp_path / "x"
    other.mkdir()
    assert L.default_repo_path(other) == str(other)
    assert Path(L.example_repo_path()).resolve() == FIXTURE_ROOT.resolve()


# ------------------------------------------------------------------ indexing and examples


def test_index_summary_and_db_location(repo, tmp_path):
    s = L.index(repo)
    assert (s.files, s.added, s.unchanged) == (13, 13, 0)
    assert s.chunks > 0 and Path(s.db_path).exists()
    assert Path(s.db_path).parent == tmp_path / "dbs"
    again = L.index(repo)
    assert (again.files, again.added, again.unchanged) == (13, 0, 13)


def test_example_questions_use_real_symbols(repo):
    s = L.index(repo)
    qs = L.example_questions(s.db_path)
    assert len(qs) == 3 and len(set(qs)) == 3
    from helpers import oracle

    names = {sym.name for sym in oracle.extract_repo(repo)}
    for q in qs:
        assert "`" in q
        name = q.split("`")[1]
        assert name in names, q
        assert not name.startswith("_")
    assert qs == L.example_questions(s.db_path)  # deterministic


# ------------------------------------------------------------------ snippets


def test_snippet_has_real_line_numbers(repo):
    text = L.read_snippet(repo, GOOD_CITE)
    lines = text.splitlines()
    assert lines[0].lstrip().startswith("9") and "def slugify" in lines[0]
    assert lines[-1].lstrip().startswith("13")


@pytest.mark.parametrize("cite", [
    {**GOOD_CITE, "path": "../outside.py"}, {**GOOD_CITE, "path": "/etc/passwd"},
    {**GOOD_CITE, "path": "nope.py"}, {**GOOD_CITE, "start_line": 0},
    {**GOOD_CITE, "start_line": 20, "end_line": 10}, {**GOOD_CITE, "path": "leak.py"},
])
def test_snippet_confinement(repo, tmp_path, cite):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET\n")
    os.symlink(secret, repo / "leak.py")
    assert L.read_snippet(repo, cite) is None


def test_snippet_caps_huge_ranges(repo):
    text = L.read_snippet(repo, {**GOOD_CITE, "path": "shopcore/models.py", "start_line": 1,
                                 "end_line": 100_000}, max_lines=10)
    assert len(text.splitlines()) == 11 and "truncated" in text.splitlines()[-1]


# ------------------------------------------------------------------ errors


@pytest.mark.parametrize("failure", [
    "max_steps_exhausted", "invalid_final_answer", "refusal",
    "llm_error: ProviderAuthError: openai returned HTTP 401",
    "llm_error: ProviderRateLimitError: x", "llm_error: ProviderBlockedError: x",
    "llm_error: ProviderTimeoutError: x", "llm_error: ProviderConnectionError: x",
    "llm_error: ProviderServerError: x", "llm_error: ProviderResponseError: x",
    "llm_error: RuntimeError: weird", "something_new"])
def test_friendly_failures_have_message_and_suggestion(failure):
    message, suggestion = L.friendly_failure(failure)
    assert message and suggestion and message != suggestion


def test_specific_suggestions():
    assert "key" in L.friendly_failure("llm_error: ProviderAuthError: x")[1].lower()
    assert "larger model" in L.friendly_failure("max_steps_exhausted")[1].lower()
    assert "CODERAG_ALLOW_INSECURE_LOCAL" in L.friendly_failure(
        "llm_error: ProviderBlockedError: x")[1]


def test_messages_never_contain_secrets():
    for text in [f"llm_error: ProviderAuthError: bad key {KEY}", f"weird {PATTERN_KEY}"]:
        msg, sug = L.friendly_failure(text, secrets=(KEY,))
        assert KEY not in msg + sug and PATTERN_KEY not in msg + sug
    msg, sug = L.friendly_exception(ValueError(f"bad thing {KEY} {PATTERN_KEY}"), secrets=(KEY,))
    assert KEY not in msg + sug and PATTERN_KEY not in msg + sug


# ------------------------------------------------------------------ asking


def ask(repo, **kw):
    params = {"provider": "openai", "model": "m-1", "key": KEY, "base_url": None,
              "demo": False, "retriever": "bm25", "max_steps": 8}
    params.update(kw)
    return L.run_question(repo, "What does slugify do?", **params)


def test_demo_mode_is_labeled_and_needs_no_key(repo):
    L.index(repo)
    out = ask(repo, demo=True, key=None, provider="", model="")
    assert out.mode == "demo" and "scripted" in out.label.lower()
    assert "not a model answer" in out.label.lower()
    assert out.ok and out.citations
    assert out.citations[0]["snippet"]


def test_fake_provider_verified_and_unverified(repo):
    calls = []

    def factory(provider, model, key, base_url):
        calls.append((provider, model, key, base_url))
        return ScriptedLLM([final(GOOD_CITE, BAD_CITE), final(GOOD_CITE, BAD_CITE)])

    L.set_client_factory(factory)
    out = ask(repo)
    assert calls == [("openai", "m-1", KEY, None)]
    assert out.ok and out.mode == "model" and not out.unsupported
    assert [c["verified"] for c in out.citations] == [True, False]
    assert out.citations[1]["reason"] == "symbol_not_in_range"
    assert "def slugify" in out.citations[0]["snippet"]
    assert out.usage["input_tokens"] > 0 and out.steps >= 1


def test_unsupported_answer(repo):
    L.set_client_factory(lambda *a: ScriptedLLM([final(BAD_CITE), final(BAD_CITE)]))
    out = ask(repo)
    assert out.ok and out.unsupported


def test_auto_reindex_before_question(repo):
    L.index(repo)
    (repo / "shopcore" / "extra.py").write_text("def brand_new_helper():\n    return 1\n")
    L.set_client_factory(lambda *a: ScriptedLLM([final(
        {"path": "shopcore/extra.py", "start_line": 1, "end_line": 2,
         "symbol": "brand_new_helper"})]))
    out = ask(repo)
    assert out.citations[0]["verified"] is True


def test_provider_error_echoing_key_is_friendly_and_scrubbed(repo, caplog):
    def factory(provider, model, key, base_url):
        def handler(req):
            logging.getLogger("coderag.test.app").warning("provider saw %s", key)
            return httpx.Response(401, json={"error": {"message": f"bad key {key}"}})

        return OpenAICompatClient(base_url="https://api.example.com/v1", model=model,
                                  api_key=key, http=httpx.Client(
                                      transport=httpx.MockTransport(handler)))

    caplog.set_level(logging.DEBUG)
    L.set_client_factory(factory)
    for key in (KEY, PATTERN_KEY):
        out = ask(repo, key=key)
        assert not out.ok and "key" in out.message.lower()
        blob = json.dumps(L.outcome_to_dict(out))
        assert key not in blob and key not in caplog.text


def test_bad_settings_become_friendly_errors(repo):
    for kw in [{"provider": "nope"}, {"model": ""}, {"key": None},
               {"provider": "openai_compatible", "base_url": "https://169.254.169.254/"}]:
        out = ask(repo, **kw)
        assert not out.ok and out.message and KEY not in out.message + out.suggestion


def test_question_validation(repo):
    assert not L.run_question(repo, "   ", provider="", model="", key=None, base_url=None,
                              demo=True, retriever="bm25", max_steps=8).ok
    out = L.run_question(repo, "x" * 4001, provider="", model="", key=None, base_url=None,
                         demo=True, retriever="bm25", max_steps=8)
    assert not out.ok and "4000" in out.message


def test_unexpected_exception_is_scrubbed(repo, caplog):
    def factory(provider, model, key, base_url):
        raise RuntimeError(f"exploded with {key}")

    caplog.set_level(logging.DEBUG)
    L.set_client_factory(factory)
    out = ask(repo)
    assert not out.ok and KEY not in json.dumps(L.outcome_to_dict(out))
    assert KEY not in caplog.text


def test_trace_summary_is_clean(repo):
    L.set_client_factory(lambda *a: ScriptedLLM([
        ScriptedLLM.tool("search_code", {"query": "slugify"}),
        ScriptedLLM.tool("read_file", {"path": "/etc/passwd"}),
        final(GOOD_CITE)]))
    out = ask(repo)
    assert [t["name"] for t in out.trace] == ["search_code", "read_file"]
    assert out.trace[0]["status"] == "ok" and out.trace[1]["status"].startswith("error")
    assert "<tool_output" not in json.dumps(out.trace)


def test_hybrid_falls_back_to_bm25_with_warning(repo, monkeypatch):
    def broken():
        raise RuntimeError("model missing")

    monkeypatch.setattr(L, "_load_embedder", broken)
    L.set_client_factory(lambda *a: ScriptedLLM([final(GOOD_CITE)]))
    out = ask(repo, retriever="hybrid")
    assert out.ok and out.warnings and "BM25" in out.warnings[0]


def test_provider_choices_and_env_key(monkeypatch):
    names = [p["name"] for p in L.provider_choices()]
    assert names[0] == "anthropic" and "gemini" in names and "openai_compatible" in names
    assert not L.env_key_available("openai")
    monkeypatch.setenv("OPENAI_API_KEY", PATTERN_KEY)
    assert L.env_key_available("openai")
    assert L.resolve_key("openai", "") == PATTERN_KEY
    assert L.resolve_key("openai", "typed") == "typed"
