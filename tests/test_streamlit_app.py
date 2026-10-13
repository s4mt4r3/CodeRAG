"""Web app: Streamlit UI driven offline with streamlit.testing.v1.AppTest."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import httpx
import pytest

from conftest import FIXTURE_ROOT, PROJECT_ROOT

AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="streamlit not installed (pip install -e '.[app]')").AppTest

from coderag import app_logic as L  # noqa: E402
from coderag.llm import ScriptedLLM  # noqa: E402
from coderag.providers.openai_compat import OpenAICompatClient  # noqa: E402

APP = str(PROJECT_ROOT / "src" / "coderag" / "streamlit_app.py")
KEY = "zz9plainSECRETforUI55"
GOOD_CITE = {"path": "shopcore/utils.py", "start_line": 9, "end_line": 13, "symbol": "slugify"}
BAD_CITE = {"path": "shopcore/utils.py", "start_line": 16, "end_line": 18, "symbol": "slugify"}


def final(*cites):
    return ScriptedLLM.final(json.dumps({"answer": "slugify makes slugs.",
                                         "citations": list(cites)}))


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    dst = tmp_path / "repo"
    shutil.copytree(FIXTURE_ROOT, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    for var in ["CODERAG_ALLOWED_ROOTS", "CODERAG_MODEL", "OPENAI_API_KEY", "CODERAG_LLM_KEY",
                "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "CODERAG_MAX_REPO_FILES"]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CODERAG_DB_DIR", str(tmp_path / "dbs"))
    yield
    L.reset_client_factory()


def start() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def rendered(at: AppTest) -> str:
    parts = []
    for kind in ["markdown", "caption", "code", "error", "warning", "info", "success", "text",
                 "title", "header", "subheader", "json", "exception"]:
        for el in getattr(at, kind):
            parts.append(str(getattr(el, "value", "")))
    parts += [e.label for e in at.expander]
    parts += [str(t.value) for t in at.text_input if t.key != "api_key"]
    parts += [str(t.value) for t in at.text_area]
    return "\n".join(parts)


def index_repo_in_ui(at: AppTest, repo: Path) -> AppTest:
    at.text_input(key="repo_path").set_value(str(repo)).run()
    at.button(key="index_btn").click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def ask_in_ui(at: AppTest, question="What does slugify do?") -> AppTest:
    at.text_area(key="question").set_value(question).run()
    at.button(key="ask_btn").click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_starts_in_demo_mode_without_key():
    at = start()
    assert at.toggle(key="demo").value is True
    assert at.text_input(key="api_key").type == "password" or \
        at.text_input(key="api_key").proto.type == 1
    text = rendered(at)
    assert "reads files from this machine" in text  # banner when no allowlist is set
    assert any("About safety" in e.label for e in at.expander)


def test_bundled_example_button_sets_repo_path():
    at = start()
    at.button(key="example_repo_btn").click().run()
    assert Path(at.text_input(key="repo_path").value).resolve() == FIXTURE_ROOT.resolve()


def test_index_shows_summary(repo):
    at = index_repo_in_ui(start(), repo)
    assert any("13 files" in s.value for s in at.success)
    assert len([b for b in at.button if (b.key or "").startswith("example_q_")]) == 3


def test_invalid_repo_shows_error(tmp_path):
    at = start()
    at.text_input(key="repo_path").set_value(str(tmp_path / "missing")).run()
    at.button(key="index_btn").click().run()
    assert any("directory" in e.value.lower() for e in at.error)


def test_demo_question_renders_answer_badge_and_code(repo):
    at = ask_in_ui(index_repo_in_ui(start(), repo))
    text = rendered(at)
    assert "scripted pipeline demo" in text.lower()
    assert any("verified" == s.value.strip().lower() or s.value.lower().startswith("verified")
               for s in at.success)
    assert any("|" in c.value and c.value.strip() for c in at.code)
    assert any(":" in e.label and "-" in e.label for e in at.expander)
    assert any("How it got there" in e.label for e in at.expander)


def test_example_question_button_fills_question(repo):
    at = index_repo_in_ui(start(), repo)
    btn = next(b for b in at.button if (b.key or "").startswith("example_q_"))
    btn.click().run()
    assert at.text_area(key="question").value == btn.label


def test_fake_provider_verified_unverified_and_unsupported(repo, caplog):
    caplog.set_level(logging.DEBUG)
    L.set_client_factory(lambda *a: ScriptedLLM([final(GOOD_CITE, BAD_CITE),
                                                 final(GOOD_CITE, BAD_CITE)]))
    at = start()
    at.toggle(key="demo").set_value(False).run()
    at.selectbox(key="provider").set_value("openai").run()
    at.text_input(key="model").set_value("m-1").run()
    at.text_input(key="api_key").set_value(KEY).run()
    at = ask_in_ui(index_repo_in_ui(at, repo))
    assert any(s.value.lower().startswith("verified") for s in at.success)
    assert any("unverified" in e.value.lower() and "symbol_not_in_range" in e.value
               for e in at.error)
    assert "scripted pipeline demo" not in rendered(at).lower()

    L.set_client_factory(lambda *a: ScriptedLLM([final(BAD_CITE), final(BAD_CITE)]))
    at = ask_in_ui(at, "Where is slugify?")
    assert any("unsupported" in w.value.lower() for w in at.warning)
    assert KEY not in rendered(at)
    assert KEY not in caplog.text
    assert all(KEY not in str(k) for k in at.session_state.keys())


def test_provider_error_with_key_shows_friendly_message(repo, caplog):
    def factory(provider, model, key, base_url):
        return OpenAICompatClient(
            base_url="https://api.example.com/v1", model=model, api_key=key,
            http=httpx.Client(transport=httpx.MockTransport(
                lambda req: httpx.Response(401, json={"error": {"message": f"bad {key}"}}))))

    caplog.set_level(logging.DEBUG)
    L.set_client_factory(factory)
    at = start()
    at.toggle(key="demo").set_value(False).run()
    at.selectbox(key="provider").set_value("openai").run()
    at.text_input(key="model").set_value("m-1").run()
    at.text_input(key="api_key").set_value(KEY).run()
    at = ask_in_ui(index_repo_in_ui(at, repo))
    assert any("rejected the api key" in e.value.lower() for e in at.error)
    assert KEY not in rendered(at) and KEY not in caplog.text
    assert all(KEY not in str(k) for k in at.session_state.keys())


def test_clear_history(repo):
    at = ask_in_ui(index_repo_in_ui(start(), repo))
    assert at.session_state["history"]
    at.button(key="clear_btn").click().run()
    assert at.session_state["history"] == []


def test_base_url_field_only_for_openai_compatible():
    at = start()
    assert not [t for t in at.text_input if t.key == "base_url"]
    at.selectbox(key="provider").set_value("openai_compatible").run()
    assert [t for t in at.text_input if t.key == "base_url"]


def test_streamlit_config_is_safe():
    cfg = (PROJECT_ROOT / ".streamlit" / "config.toml").read_text()
    assert 'showErrorDetails = "none"' in cfg
    assert "gatherUsageStats = false" in cfg
    assert 'address = "127.0.0.1"' in cfg
