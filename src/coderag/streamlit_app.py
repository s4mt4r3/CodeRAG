"""coderag web app (Streamlit). Widgets and rendering only; logic lives in app_logic.

Run with ``make app`` or ``coderag app``. The API key is kept only in this browser
session's state and in local variables while a question runs; nothing secret is
cached (no ``st.cache_*`` is used here at all). Every string that a model or a
repository can influence goes through ``md_escape`` before Markdown rendering.

Hosted deployments set ``CODERAG_HOSTED=1``: only the bundled example repo is
allowed, keys come only from what each visitor types, and local-only providers are
hidden.
"""

from __future__ import annotations

import os

import streamlit as st

from coderag import app_logic as L
from coderag.apiguard import ApiSettings

L.install()
HOSTED = L.hosted_mode()
if HOSTED:
    L.apply_hosted_defaults()
st.set_page_config(page_title="coderag", page_icon=None, layout="wide")
esc = L.md_escape

ss = st.session_state
ss.setdefault("history", [])
ss.setdefault("index_summary", None)
ss.setdefault("indexed_repo", None)
ss.setdefault("repo_path", L.example_repo_path() if HOSTED else L.default_repo_path())
ss.setdefault("question", "")
PROVIDERS = L.visible_providers(HOSTED)
ss.setdefault("provider", PROVIDERS[0])
ss.setdefault("demo", HOSTED or not L.env_key_available(ss["provider"]))
RETRIEVERS = {"BM25 (recommended)": "bm25"}
if L.hybrid_available():
    RETRIEVERS["Hybrid (needs local embedding model)"] = "hybrid"


def _use_example_repo() -> None:
    ss["repo_path"] = L.example_repo_path()


def _set_question(text: str) -> None:
    ss["question"] = text


def _clear_history() -> None:
    ss["history"] = []


# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.header("Model")
    provider = st.selectbox("Provider", PROVIDERS, key="provider")
    st.text_input("Model", value=os.environ.get("CODERAG_MODEL", ""), key="model",
                  help="A model id from your provider's documentation. No default is assumed.")
    st.text_input("API key", type="password", key="api_key",
                  help="Kept only in this browser session. Never saved, cached, or logged.")
    if not HOSTED and L.env_key_available(provider) and not ss.get("api_key"):
        st.caption("Key found in environment.")
    if provider == "openai_compatible":
        st.text_input("Base URL", key="base_url", placeholder="https://your-server/v1",
                      help="Must be https and resolve to a public address.")
    if provider == "ollama":
        st.caption("Ollama runs locally; start the app with CODERAG_ALLOW_INSECURE_LOCAL=1.")
    st.toggle("Demo mode (no key, scripted)", key="demo",
              help="Uses a fixed search-read-cite policy instead of a model, so you can try "
                   "the interface for free.")
    st.header("Retrieval")
    retriever_label = st.selectbox("Retriever", list(RETRIEVERS), key="retriever",
                                   help="BM25 scored higher than hybrid in this project's eval.")
    max_steps = st.slider("Max steps", 1, ApiSettings.from_env().max_steps, 8, key="max_steps")

# ------------------------------------------------------------------ main

st.title("coderag")
st.markdown("Ask questions about a Python repository and get answers whose citations are "
            "checked against the code.")
if HOSTED:
    st.info("Hosted demo: only the bundled example repository is available. Demo mode needs "
            "no key. A key you type is used only for your own questions in this browser "
            "session and is never stored, but it is sent from this server to your provider, "
            "so use one you are comfortable using here.")
if not L.allowlist_active():
    st.caption("This app reads files from this machine and sends excerpts to the model "
               "you choose.")
with st.expander("About safety"):
    st.markdown(
        "- The model can only use read-only tools confined to the repository folder.\n"
        "- Repository text is passed to the model inside delimiters and labeled as untrusted.\n"
        "- Repositories can contain text written to manipulate a model (prompt injection). "
        "The delimiters and read-only tools reduce the risk; they do not remove it.\n"
        "- Tokens are billed to the key you provide.\n"
        "- This app is meant for local or private use. Exposing it publicly needs "
        "authentication and HTTPS in front of it, which Streamlit does not provide.")

st.subheader("1. Repository")
col_path, col_example = st.columns([4, 1])
with col_path:
    st.text_input("Local path", key="repo_path")
with col_example:
    st.button("Use bundled example repo", key="example_repo_btn", on_click=_use_example_repo)

if st.button("Index", key="index_btn"):
    check = L.validate_repo(ss["repo_path"])
    if not check.ok or check.path is None:
        st.error(esc(check.message))
    else:
        with st.spinner("Indexing..."):
            summary = L.index(check.path)
        ss["index_summary"] = summary.__dict__
        ss["indexed_repo"] = str(check.path)

summary = ss["index_summary"]
if summary:
    st.success(f"Indexed {summary['files']} files, {summary['chunks']} chunks "
               f"({summary['unchanged']} unchanged) from {esc(ss['indexed_repo'])}.")
    if summary["warnings"]:
        with st.expander(f"Parse warnings ({len(summary['warnings'])})"):
            st.code("\n".join(summary["warnings"]), language=None)

st.subheader("2. Question")
if summary:
    examples = L.example_questions(summary["db_path"])
    if examples:
        st.caption("Examples from this repository:")
        cols = st.columns(len(examples))
        for i, (col, q) in enumerate(zip(cols, examples, strict=True)):
            with col:
                st.button(q, key=f"example_q_{i}", on_click=_set_question, args=(q,))
st.text_area("Your question", key="question", max_chars=L.MAX_QUESTION_CHARS, height=90)

if st.button("Ask", key="ask_btn", type="primary"):
    repo = ss["indexed_repo"] or ss["repo_path"]
    with st.spinner("Thinking..."):
        outcome = L.run_question(
            repo, ss["question"], provider=provider, model=ss.get("model", ""),
            key=L.app_key(provider, ss.get("api_key"), HOSTED) if not ss["demo"] else None,
            base_url=ss.get("base_url") if provider == "openai_compatible" else None,
            demo=ss["demo"], retriever=RETRIEVERS[retriever_label], max_steps=max_steps)
    ss["history"] = [L.outcome_to_dict(outcome)] + ss["history"]

# ------------------------------------------------------------------ answers


def render(item: dict) -> None:
    """Show one answered (or failed) question."""
    st.markdown(f"**Q:** {esc(item['question'])}")
    if item["label"]:
        st.info(item["label"])
    for warning in item["warnings"]:
        st.warning(esc(warning))
    if not item["ok"]:
        st.error(esc(item["message"]))
        if item["suggestion"]:
            st.caption(esc(item["suggestion"]))
    else:
        st.markdown(esc(item["answer"]) or "_(empty answer)_")
        if item["unsupported"]:
            st.warning("Unsupported: none of the citations could be verified against the code.")
        for c in item["citations"]:
            label = (f"{esc(str(c['path']))}:{c['start_line']}-{c['end_line']} "
                     f"{esc(str(c['symbol']))}")
            with st.expander(label):
                if c["verified"]:
                    st.success("verified")
                else:
                    st.error(f"unverified ({c['reason']})")
                if c.get("snippet"):
                    st.code(c["snippet"], language="python")
                else:
                    st.caption("These lines can't be shown (not a readable file in the repo).")
    with st.expander("How it got there"):
        for t in item["trace"]:
            name = esc(t["name"].replace("`", "'"))
            st.markdown(f"- `{name}` {esc(t['arguments'])} → {esc(t['status'])}")
        st.caption(f"Steps: {item['steps']} · tokens in/out: "
                   f"{item['usage']['input_tokens']}/{item['usage']['output_tokens']}")


if ss["history"]:
    st.subheader("3. Answers")
    st.button("Clear history", key="clear_btn", on_click=_clear_history)
    for entry in ss["history"]:
        render(entry)
        st.divider()
