"""Logic behind the Streamlit app, kept free of Streamlit imports so it is testable.

Secrets: the API key is passed in per call, used inside :func:`redact.secret_scope`,
and never stored here, cached, or included in any returned value. The client
factory hook (:func:`set_client_factory`) exists so tests can inject a fake LLM.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import random
import re
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from coderag.agent import Agent
from coderag.apiguard import ApiSettings
from coderag.cli import default_db
from coderag.evaluation import scripted_policy
from coderag.ingest import walk_repo
from coderag.llm import LLMClient
from coderag.providers.base import ProviderError
from coderag.providers.registry import build_client, key_from_env, list_providers
from coderag.redact import install_log_scrubbing, scrub, secret_scope
from coderag.store import Store, index_repo
from coderag.symbols import split_lines
from coderag.tools import Toolbox, confine_path

log = logging.getLogger("coderag.app")
def _project_root() -> Path:
    """The repository root.

    Editable installs keep this module inside the repo; a normal install (what
    Streamlit Community Cloud does with requirements.txt) puts it in site-packages, so
    fall back to the working directory, which Streamlit Cloud sets to the repo root.
    """
    package_side = Path(__file__).resolve().parents[2]
    for base in (package_side, Path.cwd()):
        if (base / "tests" / "fixtures" / "fixture_repo").is_dir():
            return base
    return package_side


PROJECT_ROOT = _project_root()
EXAMPLE_REPO = PROJECT_ROOT / "tests" / "fixtures" / "fixture_repo"
MAX_QUESTION_CHARS = 4000
DEMO_LABEL = ("Scripted pipeline demo, not a model answer: it searched, read the top hit, "
              "and cited it.")

_FAILURES: dict[str, tuple[str, str]] = {
    "ProviderAuthError": ("The provider rejected the API key.",
                          "Check the key and that it belongs to the selected provider."),
    "ProviderRateLimitError": ("The provider is rate limiting requests.",
                               "Wait a minute, or use a different model or plan."),
    "ProviderBlockedError": ("That base URL is not allowed.",
                             "Use a public https URL. For local Ollama, start the app with "
                             "CODERAG_ALLOW_INSECURE_LOCAL=1."),
    "ProviderTimeoutError": ("The provider took too long to respond.",
                             "Try again, or pick a faster model."),
    "ProviderConnectionError": ("Could not reach the provider.",
                                "Check your network connection and the base URL."),
    "ProviderServerError": ("The provider had a server error.", "Try again in a moment."),
    "ProviderResponseError": ("The provider returned an unexpected response.",
                              "Check the model id; the model may not support tool calling."),
    "refusal": ("The model declined to answer.", "Rephrase the question or try another model."),
    "invalid_final_answer": ("The model did not return a valid answer.",
                             "Try a larger model; small models often can't follow the "
                             "answer format."),
    "max_steps_exhausted": ("The model ran out of steps before answering.",
                            "Raise max steps, or try a larger model; small models often "
                            "can't call tools reliably."),
}
_GENERIC = ("Something went wrong while answering.",
            "Check the terminal output for details (secrets are scrubbed there).")

ClientFactory = Callable[[str, str, "str | None", "str | None"], LLMClient]
_client_factory: ClientFactory = build_client


def install() -> None:
    """Process setup for the app: scrub secrets from every log record."""
    install_log_scrubbing()


def set_client_factory(factory: ClientFactory) -> None:
    """Replace how LLM clients are built (tests inject fakes here)."""
    global _client_factory
    _client_factory = factory


def reset_client_factory() -> None:
    """Restore the real client factory."""
    set_client_factory(build_client)


# ------------------------------------------------------------------ hosted mode and safety

_MD_UNSAFE = re.compile(r"([\\\[\]()!<>])")


def md_escape(text: str) -> str:
    """Neutralize Markdown links, images and HTML in untrusted text before ``st.markdown``.

    Model output can be steered by a prompt-injected repository; a rendered image link
    would make the viewer's browser fetch an attacker URL carrying data. Escaping
    brackets, parentheses, ``!`` and angle brackets keeps the text readable but inert.
    """
    return _MD_UNSAFE.sub(r"\\\1", text or "")


def hosted_mode() -> bool:
    """True when ``CODERAG_HOSTED=1`` (set it as a root-level secret on Streamlit Cloud)."""
    return os.environ.get("CODERAG_HOSTED") == "1"


def apply_hosted_defaults() -> None:
    """Lock a public deployment down: bundled repo only, no localhost providers, temp index."""
    os.environ.setdefault("CODERAG_ALLOWED_ROOTS", str(EXAMPLE_REPO))
    os.environ.pop("CODERAG_ALLOW_INSECURE_LOCAL", None)
    os.environ.setdefault("CODERAG_DB_DIR", str(Path(tempfile.gettempdir()) / "coderag-index"))


def app_key(provider: str, typed: str | None, hosted: bool) -> str | None:
    """The key to use: hosted deployments only ever use what the visitor typed."""
    if hosted:
        return typed or None
    return resolve_key(provider, typed)


def visible_providers(hosted: bool) -> list[str]:
    """Provider names to offer; local-only providers are hidden when hosted."""
    return [p["name"] for p in provider_choices() if not (hosted and p["local_only"])]


def hybrid_available() -> bool:
    """Whether the optional embedding dependency (``.[embed]``) is installed."""
    return importlib.util.find_spec("fastembed") is not None


# ------------------------------------------------------------------ providers and keys


def provider_choices() -> list[dict]:
    """Providers from the registry, in registry order."""
    return list_providers()


def env_key_available(provider: str) -> bool:
    """Whether a key for ``provider`` is already set in the environment."""
    return bool(key_from_env(provider))


def resolve_key(provider: str, typed: str | None) -> str | None:
    """The typed key if any, else the environment's key for ``provider``."""
    return typed or key_from_env(provider)


# ------------------------------------------------------------------ repos and indexing


@dataclass
class RepoCheck:
    """Result of validating a repository path."""

    ok: bool
    path: Path | None
    message: str
    files: int = 0
    bytes: int = 0


@dataclass
class IndexSummary:
    """What indexing did, for display."""

    files: int
    added: int
    updated: int
    removed: int
    unchanged: int
    chunks: int
    warnings: list[str]
    db_path: str


def db_dir() -> Path:
    """Where index databases live (``CODERAG_DB_DIR``, default ``.coderag``)."""
    return Path(os.environ.get("CODERAG_DB_DIR", ".coderag"))


def allowed_roots() -> list[Path]:
    """Directories from ``CODERAG_ALLOWED_ROOTS`` (empty means any local path)."""
    raw = os.environ.get("CODERAG_ALLOWED_ROOTS", "")
    return [Path(p).resolve() for p in raw.split(os.pathsep) if p.strip()]


def allowlist_active() -> bool:
    """Whether repository paths are restricted to an allowlist."""
    return bool(allowed_roots())


def default_repo_path(project_root: Path = PROJECT_ROOT) -> str:
    """The bundled eval repo if present, else the given directory."""
    candidate = Path(project_root) / "data" / "eval_repo" / "src"
    return str(candidate if candidate.is_dir() else Path(project_root))


def example_repo_path() -> str:
    """The small synthetic repo shipped with the tests."""
    return str(EXAMPLE_REPO)


def validate_repo(path_str: str, max_files: int | None = None,
                  max_bytes: int | None = None) -> RepoCheck:
    """Check that ``path_str`` is an allowed directory with a manageable amount of Python."""
    if not path_str or not path_str.strip():
        return RepoCheck(False, None, "Enter a repository path.")
    path = Path(path_str.strip()).expanduser().resolve()
    if not path.is_dir():
        return RepoCheck(False, None, f"Not a directory: {path}")
    roots = allowed_roots()
    if roots and not any(path == r or path.is_relative_to(r) for r in roots):
        return RepoCheck(False, None, "That path is outside CODERAG_ALLOWED_ROOTS.")
    settings = ApiSettings.from_env()
    max_files = max_files if max_files is not None else settings.max_repo_files
    max_bytes = max_bytes if max_bytes is not None else settings.max_repo_bytes
    files = walk_repo(path)
    if not files:
        return RepoCheck(False, None, "No Python files found in that directory.")
    size = sum(len(f.text.encode("utf-8")) for f in files)
    if len(files) > max_files or size > max_bytes:
        return RepoCheck(False, None,
                         f"Repository too large ({len(files)} Python files); the limit is "
                         f"{max_files} files.", len(files), size)
    return RepoCheck(True, path, f"{len(files)} Python files found.", len(files), size)


def index(repo: Path) -> IndexSummary:
    """Index (incrementally) and summarize."""
    db = default_db(Path(repo), db_dir())
    db.parent.mkdir(parents=True, exist_ok=True)
    stats = index_repo(repo, db)
    return IndexSummary(files=stats.added + stats.updated + stats.unchanged, added=stats.added,
                        updated=stats.updated, removed=stats.removed,
                        unchanged=stats.unchanged, chunks=stats.chunks,
                        warnings=stats.warnings[:20], db_path=str(db))


def example_questions(db_path: str, n: int = 3, seed: int = 0) -> list[str]:
    """Up to ``n`` questions about real symbols in the index (deterministic)."""
    store = Store(db_path)
    try:
        rows = store.conn.execute(
            "SELECT name, kind, docstring FROM symbols ORDER BY path, start_line").fetchall()
        callees = [r[0] for r in store.conn.execute(
            "SELECT callee FROM edges WHERE confidence = 'resolved' AND callee IS NOT NULL "
            "GROUP BY callee ORDER BY callee").fetchall()]
    finally:
        store.close()
    counts: dict[str, int] = {}
    for name, _, _ in rows:
        counts[name] = counts.get(name, 0) + 1
    public = [r for r in rows if not r[0].startswith("_") and counts[r[0]] == 1]
    rng = random.Random(seed)
    used: set[str] = set()

    def pick(candidates: list[str]) -> str | None:
        options = [c for c in dict.fromkeys(candidates) if c not in used]
        if not options:
            return None
        choice = rng.choice(options)
        used.add(choice)
        return choice

    questions = []
    documented = pick([r[0] for r in public if r[1] != "class" and r[2]])
    if documented:
        questions.append(f"What does `{documented}` do?")
    public_names = {r[0] for r in public}
    called = pick([c.rsplit(".", 1)[-1] for c in callees
                   if c.rsplit(".", 1)[-1] in public_names])
    if called:
        questions.append(f"Which functions call `{called}`?")
    defined = pick([r[0] for r in public])
    if defined:
        questions.append(f"Where is `{defined}` defined?")
    return questions[:n]


# ------------------------------------------------------------------ snippets and errors


def read_snippet(repo_root: str | Path, citation: dict, max_lines: int = 60) -> str | None:
    """Cited lines with their real line numbers, or None if the citation can't be shown."""
    root = Path(repo_root).resolve()
    full, _ = confine_path(root, citation.get("path"))
    start, end = citation.get("start_line"), citation.get("end_line")
    if full is None or not isinstance(start, int) or not isinstance(end, int):
        return None
    if start < 1 or end < start:
        return None
    try:
        lines = split_lines(full.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return None
    if start > len(lines):
        return None
    end = min(end, len(lines))
    shown_end = min(end, start + max_lines - 1)
    width = len(str(shown_end))
    out = [f"{n:>{width}} | {lines[n - 1].rstrip()}" for n in range(start, shown_end + 1)]
    if shown_end < end:
        out.append(f"... truncated ({end - shown_end} more lines)")
    return "\n".join(out)


def friendly_failure(failure: str | None, secrets: tuple[str | None, ...] = ()
                     ) -> tuple[str, str]:
    """User-facing (message, suggestion) for an ``AgentResult.failure`` value."""
    failure = failure or ""
    if failure.startswith("llm_error:"):
        kind = failure.split(":", 2)[1].strip()
        message, suggestion = _FAILURES.get(kind, ("The model call failed.", _GENERIC[1]))
    else:
        message, suggestion = _FAILURES.get(failure, _GENERIC)
    return scrub(message, secrets), scrub(suggestion, secrets)


def friendly_exception(exc: BaseException, secrets: tuple[str | None, ...] = ()
                       ) -> tuple[str, str]:
    """User-facing (message, suggestion) for an exception raised while answering."""
    if isinstance(exc, ProviderError):
        message, suggestion = _FAILURES.get(type(exc).__name__,
                                            ("The model call failed.", _GENERIC[1]))
    elif isinstance(exc, ValueError):
        message, suggestion = "The provider settings are invalid.", str(exc)
    else:
        message, suggestion = _GENERIC
    return scrub(message, secrets), scrub(suggestion, secrets)


def summarize_trace(trace: list[dict], secrets: tuple[str | None, ...] = ()) -> list[dict]:
    """Tool calls as name / arguments / status, without raw tool output."""
    out = []
    for t in trace:
        try:
            args = json.dumps(t.get("arguments"), ensure_ascii=False)
        except (TypeError, ValueError):
            args = str(t.get("arguments"))
        status = "ok" if not t.get("error") else f"error: {t['error']}"
        out.append({"name": scrub(str(t.get("name", "")), secrets),
                    "arguments": scrub(args[:300], secrets),
                    "status": scrub(status[:300], secrets)})
    return out


# ------------------------------------------------------------------ asking


@dataclass
class AskOutcome:
    """Everything the UI shows for one question. Never contains the API key."""

    ok: bool
    question: str
    mode: str = "model"
    label: str = ""
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    unsupported: bool = False
    failure: str | None = None
    message: str = ""
    suggestion: str = ""
    trace: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    steps: int = 0
    warnings: list[str] = field(default_factory=list)


def outcome_to_dict(outcome: AskOutcome) -> dict:
    """Plain dict for ``st.session_state`` history."""
    return asdict(outcome)


@lru_cache(maxsize=1)
def _load_embedder() -> Any:
    from coderag.embeddings import LocalEmbedder

    return LocalEmbedder()


def _searcher(store: Store, retriever: str) -> tuple[Any, str | None]:
    if retriever != "hybrid":
        return None, None
    try:
        from coderag.hybrid import HybridSearcher

        return HybridSearcher(store, _load_embedder()), None
    except Exception as exc:  # any load failure: tell the user, keep answering with BM25
        log.warning("hybrid retrieval unavailable: %s", type(exc).__name__)
        return None, ("Hybrid retrieval is unavailable (the embedding model did not load); "
                      "using BM25.")


def _fail(question: str, message: str, suggestion: str = "", **kw: Any) -> AskOutcome:
    return AskOutcome(ok=False, question=question, message=message, suggestion=suggestion, **kw)


def run_question(repo: str | Path, question: str, *, provider: str, model: str,
                 key: str | None, base_url: str | None, demo: bool, retriever: str,
                 max_steps: int) -> AskOutcome:
    """Re-index if needed, run the agent, and package the result for display."""
    install()  # idempotent; guarantees scrubbing even when used outside the Streamlit app
    question = (question or "").strip()
    mode = "demo" if demo else "model"
    label = DEMO_LABEL if demo else ""
    if not question:
        return _fail(question, "Type a question first.", mode=mode, label=label)
    if len(question) > MAX_QUESTION_CHARS:
        return _fail(question, f"Questions are limited to {MAX_QUESTION_CHARS} characters.",
                     mode=mode, label=label)
    check = validate_repo(str(repo))
    if not check.ok or check.path is None:
        return _fail(question, check.message, mode=mode, label=label)
    secrets = (key,)
    with secret_scope(key):
        try:
            summary = index(check.path)
            store = Store(summary.db_path)
            try:
                searcher, warning = _searcher(store, retriever)
                warnings = [warning] if warning else []
                toolbox = Toolbox(store, searcher=searcher)
                if demo:
                    llm: LLMClient = scripted_policy(toolbox, question)
                else:
                    if not (model or "").strip():
                        return _fail(question, "Enter a model id for the selected provider.",
                                     "Pick one from the provider's model list.")
                    try:
                        llm = _client_factory(provider, model.strip(), key, base_url or None)
                    except ValueError as exc:
                        message, suggestion = friendly_exception(exc, secrets)
                        return _fail(question, message, suggestion)
                res = Agent(llm, toolbox, max_steps=max_steps).run(question)
            finally:
                store.close()
        except Exception as exc:  # logged inside the secret scope, so the key is scrubbed
            log.exception("question failed")
            message, suggestion = friendly_exception(exc, secrets)
            return _fail(question, message, suggestion, mode=mode, label=label)
        usage = {"input_tokens": res.usage.input_tokens, "output_tokens": res.usage.output_tokens}
        trace = summarize_trace(res.trace, secrets)
        if not res.ok:
            message, suggestion = friendly_failure(res.failure, secrets)
            return _fail(question, message, suggestion, mode=mode, label=label,
                         failure=scrub(res.failure or "", secrets), trace=trace, usage=usage,
                         steps=res.steps, warnings=warnings)
        citations = [{**c, "snippet": read_snippet(check.path, c)} for c in res.citations]
        return AskOutcome(ok=True, question=question, mode=mode, label=label,
                          answer=scrub(res.answer or "", secrets), citations=citations,
                          unsupported=res.unsupported, trace=trace, usage=usage,
                          steps=res.steps, warnings=warnings)
