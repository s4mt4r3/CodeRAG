"""Command line: ``coderag index``, ``coderag ask``, ``coderag eval``."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from coderag.agent import Agent
from coderag.evaluation import llm_from_env, run_eval, write_reports
from coderag.llm import LLMClient, ScriptedLLM
from coderag.providers.registry import (
    PROVIDERS,
    build_client,
    key_env_names,
    key_from_env,
)
from coderag.redact import scrub
from coderag.store import Store, index_repo
from coderag.tools import Toolbox

app = typer.Typer(add_completion=False, help="Code-aware Q&A over a Python repository.")
DEFAULT_DB_DIR = Path(".coderag")


def default_db(repo: Path, db_dir: Path = DEFAULT_DB_DIR) -> Path:
    """Per-repo index path: ``<db_dir>/<name>-<hash of absolute path>.db``."""
    digest = hashlib.sha1(str(repo.resolve()).encode()).hexdigest()[:8]
    return db_dir / f"{repo.resolve().name}-{digest}.db"


def _load_script(path: Path) -> ScriptedLLM:
    turns = []
    for item in json.loads(path.read_text()):
        if "final" in item:
            turns.append(ScriptedLLM.final(item["final"]))
        else:
            turns.append(ScriptedLLM.tool(item["tool"], item.get("arguments", {})))
    return ScriptedLLM(turns)


def _searcher(store: Store, retriever: str):
    if retriever == "bm25":
        return None
    try:
        from coderag.embeddings import LocalEmbedder
        from coderag.hybrid import HybridSearcher

        return HybridSearcher(store, LocalEmbedder())
    except Exception as exc:  # model missing or unloadable: say so and use BM25
        typer.echo(f"note: embedding model unavailable ({type(exc).__name__}); using BM25 only")
        return None


def _provider_llm(provider: str, model: str | None, base_url: str | None) -> LLMClient:
    model = model or os.environ.get("CODERAG_MODEL")
    if not model:
        typer.echo("error: pass --model (or set CODERAG_MODEL).")
        raise typer.Exit(code=2)
    key = key_from_env(provider)
    spec = PROVIDERS.get(provider.strip().lower())
    if spec is not None and not (spec.local_only or spec.requires_base_url) and not key:
        typer.echo(f"error: no API key for {spec.name}; set one of: "
                   f"{', '.join(key_env_names(provider))}")
        raise typer.Exit(code=2)
    try:
        return build_client(provider, model, key, base_url)
    except ValueError as exc:
        typer.echo(f"error: {scrub(str(exc), (key,))}")
        raise typer.Exit(code=2) from None


@app.command()
def index(
    repo: Path,
    db: Annotated[Path | None, typer.Option(help="Index database path.")] = None,
    chunker: Annotated[str, typer.Option(help="ast or fixed")] = "ast",
) -> None:
    """Index (or incrementally re-index) a repository."""
    db = db or default_db(repo)
    db.parent.mkdir(parents=True, exist_ok=True)
    stats = index_repo(repo, db, chunker=chunker)
    total = stats.added + stats.updated + stats.unchanged
    typer.echo(f"indexed {total} files ({stats.added} added, {stats.updated} updated, "
               f"{stats.removed} removed, {stats.unchanged} unchanged), {stats.chunks} chunks "
               f"-> {db}")
    for w in stats.warnings[:10]:
        typer.echo(f"warning: {w}")


@app.command()
def ask(
    repo: Path,
    question: str,
    db: Annotated[Path | None, typer.Option(help="Index database path.")] = None,
    retriever: Annotated[str, typer.Option(help="hybrid or bm25")] = "hybrid",
    scripted: Annotated[Path | None, typer.Option(help="JSON script for an offline LLM.")] = None,
    max_steps: int = 8,
    provider: Annotated[str | None, typer.Option(
        help="LLM provider (see GET /providers). The key is read from CODERAG_LLM_KEY or the "
             "provider's own env var, never from the command line.")] = None,
    model: Annotated[str | None, typer.Option(help="Model id (default: CODERAG_MODEL).")] = None,
    base_url: Annotated[str | None, typer.Option(
        help="Base URL, only for --provider openai_compatible.")] = None,
) -> None:
    """Answer a question about a repository, with verified citations."""
    if scripted:
        llm: LLMClient | None = _load_script(scripted)
    elif provider:
        llm = _provider_llm(provider, model, base_url)
    else:
        llm = llm_from_env()
    if llm is None:
        typer.echo("error: set ANTHROPIC_API_KEY and CODERAG_MODEL (or pass --scripted).")
        raise typer.Exit(code=2)
    db = db or default_db(repo)
    db.parent.mkdir(parents=True, exist_ok=True)
    index_repo(repo, db)
    store = Store(db)
    res = Agent(llm, Toolbox(store, searcher=_searcher(store, retriever)),
                max_steps=max_steps).run(question)
    if not res.ok:
        typer.echo(f"FAILED: {res.failure}")
        raise typer.Exit(code=1)
    typer.echo(res.answer or "")
    if res.unsupported:
        typer.echo("UNSUPPORTED: no citation could be verified against the repository.")
    for c in res.citations:
        mark = "verified" if c["verified"] else f"UNVERIFIED ({c['reason']})"
        typer.echo(f"  - {c['path']}:{c['start_line']}-{c['end_line']} {c['symbol']} [{mark}]")
    typer.echo(f"steps={res.steps} tokens_in={res.usage.input_tokens} "
               f"tokens_out={res.usage.output_tokens}")


@app.command("eval")
def eval_cmd(
    repo: Path,
    out: Annotated[Path, typer.Option(help="Report directory.")] = Path("reports"),
    db_dir: Annotated[Path, typer.Option(help="Where index databases live.")] = Path(
        "data/index"),
    db_prefix: Annotated[str | None, typer.Option()] = None,
    n: Annotated[int, typer.Option(help="Retrieval questions.")] = 200,
    agent_n: Annotated[int, typer.Option(help="Agent questions.")] = 30,
    seed: int = 0,
    embeddings: Annotated[bool, typer.Option(help="Use the local embedding model.")] = True,
    repo_name: Annotated[str | None, typer.Option()] = None,
    sha: Annotated[str | None, typer.Option(help="Expected git HEAD of the repo.")] = None,
) -> None:
    """Run retrieval, agent, and chunking-ablation evals; write reports."""
    embedder = None
    if embeddings:
        from coderag.embeddings import LocalEmbedder

        embedder = LocalEmbedder()
    report = run_eval(repo, db_dir=db_dir, n_questions=n, seed=seed, agent_n=agent_n,
                      embedder=embedder, llm=llm_from_env(), repo_name=repo_name, sha=sha,
                      db_prefix=db_prefix)
    write_reports(report, out)
    a = report["agent"]
    typer.echo(f"wrote {out}/eval.json, eval.md, ablation_chunking.json, ablation_chunking.md "
               f"(agent mode: {a['mode']})")


def _streamlit_available() -> bool:
    return importlib.util.find_spec("streamlit") is not None


@app.command("app")
def app_cmd(
    port: Annotated[int, typer.Option(help="Port to serve on.")] = 8501,
    host: Annotated[str, typer.Option(help="Interface to bind; keep 127.0.0.1 for local "
                                           "use.")] = "127.0.0.1",
) -> None:
    """Launch the web app (Streamlit). Keys are entered in the page or read from env vars,
    never passed on the command line."""
    if not _streamlit_available():
        typer.echo("error: the web app needs Streamlit: pip install -e '.[app]'")
        raise typer.Exit(code=2)
    script = Path(__file__).with_name("streamlit_app.py")
    argv = [sys.executable, "-m", "streamlit", "run", str(script),
            "--server.address", host, "--server.port", str(port)]
    typer.echo(f"coderag web app: http://{host}:{port}  (Ctrl+C to stop)")
    raise typer.Exit(code=subprocess.call(argv))


if __name__ == "__main__":
    app()
