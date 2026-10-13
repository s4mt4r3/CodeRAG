"""One-command evaluation: retrieval metrics, agent pipeline metrics, chunking ablation.

Agent-level metrics use a scripted policy by default (search, read the top hit, cite
it). Those numbers check the pipeline (tool dispatch, citation verification,
grounding against oracle truth); they say nothing about a real model's quality. A
real model is used only when ``ANTHROPIC_API_KEY`` and ``CODERAG_MODEL`` are set.
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from coderag import benchmark
from coderag.agent import Agent
from coderag.evalgen import Question, generate_questions
from coderag.hybrid import HybridSearcher
from coderag.llm import AnthropicClient, LLMClient, ScriptedLLM
from coderag.metrics import estimate_cost, hit, percentile
from coderag.providers.registry import build_client, key_from_env
from coderag.store import Store, index_repo
from coderag.tools import Toolbox

# USD per million tokens (input, output), from Anthropic's published price list.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}
SCRIPTED_LABEL = (
    "Pipeline check with a scripted LLM (search -> read top hit -> cite it). Measures tool "
    "dispatch, citation verification and grounding plumbing, NOT model answer quality. "
    "Token counts are synthetic scripted values."
)
REAL_LABEL = "Real model run via the {provider} API."


def _prices(model: str) -> tuple[float | None, float | None]:
    env_in, env_out = os.environ.get("CODERAG_PRICE_IN"), os.environ.get("CODERAG_PRICE_OUT")
    if env_in and env_out:
        return float(env_in), float(env_out)
    return PRICES.get(model, (None, None))


def llm_from_env() -> LLMClient | None:
    """The configured real LLM, or None for scripted mode.

    ``CODERAG_PROVIDER`` (+ ``CODERAG_MODEL``, key from ``CODERAG_LLM_KEY`` or the
    provider's own variable, optional ``CODERAG_BASE_URL``) selects any provider;
    otherwise ANTHROPIC_API_KEY + CODERAG_MODEL selects Anthropic as before.
    """
    provider = os.environ.get("CODERAG_PROVIDER")
    if provider:
        model = os.environ.get("CODERAG_MODEL")
        if not model:
            raise ValueError("CODERAG_PROVIDER is set but CODERAG_MODEL is not")
        return build_client(provider, model, key_from_env(provider),
                            os.environ.get("CODERAG_BASE_URL") or None)
    if os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("CODERAG_MODEL"):
        return AnthropicClient()
    return None


def scripted_policy(toolbox: Toolbox, question: str) -> ScriptedLLM:
    """Fixed demo policy: search, read the top hit, cite it (also answers the repair round)."""
    hits = toolbox.search_code(question, k=5)
    turns = [ScriptedLLM.tool("search_code", {"query": question, "k": 5})]
    cites = []
    if hits:
        top = hits[0]
        turns.append(ScriptedLLM.tool("read_file", {"path": top["path"],
                                                    "start": top["start_line"],
                                                    "end": top["end_line"]}))
        cites = [{"path": top["path"], "start_line": top["start_line"],
                  "end_line": top["end_line"], "symbol": top["symbol"].rsplit(".", 1)[-1]}]
    final = json.dumps({"answer": f"See {cites[0]['path']}." if cites else "Not found.",
                        "citations": cites})
    # The policy cannot fix citations, so the repair round gets the same answer.
    turns += [ScriptedLLM.final(final), ScriptedLLM.final(final)]
    return ScriptedLLM(turns)


def _agent_eval(store: Store, questions: list[Question], searcher: Any,
                llm: LLMClient | None) -> dict:
    toolbox = Toolbox(store, searcher=searcher)
    mode = "real" if llm is not None else "scripted"
    model = getattr(llm, "model", "scripted")
    provider = getattr(llm, "provider", "unknown") if llm is not None else "scripted"
    latencies, rows = [], []
    tok_in = tok_out = 0
    for q in questions:
        client = llm if llm is not None else scripted_policy(toolbox, q.question)
        t0 = time.perf_counter()
        res = Agent(client, toolbox).run(q.question)
        latencies.append((time.perf_counter() - t0) * 1000)
        tok_in += res.usage.input_tokens
        tok_out += res.usage.output_tokens
        truths = [tuple(t) for t in q.truth]
        grounded = res.ok and any(
            c.get("verified") and any(hit((c["path"], [(c["start_line"], c["end_line"])]), t)
                                      for t in truths)
            for c in res.citations)
        rows.append({"id": q.id, "type": q.type, "ok": res.ok, "failure": res.failure,
                     "unsupported": res.unsupported, "grounded": grounded,
                     "citations": len(res.citations),
                     "verified": sum(1 for c in res.citations if c.get("verified"))})
    n = len(rows)
    total_c = sum(r["citations"] for r in rows)
    p_in, p_out = _prices(model) if mode == "real" else (None, None)
    return {
        "mode": mode,
        "label": REAL_LABEL.format(provider=provider) if mode == "real" else SCRIPTED_LABEL,
        "provider": provider,
        "model": model,
        "n_questions": n,
        "citation_accuracy": sum(r["verified"] for r in rows) / total_c if total_c else None,
        "answer_grounded_rate": sum(r["grounded"] for r in rows) / n if n else None,
        "unsupported_rate": sum(r["ok"] and r["unsupported"] for r in rows) / n if n else None,
        "failure_rate": sum(not r["ok"] for r in rows) / n if n else None,
        "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)},
        "tokens": {"input": tok_in, "output": tok_out,
                   "mean_per_question": (tok_in + tok_out) / n if n else None},
        "estimated_cost_usd": estimate_cost(tok_in, tok_out, p_in, p_out),
        "price_per_mtok": {"input": p_in, "output": p_out},
        "per_question": rows,
    }


def _mean_lines(store: Store, questions: list[Question], search, k: int = 5) -> float | None:
    sizes = []
    for q in questions:
        for _, segs in search(q.question, k):
            sizes.append(sum(b - a + 1 for a, b in segs))
    return sum(sizes) / len(sizes) if sizes else None


def _summ(res: dict) -> dict:
    return {"overall": res["overall"], "by_type": res["by_type"]}


def run_eval(repo: str | Path, db_dir: str | Path, n_questions: int = 200, seed: int = 0,
             agent_n: int = 30, embedder: Any = None, llm: LLMClient | None = None,
             repo_name: str | None = None, sha: str | None = None,
             db_prefix: str | None = None) -> dict:
    """Run every evaluation and return one JSON-serializable report."""
    repo = Path(repo).resolve()
    if sha is not None and benchmark.git_head(repo) != sha:
        raise ValueError("repository HEAD does not match the expected SHA")
    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    prefix = db_prefix or repo.name
    ast_db, fixed_db = db_dir / f"{prefix}_ast.db", db_dir / f"{prefix}_fixed.db"
    index_repo(repo, ast_db)
    index_repo(repo, fixed_db, chunker="fixed")
    store, fstore = Store(ast_db), Store(fixed_db)
    questions = generate_questions(repo, n=n_questions, seed=seed)

    if embedder is not None:
        retrieval = benchmark.compare_retrieval(store, questions, embedder)
        searcher = HybridSearcher(store, embedder)
    else:
        retrieval = {"bm25": _summ(benchmark.run(questions, benchmark.bm25_search(store)))}
        searcher = None

    pool = sorted(questions, key=lambda q: q.id)
    sample = random.Random(seed).sample(pool, min(agent_n, len(pool)))
    agent = _agent_eval(store, sample, searcher, llm)

    ablation: dict = {"retrievers": ["bm25"] + (["hybrid"] if embedder else []),
                      "fixed_config": {"size": 40, "overlap": 10}, "results": {}}
    for name, st in (("ast", store), ("fixed", fstore)):
        bm = benchmark.bm25_search(st)
        entry = {"chunks": len(st.all_chunks()), "bm25": _summ(benchmark.run(questions, bm)),
                 "mean_lines_per_top5_chunk": _mean_lines(st, questions, bm)}
        if embedder is not None:
            hs = HybridSearcher(st, embedder)
            entry["hybrid"] = _summ(benchmark.run(
                questions, lambda q, k, hs=hs: benchmark.as_retrieved(hs.search(q, k))))
        ablation["results"][name] = entry

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "eval_repo": {"name": repo_name or repo.name, "sha": sha, "seed": seed,
                      "n_questions": len(questions)},
        "retrieval": retrieval,
        "agent": agent,
        "ablation_chunking": ablation,
    }


# ------------------------------------------------------------------ rendering


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _retrieval_table(retrieval: dict) -> list[str]:
    systems = [s for s in ("bm25", "vector", "hybrid") if s in retrieval]
    types = sorted(retrieval["bm25"]["by_type"])
    lines = ["| type | system | n | recall@1 | recall@5 | recall@10 | MRR |",
             "|---|---|---|---|---|---|---|"]
    for t in [*types, "overall"]:
        for s in systems:
            r = retrieval[s]["overall"] if t == "overall" else retrieval[s]["by_type"][t]
            lines.append(f"| {t} | {s} | {r['n']} | {_fmt(r['recall_at_1'])} | "
                         f"{_fmt(r['recall_at_5'])} | {_fmt(r['recall_at_10'])} | "
                         f"{_fmt(r['mrr'])} |")
    return lines


def render_ablation(report: dict) -> str:
    """Markdown table for the AST vs fixed-size chunking ablation."""
    abl = report["ablation_chunking"]
    lines = ["## Chunking ablation (AST vs fixed-size, same retriever)", "",
             f"Fixed-size chunks: {abl['fixed_config']['size']} lines, "
             f"{abl['fixed_config']['overlap']} overlap. A retrieved chunk hits when it covers "
             "at least half of the truth span, so larger chunks hit more easily; compare "
             "the lines-per-chunk column.", "",
             "| chunker | retriever | chunks | lines per top-5 chunk | recall@1 | recall@5 | "
             "recall@10 | MRR |",
             "|---|---|---|---|---|---|---|---|"]
    for name in ("ast", "fixed"):
        e = abl["results"][name]
        for retr in abl["retrievers"]:
            o = e[retr]["overall"]
            lines.append(f"| {name} | {retr} | {e['chunks']} | "
                         f"{_fmt(e['mean_lines_per_top5_chunk'], 1)} | {_fmt(o['recall_at_1'])} | "
                         f"{_fmt(o['recall_at_5'])} | {_fmt(o['recall_at_10'])} | "
                         f"{_fmt(o['mrr'])} |")
    return "\n".join(lines) + "\n"


def render_markdown(report: dict) -> str:
    """Human-readable report; every value comes from ``report``."""
    repo, agent = report["eval_repo"], report["agent"]
    lines = ["# coderag evaluation", "",
             f"Repo: {repo['name']} (sha {_fmt(repo['sha'])}), {repo['n_questions']} "
             f"auto-generated questions, seed {repo['seed']}. Generated {report['generated_at']}.",
             "", "## Retrieval", "", *_retrieval_table(report["retrieval"]), "",
             "## Agent", "",
             f"Mode: **{agent['mode']}** ({agent.get('provider', 'n/a')} / {agent['model']}). "
             f"{agent['label']}",
             "", "| metric | value |", "|---|---|",
             f"| questions | {agent['n_questions']} |",
             f"| citation accuracy (verified / cited) | {_fmt(agent['citation_accuracy'])} |",
             f"| answer grounded rate | {_fmt(agent['answer_grounded_rate'])} |",
             f"| unsupported rate | {_fmt(agent['unsupported_rate'])} |",
             f"| failure rate | {_fmt(agent['failure_rate'])} |",
             f"| latency p50 (ms) | {_fmt(agent['latency_ms']['p50'], 1)} |",
             f"| latency p95 (ms) | {_fmt(agent['latency_ms']['p95'], 1)} |",
             f"| tokens in / out | {agent['tokens']['input']} / {agent['tokens']['output']} |",
             f"| estimated cost (USD) | {_fmt(agent['estimated_cost_usd'])} |", "",
             render_ablation(report)]
    return "\n".join(lines)


def write_reports(report: dict, out_dir: str | Path) -> None:
    """Write eval.json/.md and ablation_chunking.json/.md."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "eval.md").write_text(render_markdown(report))
    abl = {"eval_repo": report["eval_repo"], **report["ablation_chunking"]}
    (out / "ablation_chunking.json").write_text(json.dumps(abl, indent=2) + "\n")
    (out / "ablation_chunking.md").write_text("# Chunking ablation\n\n" + render_ablation(report))
