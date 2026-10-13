"""FastAPI app: ``GET /health``, ``GET /providers``, ``POST /index``, ``POST /ask``.

Repository paths must resolve inside one of the configured allowed roots
(``CODERAG_ALLOWED_ROOTS``, ``os.pathsep``-separated, when not passed explicitly).

Bring your own LLM: ``/ask`` accepts ``X-LLM-Provider``, ``X-LLM-Model``, ``X-LLM-Key``
and, for ``openai_compatible`` only, ``X-LLM-Base-URL``. The client is built per
request; the key lives only in local variables and the request's secret scope, and
is scrubbed from responses and every log record. Without these headers the
server-configured LLM is used, unless ``CODERAG_BYO_LLM_ONLY=1``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from coderag.agent import Agent
from coderag.apiguard import (
    ApiSettings,
    BodyLimitMiddleware,
    HttpsOnlyMiddleware,
    RateLimiter,
    client_identity,
    require_token,
)
from coderag.cli import default_db
from coderag.evaluation import llm_from_env
from coderag.ingest import walk_repo
from coderag.llm import LLMClient
from coderag.providers.registry import build_client, list_providers
from coderag.redact import install_log_scrubbing, scrub, secret_scope
from coderag.store import Store, index_repo
from coderag.tools import Toolbox

log = logging.getLogger("coderag.api")
ClientFactory = Callable[[str, str, "str | None", "str | None"], LLMClient]


class IndexRequest(BaseModel):
    """Body of ``POST /index``."""

    repo: str = Field(min_length=1, max_length=4096)
    chunker: str = Field(default="ast", pattern="^(ast|fixed)$")


class AskRequest(BaseModel):
    """Body of ``POST /ask``. Credentials are never read from the body."""

    repo: str = Field(min_length=1, max_length=4096)
    question: str = Field(min_length=1, max_length=4000)
    max_steps: int = Field(default=8, ge=1, le=20)


def _default_client_factory(provider: str, model: str, key: str | None,
                            base_url: str | None) -> LLMClient:
    return build_client(provider, model, key, base_url)


def create_app(
    allowed_roots: list[str | Path] | None = None,
    db_dir: str | Path | None = None,
    llm_factory: Callable[[], LLMClient | None] | None = None,
    searcher_factory: Callable[[Store], Any] | None = None,
    *,
    settings: ApiSettings | None = None,
    client_factory: ClientFactory | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> FastAPI:
    """Build the app; nothing outside ``allowed_roots`` can be indexed or queried."""
    install_log_scrubbing()
    if allowed_roots is None:
        env = os.environ.get("CODERAG_ALLOWED_ROOTS", "")
        allowed_roots = [p for p in env.split(os.pathsep) if p]
    roots = [Path(r).resolve() for r in allowed_roots]
    dbs = Path(db_dir or os.environ.get("CODERAG_DB_DIR", ".coderag"))
    make_llm = llm_factory or llm_from_env
    settings = settings or ApiSettings.from_env()
    make_client = client_factory or _default_client_factory
    limiter = RateLimiter(clock=clock)
    index_slots = threading.BoundedSemaphore(max(1, settings.index_concurrency))

    app = FastAPI(title="coderag", version="0.2.0")
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(HttpsOnlyMiddleware, settings=settings)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic echoes offending input values; strip them so nothing sensitive leaks.
        errors = [{"loc": list(e.get("loc", ())), "msg": e.get("msg", ""),
                   "type": e.get("type", "")} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled error on %s: %s", request.url.path, type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    def guard(bucket: str, limit_of: Callable[[ApiSettings], int]):
        def dependency(request: Request) -> None:
            require_token(request, settings)
            wait = limiter.hit(bucket, client_identity(request, settings), limit_of(settings))
            if wait is not None:
                raise HTTPException(status_code=429, detail="rate limit exceeded",
                                    headers={"Retry-After": str(int(wait))})
        return dependency

    def confine(repo: str) -> Path:
        path = Path(repo).resolve()
        if not any(path == r or path.is_relative_to(r) for r in roots):
            raise HTTPException(status_code=403, detail="repository is outside the allowlist")
        if not path.is_dir():
            raise HTTPException(status_code=404, detail="repository directory not found")
        return path

    def resolve_llm(request: Request) -> tuple[LLMClient | None, str | None, dict]:
        h = request.headers
        provider, model = h.get("x-llm-provider"), h.get("x-llm-model")
        key, base_url = h.get("x-llm-key"), h.get("x-llm-base-url")
        if not any((provider, model, key, base_url)):
            if settings.byo_llm_only:
                raise HTTPException(status_code=401,
                                    detail="X-LLM-Provider, X-LLM-Model and X-LLM-Key required")
            return make_llm(), None, {"source": "server"}
        if not provider or not model:
            raise HTTPException(status_code=400,
                                detail="X-LLM-Provider and X-LLM-Model are both required")
        try:
            llm = make_client(provider, model, key, base_url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=scrub(str(exc), (key,))) from None
        return llm, key, {"source": "request", "provider": provider, "model": model}

    @app.get("/health")
    def health() -> dict:
        """Liveness probe."""
        return {"status": "ok", "allowed_roots": len(roots)}

    @app.get("/providers")
    def providers() -> dict:
        """Supported LLM providers (no secrets)."""
        return {"providers": list_providers()}

    @app.post("/index", dependencies=[Depends(guard("index", lambda s: s.index_per_minute))])
    def index(req: IndexRequest) -> dict:
        """Index or incrementally re-index an allowlisted repository."""
        repo = confine(req.repo)
        if not index_slots.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="indexing is busy",
                                headers={"Retry-After": "5"})
        try:
            files = walk_repo(repo)
            if len(files) > settings.max_repo_files or \
                    sum(len(f.text.encode("utf-8")) for f in files) > settings.max_repo_bytes:
                raise HTTPException(status_code=413, detail="repository exceeds size limits")
            db = default_db(repo, dbs)
            db.parent.mkdir(parents=True, exist_ok=True)
            stats = index_repo(repo, db, chunker=req.chunker)
        finally:
            index_slots.release()
        return {"files": stats.added + stats.updated + stats.unchanged, "added": stats.added,
                "updated": stats.updated, "removed": stats.removed,
                "unchanged": stats.unchanged, "chunks": stats.chunks,
                "warnings": stats.warnings[:20]}

    @app.post("/ask", dependencies=[Depends(guard("ask", lambda s: s.ask_per_minute))])
    def ask(req: AskRequest, request: Request) -> dict:
        """Answer a question about an indexed repository with verified citations."""
        repo = confine(req.repo)
        db = default_db(repo, dbs)
        if not db.exists():
            raise HTTPException(status_code=409, detail="repository is not indexed; POST /index")
        key = request.headers.get("x-llm-key")
        with secret_scope(key):
            try:
                llm, key, source = resolve_llm(request)
                if llm is None:
                    raise HTTPException(status_code=503, detail="no LLM configured")
                log.info("ask source=%s provider=%s model=%s", source["source"],
                         source.get("provider", "-"), source.get("model", "-"))
                store = Store(db)
                try:
                    searcher = searcher_factory(store) if searcher_factory else None
                    res = Agent(llm, Toolbox(store, searcher=searcher),
                                max_steps=min(req.max_steps, settings.max_steps)
                                ).run(req.question)
                finally:
                    store.close()
            except HTTPException:
                raise
            except Exception:
                # Logged inside the secret scope so even pattern-less keys are scrubbed.
                log.exception("ask failed")
                raise HTTPException(status_code=500, detail="internal server error") from None
            return {"ok": res.ok, "answer": res.answer, "citations": res.citations,
                    "unsupported": res.unsupported,
                    "failure": scrub(res.failure, (key,)) if res.failure else None,
                    "steps": res.steps,
                    "usage": {"input_tokens": res.usage.input_tokens,
                              "output_tokens": res.usage.output_tokens}}

    return app
