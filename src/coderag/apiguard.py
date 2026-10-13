"""API hardening: settings, bearer auth, rate limiting, body-size and HTTPS middleware."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import HTTPException, Request


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value and value.strip().isdigit() else default


@dataclass(frozen=True)
class ApiSettings:
    """Server-side limits and switches (see ``.env.example``)."""

    api_token: str | None = None
    byo_llm_only: bool = False
    require_https: bool = False
    trust_proxy: bool = False
    ask_per_minute: int = 30
    index_per_minute: int = 10
    max_body_bytes: int = 64_000
    max_steps: int = 12
    max_repo_files: int = 5_000
    max_repo_bytes: int = 50_000_000
    index_concurrency: int = 2

    @classmethod
    def from_env(cls) -> ApiSettings:
        """Read ``CODERAG_*`` environment variables."""
        return cls(
            api_token=os.environ.get("CODERAG_API_TOKEN") or None,
            byo_llm_only=os.environ.get("CODERAG_BYO_LLM_ONLY") == "1",
            require_https=os.environ.get("CODERAG_REQUIRE_HTTPS") == "1",
            trust_proxy=os.environ.get("CODERAG_TRUST_PROXY") == "1",
            ask_per_minute=_int_env("CODERAG_RATE_ASK_PER_MIN", 30),
            index_per_minute=_int_env("CODERAG_RATE_INDEX_PER_MIN", 10),
            max_body_bytes=_int_env("CODERAG_MAX_BODY_BYTES", 64_000),
            max_steps=_int_env("CODERAG_MAX_STEPS", 12),
            max_repo_files=_int_env("CODERAG_MAX_REPO_FILES", 5_000),
            max_repo_bytes=_int_env("CODERAG_MAX_REPO_BYTES", 50_000_000),
            index_concurrency=_int_env("CODERAG_INDEX_CONCURRENCY", 2),
        )


class RateLimiter:
    """Sliding-window limiter: at most ``limit`` events per ``window`` seconds per key."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, window: float = 60.0,
                 max_keys: int = 10_000) -> None:
        self.clock = clock
        self.window = window
        self.max_keys = max_keys
        self._events: dict[tuple[str, str], deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, bucket: str, key: str, limit: int) -> float | None:
        """Record an event; returns None if allowed, else seconds until retry."""
        now = self.clock()
        with self._lock:
            if len(self._events) > self.max_keys:
                self._events = {k: v for k, v in self._events.items()
                                if v and now - v[-1] < self.window}
            q = self._events.setdefault((bucket, key), deque())
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= limit:
                return max(1.0, math.ceil(self.window - (now - q[0])))
            q.append(now)
            return None


def client_identity(request: Request, settings: ApiSettings) -> str:
    """Rate-limit key: the hashed server token when auth is on, else the client IP."""
    auth = request.headers.get("authorization", "")
    if settings.api_token and auth.lower().startswith("bearer "):
        return "tok:" + hashlib.sha256(auth[7:].strip().encode()).hexdigest()[:16]
    if settings.trust_proxy and request.headers.get("x-forwarded-for"):
        return "ip:" + request.headers["x-forwarded-for"].split(",")[0].strip()
    return "ip:" + (request.client.host if request.client else "unknown")


def require_token(request: Request, settings: ApiSettings) -> None:
    """Enforce ``Authorization: Bearer <CODERAG_API_TOKEN>`` when a token is configured."""
    if not settings.api_token:
        return
    auth = request.headers.get("authorization", "")
    given = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    if not given or not hmac.compare_digest(given.encode(), settings.api_token.encode()):
        raise HTTPException(status_code=401, detail="missing or invalid bearer token",
                            headers={"WWW-Authenticate": "Bearer"})


def _json_response(status: int, detail: str) -> tuple[dict, dict]:
    body = json.dumps({"detail": detail}).encode()
    start = {"type": "http.response.start", "status": status,
             "headers": [(b"content-type", b"application/json"),
                         (b"content-length", str(len(body)).encode())]}
    return start, {"type": "http.response.body", "body": body}


class HttpsOnlyMiddleware:
    """Reject non-HTTPS requests (except ``/health``) when ``require_https`` is on."""

    def __init__(self, app, settings: ApiSettings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http" and self.settings.require_https \
                and scope.get("path") != "/health":
            scheme = scope.get("scheme", "http")
            if self.settings.trust_proxy:
                for name, value in scope.get("headers", []):
                    if name == b"x-forwarded-proto":
                        scheme = value.decode("latin-1").split(",")[0].strip().lower()
            if scheme != "https":
                start, body = _json_response(400, "HTTPS is required")
                await send(start)
                await send(body)
                return
        await self.app(scope, receive, send)


class BodyLimitMiddleware:
    """Return 413 for request bodies over ``max_bytes`` (declared or streamed)."""

    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers", []):
            if name == b"content-length" and value.isdigit() and int(value) > self.max_bytes:
                await self._reject(send)
                return
        chunks, total = [], 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                chunks.append(message)
                break
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                await self._reject(send)
                return
            chunks.append(message)
            if not message.get("more_body"):
                break
        replay = iter(chunks)

        async def replay_receive():
            try:
                return next(replay)
            except StopIteration:
                return await receive()

        await self.app(scope, replay_receive, send)

    @staticmethod
    async def _reject(send) -> None:
        start, body = _json_response(413, "request body too large")
        await send(start)
        await send(body)
