"""Shared pieces for provider adapters: error hierarchy, retries, history helpers.

Every error message is scrubbed of the API key (and known key shapes) at
construction, and transport exceptions are re-raised without chaining so request
details never ride along in a traceback.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator

import httpx

from coderag.netguard import BlockedDestination, ResponseTooLarge
from coderag.redact import scrub

DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
DEFAULT_MAX_TOKENS = 4096


class ProviderError(Exception):
    """Base class for provider failures; the message never contains the key."""

    def __init__(self, message: str, *, status: int | None = None,
                 secrets: tuple[str | None, ...] = ()) -> None:
        super().__init__(scrub(message, secrets))
        self.status = status


class ProviderAuthError(ProviderError):
    """401/403: bad or unauthorized key."""


class ProviderRateLimitError(ProviderError):
    """429: rate limited (retryable)."""

    def __init__(self, message: str, *, retry_after: float | None = None, **kw) -> None:
        super().__init__(message, **kw)
        self.retry_after = retry_after


class ProviderServerError(ProviderError):
    """5xx: provider-side failure (retryable)."""


class ProviderTimeoutError(ProviderError):
    """The request timed out."""


class ProviderConnectionError(ProviderError):
    """The provider could not be reached."""


class ProviderBlockedError(ProviderError, ValueError):
    """The destination was refused by the SSRF policy."""


class ProviderResponseError(ProviderError):
    """Unexpected status (4xx, redirect) or a malformed/oversized response."""


def _detail(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except (ValueError, UnicodeDecodeError):
        return resp.text[:200]
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        return str(err.get("message", ""))[:200]
    if isinstance(err, str):
        return err[:200]
    return ""


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return float(resp.headers.get("retry-after", ""))
    except ValueError:
        return None


def error_for(resp: httpx.Response, provider: str, secrets: tuple[str | None, ...]
              ) -> ProviderError:
    """Map a non-2xx response to the error hierarchy."""
    status = resp.status_code
    detail = _detail(resp)
    msg = f"{provider} returned HTTP {status}" + (f": {detail}" if detail else "")
    kw = {"status": status, "secrets": secrets}
    if status in (401, 403):
        return ProviderAuthError(msg, **kw)
    if status == 429:
        return ProviderRateLimitError(msg, retry_after=_retry_after(resp), **kw)
    if status >= 500:
        return ProviderServerError(msg, **kw)
    if 300 <= status < 400:
        return ProviderResponseError(f"{provider} returned an unexpected redirect", **kw)
    return ProviderResponseError(msg, **kw)


def send_with_retries(send: Callable[[], httpx.Response], provider: str,
                      secrets: tuple[str | None, ...], retries: int = 2,
                      backoff: float = 0.5, max_wait: float = 10.0,
                      sleep: Callable[[float], None] = time.sleep) -> httpx.Response:
    """Send, retrying only 429 and 5xx a bounded number of times."""
    for attempt in range(retries + 1):
        try:
            resp = send()
        except BlockedDestination as exc:
            raise ProviderBlockedError(f"{provider}: {exc}", secrets=secrets) from None
        except ResponseTooLarge:
            raise ProviderResponseError(f"{provider}: response too large",
                                        secrets=secrets) from None
        except httpx.TimeoutException:
            raise ProviderTimeoutError(f"{provider}: request timed out",
                                       secrets=secrets) from None
        except httpx.HTTPError as exc:
            raise ProviderConnectionError(f"{provider}: connection failed ({type(exc).__name__})",
                                          secrets=secrets) from None
        if 200 <= resp.status_code < 300:
            return resp
        err = error_for(resp, provider, secrets)
        if isinstance(err, ProviderRateLimitError | ProviderServerError) and attempt < retries:
            wait = getattr(err, "retry_after", None) or backoff * 2 ** attempt
            sleep(min(max_wait, wait))
            continue
        raise err
    raise AssertionError("unreachable")


def json_body(resp: httpx.Response, provider: str, secrets: tuple[str | None, ...]) -> dict:
    """Parse a JSON object response or raise :class:`ProviderResponseError`."""
    try:
        data = resp.json()
    except (ValueError, UnicodeDecodeError):
        raise ProviderResponseError(f"{provider}: response was not JSON",
                                    secrets=secrets) from None
    if not isinstance(data, dict):
        raise ProviderResponseError(f"{provider}: response was not a JSON object",
                                    secrets=secrets)
    return data


def blocks_of(content: object) -> list[dict]:
    """Anthropic-style blocks for a message's ``content`` (a string or block list)."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return [b for b in content or [] if isinstance(b, dict)]


def tool_result_text(block: dict) -> str:
    """Text of a ``tool_result`` block, prefixed when it reports an error."""
    content = block.get("content", "")
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    text = str(content)
    return f"ERROR (tool call failed): {text}" if block.get("is_error") else text


def parse_arguments(raw: object) -> object:
    """Tool arguments as a dict when they parse, else the raw value (the agent rejects it)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        return value if isinstance(value, dict) else raw
    return raw


def iter_tool_uses(messages: list[dict]) -> Iterator[dict]:
    """Every ``tool_use`` block in an Anthropic-style history."""
    for m in messages:
        if m.get("role") == "assistant":
            for b in blocks_of(m.get("content")):
                if b.get("type") == "tool_use":
                    yield b


def check_key(api_key: str | None, required: bool) -> str | None:
    """Reject missing keys (when required) and keys that could inject headers."""
    if not api_key:
        if required:
            raise ValueError("an API key is required for this provider")
        return None
    if len(api_key) > 512 or any(ord(ch) < 33 or ord(ch) > 126 for ch in api_key):
        raise ValueError("API key contains invalid characters")
    return api_key
