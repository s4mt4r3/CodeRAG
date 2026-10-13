"""Secret scrubbing for error messages and log records.

Two layers: known key shapes (``sk-...``, ``AIza...``, ``gsk_...``, bearer tokens,
credential headers) are always masked, and exact values registered for the current
request with :func:`secret_scope` are masked too, which covers keys that match no
known pattern. :func:`install_log_scrubbing` applies both to every log record the
process creates, including exception tracebacks.
"""

from __future__ import annotations

import contextvars
import logging
import re
import traceback
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

REDACTED = "[REDACTED]"
MIN_SECRET_LEN = 6

_PATTERNS = [
    (re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=\-]{6,}"), r"\1" + REDACTED),
    (re.compile(r"(?i)\b((?:x-llm-key|x-api-key|x-goog-api-key|authorization|api[_-]?key)"
                r"(?:\s*[:=]\s*|\"\s*:\s*\"))[^\s\"',;&]+"), r"\1" + REDACTED),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"), REDACTED),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"), REDACTED),
    (re.compile(r"\bgsk_[A-Za-z0-9]{10,}"), REDACTED),
    (re.compile(r"\bxai-[A-Za-z0-9]{10,}"), REDACTED),
]
_CURRENT: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "coderag_secrets", default=())


def scrub(text: str, secrets: Iterable[str | None] = ()) -> str:
    """Mask known key shapes plus ``secrets`` and any secrets of the current scope."""
    if not text:
        return text
    values = {s for s in (*secrets, *_CURRENT.get()) if s and len(s) >= MIN_SECRET_LEN}
    for value in sorted(values, key=len, reverse=True):
        text = text.replace(value, REDACTED)
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text


@contextmanager
def secret_scope(*secrets: str | None) -> Iterator[None]:
    """Register exact secret values for scrubbing within this context."""
    extra = tuple(s for s in secrets if s)
    token = _CURRENT.set(_CURRENT.get() + extra)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def _scrub_arg(arg: object) -> object:
    if isinstance(arg, str):
        return scrub(arg)
    if isinstance(arg, int | float | bool | bytes) or arg is None:
        return arg
    text = str(arg)
    cleaned = scrub(text)
    return arg if cleaned == text else cleaned


def scrub_record(record: logging.LogRecord) -> logging.LogRecord:
    """Scrub a record in place, keeping the shape of ``args`` (uvicorn's access
    formatter unpacks them positionally)."""
    if isinstance(record.msg, str):
        record.msg = scrub(record.msg)
    elif record.msg is not None:
        record.msg = _scrub_arg(record.msg)
    if isinstance(record.args, tuple):
        record.args = tuple(_scrub_arg(a) for a in record.args)
    elif isinstance(record.args, dict):
        record.args = {k: _scrub_arg(v) for k, v in record.args.items()}
    if record.exc_info and record.exc_info[0] is not None:
        record.exc_text = scrub("".join(traceback.format_exception(*record.exc_info)).rstrip())
        # Formatters print exc_text when exc_info is cleared, so the raw traceback is gone.
        record.exc_info = None
    elif record.exc_text:
        record.exc_text = scrub(record.exc_text)
    if record.stack_info:
        record.stack_info = scrub(record.stack_info)
    return record


def install_log_scrubbing() -> None:
    """Wrap the process-wide log record factory so every record is scrubbed (idempotent)."""
    current = logging.getLogRecordFactory()
    if getattr(current, "_coderag_scrubbing", False):
        return

    def factory(*args: object, **kwargs: object) -> logging.LogRecord:
        return scrub_record(current(*args, **kwargs))

    factory._coderag_scrubbing = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(factory)
