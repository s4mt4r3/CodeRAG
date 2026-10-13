"""Background jobs that run on an asyncio event loop."""

from __future__ import annotations

import asyncio
import functools

from shopcore.pricing import tax_rate
from storage.cache import TTLCache, memoize

from .notifications import Notifier, broadcast

_rate_cache = TTLCache(ttl=300.0)


def retry(times: int = 3):
    """Retry an async job up to ``times`` attempts before re-raising."""

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for _ in range(times):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator


@memoize(_rate_cache)
def cached_tax_rate(region: str) -> float:
    """Tax rate lookup memoized in a five-minute cache."""
    return tax_rate(region)


@retry(times=2)
async def send_daily_digest(notifier: Notifier, customers: list, body: str) -> int:
    """Broadcast the daily digest, retrying transient failures."""
    return await broadcast(notifier, customers, body)


class JobRunner:
    """Runs a batch of coroutines with bounded concurrency."""

    def __init__(self, limit: int = 4) -> None:
        self._sem = asyncio.Semaphore(limit)

    async def run_one(self, coro):
        async with self._sem:
            return await coro

    async def run_all(self, coros: list) -> list:
        """Run every coroutine, at most ``limit`` at a time."""
        return await asyncio.gather(*(self.run_one(c) for c in coros))
