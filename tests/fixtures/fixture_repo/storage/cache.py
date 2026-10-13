"""A tiny TTL cache used by the repository layer."""

import time
from typing import Any, Callable


class TTLCache:
    """Key-value cache whose entries expire after ``ttl`` seconds."""

    def __init__(self, ttl: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.ttl = ttl
        self._clock = clock
        self._data: dict[str, tuple[float, Any]] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Return a live entry or ``default`` if missing or expired."""
        entry = self._data.get(key)
        if entry is None:
            return default
        stored_at, value = entry
        if self._is_expired(stored_at):
            self._data.pop(key, None)
            return default
        return value

    def set(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key`` with the current timestamp."""
        self._data[key] = (self._clock(), value)

    def _is_expired(self, stored_at: float) -> bool:
        return self._clock() - stored_at > self.ttl

    def clear(self) -> None:
        """Drop every entry."""
        self._data.clear()


def memoize(cache: TTLCache):
    """Decorator that memoizes a single-argument function in ``cache``."""

    def decorator(func):
        def wrapper(arg):
            key = f"{func.__name__}:{arg}"
            hit = cache.get(key)
            if hit is not None:
                return hit
            result = func(arg)
            cache.set(key, result)
            return result

        return wrapper

    return decorator
