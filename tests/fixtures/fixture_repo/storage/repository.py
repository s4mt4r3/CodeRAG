"""In-memory order repository with a read-through cache."""

from __future__ import annotations

import storage.serializers as ser
from shopcore.models import Order

from .cache import TTLCache


class RepositoryError(Exception):
    """Raised when an order cannot be found or stored."""


class OrderRepository:
    """Stores serialized orders keyed by id."""

    def __init__(self, cache: TTLCache | None = None) -> None:
        self._rows: dict[int, dict] = {}
        self._cache = cache or TTLCache(ttl=30.0)

    def save(self, order: Order) -> None:
        """Persist an order, replacing any previous version."""
        row = ser.order_to_dict(order)
        if not ser.validate(row):
            raise RepositoryError(f"invalid order {order.order_id}")
        self._rows[order.order_id] = row
        self._cache.set(str(order.order_id), order)

    def get(self, order_id: int) -> Order:
        """Fetch an order by id, using the cache when possible."""
        cached = self._cache.get(str(order_id))
        if cached is not None:
            return cached
        row = self._rows.get(order_id)
        if row is None:
            raise RepositoryError(f"order {order_id} not found")
        order = ser.order_from_dict(row)
        self._cache.set(str(order_id), order)
        return order

    def list_ids(self) -> list[int]:
        return sorted(self._rows)

    def delete(self, order_id: int) -> bool:
        """Remove an order; return True if it existed."""
        self._cache.clear()
        return self._rows.pop(order_id, None) is not None
