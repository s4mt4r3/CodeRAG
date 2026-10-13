"""Outbound customer notifications (email is simulated)."""

from __future__ import annotations

import asyncio
import logging

from shopcore.models import Customer, Order
from shopcore.utils import format_money

log = logging.getLogger(__name__)


class Notifier:
    """Formats and 'sends' messages to customers."""

    def __init__(self, sender: str = "orders@example.com") -> None:
        self.sender = sender
        self.outbox: list[tuple[str, str]] = []

    def render_receipt(self, order: Order, total: float) -> str:
        """Build the plain-text receipt body for an order."""
        lines = [f"Order #{order.order_id} for {order.customer.display_name()}"]
        for item in order.items:
            lines.append(f"  {item.quantity} x {item.sku}: {format_money(item.subtotal())}")
        lines.append(f"Total: {format_money(total)}")
        return "\n".join(lines)

    def send(self, customer: Customer, body: str) -> None:
        """Queue a message for the customer."""
        self.outbox.append((customer.email, body))
        log.info("queued message for %s", customer.email)

    async def send_async(self, customer: Customer, body: str, delay: float = 0.0) -> None:
        """Send after an optional delay without blocking the event loop."""
        await asyncio.sleep(delay)
        self.send(customer, body)


async def broadcast(notifier: Notifier, customers: list[Customer], body: str) -> int:
    """Send the same message to many customers concurrently."""
    await asyncio.gather(*(notifier.send_async(c, body) for c in customers))
    return len(customers)
