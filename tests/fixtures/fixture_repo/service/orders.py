"""Order workflow: create, price, pay, and notify."""

from __future__ import annotations

from shopcore import pricing
from shopcore.models import Customer, Order
from storage import OrderRepository

from .notifications import Notifier


class OrderService:
    """High-level operations on orders."""

    def __init__(self, repo: OrderRepository, notifier: Notifier) -> None:
        self.repo = repo
        self.notifier = notifier
        self._next_id = 1

    def create_order(self, customer: Customer) -> Order:
        """Open a new draft order for a customer."""
        order = Order(order_id=self._allocate_id(), customer=customer)
        self.repo.save(order)
        return order

    def _allocate_id(self) -> int:
        order_id = self._next_id
        self._next_id += 1
        return order_id

    def quote(self, order_id: int, region: str = "NJ") -> float:
        """Return the amount the customer would pay right now."""
        order = self.repo.get(order_id)
        return pricing.compute_total(order, region)

    def checkout(self, order_id: int, region: str = "NJ") -> float:
        """Charge the order, award loyalty points, and send a receipt."""
        order = self.repo.get(order_id)
        if order.is_empty:
            raise ValueError("cannot check out an empty order")
        total = self.quote(order_id, region)
        order.mark_paid()
        order.customer.add_points(int(order.gross_total()))
        self.repo.save(order)
        receipt = self.notifier.render_receipt(order, total)
        self.notifier.send(order.customer, receipt)
        return total
