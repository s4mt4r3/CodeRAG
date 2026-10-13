"""Domain models: customers, line items, and orders."""

from __future__ import annotations

from dataclasses import dataclass, field

from .utils import normalize_sku, slugify


@dataclass
class Customer:
    """A registered customer who can place orders."""

    customer_id: int
    name: str
    email: str
    loyalty_points: int = 0

    def display_name(self) -> str:
        """Return a human-friendly name for receipts."""
        return f"{self.name} <{self.email}>"

    def handle(self) -> str:
        return slugify(self.name)

    def add_points(self, points: int) -> int:
        """Credit loyalty points and return the new balance."""
        if points < 0:
            raise ValueError("points must be non-negative")
        self.loyalty_points += points
        return self.loyalty_points


@dataclass
class LineItem:
    """One product line within an order."""

    sku: str
    quantity: int
    unit_price: float

    def __post_init__(self) -> None:
        self.sku = normalize_sku(self.sku)

    def subtotal(self) -> float:
        """Price of this line before discounts."""
        return round(self.quantity * self.unit_price, 2)


@dataclass
class Order:
    """A customer order made of line items."""

    order_id: int
    customer: Customer
    items: list[LineItem] = field(default_factory=list)
    status: str = "draft"

    def add_item(self, sku: str, quantity: int, unit_price: float) -> LineItem:
        """Append a new line item and return it."""
        item = LineItem(sku, quantity, unit_price)
        self.items.append(item)
        return item

    def item_count(self) -> int:
        """Total number of units across all lines."""
        return sum(item.quantity for item in self.items)

    def gross_total(self) -> float:
        """Sum of line subtotals before discounts."""
        return round(sum(item.subtotal() for item in self.items), 2)

    @property
    def is_empty(self) -> bool:
        return not self.items

    def mark_paid(self) -> None:
        """Transition the order to the paid state."""
        self.status = "paid"
