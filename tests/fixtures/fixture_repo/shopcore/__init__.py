"""Core domain types and pricing rules for the shop."""

from .models import Customer, LineItem, Order
from .pricing import apply_discount, compute_total

__all__ = ["Customer", "LineItem", "Order", "apply_discount", "compute_total"]
