"""Pricing rules: discounts, tax, and totals."""

from functools import lru_cache

from .models import Order
from .utils import clamp as clamp_value

TAX_RATES = {"NJ": 0.06625, "NY": 0.04, "DE": 0.0}


@lru_cache(maxsize=32)
def tax_rate(region: str) -> float:
    """Look up the sales tax rate for a region code."""
    return TAX_RATES.get(region.upper(), 0.05)


def apply_discount(amount: float, percent: float) -> float:
    """Apply a percentage discount, clamped to 0..100 percent."""
    pct = clamp_value(percent, 0.0, 100.0)
    return round(amount * (1 - pct / 100.0), 2)


def loyalty_discount(points: int) -> float:
    """Convert loyalty points into a discount percentage."""

    def tier(p: int) -> int:
        if p >= 1000:
            return 3
        if p >= 250:
            return 2
        return 1 if p > 0 else 0

    return float(tier(points) * 5)


def compute_total(order: Order, region: str = "NJ") -> float:
    """Gross total minus loyalty discount, plus tax."""
    gross = order.gross_total()
    discounted = apply_discount(gross, loyalty_discount(order.customer.loyalty_points))
    return round(discounted * (1 + tax_rate(region)), 2)
