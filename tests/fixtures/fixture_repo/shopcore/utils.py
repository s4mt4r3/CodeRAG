"""Small string and number helpers shared across packages."""

import re
import unicodedata

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Turn arbitrary text into a lowercase, dash-separated slug."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return _SLUG_RE.sub("-", ascii_text.lower()).strip("-")


def normalize_sku(sku: str) -> str:
    """Uppercase a SKU and strip surrounding whitespace."""
    return sku.strip().upper()


def clamp(value: float, low: float, high: float) -> float:
    """Restrict value to the closed interval [low, high]."""
    return max(low, min(high, value))


def format_money(amount: float, currency: str = "USD") -> str:
    symbol = {"USD": "$", "EUR": "€"}.get(currency, currency + " ")
    return f"{symbol}{amount:,.2f}"


def größe_bucket(quantity: int) -> str:
    """Bucket an order quantity into a coarse size label."""
    if quantity < 10:
        return "small"
    return "bulk"
