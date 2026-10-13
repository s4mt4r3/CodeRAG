"""Convert domain objects to and from plain dictionaries."""

import json

from shopcore.models import Customer, LineItem, Order


def customer_to_dict(customer: Customer) -> dict:
    """Serialize a customer to a JSON-safe dict."""
    return {
        "customer_id": customer.customer_id,
        "name": customer.name,
        "email": customer.email,
        "loyalty_points": customer.loyalty_points,
    }


def order_to_dict(order: Order) -> dict:
    """Serialize an order, including its customer and items."""
    return {
        "order_id": order.order_id,
        "status": order.status,
        "customer": customer_to_dict(order.customer),
        "items": [
            {"sku": i.sku, "quantity": i.quantity, "unit_price": i.unit_price}
            for i in order.items
        ],
    }


def order_from_dict(data: dict) -> Order:
    """Rebuild an order from ``order_to_dict`` output."""
    customer = Customer(**data["customer"])
    order = Order(order_id=data["order_id"], customer=customer, status=data["status"])
    for raw in data["items"]:
        order.items.append(LineItem(**raw))
    return order


def dumps(order: Order) -> str:
    return json.dumps(order_to_dict(order), sort_keys=True)


def validate(data: dict) -> bool:
    """Check that a serialized order has the required top-level keys."""
    return {"order_id", "status", "customer", "items"} <= set(data)
