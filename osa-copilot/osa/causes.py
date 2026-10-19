"""The fixed cause list (section 5, "Cause list and playbook").

The model may only choose causes from this list. Anything it cannot match
goes to "Needs investigation" (section 16: "anything unmatched goes to Needs
investigation"). The one-line definitions are what the model sees, so they
describe the evidence pattern, not just the name.
"""

from __future__ import annotations

DC_SHORTAGE = "DC_SHORTAGE"
ORDER_CUT = "ORDER_CUT"
STORE_ORDERING_MISS = "STORE_ORDERING_MISS"
PHANTOM_INVENTORY = "PHANTOM_INVENTORY"
SHELF_EXECUTION_GAP = "SHELF_EXECUTION_GAP"
DELISTING = "DELISTING"
PROMO_DEMAND_SPIKE = "PROMO_DEMAND_SPIKE"

CAUSES: dict[str, dict[str, str]] = {
    DC_SHORTAGE: {
        "label": "DC shortage",
        "definition": "The DC serving the store ran low on the SKU even though the manufacturer "
                      "delivered what the DC ordered, so store orders were short-shipped.",
    },
    ORDER_CUT: {
        "label": "Order cuts",
        "definition": "The manufacturer cut the DC's orders (low manufacturer fill rate), so the DC "
                      "short-shipped stores. Takes precedence over DC shortage when manufacturer "
                      "fill rate is low.",
    },
    STORE_ORDERING_MISS: {
        "label": "Store ordering miss",
        "definition": "The store stopped ordering or under-ordered while the DC had stock and the SKU "
                      "is still ranged, so store on-hand ran down to empty.",
    },
    PHANTOM_INVENTORY: {
        "label": "Phantom inventory",
        "definition": "The system shows healthy store on-hand that is not physically there: on-hand "
                      "stays flat, sales stop and no replenishment orders are triggered.",
    },
    SHELF_EXECUTION_GAP: {
        "label": "Shelf execution gap",
        "definition": "Stock is in the store (on-hand healthy, deliveries continuing) but not on the "
                      "shelf: audit shows the SKU missing or below planogram facings.",
    },
    DELISTING: {
        "label": "Delisting / ranging change",
        "definition": "The SKU was removed from the store's planogram or range, so orders stop and "
                      "the shelf is not refilled.",
    },
    PROMO_DEMAND_SPIKE: {
        "label": "Promo demand spike",
        "definition": "A promotion lifted sales well above baseline while orders and shipments were "
                      "not increased, so the store sold out.",
    },
}

UPSTREAM_CAUSES = {DC_SHORTAGE, ORDER_CUT}   # section 9.2 "Upstream cause"

CAUSE_KEYS = list(CAUSES)


def label(cause: str) -> str:
    return CAUSES.get(cause, {}).get("label", cause)
