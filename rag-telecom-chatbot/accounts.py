"""Simulated user / account data for the refund capability (FR-14 .. FR-18a).

Hard-coded for this iteration: no CRM, billing system, customer database, or
authentication is involved (Non-Goals, section 4).

`DEMO_USER_ID` stands in for an id that would arrive from authenticated session
context (FR-16). The customer is never asked for it (FR-17).

Reads hand out deep copies so nothing outside this module can quietly mutate the
store - refund status changes only through `mark_refunded()` (GR-03).
"""

from __future__ import annotations

import copy
from typing import Any

DEMO_USER_ID = "user_123"

# The demonstration user has two recharges (FR-18a):
#   rch_457  Rs 499  - at the approval limit, refundable by the application
#   rch_456  Rs 999  - above the limit, goes to the Support Team (section 6.7)
_SEED: dict[str, dict[str, Any]] = {
    "user_123": {
        "name": "Rahul",
        "plan": "Pro",
        "plan_amount": 999,
        "recharges": [
            {
                "recharge_id": "rch_456",
                "recharge_amount": 999,
                "recharge_status": "completed",
                "recharge_age_days": 2,
                "refund_status": "not_refunded",
            },
            {
                "recharge_id": "rch_457",
                "recharge_amount": 499,
                "recharge_status": "completed",
                "recharge_age_days": 1,
                "refund_status": "not_refunded",
            },
        ],
    }
}

_USERS: dict[str, dict[str, Any]] = copy.deepcopy(_SEED)


def account_snapshot(user_id: str) -> dict[str, Any] | None:
    """A read-only copy of the account, or None if the id is unknown."""
    account = _USERS.get(user_id)
    return copy.deepcopy(account) if account is not None else None


def find_recharge(user_id: str, recharge_id: str) -> dict[str, Any] | None:
    """A read-only copy of one recharge."""
    account = _USERS.get(user_id)
    if account is None:
        return None
    for recharge in account["recharges"]:
        if recharge["recharge_id"] == recharge_id:
            return copy.deepcopy(recharge)
    return None


def mark_refunded(user_id: str, recharge_id: str) -> bool:
    """Flip one recharge to `refunded` in the simulated store (FR-33).

    A refund consumes the whole recharge; no remaining balance is tracked
    (FR-28a). Returns False if the recharge is unknown or already refunded.
    """
    account = _USERS.get(user_id)
    if account is None:
        return False
    for recharge in account["recharges"]:
        if recharge["recharge_id"] != recharge_id:
            continue
        if recharge["refund_status"] != "not_refunded":
            return False
        recharge["refund_status"] = "refunded"
        return True
    return False


def reset_accounts() -> None:
    """Restore the seed data. Used by the tests, and handy between demo runs."""
    global _USERS
    _USERS = copy.deepcopy(_SEED)
