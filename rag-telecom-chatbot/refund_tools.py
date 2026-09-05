"""Refund tools and the deterministic validation rules behind them.

Section 6.4 defines four tools. Only the first two are ever handed to the LLM:

    get_user_account   FR-19   exposed to the agent
    validate_refund    FR-20   exposed to the agent
    process_refund     FR-21   application-only (FR-23a)
    submit_to_support  FR-21a  application-only (FR-23a)

Eligibility is decided here, in plain Python, never by the model (FR-29, FR-30,
GR-05). The agent may choose *which* tool to call and with what amount; it can
never choose the answer.
"""

from __future__ import annotations

import uuid
from typing import Any

from config import REFUND_WINDOW_DAYS
from accounts import account_snapshot, mark_refunded

# --- reasons (single source of wording) ------------------------------------
REASON_ELIGIBLE = "Refund is eligible."
REASON_NO_ACCOUNT = "No account found for this user."
REASON_NO_RECHARGE = "No recharge found on this account."
REASON_NON_POSITIVE = "Refund amount must be greater than zero."
REASON_EXCEEDS = "Requested amount exceeds the recharge amount."
REASON_NOT_COMPLETED = "The recharge is not in a completed state."
REASON_ALREADY_REFUNDED = "This recharge has already been refunded."
REASON_TOO_OLD = f"The recharge is outside the {REFUND_WINDOW_DAYS}-day refund window."


# --- helpers ---------------------------------------------------------------
def _ordered(recharges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Stable ordering: most recent first, then by id. Keeps selection reproducible."""
    return sorted(recharges, key=lambda r: (r["recharge_age_days"], r["recharge_id"]))


def _select_recharge(recharges: list[dict[str, Any]], amount: int) -> dict[str, Any] | None:
    """Pick the recharge a refund of `amount` applies to (FR-27a).

    An exact amount match identifies that recharge even when it is already
    refunded - otherwise asking to refund an already-refunded amount would
    silently slide onto a different recharge instead of being rejected.
    """
    ordered = _ordered(recharges)

    exact = [r for r in ordered if r["recharge_amount"] == amount]
    if exact:
        return exact[0]

    # No exact match: the smallest recharge that still covers the amount,
    # preferring one that has not been refunded yet.
    covering = [r for r in ordered if r["recharge_amount"] >= amount]
    if not covering:
        return None
    covering.sort(
        key=lambda r: (
            r["refund_status"] != "not_refunded",
            r["recharge_amount"],
            r["recharge_age_days"],
            r["recharge_id"],
        )
    )
    return covering[0]


def _result(
    eligible: bool,
    reason: str,
    requested_amount: int | None = None,
    recharge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "eligible": eligible,
        "reason": reason,
        "requested_amount": requested_amount,
        "recharge_amount": recharge["recharge_amount"] if recharge else None,
        "recharge_id": recharge["recharge_id"] if recharge else None,
    }


def is_refundable(recharge: dict[str, Any]) -> bool:
    """Would this recharge pass validation for its own full amount?"""
    return (
        recharge["recharge_status"] == "completed"
        and recharge["refund_status"] == "not_refunded"
        and recharge["recharge_age_days"] <= REFUND_WINDOW_DAYS
    )


def refundable_recharges(user_id: str) -> list[dict[str, Any]]:
    """Every recharge currently eligible for refund, most recent first (FR-41, FR-42)."""
    account = account_snapshot(user_id)
    if account is None:
        return []
    return [r for r in _ordered(account["recharges"]) if is_refundable(r)]


# --- tools exposed to the agent -------------------------------------------
def get_user_account(user_id: str) -> dict[str, Any]:
    """Retrieve the account and its recharges (FR-19).

    The user id comes from session context, never from the customer (FR-16,
    FR-17), and this snapshot is the only trusted source of account facts -
    anything the customer asserts about their recharges is ignored (GR-03).
    """
    account = account_snapshot(user_id)
    if account is None:
        return {"found": False, "reason": REASON_NO_ACCOUNT, "user_id": user_id}
    return {"found": True, "user_id": user_id, **account}


def validate_refund(user_id: str, amount: int) -> dict[str, Any]:
    """Decide refund eligibility deterministically (FR-20, FR-24 .. FR-29).

    Eligibility only. Whether an eligible refund may be executed *here* is a
    separate question, settled by the approval limit in `refund_agent`.
    """
    account = account_snapshot(user_id)
    if account is None:
        return _result(False, REASON_NO_ACCOUNT, amount)

    try:
        amount = int(amount)
    except (TypeError, ValueError):
        return _result(False, REASON_NON_POSITIVE, None)

    if amount <= 0:                                             # FR-26
        return _result(False, REASON_NON_POSITIVE, amount)

    recharges = account["recharges"]
    if not recharges:
        return _result(False, REASON_NO_RECHARGE, amount)

    recharge = _select_recharge(recharges, amount)               # FR-27a
    if recharge is None:                                        # FR-27
        return _result(False, REASON_EXCEEDS, amount)

    if recharge["recharge_status"] != "completed":              # FR-24
        return _result(False, REASON_NOT_COMPLETED, amount, recharge)

    if recharge["refund_status"] != "not_refunded":             # FR-28
        return _result(False, REASON_ALREADY_REFUNDED, amount, recharge)

    if recharge["recharge_age_days"] > REFUND_WINDOW_DAYS:      # FR-25
        return _result(False, REASON_TOO_OLD, amount, recharge)

    return _result(True, REASON_ELIGIBLE, amount, recharge)


# --- application-only tools (never bound to the LLM, FR-23a) ---------------
def process_refund(user_id: str, amount: int, recharge_id: str) -> dict[str, Any]:
    """Execute the simulated refund (FR-21, FR-23, FR-33 .. FR-35).

    Callers must have validated (FR-31) and obtained explicit confirmation
    (FR-32) first; `refund_agent` is the only caller and enforces both. No
    payment or billing API exists here - the refund is a local state flip.
    """
    check = validate_refund(user_id, amount)
    if not check["eligible"] or check["recharge_id"] != recharge_id:
        # Re-validated at the moment of execution so a stale decision cannot
        # slip through (GR-01).
        return {
            "success": False,
            "reason": check["reason"] if not check["eligible"] else REASON_NO_RECHARGE,
            "amount": amount,
            "status": "rejected",
        }

    if not mark_refunded(user_id, recharge_id):
        return {
            "success": False,
            "reason": REASON_ALREADY_REFUNDED,
            "amount": amount,
            "status": "rejected",
        }

    return {
        "success": True,
        "refund_id": f"ref_{uuid.uuid4().hex[:6]}",
        "amount": amount,
        "recharge_id": recharge_id,
        "status": "refunded",
    }


def submit_to_support(user_id: str, amount: int, recharge_id: str | None = None) -> dict[str, Any]:
    """Hand a refund above the approval limit to the Support Team (FR-21a, FR-38).

    Records the submission and nothing more. The review and whatever follows it
    happen outside this application, so no outcome is invented here (GR-09).
    """
    return {
        "submitted": True,
        "ticket_id": f"sup_{uuid.uuid4().hex[:6]}",
        "user_id": user_id,
        "amount": amount,
        "recharge_id": recharge_id,
        "status": "submitted_for_review",
    }
