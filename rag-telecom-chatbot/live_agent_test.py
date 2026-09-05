"""Live end-to-end check of the agent routes (section 8's scenarios).

Calls the real model, so it needs GROQ_API_KEY and network. The deterministic
guardrails are covered offline in `test_refund.py`; what this exercises is the
part only a live run can show - intent classification and the agent's tool loop.

    python live_agent_test.py
"""

from __future__ import annotations

import sys

from accounts import DEMO_USER_ID, account_snapshot, reset_accounts
from conversation import Turn
from refund_agent import RefundState
from router import REFUND_REQUEST, Assistant

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []

# The rupee sign is outside the Windows console's default codepage.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((PASS if ok else FAIL, name, detail))
    print(f"  {PASS if ok else FAIL}  {name}" + (f"  -- {detail}" if detail else ""))


def refund_status(recharge_id: str) -> str:
    account = account_snapshot(DEMO_USER_ID)
    return next(
        r["refund_status"] for r in account["recharges"] if r["recharge_id"] == recharge_id
    )


def show(reply) -> None:
    print(f"    agent: {reply.text.splitlines()[0][:96]}")
    for step in reply.trace:
        print(f"      {step}")


def main() -> int:
    assistant = Assistant(DEMO_USER_ID)

    # --- Scenario 5: an ordinary support question stays on the RAG route ----
    print("\n[Scenario 5] Existing RAG query")
    intent = assistant.route("Why is my mobile internet slow?", RefundState(), [])
    record("slow internet -> requirement_inquiry", intent == "requirement_inquiry", intent)

    # --- FR-05a: asking *how* refunds work is not a refund request ---------
    print("\n[FR-05a] Informational vs actionable")
    intent = assistant.route("How do I cancel a recharge?", RefundState(), [])
    record("how to cancel -> requirement_inquiry", intent == "requirement_inquiry", intent)
    intent = assistant.route("I want a refund of 499.", RefundState(), [])
    record("refund ask -> refund_request", intent == REFUND_REQUEST, intent)

    # --- Scenario 1: eligible refund, confirm, execute ----------------------
    print("\n[Scenario 1] Successful refund of 499")
    reset_accounts()
    state = RefundState()
    reply = assistant.handle_refund("I want a refund of 499.", state, [])
    show(reply)
    record("asks for confirmation", reply.awaiting_confirmation)
    record("not yet refunded", refund_status("rch_457") == "not_refunded")

    done = assistant.agent.confirm(reply.state)
    show(done)
    record("refund executed", refund_status("rch_457") == "refunded")
    record("returns a reference", "ref_" in done.text)

    # --- Scenario 6: above the approval limit -------------------------------
    print("\n[Scenario 6] Refund of 999 goes to the Support Team")
    reset_accounts()
    reply = assistant.handle_refund("I want a refund of 999.", RefundState(), [])
    show(reply)
    record("submitted to support", "Support Team" in reply.text)
    record("no confirmation step", not reply.awaiting_confirmation)
    record("recharge untouched", refund_status("rch_456") == "not_refunded")
    record(
        "claims no outcome",
        not any(w in reply.text.lower() for w in ("approved", "rejected", "has been processed")),
    )

    # --- Scenario 2: amount exceeds the recharge ----------------------------
    print("\n[Scenario 2] Refund of 2000 exceeds the recharge")
    reset_accounts()
    reply = assistant.handle_refund("Give me a refund of 2,000.", RefundState(), [])
    show(reply)
    record("rejected", not reply.awaiting_confirmation)
    record("explains why", "exceeds" in reply.text.lower())

    # --- Section 6.8: no amount given ---------------------------------------
    print("\n[6.8] Refund with no amount")
    reset_accounts()
    reply = assistant.handle_refund("I want a refund.", RefundState(), [])
    show(reply)
    record("asks which amount", "499" in reply.text and "999" in reply.text)

    # --- Scenario 7: follow-up resolved from history ------------------------
    print("\n[Scenario 7] Follow-up question uses conversation context")
    first = "How do I activate international roaming?"
    answer, _ = assistant.answer(first, [])
    print(f"    first answer: {answer.splitlines()[0][:96]}")
    history = [Turn("user", first), Turn("assistant", answer)]
    follow_up = "Should I do that before I travel?"
    intent = assistant.route(follow_up, RefundState(), history)
    record("follow-up -> requirement_inquiry", intent == "requirement_inquiry", intent)
    second, _ = assistant.answer(follow_up, history)
    print(f"    follow-up answer: {second.splitlines()[0][:120]}")
    record(
        "resolves 'that' as roaming",
        any(w in second.lower() for w in ("roaming", "before you travel", "24 hours")),
    )

    reset_accounts()

    failures = [r for r in results if r[0] == FAIL]
    print(f"\n{'=' * 70}\n{len(results) - len(failures)}/{len(results)} checks passed")
    for _, name, detail in failures:
        print(f"  FAILED: {name} {detail}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
