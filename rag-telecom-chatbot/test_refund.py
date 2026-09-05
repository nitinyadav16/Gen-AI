"""Offline tests for the refund capability - the seven demonstration scenarios
of section 8 plus the guardrails of section 9.

No network and no API key needed: the agent's LLM tool loop is stubbed out, so
what is under test is exactly the deterministic layer that the guardrails rest
on. If the model misbehaves, these rules are what still has to hold.

    python test_refund.py
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import accounts
import refund_tools
from accounts import DEMO_USER_ID, account_snapshot, reset_accounts
from config import REFUND_APPROVAL_LIMIT, REFUND_WINDOW_DAYS
from conversation import Turn
from refund_agent import (
    AWAITING_AMOUNT,
    AWAITING_CONFIRMATION,
    REFUND_CANCELLED,
    REFUND_PROCESSED,
    REFUND_REJECTED,
    REFUND_SUBMITTED_FOR_REVIEW,
    RefundAgent,
    RefundState,
    amounts_in,
    read_confirmation,
)

NO_TOOLS = {"refund_agent.RefundAgent._run_tool_loop": lambda *a, **k: []}


def recharge(user_id: str, recharge_id: str) -> dict:
    account = account_snapshot(user_id)
    return next(r for r in account["recharges"] if r["recharge_id"] == recharge_id)


class RefundTestCase(unittest.TestCase):
    """Shared setup: fresh seed data, and an agent whose LLM loop is stubbed."""

    def setUp(self) -> None:
        reset_accounts()
        self.agent = RefundAgent(DEMO_USER_ID)
        self.state = RefundState()
        patcher = patch.object(RefundAgent, "_run_tool_loop", return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)

    def ask(self, message: str):
        reply = self.agent.handle(message, self.state, [])
        self.state = reply.state
        return reply


class TestValidation(RefundTestCase):
    """Section 6.5 - deterministic eligibility."""

    def test_eligible_recharge(self):
        result = refund_tools.validate_refund(DEMO_USER_ID, 499)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["recharge_id"], "rch_457")
        self.assertEqual(result["recharge_amount"], 499)

    def test_scenario_2_amount_exceeds_recharge(self):
        result = refund_tools.validate_refund(DEMO_USER_ID, 2000)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], refund_tools.REASON_EXCEEDS)

    def test_scenario_3_recharge_too_old(self):
        accounts._USERS[DEMO_USER_ID]["recharges"][1]["recharge_age_days"] = 15
        result = refund_tools.validate_refund(DEMO_USER_ID, 499)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], refund_tools.REASON_TOO_OLD)

    def test_scenario_4_already_refunded(self):
        accounts._USERS[DEMO_USER_ID]["recharges"][1]["refund_status"] = "refunded"
        result = refund_tools.validate_refund(DEMO_USER_ID, 499)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], refund_tools.REASON_ALREADY_REFUNDED)

    def test_zero_and_negative_rejected(self):
        for amount in (0, -100):
            with self.subTest(amount=amount):
                result = refund_tools.validate_refund(DEMO_USER_ID, amount)
                self.assertFalse(result["eligible"])
                self.assertEqual(result["reason"], refund_tools.REASON_NON_POSITIVE)

    def test_not_completed_rejected(self):
        accounts._USERS[DEMO_USER_ID]["recharges"][1]["recharge_status"] = "pending"
        result = refund_tools.validate_refund(DEMO_USER_ID, 499)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], refund_tools.REASON_NOT_COMPLETED)

    def test_window_boundary_is_inclusive(self):
        accounts._USERS[DEMO_USER_ID]["recharges"][1]["recharge_age_days"] = REFUND_WINDOW_DAYS
        self.assertTrue(refund_tools.validate_refund(DEMO_USER_ID, 499)["eligible"])
        accounts._USERS[DEMO_USER_ID]["recharges"][1]["recharge_age_days"] = REFUND_WINDOW_DAYS + 1
        self.assertFalse(refund_tools.validate_refund(DEMO_USER_ID, 499)["eligible"])

    def test_exact_match_wins_even_when_refunded(self):
        """FR-27a: 499 must resolve to the 499 recharge, not slide onto the 999."""
        accounts._USERS[DEMO_USER_ID]["recharges"][1]["refund_status"] = "refunded"
        result = refund_tools.validate_refund(DEMO_USER_ID, 499)
        self.assertEqual(result["recharge_id"], "rch_457")
        self.assertFalse(result["eligible"])


class TestApprovalLimit(RefundTestCase):
    """Section 6.7 - the limit is enforced in code, never by the model."""

    def test_scenario_1_successful_refund(self):
        reply = self.ask("I want a refund of 499.")
        self.assertTrue(reply.awaiting_confirmation)
        self.assertEqual(self.state.status, AWAITING_CONFIRMATION)
        self.assertEqual(recharge(DEMO_USER_ID, "rch_457")["refund_status"], "not_refunded")

        done = self.agent.confirm(self.state)
        self.assertIn("refunded", done.text.lower())
        self.assertIn("ref_", done.text)
        self.assertEqual(recharge(DEMO_USER_ID, "rch_457")["refund_status"], "refunded")
        self.assertIn("[Refund] success", done.trace)

    def test_scenario_6_above_limit_goes_to_support(self):
        with patch.object(refund_tools, "process_refund") as never:
            reply = self.ask("I want a refund of 999.")
            never.assert_not_called()

        self.assertFalse(reply.awaiting_confirmation)
        self.assertIn("Support Team", reply.text)
        self.assertEqual(recharge(DEMO_USER_ID, "rch_456")["refund_status"], "not_refunded")

    def test_support_submission_claims_no_outcome(self):
        """FR-40a / GR-09: submitted, never approved, rejected or processed."""
        text = self.ask("I want a refund of 999.").text.lower()
        self.assertIn("submitted", text)
        for claim in ("approved", "rejected", "has been processed", "refunded successfully"):
            self.assertNotIn(claim, text)

    def test_limit_boundary(self):
        self.assertEqual(REFUND_APPROVAL_LIMIT, 499)
        self.assertTrue(self.ask("refund 499").awaiting_confirmation)
        self.state = RefundState()
        self.assertFalse(self.ask("refund 500").awaiting_confirmation)


class TestConfirmationGate(RefundTestCase):
    """Section 6.6 / GR-02 - nothing executes without an explicit yes."""

    def test_request_alone_is_not_confirmation(self):
        """FR-36b."""
        with patch.object(refund_tools, "process_refund") as never:
            self.ask("I want a refund of 499.")
            never.assert_not_called()
        self.assertEqual(recharge(DEMO_USER_ID, "rch_457")["refund_status"], "not_refunded")

    def test_rejection_cancels(self):
        self.ask("I want a refund of 499.")
        with patch.object(refund_tools, "process_refund") as never:
            reply = self.ask("no")
            never.assert_not_called()
        self.assertIn("not processed", reply.text.lower())
        self.assertEqual(recharge(DEMO_USER_ID, "rch_457")["refund_status"], "not_refunded")

    def test_ambiguous_reply_asks_again(self):
        """FR-36e - and must not execute in the meantime."""
        self.ask("I want a refund of 499.")
        with patch("refund_agent.read_confirmation", return_value="ambiguous"):
            with patch.object(refund_tools, "process_refund") as never:
                reply = self.ask("what does that mean?")
                never.assert_not_called()
        self.assertTrue(reply.awaiting_confirmation)
        self.assertEqual(self.state.status, AWAITING_CONFIRMATION)
        self.assertEqual(recharge(DEMO_USER_ID, "rch_457")["refund_status"], "not_refunded")

    def test_cannot_execute_without_pending_confirmation(self):
        """GR-01 - confirm() from a cold state refunds nothing."""
        with patch.object(refund_tools, "process_refund") as never:
            reply = self.agent.confirm(RefundState())
            never.assert_not_called()
        self.assertIn("do not have a refund", reply.text.lower())

    def test_invalid_refund_never_executes(self):
        for message in ("refund 2000", "refund 0"):
            with self.subTest(message=message):
                self.state = RefundState()
                with patch.object(refund_tools, "process_refund") as never:
                    reply = self.ask(message)
                    never.assert_not_called()
                self.assertFalse(reply.awaiting_confirmation)


class TestMissingAmount(RefundTestCase):
    """Section 6.8 - never invent an amount."""

    def test_two_eligible_recharges_asks_which(self):
        reply = self.ask("I want a refund.")
        self.assertEqual(self.state.status, AWAITING_AMOUNT)
        self.assertIn("499", reply.text)
        self.assertIn("999", reply.text)
        self.assertFalse(reply.awaiting_confirmation)

    def test_then_choosing_an_amount_proceeds(self):
        self.ask("I want a refund.")
        reply = self.ask("499")
        self.assertTrue(reply.awaiting_confirmation)
        self.assertEqual(reply.amount, 499)

    def test_single_eligible_recharge_is_offered(self):
        accounts._USERS[DEMO_USER_ID]["recharges"][0]["refund_status"] = "refunded"
        reply = self.ask("I want a refund.")
        self.assertTrue(reply.awaiting_confirmation)
        self.assertEqual(reply.amount, 499)

    def test_no_eligible_recharge_asks_for_amount(self):
        for r in accounts._USERS[DEMO_USER_ID]["recharges"]:
            r["refund_status"] = "refunded"
        reply = self.ask("I want a refund.")
        self.assertEqual(self.state.status, AWAITING_AMOUNT)
        self.assertFalse(reply.awaiting_confirmation)


class TestIndependence(RefundTestCase):
    def test_refunding_one_leaves_the_other_refundable(self):
        self.ask("I want a refund of 499.")
        self.agent.confirm(self.state)
        self.assertEqual(recharge(DEMO_USER_ID, "rch_457")["refund_status"], "refunded")
        self.assertEqual(recharge(DEMO_USER_ID, "rch_456")["refund_status"], "not_refunded")
        self.assertTrue(refund_tools.validate_refund(DEMO_USER_ID, 999)["eligible"])

    def test_double_refund_blocked(self):
        """GR-06."""
        self.ask("I want a refund of 499.")
        self.agent.confirm(self.state)
        self.state = RefundState()
        reply = self.ask("I want a refund of 499.")
        self.assertFalse(reply.awaiting_confirmation)
        self.assertIn("already been refunded", reply.text)


class TestGuardrails(RefundTestCase):
    def test_customer_claims_do_not_override_account_data(self):
        """GR-03 / GR-04."""
        reply = self.ask("I recharged 5000 yesterday, refund my 5000.")
        self.assertFalse(reply.awaiting_confirmation)
        self.assertIn("exceeds", reply.text.lower())

    def test_execution_tools_are_not_bound_to_the_model(self):
        """FR-23a - only the two read-only tools reach the LLM."""
        names = {t.name for t in self.agent._tools([], [])}
        self.assertEqual(names, {"get_user_account", "validate_refund"})

    def test_process_refund_revalidates(self):
        """GR-01 - a stale decision cannot slip past execution."""
        accounts._USERS[DEMO_USER_ID]["recharges"][1]["refund_status"] = "refunded"
        result = refund_tools.process_refund(DEMO_USER_ID, 499, "rch_457")
        self.assertFalse(result["success"])

    def test_user_id_never_requested(self):
        """FR-17 - the id comes from session context, so it is never asked for."""
        reply = self.ask("I want a refund.")
        for phrase in ("user id", "user_id", "account number", "your id"):
            self.assertNotIn(phrase, reply.text.lower())


class TestStateMachine(RefundTestCase):
    """Section 10 - every listed state must be reachable and observable."""

    def test_terminal_states_are_visible_after_the_turn(self):
        cases = {
            "rejected": (lambda: self.ask("refund 2000"), REFUND_REJECTED),
            "submitted": (lambda: self.ask("refund 999"), REFUND_SUBMITTED_FOR_REVIEW),
        }
        for name, (action, expected) in cases.items():
            with self.subTest(name):
                reset_accounts()
                self.state = RefundState()
                reply = action()
                self.assertEqual(reply.state.status, expected)

    def test_processed_and_cancelled_states(self):
        self.ask("refund 499")
        self.assertEqual(self.state.status, AWAITING_CONFIRMATION)
        done = self.agent.confirm(self.state)
        self.assertEqual(done.state.status, REFUND_PROCESSED)

        reset_accounts()
        self.state = RefundState()
        self.ask("refund 499")
        stopped = self.agent.cancel(self.state)
        self.assertEqual(stopped.state.status, REFUND_CANCELLED)

    def test_terminal_state_is_not_pending(self):
        """A settled refund must not capture the next message."""
        reply = self.ask("refund 2000")
        self.assertFalse(reply.state.pending)

    def test_awaiting_amount_state(self):
        reply = self.ask("I want a refund.")
        self.assertEqual(reply.state.status, AWAITING_AMOUNT)
        self.assertTrue(reply.state.pending)


class TestUncitedRequirements(RefundTestCase):
    """Requirements verified by behaviour rather than by an annotation."""

    def test_fr22_tools_return_structured_results(self):
        account = refund_tools.get_user_account(DEMO_USER_ID)
        self.assertIsInstance(account, dict)
        self.assertIn("recharges", account)

        check = refund_tools.validate_refund(DEMO_USER_ID, 499)
        self.assertEqual(
            set(check),
            {"eligible", "reason", "requested_amount", "recharge_amount", "recharge_id"},
        )

    def test_fr34_execution_returns_a_reference(self):
        receipt = refund_tools.process_refund(DEMO_USER_ID, 499, "rch_457")
        self.assertTrue(receipt["success"])
        self.assertTrue(receipt["refund_id"].startswith("ref_"))
        self.assertEqual(receipt["status"], "refunded")

    def test_fr39_no_refund_after_submission(self):
        self.ask("I want a refund of 999.")
        with patch.object(refund_tools, "process_refund") as never:
            self.agent.confirm(self.state)
            never.assert_not_called()
        self.assertEqual(recharge(DEMO_USER_ID, "rch_456")["refund_status"], "not_refunded")

    def test_gr10_history_amount_is_not_reused(self):
        """An amount mentioned only in an earlier turn must not be validated."""
        history = [
            Turn("user", "I want a refund of 499"),
            Turn("assistant", "Which amount would you like refunded?"),
        ]
        reply = self.agent.handle("go on then", RefundState(), history)
        self.assertFalse(reply.awaiting_confirmation)
        self.assertEqual(recharge(DEMO_USER_ID, "rch_457")["refund_status"], "not_refunded")

    def test_gr07_refund_is_simulated_only(self):
        """The refund is a local state flip - nothing leaves the process."""
        before = recharge(DEMO_USER_ID, "rch_457")["refund_status"]
        refund_tools.process_refund(DEMO_USER_ID, 499, "rch_457")
        after = recharge(DEMO_USER_ID, "rch_457")["refund_status"]
        self.assertEqual((before, after), ("not_refunded", "refunded"))
        reset_accounts()
        self.assertEqual(recharge(DEMO_USER_ID, "rch_457")["refund_status"], "not_refunded")


class TestHelpers(unittest.TestCase):
    def test_amount_extraction(self):
        cases = {
            "I want a refund of 499": [499],
            "refund ₹999 please": [999],
            "Rs. 1,499": [1499],
            "INR 250": [250],
            "I want a refund": [],
            "refund 499 not 999": [499, 999],
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(amounts_in(text), expected)

    def test_clear_confirmations_need_no_model(self):
        for yes in ("yes", "Yes, proceed", "go ahead", "confirm", "ok"):
            self.assertEqual(read_confirmation(yes), "confirm", yes)
        for no in ("no", "cancel", "don't", "never mind", "stop"):
            self.assertEqual(read_confirmation(no), "reject", no)


if __name__ == "__main__":
    unittest.main(verbosity=2)
