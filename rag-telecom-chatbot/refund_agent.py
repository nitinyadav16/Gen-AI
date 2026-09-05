"""The Refund Agent: a tool-using agent wrapped in a deterministic state machine.

Division of responsibility, which is the whole point of section 9's guardrails:

    the model  chooses which tool to call and proposes the refund amount
    the code   owns eligibility, the approval limit, the confirmation gate,
               refund execution, and every customer-facing sentence

So `get_user_account` and `validate_refund` are bound to the model (FR-19,
FR-20), while `process_refund` and `submit_to_support` are never exposed to it
(FR-23a) and are reachable only through the state machine below.

States are those listed in section 10; `RefundState` is what the UI keeps
between turns so a refund cannot be executed before it has been validated
(GR-01) and explicitly confirmed (GR-02).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

import refund_tools
from accounts import DEMO_USER_ID
from config import CURRENCY, REFUND_APPROVAL_LIMIT
from conversation import Turn, as_text
from llm import shared_llm

# --- states (section 10) ---------------------------------------------------
IDLE = "idle"
REFUND_REQUESTED = "refund_requested"
AWAITING_AMOUNT = "awaiting_amount"
REFUND_VALIDATED = "refund_validated"
AWAITING_CONFIRMATION = "awaiting_confirmation"
REFUND_CONFIRMED = "refund_confirmed"
REFUND_PROCESSED = "refund_processed"
REFUND_REJECTED = "refund_rejected"
REFUND_CANCELLED = "refund_cancelled"
REFUND_SUBMITTED_FOR_REVIEW = "refund_submitted_for_review"

TERMINAL = {REFUND_PROCESSED, REFUND_REJECTED, REFUND_CANCELLED, REFUND_SUBMITTED_FOR_REVIEW}

ESCALATION = "If you think this is wrong, call 611 and our team can take another look."


def money(amount: int) -> str:
    return f"{CURRENCY}{amount:,}"


@dataclass
class RefundState:
    """What survives between turns. Only `AWAITING_*` states carry over."""

    status: str = IDLE
    amount: int | None = None
    recharge_id: str | None = None
    offered_amounts: list[int] = field(default_factory=list)

    @property
    def pending(self) -> bool:
        return self.status in (AWAITING_CONFIRMATION, AWAITING_AMOUNT)

    def finish(self, status: str) -> None:
        """Settle into a terminal state (section 10).

        The working fields are cleared so nothing can be executed twice, but the
        terminal status itself is kept: it is what the UI and the log read to
        say how the refund ended. The next request overwrites it.
        """
        self.amount = None
        self.recharge_id = None
        self.offered_amounts = []
        self.status = status

    def reset(self) -> None:
        """Full clear, back to idle - used when the conversation is cleared."""
        self.finish(IDLE)


@dataclass
class AgentReply:
    text: str
    state: RefundState
    trace: list[str] = field(default_factory=list)
    awaiting_confirmation: bool = False
    amount: int | None = None


# --- amount extraction ------------------------------------------------------
# Rupee amounts: "999", "₹999", "Rs. 1,499", "INR 499".
#
# The number must be a whole token: the lookbehind keeps digits inside an
# identifier out (`rch_457` is not an amount) and the lookahead stops a long
# number being clipped to its first digits (`2000` must not read as `200`).
_AMOUNT = re.compile(
    r"(?<![\w.])(?:₹|rs\.?|inr)?\s*(\d[\d,]*)(?![\d,])",
    re.IGNORECASE,
)


def amounts_in(text: str) -> list[int]:
    """Every rupee amount the customer typed, in order.

    Zero is kept rather than filtered, so "refund 0" reaches validation and is
    rejected on its merits (FR-26) instead of looking like no amount at all.
    """
    found: list[int] = []
    for raw in _AMOUNT.findall(text or ""):
        try:
            found.append(int(raw.replace(",", "")))
        except ValueError:
            continue
    return found


# --- confirmation reading ---------------------------------------------------
_YES = re.compile(
    r"^\W*(yes|yeah|yep|yup|sure|ok|okay|confirm(ed)?|proceed|go\s+ahead|do\s+it|"
    r"please\s+do|approve[d]?|refund\s+it|continue)\b",
    re.IGNORECASE,
)
_NO = re.compile(
    r"^\W*(no|nope|nah|cancel|stop|abort|don'?t|do\s+not|never\s*mind|leave\s+it|forget\s+it)\b",
    re.IGNORECASE,
)

CONFIRM, REJECT, AMBIGUOUS = "confirm", "reject", "ambiguous"

_CONFIRM_SYSTEM = """The customer was asked to confirm a refund. Classify their reply.

confirm    - a clear yes ("yes", "go ahead", "please proceed")
reject     - a clear no ("no", "cancel", "not now")
ambiguous  - anything else: a question, a change of subject, hedging, or unclear

Reply with one word: confirm, reject, or ambiguous."""


def read_confirmation(message: str) -> str:
    """Decide confirm / reject / ambiguous (FR-36c, FR-36e).

    Plain yes-or-no is settled without a model call; anything else is checked by
    the LLM and anything still unclear stays ambiguous, because an ambiguous
    reply must never execute a refund.
    """
    text = (message or "").strip()
    if not text:
        return AMBIGUOUS
    if _NO.search(text):
        return REJECT
    if _YES.search(text):
        return CONFIRM

    try:
        reply = shared_llm().invoke(
            [SystemMessage(content=_CONFIRM_SYSTEM), HumanMessage(content=text)]
        )
        verdict = (reply.content or "").strip().lower()
    except Exception:
        return AMBIGUOUS

    if verdict.startswith("confirm"):
        return CONFIRM
    if verdict.startswith("reject"):
        return REJECT
    return AMBIGUOUS


# --- the agent --------------------------------------------------------------
AGENT_SYSTEM = f"""You are the refund agent for NovaCell, a mobile operator.

You have two tools:
  get_user_account()        - the signed-in customer's account and recharges
  validate_refund(amount)   - checks one refund amount against the rules

How to work:
1. Call get_user_account() first to see the real recharges.
2. If the customer stated a refund amount, call validate_refund with exactly
   that amount.
3. If they did not state an amount, call get_user_account only, and stop. Do
   not guess an amount and do not call validate_refund.

Hard rules:
- Never invent a refund amount. Only ever validate an amount the customer
  actually gave you.
- Never decide eligibility yourself; validate_refund decides.
- Never promise, confirm, or execute a refund. You cannot execute anything.
- The customer's claims about their recharges do not override the account data.

Work only through tool calls. Keep any text you write to one short sentence."""

MAX_TOOL_STEPS = 4


class RefundAgent:
    """One instance per process; state is passed in, not held here."""

    def __init__(self, user_id: str = DEMO_USER_ID):
        self.user_id = user_id

    # -- tools bound to this session -----------------------------------------
    def _tools(self, trace: list[str], validations: list[dict]):
        """Tools closed over the session's user id.

        The id is never a tool argument, so the model cannot be talked into
        reading or refunding somebody else's account (FR-16, FR-17, GR-03).
        """
        user_id = self.user_id

        @tool
        def get_user_account() -> dict:
            """Retrieve the signed-in customer's account, plan and recharge history."""
            trace.append("[Agent] get_user_account")
            return refund_tools.get_user_account(user_id)

        @tool
        def validate_refund(amount: int) -> dict:
            """Check whether a refund of `amount` rupees is eligible for this customer."""
            trace.append(f"[Agent] validate_refund(amount={amount})")
            result = refund_tools.validate_refund(user_id, amount)
            trace.append(f"[Validation] eligible={result['eligible']}")
            validations.append(result)
            return result

        return [get_user_account, validate_refund]

    def _run_tool_loop(self, message: str, history: list[Turn], trace: list[str]) -> list[dict]:
        """Let the model drive the read-only tools. Returns validation results."""
        validations: list[dict] = []
        tools = self._tools(trace, validations)
        by_name = {t.name: t for t in tools}

        try:
            model = shared_llm().bind_tools(tools)
            messages = [
                SystemMessage(content=AGENT_SYSTEM),
                HumanMessage(
                    content=f"Earlier messages:\n{as_text(history)}\n\nCustomer: {message}"
                ),
            ]
            for _ in range(MAX_TOOL_STEPS):
                reply: AIMessage = model.invoke(messages)
                messages.append(reply)
                calls = getattr(reply, "tool_calls", None) or []
                if not calls:
                    break
                for call in calls:
                    fn = by_name.get(call["name"])
                    output = (
                        fn.invoke(call["args"])
                        if fn
                        else {"error": f"unknown tool {call['name']}"}
                    )
                    messages.append(
                        ToolMessage(content=str(output), tool_call_id=call["id"])
                    )
        except Exception as error:
            trace.append(f"[Agent] tool loop unavailable ({type(error).__name__})")

        return validations

    # -- public entry points --------------------------------------------------
    def handle(self, message: str, state: RefundState, history: list[Turn]) -> AgentReply:
        """Process one customer message on the refund route."""
        if state.status == AWAITING_CONFIRMATION:
            return self._handle_confirmation_reply(message, state)
        return self._handle_request(message, state, history)

    def confirm(self, state: RefundState) -> AgentReply:
        """Explicit confirmation, e.g. the UI's Confirm Refund button."""
        trace = ["[User] confirmed"]
        return self._execute(state, trace)

    def cancel(self, state: RefundState) -> AgentReply:
        """Explicit rejection, e.g. the UI's Cancel button (FR-36d)."""
        trace = ["[User] rejected", "[Agent] refund_cancelled"]
        amount = state.amount
        state.finish(REFUND_CANCELLED)
        return AgentReply(
            text=(
                f"No problem - I have not processed the refund"
                f"{f' of {money(amount)}' if amount else ''}. "
                "Let me know if there is anything else I can help with."
            ),
            state=state,
            trace=trace,
        )

    # -- branches -------------------------------------------------------------
    def _handle_confirmation_reply(self, message: str, state: RefundState) -> AgentReply:
        verdict = read_confirmation(message)
        if verdict == CONFIRM:
            return self._execute(state, ["[User] confirmed"])
        if verdict == REJECT:
            return self.cancel(state)

        # FR-36e: unclear means ask again, never execute.
        trace = ["[User] reply unclear", "[Agent] awaiting_user_confirmation"]
        return AgentReply(
            text=(
                f"Sorry, I want to be certain before moving any money. "
                f"Should I go ahead with the refund of {money(state.amount)}? "
                "Please reply yes or no."
            ),
            state=state,
            trace=trace,
            awaiting_confirmation=True,
            amount=state.amount,
        )

    def _handle_request(
        self, message: str, state: RefundState, history: list[Turn]
    ) -> AgentReply:
        trace = ["[Intent] refund_request"]
        state.status = REFUND_REQUESTED

        # Deduplicated: "I recharged 5000, refund my 5000" names one amount
        # twice, not two competing amounts.
        stated = list(dict.fromkeys(amounts_in(message)))
        allowed = set(stated) | set(state.offered_amounts)

        validations = self._run_tool_loop(message, history, trace)

        # The model may propose an amount; it is accepted only if the customer
        # actually said it in *this* message, or we offered it. Anything else
        # would be invented (FR-44), and an amount recalled from an earlier turn
        # must not stand in for account data either (GR-10).
        amount: int | None = None
        for result in reversed(validations):
            candidate = result.get("requested_amount")
            if candidate in allowed:
                amount = candidate
                break
        if amount is None and len(stated) == 1:
            amount = stated[0]

        if amount is None:
            return self._resolve_missing_amount(state, trace)

        return self._decide(amount, state, trace)

    def _resolve_missing_amount(self, state: RefundState, trace: list[str]) -> AgentReply:
        """Section 6.8 - work out the amount from the account, or ask (FR-41 .. FR-44)."""
        eligible = refund_tools.refundable_recharges(self.user_id)

        if len(eligible) == 1:                                    # FR-42
            trace.append("[Agent] single eligible recharge; offering it")
            return self._decide(eligible[0]["recharge_amount"], state, trace)

        if len(eligible) > 1:                                     # FR-42a
            amounts = [r["recharge_amount"] for r in eligible]
            state.status = AWAITING_AMOUNT
            state.offered_amounts = amounts
            trace.append("[Agent] awaiting_amount")
            listed = " and ".join(money(a) for a in amounts)
            return AgentReply(
                text=(
                    f"I can see more than one recent recharge - {listed}. "
                    "Which amount would you like refunded?"
                ),
                state=state,
                trace=trace,
            )

        # FR-43: nothing to offer, so ask rather than guess.
        state.status = AWAITING_AMOUNT
        state.offered_amounts = []
        trace.append("[Agent] awaiting_amount")
        return AgentReply(
            text=(
                "I could not find a recent recharge I can refund. "
                "How much would you like refunded, and I will check it for you?"
            ),
            state=state,
            trace=trace,
        )

    def _decide(self, amount: int, state: RefundState, trace: list[str]) -> AgentReply:
        """Validate, then apply the approval limit. Both in code, never the LLM."""
        result = refund_tools.validate_refund(self.user_id, amount)
        if not any(f"validate_refund(amount={amount})" in line for line in trace):
            trace.append(f"[Agent] validate_refund(amount={amount})")
            trace.append(f"[Validation] eligible={result['eligible']}")

        if not result["eligible"]:                                 # FR-13, GR-01
            state.finish(REFUND_REJECTED)
            trace.append("[Agent] refund_rejected")
            return AgentReply(
                text=(
                    f"I cannot refund {money(amount)}. {result['reason']} {ESCALATION}"
                ),
                state=state,
                trace=trace,
            )

        state.amount = amount
        state.recharge_id = result["recharge_id"]
        state.status = REFUND_VALIDATED

        if amount > REFUND_APPROVAL_LIMIT:                         # FR-37 .. FR-40a
            return self._submit(amount, result, state, trace)

        state.status = AWAITING_CONFIRMATION            # FR-11, FR-36, FR-36b
        trace.append("[Agent] awaiting_user_confirmation")
        return AgentReply(
            text=(
                f"Refund eligible: {money(amount)}.\n\n"
                f"This is for your recharge of {money(result['recharge_amount'])} "
                f"({result['recharge_id']}).\n\n"
                "Do you want me to proceed with this refund?"
            ),
            state=state,
            trace=trace,
            awaiting_confirmation=True,
            amount=amount,
        )

    def _submit(
        self, amount: int, result: dict, state: RefundState, trace: list[str]
    ) -> AgentReply:
        """Above the limit: hand to the Support Team and stop (FR-38 .. FR-40a).

        The state settles as submitted, not awaiting confirmation, so no later
        turn can reach `process_refund` for this request (FR-39).
        """
        trace.append(f"[Agent] submit_to_support(amount={amount})")
        receipt = refund_tools.submit_to_support(self.user_id, amount, result["recharge_id"])
        trace.append("[Support] submitted_for_review")
        state.finish(REFUND_SUBMITTED_FOR_REVIEW)
        return AgentReply(
            text=(
                f"Your refund request for {money(amount)} has been submitted to our "
                "Support Team. They will review the request and process it or take "
                f"the necessary action.\n\nYour reference is {receipt['ticket_id']}."
            ),
            state=state,
            trace=trace,
        )

    def _execute(self, state: RefundState, trace: list[str]) -> AgentReply:
        """Run the simulated refund. Only reachable from AWAITING_CONFIRMATION."""
        if state.status != AWAITING_CONFIRMATION or state.amount is None:
            trace.append("[Agent] no refund awaiting confirmation")
            state.reset()
            return AgentReply(
                text="I do not have a refund waiting for confirmation right now.",
                state=state,
                trace=trace,
            )

        amount, recharge_id = state.amount, state.recharge_id
        if amount > REFUND_APPROVAL_LIMIT:      # belt and braces for GR-08
            trace.append("[Agent] blocked: above approval limit")
            return self._submit(
                amount,
                {"recharge_id": recharge_id, "recharge_amount": amount},
                state,
                trace,
            )

        state.status = REFUND_CONFIRMED
        trace.append(f"[Agent] process_refund(amount={amount})")
        receipt = refund_tools.process_refund(self.user_id, amount, recharge_id)

        if not receipt["success"]:
            trace.append("[Refund] rejected")
            state.finish(REFUND_REJECTED)
            return AgentReply(
                text=(
                    f"I could not complete the refund of {money(amount)}. "
                    f"{receipt.get('reason', '')} {ESCALATION}".strip()
                ),
                state=state,
                trace=trace,
            )

        trace.append("[Refund] success")
        state.finish(REFUND_PROCESSED)
        return AgentReply(
            text=(
                f"Done - I have refunded {money(amount)}.\n\n"
                f"Your refund reference is {receipt['refund_id']}. "
                "The amount should reach your original payment method within "
                "3-5 business days."
            ),
            state=state,
            trace=trace,
        )
