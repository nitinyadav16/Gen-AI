"""Intent detection: `requirement_inquiry` or `refund_request` (FR-01 .. FR-05a).

Runs on the existing Groq model (FR-04) and returns one of two fixed values
(FR-05). The distinction that matters is FR-05a: *asking to have money back* is
a refund request, whereas *asking how refunds work* is an ordinary support
question that belongs on the RAG route.
"""

from __future__ import annotations

import re

from langchain_core.prompts import ChatPromptTemplate

from conversation import Turn, as_text
from llm import shared_llm

REQUIREMENT_INQUIRY = "requirement_inquiry"
REFUND_REQUEST = "refund_request"

SYSTEM_PROMPT = """You label a telecom customer's latest message with exactly one intent.

refund_request
  The customer wants money back, or wants a recent recharge cancelled or
  reversed, or is replying about a refund already under discussion.
  Examples: "I want a refund of 999", "refund my last recharge",
  "cancel my recharge and return the money", "I want a refund".

requirement_inquiry
  Everything else: support questions, troubleshooting, billing explanations,
  and questions *about* how refunds or cancellations work rather than asking
  for one.
  Examples: "Why is my internet slow?", "How do I cancel a recharge?",
  "What is your refund policy?", "How do I activate roaming?".

The deciding question: is the customer asking for a refund to happen (refund_request),
or asking for information (requirement_inquiry)?

Reply with the single word requirement_inquiry or refund_request. Nothing else."""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Earlier messages:\n{history}\n\nLatest message:\n{question}"),
    ]
)

# Wanting money back, as opposed to asking how refunds work.
_REFUND_ASK = re.compile(
    r"\b(refund|reimburse|money\s+back|charge\s*back|chargeback)\b", re.IGNORECASE
)
_REVERSE_ASK = re.compile(
    r"\b(cancel|reverse|undo|revert)\b.{0,30}\b(recharge|payment|transaction|top[\s-]?up|plan)\b",
    re.IGNORECASE,
)
# Phrasings that make it a question about policy rather than a request.
_INFORMATIONAL = re.compile(
    r"\b(how|what|when|where|why|does|do|can\s+i\s+know|policy|eligible|eligibility)\b",
    re.IGNORECASE,
)
_WANTS = re.compile(
    r"\b(i\s+want|i\s+need|i'?d\s+like|give\s+me|please\s+refund|get\s+me|issue\s+me|"
    r"process\s+my|claim)\b",
    re.IGNORECASE,
)


def _rule_intent(question: str) -> str | None:
    """A confident rule-based answer, or None to defer to the model.

    Only the clear-cut ends are decided here; anything in between goes to the
    LLM rather than being forced into a guess.
    """
    mentions_refund = bool(_REFUND_ASK.search(question) or _REVERSE_ASK.search(question))
    if not mentions_refund:
        return None
    if _WANTS.search(question):
        # "I want a refund" stays a request even though it contains no question word.
        return REFUND_REQUEST
    if _INFORMATIONAL.search(question):
        return None  # "how do I get a refund?" is genuinely ambiguous - let the LLM judge
    return None


def classify_intent(question: str, history: list[Turn] | None = None) -> str:
    """Return `requirement_inquiry` or `refund_request`.

    Falls back to the RAG route on any error: a support answer for a refund
    request is a poor experience, but a refund flow triggered by a misfire is a
    worse one.
    """
    ruled = _rule_intent(question)
    if ruled is not None:
        return ruled

    try:
        reply = (PROMPT | shared_llm()).invoke(
            {"question": question, "history": as_text(history or [])}
        )
        text = (reply.content if hasattr(reply, "content") else str(reply)).strip().lower()
    except Exception:
        return REQUIREMENT_INQUIRY

    if REFUND_REQUEST in text:
        return REFUND_REQUEST
    return REQUIREMENT_INQUIRY
