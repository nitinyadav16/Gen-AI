"""Intent routing across the two flows (section 7).

    requirement_inquiry  ->  the existing RAG pipeline, unchanged (FR-02)
    refund_request       ->  the Refund Agent (FR-03)

Both interfaces (Streamlit and the CLI) go through `Assistant` so the routing,
history handling and logging behave identically in each.
"""

from __future__ import annotations

from accounts import DEMO_USER_ID
from conversation import Turn, recent
from intent import REFUND_REQUEST, REQUIREMENT_INQUIRY, classify_intent
from refund_agent import AgentReply, RefundAgent, RefundState

__all__ = [
    "Assistant",
    "AgentReply",
    "RefundState",
    "REFUND_REQUEST",
    "REQUIREMENT_INQUIRY",
]


class Assistant:
    """Routes a message, then hands off to RAG or the refund agent."""

    def __init__(self, user_id: str = DEMO_USER_ID):
        self.agent = RefundAgent(user_id)
        self._pipeline = None

    @property
    def pipeline(self):
        """The RAG pipeline, built on first use.

        Loading it pulls in the embedding model, which a pure refund
        conversation never needs.
        """
        if self._pipeline is None:
            from chain import RagPipeline

            self._pipeline = RagPipeline()
        return self._pipeline

    def history(self, turns: list[Turn]) -> list[Turn]:
        """The recent window shared by both routes (FR-47, FR-50)."""
        return recent(turns)

    def route(self, message: str, state: RefundState, turns: list[Turn]) -> str:
        """Decide which flow handles this message.

        A refund already awaiting an amount or a confirmation keeps the refund
        route, so a bare "yes" or "499" is read as part of that exchange rather
        than being reclassified from scratch.
        """
        if state.pending:
            return REFUND_REQUEST
        return classify_intent(message, self.history(turns))

    def handle_refund(
        self, message: str, state: RefundState, turns: list[Turn]
    ) -> AgentReply:
        return self.agent.handle(message, state, self.history(turns))

    def retrieve(self, question: str):
        return self.pipeline.retrieve(question)

    def stream(self, question: str, documents, turns: list[Turn]):
        return self.pipeline.stream(question, documents, self.history(turns))

    def answer(self, question: str, turns: list[Turn]):
        return self.pipeline.answer(question, self.history(turns))
