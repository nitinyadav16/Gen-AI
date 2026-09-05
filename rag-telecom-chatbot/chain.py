"""LCEL chain: retrieve -> prompt -> Groq LLM -> streamed string (FR-10 .. FR-13)."""

from __future__ import annotations

from collections.abc import Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import ESCALATION_HINT, TOP_K
from conversation import Turn, to_messages
from llm import build_llm
from retriever import MergedRetriever, RetrievedDoc, format_context

SYSTEM_PROMPT = f"""You are the NovaCell telecom customer-care assistant. You help \
subscribers resolve everyday mobile problems: connectivity, data, billing, SIM and \
eSIM, roaming, voice calls, devices, and account or app questions.

Answer using ONLY the CONTEXT below. The context is retrieved from NovaCell's own \
knowledge sources: published FAQ entries, resolved support tickets, and official user \
guides.

Rules you must follow:
1. Never use knowledge from outside the CONTEXT. Do not rely on anything you know \
about other operators, prices, plans, or policies.
2. Never invent prices, plan names, timelines, phone numbers, URLs, or policy details. \
If a figure is not in the CONTEXT, do not state one.
3. If the CONTEXT does not contain enough information to answer confidently, say so \
plainly in one sentence and tell the customer to {ESCALATION_HINT}. Do not guess.
4. You have no access to the customer's account, balance, or live billing data. For \
anything account-specific ("my balance", "my last bill", "my plan"), explain what the \
CONTEXT says about how to check it in the MyTelecom app, and otherwise direct the \
customer to {ESCALATION_HINT}.
5. Answer in plain English, no jargon. Prefer short numbered steps for troubleshooting.
6. Be concise: a short paragraph, or up to about six steps.
7. Never refer to your own sources in the answer. Do not write "the context", "the \
documents", "the provided information", "based on the information given", or any source \
identifier - the interface displays sources separately. Speak straight to the customer: \
say "NovaCell does not offer that" or "I do not have information on that", never "the \
provided information does not contain that".

CONTEXT:
{{context}}"""

# Earlier turns sit between the grounding rules and the new question, so a
# follow-up like "should I do that before I travel?" can resolve "that" (FR-46).
PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history", optional=True),
        ("human", "{question}"),
    ]
)


def build_generation_chain():
    """prompt | llm | parser - streams tokens as they arrive (FR-05)."""
    return PROMPT | build_llm() | StrOutputParser()


class RagPipeline:
    """Retrieval + generation, exposed so the UI can show sources alongside the answer.

    `retrieve()` is called first so the interface can render the Sources panel
    (FR-13a) while the answer is still streaming.
    """

    def __init__(self, k: int = TOP_K):
        self.retriever = MergedRetriever(k=k)
        self.chain = build_generation_chain()

    def retrieve(self, question: str) -> list[RetrievedDoc]:
        """Retrieval runs on the question exactly as asked.

        Conversation history is deliberately kept out of this call: it informs
        the answer, never the search query (FR-49).
        """
        return self.retriever.retrieve(question)

    def _payload(
        self, question: str, documents: list[RetrievedDoc], history: list[Turn] | None
    ) -> dict:
        return {
            "context": format_context(documents),
            "question": question,
            "history": to_messages(history or []),
        }

    def stream(
        self,
        question: str,
        documents: list[RetrievedDoc],
        history: list[Turn] | None = None,
    ) -> Iterator[str]:
        yield from self.chain.stream(self._payload(question, documents, history))

    def answer(
        self, question: str, history: list[Turn] | None = None
    ) -> tuple[str, list[RetrievedDoc]]:
        documents = self.retrieve(question)
        return self.chain.invoke(self._payload(question, documents, history)), documents
