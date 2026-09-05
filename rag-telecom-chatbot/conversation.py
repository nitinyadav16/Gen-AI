"""Session conversation history, shared by both routes (FR-45 .. FR-50).

The history is passed to the RAG pipeline and to the refund agent as prior
turns. It never touches the retrieval query: `MergedRetriever.retrieve()` still
receives the user's question verbatim (FR-49).
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from config import HISTORY_EXCHANGES


@dataclass(frozen=True)
class Turn:
    role: str      # "user" | "assistant"
    content: str


def recent(turns: list[Turn], exchanges: int = HISTORY_EXCHANGES) -> list[Turn]:
    """The last `exchanges` user+assistant pairs (FR-50).

    Capping the window keeps the prompt small and stops an old, unrelated topic
    from colouring a fresh question.
    """
    if exchanges <= 0:
        return []
    return turns[-(exchanges * 2) :]


def to_messages(turns: list[Turn]) -> list[BaseMessage]:
    """Prior turns as LangChain messages, for a MessagesPlaceholder."""
    out: list[BaseMessage] = []
    for turn in turns:
        if not turn.content:
            continue
        out.append(
            HumanMessage(content=turn.content)
            if turn.role == "user"
            else AIMessage(content=turn.content)
        )
    return out


def as_text(turns: list[Turn]) -> str:
    """Prior turns as a plain transcript, for prompts that take a string."""
    if not turns:
        return "(no earlier messages)"
    label = {"user": "Customer", "assistant": "Assistant"}
    return "\n".join(f"{label.get(t.role, t.role)}: {t.content}" for t in turns if t.content)


def from_messages(messages: list[dict]) -> list[Turn]:
    """Adapt the UI's message dicts (which carry sources, ids, ...) to turns."""
    return [
        Turn(role=m["role"], content=m.get("content", ""))
        for m in messages
        if m.get("role") in ("user", "assistant")
    ]
