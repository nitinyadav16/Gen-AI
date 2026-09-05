"""Streamlit UI for the RAG telecom assistant and its refund agent.

    streamlit run app.py

Behaviour lives here; all styling lives in `ui.py`. One chat surface serves both
routes (section 11) - there is no separate refund dashboard.
"""

from __future__ import annotations

import uuid

import streamlit as st

import ui
from accounts import DEMO_USER_ID, account_snapshot, reset_accounts
from config import CHROMA_DIR, GROQ_API_KEY, GROQ_MODEL
from conversation import from_messages
from interaction_log import log_answer, log_feedback, log_workflow
from refund_agent import AWAITING_CONFIRMATION, RefundState
from retriever import COLLECTIONS
from router import REFUND_REQUEST, Assistant
from sample_questions import SAMPLE_QUESTIONS
from vectorstore import count

st.set_page_config(
    page_title="NovaCell Support Assistant",
    page_icon="📶",
    layout="centered",
    initial_sidebar_state="expanded",
)

FALLBACK_ANSWER = (
    "Sorry - I could not reach the answering service just now. "
    "Please try again, or call 611 / use the MyTelecom app."
)
SUBTITLE = (
    "Ask about data, billing, SIM and eSIM, roaming, calls, or your account - "
    "or request a refund on a recent recharge."
)


@st.cache_resource(show_spinner="Loading the knowledge base…")
def get_assistant() -> Assistant:
    return Assistant(DEMO_USER_ID)


def knowledge_status() -> list[tuple]:
    return [(spec, count(spec.name)) for spec in COLLECTIONS]


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending", None)
    st.session_state.setdefault("logged_feedback", {})
    st.session_state.setdefault("refund_state", RefundState())


def clear_conversation() -> None:
    """Reset history and any refund in flight (FR-48, section 11)."""
    st.session_state.messages = []
    st.session_state.pending = None
    st.session_state.logged_feedback = {}
    st.session_state.refund_state = RefundState()


def sidebar(status: list[tuple]) -> None:
    with st.sidebar:
        ui.sidebar_brand()

        ui.label("Try asking")
        for index, question in enumerate(SAMPLE_QUESTIONS):
            if st.button(question, key=f"sample-{index}"):
                st.session_state.pending = question

        ui.label("Session")
        if st.button("Clear conversation", key="clear", type="primary"):
            clear_conversation()
            st.rerun()

        ui.label("Demo account")
        ui.account_panel(account_snapshot(DEMO_USER_ID))
        if st.button("Reset demo recharges", key="reset-accounts"):
            reset_accounts()
            clear_conversation()
            st.rerun()

        ui.label("Knowledge base")
        ui.knowledge_rows(status)
        ui.meta(GROQ_MODEL, f"{CHROMA_DIR.name}/")


def render_sources(sources: list[dict]) -> None:
    """Expandable Sources panel listing every retrieved document (FR-13a)."""
    if not sources:
        return
    with st.expander(f"Sources · {len(sources)} documents"):
        ui.source_cards(sources)


def render_trace(trace: list[str]) -> None:
    """The workflow steps this turn took (section 12). No model reasoning."""
    if not trace:
        return
    with st.expander("Workflow trace"):
        ui.trace_block(trace)


def render_feedback(message: dict) -> None:
    """👍 / 👎 control written to the interaction log (FR-05a)."""
    turn_id = message["turn_id"]
    choice = st.feedback("thumbs", key=f"feedback-{turn_id}")
    if choice is None:
        return
    rating = "up" if choice == 1 else "down"
    if st.session_state.logged_feedback.get(turn_id) != rating:
        st.session_state.logged_feedback[turn_id] = rating
        log_feedback(turn_id, message.get("question", ""), rating)


def render_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))
                render_trace(message.get("trace", []))
                render_feedback(message)


def confirmation_controls() -> None:
    """Explicit confirm / cancel for an eligible refund (section 11, FR-36)."""
    state = st.session_state.refund_state
    if state.status != AWAITING_CONFIRMATION:
        return

    assistant = get_assistant()
    left, right, _ = st.columns([1, 1, 2])
    with left:
        if st.button("Confirm refund", key="confirm-refund", type="primary"):
            reply = assistant.agent.confirm(state)
            record_assistant(reply.text, trace=reply.trace, question="[confirm refund]",
                             intent=REFUND_REQUEST)
            st.rerun()
    with right:
        if st.button("Cancel", key="cancel-refund"):
            reply = assistant.agent.cancel(state)
            record_assistant(reply.text, trace=reply.trace, question="[cancel refund]",
                             intent=REFUND_REQUEST)
            st.rerun()


def record_assistant(
    text: str,
    sources: list[dict] | None = None,
    trace: list[str] | None = None,
    question: str = "",
    intent: str = "",
) -> None:
    turn_id = uuid.uuid4().hex[:12]
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": text,
            "sources": sources or [],
            "trace": trace or [],
            "turn_id": turn_id,
            "question": question,
        }
    )
    log_answer(turn_id, question, text, [s["citation"] for s in (sources or [])])
    if trace:
        log_workflow(turn_id, question, intent, trace)


def welcome(status: list[tuple]) -> None:
    """Hero + suggestion chips, shown only while the conversation is empty."""
    ui.hero(SUBTITLE)
    ui.knowledge_pills(status)
    ui.label("Popular questions")
    columns = st.columns(2)
    for index, question in enumerate(SAMPLE_QUESTIONS[:4]):
        with columns[index % 2]:
            if st.button(question, key=f"chip-{index}"):
                st.session_state.pending = question
                st.rerun()


def answer_question(question: str) -> None:
    assistant = get_assistant()
    state = st.session_state.refund_state

    # History is the conversation *before* this message (FR-46).
    history = from_messages(st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            intent = assistant.route(question, state, history)
        except Exception:
            intent = "requirement_inquiry"

        if intent == REFUND_REQUEST:
            with st.spinner("Checking your account…"):
                reply = assistant.handle_refund(question, state, history)
            st.markdown(reply.text)
            render_trace(reply.trace)
            st.session_state.refund_state = reply.state
            record_assistant(reply.text, trace=reply.trace, question=question, intent=intent)
            st.rerun()
            return

        # requirement_inquiry: the existing RAG pipeline, unchanged (FR-02).
        trace = ["[Intent] requirement_inquiry"]
        try:
            with st.spinner("Searching NovaCell's knowledge…"):
                documents = assistant.retrieve(question)
            trace.append(f"[RAG] retrieved {len(documents)} documents")
            answer = st.write_stream(assistant.stream(question, documents, history))
        except Exception as error:
            answer = FALLBACK_ANSWER
            documents = []
            trace.append("[RAG] failed")
            st.error(f"{FALLBACK_ANSWER}\n\nDetails: {error}")

        sources = [
            {
                "label": doc.source_label,
                "ident": doc.source_id,
                "citation": doc.citation,
                "title": doc.title,
                "content": doc.content,
            }
            for doc in documents
        ]
        render_sources(sources)
        render_trace(trace)

    record_assistant(answer, sources=sources, trace=trace, question=question, intent=intent)
    st.rerun()


def main() -> None:
    init_state()
    ui.inject_css()
    status = knowledge_status()
    sidebar(status)

    if all(n == 0 for _, n in status):
        ui.hero(SUBTITLE)
        st.warning("The knowledge base is empty. Run `python ingest_all.py` once, then reload.")
        return

    if not GROQ_API_KEY:
        ui.hero(SUBTITLE)
        st.error(
            "No `GROQ_API_KEY` found. Copy `.env.example` to `.env`, add your key from "
            "https://console.groq.com/keys, then restart the app."
        )
        return

    if st.session_state.messages:
        ui.compact_header("Grounded in FAQ, tickets and guides")
        render_history()
        confirmation_controls()
    else:
        welcome(status)

    typed = st.chat_input("Ask a question, or request a refund…")
    question = typed or st.session_state.pending
    if question:
        st.session_state.pending = None
        answer_question(question)


main()
