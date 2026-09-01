"""Streamlit UI for the RAG telecom customer-care chatbot.

    streamlit run app.py

Behaviour lives here; all styling lives in `ui.py`.
"""

from __future__ import annotations

import uuid

import streamlit as st

import ui
from config import CHROMA_DIR, GROQ_API_KEY, GROQ_MODEL
from interaction_log import log_answer, log_feedback
from retriever import COLLECTIONS
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
    "Ask about data, billing, SIM and eSIM, roaming, calls, or your account. "
    "Every answer is drawn from NovaCell's own knowledge - never invented."
)


@st.cache_resource(show_spinner="Loading the knowledge base…")
def get_pipeline():
    from chain import RagPipeline

    return RagPipeline()


def knowledge_status() -> list[tuple]:
    return [(spec, count(spec.name)) for spec in COLLECTIONS]


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending", None)
    st.session_state.setdefault("logged_feedback", {})


def sidebar(status: list[tuple]) -> None:
    with st.sidebar:
        ui.sidebar_brand()

        ui.label("Try asking")
        for index, question in enumerate(SAMPLE_QUESTIONS):
            if st.button(question, key=f"sample-{index}"):
                st.session_state.pending = question

        ui.label("Session")
        if st.button("Clear conversation", key="clear", type="primary"):
            st.session_state.messages = []
            st.session_state.pending = None
            st.session_state.logged_feedback = {}
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
                render_feedback(message)


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
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    turn_id = uuid.uuid4().hex[:12]

    with st.chat_message("assistant"):
        try:
            pipeline = get_pipeline()
            with st.spinner("Searching NovaCell's knowledge…"):
                documents = pipeline.retrieve(question)
            answer = st.write_stream(pipeline.stream(question, documents))
        except Exception as error:  # network / API-key / model failures
            answer = FALLBACK_ANSWER
            documents = []
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

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "turn_id": turn_id,
            "question": question,
        }
    )
    log_answer(turn_id, question, answer, [s["citation"] for s in sources])
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
    else:
        welcome(status)

    typed = st.chat_input("Ask a question about your NovaCell service…")
    question = typed or st.session_state.pending
    if question:
        st.session_state.pending = None
        answer_question(question)


main()
