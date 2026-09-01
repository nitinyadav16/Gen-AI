"""Interactive CLI for the RAG telecom customer-care chatbot (FR-18, FR-19).

    python main.py

Type `quit` (or `exit`) to leave the session.
"""

from __future__ import annotations

import sys
import uuid

from interaction_log import log_answer
from retriever import COLLECTIONS, knowledge_base_is_empty
from sample_questions import SAMPLE_QUESTIONS
from vectorstore import count

BANNER = """
================================================================
 NovaCell Support Assistant  -  grounded in FAQ, tickets, guides
 Type your question, or `quit` to exit.  `samples` lists examples.
================================================================
"""


def print_status() -> None:
    parts = [f"{spec.source_label}: {count(spec.name)}" for spec in COLLECTIONS]
    print("Knowledge base -> " + " | ".join(parts))


def main() -> int:
    print(BANNER)

    if knowledge_base_is_empty():
        print("The knowledge base is empty. Run `python ingest_all.py` first.")
        return 1

    print_status()

    try:
        from chain import RagPipeline

        pipeline = RagPipeline()
    except Exception as error:
        print(f"Could not start the assistant: {error}")
        return 1

    while True:
        try:
            question = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Goodbye.")
            return 0
        if question.lower() == "samples":
            for sample in SAMPLE_QUESTIONS:
                print(f"  - {sample}")
            continue

        turn_id = uuid.uuid4().hex[:12]
        try:
            documents = pipeline.retrieve(question)
            print("\nAssistant > ", end="", flush=True)
            chunks: list[str] = []
            for token in pipeline.stream(question, documents):
                chunks.append(token)
                print(token, end="", flush=True)
            answer = "".join(chunks)
            print()
        except Exception as error:
            print(f"\n[error] {error}")
            continue

        citations = [doc.citation for doc in documents]
        if citations:
            print("\nSources:")
            for citation in citations:
                print(f"  - {citation}")

        log_answer(turn_id, question, answer, citations)


if __name__ == "__main__":
    sys.exit(main())
