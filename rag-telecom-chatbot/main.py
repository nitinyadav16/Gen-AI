"""Interactive CLI for the telecom assistant and its refund agent (FR-18, FR-19).

    python main.py

Type `quit` (or `exit`) to leave the session, `samples` for example questions,
`account` to inspect the simulated demo account, and `clear` to reset the
conversation and any refund in flight.
"""

from __future__ import annotations

import sys
import uuid

from accounts import DEMO_USER_ID, account_snapshot, reset_accounts
from config import CURRENCY
from conversation import Turn
from interaction_log import log_answer, log_workflow
from refund_agent import AWAITING_CONFIRMATION, RefundState
from retriever import COLLECTIONS, knowledge_base_is_empty
from router import REFUND_REQUEST, Assistant
from sample_questions import SAMPLE_QUESTIONS
from vectorstore import count

BANNER = """
================================================================
 NovaCell Support Assistant  -  grounded answers + refund agent
 Ask a question, or request a refund on a recent recharge.
 Commands: samples | account | clear | quit
================================================================
"""


def print_status() -> None:
    parts = [f"{spec.source_label}: {count(spec.name)}" for spec in COLLECTIONS]
    print("Knowledge base -> " + " | ".join(parts))


def print_account() -> None:
    account = account_snapshot(DEMO_USER_ID)
    if not account:
        print("No demo account loaded.")
        return
    print(f"\n{account['name']} - {account['plan']} plan")
    for recharge in sorted(account["recharges"], key=lambda r: r["recharge_age_days"]):
        state = (
            "refunded" if recharge["refund_status"] != "not_refunded" else "refundable"
        )
        print(
            f"  {CURRENCY}{recharge['recharge_amount']:<6,} {recharge['recharge_id']}"
            f"  {recharge['recharge_age_days']}d ago  [{state}]"
        )


def main() -> int:
    # The rupee sign is outside the Windows console's default codepage.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print(BANNER)

    if knowledge_base_is_empty():
        print("The knowledge base is empty. Run `python ingest_all.py` first.")
        return 1

    print_status()

    assistant = Assistant(DEMO_USER_ID)
    refund_state = RefundState()
    turns: list[Turn] = []

    while True:
        try:
            prompt = "Confirm? > " if refund_state.status == AWAITING_CONFIRMATION else "You > "
            message = input(f"\n{prompt}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if not message:
            continue
        lowered = message.lower()
        if lowered in {"quit", "exit"}:
            print("Goodbye.")
            return 0
        if lowered == "samples":
            for sample in SAMPLE_QUESTIONS:
                print(f"  - {sample}")
            continue
        if lowered == "account":
            print_account()
            continue
        if lowered == "clear":
            turns = []
            refund_state = RefundState()
            print("Conversation cleared.")
            continue
        if lowered == "reset":
            reset_accounts()
            turns = []
            refund_state = RefundState()
            print("Demo recharges restored.")
            continue

        turn_id = uuid.uuid4().hex[:12]
        history = list(turns)

        try:
            intent = assistant.route(message, refund_state, history)
        except Exception as error:
            print(f"\n[error] {error}")
            continue

        turns.append(Turn("user", message))

        if intent == REFUND_REQUEST:
            try:
                reply = assistant.handle_refund(message, refund_state, history)
            except Exception as error:
                print(f"\n[error] {error}")
                continue
            refund_state = reply.state
            print(f"\nAssistant > {reply.text}")
            if reply.trace:
                print("\nWorkflow:")
                for step in reply.trace:
                    print(f"  {step}")
            turns.append(Turn("assistant", reply.text))
            log_answer(turn_id, message, reply.text, [])
            log_workflow(turn_id, message, REFUND_REQUEST, reply.trace)
            continue

        # requirement_inquiry -> the existing RAG pipeline
        trace = ["[Intent] requirement_inquiry"]
        try:
            documents = assistant.retrieve(message)
            trace.append(f"[RAG] retrieved {len(documents)} documents")
            print("\nAssistant > ", end="", flush=True)
            chunks: list[str] = []
            for token in assistant.stream(message, documents, history):
                chunks.append(token)
                print(token, end="", flush=True)
            answer = "".join(chunks)
            print()
        except Exception as error:
            print(f"\n[error] {error}")
            turns.pop()
            continue

        citations = [doc.citation for doc in documents]
        if citations:
            print("\nSources:")
            for citation in citations:
                print(f"  - {citation}")

        turns.append(Turn("assistant", answer))
        log_answer(turn_id, message, answer, citations)
        log_workflow(turn_id, message, "requirement_inquiry", trace)


if __name__ == "__main__":
    sys.exit(main())
