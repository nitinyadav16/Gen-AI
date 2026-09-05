"""Append-only interaction log for answers and 👍/👎 feedback (FR-05a)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from config import INTERACTION_LOG, LOG_DIR


def _write(record: dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    with INTERACTION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_answer(turn_id: str, question: str, answer: str, citations: list[str]) -> None:
    _write(
        {
            "event": "answer",
            "turn_id": turn_id,
            "question": question,
            "answer": answer,
            "sources": citations,
        }
    )


def log_feedback(turn_id: str, question: str, rating: str) -> None:
    """rating is "up" or "down"."""
    _write(
        {
            "event": "feedback",
            "turn_id": turn_id,
            "question": question,
            "rating": rating,
        }
    )


def log_workflow(turn_id: str, question: str, intent: str, trace: list[str]) -> None:
    """Record the routed intent and the agent's workflow steps (section 12).

    Only the discrete steps the application took - never the model's private
    reasoning.
    """
    _write(
        {
            "event": "workflow",
            "turn_id": turn_id,
            "question": question,
            "intent": intent,
            "trace": trace,
        }
    )
