"""Offline smoke test: ingestion + retrieval, no Groq call required.

    python smoke_test.py
"""

from __future__ import annotations

import sys

from retriever import COLLECTIONS, MergedRetriever, format_context
from vectorstore import count

QUESTIONS = [
    "Why is my mobile internet so slow?",
    "How do I activate roaming before I travel?",
    "My SIM is not recognised.",
]


def main() -> int:
    failures: list[str] = []

    print("== collections ==")
    for spec in COLLECTIONS:
        n = count(spec.name)
        print(f"  {spec.name:<8} {n:>4} docs")
        if n == 0:
            failures.append(f"collection '{spec.name}' is empty - run ingest_all.py")

    retriever = MergedRetriever()
    for question in QUESTIONS:
        documents = retriever.retrieve(question)
        labels = sorted({doc.collection for doc in documents})
        print(f"\n== {question}\n  {len(documents)} docs from {labels}")
        for doc in documents:
            print(f"    {doc.citation:<22} {doc.content[:70].replace(chr(10), " ")}...")
        if len(documents) != len(COLLECTIONS) * retriever.k:
            failures.append(f"expected 9 documents for {question!r}, got {len(documents)}")

    context = format_context(retriever.retrieve(QUESTIONS[0]))
    for spec in COLLECTIONS:
        if f"SOURCE={spec.context_label}" not in context:
            failures.append(f"context block is missing a {spec.context_label} label")

    print("\n== context preview ==")
    print(context[:400] + "...")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
