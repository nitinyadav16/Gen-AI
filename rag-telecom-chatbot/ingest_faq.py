"""Ingest data/faq.csv into the `faq` Chroma collection (FR-14).

One CSV row becomes one vector document. Re-running the script upserts on the
stable id `FAQ-<id>`, so it is safe to run after every FAQ edit (US-06).

    python ingest_faq.py [--reset]
"""

from __future__ import annotations

import argparse
import csv
import sys

from langchain_core.documents import Document

from config import FAQ_CSV
from vectorstore import count, reset_collection, upsert

COLLECTION = "faq"


def load_documents() -> tuple[list[Document], list[str]]:
    if not FAQ_CSV.exists():
        raise FileNotFoundError(f"FAQ source not found: {FAQ_CSV}")

    documents: list[Document] = []
    ids: list[str] = []

    with FAQ_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for position, row in enumerate(csv.DictReader(handle), start=1):
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not question or not answer:
                continue

            raw_id = (row.get("id") or str(position)).strip()
            chroma_key = f"FAQ-{raw_id}"
            category = (row.get("category") or "general").strip()

            documents.append(
                Document(
                    page_content=f"Q: {question}\nA: {answer}",
                    metadata={
                        "source": COLLECTION,
                        "source_id": raw_id,
                        "title": question,
                        "category": category,
                    },
                )
            )
            ids.append(chroma_key)

    return documents, ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest FAQ entries into ChromaDB.")
    parser.add_argument(
        "--reset", action="store_true", help="drop the collection before ingesting"
    )
    args = parser.parse_args()

    if args.reset:
        reset_collection(COLLECTION)

    documents, ids = load_documents()
    written = upsert(COLLECTION, documents, ids)
    print(f"[faq] ingested {written} document(s); collection now holds {count(COLLECTION)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
