"""Ingest resolved tickets from data/tickets.db into the `tickets` collection (FR-15).

One resolved ticket becomes one vector document holding the customer-reported
issue and the resolution that closed it. Re-running upserts on `ticket_id`, so
newly seeded cases are picked up without duplicating existing ones (US-07).

    python ingest_tickets.py [--reset]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys

from langchain_core.documents import Document

from config import TICKETS_DB
from vectorstore import count, reset_collection, upsert

COLLECTION = "tickets"

QUERY = """
    SELECT ticket_id, category, issue_type, description, resolution
    FROM tickets
    WHERE lower(status) = 'resolved'
    ORDER BY id
"""


def load_documents() -> tuple[list[Document], list[str]]:
    if not TICKETS_DB.exists():
        raise FileNotFoundError(f"Ticket database not found: {TICKETS_DB}")

    connection = sqlite3.connect(TICKETS_DB)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(QUERY).fetchall()
    finally:
        connection.close()

    documents: list[Document] = []
    ids: list[str] = []

    for row in rows:
        ticket_id = row["ticket_id"]
        body = (
            f"Issue: {row['issue_type']}\n"
            f"Category: {row['category']}\n"
            f"Customer report: {row['description']}\n"
            f"Resolution: {row['resolution']}"
        )
        documents.append(
            Document(
                page_content=body,
                metadata={
                    "source": COLLECTION,
                    "source_id": ticket_id,
                    "title": row["issue_type"],
                    "category": row["category"],
                },
            )
        )
        ids.append(ticket_id)

    return documents, ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest resolved tickets into ChromaDB.")
    parser.add_argument(
        "--reset", action="store_true", help="drop the collection before ingesting"
    )
    args = parser.parse_args()

    if args.reset:
        reset_collection(COLLECTION)

    documents, ids = load_documents()
    written = upsert(COLLECTION, documents, ids)
    print(f"[tickets] ingested {written} document(s); collection now holds {count(COLLECTION)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
