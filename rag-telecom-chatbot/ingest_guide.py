"""Ingest data/telecom_guide.pdf into the `guides` collection (FR-16).

The PDF is split into 600-character chunks with 100 characters of overlap.
Chunk boundaries move whenever the PDF changes, so this script always rebuilds
the collection from scratch - which keeps repeated runs idempotent (FR-17).

    python ingest_guide.py
"""

from __future__ import annotations

import argparse
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, GUIDE_PDF
from vectorstore import count, reset_collection, upsert

COLLECTION = "guides"


def load_documents() -> tuple[list[Document], list[str]]:
    if not GUIDE_PDF.exists():
        raise FileNotFoundError(f"Guide PDF not found: {GUIDE_PDF}")

    pages = PyPDFLoader(str(GUIDE_PDF)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    documents: list[Document] = []
    ids: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        if not chunk.page_content.strip():
            continue
        page_number = int(chunk.metadata.get("page", 0)) + 1
        chunk_id = f"GUIDE-{index:04d}"
        display_id = f"p.{page_number} #{index}"
        documents.append(
            Document(
                page_content=chunk.page_content.strip(),
                metadata={
                    "source": COLLECTION,
                    "source_id": display_id,
                    "title": f"{GUIDE_PDF.name} p.{page_number}",
                    "page": page_number,
                },
            )
        )
        ids.append(chunk_id)

    return documents, ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest the telecom guide PDF into ChromaDB.")
    parser.add_argument(
        "--keep",
        action="store_true",
        help="upsert without dropping the collection first (default: rebuild)",
    )
    args = parser.parse_args()

    if not args.keep:
        reset_collection(COLLECTION)

    documents, ids = load_documents()
    written = upsert(COLLECTION, documents, ids)
    print(f"[guides] ingested {written} chunk(s); collection now holds {count(COLLECTION)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
