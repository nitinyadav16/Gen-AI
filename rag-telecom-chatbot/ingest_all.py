"""Run every ingest script in one pass.

    python ingest_all.py
"""

from __future__ import annotations

import sys

import ingest_faq
import ingest_guide
import ingest_tickets
from vectorstore import count, reset_collection, upsert

# (module, rebuild-from-scratch?) - the guide is rebuilt because chunk
# boundaries shift whenever the PDF changes.
SOURCES = ((ingest_faq, False), (ingest_tickets, False), (ingest_guide, True))


def main() -> int:
    for module, rebuild in SOURCES:
        documents, ids = module.load_documents()
        if rebuild:
            reset_collection(module.COLLECTION)
        written = upsert(module.COLLECTION, documents, ids)
        print(
            f"[{module.COLLECTION}] ingested {written} document(s); "
            f"collection now holds {count(module.COLLECTION)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
