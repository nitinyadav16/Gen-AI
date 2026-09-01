"""Merged multi-collection retriever (FR-06, FR-07, FR-08).

Three Chroma collections are queried in parallel and their hits are merged into
a single, source-labelled context block.

Adding a knowledge source (NFR-06):
    1. write `ingest_<source>.py` that upserts into a new collection
    2. append a `CollectionSpec` to `COLLECTIONS` below
Nothing else in the app needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.runnables import RunnableParallel

from config import TOP_K
from vectorstore import count, get_vectorstore


@dataclass(frozen=True)
class CollectionSpec:
    name: str            # Chroma collection name
    context_label: str   # header used inside the prompt context (FR-08)
    source_label: str    # label shown in the UI Sources panel (FR-13a)
    description: str     # shown on the app's knowledge-base status panel
    ingest_script: str


COLLECTIONS: tuple[CollectionSpec, ...] = (
    CollectionSpec("faq", "FAQ", "FAQ", "Published FAQ entries", "ingest_faq.py"),
    CollectionSpec("tickets", "TICKETS", "TICKET", "Resolved support tickets", "ingest_tickets.py"),
    CollectionSpec("guides", "GUIDES", "GUIDE", "Telecom user-guide chunks", "ingest_guide.py"),
)


@dataclass(frozen=True)
class RetrievedDoc:
    collection: str
    context_label: str
    source_label: str
    source_id: str
    title: str
    content: str

    @property
    def citation(self) -> str:
        """e.g. `TICKET TK-004`."""
        return f"{self.source_label} {self.source_id}"


class MergedRetriever:
    """Fan out one question to every registered collection at once."""

    def __init__(self, k: int = TOP_K, specs: tuple[CollectionSpec, ...] = COLLECTIONS):
        self.k = k
        self.specs = specs
        self._parallel = RunnableParallel(
            {
                spec.name: get_vectorstore(spec.name).as_retriever(search_kwargs={"k": k})
                for spec in specs
            }
        )

    def retrieve(self, question: str) -> list[RetrievedDoc]:
        hits: dict[str, list[Document]] = self._parallel.invoke(question)
        merged: list[RetrievedDoc] = []
        for spec in self.specs:
            for document in hits.get(spec.name, []):
                metadata = document.metadata or {}
                merged.append(
                    RetrievedDoc(
                        collection=spec.name,
                        context_label=spec.context_label,
                        source_label=spec.source_label,
                        source_id=str(metadata.get("source_id", "unknown")),
                        title=str(metadata.get("title", "")).strip(),
                        content=document.page_content.strip(),
                    )
                )
        return merged

    def status(self) -> list[tuple[CollectionSpec, int]]:
        """(spec, document count) for each collection - used by the UI banner."""
        return [(spec, count(spec.name)) for spec in self.specs]


def format_context(documents: list[RetrievedDoc]) -> str:
    """Render retrieved documents as a labelled context block for the prompt."""
    if not documents:
        return "(no relevant documents were retrieved)"

    blocks = []
    for index, doc in enumerate(documents, start=1):
        header = f"[{index}] SOURCE={doc.context_label} ID={doc.source_id}"
        if doc.title:
            header += f" TITLE={doc.title}"
        blocks.append(f"{header}\n{doc.content}")
    return "\n\n".join(blocks)


def knowledge_base_is_empty() -> bool:
    return all(count(spec.name) == 0 for spec in COLLECTIONS)
