"""ChromaDB access helpers.

Every collection lives in the same on-disk store (`chroma_store/`) so the app
never re-ingests at startup (NFR-05).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import CHROMA_DIR
from embeddings import get_embeddings


@lru_cache(maxsize=1)
def _client() -> chromadb.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


@lru_cache(maxsize=None)
def get_vectorstore(collection: str) -> Chroma:
    """Open (or create) a collection. Cached - loading the embedder is expensive."""
    return Chroma(
        client=_client(),
        collection_name=collection,
        embedding_function=get_embeddings(),
    )


def count(collection: str) -> int:
    """Number of documents in a collection (0 if it does not exist yet).

    Uses the raw Chroma client so a simple status check never has to load the
    embedding model.
    """
    try:
        return _client().get_collection(collection).count()
    except Exception:
        return 0


def upsert(collection: str, documents: Sequence[Document], ids: Sequence[str]) -> int:
    """Write documents under stable ids.

    Chroma upserts on a repeated id, which is what makes the ingest scripts
    idempotent: re-running one refreshes existing rows instead of duplicating
    them (FR-17).
    """
    if not documents:
        return 0
    get_vectorstore(collection).add_documents(documents=list(documents), ids=list(ids))
    return len(documents)


def reset_collection(collection: str) -> None:
    """Drop a collection so it can be rebuilt from scratch."""
    try:
        _client().delete_collection(collection)
    except Exception:
        pass
    # Any cached handle now points at a deleted collection.
    get_vectorstore.cache_clear()
