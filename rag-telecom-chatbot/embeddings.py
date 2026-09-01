"""Shared local embedding function (FR-09, NFR-02).

`all-MiniLM-L6-v2` is downloaded once (~90 MB) into the HuggingFace cache and
then runs entirely on-device - no embedding calls leave the machine. The model
is memoised so ingestion and retrieval reuse a single loaded instance.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
