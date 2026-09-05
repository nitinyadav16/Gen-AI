"""Shared Groq LLM factory (FR-04).

Extracted so intent detection and the refund agent reuse the exact same model
configuration as the RAG chain, without importing the retrieval stack (and its
Chroma / sentence-transformers start-up cost) to get at it.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_groq import ChatGroq

from config import GROQ_MODEL, REASONING_EFFORT, TEMPERATURE, require_api_key


def build_llm() -> ChatGroq:
    """Groq-hosted Qwen, deterministic and with reasoning disabled (FR-12, FR-13)."""
    api_key = require_api_key()
    kwargs = dict(model=GROQ_MODEL, temperature=TEMPERATURE, api_key=api_key)
    try:
        return ChatGroq(reasoning_effort=REASONING_EFFORT, **kwargs)
    except TypeError:
        # Older langchain-groq builds pass provider-specific fields through model_kwargs.
        return ChatGroq(model_kwargs={"reasoning_effort": REASONING_EFFORT}, **kwargs)


@lru_cache(maxsize=1)
def shared_llm() -> ChatGroq:
    """One reusable client for the short classification / extraction calls."""
    return build_llm()
