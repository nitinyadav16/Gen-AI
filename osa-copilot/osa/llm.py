"""Hosted LLM access (section 5). One place to swap model or provider.

`JsonLLM` is the narrow interface the diagnosis step needs; the Q&A agent
uses a LangChain chat model directly because it needs tool calling. Tests
pass fakes for both, so the pipeline runs offline.
"""

from __future__ import annotations

from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from osa import config


class JsonLLM(Protocol):
    model: str

    def complete_json(self, system: str, user: str) -> str: ...


def chat_model(model: str):
    from langchain_groq import ChatGroq

    return ChatGroq(model=model, temperature=config.TEMPERATURE, api_key=config.require_api_key(),
                    timeout=config.LLM_TIMEOUT_S, max_retries=config.LLM_MAX_RETRIES,
                    reasoning_effort=config.REASONING_EFFORT)


class GroqJsonLLM:
    """JSON-mode completion: the response is always a single JSON object."""

    def __init__(self, model: str | None = None):
        self.model = model or config.DIAGNOSIS_MODEL
        self._llm = chat_model(self.model).bind(response_format={"type": "json_object"})
        self.last_usage: dict = {}

    def complete_json(self, system: str, user: str) -> str:
        msg = self._llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        self.last_usage = dict(msg.usage_metadata or {})
        return msg.content
