"""Central configuration for the RAG telecom customer-care chatbot.

All paths are resolved relative to this file so the app can be started from
any working directory. Secrets are read from `.env` only (NFR-03).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


def _resolve_dir(name: str) -> Path:
    """Return BASE_DIR/name, tolerating a differently-cased folder on disk."""
    exact = BASE_DIR / name
    if exact.exists():
        return exact
    for candidate in BASE_DIR.iterdir():
        if candidate.is_dir() and candidate.name.lower() == name.lower():
            return candidate
    return exact


# --- Paths -----------------------------------------------------------------
DATA_DIR = _resolve_dir("data")
CHROMA_DIR = BASE_DIR / "chroma_store"
LOG_DIR = BASE_DIR / "logs"
INTERACTION_LOG = LOG_DIR / "interactions.jsonl"

FAQ_CSV = DATA_DIR / "faq.csv"
TICKETS_DB = DATA_DIR / "tickets.db"
GUIDE_PDF = DATA_DIR / "telecom_guide.pdf"

# --- Retrieval -------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local, offline (FR-09)
TOP_K = 3           # documents fetched per collection (FR-07)
CHUNK_SIZE = 600    # PDF guide chunking (FR-16)
CHUNK_OVERLAP = 100

# --- Generation ------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b").strip()  # FR-13
TEMPERATURE = 0                       # FR-12
REASONING_EFFORT = "none"             # FR-13

ESCALATION_HINT = "call 611 or use the MyTelecom app"

# --- Conversation memory ---------------------------------------------------
# One exchange = one user message plus one assistant reply. The most recent
# exchanges are replayed as context on the next turn (FR-46, FR-50).
HISTORY_EXCHANGES = 4

# --- Refund capability -----------------------------------------------------
CURRENCY = "₹"              # rupee sign, used in every customer-facing amount
REFUND_WINDOW_DAYS = 7           # FR-25
REFUND_APPROVAL_LIMIT = 499      # FR-36 / FR-37, enforced in code, never by the LLM (FR-36a)


def require_api_key() -> str:
    """Return the Groq API key, or raise a clear, actionable error."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://console.groq.com/keys"
        )
    return GROQ_API_KEY
