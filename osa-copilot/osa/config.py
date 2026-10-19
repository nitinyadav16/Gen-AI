"""Central configuration for the OSA Root-Cause Copilot.

Every threshold the PRD calls "configurable" lives here, so product can tune
the flagging rules per retailer without touching the pipeline (section 16:
"keep it configurable per retailer"). Secrets come from `.env` only (NFR
Security: API keys held server-side only).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

# --- Paths -----------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
DECISION_LOG = LOG_DIR / "decisions.jsonl"      # FR-9 audit log
DIAGNOSIS_LOG = LOG_DIR / "diagnoses.jsonl"     # every diagnosis shown, with versions
REQUEST_LOG = LOG_DIR / "requests.jsonl"        # Q&A requests, incl. declined ones
DIAGNOSIS_CACHE = LOG_DIR / "diagnosis_cache.json"
EVAL_DIR = BASE_DIR / "eval_results"

# --- Model strategy (section 5) ---------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
DIAGNOSIS_MODEL = os.getenv("DIAGNOSIS_MODEL", "openai/gpt-oss-120b").strip()
QA_MODEL = os.getenv("QA_MODEL", "openai/gpt-oss-20b").strip()       # smaller model for Q&A (section 5)
TEMPERATURE = 0
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low").strip()
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

# Bumped whenever a prompt changes, and logged with every diagnosis (FR-9).
DIAGNOSIS_PROMPT_VERSION = "diag-v1.0"
QA_PROMPT_VERSION = "qa-v1.0"

CURRENCY = "$"

# Standard operating procedure for managers, linked from the app sidebar.
SOP_URL = os.getenv("SOP_URL", "https://claude.ai/code/artifact/c9c380ca-495b-4f56-8f9a-c5bc1ee7f29e").strip()


# --- Detection rules (FR-3, FR-2) -------------------------------------------
@dataclass(frozen=True)
class Rules:
    osa_target: float = 92.0          # flag when latest OSA is below this (%)
    drop_points: float = 8.0          # ...or when it fell more than this vs baseline
    baseline_weeks: int = 4           # the 4-week baseline
    min_history_weeks: int = 6        # never flag an item with less history than this
    stale_after_days: int = 7         # "more than one week stale" = data gap


DEFAULT_RULES = Rules()

# Per-retailer overrides. Empty by default; add e.g. {"Retailer B": Rules(osa_target=90)}.
RETAILER_RULES: dict[str, Rules] = {}


def rules_for(retailer: str) -> Rules:
    return RETAILER_RULES.get(retailer, DEFAULT_RULES)


def require_api_key() -> str:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://console.groq.com/keys"
        )
    return GROQ_API_KEY
