"""Append-only JSONL audit log (FR-9, section 5 "Audit log")."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append(path: Path, record: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": now(), **record}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
