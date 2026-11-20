"""Append finished matches to ``data/match_history.jsonl`` (one JSON object per line)."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from config import MATCH_HISTORY_FILE


def build_record(match, source: str, tournament_id: str | None = None) -> dict[str, Any]:
    summary = match.summary()
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,                    # "arena", "human", "tournament"
        "tournament_id": tournament_id,
        "summary": summary,
        "specs": [s.to_dict() for s in match.specs],
        "decisions": match.decision_records(),
    }


def append_record(record: dict[str, Any], path: str = MATCH_HISTORY_FILE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def log_match(match, source: str, tournament_id: str | None = None, path: str = MATCH_HISTORY_FILE) -> dict[str, Any]:
    rec = build_record(match, source, tournament_id)
    append_record(rec, path)
    return rec


def load_history(path: str = MATCH_HISTORY_FILE) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue  # skip a corrupted line rather than losing the whole history
    except OSError:
        pass
    return out


def clear_history(path: str = MATCH_HISTORY_FILE) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
