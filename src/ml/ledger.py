"""Prediction ledger: record every prediction BEFORE its draw happens.

Predictions are appended to predictions/ledger.jsonl. Locking a prediction
in before the result exists is what makes later scoring honest (no
hindsight). A separate scoring step (Phase 4) will match ledger entries to
actual draws and compute rolling metrics.

Ledger entry:
    {
      "game": "power_655",
      "target_date": "2026-07-21",   # the draw this predicts
      "model": "logreg",
      "version": "2026-07-20T21:30:00",
      "top6": [3, 12, 22, 34, 41, 55],
      "probs": {"1": 0.108, ...},     # per-number P(appear)
      "created_at": "2026-07-20T21:30:05",
      "scored": false
    }
"""
from __future__ import annotations

import json
from datetime import datetime

from config import PRED_DIR

LEDGER_PATH = PRED_DIR / "ledger.jsonl"


def append(entry: dict) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    entry.setdefault("created_at", datetime.now().isoformat())
    entry.setdefault("scored", False)
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    with LEDGER_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find(game: str, target_date: str) -> list[dict]:
    return [e for e in load()
            if e.get("game") == game and e.get("target_date") == target_date]


def pending_keys() -> set:
    """(game, model, target_date) tuples already predicted but not yet scored."""
    return {(e["game"], e.get("model"), e["target_date"])
            for e in load() if not e.get("scored")}
