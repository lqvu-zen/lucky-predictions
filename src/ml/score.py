"""Score logged predictions against actual results (the honest feedback loop).

Flow:
  1. `score_pending()` — for every ledger entry whose target draw now has a
     real result (and isn't scored yet), compute how the prediction did:
     hits (top-6 ∩ actual), Brier score and log-loss of the per-number
     probabilities, and the same for the trivial base-rate predictor.
     Results are appended to predictions/scored.jsonl and the ledger entry
     is marked scored.
  2. `rebuild_scorecard()` — aggregate scored.jsonl into a rolling scorecard
     (per game + model) and attach the current pending prediction. Written
     to predictions/scorecard.json for the dashboard to read.

No look-ahead: a prediction can only be scored once its draw has happened
and been crawled, and it was logged before that.
"""
from __future__ import annotations

import json
import math
from datetime import datetime

from analyze import load_draws
from config import PRED_DIR, PRODUCTS, get_product
from ml import ledger

SCORED_PATH = PRED_DIR / "scored.jsonl"
SCORECARD_PATH = PRED_DIR / "scorecard.json"


def _results_map(game: str) -> dict[str, list[int]]:
    product = get_product(game)
    return {d["date"]: d["main"] for d in load_draws(product)}


def _brier_and_logloss(probs: dict, actual: set[int], product) -> tuple:
    n = product.max_value
    sq = 0.0
    ll = 0.0
    for k in range(1, n + 1):
        p = float(probs.get(str(k), product.main_count / n))
        y = 1.0 if k in actual else 0.0
        sq += (p - y) ** 2
        pc = min(max(p, 1e-12), 1 - 1e-12)
        ll += -(y * math.log(pc) + (1 - y) * math.log(1 - pc))
    return sq / n, ll / n


def _append_scored(rows: list[dict]) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    with SCORED_PATH.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _rewrite_ledger(entries: list[dict]) -> None:
    with ledger.LEDGER_PATH.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def score_pending() -> list[dict]:
    entries = ledger.load()
    if not entries:
        return []
    results_cache: dict[str, dict] = {}
    newly = []
    for e in entries:
        if e.get("scored"):
            continue
        game = e["game"]
        results = results_cache.setdefault(game, _results_map(game))
        actual_list = results.get(e["target_date"])
        if actual_list is None:
            continue  # draw hasn't happened / not crawled yet
        product = get_product(game)
        actual = set(actual_list[: product.main_count])
        hits = len(actual.intersection(e["top6"]))
        brier, ll = _brier_and_logloss(e.get("probs", {}), actual, product)
        base = product.main_count / product.max_value
        brier_base = sum((base - (1 if k in actual else 0)) ** 2
                         for k in range(1, product.max_value + 1)) / product.max_value
        row = {
            "game": game, "model": e.get("model"), "version": e.get("version"),
            "target_date": e["target_date"], "top6": e["top6"],
            "actual": sorted(actual), "hits": hits,
            "baseline_hits": product.main_count ** 2 / product.max_value,
            "brier": round(brier, 6), "brier_base": round(brier_base, 6),
            "logloss": round(ll, 6),
            "scored_at": datetime.now().isoformat(),
        }
        newly.append(row)
        e["scored"] = True
    if newly:
        _append_scored(newly)
        _rewrite_ledger(entries)
    return newly


def _load_scored() -> list[dict]:
    if not SCORED_PATH.exists():
        return []
    with SCORED_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rebuild_scorecard() -> dict:
    scored = _load_scored()
    entries = ledger.load()
    games = {}
    for name in PRODUCTS:
        product = get_product(name)
        rows = [s for s in scored if s["game"] == name]
        models = {}
        for kind in sorted({s["model"] for s in rows}):
            mr = [s for s in rows if s["model"] == kind]
            n = len(mr)
            models[kind] = {
                "scored": n,
                "mean_hits": round(sum(s["hits"] for s in mr) / n, 3),
                "baseline_hits": round(product.main_count ** 2 / product.max_value, 3),
                "mean_brier": round(sum(s["brier"] for s in mr) / n, 6),
                "mean_brier_base": round(sum(s["brier_base"] for s in mr) / n, 6),
                "best_hits": max(s["hits"] for s in mr),
            }
        # pending (unscored) predictions for the upcoming draw
        pending = [e for e in entries if e["game"] == name and not e.get("scored")]
        next_pred = None
        if pending:
            td = max(e["target_date"] for e in pending)
            next_pred = {
                "target_date": td,
                "by_model": {e["model"]: e["top6"]
                             for e in pending if e["target_date"] == td},
            }
        games[name] = {"label": product.label, "models": models,
                       "next_prediction": next_pred, "total_scored": len(rows)}
    card = {"generated": datetime.now().isoformat(), "games": games}
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    SCORECARD_PATH.write_text(json.dumps(card, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    return card


def load_scorecard() -> dict | None:
    if not SCORECARD_PATH.exists():
        return None
    return json.loads(SCORECARD_PATH.read_text(encoding="utf-8"))
