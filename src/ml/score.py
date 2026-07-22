"""Score logged predictions against actual results (the honest feedback loop).

Both surviving models — positional (ordered) and joint number x position —
produce a ticket (6 numbers), so scoring is by hits: how many of the ticket's
numbers actually came up. Predictions are logged BEFORE the draw, then scored
once the real result is crawled.

Flow:
  1. score_pending()  — for every ledger entry whose draw now has a result,
     compute hits (ticket ∩ actual), append to predictions/scored.jsonl and
     mark the ledger entry scored.
  2. rebuild_scorecard() — aggregate per game + model into a rolling scorecard
     (mean hits vs the random baseline) for the dashboard.
"""
from __future__ import annotations

import json
from datetime import datetime

from analyze import load_draws
from config import PRED_DIR, PRODUCTS, get_product
from ml import ledger

SCORED_PATH = PRED_DIR / "scored.jsonl"
SCORECARD_PATH = PRED_DIR / "scorecard.json"


def _ticket(entry: dict) -> list[int]:
    # tolerate old per-number entries that used "top6"
    return entry.get("ticket") or entry.get("top6") or []


def _results_map(game: str) -> dict[str, list[int]]:
    product = get_product(game)
    return {d["date"]: d["main"] for d in load_draws(product)}


def score_pending() -> list[dict]:
    entries = ledger.load()
    if not entries:
        return []
    cache: dict[str, dict] = {}
    newly = []
    for e in entries:
        if e.get("scored"):
            continue
        game = e["game"]
        results = cache.setdefault(game, _results_map(game))
        actual_list = results.get(e["target_date"])
        if actual_list is None:
            continue  # draw not available yet
        product = get_product(game)
        k = product.main_count
        actual_sorted = sorted(actual_list[:k])
        actual = set(actual_sorted)
        ticket = _ticket(e)
        ticket_sorted = sorted(ticket)
        hits = len(actual.intersection(ticket))
        # position accuracy: correct number at the correct sorted position
        pos_hits = sum(1 for i in range(k)
                       if i < len(ticket_sorted) and ticket_sorted[i] == actual_sorted[i])
        newly.append({
            "game": game, "model": e.get("model"), "version": e.get("version"),
            "target_date": e["target_date"], "ticket": ticket,
            "actual": actual_sorted, "hits": hits, "pos_hits": pos_hits,
            "baseline_hits": k ** 2 / product.max_value,
            "scored_at": datetime.now().isoformat(),
        })
        e["scored"] = True
    if newly:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        with SCORED_PATH.open("a", encoding="utf-8") as f:
            for r in newly:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with ledger.LEDGER_PATH.open("w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
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
            k = product.main_count
            mean_pos = sum(s.get("pos_hits", 0) for s in mr) / n
            models[kind] = {
                "scored": n,
                "mean_hits": round(sum(s["hits"] for s in mr) / n, 3),
                "mean_pos_hits": round(mean_pos, 3),
                # k/n score: fraction of the 6 that were the right number at
                # the right sorted position, averaged over scored draws
                "mean_pos_score": round(mean_pos / k, 4),
                "baseline_hits": round(k ** 2 / product.max_value, 3),
                "best_hits": max(s["hits"] for s in mr),
                "best_pos_hits": max(s.get("pos_hits", 0) for s in mr),
            }
        pending = [e for e in entries if e["game"] == name and not e.get("scored")]
        next_pred = None
        if pending:
            td = max(e["target_date"] for e in pending)
            by_model = {e["model"]: _ticket(e)
                        for e in pending if e["target_date"] == td}
            # consensus: how many predictors picked each number (exclude the
            # saved "consensus" prediction so it doesn't vote for itself)
            voters = {m: t for m, t in by_model.items() if m != "consensus"}
            votes: dict[int, int] = {}
            for ticket in voters.values():
                for num in ticket:
                    votes[num] = votes.get(num, 0) + 1
            consensus = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))
            n_models = len(voters)
            next_pred = {
                "target_date": td, "by_model": by_model, "n_models": n_models,
                "consensus": [[int(num), int(c)] for num, c in consensus],
                # the top main_count numbers by vote as the "consensus ticket"
                "consensus_ticket": sorted(int(num) for num, _ in consensus[:product.main_count]),
            }
        # theoretical position baseline: best possible mean pos-hits if you
        # always guessed each position's most likely value (the mode)
        try:
            from ml.joint import closed_form_grid
            g = closed_form_grid(product)
            pos_base = sum(max(g[v][pos] for v in range(1, product.max_value + 1))
                           for pos in range(product.main_count))
        except Exception:
            pos_base = 0.0
        # recent history: per past draw, the actual result + every predictor's
        # ticket and score, newest first (last few draws)
        by_date: dict[str, dict] = {}
        for s in rows:
            h = by_date.setdefault(s["target_date"],
                                   {"date": s["target_date"], "actual": s["actual"], "preds": []})
            h["preds"].append({"model": s["model"], "ticket": s["ticket"],
                               "pos": s.get("pos_hits", 0), "hits": s["hits"]})
        history = sorted(by_date.values(), key=lambda d: d["date"], reverse=True)[:20]
        for h in history:
            h["preds"].sort(key=lambda x: (-x["pos"], -x["hits"]))

        games[name] = {"label": product.label, "models": models,
                       "next_prediction": next_pred, "total_scored": len(rows),
                       "pos_baseline": round(pos_base, 3),
                       "pos_baseline_score": round(pos_base / product.main_count, 4),
                       "history": history}
    card = {"generated": datetime.now().isoformat(), "games": games}
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    SCORECARD_PATH.write_text(json.dumps(card, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    return card


def load_scorecard() -> dict | None:
    if not SCORECARD_PATH.exists():
        return None
    return json.loads(SCORECARD_PATH.read_text(encoding="utf-8"))
