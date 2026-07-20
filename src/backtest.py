"""Backtest the ticket strategies against real Vietlott history.

The idea: walk forward through the recorded draws. At each draw, use ONLY
the draws that came before it to build each strategy's weights, generate
some tickets, and count how many of the 6 main numbers each ticket got
right. Average those hit counts per strategy and compare them to the
mathematical baseline for uniformly random guessing.

The point of this exercise is honesty: because draws are independent and
uniform, every strategy — hot, cold, overdue, balanced — lands right on
top of the random baseline. Past numbers carry no information about future
ones. The backtest lets you *see* that instead of taking it on faith.

Baseline (expected matches for a random ticket) follows the hypergeometric
mean: pick 6 of N, actual is 6 of N  =>  E[hits] = 6 * 6 / N.
  - Power 6/55: 36/55 ≈ 0.655 matches per ticket
  - Power 6/45: 36/45 = 0.800 matches per ticket
"""
from __future__ import annotations

import random
from datetime import datetime

from analyze import load_draws
from config import get_product
from predict import STRATEGIES, _weighted_sample, build_weights


def _ref_date(draw: dict):
    return datetime.fromisoformat(draw["date"]).date()


def run(product_name: str, warmup: int = 200, tickets: int = 10,
        window: int = 60, seed: int = 0) -> dict:
    """Walk-forward backtest. Returns per-strategy stats.

    warmup  : draws to skip at the start (need history before predicting)
    tickets : tickets generated per strategy per test draw (averaged)
    window  : recency window passed to the strategies
    """
    product = get_product(product_name)
    draws = load_draws(product)
    n = len(draws)
    if n <= warmup + 1:
        return {"product": product.label, "tested": 0}

    rng = random.Random(seed)
    k = product.main_count
    max_hits = k + 1  # 0..6 inclusive

    stats = {s: {"hit_sum": 0, "count": 0, "dist": [0] * max_hits,
                 "big_wins": 0} for s in STRATEGIES}

    for i in range(warmup, n):
        prior = draws[:i]
        actual = set(draws[i]["main"])
        ref = _ref_date(draws[i])
        for s in STRATEGIES:
            weights = build_weights(prior, product, s, window, ref=ref)
            for _ in range(tickets):
                line = _weighted_sample(weights, k, rng)
                hits = len(actual.intersection(line))
                st = stats[s]
                st["hit_sum"] += hits
                st["count"] += 1
                st["dist"][hits] += 1
                if hits >= 4:
                    st["big_wins"] += 1

    baseline = k * k / product.max_value  # hypergeometric mean
    results = {}
    for s in STRATEGIES:
        st = stats[s]
        mean = st["hit_sum"] / st["count"] if st["count"] else 0.0
        results[s] = {
            "mean_hits": mean,
            "vs_baseline": mean - baseline,
            "big_wins": st["big_wins"],     # tickets with >=4 matches
            "dist": st["dist"],
            "tickets": st["count"],
        }
    return {
        "product": product.label,
        "tested_draws": n - warmup,
        "tickets_per_strategy": (n - warmup) * tickets,
        "baseline": baseline,
        "results": results,
    }


def format_report(bt: dict) -> str:
    if not bt.get("tested_draws"):
        return f"{bt['product']}: not enough data to backtest."
    lines = [
        f"{bt['product']} — backtest over {bt['tested_draws']} draws, "
        f"{bt['tickets_per_strategy']:,} tickets per strategy",
        f"Random baseline (expected matches per ticket): {bt['baseline']:.3f}",
        "",
        f"{'strategy':10s} {'mean hits':>10s} {'vs random':>10s} {'4+ hits':>9s}",
        "-" * 42,
    ]
    for s, r in sorted(bt["results"].items(),
                       key=lambda kv: -kv[1]["mean_hits"]):
        lines.append(f"{s:10s} {r['mean_hits']:>10.3f} "
                     f"{r['vs_baseline']:>+10.3f} {r['big_wins']:>9,d}")
    lines += [
        "",
        "Read: every strategy sits within noise of the random baseline. "
        "None has an edge — as expected for a uniform random draw.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "power_655"
    print(format_report(run(name)))
