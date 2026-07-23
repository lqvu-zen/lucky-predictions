"""Bankroll / expected-value simulator — the honest capstone.

"What if you actually bought a ticket every single draw?" We replay the whole
history: for each draw, each player buys one line (cost from config), we award
the prize for however many numbers matched, and track the running profit/loss.

Because a lottery is a negative-expectation game, every line trends steadily
down — you spend far more than you ever win back. This turns the abstract
"house edge" into a concrete losing curve.

Players are the five for-fun strategies (leakage-safe, pure-Python). Since any
ticket has the same expected return, they all converge to the same decline —
that's the point. Ticket cost and prize tiers live in `config.py`.
"""
from __future__ import annotations

import random

import predict
from analyze import load_draws
from config import get_product


def _fmt_vnd(x: int) -> str:
    return f"{x:,.0f}"


def simulate(product_name: str, warmup: int = 50, chart_points: int = 180) -> dict:
    product = get_product(product_name)
    draws = load_draws(product)
    tiers = dict(product.prize_tiers)
    cost = product.ticket_cost
    k = product.main_count
    players = predict.STRATEGIES
    if len(draws) <= warmup + 1:
        return {"product": product.label, "draws": 0}

    rng = {s: random.Random(1000 + i) for i, s in enumerate(players)}
    cum = {s: 0 for s in players}
    series = {s: [] for s in players}
    labels = []
    n = 0
    for t in range(warmup, len(draws)):
        hist = draws[:t]
        actual = set(draws[t]["main"][:k])
        for s in players:
            w = predict.build_weights(hist, product, s)
            ticket = predict._weighted_sample(w, k, rng[s])
            m = len(actual.intersection(ticket))
            cum[s] += tiers.get(m, 0) - cost
        labels.append(draws[t]["date"])
        for s in players:
            series[s].append(cum[s])
        n += 1

    # downsample the curves for the chart (keep every step-th point + last)
    step = max(1, n // chart_points)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    chart = {"labels": [labels[i] for i in idx],
             "series": {s: [series[s][i] for i in idx] for s in players}}

    totals = {}
    for s in players:
        spent = n * cost
        net = cum[s]
        won = net + spent
        totals[s] = {"spent": spent, "won": won, "net": net,
                     "return_pct": round(100.0 * won / spent - 100.0, 1)}
    return {"product": product.label, "draws": n, "cost": cost,
            "chart": chart, "totals": totals}


def format_report(b: dict) -> str:
    if not b.get("draws"):
        return f"{b['product']}: not enough data."
    lines = [
        f"{b['product']} — bankroll over {b['draws']} draws "
        f"(1 line/draw @ {_fmt_vnd(b['cost'])} VND)",
        "",
        f"{'strategy':10s} {'spent':>14s} {'won':>14s} {'net':>15s} {'return':>8s}",
        "-" * 65,
    ]
    for s, v in b["totals"].items():
        lines.append(f"{s:10s} {_fmt_vnd(v['spent']):>14s} {_fmt_vnd(v['won']):>14s} "
                     f"{_fmt_vnd(v['net']):>15s} {v['return_pct']:>7.1f}%")
    lines += [
        "",
        "  Every strategy bleeds money — that's the house edge made concrete. "
        "No ticket, model, or system changes it.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    print(format_report(simulate(sys.argv[1] if len(sys.argv) > 1 else "power_655")))
