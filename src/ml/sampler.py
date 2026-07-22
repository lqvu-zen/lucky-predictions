"""Empirical position sampler — a stochastic position-based model.

For each ordered position we build its empirical distribution over numbers
(how often each value has landed there), then sample a ticket sequentially:
p1 from its distribution, p2 from its distribution restricted to values above
p1 (leaving room for the remaining positions), and so on. This yields varied
but position-realistic tickets — unlike the deterministic models that always
return the average spread.

Pure Python (no ML extras). It's seeded by the next draw date so the pick
stays locked for a given draw. Still can't beat the odds — the distributions
are the fixed order-statistic law.
"""
from __future__ import annotations

import random

from analyze import load_draws
from config import Product, get_product


def _position_counts(product: Product, draws) -> list[dict]:
    """counts[pos][number] = times that number landed at that position."""
    k = product.main_count
    counts = [dict() for _ in range(k)]
    for d in draws:
        for pos, val in enumerate(sorted(d["main"])):
            counts[pos][val] = counts[pos].get(val, 0) + 1
    return counts


def _sample_ticket(counts, product: Product, rng: random.Random) -> list[int]:
    k, N = product.main_count, product.max_value
    ticket, prev = [], 0
    for pos in range(k):
        # must leave room for the remaining (k-1-pos) larger numbers
        hi = N - (k - 1 - pos)
        dist = {v: c for v, c in counts[pos].items() if prev < v <= hi}
        if dist:
            vals, weights = list(dist.keys()), list(dist.values())
            pick = rng.choices(vals, weights=weights, k=1)[0]
        else:
            pick = prev + 1  # fallback keeps it valid
        pick = min(max(pick, prev + 1), hi)
        ticket.append(pick)
        prev = pick
    return ticket


def predict_next(product_name: str) -> dict:
    product = get_product(product_name)
    draws = load_draws(product)
    target = product.next_draw_date()
    rng = random.Random(int(target.strftime("%Y%m%d")))
    counts = _position_counts(product, draws)
    return {"product": product.label, "model": "sampler",
            "target_date": target.isoformat(),
            "ticket": _sample_ticket(counts, product, rng)}


def _bootstrap_ci(samples, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(samples)
    means = sorted(sum(rng.choice(samples) for _ in range(n)) / n
                   for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def backtest(product_name: str, test_draws: int = 120, min_history: int = 50) -> dict:
    product = get_product(product_name)
    draws = load_draws(product)
    k, n = product.main_count, product.max_value
    start = max(min_history, len(draws) - test_draws)
    hits = []
    for t in range(start, len(draws)):
        counts = _position_counts(product, draws[:t])
        # seed by the draw's own date for reproducibility
        rng = random.Random(int(draws[t]["date"].replace("-", "")))
        ticket = _sample_ticket(counts, product, rng)
        actual = set(draws[t]["main"][:k])
        hits.append(len(actual.intersection(ticket)))
    if not hits:
        return {"product": product.label, "tested": 0}
    lo, hi = _bootstrap_ci(hits)
    base = k * k / n
    return {"product": product.label, "model": "sampler", "tested": len(hits),
            "mean_hits": sum(hits) / len(hits), "hits_lo": lo, "hits_hi": hi,
            "baseline_hits": base, "beats_baseline": lo > base}


def format_backtest(r: dict) -> str:
    if not r.get("tested"):
        return f"{r['product']}: not enough data."
    verdict = ("⚑ CI above baseline — investigate!" if r["beats_baseline"]
               else "within noise of random (CI spans the baseline)")
    return "\n".join([
        f"{r['product']} — empirical sampler backtest ({r['model']}) "
        f"over {r['tested']} draws",
        "",
        f"  mean hits       {r['mean_hits']:.3f}  "
        f"[95% CI {r['hits_lo']:.3f}, {r['hits_hi']:.3f}]",
        f"  random baseline {r['baseline_hits']:.3f}   → {verdict}",
        "",
        "  Samples each position from its real distribution — varied tickets, "
        "same non-edge.",
    ])


if __name__ == "__main__":
    import sys
    print(format_backtest(backtest(sys.argv[1] if len(sys.argv) > 1 else "power_655")))
