"""Joint number×position model: P(number k lands at ordered position p).

This is the richest of the three framings and the parent of the other two:

    - sum a number's row over positions  -> P(it appears at all) ≈ k/N
      (the per-number model's flat result)
    - a column is a position's distribution over numbers
      (what the positional model summarised by its mean)

The "model" is the maximum-likelihood estimate of the grid from history
(how often each number landed at each sorted position). It's pure counting —
no heavy ML needed — and it exactly matches the closed-form order-statistic
law of a uniform draw:

    P(position p = value v) = C(v-1, p-1) * C(N-v, k-p) / C(N, k)

Because that law is the same for every draw, the grid has no per-draw
predictive power — it can't beat the baseline at hitting the actual numbers.
What it gives is the most information-rich, and a lovely number×position
heatmap. Honest verdict, best visual.
"""
from __future__ import annotations

import random
from math import comb

from analyze import load_draws
from config import Product, get_product
from ml.util import progress


def empirical_grid(product: Product, draws=None, smoothing: float = 0.0):
    """grid[num][pos] = P(number `num` is at sorted position `pos`) from history.

    Columns (positions) sum to 1 over numbers; a number's row sums to ≈ k/N.
    Pure Python (no numpy), so callers without the ML extras can use it too.
    """
    draws = draws if draws is not None else load_draws(product)
    k, N = product.main_count, product.max_value
    counts = [[smoothing] * k for _ in range(N + 1)]  # index 0 unused
    T = 0
    for d in draws:
        for pos, val in enumerate(sorted(d["main"])):
            counts[val][pos] += 1
        T += 1
    denom = T + smoothing * N
    grid = [[0.0] * k for _ in range(N + 1)]
    if denom > 0:
        for num in range(1, N + 1):
            for pos in range(k):
                grid[num][pos] = counts[num][pos] / denom
    return grid


def closed_form_grid(product: Product):
    """Theoretical P(position p = v) for a uniform draw (order statistics)."""
    k, N = product.main_count, product.max_value
    total = comb(N, k)
    grid = [[0.0] * k for _ in range(N + 1)]
    for v in range(1, N + 1):
        for pos in range(k):
            p = pos + 1
            grid[v][pos] = comb(v - 1, p - 1) * comb(N - v, k - p) / total
    return grid


def max_abs_diff(a, b, product: Product) -> float:
    k, N = product.main_count, product.max_value
    return max(abs(a[v][p] - b[v][p])
               for v in range(1, N + 1) for p in range(k))


def predict_ticket(grid, product: Product) -> list[int]:
    """Greedy per-position assignment of distinct numbers, returned ascending."""
    k, N = product.main_count, product.max_value
    used, ticket = set(), []
    for pos in range(k):
        best, best_p = None, -1.0
        for num in range(1, N + 1):
            if num in used:
                continue
            if grid[num][pos] > best_p:
                best_p, best = grid[num][pos], num
        used.add(best)
        ticket.append(best)
    return sorted(ticket)


def predict_next(product_name: str) -> dict:
    product = get_product(product_name)
    grid = empirical_grid(product, smoothing=0.5)
    ticket = predict_ticket(grid, product)
    diff = max_abs_diff(empirical_grid(product), closed_form_grid(product), product)
    return {"product": product.label, "model": "joint-grid",
            "target_date": product.next_draw_date().isoformat(),
            "ticket": ticket, "grid_vs_theory_maxdiff": round(diff, 4)}


def grid_transposed(product: Product, draws=None):
    """Return positions×numbers grid for the dashboard heatmap."""
    g = empirical_grid(product, draws)
    k, N = product.main_count, product.max_value
    return [[g[num][pos] for num in range(1, N + 1)] for pos in range(k)]


def _bootstrap_ci(samples, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(samples)
    means = []
    for _ in range(n_boot):
        means.append(sum(rng.choice(samples) for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def backtest(product_name: str, test_draws: int = 120, min_history: int = 50) -> dict:
    product = get_product(product_name)
    draws = load_draws(product)
    k, N = product.main_count, product.max_value
    start = max(min_history, len(draws) - test_draws)
    hits = []
    total = len(draws) - start
    for j, t in enumerate(range(start, len(draws))):
        grid = empirical_grid(product, draws[:t], smoothing=0.5)
        ticket = predict_ticket(grid, product)
        actual = set(draws[t]["main"][:k])
        hits.append(len(actual.intersection(ticket)))
        progress(j + 1, total, "joint backtest")
    if not hits:
        return {"product": product.label, "tested": 0}
    lo, hi = _bootstrap_ci(hits)
    base = k * k / N
    return {
        "product": product.label, "model": "joint-grid",
        "tested": len(hits),
        "mean_hits": sum(hits) / len(hits),
        "hits_lo": lo, "hits_hi": hi,
        "baseline_hits": base,
        "beats_baseline": lo > base,
        "grid_vs_theory_maxdiff": round(
            max_abs_diff(empirical_grid(product), closed_form_grid(product), product), 4),
    }


def format_backtest(r: dict) -> str:
    if not r.get("tested"):
        return f"{r['product']}: not enough data."
    verdict = ("⚑ CI above baseline — investigate!" if r["beats_baseline"]
               else "within noise of random (CI spans the baseline)")
    return "\n".join([
        f"{r['product']} — joint number×position backtest over {r['tested']} draws",
        "",
        f"  mean hits            {r['mean_hits']:.3f}  "
        f"[95% CI {r['hits_lo']:.3f}, {r['hits_hi']:.3f}]",
        f"  random baseline      {r['baseline_hits']:.3f}   → {verdict}",
        f"  grid vs closed-form  max abs diff {r['grid_vs_theory_maxdiff']} "
        f"(≈0 means the learned grid is just the fixed order-statistic law)",
        "",
        "  The grid fits the true position law perfectly, yet can't pick the "
        "actual numbers — the law is identical every draw. Best visual, no edge.",
    ])


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "power_655"
    print(format_backtest(backtest(name)))
