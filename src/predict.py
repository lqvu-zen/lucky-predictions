"""Ticket suggestion strategies for Vietlott Power 6/55 and 6/45.

⚠️  Reality check: a lottery draw is a uniform random selection. Every
combination is equally likely on every draw, so NONE of these strategies
can improve your odds. They are transparent, reproducible ways to pick
numbers "for fun" using the historical data — nothing more.

Strategies:
  - random:        uniform random pick (the honest baseline)
  - hot:           weighted toward numbers drawn most in a recent window
  - cold:          weighted toward numbers drawn least in a recent window
  - overdue:       weighted toward numbers with the longest absence
  - balanced:      blend of frequency + recency + overdue signals
"""
from __future__ import annotations

import random

from analyze import (days_since_last, frequency, load_draws,
                     recent_frequency)
from config import Product, get_product


def _weighted_sample(weights: dict[int, float], k: int, rng: random.Random) -> list[int]:
    """Sample k distinct numbers without replacement, proportional to weight."""
    pool = list(weights.keys())
    w = [max(weights[n], 1e-9) for n in pool]
    chosen: list[int] = []
    for _ in range(k):
        total = sum(w)
        r = rng.uniform(0, total)
        upto = 0.0
        for i, weight in enumerate(w):
            upto += weight
            if upto >= r:
                chosen.append(pool[i])
                w[i] = 0.0  # remove (no replacement)
                break
    return sorted(chosen)


def _normalize(d: dict[int, float]) -> dict[int, float]:
    lo = min(d.values())
    hi = max(d.values())
    if hi == lo:
        return {k: 1.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def build_weights(draws: list[dict], product: Product, strategy: str,
                  window: int = 60, ref=None) -> dict[int, float]:
    # `ref` = the date to measure "days since last" against. Defaults to
    # today; the backtest passes the historical draw date so overdue/
    # balanced weights reflect what was known at that point in time.
    nums = range(product.min_value, product.max_value + 1)
    freq = recent_frequency(draws, product, window)
    dsl = days_since_last(draws, product, ref)

    if strategy == "random":
        return {n: 1.0 for n in nums}

    if strategy == "hot":
        return {n: float(freq[n]) + 0.5 for n in nums}

    if strategy == "cold":
        mx = max(freq.values()) if freq else 0
        return {n: float(mx - freq[n]) + 0.5 for n in nums}

    if strategy == "overdue":
        return {n: float((dsl[n]["days_since"] or 0)) + 0.5 for n in nums}

    if strategy == "balanced":
        f = _normalize({n: float(freq[n]) for n in nums})
        o = _normalize({n: float(dsl[n]["days_since"] or 0) for n in nums})
        # reward both moderately-frequent and overdue numbers
        return {n: 0.6 * f[n] + 0.4 * o[n] + 0.2 for n in nums}

    raise ValueError(f"Unknown strategy '{strategy}'")


STRATEGIES = ["random", "hot", "cold", "overdue", "balanced"]


def suggest(product_name: str, strategy: str = "balanced",
            tickets: int = 3, window: int = 60, seed: int | None = None) -> dict:
    """Generate `tickets` suggested lines for a product using a strategy."""
    product = get_product(product_name)
    draws = load_draws(product)
    rng = random.Random(seed)
    weights = build_weights(draws, product, strategy, window)
    lines = [
        _weighted_sample(weights, product.main_count, rng)
        for _ in range(tickets)
    ]
    return {
        "product": product.label,
        "strategy": strategy,
        "window": window,
        "tickets": lines,
    }


def suggest_all(product_name: str, tickets: int = 1,
                seed: int | None = None) -> dict:
    """One suggested line per strategy — handy for a daily report.

    Each strategy gets an offset seed so their lines are decorrelated
    (otherwise a shared fixed seed makes every strategy look identical).
    """
    out = {}
    for i, strat in enumerate(STRATEGIES):
        s = None if seed is None else seed + i
        out[strat] = suggest(product_name, strat, tickets, seed=s)["tickets"]
    return out


if __name__ == "__main__":
    import json
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "power_655"
    print(json.dumps(suggest_all(name, tickets=2), ensure_ascii=False, indent=2))
