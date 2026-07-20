"""Statistical analysis of Vietlott draw history.

All functions operate on the 6 MAIN numbers of each draw (the bonus
number, stored as the 7th element, is ignored for these stats).

Important: lottery draws are independent and random. These statistics
describe the *past*; they carry no predictive power over future draws.
They are provided for exploration and fun only.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import date, datetime
from pathlib import Path

from config import Product, get_product


def load_draws(product: Product) -> list[dict]:
    """Load draws sorted oldest -> newest. Adds 'main' (first 6 numbers)."""
    path = product.raw_path
    if not path.exists():
        return []
    draws = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["main"] = r["result"][: product.main_count]
            draws.append(r)
    draws.sort(key=lambda r: (r["date"], r["id"]))
    return draws


def frequency(draws: list[dict], product: Product) -> Counter:
    """Count how often each number appears across all main draws."""
    c: Counter = Counter()
    for d in draws:
        c.update(d["main"])
    # ensure every valid number is present, even with 0
    for n in range(product.min_value, product.max_value + 1):
        c.setdefault(n, 0)
    return c


def recent_frequency(draws: list[dict], product: Product, n_draws: int) -> Counter:
    return frequency(draws[-n_draws:], product)


def days_since_last(draws: list[dict], product: Product,
                    ref: date | None = None) -> dict[int, dict]:
    """For each number, the last date it appeared and days since then."""
    ref = ref or datetime.now().date()
    last: dict[int, str] = {}
    for d in draws:
        for num in d["main"]:
            last[num] = d["date"]  # draws are sorted ascending
    out = {}
    for num in range(product.min_value, product.max_value + 1):
        ld = last.get(num)
        if ld:
            gap = (ref - datetime.fromisoformat(ld).date()).days
            out[num] = {"last_date": ld, "days_since": gap}
        else:
            out[num] = {"last_date": None, "days_since": None}
    return out


def hot_cold(draws: list[dict], product: Product, window: int = 30):
    """Return (hot, cold) lists of (number, count) over the last `window` draws."""
    c = recent_frequency(draws, product, window)
    ranked = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
    hot = ranked[:10]
    cold = sorted(c.items(), key=lambda kv: (kv[1], kv[0]))[:10]
    return hot, cold


def summary(product_name: str) -> dict:
    """Compute a full analysis bundle for a product."""
    product = get_product(product_name)
    draws = load_draws(product)
    if not draws:
        return {"product": product.label, "draws": 0}

    freq = frequency(draws, product)
    dsl = days_since_last(draws, product)
    hot30, cold30 = hot_cold(draws, product, 30)
    ranked_all = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    overdue = sorted(
        (v["days_since"], k) for k, v in dsl.items() if v["days_since"] is not None
    )
    overdue_top = [(num, gap) for gap, num in sorted(overdue, reverse=True)[:10]]

    return {
        "product": product.label,
        "draws": len(draws),
        "date_range": (draws[0]["date"], draws[-1]["date"]),
        "latest": draws[-1],
        "frequency": dict(freq),
        "most_common": ranked_all[:10],
        "least_common": ranked_all[-10:],
        "hot_30": hot30,
        "cold_30": cold30,
        "days_since_last": dsl,
        "most_overdue": overdue_top,
    }


if __name__ == "__main__":
    import sys

    s = summary(sys.argv[1] if len(sys.argv) > 1 else "power_655")
    print(f"{s['product']}: {s['draws']} draws, {s['date_range']}")
    print("Most common:", s["most_common"])
    print("Hot (30):", s["hot_30"])
    print("Most overdue:", s["most_overdue"])
