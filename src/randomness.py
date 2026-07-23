"""Statistical randomness tests on the draw history.

The whole project rests on one claim: the draws are uniform and independent.
This module *checks* that claim with standard tests, so the "no edge" story is
demonstrated, not asserted:

  - chi-square goodness-of-fit: are all numbers drawn equally often?
  - odd/even balance
  - mean repeats between consecutive draws vs the hypergeometric expectation

p-values use a pure-Python Wilson-Hilferty approximation of the chi-square
survival function (no scipy needed), which is plenty accurate here.

Expected outcome: high p-values — we cannot reject randomness. That's the
point.
"""
from __future__ import annotations

import math
from collections import Counter

from analyze import load_draws
from config import get_product


def _chi2_sf(x: float, k: int) -> float:
    """P(X > x) for a chi-square with k dof (Wilson-Hilferty normal approx)."""
    if x <= 0 or k <= 0:
        return 1.0
    t = (x / k) ** (1.0 / 3.0)
    mean = 1.0 - 2.0 / (9.0 * k)
    sd = math.sqrt(2.0 / (9.0 * k))
    z = (t - mean) / sd
    return 0.5 * math.erfc(z / math.sqrt(2.0))   # normal survival function


def uniformity(draws, product) -> dict:
    lo, hi = product.min_value, product.max_value
    c: Counter = Counter()
    for d in draws:
        c.update(d["main"])
    total = sum(c.values())
    n_numbers = hi - lo + 1
    exp = total / n_numbers
    chi2 = sum((c.get(n, 0) - exp) ** 2 / exp for n in range(lo, hi + 1))
    dof = n_numbers - 1
    return {"chi2": round(chi2, 1), "dof": dof, "p": round(_chi2_sf(chi2, dof), 4),
            "expected_per_number": round(exp, 1)}


def odd_even(draws, product) -> dict:
    lo, hi = product.min_value, product.max_value
    odd_in_range = sum(1 for n in range(lo, hi + 1) if n % 2)
    total = sum(len(d["main"]) for d in draws)
    odd_picks = sum(1 for d in draws for n in d["main"] if n % 2)
    even_picks = total - odd_picks
    exp_odd = total * odd_in_range / (hi - lo + 1)
    exp_even = total - exp_odd
    chi2 = ((odd_picks - exp_odd) ** 2 / exp_odd
            + (even_picks - exp_even) ** 2 / exp_even)
    return {"odd": odd_picks, "even": even_picks,
            "exp_odd": round(exp_odd), "p": round(_chi2_sf(chi2, 1), 4)}


def repeats(draws, product) -> dict:
    k, n = product.main_count, product.max_value
    expected = k * k / n     # hypergeometric mean overlap of two draws
    reps = [len(set(draws[i - 1]["main"]) & set(draws[i]["main"]))
            for i in range(1, len(draws))]
    mean = sum(reps) / len(reps) if reps else 0.0
    return {"mean_repeat": round(mean, 3), "expected": round(expected, 3)}


def summary(product_name: str) -> dict:
    product = get_product(product_name)
    draws = load_draws(product)
    if not draws:
        return {"product": product.label, "draws": 0}
    u = uniformity(draws, product)
    verdict = ("Consistent with a fair, uniform random draw — cannot reject "
               "randomness." if u["p"] > 0.05 else
               "Frequencies deviate more than expected — worth a closer look.")
    return {"product": product.label, "draws": len(draws),
            "uniformity": u, "odd_even": odd_even(draws, product),
            "repeats": repeats(draws, product), "verdict": verdict}


def format_report(s: dict) -> str:
    if not s.get("draws"):
        return f"{s['product']}: no data."
    u, oe, r = s["uniformity"], s["odd_even"], s["repeats"]
    return "\n".join([
        f"{s['product']} — randomness tests over {s['draws']} draws",
        "",
        f"  chi-square uniformity:  X2={u['chi2']} (dof {u['dof']}), "
        f"p={u['p']}  [each number expected ~{u['expected_per_number']} times]",
        f"  odd/even balance:       odd={oe['odd']} even={oe['even']} "
        f"(expected odd ~{oe['exp_odd']}), p={oe['p']}",
        f"  repeats vs previous:    mean {r['mean_repeat']} "
        f"(expected {r['expected']})",
        "",
        f"  Verdict: {s['verdict']}",
    ])


if __name__ == "__main__":
    import sys
    print(format_report(summary(sys.argv[1] if len(sys.argv) > 1 else "power_655")))
