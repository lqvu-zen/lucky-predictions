"""Does one draw tell you anything about the next? (roadmap #16)

Every other test in this project checks a *marginal* distribution: does number K
land at position I as often as theory says (`ml.residual`), do the numbers come
up uniformly (`randomness`). All of those could pass perfectly while the draws
were still predictable — imagine a machine that produced a correct-looking
distribution but always followed 7 with 23. Nothing built so far would notice.

That matters because independence between draws is the assumption the *entire*
project rests on. Every model here reads the past to guess the future, and every
one of them fails; but that is indirect evidence. This module tests the
assumption head-on, three ways:

1. **Per-position autocorrelation.** For each sorted position p and lag L,
   correlate the value at p in draw t with the value at p in draw t−L. Under
   independence every one of these is ~0, with Fisher-transformed sampling error
   1/√(n−3).

2. **Overlap with an earlier draw.** How many numbers does draw t share with
   draw t−L? Under independence that is exactly Hypergeometric(N, k, k) — the
   *same law* as "how many of your ticket's numbers come up", since in both
   cases you are comparing a fixed k-subset with a fresh random one. Mean k²/N.

3. **Shape of that overlap.** Not just the mean: χ² the whole distribution of
   overlap counts against the hypergeometric pmf.

With k positions × L lags we run a lot of tests at once, so the report applies a
Šidák correction to the most extreme result — the same discipline as `residual`
and `ceiling`. Without it, "the largest of 60 correlations exceeded 2σ" would
read as a discovery instead of arithmetic.

Pure Python.
"""
from __future__ import annotations

import math

from analyze import load_draws
from config import Product, get_product
from randomness import _chi2_sf
from ticket_ev import match_probability


def _normal_sf(z: float) -> float:
    """P(|Z| > |z|) for a standard normal."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def _pearson(xs, ys) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return 0.0
    return sxy / math.sqrt(sxx * syy)


def _fisher_p(r: float, n: int) -> float:
    """Two-sided p-value for a correlation via Fisher's z transform."""
    if n < 5 or abs(r) >= 1.0:
        return 1.0
    z = math.atanh(r) * math.sqrt(n - 3)
    return _normal_sf(z)


def lag_correlations(product: Product, draws, max_lag: int = 10) -> dict:
    """Correlation of each sorted position with itself, at lags 1..max_lag."""
    k = product.main_count
    S = [sorted(d["main"][:k]) for d in draws]
    cells, worst = [], None
    for p in range(k):
        col = [s[p] for s in S]
        for lag in range(1, max_lag + 1):
            if len(col) <= lag + 5:
                continue
            xs, ys = col[lag:], col[:-lag]
            r = _pearson(xs, ys)
            pv = _fisher_p(r, len(xs))
            cell = {"position": p + 1, "lag": lag, "r": r, "p": pv,
                    "n": len(xs)}
            cells.append(cell)
            if worst is None or abs(r) > abs(worst["r"]):
                worst = cell
    n_tests = len(cells) or 1
    beyond2 = sum(1 for c in cells if abs(c["r"]) * math.sqrt(c["n"] - 3) > 2)
    min_p = min((c["p"] for c in cells), default=1.0)
    return {
        "cells": cells, "n_tests": n_tests, "worst": worst,
        "n_beyond_2sigma": beyond2,
        "expected_beyond_2sigma": 0.0455 * n_tests,
        "min_p": min_p,
        "sidak_p": 1.0 - (1.0 - min_p) ** n_tests,
    }


def overlap_by_lag(product: Product, draws, max_lag: int = 10) -> list[dict]:
    """Mean shared numbers with the draw L back, vs the hypergeometric law."""
    k, N = product.main_count, product.max_value
    sets = [set(d["main"][:k]) for d in draws]
    mu = k * k / N
    var = k * (k / N) * ((N - k) / N) * ((N - k) / (N - 1))
    rows = []
    for lag in range(1, max_lag + 1):
        if len(sets) <= lag + 5:
            continue
        obs = [len(sets[i] & sets[i - lag]) for i in range(lag, len(sets))]
        n = len(obs)
        mean = sum(obs) / n
        se = math.sqrt(var / n)
        z = (mean - mu) / se if se > 0 else 0.0
        rows.append({"lag": lag, "n": n, "mean": mean, "expected": mu,
                     "z": z, "p": _normal_sf(z)})
    return rows


def overlap_distribution(product: Product, draws, lag: int = 1) -> dict:
    """χ² of the full overlap-count distribution vs hypergeometric."""
    k = product.main_count
    sets = [set(d["main"][:k]) for d in draws]
    if len(sets) <= lag + 5:
        return {"lag": lag, "n": 0}
    obs_counts = [0] * (k + 1)
    for i in range(lag, len(sets)):
        obs_counts[len(sets[i] & sets[i - lag])] += 1
    n = sum(obs_counts)
    # bin the tail so every expected count is >= 5
    exp = [n * match_probability(product, j) for j in range(k + 1)]
    o_b, e_b = [], []
    o_acc = e_acc = 0.0
    for j in range(k + 1):
        o_acc += obs_counts[j]
        e_acc += exp[j]
        if e_acc >= 5.0:
            o_b.append(o_acc)
            e_b.append(e_acc)
            o_acc = e_acc = 0.0
    if e_acc > 0 and e_b:
        o_b[-1] += o_acc
        e_b[-1] += e_acc
    chi2 = sum((o - e) ** 2 / e for o, e in zip(o_b, e_b) if e > 0)
    dof = max(len(e_b) - 1, 1)
    return {"lag": lag, "n": n, "observed": obs_counts,
            "expected": [round(x, 1) for x in exp],
            "chi2": chi2, "dof": dof, "p": _chi2_sf(chi2, dof),
            "bins": len(e_b)}


def summary(product_name: str, max_lag: int = 10) -> dict:
    product = get_product(product_name)
    draws = load_draws(product)
    if len(draws) < 60:
        return {"product": product.label, "draws": len(draws)}
    corr = lag_correlations(product, draws, max_lag)
    laps = overlap_by_lag(product, draws, max_lag)
    dist = overlap_distribution(product, draws, 1)

    lap_min_p = min((r["p"] for r in laps), default=1.0)
    lap_sidak = 1.0 - (1.0 - lap_min_p) ** (len(laps) or 1)
    clean = (corr["sidak_p"] > 0.05 and lap_sidak > 0.05
             and dist.get("p", 1.0) > 0.05)
    return {
        "product": product.label, "game": product_name,
        "draws": len(draws), "max_lag": max_lag,
        "corr": corr, "overlap": laps, "overlap_dist": dist,
        "overlap_sidak_p": lap_sidak,
        "clean": clean,
        "verdict": (
            "No detectable link between draws: correlations, overlaps and their "
            "distribution all match independence. The assumption every model "
            "here relies on holds."
            if clean else
            "Something survived multiple-comparison correction — worth a hard "
            "look before trusting it."),
    }


def format_report(r: dict) -> str:
    if not r.get("draws") or "corr" not in r:
        return f"{r['product']}: not enough data."
    c, w = r["corr"], r["corr"]["worst"]
    out = [
        f"{r['product']} — does one draw predict the next? "
        f"({r['draws']} draws, lags 1..{r['max_lag']})",
        "",
        "  1) autocorrelation of each sorted position with itself",
        f"     tests run                {c['n_tests']}",
        f"     |z| > 2                  {c['n_beyond_2sigma']}  "
        f"(expect ~{c['expected_beyond_2sigma']:.1f})",
        f"     strongest               r = {w['r']:+.4f} "
        f"(position {w['position']}, lag {w['lag']}, p = {w['p']:.4f})",
        f"     corrected for {c['n_tests']} tests   Šidák p = {c['sidak_p']:.4f}",
        "",
        "  2) numbers shared with the draw L back "
        f"(expect {r['overlap'][0]['expected']:.4f})",
        f"     {'lag':>5}{'mean':>10}{'z':>9}{'p':>9}",
    ]
    for row in r["overlap"]:
        out.append(f"     {row['lag']:>5}{row['mean']:>10.4f}"
                   f"{row['z']:>+9.2f}{row['p']:>9.4f}")
    out.append(f"     corrected  Šidák p = {r['overlap_sidak_p']:.4f}")

    d = r["overlap_dist"]
    if d.get("n"):
        out += [
            "",
            "  3) shape of the lag-1 overlap distribution vs hypergeometric",
            f"     observed  {d['observed']}",
            f"     expected  {d['expected']}",
            f"     chi2 = {d['chi2']:.2f} (dof {d['dof']}, {d['bins']} bins), "
            f"p = {d['p']:.4f}",
        ]
    out += ["", f"  {r['verdict']}"]
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "power_655"
    print(format_report(summary(name)))
