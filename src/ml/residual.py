"""Is the observed number×position grid really the theoretical one? (roadmap #12)

The dashboard already draws a number×position heatmap, and `joint.py` reports a
max-absolute-difference against the closed form. Both are descriptive: "close"
and "not close" are eyeballed. This turns that comparison into an actual test.

For a fixed position p, each draw independently puts value v there with the
known probability q = closed_form_grid[v][p]. Over T draws the count is
Binomial(T, q), so the standardised residual is

    z = (observed - T·q) / sqrt(T·q·(1-q))

Under the null (a fair draw) those z's are ~N(0,1): about 5% should exceed |2|
and about 0.3% should exceed |3|. If the grid held a real pattern, some cells
would blow far past that.

The per-position χ² needs care: most cells have tiny expected counts (q is often
< 0.01, so T·q < 5) and the χ² approximation breaks down there. So we **bin
adjacent numbers until each bin expects at least `MIN_EXPECTED`**, then test the
binned table. That is the difference between a real test and a number that just
looks like one.

Pure Python.
"""
from __future__ import annotations

import math

from analyze import load_draws
from config import Product, get_product
from ml.joint import closed_form_grid
from randomness import _chi2_sf

MIN_EXPECTED = 5.0


def observed_counts(product: Product, draws=None):
    """counts[v][p] = how often number v landed at sorted position p."""
    draws = draws if draws is not None else load_draws(product)
    k, N = product.main_count, product.max_value
    counts = [[0] * k for _ in range(N + 1)]
    for d in draws:
        for pos, val in enumerate(sorted(d["main"])):
            counts[val][pos] += 1
    return counts, len(draws)


def residual_grid(product: Product, draws=None):
    """z-score per cell, plus summary stats of how extreme they get."""
    counts, T = observed_counts(product, draws)
    theory = closed_form_grid(product)
    k, N = product.main_count, product.max_value
    z = [[0.0] * k for _ in range(N + 1)]
    flat = []
    for v in range(1, N + 1):
        for p in range(k):
            q = theory[v][p]
            if q <= 0 or q >= 1 or T == 0:
                continue
            exp = T * q
            sd = (T * q * (1.0 - q)) ** 0.5
            if sd <= 0:
                continue
            zz = (counts[v][p] - exp) / sd
            z[v][p] = zz
            flat.append(zz)
    n = len(flat) or 1
    return {
        "z": z, "draws": T, "cells": len(flat),
        "max_abs_z": max((abs(x) for x in flat), default=0.0),
        "frac_gt2": sum(1 for x in flat if abs(x) > 2) / n,
        "frac_gt3": sum(1 for x in flat if abs(x) > 3) / n,
        "mean_z": sum(flat) / n,
    }


def _binned_chi2(counts_col, theory_col, T, N):
    """χ² over adjacent-number bins, each expecting >= MIN_EXPECTED."""
    obs_bin, exp_bin = [], []
    o_acc = e_acc = 0.0
    for v in range(1, N + 1):
        o_acc += counts_col[v]
        e_acc += T * theory_col[v]
        if e_acc >= MIN_EXPECTED:
            obs_bin.append(o_acc)
            exp_bin.append(e_acc)
            o_acc = e_acc = 0.0
    if e_acc > 0:                      # fold any remainder into the last bin
        if exp_bin:
            obs_bin[-1] += o_acc
            exp_bin[-1] += e_acc
        else:
            obs_bin.append(o_acc)
            exp_bin.append(e_acc)
    chi2 = sum((o - e) ** 2 / e for o, e in zip(obs_bin, exp_bin) if e > 0)
    dof = max(len(exp_bin) - 1, 1)
    return chi2, dof, _chi2_sf(chi2, dof), len(exp_bin)


def chi2_by_position(product: Product, draws=None):
    counts, T = observed_counts(product, draws)
    theory = closed_form_grid(product)
    k, N = product.main_count, product.max_value
    rows = []
    for p in range(k):
        col_o = [0] + [counts[v][p] for v in range(1, N + 1)]
        col_t = [0.0] + [theory[v][p] for v in range(1, N + 1)]
        chi2, dof, pv, nbins = _binned_chi2(col_o, col_t, T, N)
        rows.append({"position": p + 1, "chi2": round(chi2, 2), "dof": dof,
                     "p": round(pv, 4), "bins": nbins, "ok": pv > 0.05})
    return rows


def _binom_tail(n: int, k: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i)
               for i in range(k, n + 1))


def _max_z_pvalue(max_abs_z: float, n_cells: int) -> float:
    """P(largest |z| among n_cells iid normals exceeds the observed one)."""
    if max_abs_z <= 0 or n_cells <= 0:
        return 1.0
    per_cell = math.erfc(max_abs_z / math.sqrt(2.0))     # P(|z| > t)
    return 1.0 - (1.0 - per_cell) ** n_cells


def summary(product_name: str) -> dict:
    product = get_product(product_name)
    draws = load_draws(product)
    if not draws:
        return {"product": product.label, "draws": 0}
    res = residual_grid(product, draws)
    rows = chi2_by_position(product, draws)
    n_bad = sum(1 for r in rows if not r["ok"])
    k = len(rows)
    min_p = min((r["p"] for r in rows), default=1.0)

    # Multiple comparisons: we ran k tests, so "some position had p < 0.05" is
    # not surprising on its own. Two standard corrections:
    #   family_p  — chance of >= n_bad failures out of k under the null
    #   sidak_p   — the smallest p, adjusted for having taken the minimum of k
    family_p = _binom_tail(k, n_bad, 0.05) if n_bad else 1.0
    sidak_p = 1.0 - (1.0 - min_p) ** k
    max_z_p = _max_z_pvalue(res["max_abs_z"], res["cells"])
    clean = sidak_p > 0.05 and family_p > 0.05

    if n_bad == 0:
        verdict = ("Every position matches the theoretical law (no p < 0.05) — "
                   "the grid is exactly what a fair draw makes.")
    elif clean:
        verdict = (
            f"{n_bad} of {k} positions fell under p < 0.05, but that is what "
            f"running {k} tests does: the chance of at least that many is "
            f"{family_p:.2f}, and the smallest p ({min_p:.4f}) becomes "
            f"{sidak_p:.3f} once corrected. No real deviation.")
    else:
        verdict = (
            f"{n_bad} of {k} positions deviate and it survives correction "
            f"(Šidák p = {sidak_p:.4f}) — worth investigating.")

    return {
        "product": product.label, "game": product_name,
        "draws": res["draws"], "cells": res["cells"],
        "max_abs_z": round(res["max_abs_z"], 2),
        "max_z_p": round(max_z_p, 4),
        "frac_gt2": round(res["frac_gt2"], 4),
        "frac_gt3": round(res["frac_gt3"], 4),
        "mean_z": round(res["mean_z"], 4),
        "positions": rows,
        "n_positions_failing": n_bad,
        "min_p": min_p, "family_p": round(family_p, 4),
        "sidak_p": round(sidak_p, 4), "clean": clean,
        "verdict": verdict,
    }


def format_report(r: dict) -> str:
    if not r.get("draws"):
        return f"{r['product']}: no data."
    out = [
        f"{r['product']} — observed grid vs the theoretical law "
        f"({r['draws']} draws, {r['cells']} cells)",
        "",
        "  standardised residuals z = (observed - expected)/sd",
        f"    mean z        {r['mean_z']:+.4f}   (expect ~0)",
        f"    |z| > 2       {100*r['frac_gt2']:.2f}%    (expect ~4.6%)",
        f"    |z| > 3       {100*r['frac_gt3']:.2f}%    (expect ~0.3%)",
        f"    largest |z|   {r['max_abs_z']:.2f}   "
        f"(p = {r['max_z_p']:.3f} for the largest of {r['cells']} cells)",
        "",
        "  per-position goodness of fit (numbers binned so each expects >= 5)",
        f"    {'pos':<6}{'chi2':>10}{'dof':>6}{'bins':>6}{'p':>10}",
    ]
    for row in r["positions"]:
        flag = "" if row["ok"] else "   <-- p < 0.05"
        out.append(f"    {row['position']:<6}{row['chi2']:>10.2f}{row['dof']:>6}"
                   f"{row['bins']:>6}{row['p']:>10.4f}{flag}")
    out += [
        "",
        f"  multiple comparisons over {len(r['positions'])} positions: "
        f"smallest p {r['min_p']:.4f} -> Šidák {r['sidak_p']:.4f}, "
        f"P(>= {r['n_positions_failing']} failures) = {r['family_p']:.3f}",
        "",
        f"  {r['verdict']}",
    ]
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "power_655"
    print(format_report(summary(name)))
