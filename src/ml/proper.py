"""Proper scoring rules for the number×position grid (roadmap #18).

The k/6 position score is honest but *very* noisy: six positions per draw, each
almost always missed, so a predictor's score is mostly 0s with the occasional 1.
Ranking models on it is close to ranking coin flips.

A proper scoring rule fixes that by grading the whole predicted distribution
instead of one hard guess. For each draw we take the model's grid
P(number | position), look at the number that actually landed at that position,
and charge it:

    log-loss  = -ln P(actual)                     (nats; lower is better)
    Brier     = sum_v (P(v) - 1{v = actual})^2    (lower is better)

Both are *proper*: they are minimised only by reporting your true beliefs, so a
model cannot game them by being over-confident.

The reason this matters here is the floor. For a uniform draw the position law
is fixed and known (`joint.closed_form_grid`), so the best achievable expected
log-loss is the entropy of that law:

    H = -sum_v P(v) ln P(v)      (averaged over positions)

No model that only reads past draws can beat H — the draws carry no information
about each other. `entropy_floor` below computes it, and the report prints every
model against it. Expect: theory ≈ floor, empirical a hair worse (estimation
noise), uniform far worse. That gap between uniform and theory is real
structure — position 1 really is usually small — but it is structure that is
identical every draw, so it still buys you nothing.

Pure Python: no numpy needed.
"""
from __future__ import annotations

import random
from math import log

from analyze import load_draws
from config import Product, get_product
from ml.joint import closed_form_grid, empirical_grid
from ml.util import progress

EPS = 1e-12


def _column(grid, pos: int, N: int) -> list[float]:
    """Normalised P(number | position) column, index 0 unused."""
    col = [0.0] + [max(grid[v][pos], 0.0) for v in range(1, N + 1)]
    s = sum(col)
    if s <= 0:
        return [0.0] + [1.0 / N] * N
    return [c / s for c in col]


def uniform_grid(product: Product):
    k, N = product.main_count, product.max_value
    return [[1.0 / N] * k for _ in range(N + 1)]


def shrunk_grid(product: Product, draws, strength: float = 20.0):
    """Empirical counts blended toward the closed-form prior (preview of #21).

    strength = prior weight in pseudo-draws; T/(T+strength) of the way to the
    empirical grid.
    """
    k, N = product.main_count, product.max_value
    prior = closed_form_grid(product)
    T = len(draws)
    counts = [[0.0] * k for _ in range(N + 1)]
    for d in draws:
        for pos, val in enumerate(sorted(d["main"])):
            counts[val][pos] += 1.0
    grid = [[0.0] * k for _ in range(N + 1)]
    denom = T + strength
    for v in range(1, N + 1):
        for pos in range(k):
            grid[v][pos] = (counts[v][pos] + strength * prior[v][pos]) / denom
    return grid


def entropy_floor(product: Product) -> float:
    """Mean per-position entropy of the true law — the best possible log-loss."""
    k, N = product.main_count, product.max_value
    g = closed_form_grid(product)
    total = 0.0
    for pos in range(k):
        col = _column(g, pos, N)
        total += -sum(p * log(p) for p in col[1:] if p > 0)
    return total / k


def _score_draw(grid, actual_sorted, product: Product) -> tuple[float, float]:
    """(mean log-loss, mean Brier) over the k positions of one draw."""
    k, N = product.main_count, product.max_value
    ll = br = 0.0
    for pos in range(k):
        col = _column(grid, pos, N)
        actual = actual_sorted[pos]
        p = max(col[actual], EPS)
        ll += -log(p)
        # Brier = (1-p_actual)^2 + sum of squares of the rest
        br += (1.0 - p) ** 2 + sum(q * q for i, q in enumerate(col)
                                   if i >= 1 and i != actual)
    return ll / k, br / k


def _bootstrap_ci(samples, n_boot=1000, seed=0):
    rng = random.Random(seed)
    n = len(samples)
    if n == 0:
        return 0.0, 0.0
    means = sorted(sum(rng.choice(samples) for _ in range(n)) / n
                   for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


# grid builders: name -> f(product, history_draws) -> grid
BUILDERS = {
    "uniform": lambda p, h: uniform_grid(p),
    "theory": lambda p, h: closed_form_grid(p),
    "empirical": lambda p, h: empirical_grid(p, h, smoothing=0.5),
    "empirical-100": lambda p, h: empirical_grid(p, h[-100:], smoothing=0.5),
    "shrunk": lambda p, h: shrunk_grid(p, h, strength=20.0),
}


def evaluate(product_name: str, test_draws: int = 200,
             min_history: int = 50) -> dict:
    """Walk-forward proper scoring of every grid builder."""
    product = get_product(product_name)
    draws = load_draws(product)
    k = product.main_count
    start = max(min_history, len(draws) - test_draws)
    if start >= len(draws):
        return {"product": product.label, "tested": 0}

    per_model_ll = {m: [] for m in BUILDERS}
    per_model_br = {m: [] for m in BUILDERS}
    total = len(draws) - start
    for j, t in enumerate(range(start, len(draws))):
        history = draws[:t]
        actual = sorted(draws[t]["main"][:k])
        for name, build in BUILDERS.items():
            grid = build(product, history)
            ll, br = _score_draw(grid, actual, product)
            per_model_ll[name].append(ll)
            per_model_br[name].append(br)
        progress(j + 1, total, "proper scoring")

    floor = entropy_floor(product)
    models = {}
    for name in BUILDERS:
        lls = per_model_ll[name]
        lo, hi = _bootstrap_ci(lls)
        models[name] = {
            "logloss": sum(lls) / len(lls),
            "logloss_lo": lo, "logloss_hi": hi,
            "brier": sum(per_model_br[name]) / len(per_model_br[name]),
            # how far above the irreducible floor, in nats
            "excess": sum(lls) / len(lls) - floor,
            "beats_floor": hi < floor,
        }
    return {"product": product.label, "game": product_name,
            "tested": total, "entropy_floor": floor, "models": models}


def format_report(r: dict) -> str:
    if not r.get("tested"):
        return f"{r['product']}: not enough data."
    floor = r["entropy_floor"]
    rows = sorted(r["models"].items(), key=lambda kv: kv[1]["logloss"])
    out = [
        f"{r['product']} — proper scoring of the position grid "
        f"over {r['tested']} draws",
        "",
        f"  irreducible floor (entropy of the true law): {floor:.4f} nats",
        "  no history-only model can beat this in expectation",
        "",
        f"  {'model':<15}{'log-loss':>10}{'95% CI':>20}{'excess':>10}{'Brier':>9}",
    ]
    for name, m in rows:
        ci = f"[{m['logloss_lo']:.3f}, {m['logloss_hi']:.3f}]"
        out.append(
            f"  {name:<15}{m['logloss']:>10.4f}{ci:>20}"
            f"{m['excess']:>+10.4f}{m['brier']:>9.4f}")
    best = rows[0]
    out += [
        "",
        f"  best: {best[0]} ({best[1]['logloss']:.4f}, "
        f"{best[1]['excess']:+.4f} vs floor)",
    ]
    if any(m["beats_floor"] for _, m in rows):
        out.append("  ⚑ something scored below the floor — that shouldn't happen, investigate.")
    else:
        out.append("  Nothing beats the floor, as theory says. 'uniform' is far worse,")
        out.append("  which shows the position law is real structure — but it is the")
        out.append("  same every draw, so it still predicts nothing.")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "power_655"
    print(format_report(evaluate(name)))
