"""Turn a model's output into the ticket the position score actually rewards.

The k/6 position score counts exact matches: number v at sorted position p. For
an exact-match metric the Bayes-optimal guess is the *most likely* value (the
mode), not the average. The regression models (positional, chain, gap) were
rounding their predicted mean — and the position law is skewed, so the mean is
a poor guess. In 6/55 position 1 averages ~8 (P = 5.3%) while its mode is 1
(P = 10.9%). Rounding the mean left those models at ~69% of the ceiling, the
same as a random ticket.

`residual_grid` fixes the decision without changing what the model learns:
treat the model's prediction plus its own out-of-fold residuals as a predictive
distribution per position, then `optimal.optimal_ticket` picks the ascending
ticket that maximises expected correct positions.

`law_pos_score` is the noise-free yardstick: the *expected* k/6 of a ticket
under the exact order-statistic law. A realized score over 30 draws is mostly
luck; this number is not, so it is what shows whether a change helped.

Honest note: all of this moves a model toward the ceiling
(`ceiling.py`, 0.0651 for 6/55), never past it, and changes nothing about the
odds of winning a prize.
"""
from __future__ import annotations

from functools import lru_cache

from config import Product
from ml.joint import closed_form_grid
from ml.optimal import expected_pos_hits, optimal_ticket


def residual_grid(mu, residuals, product: Product):
    """grid[v][p] = share of (mu[p] + residual) that rounds to v.

    mu: k predicted values; residuals: k sequences of (actual - predicted).
    A plain histogram. Gaussian-kernel smoothing was tried and was worse (92.7%
    vs 95.2% of the ceiling in simulation): near the ends it drags the mode of
    position 1 from 1 to 2, and that is exactly where the gain is.
    """
    k, N = product.main_count, product.max_value
    grid = [[0.0] * k for _ in range(N + 1)]
    for p in range(k):
        rs = residuals[p]
        w = 1.0 / len(rs)
        for r in rs:
            v = min(max(int(round(float(mu[p]) + float(r))), 1), N)
            grid[v][p] += w
    return grid


def decode(mu, residuals, product: Product) -> list[int]:
    """Best ascending ticket for a point prediction + residual spread."""
    return optimal_ticket(residual_grid(mu, residuals, product), product)


def oof_predictions(fit, predict, X, Y, folds: int = 5):
    """Out-of-fold predictions, so residuals are honest (not in-sample).

    fit(X, Y) -> model; predict(model, X) -> array shaped like Y.
    Draws are exchangeable (no signal across time), so plain interleaved folds
    are fine and keep every fold spread over the whole history.
    """
    import numpy as np
    n = len(X)
    out = np.zeros_like(Y, dtype=float)
    idx = np.arange(n)
    for f in range(folds):
        test = idx % folds == f
        model = fit(X[~test], Y[~test])
        out[test] = predict(model, X[test])
    return out


def residuals_from(fit, predict, X, Y, folds: int = 5):
    """Per-position out-of-fold residuals, shaped (k, n)."""
    pred = oof_predictions(fit, predict, X, Y, folds)
    return (Y - pred).T


@lru_cache(maxsize=None)
def _law(product: Product):
    return closed_form_grid(product)


def law_pos_score(ticket, product: Product) -> float:
    """Expected k/6 of this ticket under the exact law (noise-free)."""
    return expected_pos_hits(_law(product), ticket, product) / product.main_count


def ceiling_score(product: Product) -> float:
    """Best expected k/6 any ticket can have (the law's optimal ticket)."""
    return law_pos_score(optimal_ticket(_law(product), product), product)


def consensus_ticket(tickets, product: Product) -> list[int]:
    """Position-aware vote across predictors' tickets.

    grid[v][p] = how many tickets put number v at sorted position p (with a
    tiny exact-law term to break ties), then the optimal ascending ticket.
    Voting per *number* ignored positions, which is what the k/6 score grades,
    and left the old consensus at ~65% of the ceiling.
    """
    k, N = product.main_count, product.max_value
    q = _law(product)
    grid = [[1e-6 * q[v][p] for p in range(k)] for v in range(N + 1)]
    for t in tickets:
        for p, v in enumerate(sorted(t)[:k]):
            grid[v][p] += 1.0
    return optimal_ticket(grid, product)


def pos_summary(tickets, actuals, product: Product) -> dict:
    """Position-score fields every backtest adds to its result.

    Mean *hits* can't separate models — any ticket averages k^2/N — so the
    backtests also report the k/6 position score, realized and expected.
    """
    k = product.main_count
    if not tickets:
        return {}
    realized = [sum(1 for t, a in zip(sorted(tk), sorted(ac)) if t == a) / k
                for tk, ac in zip(tickets, actuals)]
    expected = [law_pos_score(tk, product) for tk in tickets]
    return {"mean_pos_score": sum(realized) / len(realized),
            "law_pos_score": sum(expected) / len(expected),
            "ceiling_score": ceiling_score(product)}


def format_pos(r: dict) -> list[str]:
    if "law_pos_score" not in r:
        return []
    pct = 100.0 * r["law_pos_score"] / r["ceiling_score"]
    return [f"  k/6 position    {r['mean_pos_score']:.4f} realized, "
            f"{r['law_pos_score']:.4f} expected = {pct:.0f}% of the "
            f"{r['ceiling_score']:.4f} ceiling"]
