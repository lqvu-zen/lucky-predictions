"""Per-position classifier — the trained ML version of the joint grid.

For each ordered position p1…p6 we train a multiclass classifier that predicts
*which number* lands at that position, from history features. Their
predict_proba outputs build a learned grid `P(number | position)`, and we
greedy-assign a distinct ascending ticket from it (same assembly as the joint
model). With no signal the learned grid just reproduces the order-statistic
marginals, so it lands on the baseline. Needs the ML extras.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from analyze import load_draws
from config import Product, get_product
from ml.joint import predict_ticket
from ml.positional import _row_features
from ml.util import progress


def build_base(product: Product, draws=None, min_history: int = 50):
    draws = draws if draws is not None else load_draws(product)
    k = product.main_count
    S = np.array([sorted(d["main"]) for d in draws])
    X, Y, di = [], [], []
    for t in range(min_history, len(draws)):
        dow = datetime.fromisoformat(draws[t]["date"]).date().weekday()
        X.append(_row_features(S[:t], dow, k))
        Y.append(S[t])
        di.append(t)
    return np.vstack(X), np.vstack(Y), np.array(di)


def train(Xbase, Y, k):
    from sklearn.linear_model import LogisticRegression
    models = []
    for i in range(k):
        clf = LogisticRegression(max_iter=300, multi_class="auto")
        clf.fit(Xbase, Y[:, i])
        models.append(clf)
    return models


def _grid(models, xrow, product: Product):
    k, N = product.main_count, product.max_value
    grid = [[0.0] * k for _ in range(N + 1)]
    row = xrow.reshape(1, -1)
    for pos, clf in enumerate(models):
        proba = clf.predict_proba(row)[0]
        for c, pv in zip(clf.classes_, proba):
            if 1 <= int(c) <= N:
                grid[int(c)][pos] = float(pv)
    return grid


def predict_next(product_name: str) -> dict:
    product = get_product(product_name)
    draws = load_draws(product)
    k = product.main_count
    Xb, Y, _ = build_base(product, draws)
    models = train(Xb, Y, k)
    target = product.next_draw_date()
    S = np.array([sorted(d["main"]) for d in draws])
    grid = _grid(models, _row_features(S, target.weekday(), k), product)
    return {"product": product.label, "model": "perpos-clf",
            "target_date": target.isoformat(),
            "ticket": predict_ticket(grid, product)}


def backtest(product_name: str, test_draws: int = 120, retrain_every: int = 25,
             min_history: int = 50) -> dict:
    product = get_product(product_name)
    draws = load_draws(product)
    k, n = product.main_count, product.max_value
    Xb, Y, di = build_base(product, draws, min_history)
    if len(di) == 0:
        return {"product": product.label, "tested": 0}
    test_idx = di[-test_draws:] if test_draws < len(di) else di
    hits, models = [], None
    for step, t in enumerate(test_idx):
        mask = di < t
        if mask.sum() == 0:
            continue
        if models is None or step % retrain_every == 0:
            models = train(Xb[mask], Y[mask], k)
        grid = _grid(models, Xb[di == t][0], product)
        ticket = predict_ticket(grid, product)
        actual = set(draws[int(t)]["main"][:k])
        hits.append(len(actual.intersection(ticket)))
        progress(step + 1, len(test_idx), "perpos-clf backtest")
    hits = np.array(hits, float)
    m = len(hits)
    rng = np.random.default_rng(0)
    boot = [hits[rng.integers(0, m, m)].mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    base = k * k / n
    return {"product": product.label, "model": "perpos-clf", "tested": m,
            "mean_hits": float(hits.mean()), "hits_lo": float(lo),
            "hits_hi": float(hi), "baseline_hits": base,
            "beats_baseline": bool(lo > base)}


def format_backtest(r: dict) -> str:
    if not r.get("tested"):
        return f"{r['product']}: not enough data."
    verdict = ("⚑ CI above baseline — investigate!" if r["beats_baseline"]
               else "within noise of random (CI spans the baseline)")
    return "\n".join([
        f"{r['product']} — per-position classifier backtest ({r['model']}) "
        f"over {r['tested']} draws",
        "",
        f"  mean hits       {r['mean_hits']:.3f}  "
        f"[95% CI {r['hits_lo']:.3f}, {r['hits_hi']:.3f}]",
        f"  random baseline {r['baseline_hits']:.3f}   → {verdict}",
        "",
        "  A trained classifier per position; it just relearns the fixed "
        "position marginals. Same honest verdict.",
    ])


if __name__ == "__main__":
    import sys
    print(format_backtest(backtest(sys.argv[1] if len(sys.argv) > 1 else "power_655")))
