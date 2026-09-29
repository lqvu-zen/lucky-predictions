"""Per-position classifiers — the trained ML versions of the joint grid.

For each ordered position p1…p6 we train a multiclass classifier that predicts
*which number* lands at that position, from history features. Their
predict_proba outputs build a learned grid `P(number | position)`, and
`optimal.optimal_ticket` picks the best ascending ticket from it. Three
learners share the pipeline:

    perpos-clf  logistic regression (scaled)
    perpos-hgb  histogram gradient boosting (sklearn's LightGBM-style trees)
    perpos-mlp  a small neural network (one hidden layer of 32)

Settings were chosen by 5-fold cross-validated log-loss on 6/55 (1353 rows).
The reference is "ignore the features, use each position's frequencies":

    frequencies only                  3.3895
    logistic, old (unscaled, C=1)     3.8228
    logistic, scaled, C=1e-4          3.4056
    MLP 32 hidden, alpha=30           3.4094
    HistGB 10 rounds, depth 2         3.4194

Every learner got *better* the more it was told to ignore its features, and
none beat plain frequencies — the features are noise, so the best a classifier
can do is relearn the fixed position law. That is the finding, not a tuning
failure. (The old logistic setting fitted the noise and sat at ~84% of the
ceiling; these land near it.) Needs the ML extras.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from analyze import load_draws
from config import Product, get_product
from ml import decode
from ml.optimal import optimal_ticket
from ml.positional import _row_features
from ml.util import progress

KINDS = {"logreg": "perpos-clf", "hgb": "perpos-hgb", "mlp": "perpos-mlp"}


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


def make_classifier(kind: str = "logreg"):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "logreg":
        # note: `multi_class` was removed in scikit-learn 1.7 — leave it
        # off; LogisticRegression is multinomial-capable by default.
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=1e-4, max_iter=2000))
    if kind == "hgb":
        # no early stopping: sklearn stratifies its validation split, which
        # fails when a number has landed at a position only once
        return HistGradientBoostingClassifier(
            max_iter=10, learning_rate=0.05, max_depth=2, min_samples_leaf=40,
            l2_regularization=10.0, early_stopping=False, random_state=0)
    if kind == "mlp":
        # early stopping halts before the output biases learn the position
        # frequencies (CV log-loss 3.77 with it vs 3.41 without)
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(32,), alpha=30.0, early_stopping=False,
            max_iter=600, random_state=0))
    raise ValueError(f"kind must be one of {list(KINDS)}")


def train(Xbase, Y, k, kind: str = "logreg"):
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    models = []
    with warnings.catch_warnings():
        # the MLP can stop at its iteration cap on this tiny, signal-free
        # problem; the fit is fine and the warning just floods the daily log
        warnings.simplefilter("ignore", ConvergenceWarning)
        for i in range(k):
            models.append(make_classifier(kind).fit(Xbase, Y[:, i]))
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


def predict_next(product_name: str, kind: str = "logreg") -> dict:
    product = get_product(product_name)
    draws = load_draws(product)
    k = product.main_count
    Xb, Y, _ = build_base(product, draws)
    models = train(Xb, Y, k, kind)
    target = product.next_draw_date()
    S = np.array([sorted(d["main"]) for d in draws])
    grid = _grid(models, _row_features(S, target.weekday(), k), product)
    return {"product": product.label, "model": KINDS[kind],
            "target_date": target.isoformat(),
            "ticket": optimal_ticket(grid, product)}


def backtest(product_name: str, kind: str = "logreg", test_draws: int = 120,
             retrain_every: int = 25, min_history: int = 50) -> dict:
    product = get_product(product_name)
    draws = load_draws(product)
    k, n = product.main_count, product.max_value
    Xb, Y, di = build_base(product, draws, min_history)
    if len(di) == 0:
        return {"product": product.label, "tested": 0}
    test_idx = di[-test_draws:] if test_draws < len(di) else di
    hits, tickets, actuals, models = [], [], [], None
    for step, t in enumerate(test_idx):
        mask = di < t
        if mask.sum() == 0:
            continue
        if models is None or step % retrain_every == 0:
            models = train(Xb[mask], Y[mask], k, kind)
        grid = _grid(models, Xb[di == t][0], product)
        ticket = optimal_ticket(grid, product)
        actual = draws[int(t)]["main"][:k]
        hits.append(len(set(actual).intersection(ticket)))
        tickets.append(ticket)
        actuals.append(actual)
        progress(step + 1, len(test_idx), f"{KINDS[kind]} backtest")
    hits = np.array(hits, float)
    m = len(hits)
    rng = np.random.default_rng(0)
    boot = [hits[rng.integers(0, m, m)].mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    base = k * k / n
    return {"product": product.label, "model": KINDS[kind], "tested": m,
            "mean_hits": float(hits.mean()), "hits_lo": float(lo),
            "hits_hi": float(hi), "baseline_hits": base,
            "beats_baseline": bool(lo > base),
            **decode.pos_summary(tickets, actuals, product)}


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
        *decode.format_pos(r),
        "",
        "  A trained classifier per position; it just relearns the fixed "
        "position marginals. Same honest verdict.",
    ])


if __name__ == "__main__":
    import sys
    print(format_backtest(backtest(sys.argv[1] if len(sys.argv) > 1 else "power_655")))
