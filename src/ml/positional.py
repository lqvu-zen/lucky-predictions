"""Positional (ordered) model — an alternative framing of the question.

Instead of asking "for each number, what's the probability it appears?",
we sort each draw's 6 numbers ascending (p1 < p2 < ... < p6) and ask:

    what value will appear at each ORDERED POSITION of the next draw?

So we train 6 regressors, one per position, each predicting that
position's value from history, then assemble a valid ascending ticket.

Why this is interesting: the ordered positions have real structure — p1 is
almost always small, p6 almost always large (these are "order statistics").
So this model produces natural-looking tickets like 4-15-24-33-42-51 by
fitting those marginal distributions. But *which* value lands within each
position's range on a given draw is still random, so it does NOT beat the
baseline at matching the actual numbers — same honest verdict, prettier
tickets. The per-position MAE just measures how well it fits the spread.

Requires the optional `ml` extras: `uv sync --extra ml`.
"""
from __future__ import annotations

import numpy as np

from analyze import load_draws
from config import Product, get_product
from ml.util import progress

WINDOWS = (20, 50)


def _feature_names(k: int) -> list[str]:
    names = []
    for i in range(k):
        names += [f"p{i+1}_last"]
        names += [f"p{i+1}_mean_w{w}" for w in WINDOWS]
        names += [f"p{i+1}_std_w50"]
    names.append("dow")
    return names


def _row_features(prior_sorted: np.ndarray, dow: int, k: int) -> np.ndarray:
    """Features from prior sorted draws (shape (t, k)) for predicting next."""
    t = prior_sorted.shape[0]
    feats = []
    for i in range(k):
        col = prior_sorted[:, i]
        feats.append(col[-1])                                  # last value
        for w in WINDOWS:
            feats.append(col[-w:].mean())                      # rolling mean
        feats.append(col[-50:].std() if t >= 2 else 0.0)       # rolling std
    feats.append(float(dow))
    return np.array(feats, dtype=np.float32)


def build_dataset(product: Product, draws=None, min_history: int = 50):
    draws = draws if draws is not None else load_draws(product)
    k = product.main_count
    from datetime import datetime
    sorted_all = np.array([sorted(d["main"]) for d in draws])
    X, Y, di = [], [], []
    for t in range(min_history, len(draws)):
        prior = sorted_all[:t]
        dow = datetime.fromisoformat(draws[t]["date"]).date().weekday()
        X.append(_row_features(prior, dow, k))
        Y.append(sorted_all[t])
        di.append(t)
    if not X:
        return (np.empty((0, len(_feature_names(k)))), np.empty((0, k)),
                np.array([]), _feature_names(k))
    return np.vstack(X), np.vstack(Y), np.array(di), _feature_names(k)


def _make_model(kind: str):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor
    if kind == "ridge":
        return Ridge(alpha=1.0)
    if kind == "gb":
        return MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                      learning_rate=0.05))
    raise ValueError("kind must be 'ridge' or 'gb'")


def train(X, Y, kind: str = "ridge"):
    m = _make_model(kind)
    m.fit(X, Y)
    return m


def _valid_ticket(vals, n: int, k: int) -> list[int]:
    """Round predictions to a strictly-increasing, distinct ticket in [1, n]."""
    v = [int(round(x)) for x in vals]
    v = [min(max(x, 1), n) for x in v]
    v.sort()
    for i in range(1, k):
        if v[i] <= v[i - 1]:
            v[i] = v[i - 1] + 1
    if v[-1] > n:                       # overflowed the top; push down
        v[-1] = n
        for i in range(k - 2, -1, -1):
            if v[i] >= v[i + 1]:
                v[i] = v[i + 1] - 1
    return [min(max(x, 1), n) for x in v]


def predict_next(product_name: str, kind: str = "ridge") -> dict:
    from datetime import datetime
    product = get_product(product_name)
    draws = load_draws(product)
    k, n = product.main_count, product.max_value
    X, Y, _, _ = build_dataset(product, draws)
    model = train(X, Y, kind)

    target = product.next_draw_date()
    sorted_all = np.array([sorted(d["main"]) for d in draws])
    xrow = _row_features(sorted_all, target.weekday(), k).reshape(1, -1)
    raw = model.predict(xrow)[0]
    ticket = _valid_ticket(raw, n, k)
    return {"product": product.label, "model": f"positional-{kind}",
            "target_date": target.isoformat(),
            "raw": [round(float(x), 1) for x in raw], "ticket": ticket}


def backtest(product_name: str, kind: str = "ridge",
             test_draws: int = 120, retrain_every: int = 20,
             min_history: int = 50) -> dict:
    product = get_product(product_name)
    draws = load_draws(product)
    k, n = product.main_count, product.max_value
    X, Y, di, _ = build_dataset(product, draws, min_history)
    if len(di) == 0:
        return {"product": product.label, "tested": 0}
    uniq = di
    test_idx = uniq[-test_draws:] if test_draws < len(uniq) else uniq

    hits_list = []
    mae = np.zeros(k)
    model = None
    for step, t in enumerate(test_idx):
        mask = di < t
        if mask.sum() == 0:
            continue
        if model is None or step % retrain_every == 0:
            model = train(X[mask], Y[mask], kind)
        row = X[di == t]
        pred = model.predict(row)[0]
        actual_sorted = Y[di == t][0]
        ticket = _valid_ticket(pred, n, k)
        hits_list.append(len(set(ticket).intersection(actual_sorted.tolist())))
        mae += np.abs(np.array(ticket) - actual_sorted)
        progress(step + 1, len(test_idx), f"positional {kind}")

    hits = np.array(hits_list, dtype=float)
    m = len(hits)
    # 95% bootstrap CI on mean hits, so a lucky spread doesn't look like signal
    rng = np.random.default_rng(0)
    boot = [hits[rng.integers(0, m, m)].mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    base = k * k / n
    return {
        "product": product.label, "model": f"positional-{kind}",
        "tested": m,
        "mean_hits": float(hits.mean()),
        "hits_lo": float(lo), "hits_hi": float(hi),
        "baseline_hits": base,
        "beats_baseline": bool(lo > base),
        "pos_mae": (mae / m).round(2).tolist(),
    }


def format_backtest(r: dict) -> str:
    if not r.get("tested"):
        return f"{r['product']}: not enough data."
    verdict = ("⚑ CI above baseline — investigate!" if r["beats_baseline"]
               else "within noise of random (CI spans the baseline)")
    lines = [
        f"{r['product']} — positional backtest ({r['model']}) over {r['tested']} draws",
        "",
        f"  mean hits (ordered ticket)  {r['mean_hits']:.3f}  "
        f"[95% CI {r['hits_lo']:.3f}, {r['hits_hi']:.3f}]",
        f"  random baseline             {r['baseline_hits']:.3f}   → {verdict}",
        f"  per-position MAE            {r['pos_mae']}",
        "",
        "  Per-position MAE shows the model learns each position's typical "
        "range; the hits CI spanning the baseline shows it still can't pick "
        "the actual numbers. Same honest verdict, different framing.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "power_655"
    print(format_backtest(backtest(name)))
