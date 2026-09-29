"""Positional (ordered) model — an alternative framing of the question.

Instead of asking "for each number, what's the probability it appears?",
we sort each draw's 6 numbers ascending (p1 < p2 < ... < p6) and ask:

    what value will appear at each ORDERED POSITION of the next draw?

So we train 6 regressors, one per position, each predicting that
position's value from history, then assemble a valid ascending ticket.

Why this is interesting: the ordered positions have real structure — p1 is
almost always small, p6 almost always large (these are "order statistics").
So this model learns those marginal distributions. But *which* value lands
within each position's range on a given draw is still random, so it does NOT
beat the baseline at matching the actual numbers. The per-position MAE just
measures how well it fits the spread.

Decoding: a regressor predicts each position's *mean*, but the position score
rewards the *most likely* value, and the two differ (6/55 position 1: mean ~8,
mode 1). So the ticket comes from `ml.decode`: prediction + out-of-fold
residuals -> a per-position distribution -> the optimal ascending ticket.
Rounding the mean instead left this model at ~69% of the ceiling.

Requires the optional `ml` extras: `uv sync --extra ml`.
"""
from __future__ import annotations

import numpy as np

from analyze import load_draws
from config import Product, get_product
from ml import decode
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


def make_ridge():
    """Scaled ridge whose strength is picked by (efficient leave-one-out) CV.

    With a hand-set alpha=1 on unscaled features the ridge chased noise, so its
    predicted means wobbled draw to draw. Letting CV choose is the honest
    setting: if the features carry no signal, CV picks heavy shrinkage by itself.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         RidgeCV(alphas=np.logspace(-2, 6, 17)))


def _make_model(kind: str):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    if kind == "ridge":
        return make_ridge()
    if kind == "gb":
        # early stopping on a held-out 20%: trees are added only while they
        # help unseen draws, so the booster stops fitting noise on its own
        return MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                      learning_rate=0.05, subsample=0.8,
                                      validation_fraction=0.2,
                                      n_iter_no_change=10, random_state=0))
    raise ValueError("kind must be 'ridge' or 'gb'")


def train(X, Y, kind: str = "ridge"):
    m = _make_model(kind)
    m.fit(X, Y)
    return m


def train_with_residuals(X, Y, kind: str = "ridge"):
    """(model, per-position out-of-fold residuals) for mode decoding."""
    res = decode.residuals_from(lambda a, b: train(a, b, kind),
                                lambda m, a: m.predict(a), X, Y)
    return train(X, Y, kind), res


def predict_next(product_name: str, kind: str = "ridge") -> dict:
    from datetime import datetime
    product = get_product(product_name)
    draws = load_draws(product)
    k, n = product.main_count, product.max_value
    X, Y, _, _ = build_dataset(product, draws)
    model, res = train_with_residuals(X, Y, kind)

    target = product.next_draw_date()
    sorted_all = np.array([sorted(d["main"]) for d in draws])
    xrow = _row_features(sorted_all, target.weekday(), k).reshape(1, -1)
    raw = model.predict(xrow)[0]
    ticket = decode.decode(raw, res, product)
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

    hits_list, tickets, actuals = [], [], []
    mae = np.zeros(k)
    model = None
    for step, t in enumerate(test_idx):
        mask = di < t
        if mask.sum() == 0:
            continue
        if model is None or step % retrain_every == 0:
            model, res = train_with_residuals(X[mask], Y[mask], kind)
        row = X[di == t]
        pred = model.predict(row)[0]
        actual_sorted = Y[di == t][0]
        ticket = decode.decode(pred, res, product)
        hits_list.append(len(set(ticket).intersection(actual_sorted.tolist())))
        tickets.append(ticket)
        actuals.append(actual_sorted.tolist())
        # MAE of the mean prediction — what the regressor itself is fit to
        mae += np.abs(np.array(pred) - actual_sorted)
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
        **decode.pos_summary(tickets, actuals, product),
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
        *decode.format_pos(r),
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
