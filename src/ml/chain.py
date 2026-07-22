"""Conditional / autoregressive positional model.

Predict the smallest number p1 from history, then p2 conditioned on the
predicted p1, then p3 on p2, and so on — each ordered position depends on the
one before it, which naturally respects the ascending order. Six Ridge
regressors; regressor i sees the base history features plus the earlier
positions of the same draw.

Because the draw is random the conditional means still collapse to the usual
tidy spread, so it lands on the baseline like the rest — a more sophisticated
framing, same honest result. Needs the ML extras (`uv sync --extra ml`).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from analyze import load_draws
from config import Product, get_product
from ml.positional import _row_features, _valid_ticket
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
    """One regressor per position; position i also sees positions 0..i-1."""
    from sklearn.linear_model import Ridge
    models = []
    for i in range(k):
        Xi = np.hstack([Xbase, Y[:, :i]]) if i > 0 else Xbase
        models.append(Ridge(alpha=1.0).fit(Xi, Y[:, i]))
    return models


def _predict_row(models, xbase_row, n, k):
    prev = []
    for i in range(k):
        xi = np.hstack([xbase_row, np.array(prev, dtype=float)]) if i > 0 else xbase_row
        prev.append(float(models[i].predict(xi.reshape(1, -1))[0]))
    return _valid_ticket(prev, n, k), prev


def predict_next(product_name: str) -> dict:
    product = get_product(product_name)
    draws = load_draws(product)
    k, n = product.main_count, product.max_value
    Xb, Y, _ = build_base(product, draws)
    models = train(Xb, Y, k)
    target = product.next_draw_date()
    S = np.array([sorted(d["main"]) for d in draws])
    xrow = _row_features(S, target.weekday(), k)
    ticket, raw = _predict_row(models, xrow, n, k)
    return {"product": product.label, "model": "chain-ridge",
            "target_date": target.isoformat(), "ticket": ticket,
            "raw": [round(v, 1) for v in raw]}


def backtest(product_name: str, test_draws: int = 120, retrain_every: int = 20,
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
        ticket, _ = _predict_row(models, Xb[di == t][0], n, k)
        actual = set(draws[int(t)]["main"][:k])
        hits.append(len(actual.intersection(ticket)))
        progress(step + 1, len(test_idx), "chain backtest")
    hits = np.array(hits, float)
    m = len(hits)
    rng = np.random.default_rng(0)
    boot = [hits[rng.integers(0, m, m)].mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    base = k * k / n
    return {"product": product.label, "model": "chain-ridge", "tested": m,
            "mean_hits": float(hits.mean()), "hits_lo": float(lo),
            "hits_hi": float(hi), "baseline_hits": base,
            "beats_baseline": bool(lo > base)}


def format_backtest(r: dict) -> str:
    if not r.get("tested"):
        return f"{r['product']}: not enough data."
    verdict = ("⚑ CI above baseline — investigate!" if r["beats_baseline"]
               else "within noise of random (CI spans the baseline)")
    return "\n".join([
        f"{r['product']} — conditional/autoregressive backtest ({r['model']}) "
        f"over {r['tested']} draws",
        "",
        f"  mean hits       {r['mean_hits']:.3f}  "
        f"[95% CI {r['hits_lo']:.3f}, {r['hits_hi']:.3f}]",
        f"  random baseline {r['baseline_hits']:.3f}   → {verdict}",
        "",
        "  Each position is predicted from the previous one; still can't pick "
        "the actual draw. Same honest verdict, richer framing.",
    ])


if __name__ == "__main__":
    import sys
    print(format_backtest(backtest(sys.argv[1] if len(sys.argv) > 1 else "power_655")))
