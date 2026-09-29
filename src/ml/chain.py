"""Conditional / autoregressive positional model.

Predict the smallest number p1 from history, then p2 conditioned on the
predicted p1, then p3 on p2, and so on — each ordered position depends on the
one before it, which naturally respects the ascending order. Six Ridge
regressors; regressor i sees the base history features plus the earlier
positions of the same draw.

Because the draw is random the conditional means still collapse to the usual
tidy spread, so it lands on the baseline like the rest — a more sophisticated
framing, same honest result. The ticket is mode-decoded from the predicted
means + out-of-fold residuals (see `ml.decode`). Needs the ML extras
(`uv sync --extra ml`).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from analyze import load_draws
from config import Product, get_product
from ml import decode
from ml.positional import _row_features, make_ridge
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
    models = []
    for i in range(k):
        Xi = np.hstack([Xbase, Y[:, :i]]) if i > 0 else Xbase
        models.append(make_ridge().fit(Xi, Y[:, i]))
    return models


def predict_mu(models, Xbase):
    """Chained means for each row: position i sees the predicted 0..i-1."""
    prev = np.zeros((len(Xbase), 0))
    for m in models:
        col = m.predict(np.hstack([Xbase, prev]))
        prev = np.hstack([prev, col.reshape(-1, 1)])
    return prev


def train_with_residuals(Xbase, Y, k):
    res = decode.residuals_from(lambda a, b: train(a, b, k), predict_mu, Xbase, Y)
    return train(Xbase, Y, k), res


def predict_next(product_name: str) -> dict:
    product = get_product(product_name)
    draws = load_draws(product)
    k, n = product.main_count, product.max_value
    Xb, Y, _ = build_base(product, draws)
    models, res = train_with_residuals(Xb, Y, k)
    target = product.next_draw_date()
    S = np.array([sorted(d["main"]) for d in draws])
    xrow = _row_features(S, target.weekday(), k)
    raw = [float(v) for v in predict_mu(models, xrow.reshape(1, -1))[0]]
    ticket = decode.decode(raw, res, product)
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
    hits, tickets, actuals, models = [], [], [], None
    for step, t in enumerate(test_idx):
        mask = di < t
        if mask.sum() == 0:
            continue
        if models is None or step % retrain_every == 0:
            models, res = train_with_residuals(Xb[mask], Y[mask], k)
        ticket = decode.decode(predict_mu(models, Xb[di == t])[0], res, product)
        actual = draws[int(t)]["main"][:k]
        hits.append(len(set(actual).intersection(ticket)))
        tickets.append(ticket)
        actuals.append(actual)
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
            "beats_baseline": bool(lo > base),
            **decode.pos_summary(tickets, actuals, product)}


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
        *decode.format_pos(r),
        "",
        "  Each position is predicted from the previous one; still can't pick "
        "the actual draw. Same honest verdict, richer framing.",
    ])


if __name__ == "__main__":
    import sys
    print(format_backtest(backtest(sys.argv[1] if len(sys.argv) > 1 else "power_655")))
