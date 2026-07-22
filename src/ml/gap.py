"""Gap / spacing model — a position-based framing on the gaps between numbers.

Instead of the absolute value at each position, we model the **gaps** of a
sorted draw:  g1 = p1,  g2 = p2 - p1,  …,  g6 = p6 - p5  (every gap >= 1
because the numbers are distinct and ascending). One Ridge regressor per gap
predicts that gap from the history of gaps; a cumulative sum turns the
predicted gaps back into an ascending ticket.

Like the other position models it just learns each gap's average and predicts
a tidy, evenly-spaced ticket — honest verdict, different lens. Needs the ML
extras (`uv sync --extra ml`).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from analyze import load_draws
from config import Product, get_product
from ml.positional import _valid_ticket
from ml.util import progress

WINDOWS = (20, 50)


def _gaps(sorted_vals) -> np.ndarray:
    return np.diff(np.concatenate([[0], sorted_vals]))


def _row_features(prior_gaps: np.ndarray, dow: int, k: int) -> np.ndarray:
    feats = []
    for i in range(k):
        col = prior_gaps[:, i]
        feats.append(col[-1])
        for w in WINDOWS:
            feats.append(col[-w:].mean())
        feats.append(col[-50:].std() if prior_gaps.shape[0] >= 2 else 0.0)
    feats.append(float(dow))
    return np.array(feats, dtype=np.float32)


def build_dataset(product: Product, draws=None, min_history: int = 50):
    draws = draws if draws is not None else load_draws(product)
    k = product.main_count
    G = np.array([_gaps(sorted(d["main"])) for d in draws])
    X, Y, di = [], [], []
    for t in range(min_history, len(draws)):
        dow = datetime.fromisoformat(draws[t]["date"]).date().weekday()
        X.append(_row_features(G[:t], dow, k))
        Y.append(G[t])
        di.append(t)
    return np.vstack(X), np.vstack(Y), np.array(di)


def train(X, Y):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=1.0).fit(X, Y)   # Ridge handles multi-output natively


def _ticket_from_gaps(gaps, n: int, k: int) -> list[int]:
    cum, acc = [], 0.0
    for g in gaps:
        acc += max(float(g), 1.0)
        cum.append(acc)
    return _valid_ticket(cum, n, k)


def predict_next(product_name: str) -> dict:
    product = get_product(product_name)
    draws = load_draws(product)
    k, n = product.main_count, product.max_value
    X, Y, _ = build_dataset(product, draws)
    model = train(X, Y)
    target = product.next_draw_date()
    G = np.array([_gaps(sorted(d["main"])) for d in draws])
    xrow = _row_features(G, target.weekday(), k).reshape(1, -1)
    gaps = model.predict(xrow)[0]
    return {"product": product.label, "model": "gap-ridge",
            "target_date": target.isoformat(),
            "ticket": _ticket_from_gaps(gaps, n, k),
            "gaps": [round(float(g), 1) for g in gaps]}


def backtest(product_name: str, test_draws: int = 120, retrain_every: int = 20,
             min_history: int = 50) -> dict:
    product = get_product(product_name)
    draws = load_draws(product)
    k, n = product.main_count, product.max_value
    X, Y, di = build_dataset(product, draws, min_history)
    if len(di) == 0:
        return {"product": product.label, "tested": 0}
    test_idx = di[-test_draws:] if test_draws < len(di) else di
    hits, model = [], None
    for step, t in enumerate(test_idx):
        mask = di < t
        if mask.sum() == 0:
            continue
        if model is None or step % retrain_every == 0:
            model = train(X[mask], Y[mask])
        gaps = model.predict(X[di == t])[0]
        ticket = _ticket_from_gaps(gaps, n, k)
        actual = set(draws[int(t)]["main"][:k])
        hits.append(len(actual.intersection(ticket)))
        progress(step + 1, len(test_idx), "gap backtest")
    hits = np.array(hits, float)
    m = len(hits)
    rng = np.random.default_rng(0)
    boot = [hits[rng.integers(0, m, m)].mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    base = k * k / n
    return {"product": product.label, "model": "gap-ridge", "tested": m,
            "mean_hits": float(hits.mean()), "hits_lo": float(lo),
            "hits_hi": float(hi), "baseline_hits": base,
            "beats_baseline": bool(lo > base)}


def format_backtest(r: dict) -> str:
    if not r.get("tested"):
        return f"{r['product']}: not enough data."
    verdict = ("⚑ CI above baseline — investigate!" if r["beats_baseline"]
               else "within noise of random (CI spans the baseline)")
    return "\n".join([
        f"{r['product']} — gap/spacing backtest ({r['model']}) over {r['tested']} draws",
        "",
        f"  mean hits       {r['mean_hits']:.3f}  "
        f"[95% CI {r['hits_lo']:.3f}, {r['hits_hi']:.3f}]",
        f"  random baseline {r['baseline_hits']:.3f}   → {verdict}",
        "",
        "  Models the spacing between numbers; still can't pick the actual "
        "draw. Same honest verdict, another lens.",
    ])


if __name__ == "__main__":
    import sys
    print(format_backtest(backtest(sys.argv[1] if len(sys.argv) > 1 else "power_655")))
