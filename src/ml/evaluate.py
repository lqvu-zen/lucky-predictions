"""Walk-forward evaluation of the ML models — the honest scoreboard.

Procedure (no look-ahead):
  1. Build the full leakage-safe dataset once.
  2. Step through the last `test_draws` draws. For each test draw t, train
     on every row from draws before t (retraining every `retrain_every`
     draws to keep it fast), predict the N per-number probabilities for t,
     take the top-6 as the ticket, and score it against the real draw.

Metrics:
  - mean_hits      average matched numbers per ticket (top-6)
  - baseline_hits  analytic random expectation = k*k/N (0.655 / 0.800)
  - brier          mean squared error of per-number probabilities (lower=better)
  - brier_base     same metric for the trivial "everyone = k/N" predictor
  - logloss        cross-entropy of the probabilities
  - big_wins       test tickets with >= 4 matches

If the probabilities carried real signal, `brier` would beat `brier_base`
and `mean_hits` would beat `baseline_hits`. For a uniform lottery they won't
— that's the expected, honest result.
"""
from __future__ import annotations

import numpy as np

from config import get_product
from ml import features as F
from ml import model as M
from ml.util import progress


def _logloss(p, y, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def walk_forward(product_name: str, kind: str = "logreg",
                 test_draws: int = 120, retrain_every: int = 15,
                 min_history: int = 50) -> dict:
    product = get_product(product_name)
    X, y, di, nums, _ = F.build_dataset(product, min_history=min_history)
    if len(y) == 0:
        return {"product": product.label, "tested": 0}

    unique_draws = np.unique(di)
    test_set = unique_draws[-test_draws:] if test_draws < len(unique_draws) else unique_draws
    k = product.main_count
    N = product.max_value
    base_rate = k / N

    hit_sum = big = 0
    all_p, all_y = [], []
    model = None
    for step, t in enumerate(test_set):
        train_mask = di < t
        if train_mask.sum() == 0:
            continue
        if model is None or step % retrain_every == 0:
            model = M.train(X[train_mask], y[train_mask], kind)
        test_mask = di == t
        Xt, yt, nt = X[test_mask], y[test_mask], nums[test_mask]
        probs = M.predict_proba(model, Xt)
        top6, _ = M.rank_topk(probs, nt, k)
        actual = set(int(nt[i]) for i in range(len(nt)) if yt[i] == 1)
        hits = len(actual.intersection(top6))
        hit_sum += hits
        big += 1 if hits >= 4 else 0
        all_p.append(probs)
        all_y.append(yt)
        progress(step + 1, len(test_set), f"{kind} backtest")

    n_tested = len(test_set)
    p = np.concatenate(all_p)
    yy = np.concatenate(all_y).astype(float)
    return {
        "product": product.label,
        "model": kind,
        "tested": n_tested,
        "mean_hits": hit_sum / n_tested,
        "baseline_hits": k * k / N,
        "brier": float(np.mean((p - yy) ** 2)),
        "brier_base": float(np.mean((base_rate - yy) ** 2)),
        "logloss": _logloss(p, yy),
        "logloss_base": _logloss(np.full_like(yy, base_rate), yy),
        "big_wins": big,
    }


def format_report(r: dict) -> str:
    if not r.get("tested"):
        return f"{r['product']}: not enough data."
    d_hits = r["mean_hits"] - r["baseline_hits"]
    d_brier = r["brier"] - r["brier_base"]
    lines = [
        f"{r['product']} — ML backtest ({r['model']}) over {r['tested']} draws",
        "",
        f"  mean hits (top-6)     {r['mean_hits']:.3f}   "
        f"(random baseline {r['baseline_hits']:.3f}, diff {d_hits:+.3f})",
        f"  Brier score           {r['brier']:.5f}   "
        f"(base-rate {r['brier_base']:.5f}, diff {d_brier:+.5f})",
        f"  log-loss              {r['logloss']:.5f}   "
        f"(base-rate {r['logloss_base']:.5f})",
        f"  tickets with 4+ hits  {r['big_wins']} / {r['tested']}",
        "",
    ]
    edge = d_hits > 0.05 and d_brier < -0.0002
    if edge:
        lines.append("  ⚑ Model appears to beat baseline — investigate for leakage!")
    else:
        lines.append("  Verdict: on par with random, as expected. No usable signal.")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "power_655"
    print(format_report(walk_forward(name)))
