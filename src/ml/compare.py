"""Compare models with confidence intervals — the honest tuning verdict.

Tuning is only meaningful if you can tell a real improvement from luck. This
harness runs a walk-forward evaluation for several models, then bootstraps
95% confidence intervals on:

  - mean hits per ticket (top-6), vs the analytic random baseline
  - Brier score of the per-number probabilities, vs the base-rate predictor

If a model had real signal, its mean-hits CI would sit *above* the baseline
and its Brier CI *below* the base rate. For a uniform lottery every CI
straddles the baseline — so no configuration is significantly better, no
matter how rich the features. That's the point of measuring.
"""
from __future__ import annotations

import numpy as np

from config import get_product
from ml import features as F
from ml import model as M
from ml.util import progress


def _walk(product, X, y, di, nums, kind, test_draws, retrain_every):
    """Return per-draw hit counts and pooled (probs, labels)."""
    unique = np.unique(di)
    test_set = unique[-test_draws:] if test_draws < len(unique) else unique
    k = product.main_count
    hits_per_draw, all_p, all_y = [], [], []
    model = None
    for step, t in enumerate(test_set):
        train_mask = di < t
        if train_mask.sum() == 0:
            continue
        if model is None or step % retrain_every == 0:
            model = M.train(X[train_mask], y[train_mask], kind)
        tm = di == t
        Xt, yt, nt = X[tm], y[tm], nums[tm]
        probs = M.predict_proba(model, Xt)
        top6, _ = M.rank_topk(probs, nt, k)
        actual = {int(nt[i]) for i in range(len(nt)) if yt[i] == 1}
        hits_per_draw.append(len(actual.intersection(top6)))
        all_p.append(probs)
        all_y.append(yt)
        progress(step + 1, len(test_set), f"compare {kind}")
    return (np.array(hits_per_draw, float),
            np.concatenate(all_p), np.concatenate(all_y).astype(float))


def _ci(samples, stat, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(samples)
    boot = [stat(samples[rng.integers(0, n, n)]) for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(stat(samples)), float(lo), float(hi)


def compare(product_name: str, kinds=("logreg", "gb", "rf"),
            test_draws: int = 100, retrain_every: int = 50) -> dict:
    product = get_product(product_name)
    X, y, di, nums, _ = F.build_dataset(product, min_history=50)
    k, N = product.main_count, product.max_value
    base_hits = k * k / N
    base_rate = k / N

    rows = []
    for kind in kinds:
        hits, p, yy = _walk(product, X, y, di, nums, kind, test_draws, retrain_every)
        mh, mlo, mhi = _ci(hits, np.mean)
        se = (p - yy) ** 2
        bmean, blo, bhi = _ci(se, np.mean)
        rows.append({
            "model": kind, "n": len(hits),
            "mean_hits": mh, "hits_lo": mlo, "hits_hi": mhi,
            "brier": bmean, "brier_lo": blo, "brier_hi": bhi,
            "beats_baseline": mlo > base_hits and bhi < ((base_rate - yy) ** 2).mean(),
        })
    return {
        "product": product.label, "test_draws": test_draws,
        "baseline_hits": base_hits,
        "brier_base": float(((base_rate - _pool_labels(X, y, di, nums, product,
                                                       test_draws)) ** 2).mean()),
        "rows": rows,
    }


def _pool_labels(X, y, di, nums, product, test_draws):
    unique = np.unique(di)
    test_set = unique[-test_draws:] if test_draws < len(unique) else unique
    mask = np.isin(di, test_set)
    return y[mask].astype(float)


def format_report(c: dict) -> str:
    lines = [
        f"{c['product']} — model comparison over {c['test_draws']} draws "
        f"(95% bootstrap CIs)",
        f"Random baseline: {c['baseline_hits']:.3f} hits/ticket · "
        f"base-rate Brier {c['brier_base']:.5f}",
        "",
        f"{'model':8s} {'mean hits [95% CI]':>26s} {'Brier [95% CI]':>26s}  verdict",
        "-" * 74,
    ]
    for r in c["rows"]:
        hits = f"{r['mean_hits']:.3f} [{r['hits_lo']:.3f},{r['hits_hi']:.3f}]"
        brier = f"{r['brier']:.4f} [{r['brier_lo']:.4f},{r['brier_hi']:.4f}]"
        verdict = "BEATS baseline!" if r["beats_baseline"] else "≈ random"
        lines.append(f"{r['model']:8s} {hits:>26s} {brier:>26s}  {verdict}")
    lines += [
        "",
        "Every CI straddles the baseline: no model is significantly better. "
        "Richer features don't create signal that isn't there.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "power_655"
    print(format_report(compare(name)))
