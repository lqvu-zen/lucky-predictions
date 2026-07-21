"""Feature importance — which signals the model leans on.

Two views:
  - Built-in importances: tree models expose `feature_importances_`;
    logistic regression exposes standardized coefficients (|coef|, since the
    pipeline scales inputs first, so they're comparable).
  - Permutation importance (optional, model-agnostic): shuffle each feature
    and measure how much the Brier score worsens on a holdout.

Honest caveat: a feature having high "importance" only means the model USES
it to fit the training data — not that it predicts future draws. The
evaluation harness already shows the models don't beat the baseline, so
these rankings describe internal structure, not a real edge.
"""
from __future__ import annotations

import numpy as np

from config import get_product
from ml import features as F
from ml import model as M


def _final_estimator(fitted):
    if hasattr(fitted, "named_steps"):
        return list(fitted.named_steps.values())[-1]
    return fitted


def _builtin_importances(fitted) -> np.ndarray | None:
    est = _final_estimator(fitted)
    if hasattr(est, "feature_importances_"):
        return np.asarray(est.feature_importances_, dtype=float)
    if hasattr(est, "coef_"):
        return np.abs(np.asarray(est.coef_, dtype=float)).ravel()
    return None


def importances(product_name: str, kind: str = "rf",
                permutation: bool = False, sample: int = 6000) -> dict:
    product = get_product(product_name)
    X, y, di, nums, names = F.build_dataset(product, min_history=50)
    fitted = M.train(X, y, kind)

    raw = _builtin_importances(fitted)
    if raw is None:
        raw = np.zeros(len(names))
    total = raw.sum() or 1.0
    builtin = raw / total

    perm = None
    if permutation:
        from sklearn.inspection import permutation_importance
        # use the most recent `sample` rows to keep it fast
        Xs, ys = X[-sample:], y[-sample:]
        r = permutation_importance(fitted, Xs, ys, n_repeats=5,
                                   random_state=0, scoring="neg_brier_score")
        perm = r.importances_mean

    ranked = sorted(
        [{"feature": names[i], "importance": float(builtin[i]),
          "perm": (float(perm[i]) if perm is not None else None)}
         for i in range(len(names))],
        key=lambda d: -d["importance"],
    )
    return {"product": product.label, "model": kind, "features": ranked}


def format_report(r: dict) -> str:
    lines = [
        f"{r['product']} — feature importance ({r['model']})",
        "",
        f"{'feature':14s} {'importance':>11s}   share",
    ]
    lines.append("-" * 46)
    for f in r["features"]:
        bar = "█" * max(1, round(f["importance"] * 40))
        extra = ""
        if f["perm"] is not None:
            extra = f"   perm Δbrier {f['perm']:+.5f}"
        lines.append(f"{f['feature']:14s} {f['importance']:>11.3f}   {bar}{extra}")
    lines += [
        "",
        "Note: importance = how much the model USES a feature to fit history, "
        "not evidence it predicts the future. The models still match the "
        "random baseline (see ml-backtest / ml-compare).",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "power_655"
    print(format_report(importances(name)))
