"""Per-number probability models (scikit-learn).

Two tabular models predict P(number appears in the next draw):
  - "logreg": StandardScaler + LogisticRegression (fast, interpretable)
  - "gb":     GradientBoostingClassifier (captures nonlinearities)

To turn probabilities into a ticket we rank all numbers by predicted
probability and take the top `main_count`. (Ranking, not thresholding,
guarantees exactly 6 numbers.)

Requires the optional `ml` dependencies: `uv sync --extra ml`.
"""
from __future__ import annotations

import numpy as np

MODELS = ("logreg", "gb", "rf")


def make_model(kind: str = "logreg"):
    from sklearn.ensemble import (GradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logreg":
        # No class_weight rebalancing: we want *calibrated* probabilities so
        # the Brier/log-loss comparison against the base rate is meaningful.
        # Ranking for the top-6 ticket is unaffected.
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0),
        )
    if kind == "gb":
        return GradientBoostingClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.05, subsample=0.8
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=20,
            n_jobs=-1, random_state=0
        )
    raise ValueError(f"Unknown model kind '{kind}'. Choices: {MODELS}")


def train(X: np.ndarray, y: np.ndarray, kind: str = "logreg"):
    model = make_model(kind)
    model.fit(X, y)
    return model


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """P(class=1) per row, robust to a model that saw only one class."""
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        return proba[:, classes.index(1)]
    return np.zeros(X.shape[0])


def rank_topk(probs: np.ndarray, numbers: np.ndarray, k: int):
    order = np.argsort(-probs)
    top = sorted(int(numbers[i]) for i in order[:k])
    prob_map = {int(numbers[i]): float(probs[i]) for i in range(len(numbers))}
    return top, prob_map
