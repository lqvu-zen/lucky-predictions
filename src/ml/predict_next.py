"""Train on all history and predict (and log) the next scheduled draw."""
from __future__ import annotations

from datetime import datetime

from config import get_product
from ml import features as F
from ml import ledger
from ml import model as M


def predict_next(product_name: str, kind: str = "logreg", log: bool = True) -> dict:
    product = get_product(product_name)
    target = product.next_draw_date()
    X, y, _, _, _ = F.build_dataset(product, min_history=50)
    trained = M.train(X, y, kind)

    Xn, numbers, _ = F.build_next_features(product, target.weekday())
    probs = M.predict_proba(trained, Xn)
    top6, prob_map = M.rank_topk(probs, numbers, product.main_count)

    entry = {
        "game": product_name,
        "target_date": target.isoformat(),
        "model": kind,
        "version": datetime.now().replace(microsecond=0).isoformat(),
        "top6": top6,
        "probs": {str(k): round(v, 4) for k, v in sorted(prob_map.items())},
    }
    if log:
        ledger.append(entry)
    return entry
