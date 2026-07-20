"""Leakage-safe feature engineering for per-number appearance prediction.

We reframe "predict the next 6 numbers" as N independent binary questions:
for each number k in 1..N, what is P(k appears in the next draw)?

For every historical draw t we emit one row per number, computed using
ONLY the draws before t (strictly no look-ahead), labelled 1 if that number
appeared in draw t. A single forward pass keeps incremental state (counts,
gaps, rolling windows), so building the whole dataset is O(draws * N).

Features per (draw t, number k), all derived from draws[:t]:
    rate_all      appearances / draws_so_far
    rate_w10/30/50 appearances in last W draws / window size
    draws_since   draws since k last appeared (capped)
    gap_mean      mean gap between appearances so far
    gap_ratio     draws_since / gap_mean  (>1 => "overdue")
    appeared_last 1 if k was in the immediately previous draw
    value_norm    k / N
    is_odd        k % 2
    dow           weekday of the TARGET draw (0=Mon..6=Sun) — a known covariate
"""
from __future__ import annotations

from collections import deque
from datetime import datetime

import numpy as np

from analyze import load_draws
from config import Product

WINDOWS = (10, 30, 50, 100)
EMA_DECAY = 0.98
FEATURE_NAMES = (
    ["rate_all"]
    + [f"rate_w{w}" for w in WINDOWS]
    + ["ema_rate", "pair_lift", "draws_since", "gap_mean", "gap_ratio",
       "appeared_last", "value_norm", "is_odd", "dow"]
)


class _State:
    """Incremental history state for numbers 1..N (index 0 unused)."""

    def __init__(self, product: Product):
        self.p = product
        n = product.max_value
        self.N = n
        self.count = np.zeros(n + 1)
        self.last_seen = np.full(n + 1, -1)     # draw index of last appearance
        self.gap_sum = np.zeros(n + 1)
        self.gap_num = np.zeros(n + 1)
        self.prev_set: set[int] = set()
        self.t = 0                               # draws processed so far
        self.win = {w: deque() for w in WINDOWS}
        self.wcount = {w: np.zeros(n + 1) for w in WINDOWS}
        self.ema = np.zeros(n + 1)               # exp-decayed appearance rate
        self.cooc = np.zeros((n + 1, n + 1))     # co-occurrence counts

    def features(self, dow: int) -> np.ndarray:
        """Feature matrix for all numbers, given the target draw's weekday."""
        n = self.N
        t = max(self.t, 1)
        nums = np.arange(1, n + 1)
        rate_all = self.count[1:] / t
        rows = [rate_all]
        for w in WINDOWS:
            denom = max(len(self.win[w]), 1)
            rows.append(self.wcount[w][1:] / denom)
        # exp-decayed rate (recency-weighted frequency)
        rows.append(self.ema[1:].copy())
        # pair lift: how often each number co-occurs with the previous draw's
        # numbers, normalised by how often those numbers appear overall
        prev = list(self.prev_set)
        if prev:
            sub = self.cooc[1:, prev]                       # (N, |prev|)
            denom = np.maximum(self.count[prev], 1.0)
            pair_lift = (sub / denom).mean(axis=1)
        else:
            pair_lift = np.zeros(n)
        rows.append(pair_lift)
        last = self.last_seen[1:]
        draws_since = np.where(last < 0, t, self.t - 1 - last).astype(float)
        gap_mean = np.where(self.gap_num[1:] > 0,
                            self.gap_sum[1:] / np.maximum(self.gap_num[1:], 1),
                            float(t))
        gap_ratio = draws_since / np.maximum(gap_mean, 1e-6)
        appeared_last = np.array([1.0 if k in self.prev_set else 0.0 for k in nums])
        value_norm = nums / n
        is_odd = (nums % 2).astype(float)
        dow_col = np.full(n, float(dow))
        rows += [draws_since, gap_mean, gap_ratio, appeared_last,
                 value_norm, is_odd, dow_col]
        return np.column_stack(rows).astype(np.float32)

    def update(self, draw_numbers) -> None:
        s = set(draw_numbers)
        for k in s:
            if self.last_seen[k] >= 0:
                gap = self.t - self.last_seen[k]
                self.gap_sum[k] += gap
                self.gap_num[k] += 1
            self.last_seen[k] = self.t
            self.count[k] += 1
        # rolling windows
        for w in WINDOWS:
            self.win[w].append(s)
            for k in s:
                self.wcount[w][k] += 1
            if len(self.win[w]) > w:
                old = self.win[w].popleft()
                for k in old:
                    self.wcount[w][k] -= 1
        # exp-decayed rate
        self.ema *= EMA_DECAY
        for k in s:
            self.ema[k] += (1 - EMA_DECAY)
        # co-occurrence
        sl = list(s)
        for i, a in enumerate(sl):
            for b in sl[i + 1:]:
                self.cooc[a, b] += 1
                self.cooc[b, a] += 1
        self.prev_set = s
        self.t += 1


def _dow(date_str: str) -> int:
    return datetime.fromisoformat(date_str).date().weekday()


def build_dataset(product: Product, draws=None, min_history: int = 50):
    """Return (X, y, draw_index, numbers, feature_names).

    Rows are emitted for draws from index `min_history` onward.
    """
    draws = draws if draws is not None else load_draws(product)
    st = _State(product)
    X, y, di, nums_col = [], [], [], []
    n = product.max_value
    for t, d in enumerate(draws):
        if t >= min_history:
            feats = st.features(_dow(d["date"]))
            actual = set(d["main"])
            labels = np.array([1 if k in actual else 0 for k in range(1, n + 1)])
            X.append(feats)
            y.append(labels)
            di.append(np.full(n, t))
            nums_col.append(np.arange(1, n + 1))
        st.update(d["main"])
    if not X:
        empty = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        return empty, np.array([]), np.array([]), np.array([]), list(FEATURE_NAMES)
    return (np.vstack(X), np.concatenate(y), np.concatenate(di),
            np.concatenate(nums_col), list(FEATURE_NAMES))


def build_next_features(product: Product, target_dow: int, draws=None):
    """Feature matrix (N rows) for the upcoming draw, using ALL history."""
    draws = draws if draws is not None else load_draws(product)
    st = _State(product)
    for d in draws:
        st.update(d["main"])
    numbers = np.arange(1, product.max_value + 1)
    return st.features(target_dow), numbers, list(FEATURE_NAMES)
