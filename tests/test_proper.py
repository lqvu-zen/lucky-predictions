"""Proper scoring rules (log-loss / Brier) over the number×position grid."""
from math import log

import pytest

from config import get_product
from ml import proper
from ml.joint import closed_form_grid


def test_uniform_logloss_is_log_n():
    """A model that knows nothing must score exactly ln(N) per position."""
    product = get_product("power_655")
    grid = proper.uniform_grid(product)
    actual = [1, 2, 3, 4, 5, 6]
    ll, _ = proper._score_draw(grid, actual, product)
    assert ll == pytest.approx(log(product.max_value), abs=1e-9)


def test_entropy_floor_below_uniform():
    """The true law is more informative than uniform, so its entropy is lower."""
    product = get_product("power_655")
    floor = proper.entropy_floor(product)
    assert 0 < floor < log(product.max_value)


def test_closed_form_columns_are_distributions():
    product = get_product("power_655")
    g = closed_form_grid(product)
    N, k = product.max_value, product.main_count
    for pos in range(k):
        col_sum = sum(g[v][pos] for v in range(1, N + 1))
        assert col_sum == pytest.approx(1.0, abs=1e-9)


def test_confident_beats_vague():
    """Log-loss must reward putting mass on what actually happened."""
    product = get_product("power_645")
    N, k = product.max_value, product.main_count
    actual = [2, 10, 20, 30, 40, 44]
    vague = proper.uniform_grid(product)
    sharp = [[1e-6] * k for _ in range(N + 1)]
    for pos, v in enumerate(actual):
        sharp[v][pos] = 1.0
    ll_sharp, br_sharp = proper._score_draw(sharp, actual, product)
    ll_vague, br_vague = proper._score_draw(vague, actual, product)
    assert ll_sharp < ll_vague
    assert br_sharp < br_vague


def test_shrunk_grid_is_a_distribution():
    product = get_product("power_655")
    draws = [{"main": [1, 2, 3, 4, 5, 6]}, {"main": [10, 20, 30, 40, 50, 55]}]
    g = proper.shrunk_grid(product, draws, strength=5.0)
    N, k = product.max_value, product.main_count
    for pos in range(k):
        assert sum(g[v][pos] for v in range(1, N + 1)) == pytest.approx(1.0, abs=1e-9)


def test_evaluate_runs_and_ranks_sensibly():
    """End-to-end: uniform must be the worst model on real history."""
    r = proper.evaluate("power_655", test_draws=20)
    assert r["tested"] > 0
    models = r["models"]
    worst = max(models.items(), key=lambda kv: kv[1]["logloss"])[0]
    assert worst == "uniform"
    # and nothing should be significantly under the irreducible floor
    assert not any(m["beats_floor"] for m in models.values())
