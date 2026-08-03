"""Independence between draws — the assumption the whole project rests on (#16)."""
import math
import random

import pytest

from config import get_product
from ml import independence


def test_pearson_matches_known_cases():
    assert independence._pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert independence._pearson([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)
    assert independence._pearson([1, 1, 1], [1, 2, 3]) == 0.0


def test_fisher_p_flags_a_strong_correlation():
    assert independence._fisher_p(0.0, 500) == pytest.approx(1.0)
    assert independence._fisher_p(0.9, 500) < 1e-6
    # a weak correlation over few points is not significant
    assert independence._fisher_p(0.1, 20) > 0.05


def test_detects_a_planted_dependency():
    """The test must FAIL to clear data that really is predictable.

    Build draws where position 1 copies the previous draw's position 1. If the
    module still reported independence, it would be worthless.
    """
    product = get_product("power_655")
    rng = random.Random(4)
    draws, first = [], 8
    for _ in range(400):
        # random walk on the smallest number => strong lag-1 autocorrelation,
        # while every draw still looks like a plausible ascending ticket
        first = max(1, min(15, first + rng.choice([-2, -1, 1, 2])))
        rest = rng.sample(range(20, 56), 5)
        draws.append({"main": sorted([first] + rest)})
    corr = independence.lag_correlations(product, draws, max_lag=3)
    assert corr["sidak_p"] < 0.05, "a planted dependency must be detected"
    assert abs(corr["worst"]["r"]) > 0.3, "the planted signal should be obvious"


def test_real_draws_look_independent():
    for name in ("power_655", "power_645"):
        r = independence.summary(name, max_lag=10)
        assert r["draws"] > 100
        # nothing should survive correction on genuinely random data
        assert r["corr"]["sidak_p"] > 0.01
        assert r["overlap_sidak_p"] > 0.01
        assert r["overlap_dist"]["p"] > 0.01


def test_overlap_mean_matches_hypergeometric_expectation():
    product = get_product("power_655")
    rows = independence.overlap_by_lag(product, _load(product), max_lag=5)
    expected = product.main_count ** 2 / product.max_value
    for row in rows:
        assert row["expected"] == pytest.approx(expected)
        assert abs(row["z"]) < 4, "no lag should deviate wildly"


def test_number_of_extreme_correlations_is_about_right():
    """~4.6% of tests should exceed 2 sigma — not 0, not half of them."""
    product = get_product("power_655")
    corr = independence.lag_correlations(product, _load(product), max_lag=10)
    n = corr["n_tests"]
    assert corr["n_beyond_2sigma"] <= max(3 * corr["expected_beyond_2sigma"], 3)
    assert n == product.main_count * 10


def test_normal_sf_is_a_two_sided_tail():
    assert independence._normal_sf(0.0) == pytest.approx(1.0)
    assert independence._normal_sf(1.96) == pytest.approx(0.05, abs=0.002)
    assert independence._normal_sf(-1.96) == pytest.approx(0.05, abs=0.002)


def _load(product):
    from analyze import load_draws
    return load_draws(product)
