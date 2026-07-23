from bankroll import simulate
from randomness import _chi2_sf, summary


def test_chi2_sf_in_range_and_monotone():
    assert 0.0 <= _chi2_sf(50, 54) <= 1.0
    # larger statistic -> smaller survival probability
    assert _chi2_sf(80, 54) < _chi2_sf(40, 54)


def test_uniformity_pvalue_valid():
    s = summary("power_655")
    assert 0.0 <= s["uniformity"]["p"] <= 1.0


def test_bankroll_loses_money():
    # small window keeps it fast; every strategy should end negative
    b = simulate("power_655", warmup=1150)
    assert b["draws"] > 0
    assert all(v["net"] < 0 for v in b["totals"].values())
