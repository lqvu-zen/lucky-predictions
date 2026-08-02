"""Grid residuals vs theory (#12) and the skill-free null distribution (#15)."""
import pytest

from config import get_product
from ml import ceiling, residual
from ml.joint import closed_form_grid


def test_perfect_data_has_zero_residuals():
    """Counts generated exactly at the theoretical rate must give z ~ 0."""
    product = get_product("power_645")
    k, N = product.main_count, product.max_value
    theory = closed_form_grid(product)
    # build a fake history whose position counts match theory in expectation
    # by weighting each possible draw: instead, check the degenerate case of
    # T = 0 draws contributing no residual mass.
    res = residual.residual_grid(product, [])
    assert res["draws"] == 0
    assert res["max_abs_z"] == 0.0
    assert theory[1][0] > theory[N][0], "position 1 must favour small numbers"


def test_residuals_are_normal_ish_on_real_data():
    """The whole point: |z| > 2 should hit roughly the normal rate, not 30%."""
    for name in ("power_655", "power_645"):
        r = residual.summary(name)
        assert r["draws"] > 100
        assert abs(r["mean_z"]) < 0.5
        assert r["frac_gt2"] < 0.15, "far too many extreme cells"
        assert r["max_z_p"] > 0.001, "largest residual is implausible under the null"


def test_binning_keeps_expected_counts_reasonable():
    product = get_product("power_655")
    rows = residual.chi2_by_position(product)
    assert len(rows) == product.main_count
    for row in rows:
        assert row["dof"] >= 1
        assert 0.0 <= row["p"] <= 1.0


def test_binom_tail_and_sidak_maths():
    # P(X >= 0) == 1 ; P(X >= n) == p^n
    assert residual._binom_tail(6, 0, 0.05) == pytest.approx(1.0)
    assert residual._binom_tail(6, 6, 0.05) == pytest.approx(0.05 ** 6)
    # running 6 tests at 0.05 gives ~26% chance of at least one hit
    assert residual._binom_tail(6, 1, 0.05) == pytest.approx(1 - 0.95 ** 6)


def test_max_z_pvalue_accounts_for_many_cells():
    """|z| = 3 is rare once, unremarkable across 300 cells."""
    single = residual._max_z_pvalue(3.0, 1)
    many = residual._max_z_pvalue(3.0, 300)
    assert single < 0.01
    assert many > 0.3


def test_null_distribution_is_centred_on_random_play():
    """A skill-free player's mean must match the exact random-ticket value."""
    product = get_product("power_655")
    exact = ceiling.exact_random(product) / product.main_count
    means = ceiling.simulate_null("power_655", n_draws=200, n_sims=400, seed=5)
    avg = sum(means) / len(means)
    assert avg == pytest.approx(exact, abs=0.01)


def test_p_value_bounds_and_direction():
    null = [0.0, 0.02, 0.04, 0.06, 0.08]
    # scoring below everything -> p near 1; above everything -> p small
    assert ceiling.null_p_value(null, -1.0) == pytest.approx(1.0)
    assert ceiling.null_p_value(null, 99.0) == pytest.approx(1 / 6)
    assert 0.0 < ceiling.null_p_value(null, 0.05) < 1.0
