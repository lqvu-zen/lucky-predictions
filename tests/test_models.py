"""Every model must return a valid ticket: 6 distinct, ascending, in range."""
import pytest

from config import get_product
from ml import joint, sampler


def _assert_valid(ticket, product):
    k = product.main_count
    assert len(ticket) == k
    assert ticket == sorted(ticket)          # ascending
    assert len(set(ticket)) == k             # distinct
    assert all(product.min_value <= n <= product.max_value for n in ticket)


@pytest.mark.parametrize("game", ["power_655", "power_645"])
def test_joint_ticket_valid(game):
    _assert_valid(joint.predict_next(game)["ticket"], get_product(game))


def test_joint_matches_closed_form():
    # the learned grid should recover the fixed order-statistic law
    p = get_product("power_655")
    diff = joint.max_abs_diff(joint.empirical_grid(p),
                              joint.closed_form_grid(p), p)
    assert diff < 0.05


@pytest.mark.parametrize("game", ["power_655", "power_645"])
def test_sampler_ticket_valid(game):
    _assert_valid(sampler.predict_next(game)["ticket"], get_product(game))


def test_ml_models_valid():
    pytest.importorskip("sklearn")
    from ml import chain, gap, positional
    p = get_product("power_655")
    _assert_valid(positional.predict_next("power_655")["ticket"], p)
    _assert_valid(gap.predict_next("power_655")["ticket"], p)
    _assert_valid(chain.predict_next("power_655")["ticket"], p)
