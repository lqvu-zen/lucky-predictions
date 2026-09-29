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


@pytest.mark.parametrize("kind", ["logreg", "hgb", "mlp"])
def test_classifier_tickets_valid(kind):
    pytest.importorskip("sklearn")
    from ml import perpos
    p = get_product("power_645")
    _assert_valid(perpos.predict_next("power_645", kind=kind)["ticket"], p)


def test_joint_ticket_is_near_the_ceiling():
    # the joint grid's ticket should be (close to) the law's optimum
    from ml.decode import ceiling_score, law_pos_score
    p = get_product("power_655")
    t = joint.predict_next("power_655")["ticket"]
    assert law_pos_score(t, p) >= 0.97 * ceiling_score(p)
