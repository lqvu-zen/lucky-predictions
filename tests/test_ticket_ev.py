"""Expected value of a line — exact hypergeometric maths (#8)."""
from math import comb

import pytest

import ticket_ev
from config import PRODUCTS, get_product


def test_match_probabilities_form_a_distribution():
    """P(0..k matches) must sum to exactly 1."""
    for name in PRODUCTS:
        product = get_product(name)
        total = sum(ticket_ev.match_probability(product, j)
                    for j in range(product.main_count + 1))
        assert total == pytest.approx(1.0, abs=1e-12)


def test_jackpot_odds_are_one_over_combinations():
    for name in PRODUCTS:
        product = get_product(name)
        p = ticket_ev.match_probability(product, product.main_count)
        assert p == pytest.approx(1 / comb(product.max_value, product.main_count))


def test_known_values_for_655():
    """Spot-check against hand-computed odds."""
    product = get_product("power_655")
    assert ticket_ev.match_probability(product, 6) == pytest.approx(1 / 28_989_675)
    # 5 matches: C(6,5)*C(49,1)/C(55,6)
    assert ticket_ev.match_probability(product, 5) == pytest.approx(
        6 * 49 / 28_989_675)
    # 3 matches: C(6,3)*C(49,3)/C(55,6)
    assert ticket_ev.match_probability(product, 3) == pytest.approx(
        20 * comb(49, 3) / 28_989_675)


def test_expected_value_is_a_loss():
    """The house edge must show up as a negative EV for every game."""
    for name in PRODUCTS:
        s = ticket_ev.summary(name)
        assert s["net_ev"] < 0
        assert s["return_pct"] < 0
        assert s["gross_ev"] < s["ticket_cost"]


def test_impossible_match_counts_are_zero():
    product = get_product("power_655")
    assert ticket_ev.match_probability(product, -1) == 0.0
    assert ticket_ev.match_probability(product, product.main_count + 1) == 0.0


def test_ev_does_not_depend_on_the_numbers_chosen():
    """The lesson of the calculator, asserted: no ticket is special.

    match_probability takes no ticket argument at all, so any two lines share
    the same EV by construction — this test pins that property down so a future
    change can't quietly introduce number-dependent odds.
    """
    s = ticket_ev.summary("power_655")
    ev_from_tiers = sum(r["contribution"] for r in s["tiers"]) - s["ticket_cost"]
    assert ev_from_tiers == pytest.approx(s["net_ev"])
    import inspect
    src = inspect.signature(ticket_ev.match_probability).parameters
    assert "ticket" not in src and "numbers" not in src
