"""Optimal (DP) ticket assignment vs the greedy one."""
import pytest

from config import get_product
from ml.joint import closed_form_grid, empirical_grid, predict_ticket
from ml.optimal import expected_pos_hits, optimal_ticket


def _blank(product):
    k, N = product.main_count, product.max_value
    return [[0.0] * k for _ in range(N + 1)]


def test_dp_beats_greedy_on_an_adversarial_grid():
    """The whole point: greedy can steal a number a later position needed more.

    Position 0 slightly prefers number 2, but position 1 needs number 2 badly.
    Greedy grabs it at position 0 and wrecks position 1; the DP does not.
    """
    product = get_product("power_645")
    k, N = product.main_count, product.max_value
    grid = _blank(product)
    grid[1][0], grid[2][0] = 0.50, 0.51
    grid[2][1], grid[3][1] = 0.90, 0.05
    for p in range(2, k):           # later positions: force an ascending tail
        for v in range(10 + p, N + 1):
            grid[v][p] = 0.01

    greedy = predict_ticket(grid, product)
    best = optimal_ticket(grid, product)
    g_e = expected_pos_hits(grid, greedy, product)
    o_e = expected_pos_hits(grid, best, product)

    assert o_e > g_e, "DP must beat greedy when greedy is provably suboptimal"
    assert best[0] == 1 and best[1] == 2, f"expected 1,2 at the front, got {best}"


def test_optimal_is_ascending_and_distinct():
    for name in ("power_655", "power_645"):
        product = get_product(name)
        grid = empirical_grid(product, smoothing=0.5)
        t = optimal_ticket(grid, product)
        assert len(t) == product.main_count
        assert len(set(t)) == product.main_count
        assert t == sorted(t)
        assert all(1 <= v <= product.max_value for v in t)


def test_optimal_never_worse_than_greedy_on_real_grids():
    for name in ("power_655", "power_645"):
        product = get_product(name)
        for grid in (closed_form_grid(product),
                     empirical_grid(product, smoothing=0.5)):
            g = expected_pos_hits(grid, predict_ticket(grid, product), product)
            o = expected_pos_hits(grid, optimal_ticket(grid, product), product)
            assert o >= g - 1e-12


def test_matches_brute_force_on_a_small_case():
    """Exhaustive check of the DP against every ascending combination."""
    from itertools import combinations
    product = get_product("power_645")
    k, N = product.main_count, product.max_value
    import random
    rng = random.Random(7)
    grid = _blank(product)
    small = 12                      # restrict to 1..12 so brute force is cheap
    for v in range(1, small + 1):
        for p in range(k):
            grid[v][p] = rng.random()
    best_combo = max(combinations(range(1, small + 1), k),
                     key=lambda c: sum(grid[v][p] for p, v in enumerate(c)))
    brute = sum(grid[v][p] for p, v in enumerate(best_combo))
    dp = expected_pos_hits(grid, optimal_ticket(grid, product), product)
    assert dp == pytest.approx(brute, abs=1e-9)
