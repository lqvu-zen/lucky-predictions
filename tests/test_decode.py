"""Mode decoding, consensus voting and the CV-chosen shrinkage strength."""
import random

import pytest

from config import get_product
from ml.decode import (ceiling_score, consensus_ticket, law_pos_score,
                       residual_grid)
from ml.joint import closed_form_grid
from ml.proper import choose_strength, strength_losses


def _valid(ticket, product):
    k = product.main_count
    return (len(ticket) == k and ticket == sorted(ticket)
            and len(set(ticket)) == k
            and all(1 <= v <= product.max_value for v in ticket))


def test_mode_beats_mean_for_the_position_score():
    """The finding behind the decoding change: rounding each position's mean
    scores worse than the law's optimal (mode) ticket."""
    p = get_product("power_655")
    mean_ticket = [8, 16, 24, 32, 40, 48]    # E[X_(i)] = i*(N+1)/(k+1)
    assert law_pos_score(mean_ticket, p) < 0.75 * ceiling_score(p)


def test_residual_grid_columns_are_distributions():
    pytest.importorskip("numpy")
    p = get_product("power_645")
    rng = random.Random(0)
    res = [[rng.gauss(0, 6) for _ in range(300)] for _ in range(p.main_count)]
    grid = residual_grid([5, 12, 20, 27, 34, 41], res, p)
    for pos in range(p.main_count):
        col = sum(grid[v][pos] for v in range(1, p.max_value + 1))
        assert col == pytest.approx(1.0, abs=1e-9)


def test_decoding_finds_the_skewed_mode():
    """Mean ~8 plus residuals drawn from the real position-1 law must decode to
    1 — the mode — not to the rounded mean."""
    pytest.importorskip("numpy")
    p = get_product("power_655")
    q = closed_form_grid(p)
    rng = random.Random(1)
    values = list(range(1, p.max_value + 1))
    res = []
    for pos in range(p.main_count):
        col = [q[v][pos] for v in values]
        mean = sum(v * w for v, w in zip(values, col))
        res.append([v - mean for v in rng.choices(values, col, k=4000)])
    means = [8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
    from ml.decode import decode
    ticket = decode(means, res, p)
    assert _valid(ticket, p)
    # P(1) = 10.9% vs P(2) = 9.1% is a clear gap; the top end (P(55) 10.9% vs
    # P(54) 9.9%) is within sampling noise, so only the bottom is pinned
    assert ticket[0] == 1
    assert law_pos_score(ticket, p) > 0.95 * ceiling_score(p)


def test_consensus_votes_by_position():
    p = get_product("power_645")
    same = [3, 9, 18, 27, 36, 44]
    assert consensus_ticket([same] * 3 + [[1, 2, 3, 4, 5, 6]], p) == same
    # number 9 is voted at position 2 by everyone, so it must stay there even
    # though the other positions disagree
    t = consensus_ticket([[1, 9, 20, 30, 40, 45], [2, 9, 21, 31, 41, 44],
                          [4, 9, 22, 32, 42, 43]], p)
    assert _valid(t, p) and t[1] == 9


def test_cv_strength_trusts_history_when_history_deviates():
    """Guard against a vacuous 'always pick the law': if the draws really do
    break the order-statistic law, CV must hand most of the weight to history
    (weight T/(T+strength), T ~ 350 tested-on history here)."""
    p = get_product("power_645")
    rng = random.Random(2)
    fixed = [2, 4, 6, 8, 10, 12]
    draws = [{"main": fixed if rng.random() < 0.5
              else sorted(rng.sample(range(1, 46), 6))} for _ in range(400)]
    s = choose_strength(p, draws)
    assert 350 / (350 + s) > 2 / 3


def test_cv_strength_losses_on_the_real_draws():
    p = get_product("power_655")
    from analyze import load_draws
    losses = strength_losses(p, load_draws(p))
    assert set(losses) and all(v > 0 for v in losses.values())
    # the exact law must beat raw counts on real (uniform) draws
    assert losses[float("inf")] < losses[0.0]
