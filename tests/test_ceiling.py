"""The theoretical ceiling: best possible score, and its noise band."""
import pytest

from config import get_product
from ml import ceiling
from ml.joint import closed_form_grid
from ml.optimal import expected_pos_hits


def test_knowing_the_law_beats_a_random_ticket_but_only_just():
    for name in ("power_655", "power_645"):
        product = get_product(name)
        _, best = ceiling.exact_optimal(product)
        rnd = ceiling.exact_random(product)
        assert best > rnd, "the optimal ticket must beat a random one"
        # ...and the whole advantage is worth well under one position in six
        assert best - rnd < 1.0


def test_ceiling_ticket_is_valid():
    for name in ("power_655", "power_645"):
        product = get_product(name)
        ticket, _ = ceiling.exact_optimal(product)
        assert len(ticket) == product.main_count
        assert len(set(ticket)) == product.main_count
        assert ticket == sorted(ticket)


def test_exact_random_matches_definition():
    """sum_p sum_v P(v at p)^2 — the collision probability."""
    product = get_product("power_645")
    g = closed_form_grid(product)
    k, N = product.main_count, product.max_value
    expected = sum(g[v][p] ** 2 for p in range(k) for v in range(1, N + 1))
    assert ceiling.exact_random(product) == pytest.approx(expected, abs=1e-12)


def test_no_ticket_can_beat_the_optimal_one():
    """Spot-check the ceiling really is a ceiling, against random tickets."""
    import random
    product = get_product("power_655")
    grid = closed_form_grid(product)
    _, best = ceiling.exact_optimal(product)
    rng = random.Random(3)
    for _ in range(300):
        t = sorted(rng.sample(range(1, product.max_value + 1), product.main_count))
        assert expected_pos_hits(grid, t, product) <= best + 1e-12


def test_simulated_band_brackets_the_ceiling():
    """A perfect model's simulated mean must centre on its exact ceiling."""
    r = ceiling.simulate("power_655", n_draws=200, n_sims=400, seed=1)
    assert r["lo"] <= r["ceiling_score"] <= r["hi"]
    assert r["sim_mean"] == pytest.approx(r["ceiling_score"], abs=0.01)


def test_band_widens_as_sample_shrinks():
    """The whole point of per-model bands: fewer draws => much wider band."""
    few = ceiling.simulate("power_655", n_draws=10, n_sims=400, seed=2)
    many = ceiling.simulate("power_655", n_draws=300, n_sims=400, seed=2)
    assert (few["hi"] - few["lo"]) > (many["hi"] - many["lo"])
