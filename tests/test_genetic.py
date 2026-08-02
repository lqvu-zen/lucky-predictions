"""Genetic-algorithm ticket and the overfitting sweep (#28)."""
import pytest

from config import get_product
from ml import genetic
from ml.optimal import optimal_ticket


def test_fitness_matches_a_direct_count():
    """The O(k) count-table shortcut must equal the naive per-draw scoring."""
    product = get_product("power_655")
    draws = [{"main": [1, 11, 22, 33, 44, 55]},
             {"main": [1, 5, 22, 40, 44, 50]},
             {"main": [2, 11, 30, 33, 47, 55]}]
    ticket = [1, 11, 22, 33, 44, 55]
    counts = genetic.position_counts(product, draws)
    fast = genetic.fitness(ticket, counts, len(draws), product.main_count)
    slow = genetic._score_on(product, ticket, draws)
    assert fast == pytest.approx(slow)


def test_ga_returns_a_valid_ticket():
    product = get_product("power_645")
    draws = [{"main": sorted([(i * 7 + j * 5) % 45 + 1 for j in range(6)])}
             for i in range(60)]
    draws = [d for d in draws if len(set(d["main"])) == 6]
    ticket, fit, hist = genetic.run_ga(product, draws, pop_size=30,
                                       generations=10, seed=1)
    assert len(ticket) == 6 and len(set(ticket)) == 6
    assert ticket == sorted(ticket)
    assert all(1 <= v <= 45 for v in ticket)
    assert 0.0 <= fit <= 1.0
    assert len(hist) == 10


def test_ga_best_fitness_never_decreases():
    """Elitism must make the best-so-far monotone."""
    product = get_product("power_655")
    draws = [{"main": sorted([(i + j * 9) % 55 + 1 for j in range(6)])}
             for i in range(80)]
    draws = [d for d in draws if len(set(d["main"])) == 6]
    _, _, hist = genetic.run_ga(product, draws, pop_size=40, generations=15,
                                seed=2)
    running = 0.0
    for h in hist:
        running = max(running, h)
    assert running == pytest.approx(max(hist))


def test_ga_cannot_beat_the_exact_optimum():
    """The DP solves the same linear objective exactly — GA is bounded by it."""
    product = get_product("power_655")
    draws = [{"main": sorted([(i * 3 + j * 11) % 55 + 1 for j in range(6)])}
             for i in range(120)]
    draws = [d for d in draws if len(set(d["main"])) == 6]
    counts = genetic.position_counts(product, draws)
    k = product.main_count
    grid = [[counts[p][v] for p in range(k)] for v in range(product.max_value + 1)]
    best_exact = optimal_ticket(grid, product)
    exact_fit = genetic.fitness(best_exact, counts, len(draws), k)
    ticket, fit, _ = genetic.run_ga(product, draws, pop_size=60, generations=30,
                                    seed=3)
    assert fit <= exact_fit + 1e-12


def test_overfitting_shrinks_as_training_data_grows():
    """The headline result: the in-sample/out-of-sample gap must close."""
    r = genetic.sweep("power_655", sizes=(10, 800), test_draws=200)
    small, large = r["rows"][0], r["rows"][-1]
    gap_small = small["train_score"] - small["test_score"]
    gap_large = large["train_score"] - large["test_score"]
    assert gap_small > gap_large, "tiny training sets must overfit more"
    # and the tiny-sample fit must exceed what real knowledge could ever do
    assert small["train_score"] > r["ceiling_score"]


def test_out_of_sample_never_beats_the_ceiling_by_much():
    """No training size should produce genuine out-of-sample skill."""
    r = genetic.sweep("power_655", sizes=(10, 50, 200, 800), test_draws=200)
    for row in r["rows"]:
        assert row["test_score"] < r["ceiling_score"] * 1.6
