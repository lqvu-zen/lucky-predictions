"""How well could a *perfect* model possibly do? (roadmap #14)

The leaderboard keeps reshuffling and no predictor ever pulls away. Two
questions that answers only in combination:

  1. What is the best score anyone could achieve *even knowing the exact law*?
  2. How much does that score wobble over a finite run of draws?

**(1) is exact, no simulation required.** The optimal ticket under the true
order-statistic law scores, in expectation,

    E[pos-hits] = sum_p  grid[v_p][p]        with v = the optimal assignment

and a *randomly chosen* ticket scores

    E[pos-hits] = sum_p sum_v grid[v][p]^2

(the chance the random ticket and the draw independently put the same number at
the same position). The gap between those two numbers is the entire value of
knowing the law perfectly — and it is small.

**(2) needs simulation.** A perfect model still only *averages* its ceiling;
over 130 draws the observed mean bounces around it a lot. That band is the point
of this module: once you know a perfect model would score anywhere in
[lo, hi] over your sample size, you can see that every real predictor sits
inside that band — so their differences are noise, not skill.

Pure Python.
"""
from __future__ import annotations

import random

from config import Product, get_product
from ml.joint import closed_form_grid
from ml.optimal import expected_pos_hits, optimal_ticket


def exact_optimal(product: Product) -> tuple[list[int], float]:
    """The best ticket under the true law, and its expected pos-hits."""
    grid = closed_form_grid(product)
    ticket = optimal_ticket(grid, product)
    return ticket, expected_pos_hits(grid, ticket, product)


def exact_random(product: Product) -> float:
    """Expected pos-hits of a uniformly random ascending ticket."""
    grid = closed_form_grid(product)
    k, N = product.main_count, product.max_value
    return sum(grid[v][p] ** 2 for p in range(k) for v in range(1, N + 1))


def _draw(rng: random.Random, product: Product) -> list[int]:
    return sorted(rng.sample(range(1, product.max_value + 1), product.main_count))


def simulate(product_name: str, n_draws: int = 130, n_sims: int = 2000,
             seed: int = 0) -> dict:
    """Sampling distribution of a perfect model's mean k/6 over n_draws draws."""
    product = get_product(product_name)
    k = product.main_count
    ticket, exact = exact_optimal(product)
    rnd_exact = exact_random(product)
    rng = random.Random(seed)

    means = []
    for _ in range(n_sims):
        total = 0
        for _ in range(n_draws):
            actual = _draw(rng, product)
            total += sum(1 for i in range(k) if actual[i] == ticket[i])
        means.append(total / n_draws / k)      # as a k/6 score
    means.sort()

    def pct(q):
        return means[min(len(means) - 1, int(q * len(means)))]

    return {
        "product": product.label, "game": product_name,
        "n_draws": n_draws, "n_sims": n_sims,
        "optimal_ticket": ticket,
        "ceiling_pos_hits": exact,
        "ceiling_score": exact / k,
        "random_pos_hits": rnd_exact,
        "random_score": rnd_exact / k,
        "sim_mean": sum(means) / len(means),
        "lo": pct(0.025), "hi": pct(0.975),
        "p5": pct(0.05), "p95": pct(0.95),
        "max_seen": means[-1],
    }


def simulate_null(product_name: str, n_draws: int, n_sims: int = 2000,
                  seed: int = 11) -> list[float]:
    """Null distribution of the k/6 score for a *skill-free* player (roadmap #15).

    Each run picks a fresh random ascending ticket and plays it against n_draws
    random draws. The resulting spread is what "no skill whatsoever" looks like
    over that sample size — the reference for a p-value.
    """
    product = get_product(product_name)
    k, N = product.main_count, product.max_value
    rng = random.Random(seed)
    means = []
    for _ in range(n_sims):
        ticket = sorted(rng.sample(range(1, N + 1), k))
        total = 0
        for _ in range(n_draws):
            actual = _draw(rng, product)
            total += sum(1 for i in range(k) if actual[i] == ticket[i])
        means.append(total / n_draws / k)
    means.sort()
    return means


def null_p_value(null_means: list[float], observed: float) -> float:
    """P(a skill-free player scores at least this well). Add-one corrected."""
    ge = sum(1 for m in null_means if m >= observed)
    return (ge + 1) / (len(null_means) + 1)


def with_observed(product_name: str, sim: dict | None = None) -> dict:
    """Attach each real predictor's observed score and whether it's in-band.

    IMPORTANT: each predictor is judged against a band simulated for *its own*
    number of scored draws. A model with 5 results has a hugely wider band than
    one with 130; using a single band would flag short-run luck as a discovery.
    """
    from ml.score import load_scorecard
    sim = sim or simulate(product_name)
    card = load_scorecard()
    rows = []
    bands: dict[int, dict] = {}
    if card:
        g = card.get("games", {}).get(product_name, {})
        for name, m in (g.get("models") or {}).items():
            s = m.get("mean_pos_score", 0.0)
            n = int(m.get("scored", 0) or 0)
            if n <= 0:
                continue
            if n not in bands:
                b = simulate(product_name, n_draws=n, n_sims=sim["n_sims"])
                bands[n] = {"lo": b["lo"], "hi": b["hi"], "max": b["max_seen"],
                            "null": simulate_null(product_name, n,
                                                  n_sims=sim["n_sims"])}
            band = bands[n]
            rows.append({
                "model": name, "score": s, "scored": n,
                "band_lo": band["lo"], "band_hi": band["hi"],
                "in_band": band["lo"] <= s <= band["hi"],
                "above_ceiling": s > sim["ceiling_score"],
                "p_value": null_p_value(band["null"], s),
            })
    rows.sort(key=lambda r: -r["score"])

    # Multiple comparisons: with this many predictors, the *smallest* p-value is
    # not the chance of a fluke — the chance that at least one of them looks
    # good is much higher. Šidák-adjust the best one.
    m = len(rows)
    min_p = min((r["p_value"] for r in rows), default=1.0)
    sim["observed"] = rows
    sim["n_models"] = m
    sim["min_p"] = min_p
    sim["sidak_p"] = 1.0 - (1.0 - min_p) ** m if m else 1.0
    sim["any_significant"] = sim["sidak_p"] < 0.05
    sim["bands"] = {str(n): {kk: vv for kk, vv in b.items() if kk != "null"}
                    for n, b in bands.items()}
    return sim


def format_report(r: dict) -> str:
    k_lo, k_hi = r["lo"], r["hi"]
    out = [
        f"{r['product']} — how good could a perfect model be?",
        "",
        f"  optimal ticket under the true law : {r['optimal_ticket']}",
        f"  its expected correct positions    : {r['ceiling_pos_hits']:.5f} of "
        f"6  → score {r['ceiling_score']:.4f}",
        f"  a randomly chosen ticket          : {r['random_pos_hits']:.5f} of "
        f"6  → score {r['random_score']:.4f}",
        f"  value of knowing the law exactly  : "
        f"{r['ceiling_score'] - r['random_score']:+.4f} score",
        "",
        f"  Over {r['n_draws']} draws ({r['n_sims']} simulated runs), even that "
        f"perfect model lands anywhere in",
        f"  [{k_lo:.4f}, {k_hi:.4f}]  (95%), best run seen {r['max_seen']:.4f}.",
    ]
    if r.get("observed"):
        out += ["", "  observed predictors (each vs a band for its OWN sample size):"]
        for o in r["observed"]:
            flag = ("in band" if o["in_band"] else
                    ("ABOVE band — investigate" if o["score"] > o["band_hi"]
                     else "below band"))
            star = " *" if o["above_ceiling"] else ""
            out.append(
                f"    {o['model']:<18}{o['score']:.4f}  ({o['scored']:>3} scored)  "
                f"band [{o['band_lo']:.4f}, {o['band_hi']:.4f}]  "
                f"p={o['p_value']:.3f}  {flag}{star}")
        out += [
            "",
            "  * = above the ceiling's *mean*; over a handful of draws that is ordinary luck,",
            "    which is exactly why each model is judged against its own sample size.",
            "",
            f"  p = chance a skill-free player (random ticket) scores at least that well.",
            f"  Best p = {r['min_p']:.4f} across {r['n_models']} predictors; testing that many,",
            f"  the corrected value is Šidák p = {r['sidak_p']:.4f} — "
            + ("SIGNIFICANT, investigate." if r["any_significant"]
               else "nothing significant."),
        ]
    out += [
        "",
        "  Read it this way: perfect knowledge is worth almost nothing here, and the",
        "  noise band is far wider than the gap between any two predictors. That is",
        "  why the leaderboard keeps changing hands.",
    ]
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "power_655"
    print(format_report(with_observed(name)))
