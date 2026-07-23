"""Jackpot reality-check — the exact expectation of chasing the jackpot.

No simulation needed; it's a closed-form expectation:
  - odds of the jackpot with one line = 1 / C(N, k)
  - expected number of draws to win it once = C(N, k)  (geometric mean)
  - expected years = that / draws-per-year
  - expected money spent = that × ticket cost

Plus a relatable comparison to the yearly odds of being struck by lightning.
One memorable set of numbers that captions the whole project.
"""
from __future__ import annotations

from math import comb

from config import get_product

# rough yearly odds of being struck by lightning (~1 in 1.2 million)
_LIGHTNING_ANNUAL = 1 / 1_200_000


def summary(product_name: str) -> dict:
    p = get_product(product_name)
    combos = comb(p.max_value, p.main_count)          # 1 / jackpot odds
    draws_per_year = max(len(p.draw_days), 1) * 52
    jackpot = dict(p.prize_tiers).get(p.main_count, 0)
    expected_cost = combos * p.ticket_cost
    # how many times more likely is lightning than a single-line jackpot?
    lightning_ratio = _LIGHTNING_ANNUAL * combos
    return {
        "product": p.label,
        "one_in": combos,
        "expected_draws": combos,
        "expected_years": round(combos / draws_per_year),
        "draws_per_year": draws_per_year,
        "expected_cost": expected_cost,
        "jackpot_nominal": jackpot,
        "cost_vs_jackpot": round(expected_cost / jackpot, 1) if jackpot else None,
        "lightning_ratio": round(lightning_ratio, 1),
        "ticket_cost": p.ticket_cost,
    }


def _vnd(x: int) -> str:
    if x >= 1_000_000_000:
        return f"{x/1_000_000_000:.1f} billion VND"
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f} million VND"
    return f"{x:,} VND"


def format_report(s: dict) -> str:
    return "\n".join([
        f"{s['product']} — jackpot reality check",
        "",
        f"  Odds per line:        1 in {s['one_in']:,}",
        f"  Expected draws to win once: {s['expected_draws']:,}",
        f"  ... at {s['draws_per_year']} draws/year: about {s['expected_years']:,} years",
        f"  Expected money spent:  {_vnd(s['expected_cost'])}"
        + (f"  (~{s['cost_vs_jackpot']}x the {_vnd(s['jackpot_nominal'])} jackpot)"
           if s['cost_vs_jackpot'] else ""),
        f"  You are ~{s['lightning_ratio']}x more likely to be struck by "
        f"lightning this year than to win one jackpot with a single line.",
        "",
        "  The house edge, in perspective: you'd spend many times the prize, "
        "over many lifetimes, to win it once on average.",
    ])


if __name__ == "__main__":
    import sys
    print(format_report(summary(sys.argv[1] if len(sys.argv) > 1 else "power_655")))
