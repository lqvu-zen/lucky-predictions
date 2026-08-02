"""Expected value of a single line (roadmap #8).

Exact hypergeometric maths, no simulation. If you pick k numbers and k are
drawn from N, the chance of matching exactly j of them is

    P(j) = C(k, j) · C(N-k, k-j) / C(N, k)

Multiply each by its prize tier, add up, subtract the ticket price, and you have
the expected value of a line.

**The point of this module is what the formula does NOT contain: your numbers.**
P(j) depends only on k and N. Every one of the C(55, 6) = 28,989,675 possible
tickets — birthdays, hot numbers, the model's pick, 1-2-3-4-5-6 — has exactly
the same expected value. The interactive calculator on the dashboard exists to
let someone type their favourite line and watch the number refuse to move.
"""
from __future__ import annotations

from math import comb

from config import Product, get_product


def match_probability(product: Product, matches: int) -> float:
    """P(exactly `matches` of the ticket's numbers are drawn)."""
    k, N = product.main_count, product.max_value
    if matches < 0 or matches > k:
        return 0.0
    return comb(k, matches) * comb(N - k, k - matches) / comb(N, k)


def tier_table(product: Product) -> list[dict]:
    """One row per match count that pays, plus its contribution to the EV."""
    tiers = dict(product.prize_tiers)
    rows = []
    for matches in sorted(tiers):
        p = match_probability(product, matches)
        prize = tiers[matches]
        rows.append({
            "matches": matches,
            "prob": p,
            "one_in": (1 / p) if p > 0 else None,
            "prize": prize,
            "contribution": p * prize,
        })
    return rows


def summary(product_name: str) -> dict:
    product = get_product(product_name)
    rows = tier_table(product)
    gross = sum(r["contribution"] for r in rows)
    cost = product.ticket_cost
    return {
        "product": product.label, "game": product_name,
        "max_value": product.max_value, "main_count": product.main_count,
        "ticket_cost": cost,
        "total_tickets": comb(product.max_value, product.main_count),
        "tiers": rows,
        "gross_ev": gross,
        "net_ev": gross - cost,
        "return_pct": (100.0 * gross / cost - 100.0) if cost else 0.0,
        # chance of winning anything at all
        "p_any_prize": sum(r["prob"] for r in rows),
    }


def _vnd(x: float) -> str:
    if abs(x) >= 1_000_000_000:
        return f"{x/1_000_000_000:,.1f}bn"
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:,.1f}m"
    return f"{x:,.0f}"


def format_report(s: dict) -> str:
    out = [
        f"{s['product']} — expected value of one line "
        f"({s['ticket_cost']:,} VND)",
        "",
        f"  {'match':<8}{'probability':>16}{'1 in':>16}{'prize':>14}{'EV part':>12}",
    ]
    for r in s["tiers"]:
        out.append(
            f"  {r['matches']:<8}{r['prob']:>16.3e}"
            f"{r['one_in']:>16,.0f}{_vnd(r['prize']):>14}"
            f"{r['contribution']:>12,.0f}")
    out += [
        "",
        f"  chance of any prize   {s['p_any_prize']:.4%}  "
        f"(1 in {1/s['p_any_prize']:,.1f})",
        f"  expected return       {_vnd(s['gross_ev'])} VND per "
        f"{s['ticket_cost']:,} VND line",
        f"  expected loss         {_vnd(s['net_ev'])} VND  "
        f"({s['return_pct']:+.1f}%)",
        "",
        f"  None of this depends on which numbers you choose. All "
        f"{s['total_tickets']:,} possible",
        "  tickets share this exact expected value — which is the whole lesson.",
    ]
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    print(format_report(summary(sys.argv[1] if len(sys.argv) > 1 else "power_655")))
