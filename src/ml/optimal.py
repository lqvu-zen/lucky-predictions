"""Optimal ticket selection from a number×position grid (roadmap #20).

Every grid-based model here has to turn a grid P(number v at ordered position p)
into one ticket. `joint.predict_ticket` does it greedily: walk positions left to
right, take the highest-probability number not already used. That is *not*
optimal — an early position can grab a number that a later position needed far
more.

What we actually want to maximise is the expected number of correct positions:

    E[pos-hits] = sum_p  P(v_p at position p)

which is linear in the choice, so the best ticket is a maximum-weight
assignment. The textbook tool is the Hungarian algorithm — but here the ticket
must also be *ascending* (v_1 < v_2 < ... < v_k), and that constraint turns the
assignment into a chain, which a simple dynamic program solves exactly:

    dp[p][v] = grid[v][p] + max over v' < v of dp[p-1][v']

Running a prefix maximum makes it O(N·k) — simpler and faster than Hungarian,
and it returns the true optimum under the ordering constraint rather than
Hungarian's unconstrained one (which could hand back a non-monotone assignment
that isn't a valid sorted ticket at all).

Honest note: this is a genuine improvement in *decision-making* — it extracts
the most the grid can offer. It is not an improvement in *knowledge*. The grid
is the same fixed order-statistic law every draw, so a better ticket from it
still cannot beat the odds. What you should expect is a small, real gain in
expected position-hits, and no gain in actual winnings.
"""
from __future__ import annotations

import random

from analyze import load_draws
from config import Product, get_product
from ml.joint import closed_form_grid, empirical_grid, predict_ticket
from ml.proper import shrunk_grid
from ml.util import progress

NEG = float("-inf")


def optimal_ticket(grid, product: Product) -> list[int]:
    """Ascending ticket maximising sum_p grid[v_p][p]  (exact, O(N*k))."""
    k, N = product.main_count, product.max_value
    # dp[p] indexed by number v (1..N); parent[p][v] = chosen v at position p-1
    dp = [[NEG] * (N + 1) for _ in range(k)]
    parent = [[0] * (N + 1) for _ in range(k)]

    for v in range(1, N + 1):
        dp[0][v] = grid[v][0]

    for p in range(1, k):
        best_val, best_arg = NEG, 0
        for v in range(1, N + 1):
            # prefix max over v' < v from the previous position
            prev = dp[p - 1][v - 1]
            if prev > best_val:
                best_val, best_arg = prev, v - 1
            if best_val > NEG:
                dp[p][v] = grid[v][p] + best_val
                parent[p][v] = best_arg

    end, best = 0, NEG
    for v in range(1, N + 1):
        if dp[k - 1][v] > best:
            best, end = dp[k - 1][v], v

    ticket = [0] * k
    v = end
    for p in range(k - 1, -1, -1):
        ticket[p] = v
        v = parent[p][v]
    return ticket


def expected_pos_hits(grid, ticket, product: Product) -> float:
    """Expected count of correct number-at-position for this ticket."""
    return sum(grid[v][p] for p, v in enumerate(sorted(ticket)))


def compare(product_name: str) -> dict:
    """Greedy vs optimal on the theoretical and learned grids."""
    product = get_product(product_name)
    draws = load_draws(product)
    out = {"product": product.label, "game": product_name, "grids": {}}
    for label, grid in (("theory", closed_form_grid(product)),
                        ("empirical", empirical_grid(product, draws, smoothing=0.5)),
                        ("shrunk", shrunk_grid(product, draws, strength=20.0))):
        g_t = predict_ticket(grid, product)
        o_t = optimal_ticket(grid, product)
        g_e = expected_pos_hits(grid, g_t, product)
        o_e = expected_pos_hits(grid, o_t, product)
        out["grids"][label] = {
            "greedy_ticket": g_t, "optimal_ticket": o_t,
            "greedy_expected": g_e, "optimal_expected": o_e,
            "gain": o_e - g_e,
            "gain_pct": (100.0 * (o_e - g_e) / g_e) if g_e > 0 else 0.0,
            "same": g_t == o_t,
        }
    return out


def predict_next(product_name: str) -> dict:
    product = get_product(product_name)
    draws = load_draws(product)
    grid = shrunk_grid(product, draws, strength=20.0)
    return {"product": product.label, "model": "optimal-grid",
            "target_date": product.next_draw_date().isoformat(),
            "ticket": optimal_ticket(grid, product)}


def _bootstrap_ci(samples, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(samples)
    if n == 0:
        return 0.0, 0.0
    means = sorted(sum(rng.choice(samples) for _ in range(n)) / n
                   for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def backtest(product_name: str, test_draws: int = 120,
             min_history: int = 50) -> dict:
    """Walk-forward: does the optimal ticket really land more positions?"""
    product = get_product(product_name)
    draws = load_draws(product)
    k, N = product.main_count, product.max_value
    start = max(min_history, len(draws) - test_draws)
    g_pos, o_pos, g_hits, o_hits = [], [], [], []
    total = len(draws) - start
    for j, t in enumerate(range(start, len(draws))):
        grid = shrunk_grid(product, draws[:t], strength=20.0)
        actual = sorted(draws[t]["main"][:k])
        for ticket, pos_acc, hit_acc in ((predict_ticket(grid, product), g_pos, g_hits),
                                         (optimal_ticket(grid, product), o_pos, o_hits)):
            tk = sorted(ticket)
            pos_acc.append(sum(1 for i in range(k) if tk[i] == actual[i]))
            hit_acc.append(len(set(tk) & set(actual)))
        progress(j + 1, total, "optimal vs greedy")
    if not g_pos:
        return {"product": product.label, "tested": 0}
    g_lo, g_hi = _bootstrap_ci(g_pos)
    o_lo, o_hi = _bootstrap_ci(o_pos)
    return {
        "product": product.label, "tested": len(g_pos),
        "greedy_pos": sum(g_pos) / len(g_pos), "greedy_lo": g_lo, "greedy_hi": g_hi,
        "optimal_pos": sum(o_pos) / len(o_pos), "optimal_lo": o_lo, "optimal_hi": o_hi,
        "greedy_hits": sum(g_hits) / len(g_hits),
        "optimal_hits": sum(o_hits) / len(o_hits),
        "baseline_hits": k * k / N,
    }


def format_compare(r: dict) -> str:
    out = [f"{r['product']} — greedy vs optimal ticket from the same grid", ""]
    for label, g in r["grids"].items():
        out += [
            f"  [{label}]",
            f"    greedy   {g['greedy_ticket']}   E[pos-hits] = {g['greedy_expected']:.5f}",
            f"    optimal  {g['optimal_ticket']}   E[pos-hits] = {g['optimal_expected']:.5f}",
            f"    gain     {g['gain']:+.5f}  ({g['gain_pct']:+.2f}%)"
            + ("   — identical tickets" if g["same"] else ""),
            "",
        ]
    out += [
        "  The optimal ticket maximises expected correct-position count exactly;",
        "  greedy can be beaten because an early position steals a number a later",
        "  one needed more. Better extraction from the grid — but the grid is the",
        "  same fixed law every draw, so this still wins no money.",
    ]
    return "\n".join(out)


def format_backtest(r: dict) -> str:
    if not r.get("tested"):
        return f"{r['product']}: not enough data."
    return "\n".join([
        f"{r['product']} — optimal vs greedy over {r['tested']} draws",
        "",
        f"  greedy   mean pos-hits {r['greedy_pos']:.4f}  "
        f"[95% CI {r['greedy_lo']:.3f}, {r['greedy_hi']:.3f}]   "
        f"mean hits {r['greedy_hits']:.3f}",
        f"  optimal  mean pos-hits {r['optimal_pos']:.4f}  "
        f"[95% CI {r['optimal_lo']:.3f}, {r['optimal_hi']:.3f}]   "
        f"mean hits {r['optimal_hits']:.3f}",
        f"  random-baseline hits {r['baseline_hits']:.3f}",
        "",
        "  Expect the CIs to overlap heavily: the theoretical gain is real but",
        "  tiny, and one draw's worth of luck dwarfs it. Both remain at chance.",
    ])


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "power_655"
    print(format_compare(compare(name)))
    print()
    print(format_backtest(backtest(name)))
