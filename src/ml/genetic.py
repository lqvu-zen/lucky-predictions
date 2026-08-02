"""Genetic algorithm ticket — a live overfitting demonstration (roadmap #28).

Every other model here is built to be honest. This one is built the way an
over-eager optimiser would build it: take the historical draws, and evolve the
single ticket that would have scored best on them. No leakage-safe walk-forward,
no held-out data during the search — just "optimise the metric".

It works spectacularly. On the data it trained on, the evolved ticket scores far
above the theoretical ceiling (`ml.ceiling`) — which is the tell, because the
ceiling is the best any *genuine* knowledge could achieve. Beating it is proof
of memorisation, not skill. Then the same ticket is scored on draws it never
saw, and it collapses to the random baseline.

A second, quieter lesson lives in the fitness function. Scoring a ticket against
T draws looks like O(T·k) work, but

    fitness(ticket) = (1/(T·k)) · sum_p  counts[p][ticket_p]

where `counts[p][v]` is how often value v appeared at position p. That is O(k),
and it also means the GA is maximising a *linear* objective over ascending
tickets — exactly the problem `ml.optimal`'s DP solves exactly. So we can
compute the true in-sample optimum directly and check what the GA converged to.
An evolutionary search is not finding anything subtle here; it is rediscovering
a table lookup, slowly.

Pure Python.
"""
from __future__ import annotations

import random

from analyze import load_draws
from config import Product, get_product
from ml.ceiling import exact_optimal, exact_random
from ml.optimal import optimal_ticket


def position_counts(product: Product, draws) -> list[list[int]]:
    """counts[p][v] = how often value v landed at sorted position p."""
    k, N = product.main_count, product.max_value
    counts = [[0] * (N + 1) for _ in range(k)]
    for d in draws:
        for p, v in enumerate(sorted(d["main"][:k])):
            counts[p][v] += 1
    return counts


def fitness(ticket, counts, n_draws: int, k: int) -> float:
    """Mean k/6 position score of this ticket over the counted draws — O(k)."""
    if n_draws <= 0:
        return 0.0
    return sum(counts[p][v] for p, v in enumerate(ticket)) / (n_draws * k)


def _random_ticket(rng: random.Random, product: Product) -> list[int]:
    return sorted(rng.sample(range(1, product.max_value + 1),
                             product.main_count))


def _crossover(a, b, rng: random.Random, product: Product) -> list[int]:
    """Take a random mix of both parents' numbers, top up randomly, sort."""
    k, N = product.main_count, product.max_value
    pool = list({*a, *b})
    rng.shuffle(pool)
    child = set(pool[:k])
    while len(child) < k:
        child.add(rng.randint(1, N))
    return sorted(child)


def _mutate(ticket, rng: random.Random, product: Product, rate: float):
    k, N = product.main_count, product.max_value
    out = set(ticket)
    for v in list(out):
        if rng.random() < rate:
            out.discard(v)
            while len(out) < k:
                out.add(rng.randint(1, N))
    while len(out) < k:
        out.add(rng.randint(1, N))
    return sorted(out)


def run_ga(product: Product, draws, pop_size: int = 120, generations: int = 80,
           mutation: float = 0.15, elite: int = 4, seed: int = 0):
    """Evolve the ticket that best fits `draws`. Returns (ticket, history)."""
    rng = random.Random(seed)
    k = product.main_count
    counts = position_counts(product, draws)
    T = len(draws)

    pop = [_random_ticket(rng, product) for _ in range(pop_size)]
    history = []
    best, best_fit = pop[0], -1.0
    for _ in range(generations):
        scored = sorted(((fitness(t, counts, T, k), t) for t in pop),
                        key=lambda x: -x[0])
        if scored[0][0] > best_fit:
            best_fit, best = scored[0]
        history.append(scored[0][0])
        survivors = [t for _, t in scored[:max(elite, pop_size // 4)]]
        nxt = [t for _, t in scored[:elite]]          # elitism
        while len(nxt) < pop_size:
            a = rng.choice(survivors)
            b = rng.choice(survivors)
            nxt.append(_mutate(_crossover(a, b, rng, product), rng, product,
                               mutation))
        pop = nxt
    return best, best_fit, history


def _score_on(product: Product, ticket, draws) -> float:
    """Honest mean k/6 of a fixed ticket over a set of draws."""
    k = product.main_count
    if not draws:
        return 0.0
    total = sum(sum(1 for i in range(k) if sorted(d["main"][:k])[i] == ticket[i])
                for d in draws)
    return total / len(draws) / k


def evaluate(product_name: str, test_draws: int = 200, **ga_kwargs) -> dict:
    """Fit on the past, then score on draws the search never saw."""
    product = get_product(product_name)
    draws = load_draws(product)
    k = product.main_count
    split = len(draws) - test_draws
    if split < 50:
        return {"product": product.label, "tested": 0}
    train, test = draws[:split], draws[split:]

    ticket, train_fit, history = run_ga(product, train, **ga_kwargs)

    # the exact in-sample optimum: same linear objective, solved by DP (#20)
    counts = position_counts(product, train)
    grid = [[counts[p][v] for p in range(k)]
            for v in range(product.max_value + 1)]
    exact_ticket = optimal_ticket(grid, product)
    exact_fit = fitness(exact_ticket, counts, len(train), k)

    _, ceil_hits = exact_optimal(product)
    return {
        "product": product.label, "game": product_name,
        "n_train": len(train), "n_test": len(test),
        "ga_ticket": ticket,
        "train_score": train_fit,
        "test_score": _score_on(product, ticket, test),
        "exact_ticket": exact_ticket,
        "exact_train_score": exact_fit,
        "exact_test_score": _score_on(product, exact_ticket, test),
        "ga_found_optimum": ticket == exact_ticket,
        "ceiling_score": ceil_hits / k,
        "random_score": exact_random(product) / k,
        "history": history,
    }


def sweep(product_name: str, sizes=(10, 20, 30, 50, 100, 200, 400, 800),
          test_draws: int = 200, seed: int = 0) -> dict:
    """Train on N past draws, test on unseen ones — for a range of N.

    This is where the overfitting actually shows. A ticket is a tiny model (six
    choices), so against 1000+ draws there is no noise left to memorise and the
    in-sample optimum is basically the theoretical one. Shrink the training set
    and the gap explodes: with a handful of draws the evolved ticket looks
    brilliant on them and is worthless everywhere else.
    """
    product = get_product(product_name)
    draws = load_draws(product)
    k = product.main_count
    split = len(draws) - test_draws
    train_all, test = draws[:split], draws[split:]
    _, ceil_hits = exact_optimal(product)

    rows = []
    for n in sizes:
        if n > len(train_all):
            continue
        window = train_all[-n:]                  # most recent n training draws
        counts = position_counts(product, window)
        grid = [[counts[p][v] for p in range(k)]
                for v in range(product.max_value + 1)]
        ticket = optimal_ticket(grid, product)   # exact in-sample best
        rows.append({
            "n_train": n,
            "ticket": ticket,
            "train_score": fitness(ticket, counts, n, k),
            "test_score": _score_on(product, ticket, test),
        })
    return {
        "product": product.label, "game": product_name,
        "n_test": len(test), "rows": rows,
        "ceiling_score": ceil_hits / k,
        "random_score": exact_random(product) / k,
    }


def format_sweep(r: dict) -> str:
    out = [
        f"{r['product']} — overfitting vs training-set size "
        f"(always tested on the same {r['n_test']} unseen draws)",
        "",
        f"  {'train':>7}{'in-sample':>12}{'out-of-sample':>16}{'gap':>10}",
    ]
    for row in r["rows"]:
        gap = row["train_score"] - row["test_score"]
        out.append(f"  {row['n_train']:>7}{row['train_score']:>12.4f}"
                   f"{row['test_score']:>16.4f}{gap:>+10.4f}")
    out += [
        "",
        f"  theoretical ceiling {r['ceiling_score']:.4f}   "
        f"random ticket {r['random_score']:.4f}",
        "",
        "  Small training sets produce spectacular in-sample scores far above the",
        "  ceiling — impossible for real knowledge, so it is pure memorisation — and",
        "  they buy nothing out of sample. As the training set grows the illusion",
        "  fades: with enough draws the 'optimised' ticket is just the theoretical",
        "  one, and in-sample and out-of-sample agree. Overfitting is not a property",
        "  of the algorithm; it is the ratio of freedom to evidence.",
    ]
    return "\n".join(out)


def format_report(r: dict) -> str:
    if not r.get("n_train"):
        return f"{r['product']}: not enough data."
    over = r["train_score"] / r["ceiling_score"] if r["ceiling_score"] else 0
    out = [
        f"{r['product']} — genetic algorithm ticket "
        f"(fit on {r['n_train']} draws, tested on {r['n_test']})",
        "",
        f"  evolved ticket        {r['ga_ticket']}",
        f"  exact in-sample best  {r['exact_ticket']}"
        + ("   (GA found it)" if r["ga_found_optimum"] else "   (GA fell short)"),
        "",
        f"  score on the data it trained on   {r['train_score']:.4f}",
        f"  score on unseen draws             {r['test_score']:.4f}",
        "",
        f"  theoretical ceiling               {r['ceiling_score']:.4f}",
        f"  a random ticket                   {r['random_score']:.4f}",
        "",
        f"  Training score is {over:.2f}x the ceiling.",
    ]
    if over < 1.3:
        out += [
            "",
            "  Note the anticlimax: with this many training draws the search barely",
            "  overfits at all, and the ticket it 'discovered' is essentially the",
            "  theoretical optimum. A ticket is a six-choice model, and a thousand",
            "  draws leave it no noise to memorise. Run the sweep (`--sweep`) to see",
            "  the overfitting appear as the training set shrinks.",
        ]
    else:
        out += [
            "",
            "  Nothing can legitimately exceed the ceiling, so the excess is memorised",
            "  noise, not skill — and out of sample it falls back to chance.",
        ]
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "power_655"
    print(format_report(evaluate(name)))
