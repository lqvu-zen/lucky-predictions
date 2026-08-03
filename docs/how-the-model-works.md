# How the models work — and how we know they don't

This project keeps six **position-based** models. They all sort each draw's six
numbers ascending (p1 < p2 < … < p6) and reason about *where* numbers land. None
of them beats random odds, and most of the project's effort goes into
**demonstrating that rigorously** rather than into the models themselves.

> **TL;DR** You can model the *structure* of a lottery draw (which slots tend to
> hold small vs large numbers) in ever richer detail, but never the *outcome*.
> The structure is real; it is also identical every draw, so it predicts nothing.
> The value here is the pipeline and the honest measurement.

## The shared idea: order statistics

Sort any draw ascending and each position has a stable, real distribution: p1
(the smallest) is almost always low, p6 (the largest) almost always high. These
are **order statistics**. For a uniform draw they are fully determined by
combinatorics:

```
P(position p = value v) = C(v-1, p-1) · C(N-v, k-p) / C(N, k)
```

`ml/joint.closed_form_grid` computes this exactly. Every model below is, in the
end, an attempt to estimate that same fixed grid from data — which is why they
all arrive at the same place.

---

## The six models

### 1. Positional regression — `ml/positional.py`

Six regressors (`ridge` / `gb`), one per ordered position, predicting that
position's value from rolling means, the previous draw, and the weekday. Squared
error is minimised by predicting the mean, and the features carry no signal, so
each regressor falls back to that position's historical average:

```
Position:               p1    p2    p3    p4    p5    p6
Model prediction (raw): 8.6  18.1  26.0  33.8  42.1  49.9
Historical mean:        8.2  16.3  24.5  32.4  40.3  47.9
Theoretical (uniform):  8.0  16.0  24.0  32.0  40.0  48.0   (= (N+1)·i/(k+1))
```

All three rows nearly coincide. Prettier tickets, no edge.

### 2. Joint number×position grid — `ml/joint.py`

Maximum-likelihood estimate of the full N×6 grid: count how often each number
landed at each sorted position, normalise. Pure counting, no ML. It reproduces
the closed-form law to a max absolute difference of ~0.015 — the data is just
re-estimating a formula we already know. Renders as the dashboard heatmap.

### 3. Gap / spacing — `ml/gap.py`

Models the *distances* between consecutive numbers (p1, p2−p1, p3−p2, …) and
rebuilds a ticket by cumulative sum. A different parameterisation of the same
structure.

### 4. Conditional / autoregressive — `ml/chain.py`

Predicts p1, then each position conditioned on the previous one, respecting the
ascending order.

### 5. Per-position classifier — `ml/perpos.py`

A LogisticRegression per position predicting *which number* lands there — the
trained-ML version of the joint grid. Its learned probabilities converge to the
same order-statistic marginals.

### 6. Empirical sampler — `ml/sampler.py`

Samples each position from its real historical distribution instead of taking
the mode. Produces varied but position-realistic tickets.

Plus five **for-fun heuristic lines** (`random / hot / cold / overdue /
balanced`) and a **consensus** ticket built from the numbers most predictors
agree on.

---

## Turning a grid into a ticket — and the limit of that

Given a grid, which six numbers should you pick? Maximising the expected count
of correct positions is a *linear* objective, so it has an exact answer. Because
the ticket must be ascending, the assignment forms a chain and a simple DP solves
it in O(N·k) — `ml/optimal.py`:

```
dp[p][v] = grid[v][p] + max over v' < v of dp[p-1][v']
```

The greedy per-position pick that the other models use turns out to be **already
optimal** on this grid family (its columns are ordered by monotone likelihood
ratio, so argmaxes never collide). Verified, not assumed: the DP strictly beats
greedy on a hand-built adversarial grid and matches brute-force enumeration
exactly on a random one (`tests/test_optimal.py`).

---

## How we know none of it works

Six models all landing near baseline is suggestive, not proof. Five independent
tests make it rigorous. Figures are Power 6/55.

### The information floor — `run.py proper-score`

The k/6 hit score is very noisy (six positions, almost always missed). So models
are also graded by **log-loss** on the probability they assigned to the number
that actually landed at each position — a proper scoring rule, which uses the
whole distribution and needs far less data to separate models.

The floor is the **entropy of the true position law**: no model reading only past
draws can beat it.

```
floor (entropy of the true law)  3.3623 nats
theory  3.3548 · shrunk  3.3801 · empirical  3.3811
empirical-100  3.5625 · uniform  4.0073   (= ln 55 exactly)
```

Nothing beats the floor. Learning from history lands slightly *worse* than the
closed-form law — estimation noise added to a formula we already knew. A
100-draw window is clearly worse: more data helps you estimate a fixed law, not
predict a draw.

### The ceiling and its noise band — `run.py ceiling`

How good could a model be if it knew the law *exactly*? That has a closed form
too: the optimal ticket's expected score is **0.0651**, against **0.0423** for a
random ticket. Perfect knowledge is worth **+0.023** of a k/6 point.

Simulation adds the part that matters — the **noise band**. Over 130 draws even a
perfect model lands anywhere in [0.0487, 0.0821]; over 13 draws, [0.0128,
0.1282]. Each predictor is judged against a band for *its own* sample size, and
every one sits inside its band.

### p-values, and the trap — `run.py ceiling`

Each predictor also gets p = P(a skill-free random ticket scores this well).
One strategy scored **p = 0.0125** — by the usual α = 0.05 that reads as a
discovery. But it is the best of **13** predictors, and correcting for that gives
**Šidák p = 0.151**. Test enough candidates and one always looks gifted.

### Does the grid match theory? — `run.py residual`

Each cell compared with its exact theoretical rate as z = (observed −
expected)/sd, over 1379 draws: mean z +0.021, |z| > 2 in 5.0% of cells (expect
4.6%), largest |z| = 3.28 → p = 0.27 once you account for 300 cells. Numbers are
binned so every χ² bin expects ≥ 5, without which the test would look rigorous
and mean nothing.

### Does one draw predict the next? — `run.py independence`

Every test above checks a *marginal* distribution. All would pass on a machine
that produced perfect frequencies but always followed 7 with 23. Three direct
tests:

- per-position autocorrelation, lags 1–10: 60 tests, 2 beyond 2σ (expect 2.7),
  strongest r = −0.059 → **Šidák p = 0.82**
- overlap with the draw L back vs Hypergeometric(N,k,k): worst lag p = 0.015 raw
  → **Šidák p = 0.14**
- shape of the lag-1 overlap distribution: **χ² p = 0.35**

(The overlap between two draws follows the *same* hypergeometric law as "how many
of your ticket's numbers come up" — comparing a fixed k-subset with a fresh
random one is the same problem either way.)

### And why backtests lie — `run.py overfit`

A ticket optimised against N past draws, then scored on 200 it never saw:

| training draws | score on them | score on unseen | gap |
|---:|---:|---:|---:|
| 10 | **0.2167** | 0.0392 | +0.1775 |
| 30 | 0.1500 | 0.0642 | +0.0858 |
| 100 | 0.1117 | 0.0442 | +0.0675 |
| 800 | 0.0729 | 0.0567 | +0.0163 |

With 10 draws it scores **3.3× the theoretical ceiling** — flatly impossible for
genuine knowledge — then performs *worse than a random ticket* on new draws. The
gap closes monotonically as data grows. Overfitting is not a flaw of one
algorithm; it is the ratio of freedom to evidence.

---

## How predictions are scored

Every predictor logs one ticket **before** each draw (`ml/ledger.py`), and is
scored after (`ml/score.py`) two ways:

- **Hits** — number overlap, 0–6. Random baseline `k²/N` (0.655 for 6/55).
- **Pos-hits** — correct number at the correct sorted position. Reported as a
  k/6 score.

Backtests add bootstrap 95% CIs. For a fair lottery, *any* fixed ticket has
expected hits of exactly `k²/N`, so modelling position structure in more detail
changes the tickets' *shape*, never their *hit rate*.

## And the money

`run.py ticket-ev`. One 10,000 VND line on 6/55 returns 2,380 VND in
expectation — a loss of **7,620 VND (−76%)**, with a 1.33% chance of any prize.
`match_probability` takes no ticket argument, because the odds depend only on
choosing k from N: all **28,989,675** possible tickets share that identical
expected value.

## Commands

```bash
uv run python run.py ml-predict-pos    power_655   # a model's next ticket
uv run python run.py ml-backtest-pos   power_655   # walk-forward, with CIs
uv run python run.py proper-score      power_655   # log-loss vs the entropy floor
uv run python run.py ceiling           power_655   # ceiling, bands, p-values
uv run python run.py residual          power_655   # grid vs theory
uv run python run.py independence      power_655   # does one draw predict the next?
uv run python run.py overfit           power_655   # in-sample vs out-of-sample
uv run python run.py optimal-ticket    power_655   # greedy vs provably optimal
uv run python run.py ticket-ev         power_655   # what a line is worth
```
