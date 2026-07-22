# How the models predict the next draw

This project keeps two **position-based** models. Both sort each draw's six
numbers ascending (p1 < p2 < … < p6) and reason about *where* numbers land —
then, honestly, neither beats random odds. This doc explains how each works
and why the honest result is what it is.

> **TL;DR** You can model the *structure* of a lottery draw (which slots tend
> to hold small vs large numbers) in ever richer detail, but never the
> *outcome*. The value is the pipeline and the honest measurement.

## The shared idea: order statistics

Sort any draw ascending and each position has a stable, real distribution:
p1 (the smallest) is almost always low, p6 (the largest) almost always high.
These are **order statistics**. For a uniform draw they're fully determined by
combinatorics — no prediction involved. Both models fit this structure; the
randomness is *which* value lands in each position on a given night.

## Model 1 — Positional (ordered) regression

`src/ml/positional.py`. Question: *what value will appear at each ordered
position of the next draw?*

**How it reasons**

1. **Features per position** — from history it computes, for each position:
   its value in the previous draw, rolling means over the last 20/50 draws, a
   rolling std, plus the target draw's weekday.
2. **Six regressors** (`ridge` or `gb`) — one per position, each learning
   `value = weighted sum of features`. Since squared error is minimised by
   predicting the mean, and the features carry no real signal, each regressor
   falls back to that position's historical average (regression to the mean).
3. **Assemble a ticket** — round the six predictions and force them strictly
   increasing and distinct within `[1, N]`.

**It just predicts the average position.** For Power 6/55:

```
Position:               p1    p2    p3    p4    p5    p6
Model prediction (raw): 8.6  18.1  26.0  33.8  42.1  49.9
Historical mean:        8.2  16.3  24.5  32.4  40.3  47.9
Theoretical (uniform):  8.0  16.0  24.0  32.0  40.0  48.0   (= (N+1)·i/(k+1))
```

All three rows nearly coincide, and the coefficients on the "recent history"
features are tiny compared with how much each position actually varies. So the
model outputs roughly the same tidy spread every draw (e.g.
`09 - 18 - 26 - 34 - 42 - 50`). Prettier tickets, no edge.

```bash
uv run python run.py ml-predict-pos  power_655
uv run python run.py ml-backtest-pos power_655   # mean hits (95% CI) + per-position MAE
```

The per-position MAE shows it fits each position's range; the hits CI spanning
the baseline shows it still can't pick the actual numbers.

## Model 2 — Joint number×position grid

`src/ml/joint.py`. Question: *for every number k and position p, what is
`P(number k lands at position p)`?* — a full N×6 grid. It's the parent of the
positional view: a column is a position's distribution over numbers, and a
number's row summed over positions gives its overall appearance rate (≈ k/N).

**How it reasons.** The model is the maximum-likelihood estimate of the grid —
count how often each number landed at each sorted position, and normalise.
Pure counting. To make a ticket it picks, per position, the most likely
still-unused number, then sorts them.

**It recovers a fixed law.** The grid a uniform lottery produces is known in
closed form (order statistics):

```
P(position p = value v) = C(v-1, p-1) · C(N-v, k-p) / C(N, k)
```

The learned grid matches this to a max absolute difference of ~0.015 — the
data just re-estimates that fixed formula. Real cells for 6/55:

```
P(number k at position p)   [p1    p2    p3    p4    p5    p6]
  number  2:  0.084 0.012 0.000 0.000 0.000 0.000
  number 28:  0.001 0.015 0.031 0.036 0.008 0.002
  number 54:  0.000 0.000 0.000 0.000 0.009 0.092
```

Small numbers concentrate in low positions, large in high — a clean diagonal
band, which the dashboard renders as a number×position heatmap. Because that
band is identical every draw, the grid has no per-draw predictive power.

```bash
uv run python run.py ml-predict-joint  power_655
uv run python run.py ml-backtest-joint power_655
```

## How we score them (and why it's always "≈ random")

Both models output a 6-number ticket, so scoring is by **hits** — how many of
the ticket's numbers actually came up — averaged over many draws and compared
to the random baseline `6×6/N` (0.655 for 6/55, 0.800 for 6/45). Backtests add
a **bootstrap 95% CI** so a lucky run can't look like signal; the live loop
(`ml-loop`) logs each prediction *before* the draw and scores it after,
building the dashboard scorecard.

For a fair lottery the actual numbers are independent of any feature, and for
*any* fixed 6-number ticket the expected hits are exactly `36/N`. So modelling
the position structure in more detail changes the tickets' *shape*, never
their *hit rate*. Every model's CI straddles the baseline — the honest,
expected result, and being able to demonstrate it is the real skill here.
