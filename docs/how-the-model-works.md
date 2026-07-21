# How the ML models predict the next draw

This document explains, end to end, how the machine-learning pipeline in
`src/ml/` turns lottery history into a predicted ticket — and why, honestly,
it can't beat random odds.

> **TL;DR** The model is a calibrated *"how due is this number?"* scorer
> trained on history. Because the draw is uniform and independent, every
> number ends up equally due, so the prediction has no real edge. The value
> of the project is the pipeline and the honest measurement, not the ticket.

## 1. The reframing (the key trick)

We don't ask the model to output "the six winning numbers" directly — that's
a ~28-million-way choice. Instead we flip it into **N independent yes/no
questions** (N = 55 for Power 6/55, 45 for 6/45):

> For each number *k*, what is the probability it appears in the next draw?

Each number is scored on its own, and at the end we simply take the six with
the highest probability. "Predict the draw" becomes 55 ordinary binary
classification problems.

## 2. Turning history into a table (`ml/features.py`)

For every past draw *t* and every number *k*, we build one row of features
computed from **only the draws before *t*** (the "past-only" rule is what
makes the dataset leakage-safe). The 14 features:

| Feature | Meaning |
| --- | --- |
| `rate_all` | share of all past draws containing *k* |
| `rate_w10/30/50/100` | share within the last 10/30/50/100 draws |
| `ema_rate` | exponentially-decayed (recency-weighted) rate |
| `pair_lift` | how often *k* co-occurs with the previous draw's numbers |
| `draws_since` | draws since *k* last appeared |
| `gap_mean` | *k*'s average gap between appearances so far |
| `gap_ratio` | `draws_since / gap_mean` (>1 means "overdue") |
| `appeared_last` | 1 if *k* was in the immediately previous draw |
| `value_norm` | `k / N` (position in the range) |
| `is_odd` | parity of *k* |
| `dow` | weekday of the target draw (a known covariate) |

Each row gets a **label**: 1 if number *k* actually appeared in draw *t*,
else 0. With ~1,300 draws × 55 numbers, that's ~70,000 labelled examples,
built in a single incremental forward pass.

## 3. What "training" does (`ml/model.py`)

A model learns a function `features → probability of a 1`:

- **Logistic regression** — learns one weight per feature, computes a
  weighted sum, and squashes it through an S-curve (sigmoid) into a 0–1
  probability. We deliberately *don't* rebalance classes, so the
  probabilities stay calibrated (near the true base rate) — which makes the
  Brier-score comparison meaningful.
- **Gradient boosting / random forest** — build many small decision trees
  ("if `gap_ratio` > 1.4 and `rate_w50` < 0.1 then lean higher…") and
  average them, capturing nonlinear feature combinations.

## 4. Predicting the next draw (`ml/predict_next.py`)

1. Take **all** history and compute the same 14 features for every number as
   of the upcoming draw date (weekday known from the schedule in
   `config.py`).
2. Feed those 55 rows through the trained model → 55 probabilities.
3. Sort, and the **top 6** become the predicted ticket.
4. Log it to `predictions/ledger.jsonl` *before* the draw happens, so it can
   be scored honestly later (`ml/score.py`).

## 5. Worked example — ball 22 in Power 6/55

Real feature values for number 22, predicting the 2026‑07‑23 draw:

```
rate_all       0.128     draws_since    0.00
rate_w10       0.200     gap_mean       7.75
rate_w30       0.133     gap_ratio      0.00
rate_w50       0.200     appeared_last  1.00
rate_w100      0.180     value_norm     0.40
ema_rate       0.173     is_odd         0.00
pair_lift      0.089     dow            3.00
```

The model turns these into:

```
Ball 22  →  P(appear) = 0.1044   (base rate = 0.1091)   rank #51 of 55
```

Ball 22 is "hot" by folk logic — it appeared in the very last draw
(`appeared_last = 1`, `draws_since = 0`) — yet the model ranks it near the
**bottom**. It picked up a faint "just appeared → slightly less likely next
time" wrinkle from the history, which is itself just noise.

And the crucial part is the **spread across all 55 numbers**:

```
min P = 0.103    max P = 0.119    range ≈ 0.016
```

Every number lands within ~1.5 percentage points of the base rate. The
probabilities are practically tied, so the "top 6" is an arbitrary
tie-break, not a confident pick.

## 6. Why the answer is always "≈ random"

For a fair lottery the label is genuinely independent of every feature — a
number being "hot" or "overdue" tells you nothing about the next draw. So in
training the model can only fit faint accidental patterns in historical
noise, which don't carry forward. The result:

- per-number probabilities collapse to ≈ `6/N`,
- the top-6 is essentially arbitrary,
- and the evaluation confirms it: mean hits sit on the `k²/N` baseline
  (0.655 for 6/55, 0.800 for 6/45), and the Brier score matches the trivial
  base-rate predictor to five decimals.

See it yourself:

```bash
uv run python run.py ml-backtest   power_655 --model both     # hits + Brier vs baseline
uv run python run.py ml-compare    power_655                  # models with bootstrap CIs
uv run python run.py ml-importance power_655 --model rf       # which features it leans on
```

The machinery is real and correct; there is simply no signal for it to find.
That's the honest lesson — and being able to *demonstrate* it with proper
evaluation is the actual skill this project practises.

## 7. An alternative framing — the positional (ordered) model

`src/ml/positional.py` asks a different question. Instead of scoring each
number independently, it sorts every draw ascending (p1 < p2 < … < p6) and
treats each **ordered position** as its own regression target:

> what value will land at position p1 (the smallest), p2, …, p6 next time?

**How it reasons**

1. **Features per position** — from history it computes, for each position:
   its value in the previous draw, rolling means over the last 20/50 draws,
   a rolling std, plus the target draw's weekday. This describes *where each
   position has recently been landing*.
2. **Six regressors** — one Ridge model per position learns
   `value = weighted sum of features`. Since squared error is minimised by
   predicting the mean, and the history features carry no real signal, each
   regressor essentially **falls back to that position's historical average**
   (regression to the mean).
3. **Assemble a valid ticket** — the six predicted values are rounded and
   forced strictly increasing and distinct within `[1, N]`.

**It just predicts the average position.** For Power 6/55:

```
Position:               p1    p2    p3    p4    p5    p6
Model prediction (raw): 8.6  18.1  26.0  33.8  42.1  49.9
Historical mean:        8.2  16.3  24.5  32.4  40.3  47.9
Theoretical (uniform):  8.0  16.0  24.0  32.0  40.0  48.0   (= (N+1)·i/(k+1))
```

All three rows nearly coincide. The magnitude of the Ridge coefficients on
the "what happened recently" features is tiny (~0.37) compared with how much
each position actually varies (std ≈ 6.5–9.3). So the model **all but ignores
recent history** and outputs the same typical spread every draw. Rounded, the
example above becomes the ticket `09 - 18 - 26 - 34 - 42 - 50`.

**Prettier tickets, same non-edge.** This framing nails the *shape* of a draw
(evenly spaced numbers), so its tickets look natural. But each position still
swings randomly by ±8 around its mean, and which value actually appears is
independent of the past. For any fixed 6-number ticket the expected hits are
`36/N` regardless, so the backtest's mean-hits confidence interval still spans
the baseline:

```bash
uv run python run.py ml-predict-pos  power_655   # the ordered ticket
uv run python run.py ml-backtest-pos power_655   # hits (with 95% CI) + per-position MAE
```

The per-position MAE shows the model fits each position's range well; the
hits CI spanning the baseline shows it still can't pick the actual numbers.
Two different questions, one honest answer.

## 8. The richest framing — the joint number×position grid

`src/ml/joint.py` asks the most general version of the question:

> for every number k and every ordered position p, what is
> `P(number k lands at position p in the next draw)`?

That's a full N×6 grid. It's the **parent of the other two models**:

- sum a *row* over positions → `P(number appears at all)` ≈ k/N — the flat
  per-number result;
- a *column* is a position's full distribution over numbers — the positional
  model kept only its mean.

**How it reasons.** The model is the maximum-likelihood estimate of the grid:
count how often each number landed at each sorted position in history, and
normalise. Pure counting — no heavy ML. To make a ticket it picks, per
position, the most likely still-unused number, then sorts them.

**It just recovers a fixed law.** The grid a uniform lottery produces is known
in closed form (order statistics):

```
P(position p = value v) = C(v-1, p-1) · C(N-v, k-p) / C(N, k)
```

The learned grid matches this to a max absolute difference of ~0.015 — i.e.
the data is just re-estimating that fixed formula. Real cell values for 6/55:

```
P(number k at position p)   [p1    p2    p3    p4    p5    p6]
  number  2:  0.084 0.012 0.000 0.000 0.000 0.000
  number 28:  0.001 0.015 0.031 0.036 0.008 0.002
  number 54:  0.000 0.000 0.000 0.000 0.009 0.092
```

Small numbers concentrate in low positions, large in high — a clean diagonal
band. Because that band is identical every draw, the grid has **no per-draw
predictive power**; the backtest mean-hits CI spans the baseline just like the
others. What it wins on is information and visuals: the dashboard renders the
whole grid as a number×position heatmap.

```bash
uv run python run.py ml-predict-joint  power_655   # per-position-mode ticket + grid check
uv run python run.py ml-backtest-joint power_655   # hits (95% CI) + grid-vs-theory diff
```

Three questions — per-number, per-position, and the full joint grid — and the
same honest bottom line: you can model the *structure* of a lottery draw in
ever richer detail, but never the *outcome*.
