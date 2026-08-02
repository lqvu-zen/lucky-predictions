# Roadmap

Planned enhancements, to build one at a time. Everything stays honest: the
lottery is uniform-random, so these deepen the *analysis and the lesson*, not
the odds.

## Status (tick off as shipped)

- [x] 1. Uniformity test — `src/randomness.py`, `run.py uniformity`, dashboard card
- [x] 2. Bankroll / EV simulator — `src/bankroll.py`, prize tiers in config, `run.py bankroll`, dashboard chart
- [x] 3. Accuracy trend over time — running k/6 per predictor in scorecard + dashboard line chart
- [x] 4. Test suite — `tests/` (pytest), `dev` extra, `.github/workflows/tests.yml`
- ~~5. More games (Keno / Power 5/35)~~ — dropped (not needed)
- ~~6. Notifications (Telegram)~~ — dropped (not needed)

Round two:

- [x] 7. Jackpot expectation — `src/jackpot.py`, `run.py jackpot`, dashboard card
- [ ] 8. Interactive ticket EV calculator (type 6 numbers → odds + EV)
- [ ] 9. Meta-learner (stacking) predictor — learn per-predictor weights
- [ ] 10. Calibration curve for the joint grid
- [ ] 11. "Does more data help?" — score vs training-window size

---

## 1. Uniformity test (is the draw really random?)  ·  small, pure-Python

**Goal:** statistically confirm the draws are uniform — the foundation of the
whole "no edge" story.

**Build:**
- `src/analyze.py` (or `src/ml/tests_stats.py`): chi-square goodness-of-fit of
  each number's frequency vs the uniform expectation; also an odd/even and
  low/high split test, and lag-1 autocorrelation.
- Report the χ² statistic, degrees of freedom, and p-value.
- Dashboard card "Is it random?" showing the p-value and a plain-language
  verdict (p high → cannot reject uniform → consistent with a fair draw).

**Honest note:** expected result is "cannot reject randomness."

---

## 2. Bankroll / expected-value simulator (the honest capstone)  ·  medium

**Goal:** show, in money, that playing along loses — the most convincing proof.

**Build:**
- `src/ml/bankroll.py`: replay history; for each predictor, "buy" its ticket
  every draw (cost 10,000 VND), award prize by match count using each game's
  prize tiers (config), track cumulative profit/loss.
- Prize tiers in `config.py` per product (match-count → payout; jackpot as a
  nominal fixed amount since it varies).
- Dashboard "Bankroll" line chart (Chart.js): cumulative VND per predictor
  over draws — every line trends steadily down. Show total spent vs won and
  the realized return (~ −50% or worse, i.e. the house edge).

**Honest note:** the punchline of the project.

---

## 3. Accuracy trend over time  ·  small–medium

**Goal:** make the leaderboard's noise visible — the "leader" keeps changing.

**Build:**
- From `predictions/scored.jsonl`, compute each predictor's rolling mean
  pos-score over draws.
- Dashboard line chart of the rolling scores; overlay the random baseline.
- One-line significance check: bootstrap CI on the current leader vs baseline
  (almost certainly overlaps → not real).

---

## 4. Test suite  ·  small, insurance

**Goal:** guard the fragile parts (crawler parsing especially).

**Build:**
- `tests/` with pytest: `_parse_html` on a saved HTML fixture, `pos_hits` /
  scoring math, model `predict_next` returns a valid 6-number ascending ticket,
  `next_draw_date` schedule logic.
- Add `pytest` to the `ml` extra or a `dev` extra; a CI job to run it.

---

## ~~5. More games (Keno / Power 5/35)~~ — dropped

## ~~6. Notifications (Telegram)~~ — dropped

Both dropped as not needed. Kept here only as a record; delete if you like.

---

## 7. Jackpot expectation (reality check)  ·  small

Exact expectation (no sim needed): odds = 1 / C(N, k); expected draws to win =
C(N, k); expected years = that / draws-per-year; expected spend = that ×
ticket cost; plus a relatable comparison (e.g. lightning). Dashboard card +
`run.py jackpot`.

## 8. Interactive ticket EV calculator  ·  small

Client-side: user types 6 numbers, dashboard shows the hypergeometric odds of
each prize tier and the expected value of the line (deeply negative).

## 9. Meta-learner (stacking)  ·  medium

Learn weights per predictor from past pos-scores; add as a 13th predictor.
Expected to overfit noise and fall back out-of-sample — a live overfitting demo.

## 10. Calibration curve (joint grid)  ·  small

Predicted P(number at position) vs observed frequency; should sit on the
diagonal (well-calibrated to a non-predictive distribution).

## 11. "Does more data help?"  ·  small

Model score vs training-window size — a flat line: more history ≠ signal.

---

# Round three: everything aimed at "number K at position I"

The whole family below targets one metric: **P(number k lands at sorted
position i)**, and the k/6 position score built on it.

**The ceiling, stated up front.** For a uniform draw the grid is known exactly
(order statistics of sampling without replacement):

    P(X_(i) = k) = C(k-1, i-1) · C(N-k, m-i) / C(N, m)

`ml/joint.closed_form_grid` already computes it. Any model trained only on past
draws can, at best, *re-estimate this same fixed grid* — so every idea below is
bounded by the same ceiling (item 14 measures it). They are worth building as
sharper measurement and better decision rules, not as a route to an edge.

## A. Analytics (measure the position structure)

- [ ] 12. **Residual map** — empirical grid minus `closed_form_grid`, as
      standardized z-scores per cell, plus a χ² per position. Expect a noise
      field with ~5% of cells past ±2σ. Turns the existing heatmap into a test.
- [ ] 13. **Per-position distributions** — box/violin of p1…p6 (observed vs
      theoretical mean and quartiles). Shows p1 is small, p6 is large, and each
      is far too wide to pin down.
- [x] 14. **Theoretical ceiling** ⭐ — `src/ml/ceiling.py`, `run.py ceiling`,
      dashboard card, `tests/test_ceiling.py`. The *mean* ceiling turned out to
      need no simulation at all: it is `expected_pos_hits(closed_form_grid,
      optimal_ticket(...))`, and a random ticket's expectation is the collision
      sum `sum_p sum_v grid[v][p]^2`.

      6/55: ceiling **0.0651**, random ticket **0.0423** — knowing the exact law
      is worth **+0.0229** of a k/6 point. Optimal ticket `[1, 11, 22, 33, 44, 55]`.

      What simulation adds is the **noise band**, which is the real payload: over
      130 draws even a perfect model lands anywhere in [0.0487, 0.0821]; over 13
      draws, [0.0128, 0.1282]. Each predictor is compared against a band for
      *its own* sample size — comparing a 13-draw model to a 130-draw band
      wrongly flags ordinary luck as a discovery (it briefly did, and that bug is
      what `test_band_widens_as_sample_shrinks` now guards). With bands done
      correctly, **every predictor sits inside its band**, and the bands are far
      wider than the gaps between predictors — the quantitative reason the
      leaderboard keeps reshuffling.
- [ ] 15. **Null distribution of k/6** — score N random tickets to get the full
      distribution, then report each predictor's empirical percentile/p-value.
      Much more informative than comparing to a single baseline number.
- [ ] 16. **Position autocorrelation** — corr(pos i at draw t, pos i at draw
      t−lag) for lags 1..10, per position. Expect ≈0: the draw has no memory.
- [ ] 17. **Adjacent-gap distribution** — d_i = x_(i+1) − x_(i) observed vs
      theoretical. A different lens on the same structure.
- [x] 18. **Log-loss / Brier scoring of the grid** ⭐ — `src/ml/proper.py`,
      `run.py proper-score`, dashboard card, `tests/test_proper.py`.
      Result (6/55, 200 draws): floor 3.3623 nats; theory 3.3548, shrunk 3.3801,
      empirical 3.3811, empirical-100 3.5625, uniform 4.0073 (= ln 55 exactly,
      a nice implementation check). Nothing beats the floor; learning from
      history lands a hair *above* the closed-form law (estimation noise), and a
      100-draw window is clearly worse — more data helps you estimate a fixed
      law, not predict a draw. Same story for 6/45 (floor 3.1496, uniform
      3.8067 = ln 45).
- [ ] 19. **Position error profile** — for each position, distribution of
      (predicted − actual), and a confusion-style band chart. Shows models are
      right about the *shape* and helpless about the *value*.

## B. Models / decision rules (produce the ticket)

- [x] 20. **Optimal assignment** ⭐ — `src/ml/optimal.py`, `run.py optimal-ticket`,
      `tests/test_optimal.py`. Not Hungarian in the end: the ascending constraint
      turns the assignment into a chain, so an exact O(N·k) DP
      (`dp[p][v] = grid[v][p] + max_{v'<v} dp[p-1][v']`) solves it — simpler, and
      it respects the ordering that unconstrained Hungarian would ignore.

      **Result: no gain — greedy was already optimal.** On theory, empirical and
      shrunk grids for both games the two tickets are byte-identical
      (6/55 → `[1, 11, 29, 34, 47, 55]`, gain +0.00000). The reason is a property
      of the order-statistic law: its columns are ordered by monotone likelihood
      ratio, so each position's argmax is already increasing and greedy never
      steals a number a later position needed.

      The DP is verified, not vacuous: `test_optimal.py` shows it strictly beats
      greedy on an adversarial grid (where greedy grabs a number position 1
      needed) and matches brute-force enumeration exactly on a random grid.
      Deliberately **not** registered as a predictor — it would duplicate
      `joint-grid` and add a fake name to the leaderboard.
- [ ] 21. **Bayesian shrinkage grid** — Dirichlet–multinomial with the
      closed-form grid as prior; posterior = shrunk empirical grid. The
      principled fix for overfitting sparse cells, and shrinkage strength
      becomes a tunable knob.
- [ ] 22. **Positional Markov chain** — discrete P(x_i | x_(i−1)) transition
      matrices + Viterbi/beam search over ascending paths. Differs from the
      existing `chain-ridge` (regression) by being fully probabilistic.
- [ ] 23. **Quantile / median regression per position** — predict the median or
      mode rather than the mean; better matched to an exact-match metric than
      ridge's squared-error target.
- [ ] 24. **Global beam search** — instead of choosing each position greedily,
      search the top-B ascending combinations by total grid log-probability.
- [ ] 25. **Copula / order-statistic resampling** — sample tickets preserving
      the dependence between positions, not just the marginals.
- [ ] 26. **LightGBM per-position classifier** — stronger learner than the
      current logistic `perpos-clf`; useful precisely because it will *also*
      land on the baseline.
- [ ] 27. **Neural sequence model** — small LSTM/Transformer over the draw
      sequence. The most persuasive negative result for anyone who assumes deep
      learning would find something. (Needs torch — heavy.)
- [ ] 28. **Genetic-algorithm ticket** — optimize a ticket directly against
      historical k/6. Will look brilliant in-sample and collapse out-of-sample:
      the clearest overfitting demonstration in the project.
- [ ] 29. **Hidden Markov model** — latent-state model over positions; another
      "sophisticated method, same ceiling" data point.

**Suggested order:** 18 → 14 → 20 → 21 (better metric, then a reference line,
then a provably better decision rule, then better estimation), with 15 and 12
as quick analytics wins alongside.

---

_This file is the place for future ideas — add them above._
