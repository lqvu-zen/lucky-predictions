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

_This file is the place for future ideas — add them above._
