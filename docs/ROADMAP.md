# Roadmap

Planned enhancements, to build one at a time. Everything stays honest: the
lottery is uniform-random, so these deepen the *analysis and the lesson*, not
the odds.

## Status (tick off as shipped)

- [x] 1. Uniformity test — `src/randomness.py`, `run.py uniformity`, dashboard card
- [ ] 2. Bankroll / EV simulator
- [ ] 3. Accuracy trend over time
- [ ] 4. Test suite
- [ ] 5. More games (Keno / Power 5/35)
- [ ] 6. Notifications (Telegram)

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

## 5. More games (config-only-ish)  ·  small

**Goal:** add Keno and/or Power 5/35.

**Build:**
- Add `Product` entries in `config.py` (endpoint key, range, main_count,
  draw schedule). Keno needs its own request/parse shape (many draws/day).
- Verify the crawler parse for the new response; seed some history.

---

## 6. Notifications  ·  small, needs a secret

**Goal:** get the daily prediction + last result without opening the dashboard.

**Build:**
- `src/notify.py`: send a message (Telegram bot is simplest) with the next
  draw's consensus + per-model picks and the latest actual result.
- A workflow step after `daily`, gated on a `TELEGRAM_TOKEN` / `CHAT_ID`
  secret (skips cleanly if unset).

---

## Suggested build order

1 (uniformity) → 2 (bankroll) → 3 (trend) → 4 (tests) → 5 (games) → 6 (notify).

1–3 reinforce the honest core and are the most interesting; 4 is cheap
insurance; 5–6 are nice-to-haves.
