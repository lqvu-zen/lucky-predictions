# Lucky Predictions

📊 **Live dashboard:** https://lqvu-zen.github.io/lucky-predictions/
_(updates automatically every evening after the draw)_

A small, self-contained project for **learning about web crawling, data
analysis, and honest ML evaluation** using Vietnamese lottery data. It fetches
draw results for **Power 6/55** and **Power 6/45 (Mega)**, computes statistics,
builds six position-based models for fun — and then spends most of its effort
**proving, from several independent angles, that none of them work**.

> ⚠️ **Honest disclaimer.** A lottery draw is a uniform random selection —
> every combination is equally likely on every draw. **No analysis or model
> here can improve your odds of winning.** This project is purely for learning
> (crawling, data wrangling, statistics, ML evaluation) and entertainment.
> Please gamble responsibly, if at all.

## What it does

1. **Crawl** — pulls the latest results from the lottery site's `ajaxpro` API
   and stores them as JSONL, deduping by draw id.
2. **Analyze** — frequency (all-time & recent), hot/cold numbers, and
   "days since last appearance" (overdue) stats.
3. **Predict** — two position-based models (see below) produce a ticket.
4. **Score** — every prediction is logged *before* its draw, then scored
   against the real result to keep a rolling, honest scorecard.
5. **Test** — statistical tests that check whether any of it means anything
   (see below). This is the part that matters.
6. **Dashboard** — a self-contained HTML page with a number heatmap, a
   number×position map, the model scorecard, and every test result.

## What the analysis actually found

The models are the excuse; these are the results. All figures are Power 6/55.

| Question | Answer |
|---|---|
| Does number K land at position I as theory predicts? | Yes. \|z\|>2 in 5.0% of cells (expect 4.6%), largest \|z\| = 3.28 → p = 0.27 across 300 cells |
| Does any model beat the information floor? | No. Entropy floor 3.3623 nats; theory 3.3548, learned-from-history 3.3811, uniform 4.0073 |
| How good could a *perfect* model be? | k/6 score **0.0651**, vs **0.0423** for a random ticket. Perfect knowledge is worth +0.023 |
| Is any predictor beating chance? | No. Best raw p = 0.0125, but across 13 predictors that corrects to **Šidák p = 0.151** |
| Does one draw predict the next? | No. 60 autocorrelations, strongest r = −0.059 → Šidák p = 0.82; overlap distribution χ² p = 0.35 |
| Can you "find" a winning ticket in past data? | Yes, and it means nothing — see below |
| What is a line worth? | **−7,620 VND** on a 10,000 VND ticket (−76%), identical for all 28,989,675 tickets |

The overfitting demonstration (`run.py overfit`) is the clearest lesson. A
ticket optimised against N past draws, then scored on 200 it never saw:

| training draws | score on them | score on unseen | gap |
|---:|---:|---:|---:|
| 10 | **0.2167** | 0.0392 | +0.1775 |
| 30 | 0.1500 | 0.0642 | +0.0858 |
| 100 | 0.1117 | 0.0442 | +0.0675 |
| 800 | 0.0729 | 0.0567 | +0.0163 |

With 10 draws it scores **3.3× the theoretical ceiling** — impossible for real
knowledge — then performs *worse than random* on new draws. The gap closes as
data grows. Overfitting isn't a flaw of one algorithm; it's the ratio of freedom
to evidence. Any backtest without held-out data can be made to look brilliant.

## The models

All of them reason about **where each number lands** in a draw sorted
ascending (p1 < p2 < … < p6):

- **Positional (ordered)** — 6 regressors (`ridge`/`gb`), one per ordered
  position, predict that position's value.
- **Joint number×position** — the counted grid `P(number k at position p)`;
  matches the closed-form order-statistic law and renders as a heatmap.
- **Gap / spacing** — Ridge regressors on the gaps between consecutive numbers
  (p1, p2−p1, …), reconstructed into a ticket by cumulative sum.
- **Conditional (autoregressive)** — predicts p1, then each position from the
  previous one, respecting the ascending order.
- **Per-position classifier** — a LogisticRegression per position predicting
  the number; the trained ML version of the joint grid.
- **Empirical sampler** — samples each position from its real distribution
  (varied but position-realistic tickets).

Plus the **for-fun heuristic lines** (`random / hot / cold / overdue /
balanced`). Every model lands on the random baseline in evaluation — that's the
honest result; the value is the pipeline and the measurement. See
[docs/how-the-model-works.md](docs/how-the-model-works.md).

## Project layout

```
lucky-predictions/
├── run.py               # CLI entrypoint
├── pyproject.toml
├── src/
│   ├── config.py        # product definitions (655 & 645), draw schedule, paths
│   ├── crawler.py       # fetch + parse + store draws
│   ├── analyze.py       # statistics
│   ├── randomness.py    # uniformity / odd-even / repeat tests
│   ├── bankroll.py      # what playing along would have cost
│   ├── jackpot.py       # jackpot expectation reality-check
│   ├── ticket_ev.py     # exact EV of one line (hypergeometric)
│   └── ml/
│       ├── positional.py    # ordered-position model (ridge / gb)
│       ├── joint.py         # joint number×position grid
│       ├── gap.py           # spacing model
│       ├── chain.py         # conditional / autoregressive
│       ├── perpos.py        # per-position classifier
│       ├── sampler.py       # empirical position sampler
│       ├── proper.py        # log-loss / Brier scoring of the grid
│       ├── ceiling.py       # best possible score + noise bands + p-values
│       ├── optimal.py       # provably optimal ticket from a grid (DP)
│       ├── residual.py      # observed grid vs theory, as a test
│       ├── independence.py  # does one draw predict the next?
│       ├── genetic.py       # GA ticket — the overfitting demo
│       ├── ledger.py        # logs predictions before each draw
│       ├── score.py         # scores them after, builds the scorecard
│       └── util.py          # progress bar helper
├── data/                # power655.jsonl, power645.jsonl (seeded history)
├── predictions/         # ledger + scored + scorecard (the honest record)
└── reports/             # generated dashboard + daily reports
```

Each stored draw:

```json
{"date": "2026-07-18", "id": "01373", "result": [22, 41, 45, 48, 54, 55, 16], "process_time": "..."}
```

`result` holds the **6 main numbers followed by the bonus** (last element);
6/45 stores only the 6 main numbers. Analysis uses the 6 main numbers.

## Setup

Uses [uv](https://docs.astral.sh/uv/). Install it once (PowerShell):

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Then:

```bash
cd D:\Projects\lucky-predictions
uv sync                # core deps (crawl / analyze / joint / dashboard)
uv sync --extra ml     # adds numpy + scikit-learn for the positional model
```

The `data/` folder is seeded with history, so everything works immediately.

## Usage

```bash
# Fetch the latest draws for both games
uv run python run.py crawl

# Statistics
uv run python run.py analyze power_655

# Predict the next draw
uv run python run.py ml-predict-pos   power_655        # positional (ridge)
uv run python run.py ml-predict-pos   power_655 --model gb
uv run python run.py ml-predict-joint power_655        # joint grid

# Evaluate (walk-forward backtest with bootstrap CIs)
uv run python run.py ml-backtest-pos   power_655 --model both
uv run python run.py ml-backtest-joint power_655

# The honest analysis — this is the interesting part
uv run python run.py proper-score  power_655   # log-loss / Brier vs the entropy floor
uv run python run.py ceiling       power_655   # best possible score, noise bands, p-values
uv run python run.py residual      power_655   # does number K at position I match theory?
uv run python run.py independence  power_655   # does one draw predict the next?
uv run python run.py overfit       power_655   # in-sample brilliance vs out-of-sample reality
uv run python run.py ticket-ev     power_655   # what one line is actually worth
uv run python run.py optimal-ticket power_655  # greedy vs provably-optimal ticket
uv run python run.py uniformity    power_655   # is the draw random?
uv run python run.py jackpot       power_655   # how long to win the jackpot?

# Score past predictions + log the next one (updates the scorecard)
uv run python run.py ml-loop

# Everything: crawl + analyze + report + loop + dashboard
uv run python run.py daily

# Rebuild the dashboard only
uv run python run.py dashboard
```

**Convenience scripts (Windows):** double-click `train_all.bat` to
evaluate both models on both games with live progress, or `predict_all.bat`
to print every model's next-draw prediction.

## How predictions are scored — the leaderboard

Every predictor — all 5 for-fun strategies **and** all position-based models —
logs one ticket per draw, and each is scored two ways once the real result
arrives:

- **Hits** — number overlap: how many of the ticket's 6 numbers came up (0–6).
  Random baseline `6×6/N` (0.655 for 6/55, 0.800 for 6/45).
- **Pos-hits** — position accuracy: how many are the *correct number at the
  correct sorted position* (`ticket[i] == actual[i]`). Harder; the best a
  mode-guesser can average is shown as the reference baseline.

The dashboard ranks all predictors into a **leaderboard** by mean pos-hits, and
orders the next draw's tickets by each predictor's best-ever k/6 and how often it
reached it. The honest catch: the leader keeps reshuffling, and the project now
quantifies exactly why. Each predictor is compared against the range a
*perfect* model would land in over that predictor's own number of scored draws —
every one sits inside its band, and the bands are far wider than the gaps between
predictors.

The sharpest illustration is the p-values. One strategy scored p = 0.0125 against
a skill-free null — by the usual α = 0.05 that reads as a discovery. But it is
the best of **13** predictors, and correcting for that gives **Šidák p = 0.151**.
Test enough candidates and one always looks gifted. The dashboard says so right
next to the number.

Because k/6 is so noisy (six positions, mostly missed), models are also graded by
**log-loss and Brier score** on the whole predicted distribution — a proper
scoring rule that separates models with far less data. Nothing beats the entropy
floor of the true draw law.

Tests live in `tests/` (61 of them, pytest). They cover the fragile parts —
HTML parsing, the draw schedule, scoring math, ticket validity — and the
statistics: that the optimal-ticket DP matches brute force, that the
independence test catches a *planted* dependency, and that overfitting shrinks as
training data grows.

## Draw schedule & automation

Draws (Vietnam time, encoded in `config.py`): **6/55 Tue/Thu/Sat 18:00**,
**6/45 Wed/Fri/Sun 18:00**. A run after 18:00 catches that night's result.

**The PC crawls; the cloud only deploys.** The lottery site returns `403` to
GitHub Actions runner IPs, so a cloud crawl cannot work — the request itself is
fine (the unrelated `vietvudanh/vietlott-data` project sends an identical POST
and also crawls from its own machine, never from Actions). Don't "fix" the
crawler in response to a 403.

**Local (Windows) — the only thing that crawls.**

1. `setup.bat` — `uv sync` to build the environment.
2. `daily.bat` — crawl → report → predict/score → dashboard, printing live
   progress while appending to `logs/daily.log`; then commits `data/` +
   `predictions/`, `git pull --rebase --autostash`, and pushes.
3. `install_schedule.bat` — registers the `LuckyDaily` task (default 21:00).

**Cloud (GitHub Actions) — rebuild + publish only.** No `schedule:` cron. A push
to `data/**` or `predictions/**` triggers `run.py daily --no-crawl`, which
rebuilds the dashboard from the committed data and publishes it to GitHub Pages.
It deliberately commits nothing back, which would collide with the PC's next
push. To retest whether the block ever lifts, run the workflow manually with the
`try_crawl` input set to true.

Enable publishing via **Settings → Pages → Source → GitHub Actions**.

## License / disclaimer

For personal, educational use only. Not affiliated with any lottery operator.
No gambling advice is provided.
