# Lucky Predictions

📊 **Live dashboard:** https://lqvu-zen.github.io/lucky-predictions/
_(updates automatically every evening after the draw)_

A small, self-contained project for **learning about web crawling, data
analysis, and honest ML evaluation** using Vietnamese lottery data. It
automatically fetches daily draw results for **Power 6/55** and **Power 6/45
(Mega)**, computes statistics, and builds two position-based models — for fun.

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
5. **Dashboard** — a self-contained HTML page with a number heatmap, a
   number×position map, and the model scorecard.

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
│   └── ml/
│       ├── positional.py  # ordered-position model (ridge / gb)
│       ├── joint.py       # joint number×position grid
│       ├── ledger.py      # logs predictions before each draw
│       ├── score.py       # scores them after, builds the scorecard
│       └── util.py        # progress bar helper
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

The dashboard ranks all predictors into a **leaderboard** by mean pos-hits, so
you can see who's ahead. The honest catch: the leader keeps reshuffling and
over enough draws every predictor converges — there is no real edge, so
"who's best" is luck. Backtests (`ml-backtest-*`) add bootstrap 95% CIs that
confirm each model's CI straddles the baseline.

## Draw schedule & automation

Draws (Vietnam time, encoded in `config.py`): **6/55 Tue/Thu/Sat 18:00**,
**6/45 Wed/Fri/Sun 18:00**. A run after 18:00 catches that night's result.

**Local (Windows), 3 double-clicks:**

1. `setup.bat` — `uv sync` to build the environment.
2. `install_schedule.bat` — registers the `LuckyDaily` task at **21:00**.
3. `daily.bat` — the job itself (also runs on demand).

**Cloud (GitHub Actions):** `.github/workflows/daily.yml` runs the whole
pipeline nightly, commits new draws + predictions, and publishes the dashboard
to GitHub Pages. Enable it via **Settings → Pages → Source → GitHub Actions**
and **Settings → Actions → General → Workflow permissions → Read and write**.

> The crawl must run where it can reach the lottery site directly.

## License / disclaimer

For personal, educational use only. Not affiliated with any lottery operator.
No gambling advice is provided.
