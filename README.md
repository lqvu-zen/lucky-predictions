# Lucky Predictions

A small, self-contained project for **learning about web crawling and data
analysis** using Vietnamese lottery (Vietlott) data. It automatically fetches
daily draw results for **Power 6/55** and **Power 6/45 (Mega)**, computes
statistics, and generates "for fun" suggested ticket lines.

> ⚠️ **Honest disclaimer.** A lottery draw is a uniform random selection —
> every combination is equally likely on every draw. **No analysis or model
> here can improve your odds of winning.** This project is purely for learning
> (crawling, data wrangling, statistics) and entertainment. Please gamble
> responsibly, if at all.

Inspired by [vietvudanh/vietlott-data](https://github.com/vietvudanh/vietlott-data).

## What it does

1. **Crawl** — pulls the latest results from Vietlott's `ajaxpro` API
   (the same endpoint the official results page uses) and stores them as
   JSONL, deduping by draw id.
2. **Analyze** — frequency (all-time & recent), hot/cold numbers, and
   "days since last appearance" (overdue) stats.
3. **Predict** — five transparent strategies (`random`, `hot`, `cold`,
   `overdue`, `balanced`) that turn those stats into suggested lines.
4. **Daily report** — one command crawls, analyzes, and writes a dated
   Markdown report to `reports/`.

## Project layout

```
lucky-predictions/
├── run.py              # CLI entrypoint (crawl / analyze / predict / daily)
├── requirements.txt
├── src/
│   ├── config.py       # product definitions (655 & 645) + paths
│   ├── crawler.py      # fetch + parse + store draws
│   ├── analyze.py      # statistics
│   └── predict.py      # ticket-suggestion strategies
├── data/               # power655.jsonl, power645.jsonl (seeded from reference)
└── reports/            # generated daily reports
```

Each stored draw looks like:

```json
{"date": "2026-07-18", "id": "01373", "result": [22, 41, 45, 48, 54, 55, 16], "process_time": "..."}
```

`result` holds the **6 main numbers followed by the bonus number** (last
element). Analysis uses only the 6 main numbers.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for environment and
dependency management. Install uv once (PowerShell):

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Then, from the project root:

```bash
cd D:\Projects\lucky-predictions
uv sync           # creates .venv and installs dependencies from pyproject.toml
```

The `data/` folder is already seeded with history through mid-July 2026, so
analysis and prediction work immediately — even before your first crawl.

## Usage

`uv run` executes inside the project environment (and auto-syncs if needed):

```bash
# Fetch the latest draws for both games
uv run python run.py crawl

# Fetch more history (e.g. 3 pages back) for one game
uv run python run.py crawl power_655 --pages 3

# Show statistics
uv run python run.py analyze power_655

# Suggest lines
uv run python run.py predict power_655 --strategy balanced --tickets 5
uv run python run.py predict power_645 --strategy all

# Do everything and write reports/report_YYYY-MM-DD.md
uv run python run.py daily

# Generate the HTML dashboard
uv run python run.py dashboard

# Backtest the strategies against real history
uv run python run.py backtest power_655
```

## Machine-learning experiments (`ml`)

An optional ML pipeline treats prediction as N binary questions — for each
number, *what's the probability it appears in the next draw?* — then ranks
the top 6. For a full walkthrough (with a worked example) see
[docs/how-the-model-works.md](docs/how-the-model-works.md). It exists to practice real ML engineering (leakage-safe features,
versioned models, walk-forward evaluation, a predict→score loop), not to
beat the lottery. It won't: the harness is built to *measure* that honestly.

Install the extra dependencies first:

```bash
uv sync --extra ml
```

Commands:

```bash
# Walk-forward evaluation (logistic regression, gradient boosting, or both)
uv run python run.py ml-backtest power_655 --model both

# Compare models with 95% confidence intervals (the honest tuning verdict)
uv run python run.py ml-compare power_655 --models logreg,gb,rf

# See which features a model leans on (add --permutation for a robust view)
uv run python run.py ml-importance power_655 --model rf

# Predict the next scheduled draw and log it to the prediction ledger
uv run python run.py ml-predict power_655

# The full feedback loop: score any predictions whose draw has happened,
# then predict the next draw for both games with both models
uv run python run.py ml-loop
```

**Draw schedule** is encoded in `config.py` (6/55 Tue/Thu/Sat 18:00, 6/45
Wed/Fri/Sun 18:00), so `ml-predict` targets the correct upcoming draw and
records it in `predictions/ledger.jsonl` *before* the result exists — the
basis for honest scoring later.

**What the evaluation shows.** Beyond raw hit-counts, it reports a **Brier
score** and **log-loss** on the per-number probabilities — the real test of
whether the probabilities are informative — against the trivial "every
number = 6/N" predictor:

```
Power 6/55 — ML backtest (logreg) over 120 draws
  mean hits (top-6)     0.567   (random baseline 0.655, diff -0.088)
  Brier score           0.09722 (base-rate 0.09719, diff +0.00003)
  log-loss              0.34476 (base-rate 0.34461)
  Verdict: on par with random, as expected. No usable signal.
```

The Brier score matching the base rate to five decimals is the honest
takeaway: the model's probabilities carry no more information than knowing
each number appears 6/N of the time. If a model ever *did* beat baseline,
the harness flags it (and it would almost certainly mean a data leak to fix).

**Tuning, done honestly.** The feature set is deliberately rich — multi-window
frequencies (10/30/50/100), an exponential-decay rate, gap/overdue ratios,
and pair co-occurrence with the previous draw — and three models are on
offer (`logreg`, `gb`, `rf`). `ml-compare` puts them head to head with
**bootstrap 95% confidence intervals**, the only fair way to tell a real
gain from luck:

```
Power 6/55 — model comparison over 90 draws (95% bootstrap CIs)
Random baseline: 0.655 hits/ticket · base-rate Brier 0.09719

model            mean hits [95% CI]             Brier [95% CI]  verdict
--------------------------------------------------------------------------
logreg          0.611 [0.456,0.767]     0.0972 [0.0903,0.1044]  ≈ random
gb              0.667 [0.522,0.822]     0.0973 [0.0904,0.1045]  ≈ random
rf              0.667 [0.522,0.811]     0.0973 [0.0904,0.1044]  ≈ random
```

Every interval straddles the baseline. Adding features and stronger models
changes nothing — there is no signal to find in a uniform draw, and now we
can *prove* it rather than assume it. That measurement discipline is the
real deliverable.

**Feature importance.** `ml-importance` shows which signals a model leans on
(tree `feature_importances_`, or standardized coefficients for logistic
regression; `--permutation` adds a model-agnostic view that shuffles each
feature and measures the Brier hit). The random forest leans on `rate_all`,
`pair_lift`, `ema_rate`, and the gap features — but the permutation Δ Brier
for even the top feature is ~0.0001, i.e. removing it barely changes
anything. Importance here reflects how the model *fits history*, not any
ability to predict the next draw.

**The predict→score loop.** `ml-loop` closes the feedback cycle:

1. **Score** — any ledger prediction whose draw has now happened (and been
   crawled) is matched to the real result; hits, Brier, and log-loss are
   written to `predictions/scored.jsonl`.
2. **Predict** — a fresh prediction is logged for the next scheduled draw
   of each game, with both models.
3. **Aggregate** — a rolling `predictions/scorecard.json` is rebuilt and
   surfaced on the dashboard (mean hits vs baseline, Brier vs base rate).

The daily run calls this automatically **when the ML extras are installed**
(it's skipped silently otherwise, so the lightweight cloud crawl is
unaffected). To include it in the GitHub Actions run, change the workflow's
`uv sync` step to `uv sync --extra ml`.

Over time the scorecard is the honest record of how the models actually do
on unseen draws — the data you'd use to decide whether any change is a real
improvement or just noise.

## Does any strategy actually work? (backtest)

`backtest` walks forward through history: at each past draw it builds each
strategy's weights from *only the earlier draws*, generates tickets, and
counts how many of the 6 numbers each got right — then compares the average
to the baseline for uniformly random guessing (`6 × 6 / N` matches per
ticket: ≈0.655 for 6/55, 0.800 for 6/45).

Every strategy lands within noise of that baseline — none has an edge:

```
Power 6/55 — backtest over 1173 draws, 11,730 tickets per strategy
Random baseline (expected matches per ticket): 0.655

strategy    mean hits  vs random   4+ hits
------------------------------------------
cold            0.655     +0.000         8
overdue         0.649     -0.006         7
random          0.649     -0.006         4
balanced        0.646     -0.009         3
hot             0.645     -0.010         7
```

That's the whole point: lottery draws are independent and uniform, so past
frequencies carry no predictive power. The backtest lets you *see* it.

> **Note on running the crawl:** it must run on a machine that can reach
> `vietlott.vn` directly. Draw schedule: 6/55 on Tue/Thu/Sat, 6/45 on
> Wed/Fri/Sun (evenings, Vietnam time), so running `daily` each morning
> picks up the previous night's result.

## Automating the daily run (Windows) — 3 double-clicks

Included batch files handle everything; no command typing needed:

1. **`setup.bat`** — run once. Runs `uv sync` to create the environment and
   install dependencies. (Requires uv — see Setup above.)
2. **`install_schedule.bat`** — run once. Registers a Windows scheduled task
   named `LuckyDaily` that runs every evening at **21:00 (9 PM)**.
   (To use a different time, open the file, change `21:00`, and re-run it.)
3. **`daily.bat`** — the job itself (crawl → analyze → report). The scheduler
   calls it for you; you can also double-click it anytime to run on demand.

Each run writes `reports/report_YYYY-MM-DD.md` and refreshes `reports/latest.md`,
and appends output to `logs/daily.log`.

Manage the task from a terminal if you like:

```bat
schtasks /run    /tn LuckyDaily      REM run it now
schtasks /delete /tn LuckyDaily /f   REM remove the schedule
```

> The PC must be on (not necessarily logged in) at the scheduled time for the
> task to run. Draws happen in the evening, so a morning run catches the
> previous night's result.

## Cloud automation (GitHub Actions)

`.github/workflows/daily.yml` runs the whole pipeline in the cloud, so your
PC doesn't need to be on. Every day at **21:30 Vietnam time** (14:30 UTC) it:

1. crawls the latest 6/55 and 6/45 draws,
2. commits any new draws back into `data/*.jsonl`,
3. builds the dashboard and uploads it as a downloadable **artifact**.

To enable it:

1. Push the repo to GitHub (already done).
2. In the repo: **Settings → Actions → General → Workflow permissions →**
   select **Read and write permissions** (lets the job commit new data).
3. Go to the **Actions** tab, pick **daily-crawl**, and click **Run workflow**
   once to test it (or wait for the schedule).

To view the dashboard: open the workflow run under the **Actions** tab and
download the **vietlott-dashboard** artifact (contains `dashboard.html`).

> This is the private-friendly setup — the dashboard is an artifact, not a
> public web page. A live GitHub Pages URL needs either a public repo or a
> paid plan; if you switch to one of those later, I can add a Pages deploy.
> Note: GitHub pauses scheduled workflows after ~60 days with no repo
> activity — the daily commits keep it alive.

## Strategies (and why none of them "work")

| Strategy | Idea |
| --- | --- |
| `random` | Uniform random pick — the honest baseline. |
| `hot` | Weighted toward numbers drawn most in a recent window. |
| `cold` | Weighted toward numbers drawn least recently. |
| `overdue` | Weighted toward numbers absent the longest. |
| `balanced` | A blend of frequency + overdue signals. |

"Hot", "cold", and "overdue" all rest on the **gambler's fallacy** — the
mistaken belief that past draws influence future ones. They're included
because they're fun to compare, not because they help. The `random`
strategy is, mathematically, exactly as good as the others.

## License / disclaimer

For personal, educational use only. Not affiliated with Vietlott. No
gambling advice is provided.
