# Vietlott Predictions

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
vietlott-predictions/
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
cd D:\Projects\vietlott-predictions
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
```

> **Note on running the crawl:** it must run on a machine that can reach
> `vietlott.vn` directly. Draw schedule: 6/55 on Tue/Thu/Sat, 6/45 on
> Wed/Fri/Sun (evenings, Vietnam time), so running `daily` each morning
> picks up the previous night's result.

## Automating the daily run (Windows) — 3 double-clicks

Included batch files handle everything; no command typing needed:

1. **`setup.bat`** — run once. Runs `uv sync` to create the environment and
   install dependencies. (Requires uv — see Setup above.)
2. **`install_schedule.bat`** — run once. Registers a Windows scheduled task
   named `VietlottDaily` that runs every evening at **21:00 (9 PM)**.
   (To use a different time, open the file, change `21:00`, and re-run it.)
3. **`daily.bat`** — the job itself (crawl → analyze → report). The scheduler
   calls it for you; you can also double-click it anytime to run on demand.

Each run writes `reports/report_YYYY-MM-DD.md` and refreshes `reports/latest.md`,
and appends output to `logs/daily.log`.

Manage the task from a terminal if you like:

```bat
schtasks /run    /tn VietlottDaily      REM run it now
schtasks /delete /tn VietlottDaily /f   REM remove the schedule
```

> The PC must be on (not necessarily logged in) at the scheduled time for the
> task to run. Draws happen in the evening, so a morning run catches the
> previous night's result.

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
