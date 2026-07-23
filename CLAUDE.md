# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rule: check the roadmap

`docs/ROADMAP.md` tracks planned-but-not-yet-built enhancements. **Before
wrapping up a task or when asked "what's next", consult `docs/ROADMAP.md`** to
see if anything was forgotten or is ready to build, and keep it up to date:
tick off items as they ship and add new ideas as they come up.

## What this is

An educational project that crawls Vietnamese lottery results for **Power 6/55** and **Power 6/45**, computes descriptive statistics, and builds two position-based models "for fun". The models deliberately do not — and cannot — improve real odds; lottery draws are uniform random. Keep any code and docs honest about this: evaluation is designed to *demonstrate* the absence of an edge (mean hits vs the random baseline, with bootstrap CIs).

## Commands

Uses [uv](https://docs.astral.sh/uv/) for environment/dependency management. `uv run` auto-syncs before running. The positional model needs the ML extras (`uv sync --extra ml`); everything else runs on the core deps.

```bash
uv sync                                              # core deps
uv sync --extra ml                                   # + numpy/scikit-learn (positional model)
uv run python run.py crawl                           # crawl both games, latest page
uv run python run.py analyze power_655               # print statistics
uv run python run.py ml-predict-pos power_655        # positional (ordered) ticket
uv run python run.py ml-predict-joint power_655      # joint number x position ticket
uv run python run.py ml-backtest-pos power_655 --model both
uv run python run.py ml-backtest-joint power_655
uv run python run.py ml-loop                         # score past predictions + predict next
uv run python run.py daily                           # crawl + analyze + report + loop + dashboard
```

Tests live in `tests/` (pytest). Run with `uv sync --extra ml --extra dev` then `uv run pytest`; `.github/workflows/tests.yml` runs them on push. They cover the fragile bits: HTML parsing, the draw schedule, scoring math, model ticket validity, and the stats/bankroll sanity. Also verify behaviour by running the CLI commands above.

Product keys are `power_655` and `power_645` everywhere (CLI args, `PRODUCTS` dict). Crawling requires direct network access to `vietlott.vn`; analysis and prediction work offline against the seeded `data/*.jsonl`.

## Architecture

`run.py` is the only entrypoint. It inserts `src/` onto `sys.path` and imports the core modules by bare name (`config`, `crawler`, `analyze`, `dashboard`). The ML code lives in the `ml` package (`src/ml/`), imported as `from ml import ...`; within `ml`, modules import the top-level ones bare (`from config import ...`) since `src/` is on the path.

Data flows one direction:

- **`src/config.py`** — the single source of truth. The frozen `Product` dataclass holds each game's ajaxpro `url`, magic `key`, `array_rows`, number range, `main_count`, and draw schedule (`draw_days`, `draw_hour`, `next_draw_date()` on Vietnam time). `PRODUCTS` maps name → Product. Adding/changing a game happens only here.
- **`src/crawler.py`** — POSTs to the `ashx` ajaxpro endpoint, which returns JSON wrapping an HTML table fragment (`value.HtmlContent`). `_parse_html` scrapes rows, `crawl()` merges deduping by draw `id`, sorts by `(date, id)`, rewrites the JSONL. The scraping is the fragile part most likely to break if the site changes.
- **`src/analyze.py`** — pure functions over loaded draws: `frequency`, `recent_frequency`, `hot_cold`, `days_since_last`, bundled by `summary()`.
- **`src/ml/positional.py`** — sorts each draw ascending and trains one regressor per ordered position (`ridge`/`gb`); `predict_next` assembles an ascending ticket; `backtest` reports mean hits + bootstrap CI + per-position MAE. Needs numpy/scikit-learn.
- **`src/ml/joint.py`** — pure-Python MLE of the grid `P(number k at position p)`; verifies it against the closed-form order-statistic law; `predict_next` picks a per-position-mode ticket; `backtest` reports mean hits + CI. No heavy deps.
- **`src/ml/ledger.py` + `score.py`** — the predict→score loop: predictions are logged to `predictions/ledger.jsonl` before a draw, then `score_pending` matches them to real results (hits), and `rebuild_scorecard` writes `predictions/scorecard.json` for the dashboard.
- **`src/dashboard.py`** — renders a self-contained HTML dashboard (number heatmap, number×position map, model scorecard).

### Data model

Stored draw record (one JSON object per line in `data/*.jsonl`):

```json
{"date": "2026-07-18", "id": "01373", "result": [22, 41, 45, 48, 54, 55, 16], "process_time": "..."}
```

`result` is the 6 main numbers **followed by the bonus number** (last element). **Analysis uses only the first `main_count` (6) numbers** — `analyze.load_draws` attaches `r["main"] = result[:main_count]`. 6/45 stores only 6 numbers (no bonus); `_bonus_str` in `run.py` only shows a bonus when `len(result) > main_count`.

`data/*.jsonl` and `predictions/` are committed. Generated `reports/*` and `logs/` are gitignored (see `.gitignore`).

## Windows automation

`daily.bat` (crawl → analyze → loop → report → dashboard, appending to `logs/daily.log`) is the job. `install_schedule.bat` registers a `LuckyDaily` scheduled task (default 21:00). `setup.bat` runs `uv sync`. `train_all.bat` / `predict_all.bat` run all models with live progress. `push_to_github.bat` is a git helper. `.github/workflows/daily.yml` runs the same pipeline in the cloud and publishes the dashboard to GitHub Pages.
