# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An educational project that crawls Vietlott (Vietnamese lottery) results for **Power 6/55** and **Power 6/45**, computes descriptive statistics, and generates "for fun" suggested ticket lines. The strategies deliberately do not — and cannot — improve real odds; lottery draws are uniform random. Keep any code and docs honest about this: `hot`/`cold`/`overdue` are framed as gambler's-fallacy demonstrations, not advice.

## Commands

Uses [uv](https://docs.astral.sh/uv/) for environment/dependency management. `uv run` auto-syncs before running.

```bash
uv sync                                              # create .venv, install deps
uv run python run.py crawl                           # crawl both games, latest page
uv run python run.py crawl power_655 --pages 3       # fetch 3 pages back for one game
uv run python run.py analyze power_655               # print statistics
uv run python run.py predict power_655 --strategy balanced --tickets 5
uv run python run.py predict power_645 --strategy all
uv run python run.py daily                           # crawl + analyze + write reports/report_<date>.md
```

Each `src/` module is independently runnable for quick checks, e.g. `uv run python src/analyze.py power_655` or `uv run python src/crawler.py power_655 1`.

There is **no test suite, linter config, or CI**. Verify changes by running the commands above.

Product keys are `power_655` and `power_645` everywhere (CLI args, `PRODUCTS` dict). Crawling requires direct network access to `vietlott.vn`; analysis and prediction work offline against the seeded `data/*.jsonl`.

## Architecture

`run.py` is the only entrypoint. It inserts `src/` onto `sys.path` and imports the modules by bare name (`config`, `crawler`, `analyze`, `predict`) — so within `src/` modules import each other bare (`from config import ...`), not as a package. There is no `src/__init__.py`; don't turn `src/` into a package without updating these imports.

Data flows in one direction through four modules:

- **`src/config.py`** — the single source of truth. The frozen `Product` dataclass holds each game's ajaxpro `url`, magic `key`, `array_rows`, number range, and `main_count`. `PRODUCTS` maps name → Product. Adding/changing a game happens only here. `Product.raw_path` derives the JSONL filename (`power_655` → `data/power655.jsonl`).
- **`src/crawler.py`** — POSTs to Vietlott's `ashx` ajaxpro endpoint, which returns JSON wrapping an HTML table fragment (`value.HtmlContent`). `_parse_html` scrapes rows into records, `crawl()` merges with existing data deduping by draw `id`, sorts by `(date, id)`, and rewrites the whole JSONL. The scraping (header format, `<span>` number cells, `dd/mm/yyyy` dates) is the fragile part most likely to break when Vietlott changes their page.
- **`src/analyze.py`** — pure functions over loaded draws: `frequency`, `recent_frequency`, `hot_cold`, `days_since_last`, bundled by `summary()`.
- **`src/predict.py`** — `build_weights` turns stats into per-number weights per strategy (`STRATEGIES`), then `_weighted_sample` draws distinct numbers without replacement. `suggest_all` offsets the seed per strategy so lines aren't identical.

### Data model

Stored draw record (one JSON object per line in `data/*.jsonl`):

```json
{"date": "2026-07-18", "id": "01373", "result": [22, 41, 45, 48, 54, 55, 16], "process_time": "..."}
```

`result` is the 6 main numbers **followed by the bonus number** (last element). **Analysis and prediction use only the first `main_count` (6) numbers** — `analyze.load_draws` attaches `r["main"] = result[:main_count]`. The `_bonus_str` helper in `run.py` only shows a bonus when `len(result) > main_count`.

`data/*.jsonl` is committed and seeded with history. Generated `reports/*.md` and `logs/` are gitignored (see `.gitignore`).

## Windows automation

`daily.bat` (crawl → analyze → report, appending to `logs/daily.log`) is the job. `install_schedule.bat` registers a `VietlottDaily` scheduled task (default 21:00). `setup.bat` runs `uv sync`. `push_to_github.bat` is a git helper. These target Windows; the scheduler calls `daily.bat`, which `cd`s to its own directory first.
