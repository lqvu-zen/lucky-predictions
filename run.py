#!/usr/bin/env python3
"""Vietlott predictions — command-line entrypoint.

Usage:
    python run.py crawl                 # crawl both games (latest page)
    python run.py crawl power_655 --pages 3
    python run.py analyze power_655
    python run.py predict power_655 --strategy balanced --tickets 5
    python run.py daily                 # crawl + analyze + predict + write report

Run from the project root. Requires: requests, beautifulsoup4, lxml.

⚠️  For education/fun only. Lottery draws are random; suggested tickets
    cannot beat the odds.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# make src/ importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from config import PRODUCTS, REPORTS_DIR, get_product  # noqa: E402
import analyze  # noqa: E402
import backtest  # noqa: E402
import dashboard  # noqa: E402
import predict  # noqa: E402


def _fmt_line(nums: list[int]) -> str:
    return " - ".join(f"{n:02d}" for n in nums)


def _bonus_str(result: list[int], main_count: int) -> str:
    """Return ' (bonus N)' only when the draw actually has a bonus number.

    Power 6/55 draws store 7 numbers (6 main + bonus); Power 6/45 stores
    only the 6 main numbers, so there is no bonus to show.
    """
    return f" (bonus {result[-1]})" if len(result) > main_count else ""


def cmd_crawl(args) -> None:
    import crawler
    names = [args.product] if args.product else list(PRODUCTS)
    for name in names:
        try:
            s = crawler.crawl(name, 0, args.pages)
            latest = s["latest"]
            mc = get_product(name).main_count
            print(f"[{s['product']}] +{s['new']} new (total {s['total']}). "
                  f"Latest: {latest['date']} #{latest['id']} "
                  f"{_fmt_line(latest['result'][:mc])}"
                  f"{_bonus_str(latest['result'], mc)}")
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] crawl FAILED: {type(e).__name__}: {e}", file=sys.stderr)


def cmd_analyze(args) -> None:
    s = analyze.summary(args.product)
    if not s.get("draws"):
        print(f"No data for {args.product}. Run `python run.py crawl` first "
              f"(or seed data/).")
        return
    print(f"\n{s['product']} — {s['draws']} draws "
          f"({s['date_range'][0]} → {s['date_range'][1]})")
    print(f"Latest: {s['latest']['date']} #{s['latest']['id']} "
          f"{_fmt_line(s['latest']['main'])}")
    print("\nMost common (all time):",
          ", ".join(f"{n}({c})" for n, c in s["most_common"]))
    print("Hot — last 30 draws:   ",
          ", ".join(f"{n}({c})" for n, c in s["hot_30"]))
    print("Cold — last 30 draws:  ",
          ", ".join(f"{n}({c})" for n, c in s["cold_30"]))
    print("Most overdue (days):   ",
          ", ".join(f"{n}({g}d)" for n, g in s["most_overdue"]))


def cmd_predict(args) -> None:
    if args.strategy == "all":
        allsug = predict.suggest_all(args.product, tickets=args.tickets, seed=args.seed)
        print(f"\n{get_product(args.product).label} — suggested lines:")
        for strat, lines in allsug.items():
            for ln in lines:
                print(f"  {strat:9s}: {_fmt_line(ln)}")
    else:
        r = predict.suggest(args.product, args.strategy, args.tickets, seed=args.seed)
        print(f"\n{r['product']} — {r['strategy']} (window {r['window']} draws):")
        for ln in r["tickets"]:
            print(f"  {_fmt_line(ln)}")
    print("\n(For fun only — these cannot improve real odds.)")


def _report_section(name: str, seed: int | None) -> str:
    p = get_product(name)
    s = analyze.summary(name)
    if not s.get("draws"):
        return f"## {p.label}\n\n_No data yet._\n"
    allsug = predict.suggest_all(name, tickets=1, seed=seed)
    lines = [
        f"## {p.label}",
        "",
        f"- **Draws on record:** {s['draws']} "
        f"({s['date_range'][0]} → {s['date_range'][1]})",
        f"- **Latest draw:** {s['latest']['date']} #{s['latest']['id']} — "
        f"`{_fmt_line(s['latest']['main'])}`"
        f"{_bonus_str(s['latest']['result'], p.main_count)}",
        "",
        "**Hot (last 30 draws):** "
        + ", ".join(f"{n} ({c})" for n, c in s["hot_30"]),
        "",
        "**Cold (last 30 draws):** "
        + ", ".join(f"{n} ({c})" for n, c in s["cold_30"]),
        "",
        "**Most overdue:** "
        + ", ".join(f"{n} ({g}d)" for n, g in s["most_overdue"]),
        "",
        "### Suggested lines (one per strategy)",
        "",
        "| Strategy | Numbers |",
        "| --- | --- |",
    ]
    for strat, sug in allsug.items():
        lines.append(f"| {strat} | `{_fmt_line(sug[0])}` |")
    lines.append("")
    return "\n".join(lines)


def cmd_backtest(args) -> None:
    bt = backtest.run(args.product, warmup=args.warmup,
                      tickets=args.tickets, window=args.window, seed=args.seed)
    print()
    print(backtest.format_report(bt))


def cmd_dashboard(args) -> None:
    path = dashboard.build()
    print(f"[dashboard] wrote {path}")


def cmd_daily(args) -> None:
    # 1) crawl (best effort — skip on network failure so report still runs)
    try:
        import crawler
        for name in PRODUCTS:
            try:
                s = crawler.crawl(name, 0, args.pages)
                print(f"[crawl] {s['product']}: +{s['new']} new (total {s['total']})")
            except Exception as e:  # noqa: BLE001
                print(f"[crawl] {name} skipped: {type(e).__name__}: {e}",
                      file=sys.stderr)
    except ImportError:
        print("[crawl] crawler deps missing, skipping crawl", file=sys.stderr)

    # 2) build report
    today = datetime.now().strftime("%Y-%m-%d")
    seed = int(datetime.now().strftime("%Y%m%d"))  # stable within a day
    parts = [
        f"# Vietlott daily report — {today}",
        "",
        "> ⚠️ Lottery draws are random. The statistics below describe the past "
        "and the suggested lines are for fun only — they cannot improve your odds.",
        "",
    ]
    for name in PRODUCTS:
        parts.append(_report_section(name, seed))
    report = "\n".join(parts).rstrip() + "\n"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"report_{today}.md"
    out.write_text(report, encoding="utf-8")
    latest = REPORTS_DIR / "latest.md"
    latest.write_text(report, encoding="utf-8")
    print(f"[report] wrote {out}")

    # 3) refresh the HTML dashboard
    print(f"[dashboard] wrote {dashboard.build()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Vietlott crawl + analyze + predict")
    sub = ap.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("crawl", help="fetch latest results")
    pc.add_argument("product", nargs="?", choices=list(PRODUCTS), default=None)
    pc.add_argument("--pages", type=int, default=1,
                    help="how many pages back to fetch (0=latest only)")
    pc.set_defaults(func=cmd_crawl)

    pa = sub.add_parser("analyze", help="print statistics")
    pa.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    pa.set_defaults(func=cmd_analyze)

    pp = sub.add_parser("predict", help="suggest ticket lines")
    pp.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    pp.add_argument("--strategy", default="balanced",
                    choices=predict.STRATEGIES + ["all"])
    pp.add_argument("--tickets", type=int, default=3)
    pp.add_argument("--seed", type=int, default=None)
    pp.set_defaults(func=cmd_predict)

    pb = sub.add_parser("backtest", help="score strategies against history")
    pb.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    pb.add_argument("--warmup", type=int, default=200)
    pb.add_argument("--tickets", type=int, default=10)
    pb.add_argument("--window", type=int, default=60)
    pb.add_argument("--seed", type=int, default=0)
    pb.set_defaults(func=cmd_backtest)

    pdash = sub.add_parser("dashboard", help="generate reports/dashboard.html")
    pdash.set_defaults(func=cmd_dashboard)

    pd = sub.add_parser("daily", help="crawl + analyze + predict + report + dashboard")
    pd.add_argument("--pages", type=int, default=1)
    pd.set_defaults(func=cmd_daily)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
