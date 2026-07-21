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
    # Default to a per-draw seed so lines stay locked for a given draw;
    # pass --seed to override for experimentation.
    seed = args.seed
    if seed is None:
        seed = int(get_product(args.product).next_draw_date().strftime("%Y%m%d"))
    if args.strategy == "all":
        allsug = predict.suggest_all(args.product, tickets=args.tickets, seed=seed)
        print(f"\n{get_product(args.product).label} — suggested lines:")
        for strat, lines in allsug.items():
            for ln in lines:
                print(f"  {strat:9s}: {_fmt_line(ln)}")
    else:
        r = predict.suggest(args.product, args.strategy, args.tickets, seed=seed)
        print(f"\n{r['product']} — {r['strategy']} (window {r['window']} draws):")
        for ln in r["tickets"]:
            print(f"  {_fmt_line(ln)}")
    print("\n(For fun only — these cannot improve real odds.)")


def _report_section(name: str) -> str:
    p = get_product(name)
    s = analyze.summary(name)
    if not s.get("draws"):
        return f"## {p.label}\n\n_No data yet._\n"
    # Seed by the next draw date so the suggested lines are locked per draw.
    seed = int(p.next_draw_date().strftime("%Y%m%d"))
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


def _need_ml():
    try:
        import numpy  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        sys.exit("ML features need extra deps. Install with:  uv sync --extra ml")


def cmd_ml_backtest(args) -> None:
    _need_ml()
    from ml import evaluate
    print()
    for kind in (["logreg", "gb"] if args.model == "both" else [args.model]):
        r = evaluate.walk_forward(args.product, kind=kind,
                                  test_draws=args.test, retrain_every=args.retrain)
        print(evaluate.format_report(r))
        print()


def cmd_ml_compare(args) -> None:
    _need_ml()
    from ml import compare
    kinds = args.models.split(",") if args.models else ("logreg", "gb", "rf")
    print()
    print(compare.format_report(
        compare.compare(args.product, kinds=tuple(kinds),
                        test_draws=args.test, retrain_every=args.retrain)))


def cmd_ml_importance(args) -> None:
    _need_ml()
    from ml import importance
    print()
    print(importance.format_report(
        importance.importances(args.product, kind=args.model,
                               permutation=args.permutation)))


def cmd_ml_predict(args) -> None:
    _need_ml()
    from ml.predict_next import predict_next
    e = predict_next(args.product, kind=args.model, log=not args.no_log)
    p = get_product(args.product)
    print(f"\n{p.label} — {e['model']} prediction for next draw {e['target_date']}:")
    print("  " + " - ".join(f"{n:02d}" for n in e["top6"]))
    top = sorted(e["probs"].items(), key=lambda kv: -kv[1])[:8]
    print("  most likely (P):", ", ".join(f"{n}={v:.3f}" for n, v in top))
    if not args.no_log:
        print("  logged to predictions/ledger.jsonl")
    print("\n(Reminder: expected to match random odds — logged now so we can score it honestly later.)")


def _run_ml_loop(models=("logreg", "gb"), verbose=True):
    """Score any predictions whose draw has happened, then predict the next
    draw for both games with each model. Returns the fresh scorecard."""
    from ml import ledger, score
    from ml.predict_next import predict_next

    newly = score.score_pending()
    if verbose and newly:
        for s in newly:
            print(f"[ml-score] {s['game']} {s['target_date']} {s['model']}: "
                  f"{s['hits']} hits (top6 {s['top6']} vs {s['actual']})")
    elif verbose:
        print("[ml-score] no predictions ready to score yet")

    pending = ledger.pending_keys()
    for name in PRODUCTS:
        target = get_product(name).next_draw_date().isoformat()
        for kind in models:
            if (name, kind, target) in pending:
                continue  # already predicted this draw
            e = predict_next(name, kind=kind, log=True)
            if verbose:
                print(f"[ml-predict] {name} {kind} -> {target}: {e['top6']}")

    card = score.rebuild_scorecard()
    return card


def cmd_ml_loop(args) -> None:
    _need_ml()
    card = _run_ml_loop(verbose=True)
    print("\nRolling scorecard:")
    for name, g in card["games"].items():
        if not g["models"]:
            print(f"  {g['label']}: nothing scored yet")
            continue
        for kind, m in g["models"].items():
            print(f"  {g['label']} [{kind}]: {m['scored']} scored, "
                  f"mean hits {m['mean_hits']} vs baseline {m['baseline_hits']}, "
                  f"Brier {m['mean_brier']} vs base {m['mean_brier_base']}")


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
    parts = [
        f"# Vietlott daily report — {today}",
        "",
        "> ⚠️ Lottery draws are random. The statistics below describe the past "
        "and the suggested lines are for fun only — they cannot improve your odds.",
        "",
    ]
    for name in PRODUCTS:
        parts.append(_report_section(name))
    report = "\n".join(parts).rstrip() + "\n"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"report_{today}.md"
    out.write_text(report, encoding="utf-8")
    latest = REPORTS_DIR / "latest.md"
    latest.write_text(report, encoding="utf-8")
    print(f"[report] wrote {out}")

    # 3) ML predict->score loop (best effort; skipped if ml deps not installed)
    try:
        import numpy  # noqa: F401
        import sklearn  # noqa: F401
        _run_ml_loop(verbose=True)
    except ImportError:
        print("[ml] extras not installed (uv sync --extra ml) — skipping ML loop")
    except Exception as e:  # noqa: BLE001
        print(f"[ml] loop skipped: {type(e).__name__}: {e}", file=sys.stderr)

    # 4) refresh the HTML dashboard (now includes the ML scorecard)
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

    pmb = sub.add_parser("ml-backtest", help="walk-forward ML evaluation")
    pmb.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    pmb.add_argument("--model", default="logreg", choices=["logreg", "gb", "both"])
    pmb.add_argument("--test", type=int, default=120, help="draws to evaluate")
    pmb.add_argument("--retrain", type=int, default=15, help="retrain cadence")
    pmb.set_defaults(func=cmd_ml_backtest)

    pmc = sub.add_parser("ml-compare", help="compare models with bootstrap CIs")
    pmc.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    pmc.add_argument("--models", default="", help="comma list, e.g. logreg,gb,rf")
    pmc.add_argument("--test", type=int, default=100)
    pmc.add_argument("--retrain", type=int, default=50)
    pmc.set_defaults(func=cmd_ml_compare)

    pmi = sub.add_parser("ml-importance", help="which features the model leans on")
    pmi.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    pmi.add_argument("--model", default="rf", choices=["logreg", "gb", "rf"])
    pmi.add_argument("--permutation", action="store_true",
                     help="also compute model-agnostic permutation importance")
    pmi.set_defaults(func=cmd_ml_importance)

    pmp = sub.add_parser("ml-predict", help="predict + log the next draw")
    pmp.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    pmp.add_argument("--model", default="logreg", choices=["logreg", "gb"])
    pmp.add_argument("--no-log", action="store_true", help="don't write to the ledger")
    pmp.set_defaults(func=cmd_ml_predict)

    pml = sub.add_parser("ml-loop", help="score past predictions + predict next draw")
    pml.set_defaults(func=cmd_ml_loop)

    pdash = sub.add_parser("dashboard", help="generate reports/dashboard.html")
    pdash.set_defaults(func=cmd_dashboard)

    pd = sub.add_parser("daily", help="crawl + analyze + predict + report + dashboard")
    pd.add_argument("--pages", type=int, default=1)
    pd.set_defaults(func=cmd_daily)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
