#!/usr/bin/env python3
"""Lucky Predictions — command-line entrypoint.

Usage:
    python run.py crawl                     # crawl both games (latest page)
    python run.py crawl power_655 --pages 3
    python run.py analyze power_655
    python run.py ml-predict-pos power_655  # positional (ordered) ticket
    python run.py ml-predict-joint power_655
    python run.py ml-backtest-pos power_655
    python run.py ml-backtest-joint power_655
    python run.py ml-loop                   # score past predictions + predict next
    python run.py daily                     # crawl + analyze + report + dashboard

Run from the project root. Core needs: requests, beautifulsoup4, lxml.
The positional model additionally needs the ML extras: `uv sync --extra ml`.

For education/fun only. Lottery draws are random; no ticket can beat the odds.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Print UTF-8 regardless of the Windows console code page, so characters like
# the arrow and box-drawing glyphs don't raise UnicodeEncodeError on cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# make src/ importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from config import PRODUCTS, REPORTS_DIR, get_product  # noqa: E402
import analyze  # noqa: E402
import dashboard  # noqa: E402
import predict  # noqa: E402
import randomness  # noqa: E402
import bankroll  # noqa: E402
import jackpot  # noqa: E402


def _fmt_line(nums) -> str:
    return " - ".join(f"{n:02d}" for n in nums)


def _bonus_str(result, main_count: int) -> str:
    """Return ' (bonus N)' only when the draw actually has a bonus number
    (6/55 stores 6 main + bonus; 6/45 stores only the 6 main numbers)."""
    return f" (bonus {result[-1]})" if len(result) > main_count else ""


def _need_ml():
    try:
        import numpy  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        sys.exit("The positional model needs extra deps. Install with:  uv sync --extra ml")


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
        print(f"No data for {args.product}. Run `python run.py crawl` first.")
        return
    print(f"\n{s['product']} - {s['draws']} draws "
          f"({s['date_range'][0]} to {s['date_range'][1]})")
    print(f"Latest: {s['latest']['date']} #{s['latest']['id']} "
          f"{_fmt_line(s['latest']['main'])}")
    print("\nMost common (all time):",
          ", ".join(f"{n}({c})" for n, c in s["most_common"]))
    print("Hot - last 30 draws:   ",
          ", ".join(f"{n}({c})" for n, c in s["hot_30"]))
    print("Cold - last 30 draws:  ",
          ", ".join(f"{n}({c})" for n, c in s["cold_30"]))
    print("Most overdue (days):   ",
          ", ".join(f"{n}({g}d)" for n, g in s["most_overdue"]))


def cmd_predict(args) -> None:
    # Default to a per-draw seed so the fun lines stay locked for a given draw.
    seed = args.seed
    if seed is None:
        seed = int(get_product(args.product).next_draw_date().strftime("%Y%m%d"))
    if args.strategy == "all":
        allsug = predict.suggest_all(args.product, tickets=args.tickets, seed=seed)
        print(f"\n{get_product(args.product).label} - for-fun suggested lines:")
        for strat, lines in allsug.items():
            for ln in lines:
                print(f"  {strat:9s}: {_fmt_line(ln)}")
    else:
        r = predict.suggest(args.product, args.strategy, args.tickets, seed=seed)
        print(f"\n{r['product']} - {r['strategy']} (window {r['window']} draws):")
        for ln in r["tickets"]:
            print(f"  {_fmt_line(ln)}")
    print("\n(For fun only - these cannot improve real odds.)")


def cmd_uniformity(args) -> None:
    print()
    print(randomness.format_report(randomness.summary(args.product)))


def cmd_bankroll(args) -> None:
    print()
    print(bankroll.format_report(bankroll.simulate(args.product)))


def cmd_jackpot(args) -> None:
    print()
    print(jackpot.format_report(jackpot.summary(args.product)))


def cmd_ml_backtest_joint(args) -> None:
    from ml import joint
    print()
    print(joint.format_backtest(joint.backtest(args.product, test_draws=args.test)))


def cmd_ml_predict_joint(args) -> None:
    from ml import joint
    e = joint.predict_next(args.product)
    print(f"\n{e['product']} - joint number x position model, next draw {e['target_date']}:")
    print("  ticket (per-position modes):", _fmt_line(e["ticket"]))
    print(f"  learned grid vs closed-form law: max abs diff = {e['grid_vs_theory_maxdiff']}")
    print("\n(The grid is the fixed order-statistic law - great to visualize, no real edge.)")


def cmd_ml_backtest_pos(args) -> None:
    _need_ml()
    from ml import positional
    print()
    for kind in (["ridge", "gb"] if args.model == "both" else [args.model]):
        print(positional.format_backtest(
            positional.backtest(args.product, kind=kind,
                                test_draws=args.test, retrain_every=args.retrain)))
        print()


def cmd_ml_predict_pos(args) -> None:
    _need_ml()
    from ml import positional
    e = positional.predict_next(args.product, kind=args.model)
    print(f"\n{e['product']} - {e['model']} prediction for next draw {e['target_date']}:")
    print("  ordered ticket:", _fmt_line(e["ticket"]))
    print("  raw position estimates:", e["raw"])
    print("\n(Ordered framing makes prettier tickets, but still can't beat random odds.)")


def cmd_ml_backtest_gap(args) -> None:
    _need_ml()
    from ml import gap
    print()
    print(gap.format_backtest(gap.backtest(args.product, test_draws=args.test)))


def cmd_ml_predict_gap(args) -> None:
    _need_ml()
    from ml import gap
    e = gap.predict_next(args.product)
    print(f"\n{e['product']} - gap/spacing model, next draw {e['target_date']}:")
    print("  ticket:", _fmt_line(e["ticket"]))
    print("  predicted gaps:", e["gaps"])
    print("\n(Models the spacing between numbers - for fun, no real edge.)")


def cmd_ml_backtest_chain(args) -> None:
    _need_ml()
    from ml import chain
    print()
    print(chain.format_backtest(chain.backtest(args.product, test_draws=args.test)))


def cmd_ml_predict_chain(args) -> None:
    _need_ml()
    from ml import chain
    e = chain.predict_next(args.product)
    print(f"\n{e['product']} - conditional (autoregressive) model, next draw {e['target_date']}:")
    print("  ticket:", _fmt_line(e["ticket"]))
    print("\n(Each position predicted from the previous - for fun, no real edge.)")


def cmd_ml_backtest_clf(args) -> None:
    _need_ml()
    from ml import perpos
    print()
    print(perpos.format_backtest(perpos.backtest(args.product, test_draws=args.test)))


def cmd_ml_predict_clf(args) -> None:
    _need_ml()
    from ml import perpos
    e = perpos.predict_next(args.product)
    print(f"\n{e['product']} - per-position classifier, next draw {e['target_date']}:")
    print("  ticket:", _fmt_line(e["ticket"]))
    print("\n(A trained classifier per position - for fun, no real edge.)")


def cmd_ml_backtest_sampler(args) -> None:
    from ml import sampler
    print()
    print(sampler.format_backtest(sampler.backtest(args.product, test_draws=args.test)))


def cmd_ml_predict_sampler(args) -> None:
    from ml import sampler
    e = sampler.predict_next(args.product)
    print(f"\n{e['product']} - empirical position sampler, next draw {e['target_date']}:")
    print("  ticket:", _fmt_line(e["ticket"]))
    print("\n(Sampled from each position's real distribution - for fun, no real edge.)")


def _run_ml_loop(verbose=True):
    """Score predictions whose draw has happened, then log a fresh prediction
    for the next draw of each game (positional ridge/gb + joint). Rebuilds and
    returns the scorecard. Requires the ML extras (positional)."""
    from datetime import datetime as _dt

    from ml import (chain, gap, joint, ledger, perpos, positional, sampler,
                    score)

    newly = score.score_pending()
    if verbose and newly:
        for s in newly:
            print(f"[score] {s['game']} {s['target_date']} {s['model']}: "
                  f"{s['hits']} hits (ticket {s['ticket']} vs {s['actual']})")
    elif verbose:
        print("[score] no predictions ready to score yet")

    pending = ledger.pending_keys()
    version = _dt.now().replace(microsecond=0).isoformat()
    for name in PRODUCTS:
        target = get_product(name).next_draw_date().isoformat()

        for kind in ("ridge", "gb"):
            model = f"positional-{kind}"
            if (name, model, target) in pending:
                continue
            e = positional.predict_next(name, kind=kind)
            ledger.append({"game": name, "target_date": e["target_date"],
                           "model": model, "version": version, "ticket": e["ticket"]})
            if verbose:
                print(f"[predict] {name} {model} -> {target}: {e['ticket']}")

        for mod, model in ((joint, "joint-grid"), (gap, "gap-ridge"),
                           (chain, "chain-ridge"), (perpos, "perpos-clf"),
                           (sampler, "sampler")):
            if (name, model, target) in pending:
                continue
            e = mod.predict_next(name)
            ledger.append({"game": name, "target_date": e["target_date"],
                           "model": model, "version": version, "ticket": e["ticket"]})
            if verbose:
                print(f"[predict] {name} {model} -> {target}: {e['ticket']}")

        # also log the for-fun heuristic strategies so they're evaluated too
        seed = int(get_product(name).next_draw_date().strftime("%Y%m%d"))
        allsug = predict.suggest_all(name, tickets=1, seed=seed)
        for strat, lines in allsug.items():
            model = f"fun-{strat}"
            if (name, model, target) in pending:
                continue
            ledger.append({"game": name, "target_date": target,
                           "model": model, "version": version, "ticket": lines[0]})

        # consensus: the top-k numbers by vote across every other predictor for
        # this draw, saved as its own prediction so it's scored too
        if (name, "consensus", target) not in pending:
            k = get_product(name).main_count
            votes: dict[int, int] = {}
            for e in ledger.load():
                if (e["game"] == name and e["target_date"] == target
                        and e.get("model") != "consensus" and not e.get("scored")):
                    for num in (e.get("ticket") or []):
                        votes[num] = votes.get(num, 0) + 1
            if votes:
                top = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
                ticket = sorted(int(n) for n, _ in top)
                ledger.append({"game": name, "target_date": target,
                               "model": "consensus", "version": version, "ticket": ticket})
                if verbose:
                    print(f"[predict] {name} consensus -> {target}: {ticket}")

    return score.rebuild_scorecard()


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
                  f"mean hits {m['mean_hits']} vs baseline {m['baseline_hits']}")


def _report_section(name: str) -> str:
    p = get_product(name)
    s = analyze.summary(name)
    if not s.get("draws"):
        return f"## {p.label}\n\n_No data yet._\n"
    # joint model is pure-Python, so the report always has a model pick
    try:
        from ml import joint
        pick = _fmt_line(joint.predict_next(name)["ticket"])
    except Exception:
        pick = "_unavailable_"
    seed = int(p.next_draw_date().strftime("%Y%m%d"))
    allsug = predict.suggest_all(name, tickets=1, seed=seed)
    lines = [
        f"## {p.label}",
        "",
        f"- **Draws on record:** {s['draws']} "
        f"({s['date_range'][0]} to {s['date_range'][1]})",
        f"- **Latest draw:** {s['latest']['date']} #{s['latest']['id']} - "
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
        f"**Model pick ({p.next_draw_date()}):** `{pick}`  "
        "_(joint number x position - for fun; see the dashboard for all models)_",
        "",
        f"### For-fun suggested lines for the next draw ({p.next_draw_date()})",
        "",
        "| Strategy | Numbers |",
        "| --- | --- |",
    ]
    for strat, sug in allsug.items():
        lines.append(f"| {strat} | `{_fmt_line(sug[0])}` |")
    lines.append("")
    return "\n".join(lines)


def cmd_dashboard(args) -> None:
    print(f"[dashboard] wrote {dashboard.build()}")


def cmd_daily(args) -> None:
    # 1) crawl (best effort - skip on network failure so the report still runs)
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
        f"# Lucky Predictions daily report - {today}",
        "",
        "> Lottery draws are random. The statistics below describe the past "
        "and the model picks are for fun only - they cannot improve your odds.",
        "",
    ]
    for name in PRODUCTS:
        parts.append(_report_section(name))
    report = "\n".join(parts).rstrip() + "\n"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"report_{today}.md"
    out.write_text(report, encoding="utf-8")
    (REPORTS_DIR / "latest.md").write_text(report, encoding="utf-8")
    print(f"[report] wrote {out}")

    # 3) predict->score loop (best effort; needs the ML extras for positional)
    try:
        import numpy  # noqa: F401
        import sklearn  # noqa: F401
        _run_ml_loop(verbose=True)
    except ImportError:
        print("[ml] extras not installed (uv sync --extra ml) - skipping model loop")
    except Exception as e:  # noqa: BLE001
        print(f"[ml] loop skipped: {type(e).__name__}: {e}", file=sys.stderr)

    # 4) refresh the HTML dashboard (includes the model scorecard)
    print(f"[dashboard] wrote {dashboard.build()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Lottery crawl + analyze + predict")
    sub = ap.add_subparsers(dest="command", required=True)

    pc = sub.add_parser("crawl", help="fetch latest results")
    pc.add_argument("product", nargs="?", choices=list(PRODUCTS), default=None)
    pc.add_argument("--pages", type=int, default=1,
                    help="how many pages back to fetch (0=latest only)")
    pc.set_defaults(func=cmd_crawl)

    pa = sub.add_parser("analyze", help="print statistics")
    pa.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    pa.set_defaults(func=cmd_analyze)

    pp = sub.add_parser("predict", help="for-fun heuristic suggested lines")
    pp.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    pp.add_argument("--strategy", default="all", choices=predict.STRATEGIES + ["all"])
    pp.add_argument("--tickets", type=int, default=3)
    pp.add_argument("--seed", type=int, default=None)
    pp.set_defaults(func=cmd_predict)

    pu = sub.add_parser("uniformity", help="randomness tests on the draw history")
    pu.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    pu.set_defaults(func=cmd_uniformity)

    pbk = sub.add_parser("bankroll", help="simulate buying a line every draw")
    pbk.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    pbk.set_defaults(func=cmd_bankroll)

    pj = sub.add_parser("jackpot", help="jackpot reality-check expectation")
    pj.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    pj.set_defaults(func=cmd_jackpot)

    pbp = sub.add_parser("ml-backtest-pos", help="positional (ordered) model backtest")
    pbp.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    pbp.add_argument("--model", default="ridge", choices=["ridge", "gb", "both"])
    pbp.add_argument("--test", type=int, default=120)
    pbp.add_argument("--retrain", type=int, default=20)
    pbp.set_defaults(func=cmd_ml_backtest_pos)

    ppp = sub.add_parser("ml-predict-pos", help="positional (ordered) next-draw ticket")
    ppp.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    ppp.add_argument("--model", default="ridge", choices=["ridge", "gb"])
    ppp.set_defaults(func=cmd_ml_predict_pos)

    pbj = sub.add_parser("ml-backtest-joint", help="joint number x position backtest")
    pbj.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    pbj.add_argument("--test", type=int, default=120)
    pbj.set_defaults(func=cmd_ml_backtest_joint)

    ppj = sub.add_parser("ml-predict-joint", help="joint number x position next-draw ticket")
    ppj.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    ppj.set_defaults(func=cmd_ml_predict_joint)

    pbg = sub.add_parser("ml-backtest-gap", help="gap/spacing model backtest")
    pbg.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    pbg.add_argument("--test", type=int, default=120)
    pbg.set_defaults(func=cmd_ml_backtest_gap)

    ppg = sub.add_parser("ml-predict-gap", help="gap/spacing next-draw ticket")
    ppg.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    ppg.set_defaults(func=cmd_ml_predict_gap)

    pbc = sub.add_parser("ml-backtest-chain", help="conditional (autoregressive) backtest")
    pbc.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    pbc.add_argument("--test", type=int, default=120)
    pbc.set_defaults(func=cmd_ml_backtest_chain)

    ppc = sub.add_parser("ml-predict-chain", help="conditional (autoregressive) next-draw ticket")
    ppc.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    ppc.set_defaults(func=cmd_ml_predict_chain)

    pbcl = sub.add_parser("ml-backtest-clf", help="per-position classifier backtest")
    pbcl.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    pbcl.add_argument("--test", type=int, default=120)
    pbcl.set_defaults(func=cmd_ml_backtest_clf)

    ppcl = sub.add_parser("ml-predict-clf", help="per-position classifier next-draw ticket")
    ppcl.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    ppcl.set_defaults(func=cmd_ml_predict_clf)

    pbs = sub.add_parser("ml-backtest-sampler", help="empirical position sampler backtest")
    pbs.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    pbs.add_argument("--test", type=int, default=120)
    pbs.set_defaults(func=cmd_ml_backtest_sampler)

    pps = sub.add_parser("ml-predict-sampler", help="empirical position sampler next-draw ticket")
    pps.add_argument("product", nargs="?", choices=list(PRODUCTS), default="power_655")
    pps.set_defaults(func=cmd_ml_predict_sampler)

    pml = sub.add_parser("ml-loop", help="score past predictions + predict next draw")
    pml.set_defaults(func=cmd_ml_loop)

    pdash = sub.add_parser("dashboard", help="generate reports/dashboard.html")
    pdash.set_defaults(func=cmd_dashboard)

    pd = sub.add_parser("daily", help="crawl + analyze + report + loop + dashboard")
    pd.add_argument("--pages", type=int, default=1)
    pd.set_defaults(func=cmd_daily)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
