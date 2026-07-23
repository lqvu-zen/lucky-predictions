"""Product configuration for lottery crawling and analysis.

Focused on the two jackpot games: Power 6/55 and Power 6/45 (Mega).
Each draw result stores 7 numbers: the 6 main numbers plus 1 bonus
number (the last element). Only the 6 main numbers count for analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Draws are on Vietnam time (UTC+7, no daylight saving). Compute the
# schedule against this fixed offset so it's correct anywhere it runs —
# including GitHub Actions runners, which use UTC.
VN_TZ = timezone(timedelta(hours=7))

# Project root = parent of this file's directory (src/)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
PRED_DIR = ROOT / "predictions"

# Python weekday(): Monday=0 ... Sunday=6
_WD = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}


@dataclass(frozen=True)
class Product:
    name: str            # internal key, e.g. "power_655"
    label: str           # human label, e.g. "Power 6/55"
    url: str             # ajaxpro endpoint
    key: str             # magic "Key" value the endpoint expects
    array_rows: int      # ArrayNumbers row count (5 for 655, 6 for 645)
    min_value: int
    max_value: int
    main_count: int      # numbers that count (6); result also has 1 bonus
    referer: str
    draw_days: tuple = ()   # weekdays draws happen (Mon=0..Sun=6)
    draw_hour: int = 18     # draws at 18:00 Vietnam time
    ticket_cost: int = 10000            # VND per line
    # prize (VND) by number of matched main numbers; jackpot is a nominal fixed
    # figure since the real one rolls over. Used only by the bankroll simulator.
    prize_tiers: tuple = ()             # ((match, payout), ...)

    @property
    def raw_path(self) -> Path:
        return DATA_DIR / f"{self.name.replace('power_', 'power')}.jsonl"

    def next_draw_date(self, ref: datetime | None = None) -> date:
        """The date of the next draw at/after `ref` (default: now).

        If today is a draw day and it's before the draw hour, today counts;
        otherwise it rolls forward to the next scheduled draw day.
        """
        ref = ref or datetime.now(VN_TZ)
        today = ref.date()
        for offset in range(0, 8):
            d = today + timedelta(days=offset)
            if d.weekday() in self.draw_days:
                if offset == 0 and ref.hour >= self.draw_hour:
                    continue  # today's draw already happened
                return d
        return today  # unreachable if draw_days non-empty


_BASE = "https://vietlott.vn/ajaxpro/Vietlott.PlugIn.WebParts."

POWER_655 = Product(
    name="power_655",
    label="Power 6/55",
    url=_BASE + "Game655CompareWebPart,Vietlott.PlugIn.WebParts.ashx",
    key="23bbd667",
    array_rows=5,
    min_value=1,
    max_value=55,
    main_count=6,
    referer="https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/winning-number-655",
    draw_days=(_WD["Tue"], _WD["Thu"], _WD["Sat"]),
    prize_tiers=((3, 50_000), (4, 500_000), (5, 40_000_000), (6, 30_000_000_000)),
)

POWER_645 = Product(
    name="power_645",
    label="Power 6/45",
    url=_BASE + "Game645CompareWebPart,Vietlott.PlugIn.WebParts.ashx",
    key="8290fce2",
    array_rows=6,
    min_value=1,
    max_value=45,
    main_count=6,
    referer="https://vietlott.vn/vi/trung-thuong/ket-qua-trung-thuong/winning-number-645",
    draw_days=(_WD["Wed"], _WD["Fri"], _WD["Sun"]),
    prize_tiers=((3, 30_000), (4, 300_000), (5, 10_000_000), (6, 12_000_000_000)),
)

PRODUCTS = {p.name: p for p in (POWER_655, POWER_645)}


def get_product(name: str) -> Product:
    if name not in PRODUCTS:
        raise ValueError(f"Unknown product '{name}'. Choices: {list(PRODUCTS)}")
    return PRODUCTS[name]
