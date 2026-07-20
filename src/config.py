"""Product configuration for Vietlott crawling and analysis.

Focused on the two jackpot games: Power 6/55 and Power 6/45 (Mega).
Each draw result stores 7 numbers: the 6 main numbers plus 1 bonus
number (the last element). Only the 6 main numbers count for analysis.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Project root = parent of this file's directory (src/)
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"


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

    @property
    def raw_path(self) -> Path:
        return DATA_DIR / f"{self.name.replace('power_', 'power')}.jsonl"


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
)

PRODUCTS = {p.name: p for p in (POWER_655, POWER_645)}


def get_product(name: str) -> Product:
    if name not in PRODUCTS:
        raise ValueError(f"Unknown product '{name}'. Choices: {list(PRODUCTS)}")
    return PRODUCTS[name]
