from datetime import datetime

import pytest

from config import PRODUCTS, get_product


def test_products_present():
    assert set(PRODUCTS) == {"power_655", "power_645"}


def test_unknown_product_raises():
    with pytest.raises(ValueError):
        get_product("nope")


def test_prize_tiers_and_cost():
    p = get_product("power_655")
    tiers = dict(p.prize_tiers)
    assert tiers[6] > tiers[5] > tiers[4] > tiers[3] > 0
    assert p.ticket_cost > 0


def test_next_draw_date_schedule():
    p = get_product("power_655")  # draws Tue/Thu/Sat 18:00 VN
    # Monday midday -> next is Tuesday
    assert p.next_draw_date(datetime(2026, 7, 20, 12, 0)).isoformat() == "2026-07-21"
    # Tuesday after the draw hour -> rolls to Thursday
    assert p.next_draw_date(datetime(2026, 7, 21, 19, 0)).isoformat() == "2026-07-23"
