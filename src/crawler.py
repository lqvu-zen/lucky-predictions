"""Crawl Power 6/55 and 6/45 results from the lottery's ajaxpro API.

The public results page (vietlott.vn) loads draw history via an ashx
endpoint that returns JSON containing an HTML fragment (a table of
draws). We POST to that endpoint, parse the table, and append any new
draws to a JSONL file, deduping by draw id.

Each stored record:
    {"date": "YYYY-MM-DD", "id": "01373",
     "result": [n1..n6, bonus], "process_time": ISO8601}

NOTE: This must run on a machine that can reach vietlott.vn directly
(e.g. your Windows PC). It is for educational/personal use only.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import re
import sys

import requests
from bs4 import BeautifulSoup

from config import Product, get_product

TIMEOUT = 25


def _get_cookie(session: requests.Session) -> dict:
    """The ajaxpro endpoint sets an anti-bot cookie via a small JS snippet on
    /ajaxpro/. Fetch it and return it as a cookie dict so the POST is accepted
    (needed on non-Vietnam IPs like GitHub's runners). Best effort."""
    try:
        r = session.get("https://vietlott.vn/ajaxpro/", timeout=TIMEOUT)
        m = re.search(r'document\.cookie="(.*?)"', r.text)
        if m:
            name, _, value = m.group(1).partition("=")
            return {name: value}
    except Exception as e:  # noqa: BLE001
        print(f"[crawl] cookie handshake failed: {type(e).__name__}: {e}",
              file=sys.stderr)
    return {}


def _headers(product: Product) -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) "
            "Gecko/20100101 Firefox/128.0"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "Content-Type": "text/plain; charset=utf-8",
        "X-AjaxPro-Method": "ServerSideDrawResult",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": "https://vietlott.vn",
        "Referer": product.referer,
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }


def _body(product: Product, page_index: int) -> dict:
    return {
        "ORenderInfo": {
            "SiteId": "main.frontend.vi",
            "SiteAlias": "main.vi",
            "UserSessionId": "",
            "SiteLang": "vi",
            "IsPageDesign": False,
            "SiteName": "Vietlott",
            "System": 1,
        },
        "Key": product.key,
        "GameDrawId": "",
        "ArrayNumbers": [["" for _ in range(18)] for _ in range(product.array_rows)],
        "CheckMulti": False,
        "PageIndex": page_index,
    }


def _parse_html(html: str) -> list[dict]:
    """Parse the results table HTML fragment into draw records."""
    soup = BeautifulSoup(html or "", "lxml")
    rows: list[dict] = []
    for i, tr in enumerate(soup.select("table tr")):
        if i == 0:  # header row
            continue
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue
        try:
            date = datetime.strptime(tds[0].text.strip(), "%d/%m/%Y").strftime("%Y-%m-%d")
        except ValueError:
            continue
        draw_id = tds[1].text.strip()
        numbers = [
            int(span.text)
            for span in tds[2].find_all("span")
            if span.text.strip() and span.text.strip() != "|"
        ]
        if not numbers:
            continue
        rows.append({
            "date": date,
            "id": draw_id,
            "result": numbers,
            "process_time": datetime.now().isoformat(),
        })
    return rows


def fetch_pages(product: Product, index_from: int = 0, index_to: int = 1) -> list[dict]:
    """Fetch draws from page index_from up to and including index_to.

    Page 0 is the most recent draws. Each page holds several draws.
    Returns a flat list of draw records (may contain duplicates across pages).
    """
    session = requests.Session()
    cookies = _get_cookie(session)          # anti-bot cookie handshake
    all_rows: list[dict] = []
    for page in range(index_from, index_to + 1):
        resp = session.post(
            product.url,
            data=json.dumps(_body(product, page)),
            headers=_headers(product),
            cookies=cookies,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        try:
            html = resp.json().get("value", {}).get("HtmlContent", "")
        except ValueError:
            # not JSON — usually an anti-bot / challenge page; surface it
            print(f"[crawl] {product.name} page {page}: non-JSON response "
                  f"({len(resp.text)} chars) — likely blocked/needs cookie",
                  file=sys.stderr)
            continue
        rows = _parse_html(html)
        if not rows:
            print(f"[crawl] {product.name} page {page}: 0 rows parsed",
                  file=sys.stderr)
        all_rows.extend(rows)
    return all_rows


def _load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _save(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def crawl(product_name: str, index_from: int = 0, index_to: int = 1) -> dict:
    """Crawl a product, merge with stored data, persist. Returns a summary dict."""
    product = get_product(product_name)
    fetched = fetch_pages(product, index_from, index_to)

    existing = _load_existing(product.raw_path)
    existing_ids = {r["id"] for r in existing}

    new_rows = []
    seen = set(existing_ids)
    for r in fetched:
        if r["id"] not in seen:
            new_rows.append(r)
            seen.add(r["id"])

    merged = existing + new_rows
    merged.sort(key=lambda r: (r["date"], r["id"]))
    _save(product.raw_path, merged)

    return {
        "product": product.label,
        "fetched": len(fetched),
        "new": len(new_rows),
        "total": len(merged),
        "latest": merged[-1] if merged else None,
        "new_draws": new_rows,
    }


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "power_655"
    to = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    summary = crawl(name, 0, to)
    print(json.dumps({k: v for k, v in summary.items() if k != "new_draws"},
                     ensure_ascii=False, indent=2, default=str))
