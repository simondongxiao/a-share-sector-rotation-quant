from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
NORMALIZED = ROOT / "data" / "normalized"


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_size(text: str) -> tuple[float | None, str | None]:
    text = text or ""
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(亿|万亿|万元|万份|份)", text)
    if not m:
        return None, None
    value = float(m.group(1))
    unit = m.group(2)
    if unit == "万亿":
        return value * 1e12, unit
    if unit == "亿":
        return value * 1e8, unit
    if unit == "万元":
        return value * 1e4, unit
    return value, unit


def parse_date(text: str) -> str | None:
    m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text or "")
    if not m:
        return None
    return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"


def parse_page(code: str, timeout: int = 20) -> dict:
    url = f"https://fundf10.eastmoney.com/jbgk_{code}.html"
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.content.decode("utf-8", errors="replace"), "html.parser")
    pairs: dict[str, str] = {}
    for row in soup.select("tr"):
        cells = [" ".join(cell.stripped_strings) for cell in row.select("th,td")]
        for idx in range(0, len(cells) - 1, 2):
            key = cells[idx].strip()
            value = cells[idx + 1].strip()
            if key and value and key not in pairs:
                pairs[key] = value

    def value_containing(*needles: str) -> str:
        for key, value in pairs.items():
            if any(needle in key for needle in needles):
                return value
        return ""

    tracking_label = None
    tracking_value = None
    for key in ("跟踪标的", "业绩比较基准", "标的指数"):
        if key in pairs:
            tracking_label = key
            tracking_value = pairs[key]
            break
    inception_text = value_containing("成立日期")
    size_text = value_containing("资产规模", "净资产规模")
    size_cny, size_unit = parse_size(size_text)
    return {
        "code": code,
        "inception_date": parse_date(inception_text),
        "fund_full_name": pairs.get("基金全称"),
        "fund_short_name": pairs.get("基金简称"),
        "fund_category": pairs.get("基金类型"),
        "asset_size_text": size_text,
        "asset_size_cny": size_cny,
        "asset_size_unit": size_unit,
        "asset_size_asof": parse_date(size_text),
        "tracking_label": tracking_label,
        "tracking_index": tracking_value,
        "tracking_status": "verified_page_field" if tracking_value else "missing_page_field",
        "tracking_source": url,
        "checked_at": now_text(),
        "error": None,
    }


def fetch_one(code: str, retries: int = 3) -> dict:
    last_error = None
    for attempt in range(retries):
        try:
            return parse_page(code)
        except Exception as exc:
            last_error = repr(exc)
            time.sleep(0.5 * (attempt + 1))
    return {
        "code": code,
        "inception_date": None,
        "fund_full_name": None,
        "fund_short_name": None,
        "fund_category": None,
        "asset_size_text": None,
        "asset_size_cny": None,
        "asset_size_unit": None,
        "asset_size_asof": None,
        "tracking_label": None,
        "tracking_index": None,
        "tracking_status": "request_error",
        "tracking_source": f"https://fundf10.eastmoney.com/jbgk_{code}.html",
        "checked_at": now_text(),
        "error": last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(NORMALIZED / "etf_universe.csv"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    frame = pd.read_csv(args.input, dtype={"code": str})
    codes = frame["code"].astype(str).str.zfill(6).drop_duplicates().tolist()
    results = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 16))) as pool:
        futures = {pool.submit(fetch_one, code): code for code in codes}
        for idx, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if idx % 100 == 0 or idx == len(codes):
                print(f"progress {idx}/{len(codes)}")
    review = pd.DataFrame(results).sort_values("code")
    review.to_csv(NORMALIZED / "etf_tracking_index_review.csv", index=False, encoding="utf-8-sig")
    merged = frame.drop(columns=[c for c in review.columns if c in frame.columns and c != "code"], errors="ignore").merge(review, on="code", how="left")
    merged["size_threshold_cny"] = 100_000_000
    merged["size_eligible_current"] = merged["asset_size_cny"] >= merged["size_threshold_cny"]
    merged["inception_date"] = pd.to_datetime(merged["inception_date"], errors="coerce")
    merged["backtest_start"] = pd.to_datetime("2015-01-01")
    merged.loc[merged["inception_date"] > merged["backtest_start"], "backtest_start"] = merged.loc[merged["inception_date"] > merged["backtest_start"], "inception_date"]
    merged["backtest_start"] = merged["backtest_start"].dt.date
    merged.to_csv(NORMALIZED / "etf_universe_with_tracking_review.csv", index=False, encoding="utf-8-sig")
    print("saved", NORMALIZED / "etf_tracking_index_review.csv")
    print("saved", NORMALIZED / "etf_universe_with_tracking_review.csv")


if __name__ == "__main__":
    main()
