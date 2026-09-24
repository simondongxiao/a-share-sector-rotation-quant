from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
NORMALIZED = ROOT / "data" / "normalized"
DAILY_DIR = RAW / "etf_daily"
LOG = logging.getLogger("build_etf_universe")


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def fetch_ths_etf_inventory() -> tuple[pd.DataFrame, str]:
    url = "https://fund.10jqka.com.cn/data/Net/info/ETF_rate_desc_0_0_1_9999_0_0_0_jsonp_g.html"
    response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    payload = json.loads(response.text[2:-1])
    rows = []
    for item in payload["data"]["data"].values():
        rows.append(
            {
                "code": str(item.get("code", "")).zfill(6),
                "fund_name": item.get("name"),
                "fund_type": item.get("typename"),
                "unit_nav": item.get("net"),
                "nav_change": item.get("ranges"),
                "nav_change_pct": item.get("rate"),
                "trade_date": item.get("newdate") or item.get("date"),
                "subscription_status": item.get("sgstat"),
                "redemption_status": item.get("shstat"),
            }
        )
    df = pd.DataFrame(rows)
    for col in ["unit_nav", "nav_change", "nav_change_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
    df["exchange"] = df["code"].map(lambda x: "SSE" if x.startswith(("5", "6")) else "SZSE")
    df["source"] = url
    df["retrieved_at"] = now_text()
    return df.sort_values("code").reset_index(drop=True), url


def fetch_sse_scale() -> pd.DataFrame:
    """Fetch the latest SSE ETF shares snapshot; kept optional because the endpoint can lag."""
    url = "https://query.sse.com.cn/commonQuery.do"
    try:
        params = {
            "sqlId": "COMMON_SSE_ZQPZ_ETFZL_XXPL",
            "fundType": "01",
            "pageHelp.pageSize": "5000",
            "pageHelp.pageNo": "1",
            "pageHelp.beginPage": "1",
            "pageHelp.cacheSize": "1",
        }
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.sse.com.cn/"}
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("result", []) if isinstance(payload, dict) else []
        if not rows:
            return pd.DataFrame(columns=["code", "shares", "scale_asof", "scale_source"])
        frame = pd.DataFrame(rows)
        code_col = next((c for c in frame.columns if "CODE" in str(c).upper()), frame.columns[0])
        share_col = next((c for c in frame.columns if "SHARE" in str(c).upper() or "份额" in str(c)), None)
        date_col = next((c for c in frame.columns if "DATE" in str(c).upper() or "日期" in str(c)), None)
        if share_col is None:
            return pd.DataFrame(columns=["code", "shares", "scale_asof", "scale_source"])
        out = pd.DataFrame({"code": frame[code_col].astype(str).str.extract(r"(\d{6})")[0], "shares": pd.to_numeric(frame[share_col], errors="coerce")})
        out["scale_asof"] = frame[date_col] if date_col else None
        out["scale_source"] = url
        return out.dropna(subset=["code"])
    except Exception as exc:
        LOG.warning("SSE scale unavailable: %s", exc)
        return pd.DataFrame(columns=["code", "shares", "scale_asof", "scale_source"])


def fetch_szse_scale() -> pd.DataFrame:
    url = "https://fund.szse.cn/api/report/ShowReport"
    params = {"SHOWTYPE": "xlsx", "CATALOGID": "1000_lf", "TABKEY": "tab1", "random": "0.076103531917917"}
    headers = {"Referer": "https://fund.szse.cn/marketdata/fundslist/index.html", "User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        frame = pd.read_excel(io.BytesIO(response.content), engine="openpyxl")
        code_col = frame.columns[0]
        shares_col = next((c for c in frame.columns if "规模" in str(c) or "SCALE" in str(c).upper()), frame.columns[5])
        date_col = next((c for c in frame.columns if "日期" in str(c) or "DATE" in str(c).upper()), frame.columns[4])
        out = pd.DataFrame({"code": frame[code_col].astype(str).str.extract(r"(\d{6})")[0], "shares": frame[shares_col].astype(str).str.replace(",", "", regex=False).pipe(pd.to_numeric, errors="coerce"), "scale_asof": frame[date_col], "scale_source": url})
        return out.dropna(subset=["code"])
    except Exception as exc:
        LOG.warning("SZSE scale unavailable: %s", exc)
        return pd.DataFrame(columns=["code", "shares", "scale_asof", "scale_source"])


def build_inventory() -> Path:
    NORMALIZED.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    inventory, source = fetch_ths_etf_inventory()
    inventory.to_csv(RAW / "etf_inventory_ths.csv", index=False, encoding="utf-8-sig")
    sse = fetch_sse_scale()
    szse = fetch_szse_scale()
    scale = pd.concat([sse, szse], ignore_index=True).drop_duplicates("code", keep="last")
    if scale.empty:
        inventory["shares"] = pd.NA
        inventory["scale_asof"] = pd.NaT
        inventory["scale_source"] = pd.NA
    else:
        inventory = inventory.merge(scale, on="code", how="left")
    inventory["estimated_size_cny"] = pd.to_numeric(inventory["shares"], errors="coerce") * inventory["unit_nav"]
    inventory["size_threshold_cny"] = 100_000_000
    inventory["size_eligibility"] = inventory["estimated_size_cny"] >= inventory["size_threshold_cny"]
    inventory["size_status"] = inventory["estimated_size_cny"].notna().map({True: "point_in_time_snapshot_available", False: "not_available"})
    inventory["history_start"] = pd.NaT
    inventory["history_end"] = pd.NaT
    inventory["history_status"] = "not_downloaded"
    out = NORMALIZED / "etf_universe.csv"
    inventory.to_csv(out, index=False, encoding="utf-8-sig")
    LOG.info("Saved %s ETF inventory rows to %s", len(inventory), out)
    return out


def exchange_prefix(code: str) -> str:
    return "sh" if str(code).startswith(("5", "6")) else "sz"


def fetch_one_history(code: str, start_date: str, end_date: str, retries: int = 3) -> dict:
    output = DAILY_DIR / f"{code}.csv.gz"
    if output.exists() and output.stat().st_size > 100:
        try:
            cached = pd.read_csv(output, nrows=2)
            return {"code": code, "status": "cached", "rows": -1, "history_start": None, "history_end": None, "error": None}
        except Exception:
            output.unlink(missing_ok=True)
    import akshare as ak

    last_error = None
    for attempt in range(retries):
        try:
            frame = ak.fund_etf_hist_sina(f"{exchange_prefix(code)}{code}")
            if frame is None or frame.empty:
                return {"code": code, "status": "empty", "rows": 0, "history_start": None, "history_end": None, "error": None}
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.dropna(subset=["date"])
            frame = frame[(frame["date"] >= pd.Timestamp(start_date)) & (frame["date"] <= pd.Timestamp(end_date))].copy()
            frame.insert(0, "code", code)
            frame["source"] = "Sina via AkShare fund_etf_hist_sina"
            frame["retrieved_at"] = now_text()
            frame.to_csv(output, index=False, compression="gzip", encoding="utf-8")
            return {"code": code, "status": "downloaded", "rows": len(frame), "history_start": frame["date"].min().date().isoformat() if len(frame) else None, "history_end": frame["date"].max().date().isoformat() if len(frame) else None, "error": None}
        except Exception as exc:
            last_error = repr(exc)
            time.sleep(0.8 * (attempt + 1))
    return {"code": code, "status": "error", "rows": 0, "history_start": None, "history_end": None, "error": last_error}


def download_histories(inventory_path: Path, start_date: str, end_date: str, workers: int) -> Path:
    inventory = pd.read_csv(inventory_path, dtype={"code": str})
    inventory["code"] = inventory["code"].str.zfill(6)
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    codes = inventory["code"].drop_duplicates().tolist()
    LOG.info("Downloading %d ETF histories with %d workers", len(codes), workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one_history, code, start_date, end_date): code for code in codes}
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if idx % 50 == 0 or idx == len(codes):
                ok = sum(r["status"] in {"downloaded", "cached"} for r in results)
                LOG.info("Progress %d/%d; successful=%d; errors=%d", idx, len(codes), ok, sum(r["status"] == "error" for r in results))
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(NORMALIZED / "etf_history_download_log.csv", index=False, encoding="utf-8-sig")
    inventory = inventory.drop(columns=[c for c in ["history_start", "history_end", "history_status"] if c in inventory.columns])
    inventory = inventory.merge(result_frame[["code", "status", "rows", "history_start", "history_end"]], on="code", how="left")
    inventory = inventory.rename(columns={"status": "history_status"})
    inventory.to_csv(NORMALIZED / "etf_universe.csv", index=False, encoding="utf-8-sig")
    return NORMALIZED / "etf_history_download_log.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--download-history", action="store_true")
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default=datetime.now().date().isoformat())
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    inventory_path = build_inventory()
    if args.download_history:
        download_histories(inventory_path, args.start_date, args.end_date, max(1, min(args.workers, 16)))


if __name__ == "__main__":
    main()
