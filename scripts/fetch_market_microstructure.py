"""Build sector microstructure inputs for the dynamic exit engine.

This pipeline intentionally separates:
* historical, reproducible price-return estimates of sector limit-down breadth;
* recent Eastmoney limit-down-pool observations with real sealed-order money;
* recent 30-minute ETF bars used for next-session repair confirmation.

The historical estimate is not presented as an official limit-down pool. It is
tagged with method/data-quality fields so the backtest can distinguish it from
the recent high-fidelity observations.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import akshare as ak
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "microstructure"
STOCK_RAW = RAW / "stock_daily"
NORMALIZED = ROOT / "data" / "normalized"
RAW.mkdir(parents=True, exist_ok=True)
STOCK_RAW.mkdir(parents=True, exist_ok=True)
NORMALIZED.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("microstructure")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def fetch_sw_constituents() -> pd.DataFrame:
    info = ak.sw_index_first_info()
    rows: list[pd.DataFrame] = []
    for record in info.itertuples(index=False, name=None):
        index_code = str(record[0]).split(".")[0].zfill(6)
        industry = str(record[1])
        try:
            frame = ak.index_component_sw(index_code)
            if frame.empty:
                continue
            code_col = frame.columns[1]
            asof_col = frame.columns[-1]
            out = pd.DataFrame({
                "industry_code": index_code,
                "industry": industry,
                "stock_code": frame[code_col].astype(str).str.extract(r"(\d{6})")[0],
                "constituent_asof": pd.to_datetime(frame[asof_col], errors="coerce"),
                "source": "akshare:index_component_sw",
                "retrieved_at": _now_utc(),
            })
            rows.append(out.dropna(subset=["stock_code"]))
        except Exception as exc:  # pragma: no cover - provider/network dependent
            LOG.warning("constituent fetch failed %s: %s", index_code, exc)
    if not rows:
        raise RuntimeError("No Shenwan constituent data returned")
    result = pd.concat(rows, ignore_index=True).drop_duplicates(["industry_code", "stock_code"])
    result.to_csv(NORMALIZED / "shenwan_level1_constituents.csv", index=False, encoding="utf-8-sig")
    return result


def _eastmoney_hist(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    return ak.stock_zh_a_hist(code, period="daily", start_date=start_date, end_date=end_date, adjust="")


def _tencent_hist(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    prefix = "sh" if code.startswith(("6", "68")) else "sz"
    return ak.stock_zh_a_hist_tx(prefix + code, start_date, end_date, adjust="")


def _normalize_stock_history(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    columns = {str(c): c for c in frame.columns}
    date_col = next((columns[c] for c in columns if c in {"日期", "date"}), frame.columns[0])
    open_col = next((columns[c] for c in columns if c in {"开盘", "open"}), None)
    high_col = next((columns[c] for c in columns if c in {"最高", "high"}), None)
    low_col = next((columns[c] for c in columns if c in {"最低", "low"}), None)
    close_col = next((columns[c] for c in columns if c in {"收盘", "close"}), None)
    amount_col = next((columns[c] for c in columns if c in {"成交额", "amount"}), None)
    if any(col is None for col in [open_col, high_col, low_col, close_col]):
        # Tencent's schema is already canonical.
        if set(["date", "open", "high", "low", "close"]).issubset(frame.columns):
            date_col, open_col, high_col, low_col, close_col = "date", "open", "high", "low", "close"
            amount_col = "amount" if "amount" in frame.columns else None
        else:
            return pd.DataFrame()
    out = pd.DataFrame({
        "stock_code": code,
        "date": pd.to_datetime(frame[date_col], errors="coerce"),
        "open": pd.to_numeric(frame[open_col], errors="coerce"),
        "high": pd.to_numeric(frame[high_col], errors="coerce"),
        "low": pd.to_numeric(frame[low_col], errors="coerce"),
        "close": pd.to_numeric(frame[close_col], errors="coerce"),
        "amount": pd.to_numeric(frame[amount_col], errors="coerce") if amount_col is not None else np.nan,
    })
    out = out.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
    out["daily_return"] = out["close"].pct_change()
    out["shadow_ratio"] = (out["close"] - out["low"]) / (out["high"] - out["low"]).replace(0, np.nan)
    return out


def fetch_one_stock(code: str, start_date: str, end_date: str, retries: int = 3) -> tuple[str, str, int]:
    target = STOCK_RAW / f"{code}.csv.gz"
    if target.exists():
        try:
            cached = pd.read_csv(target, compression="gzip", usecols=["date"])
            return code, "cached", int(len(cached))
        except Exception:
            target.unlink(missing_ok=True)
    last_error = ""
    for attempt in range(retries):
        try:
            raw = _eastmoney_hist(code, start_date, end_date)
            normalized = _normalize_stock_history(raw, code)
            if normalized.empty:
                raw = _tencent_hist(code, start_date, end_date)
                normalized = _normalize_stock_history(raw, code)
            if not normalized.empty:
                normalized.to_csv(target, index=False, compression="gzip")
                return code, "ok", int(len(normalized))
            last_error = "empty"
        except Exception as exc:  # pragma: no cover - provider/network dependent
            last_error = f"{type(exc).__name__}:{exc}"
            # Eastmoney is often unavailable behind a corporate proxy. Try
            # Tencent in the same attempt so the historical job can continue.
            try:
                raw = _tencent_hist(code, start_date, end_date)
                normalized = _normalize_stock_history(raw, code)
                if not normalized.empty:
                    normalized.to_csv(target, index=False, compression="gzip")
                    return code, "ok_tencent_fallback", int(len(normalized))
            except Exception as fallback_exc:  # pragma: no cover
                last_error = f"{last_error};fallback={type(fallback_exc).__name__}:{fallback_exc}"
            time.sleep(0.5 * (attempt + 1))
    return code, f"error:{last_error[:160]}", 0


def limit_threshold(code: str, date: pd.Timestamp) -> float:
    """Return the approximate down-limit magnitude for common A-share boards."""
    # ChiNext switched to 20% on 2020-08-24; STAR has been 20% since launch.
    if code.startswith("688") or (code.startswith(("300", "301")) and date >= pd.Timestamp("2020-08-24")):
        return 0.195
    # The historical pipeline excludes BSE codes; 10% is the conservative base.
    return 0.095


def aggregate_sector_microstructure(constituents: pd.DataFrame) -> pd.DataFrame:
    membership = constituents.groupby("stock_code")["industry"].apply(list).to_dict()
    industry_sizes = constituents.groupby("industry")["stock_code"].nunique().to_dict()
    buckets: dict[tuple[str, pd.Timestamp], dict[str, float]] = {}
    for idx, path in enumerate(STOCK_RAW.glob("*.csv.gz"), start=1):
        code = path.name.split(".", 1)[0]
        industries = membership.get(code, [])
        if not industries:
            continue
        try:
            frame = pd.read_csv(path, compression="gzip", parse_dates=["date"])
        except Exception:
            continue
        for row in frame.itertuples(index=False):
            date = pd.Timestamp(row.date)
            ret = _safe_float(row.daily_return)
            if pd.isna(ret):
                continue
            threshold = limit_threshold(code, date)
            down = float(ret <= -threshold)
            for industry in industries:
                key = (industry, date)
                item = buckets.setdefault(key, {"available": 0.0, "limit_down": 0.0, "return_sum": 0.0, "shadow_sum": 0.0, "shadow_n": 0.0, "amount_sum": 0.0})
                item["available"] += 1
                item["limit_down"] += down
                item["return_sum"] += ret
                shadow = _safe_float(row.shadow_ratio)
                if pd.notna(shadow):
                    item["shadow_sum"] += shadow
                    item["shadow_n"] += 1
                amount = _safe_float(row.amount)
                if pd.notna(amount):
                    item["amount_sum"] += amount
        if idx % 500 == 0:
            LOG.info("aggregated stock histories: %s", idx)
    rows = []
    for (industry, date), item in buckets.items():
        expected = industry_sizes.get(industry, np.nan)
        rows.append({
            "industry": industry,
            "date": date,
            "available_constituents": int(item["available"]),
            "expected_constituents_snapshot": int(expected) if pd.notna(expected) else None,
            "limit_down_count": int(item["limit_down"]),
            "limit_down_ratio": item["limit_down"] / item["available"] if item["available"] else np.nan,
            "mean_constituent_return": item["return_sum"] / item["available"] if item["available"] else np.nan,
            "mean_shadow_ratio": item["shadow_sum"] / item["shadow_n"] if item["shadow_n"] else np.nan,
            "constituent_amount_cny": item["amount_sum"],
            "sealed_amount_cny": np.nan,
            "limit_down_method": "price_return_estimate",
            "sealed_amount_method": "unavailable_historical",
            "data_quality": "estimated_breadth",
        })
    result = pd.DataFrame(rows).sort_values(["date", "industry"])
    result.to_csv(NORMALIZED / "shenwan_level1_microstructure_daily.csv", index=False, encoding="utf-8-sig")
    return result


def fetch_recent_limit_down_pool(dates: list[pd.Timestamp], constituents: pd.DataFrame) -> pd.DataFrame:
    code_to_industry = constituents.groupby("stock_code")["industry"].first().to_dict()
    rows: list[dict[str, Any]] = []
    for date in dates:
        date_text = date.strftime("%Y%m%d")
        try:
            frame = ak.stock_zt_pool_dtgc_em(date_text)
            for record in frame.itertuples(index=False, name=None):
                if len(record) < 12:
                    continue
                code = str(record[1]).zfill(6)
                rows.append({
                    "date": date,
                    "stock_code": code,
                    "industry": code_to_industry.get(code),
                    "change_pct": _safe_float(record[3]),
                    "amount_cny": _safe_float(record[5]),
                    "sealed_amount_cny": _safe_float(record[10]),
                    "last_sealed_time": record[11],
                    "source": "akshare:stock_zt_pool_dtgc_em",
                    "retrieved_at": _now_utc(),
                })
        except Exception as exc:  # pragma: no cover - provider/network dependent
            LOG.warning("limit-down pool unavailable for %s: %s", date_text, exc)
    result = pd.DataFrame(rows)
    if result.empty:
        result = pd.DataFrame(columns=["date", "stock_code", "industry", "change_pct", "amount_cny", "sealed_amount_cny", "last_sealed_time", "source", "retrieved_at"])
    result.to_csv(NORMALIZED / "limit_down_pool_recent.csv", index=False, encoding="utf-8-sig")
    if not result.empty:
        agg = result.groupby(["date", "industry"], dropna=False).agg(
            official_limit_down_count=("stock_code", "nunique"),
            sealed_amount_cny=("sealed_amount_cny", "sum"),
        ).reset_index()
        agg.to_csv(NORMALIZED / "shenwan_level1_microstructure_recent_official.csv", index=False, encoding="utf-8-sig")
    return result


def fetch_recent_30m_etf() -> pd.DataFrame:
    mapping = pd.read_csv(NORMALIZED / "etf_shenwan_level1_mapping_draft.csv", dtype={"code": str})
    mapping["code"] = mapping["code"].str.zfill(6)
    mapping["asset_size_cny"] = pd.to_numeric(mapping["asset_size_cny"], errors="coerce")
    mapping = mapping[(mapping["mapping_status"] == "candidate") & (mapping["size_eligible_current"] == True)].copy()  # noqa: E712
    mapping = mapping.sort_values("asset_size_cny", ascending=False).drop_duplicates("shenwan_level1")
    rows = []
    for row in mapping.itertuples(index=False):
        prefix = "sh" if row.code.startswith(("5", "6")) else "sz"
        try:
            frame = ak.stock_zh_a_minute(prefix + row.code, period="30", adjust="")
            if frame.empty:
                continue
            frame = frame.rename(columns={"day": "datetime"})
            frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
            for col in ["open", "high", "low", "close", "volume", "amount"]:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame["code"] = row.code
            frame["industry"] = row.shenwan_level1
            frame["recovery_30m"] = (frame["close"] > frame["open"]) & (frame["close"] >= frame["high"].rolling(2, min_periods=1).max())
            rows.append(frame[["datetime", "code", "industry", "open", "high", "low", "close", "volume", "amount", "recovery_30m"]])
        except Exception as exc:  # pragma: no cover - provider/network dependent
            LOG.warning("30m fetch failed %s: %s", row.code, exc)
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    result.to_csv(NORMALIZED / "etf_30m_recent.csv", index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="20150101")
    parser.add_argument("--end-date", default=pd.Timestamp.now().strftime("%Y%m%d"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-history", action="store_true")
    parser.add_argument("--skip-30m", action="store_true")
    args = parser.parse_args()

    constituents = fetch_sw_constituents()
    stock_codes = sorted(constituents["stock_code"].dropna().unique())
    if not args.skip_history:
        statuses = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(fetch_one_stock, code, args.start_date, args.end_date): code for code in stock_codes}
            for idx, future in enumerate(as_completed(futures), start=1):
                statuses.append(future.result())
                if idx % 100 == 0:
                    ok = sum(s[1] in {"ok", "cached"} for s in statuses)
                    LOG.info("stock histories %s/%s, usable=%s", idx, len(futures), ok)
        pd.DataFrame(statuses, columns=["stock_code", "status", "rows"]).to_csv(
            NORMALIZED / "stock_history_download_log.csv", index=False, encoding="utf-8-sig"
        )
        aggregate_sector_microstructure(constituents)

    dates = list(pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=35))
    fetch_recent_limit_down_pool(dates, constituents)
    if not args.skip_30m:
        fetch_recent_30m_etf()

    manifest = {
        "retrieved_at": _now_utc(),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "constituent_count": int(len(stock_codes)),
        "industry_count": int(constituents["industry"].nunique()),
        "historical_limit_down_method": "price_return_estimate",
        "recent_limit_down_source": "Eastmoney limit-down pool via AkShare, recent window only",
        "intraday_source": "Sina minute endpoint via AkShare, recent bars only",
        "notes": [
            "Historical sealed amount is unavailable from free daily OHLC and remains null.",
            "Current SW constituent snapshot is used for the historical breadth estimate; this is not a point-in-time membership reconstruction.",
        ],
    }
    (NORMALIZED / "microstructure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
