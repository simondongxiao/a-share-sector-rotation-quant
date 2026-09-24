from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NORMALIZED = ROOT / "data" / "normalized"
DAILY_DIR = ROOT / "data" / "raw" / "etf_daily"
REPORTS = ROOT / "reports"


FEE_RATE = 0.0003
START_DATE = pd.Timestamp("2015-01-01")


def load_universe() -> pd.DataFrame:
    frame = pd.read_csv(NORMALIZED / "etf_shenwan_level1_mapping_draft.csv", dtype={"code": str})
    frame["code"] = frame["code"].str.zfill(6)
    frame["asset_size_cny"] = pd.to_numeric(frame["asset_size_cny"], errors="coerce")
    frame["inception_date"] = pd.to_datetime(frame["inception_date"], errors="coerce")
    frame["size_eligible_current"] = frame["size_eligible_current"].fillna(False).astype(bool)
    frame = frame[
        (frame["size_eligible_current"])
        & (frame["tracking_status"] == "verified_page_field")
        & (frame["mapping_status"] == "candidate")
        & frame["inception_date"].notna()
    ].copy()
    frame["backtest_start"] = frame["inception_date"].clip(lower=START_DATE)
    frame = frame[frame["code"].map(lambda code: (DAILY_DIR / f"{code}.csv.gz").exists())]
    return frame.reset_index(drop=True)


def load_history(code: str, start: pd.Timestamp) -> pd.DataFrame:
    path = DAILY_DIR / f"{code}.csv.gz"
    frame = pd.read_csv(path, compression="gzip")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            frame[col] = np.nan
    frame = frame.dropna(subset=["date", "close"])
    frame = frame[frame["date"] >= start].copy()
    frame["code"] = code
    frame = frame.sort_values("date").drop_duplicates(["code", "date"], keep="last")
    frame["ma60"] = frame["close"].rolling(60, min_periods=40).mean()
    frame["mom20"] = frame["close"].pct_change(20)
    frame["mom60"] = frame["close"].pct_change(60)
    frame["trend60"] = frame["close"] / frame["ma60"] - 1
    frame["vol20"] = frame["close"].pct_change().rolling(20, min_periods=15).std() * np.sqrt(252)
    frame["amount60"] = frame["amount"].rolling(60, min_periods=20).mean()
    return frame[["code", "date", "open", "close", "volume", "amount", "ma60", "mom20", "mom60", "trend60", "vol20", "amount60"]]


def cross_sectional_rank(values: pd.Series) -> pd.Series:
    return values.rank(pct=True, method="average")


def build_features(universe: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for row in universe.itertuples(index=False):
        frame = load_history(row.code, max(pd.Timestamp(row.backtest_start), START_DATE))
        if frame.empty:
            continue
        frame["industry"] = row.shenwan_level1
        frames.append(frame)
    if not frames:
        raise RuntimeError("No eligible ETF history available")
    data = pd.concat(frames, ignore_index=True)
    data = data.replace([np.inf, -np.inf], np.nan)
    for field in ["mom20", "mom60", "trend60", "vol20", "amount60"]:
        data[f"rank_{field}"] = data.groupby("date")[field].transform(cross_sectional_rank)
    data["score"] = (
        0.35 * data["rank_mom20"]
        + 0.30 * data["rank_mom60"]
        + 0.20 * data["rank_trend60"]
        - 0.15 * data["rank_vol20"]
    )
    data["trend_ok"] = (data["close"] > data["ma60"]) & (data["mom20"] > 0)
    return data


def load_market_reference() -> pd.DataFrame:
    path = DAILY_DIR / "510300.csv.gz"
    if not path.exists():
        raise RuntimeError("Missing 510300 reference history")
    frame = pd.read_csv(path, compression="gzip")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["open"] = pd.to_numeric(frame["open"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
    frame = frame[frame["date"] >= START_DATE].copy()
    frame["ma200"] = frame["close"].rolling(200, min_periods=150).mean()
    frame["risk_on"] = frame["close"] > frame["ma200"]
    return frame[["date", "open", "close", "risk_on"]]


def build_targets(features: pd.DataFrame, market: pd.DataFrame) -> tuple[dict[pd.Timestamp, dict[str, float]], pd.DataFrame]:
    market_dates = market["date"].tolist()
    date_frame = pd.DataFrame({"date": market_dates})
    date_frame["week"] = date_frame["date"].dt.to_period("W-FRI")
    signal_dates = date_frame.groupby("week", as_index=False)["date"].max()["date"].tolist()
    market_gate = market.set_index("date")["risk_on"].to_dict()
    targets: dict[pd.Timestamp, dict[str, float]] = {}
    records = []
    for date in signal_dates:
        if date not in market_gate or not bool(market_gate[date]):
            targets[date] = {}
            records.append({"signal_date": date, "execution_date": None, "risk_on": False, "selected_count": 0, "selected": ""})
            continue
        day = features[features["date"] == date].dropna(subset=["score", "mom20", "mom60", "trend60", "vol20", "amount60"])
        day = day[day["trend_ok"]].copy()
        if day.empty:
            targets[date] = {}
            records.append({"signal_date": date, "execution_date": None, "risk_on": True, "selected_count": 0, "selected": ""})
            continue
        # One representative ETF per Shenwan level-1 industry; choose the highest score,
        # then use trailing amount as a liquidity tie-breaker.
        day = day.sort_values(["score", "amount60"], ascending=False).drop_duplicates("industry", keep="first")
        selected = day.head(5)
        if len(selected) < 3:
            targets[date] = {}
            records.append({"signal_date": date, "execution_date": None, "risk_on": True, "selected_count": int(len(selected)), "selected": ""})
            continue
        weight = 1.0 / len(selected)
        targets[date] = {row.code: weight for row in selected.itertuples(index=False)}
        records.append({"signal_date": date, "execution_date": None, "risk_on": True, "selected_count": int(len(selected)), "selected": ";".join(f"{row.industry}:{row.code}" for row in selected.itertuples(index=False))})
    # Signals are calculated at a completed close and executed on the next reference trading day.
    date_to_next = {market_dates[i]: market_dates[i + 1] for i in range(len(market_dates) - 1)}
    exec_targets = {}
    for record in records:
        signal_date = record["signal_date"]
        execution_date = date_to_next.get(signal_date)
        record["execution_date"] = execution_date
        if execution_date is not None:
            exec_targets[execution_date] = targets.get(signal_date, {})
    return exec_targets, pd.DataFrame(records)


def simulate(market: pd.DataFrame, price_data: dict[str, pd.DataFrame], targets: dict[pd.Timestamp, dict[str, float]]) -> pd.DataFrame:
    calendar = market["date"].tolist()
    codes = sorted(price_data)
    opens = pd.DataFrame({code: price_data[code].set_index("date")["open"] for code in codes}).reindex(calendar)
    closes = pd.DataFrame({code: price_data[code].set_index("date")["close"] for code in codes}).reindex(calendar)
    previous_closes = closes.shift(1)
    wealth = 1.0
    weights = {"CASH": 1.0}
    rows = []
    for idx, date in enumerate(calendar):
        if idx == 0:
            rows.append({"date": date, "portfolio_value": wealth, "portfolio_return": 0.0, "turnover": 0.0, "fee": 0.0, "holdings": "CASH:1.0000"})
            continue
        prev_close = previous_closes.loc[date]
        open_px = opens.loc[date]
        close_px = closes.loc[date]
        open_px = open_px.where(open_px.notna(), prev_close)
        close_px = close_px.where(close_px.notna(), open_px)
        old_cash = float(weights.get("CASH", 0.0))
        values = {}
        open_value = wealth * old_cash
        for code in codes:
            old_weight = float(weights.get(code, 0.0))
            if old_weight <= 0:
                continue
            prev = prev_close.get(code, np.nan)
            op = open_px.get(code, np.nan)
            if pd.isna(prev) or prev <= 0 or pd.isna(op):
                op = prev if pd.notna(prev) else np.nan
            if pd.isna(op) or pd.isna(prev) or prev <= 0:
                values[code] = 0.0
                continue
            values[code] = wealth * old_weight * float(op / prev)
            # `values` is already the position value at the open, so the
            # portfolio open value is cash plus all position values.
            open_value += values[code]
        if open_value <= 0:
            open_value = wealth
        post_gap = {"CASH": old_cash / open_value}
        for code, value in values.items():
            post_gap[code] = value / open_value
        target = targets.get(date)
        if target is None:
            target = {k: v for k, v in post_gap.items() if k == "CASH" or v > 0}
        target = {k: float(v) for k, v in target.items() if float(v) > 0}
        target.setdefault("CASH", max(0.0, 1.0 - sum(v for k, v in target.items() if k != "CASH")))
        target["CASH"] = max(0.0, 1.0 - sum(v for k, v in target.items() if k != "CASH"))
        all_codes = set(post_gap) | set(target)
        turnover = sum(abs(target.get(code, 0.0) - post_gap.get(code, 0.0)) for code in all_codes if code != "CASH")
        fee = open_value * FEE_RATE * turnover
        after_fee = max(0.0, open_value - fee)
        close_value = after_fee * target.get("CASH", 0.0)
        close_weights_value = {"CASH": after_fee * target.get("CASH", 0.0)}
        for code in codes:
            tw = target.get(code, 0.0)
            if tw <= 0:
                continue
            op = open_px.get(code, np.nan)
            cl = close_px.get(code, np.nan)
            if pd.isna(op) or op <= 0 or pd.isna(cl):
                cl = op
            contribution = after_fee * tw * float(cl / op) if pd.notna(op) and op > 0 and pd.notna(cl) else 0.0
            close_value += contribution
            close_weights_value[code] = contribution
        if close_value <= 0:
            close_value = wealth
        weights = {k: v / close_value for k, v in close_weights_value.items() if v > 0}
        holdings = ";".join(f"{k}:{v:.4f}" for k, v in sorted(weights.items()))
        rows.append({"date": date, "portfolio_value": close_value, "portfolio_return": close_value / wealth - 1, "turnover": turnover, "fee": fee, "holdings": holdings})
        wealth = close_value
    return pd.DataFrame(rows)


def metrics(daily: pd.DataFrame) -> dict:
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.dropna(subset=["portfolio_value"])
    if daily.empty:
        return {}
    start_value = float(daily.iloc[0]["portfolio_value"])
    end_value = float(daily.iloc[-1]["portfolio_value"])
    years = max((daily.iloc[-1]["date"] - daily.iloc[0]["date"]).days / 365.25, 1 / 365.25)
    ret = daily["portfolio_return"].fillna(0.0)
    curve = daily["portfolio_value"]
    drawdown = curve / curve.cummax() - 1
    return {
        "start_date": daily.iloc[0]["date"].date().isoformat(),
        "end_date": daily.iloc[-1]["date"].date().isoformat(),
        "start_value": start_value,
        "end_value": end_value,
        "total_return": end_value / start_value - 1,
        "cagr": (end_value / start_value) ** (1 / years) - 1,
        "annualized_volatility": float(ret.std(ddof=1) * np.sqrt(252)),
        "sharpe_rf0": float(ret.mean() / ret.std(ddof=1) * np.sqrt(252)) if ret.std(ddof=1) > 0 else None,
        "max_drawdown": float(drawdown.min()),
        "average_daily_turnover": float(daily["turnover"].mean()),
        "total_turnover": float(daily["turnover"].sum()),
        "total_fees": float(daily["fee"].sum()),
        "observation_days": int(len(daily)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-etfs", type=int, default=0, help="0 means all eligible ETFs")
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    universe = load_universe()
    if args.max_etfs:
        universe = universe.head(args.max_etfs).copy()
    features = build_features(universe)
    market = load_market_reference()
    targets, signal_log = build_targets(features, market)
    used_codes = sorted({code for target in targets.values() for code in target})
    price_data = {code: load_history(code, START_DATE) for code in used_codes}
    daily = simulate(market, price_data, targets)
    result_metrics = metrics(daily)
    # Market reference is a transparent benchmark; this is close-to-close and excludes fees.
    market_ret = market["close"].pct_change().fillna(0.0)
    market_curve = (1 + market_ret).cumprod()
    market_drawdown = market_curve / market_curve.cummax() - 1
    market_years = max((market["date"].iloc[-1] - market["date"].iloc[0]).days / 365.25, 1 / 365.25)
    result_metrics["benchmark_510300_total_return"] = float(market_curve.iloc[-1] - 1)
    result_metrics["benchmark_510300_cagr"] = float(market_curve.iloc[-1] ** (1 / market_years) - 1)
    result_metrics["benchmark_510300_max_drawdown"] = float(market_drawdown.min())
    result_metrics["eligible_etf_count"] = int(len(universe))
    result_metrics["eligible_industry_count"] = int(universe["shenwan_level1"].nunique())
    result_metrics["used_etf_count"] = int(len(used_codes))
    result_metrics["rebalance_signal_count"] = int((signal_log["selected_count"] >= 3).sum())
    daily.to_csv(ROOT / "data" / "normalized" / "baseline_backtest_daily.csv", index=False, encoding="utf-8-sig")
    signal_log.to_csv(ROOT / "data" / "normalized" / "baseline_backtest_signals.csv", index=False, encoding="utf-8-sig")
    (ROOT / "data" / "normalized" / "baseline_backtest_metrics.json").write_text(json.dumps(result_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    report = "# 基础 ETF 轮动回测\n\n"
    report += "本回测为当前规模池的回溯性基线，不代表无生存偏差的最终样本外结果。\n\n"
    report += "## 规则\n\n"
    report += "- ETF池：当前公开资产规模 >= 1亿元、跟踪标的已核对、申万一级行业候选映射、日线可用。\n"
    report += "- 起始日：2015-01-01；每只 ETF 从成立日期和实际首个日线数据中的较晚日期开始。\n"
    report += "- 信号：每周最后一个交易日收盘计算，20/60日动量、60日趋势、20日波动率横截面排名。\n"
    report += "- 组合：市场参考 510300 位于 200日均线之上时，按行业选分数最高的最多5个行业 ETF 等权；不足3个或市场风险关闭时持现金。\n"
    report += "- 执行：下一交易日开盘；交易费率万三；基础回测滑点为零。\n\n"
    report += "## 结果\n\n"
    for key, value in result_metrics.items():
        report += f"- {key}: {value}\n"
    report += "\n## 文件\n\n- `baseline_backtest_daily.csv`\n- `baseline_backtest_signals.csv`\n- `baseline_backtest_metrics.json`\n"
    (REPORTS / "baseline_backtest_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(result_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
