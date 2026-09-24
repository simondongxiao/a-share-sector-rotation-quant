"""Daily-close-safe dynamic reduction and rebound confirmation engine.

The engine accepts both the user's original CamelCase column names and the
project's snake_case names. A decision made from a daily close is executable
on the next trading session. Same-day 30-minute confirmation is optional and
must be supplied explicitly; it is never inferred from daily OHLC data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DynamicExitConfig:
    large_drop: float = -0.035
    rebound_gain: float = 0.02
    panic_limit_down_count: int = 2
    panic_shadow_ratio: float = 0.20
    rebound_shadow_ratio: float = 0.30
    ordinary_position: float = 0.20
    disagreement_position: float = 0.50
    rebound_position: float = 1.00
    clear_position: float = 0.00
    use_intraday_confirmation: bool = False


@dataclass(frozen=True)
class ExitDecision:
    new_position: float
    action: str
    reason: str
    execution_date: Any
    data_quality: str
    log: str


ALIASES = {
    "position": ("Position", "position"),
    "state": ("State", "state"),
    "limit_down_count": ("Limit_Down_Count", "limit_down_count"),
    "close": ("Close", "close"),
    "low": ("Low", "low"),
    "high": ("High", "high"),
    "daily_drop": ("Daily_Drop", "daily_drop", "return_1d"),
    "ma5": ("MA5", "ma5"),
    "ma10": ("MA10", "ma10"),
    "market_turnover": ("Volume", "volume", "Amount", "amount"),
}


def _value(row: pd.Series, field: str, default: Any = np.nan) -> Any:
    for name in ALIASES[field]:
        if name in row.index:
            return row[name]
    return default


def _number(value: Any) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _execution_date(index: pd.Index, current_date: Any) -> Any:
    dates = pd.Index(index)
    try:
        position = dates.get_loc(current_date)
    except KeyError:
        return None
    if isinstance(position, slice) or isinstance(position, np.ndarray):
        return None
    return dates[position + 1] if position + 1 < len(dates) else None


def _intraday_flag(intraday_confirmation: pd.DataFrame | pd.Series | None, current_date: Any, *names: str) -> bool:
    if intraday_confirmation is None:
        return False
    if isinstance(intraday_confirmation, pd.DataFrame):
        if current_date not in intraday_confirmation.index:
            return False
        row = intraday_confirmation.loc[current_date]
    else:
        row = intraday_confirmation
    for name in names:
        if name in row.index and bool(row[name]):
            return True
    return False


def evaluate_dynamic_exit(
    sector_data: pd.DataFrame,
    index_data: pd.DataFrame,
    current_date: Any,
    intraday_confirmation: pd.DataFrame | pd.Series | None = None,
    config: DynamicExitConfig | None = None,
) -> ExitDecision:
    """Evaluate one completed daily close and return a next-session decision."""
    config = config or DynamicExitConfig()
    if current_date not in sector_data.index:
        return ExitDecision(np.nan, "DATA_ERROR", "current_date_not_found", None, "invalid", "当前日期不存在")
    row = sector_data.loc[current_date]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[-1]
    position = np.clip(_number(_value(row, "position", 1.0)), 0.0, 1.0)
    state = str(_value(row, "state", "NORMAL")).upper()
    close = _number(_value(row, "close"))
    low = _number(_value(row, "low"))
    high = _number(_value(row, "high"))
    daily_drop = _number(_value(row, "daily_drop"))
    ma5 = _number(_value(row, "ma5"))
    ma10 = _number(_value(row, "ma10"))
    limit_down_count = _number(_value(row, "limit_down_count"))
    market_turnover = _number(_value(row, "market_turnover"))
    prev_position = np.nan
    prev_row = None
    try:
        loc = sector_data.index.get_loc(current_date)
        if not isinstance(loc, (slice, np.ndarray)) and loc > 0:
            prev_row = sector_data.iloc[loc - 1]
            prev_position = np.clip(_number(_value(prev_row, "position", position)), 0.0, 1.0)
    except (KeyError, TypeError):
        pass
    execution_date = _execution_date(sector_data.index, current_date)
    if pd.notna(high) and pd.notna(low) and high > low and pd.notna(close):
        shadow_ratio = float((close - low) / (high - low))
    else:
        shadow_ratio = np.nan
    data_quality = "complete" if all(pd.notna(x) for x in [close, daily_drop, ma5, ma10, shadow_ratio]) else "partial"
    if pd.isna(limit_down_count):
        data_quality = "missing_limit_down_count"

    # A prior reduction gets confirmation priority. Otherwise a still-overheated
    # row could repeatedly reduce the position and never reach the rebound branch.
    if pd.notna(prev_position) and 0.0 < prev_position < 1.0:
        intraday_rebound = config.use_intraday_confirmation and _intraday_flag(
            intraday_confirmation,
            current_date,
            "recovery_30m",
            "open_above_ma5",
            "rebound_confirmed",
        )
        if intraday_rebound or (pd.notna(close) and pd.notna(ma5) and pd.notna(daily_drop) and close > ma5 and daily_drop > config.rebound_gain):
            reason = "intraday_30m_rebound" if intraday_rebound else "daily_close_recovered_ma5"
            log = f"[{current_date}] 🔄 确认暴力修复，回补至100%，执行日={execution_date}，原因={reason}"
            return ExitDecision(config.rebound_position, "REBOUND_REENTRY", reason, execution_date, data_quality, log)
        if pd.notna(close) and pd.notna(ma10) and close < ma10:
            log = f"[{current_date}] ❌ 收盘跌破10日线，确认A杀，清空剩余仓位，执行日={execution_date}"
            return ExitDecision(config.clear_position, "CONFIRM_A_KILL", "close_below_ma10_after_reduction", execution_date, data_quality, log)

    trigger = state == "OVERHEATED" and ((pd.notna(daily_drop) and daily_drop <= config.large_drop) or (pd.notna(close) and pd.notna(ma5) and close < ma5))
    if trigger:
        # Without a trustworthy limit-down count, do not claim an A-kill. The
        # fallback can reduce risk but leaves a residual position for review.
        if pd.notna(limit_down_count) and limit_down_count >= config.panic_limit_down_count and pd.notna(shadow_ratio) and shadow_ratio < config.panic_shadow_ratio:
            log = f"[{current_date}] 🚨 跌停数={int(limit_down_count)}且下影线弱，触发恐慌退潮，清仓至0%，执行日={execution_date}"
            return ExitDecision(config.clear_position, "PANIC_CLEAR", "panic_limit_down_and_short_shadow", execution_date, data_quality, log)
        if (pd.notna(shadow_ratio) and shadow_ratio >= config.rebound_shadow_ratio) or (pd.notna(limit_down_count) and limit_down_count == 0):
            log = f"[{current_date}] ⚠️ 高位回撤但有承接，减仓至50%，执行日={execution_date}"
            return ExitDecision(config.disagreement_position, "PARTIAL_REDUCTION", "rebound_shadow_or_no_limit_down", execution_date, data_quality, log)
        new_position = config.ordinary_position if pd.notna(limit_down_count) else config.ordinary_position
        log = f"[{current_date}] ⚠️ 触发常规风控，减仓至{new_position:.0%}，执行日={execution_date}，数据质量={data_quality}"
        return ExitDecision(new_position, "ORDINARY_REDUCTION", "overheated_breakdown", execution_date, data_quality, log)

    return ExitDecision(position, "HOLD", "no_trigger", execution_date, data_quality, f"[{current_date}] 维持当前仓位{position:.0%}，执行日={execution_date}")


def dynamic_exit_and_rebound_engine(
    sector_data: pd.DataFrame,
    index_data: pd.DataFrame,
    current_date: Any,
    intraday_confirmation: pd.DataFrame | pd.Series | None = None,
    config: DynamicExitConfig | None = None,
) -> tuple[float, str]:
    """Compatibility wrapper matching the user's `(new_pos, log)` interface."""
    decision = evaluate_dynamic_exit(sector_data, index_data, current_date, intraday_confirmation, config)
    return decision.new_position, decision.log
