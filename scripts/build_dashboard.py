from __future__ import annotations

import json
import math
import gzip
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NORMALIZED = ROOT / "data" / "normalized"
DAILY_DIR = ROOT / "data" / "raw" / "etf_daily"
DOCS = ROOT / "docs"
DOCS.mkdir(parents=True, exist_ok=True)


def read_etf_latest(code: str) -> dict:
    path = DAILY_DIR / f"{code}.csv.gz"
    if not path.exists():
        return {}
    frame = pd.read_csv(path, compression="gzip", parse_dates=["date"])
    for col in ["open", "high", "low", "close", "amount"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
    if frame.empty:
        return {}
    close = frame["close"]
    latest = frame.iloc[-1]
    result = {
        "date": latest["date"].date().isoformat(),
        "close": float(latest["close"]),
        "return_1d": float(close.pct_change().iloc[-1]) if len(close) > 1 else None,
        "mom20": float(close.pct_change(20).iloc[-1]) if len(close) > 20 else None,
        "mom60": float(close.pct_change(60).iloc[-1]) if len(close) > 60 else None,
        "ma5": float(close.rolling(5).mean().iloc[-1]),
        "ma10": float(close.rolling(10).mean().iloc[-1]),
        "ma20": float(close.rolling(20).mean().iloc[-1]),
        "trend20": float(close.iloc[-1] / close.rolling(20).mean().iloc[-1] - 1) if len(close) >= 20 else None,
        "vol20": float(close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)) if len(close) >= 21 else None,
        "amount60": float(pd.to_numeric(frame["amount"], errors="coerce").rolling(60, min_periods=20).mean().iloc[-1]),
    }
    return result


def pct(value, digits=1):
    return "—" if value is None or not np.isfinite(value) else f"{value * 100:.{digits}f}%"


def number(value, digits=2):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def build_payload() -> dict:
    universe = pd.read_csv(NORMALIZED / "etf_shenwan_level1_mapping_draft.csv", dtype={"code": str})
    universe["code"] = universe["code"].str.zfill(6)
    universe["asset_size_cny"] = pd.to_numeric(universe["asset_size_cny"], errors="coerce")
    universe = universe[(universe["mapping_status"] == "candidate") & (universe["tracking_status"] == "verified_page_field") & (universe["size_eligible_current"] == True)].copy()  # noqa: E712
    rows = []
    for row in universe.itertuples(index=False):
        latest = read_etf_latest(row.code)
        if not latest:
            continue
        rows.append({
            "code": row.code,
            "name": str(row.fund_name),
            "industry": str(row.shenwan_level1),
            "tracking_index": str(row.tracking_index),
            "asset_size_cny": float(row.asset_size_cny),
            **latest,
        })
    etfs = pd.DataFrame(rows)
    for col in ["mom20", "mom60", "trend20", "vol20", "amount60"]:
        etfs[f"rank_{col}"] = etfs[col].rank(pct=True)
    etfs["score"] = 0.35 * etfs["rank_mom20"] + 0.35 * etfs["rank_mom60"] + 0.20 * etfs["rank_trend20"] - 0.10 * etfs["rank_vol20"]
    etfs["trend_ok"] = (etfs["trend20"] > 0) & (etfs["mom20"] > 0)
    etfs.to_csv(NORMALIZED / "dashboard_etf_features.csv", index=False, encoding="utf-8-sig")
    reps = etfs.sort_values(["score", "asset_size_cny"], ascending=False).drop_duplicates("industry")
    reps = reps.sort_values("score", ascending=False).head(20).copy()

    market = pd.read_csv(DAILY_DIR / "510300.csv.gz", compression="gzip", parse_dates=["date"])
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    market = market.dropna(subset=["date", "close"]).sort_values("date")
    market["ma200"] = market["close"].rolling(200, min_periods=150).mean()
    market_last = market.iloc[-1]
    risk_on = bool(market_last["close"] > market_last["ma200"]) if pd.notna(market_last["ma200"]) else False
    eligible_reps = reps[reps["trend_ok"]].head(5)
    if risk_on and len(eligible_reps) >= 3:
        eligible_reps = eligible_reps.copy()
        eligible_reps["suggested_weight"] = 1.0 / len(eligible_reps)
        cash_weight = 0.0
    else:
        eligible_reps = eligible_reps.copy()
        eligible_reps["suggested_weight"] = 0.0
        cash_weight = 1.0
    reps["suggested_weight"] = reps["code"].map(eligible_reps.set_index("code")["suggested_weight"]).fillna(0.0)

    micro = {}
    recent_path = NORMALIZED / "shenwan_level1_microstructure_recent_official.csv"
    if recent_path.exists():
        recent = pd.read_csv(recent_path, parse_dates=["date"])
        if not recent.empty:
            latest_date = recent["date"].max()
            for row in recent[recent["date"] == latest_date].itertuples(index=False):
                micro[str(row.industry)] = {
                    "limit_down_count": int(row.official_limit_down_count),
                    "sealed_amount_cny": float(row.sealed_amount_cny),
                    "date": latest_date.date().isoformat(),
                    "method": "official_recent_limit_down_pool",
                }
    for row in reps.itertuples(index=False):
        if row.industry not in micro:
            micro[row.industry] = {"limit_down_count": None, "sealed_amount_cny": None, "date": None, "method": "unavailable"}

    metrics = json.loads((NORMALIZED / "baseline_backtest_metrics.json").read_text(encoding="utf-8"))
    daily = pd.read_csv(NORMALIZED / "baseline_backtest_daily.csv", parse_dates=["date"])
    curve = daily[["date", "portfolio_value"]].dropna().tail(400).to_dict("records")
    for point in curve:
        point["date"] = pd.Timestamp(point["date"]).date().isoformat()
        point["portfolio_value"] = float(point["portfolio_value"])

    data_dates = {
        "etf_daily": str(etfs["date"].max()) if not etfs.empty else None,
        "market_daily": market_last["date"].date().isoformat(),
        "limit_down_official": max((v["date"] for v in micro.values() if v["date"]), default=None),
        "etf_30m": None,
    }
    minute_path = NORMALIZED / "etf_30m_recent.csv"
    if minute_path.exists():
        minute = pd.read_csv(minute_path, usecols=["datetime"])
        if not minute.empty:
            data_dates["etf_30m"] = pd.to_datetime(minute["datetime"], errors="coerce").max().isoformat()

    ranking = []
    for row in reps.itertuples(index=False):
        ranking.append({
            "industry": row.industry,
            "code": row.code,
            "name": row.name,
            "tracking_index": row.tracking_index,
            "asset_size_cny": row.asset_size_cny,
            "score": float(row.score),
            "return_1d": row.return_1d,
            "mom20": row.mom20,
            "mom60": row.mom60,
            "trend20": row.trend20,
            "vol20": row.vol20,
            "suggested_weight": float(row.suggested_weight),
            "microstructure": micro.get(row.industry),
        })
    return {
        "generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "as_of_date": str(etfs["date"].max()) if not etfs.empty else None,
        "execution_date": (pd.Timestamp(etfs["date"].max()) + pd.offsets.BDay(1)).date().isoformat() if not etfs.empty else None,
        "universe_count": int(len(universe)),
        "history_available_count": int(len(etfs)),
        "industry_count": int(etfs["industry"].nunique()),
        "risk_on": risk_on,
        "cash_weight": cash_weight,
        "benchmark": {"close": float(market_last["close"]), "ma200": float(market_last["ma200"]) if pd.notna(market_last["ma200"]) else None},
        "metrics": metrics,
        "ranking": ranking,
        "curve": curve,
        "data_dates": data_dates,
        "engine": {
            "status": "候选规则已实现；需继续样本外验证",
            "daily_execution": "T日收盘计算，T+1开盘执行",
            "intraday_confirmation": "已接入代表性ETF近端30分钟数据；历史30分钟回测未宣称完成",
            "panic_branch": "只有跌停家数真实可用时才允许高置信度恐慌清仓",
        },
    }


def svg_curve(points: list[dict]) -> str:
    if len(points) < 2:
        return "<svg viewBox='0 0 800 220'><text x='20' y='40'>暂无曲线数据</text></svg>"
    values = [p["portfolio_value"] for p in points]
    lo, hi = min(values), max(values)
    span = hi - lo or 1
    coords = []
    for i, value in enumerate(values):
        x = 8 + 784 * i / (len(values) - 1)
        y = 205 - 175 * (value - lo) / span
        coords.append(f"{x:.1f},{y:.1f}")
    return f"<svg viewBox='0 0 800 220' role='img' aria-label='组合净值曲线'><polyline fill='none' stroke='#2f7d68' stroke-width='3' points='{ ' '.join(coords) }'/><text x='8' y='218' fill='#6b7280'>{points[0]['date']}</text><text x='680' y='218' fill='#6b7280'>{points[-1]['date']}</text></svg>"


def render(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    rows = []
    for item in payload["ranking"]:
        micro = item["microstructure"] or {}
        rows.append(
            f"<tr><td><b>{item['industry']}</b></td><td>{item['code']}<br><span class='muted'>{item['name']}</span></td>"
            f"<td>{item['score']:.3f}</td><td>{pct(item['mom20'])}</td><td>{pct(item['mom60'])}</td>"
            f"<td>{pct(item['suggested_weight'])}</td><td>{micro.get('limit_down_count','—')}</td>"
            f"<td>{number(micro.get('sealed_amount_cny'),0) if micro.get('sealed_amount_cny') is not None else '—'}</td></tr>"
        )
    metrics = payload["metrics"]
    return f"""<!doctype html>
<html lang='zh-CN'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>A股板块轮动量化看板</title>
<style>
:root{{--ink:#18252b;--muted:#66757c;--line:#e4ece8;--bg:#f5f8f6;--card:#fff;--green:#2f7d68;--amber:#b7791f;--red:#b84a47}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI','Microsoft YaHei',sans-serif}}main{{max-width:1240px;margin:0 auto;padding:28px 18px 60px}}h1{{margin:0 0 6px;font-size:28px;letter-spacing:-.02em}}h2{{margin:0 0 14px;font-size:18px}}p{{margin:6px 0;color:var(--muted)}}.sub{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:22px 0}}.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:0 4px 16px #163c2d08}}.label{{color:var(--muted);font-size:12px}}.value{{font-size:24px;font-weight:700;margin-top:3px}}.good{{color:var(--green)}}.warn{{color:var(--amber)}}.bad{{color:var(--red)}}.panel{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin-top:14px;overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:820px}}th,td{{text-align:left;border-bottom:1px solid var(--line);padding:10px 8px;vertical-align:top}}th{{font-size:12px;color:var(--muted);font-weight:600}}.muted{{color:var(--muted);font-size:12px}}.tag{{display:inline-block;padding:3px 8px;border-radius:999px;background:#e8f3ee;color:var(--green);font-size:12px}}.tag.warn{{background:#fff4dc;color:#94620e}}svg{{width:100%;height:240px;background:linear-gradient(#fff,#fbfdfc);border-radius:10px}}.two{{display:grid;grid-template-columns:1.4fr 1fr;gap:14px}}ul{{margin:8px 0 0 18px;padding:0;color:var(--muted)}}footer{{margin-top:18px;color:var(--muted);font-size:12px}}@media(max-width:800px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.two{{grid-template-columns:1fr}}h1{{font-size:23px}}}}
</style></head><body><main>
<header><h1>A股板块轮动量化看板</h1><p>日线收盘决策 · 申万一级行业 · ETF仓位建议 · 动态减仓/回补引擎</p><p>数据截至 <b>{payload['as_of_date']}</b>，建议执行日 <b>{payload['execution_date']}</b> · 生成于 {payload['generated_at']}</p></header>
<section class='grid'>
<div class='card'><div class='label'>风险状态</div><div class='value {'good' if payload['risk_on'] else 'warn'}'>{'RISK ON' if payload['risk_on'] else 'RISK OFF'}</div><p>510300 收盘 {number(payload['benchmark']['close'])} / MA200 {number(payload['benchmark']['ma200'])}</p></div>
<div class='card'><div class='label'>建议现金</div><div class='value'>{pct(payload['cash_weight'])}</div><p>不足3个趋势合格行业时自动提高现金</p></div>
<div class='card'><div class='label'>当前候选ETF</div><div class='value'>{payload['history_available_count']}</div><p>规模≥1亿元、跟踪信息已核验</p></div>
<div class='card'><div class='label'>基础回测 CAGR / 最大回撤</div><div class='value bad'>{pct(metrics.get('cagr'))} / {pct(metrics.get('max_drawdown'))}</div><p>基线尚未达到目标，动态引擎不能预先承诺改善</p></div>
</section>
<section class='panel'><h2>板块热度与仓位建议</h2><p>评分由20/60日动量、趋势和波动率组成；跌停数/封单额仅在近期官方跌停池窗口显示，历史缺失时显示“—”。</p><table><thead><tr><th>板块</th><th>代表ETF</th><th>热度分</th><th>20日动量</th><th>60日动量</th><th>建议仓位</th><th>跌停数</th><th>封单额（元）</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section class='two'><div class='panel'><h2>基线净值曲线</h2>{svg_curve(payload['curve'])}<p>当前曲线为基础ETF轮动策略，不等同于动态退出引擎的样本外结果。</p></div>
<div class='panel'><h2>动态引擎状态</h2><p><span class='tag warn'>{payload['engine']['status']}</span></p><ul><li>{payload['engine']['daily_execution']}</li><li>{payload['engine']['intraday_confirmation']}</li><li>{payload['engine']['panic_branch']}</li></ul><h2 style='margin-top:18px'>数据新鲜度</h2><ul>{''.join(f"<li>{k}: {v or '不可用'}</li>" for k,v in payload['data_dates'].items())}</ul></div></section>
<section class='panel'><h2>回测摘要</h2><table><tbody><tr><td>总收益</td><td>{pct(metrics.get('total_return'))}</td><td>年化波动</td><td>{pct(metrics.get('annualized_volatility'))}</td></tr><tr><td>Sharpe（无风险利率0）</td><td>{number(metrics.get('sharpe_rf0'),3)}</td><td>总换手</td><td>{number(metrics.get('total_turnover'))}</td></tr><tr><td>基准510300 CAGR</td><td>{pct(metrics.get('benchmark_510300_cagr'))}</td><td>基准最大回撤</td><td>{pct(metrics.get('benchmark_510300_max_drawdown'))}</td></tr></tbody></table></section>
<footer>本页面仅提供量化决策建议，不连接券商、不自动下单。所有建议须结合数据可用性、流动性、涨跌停和执行价差复核。</footer>
<script>window.dashboardData={data};</script></main></body></html>"""


def main():
    payload = build_payload()
    (DOCS / "index.html").write_text(render(payload), encoding="utf-8")
    (DOCS / "dashboard.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"as_of_date": payload["as_of_date"], "ranking_count": len(payload["ranking"]), "output": str(DOCS / "index.html")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
