# 基础 ETF 轮动回测

本回测为当前规模池的回溯性基线，不代表无生存偏差的最终样本外结果。

## 规则

- ETF池：当前公开资产规模 >= 1亿元、跟踪标的已核对、申万一级行业候选映射、日线可用。
- 起始日：2015-01-01；每只 ETF 从成立日期和实际首个日线数据中的较晚日期开始。
- 信号：每周最后一个交易日收盘计算，20/60日动量、60日趋势、20日波动率横截面排名。
- 组合：市场参考 510300 位于 200日均线之上时，按行业选分数最高的最多5个行业 ETF 等权；不足3个或市场风险关闭时持现金。
- 执行：下一交易日开盘；交易费率万三；基础回测滑点为零。

## 结果

- start_date: 2015-01-05
- end_date: 2026-09-22
- start_value: 1.0
- end_value: 0.7065116619593212
- total_return: -0.29348833804067875
- cagr: -0.029226287201026824
- annualized_volatility: 0.18872839343399686
- sharpe_rf0: -0.06315444979223621
- max_drawdown: -0.5808777130968089
- average_daily_turnover: 0.06570648528359346
- total_turnover: 187.26348305824138
- total_fees: 0.06156877078628826
- observation_days: 2850
- benchmark_510300_total_return: 0.25081344902385005
- benchmark_510300_cagr: 0.019290956735235065
- benchmark_510300_max_drawdown: -0.46306068601583183
- eligible_etf_count: 237
- eligible_industry_count: 19
- used_etf_count: 130
- rebalance_signal_count: 177

## 文件

- `baseline_backtest_daily.csv`
- `baseline_backtest_signals.csv`
- `baseline_backtest_metrics.json`
