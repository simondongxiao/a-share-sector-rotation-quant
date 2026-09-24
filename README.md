# A股板块轮动量化系统

这是一个只做多、允许持有现金的 A 股 ETF 板块轮动研究项目。系统按已完成日线收盘计算信号，并在下一交易日执行，默认交易费率为万三、滑点首版设为零。

## 当前组成

- 全市场场内 ETF 清单、规模筛选、跟踪指数复核和申万一级行业映射；
- 2015 年以来的 ETF 基础轮动回测；
- 动态减仓 / 恐慌退潮 / V 反回补引擎；
- 近期真实跌停池和封单金额；
- 代表性 ETF 近端 30 分钟修复数据；
- 静态 HTML 看板，入口为 `docs/index.html`。

## 数据质量边界

历史板块跌停广度使用成分股日线跌幅估算，并明确标记为 `price_return_estimate`；近期跌停池使用公开接口的真实池数据和封单金额。免费公开日线接口无法还原 2015 年以来逐日历史封单额，历史封单额保持为空，不以成交额伪造。

申万成分股当前快照用于历史估算，尚未替代为完整的点时点成分股历史。该限制会在回测报告和看板中显示。

## 本地运行

```powershell
python -m pip install -r requirements.txt
python scripts/fetch_market_microstructure.py --skip-history
python scripts/build_dashboard.py
```

完整历史成分股回溯为长任务，可手动运行：

```powershell
python scripts/fetch_market_microstructure.py --workers 8 --start-date 20150101 --end-date 20260924 --skip-30m
```

## 免责声明

本项目只提供研究和决策建议，不连接券商、不自动下单，也不保证收益率、胜率或最大回撤目标。
