# Dynamic exit and rebound engine

The attached code and images are treated as candidate rules. The engine has four stages:

1. Detect an overheated breakdown using state, daily drop, and MA5.
2. Diagnose panic versus disagreement using limit-down count and candle shadow ratio.
3. If the prior session reduced the position, give rebound confirmation priority; a daily-close confirmation is executed on the next session.
4. If the remaining position closes below MA10, clear the remainder.

## Execution timing

Daily data cannot prove a same-day 30-minute repair. In daily-close mode:

- T close computes the exit/re-entry decision;
- T+1 open executes it;
- a rebound seen at T+1 close can only be re-entered at T+2 open.

An optional intraday confirmation table can enable same-day 30-minute recovery, but it must contain a separately timestamped `recovery_30m` or equivalent field. Missing intraday data never counts as confirmation.

## Missing-data rule

If limit-down count is unavailable, the engine may reduce to the ordinary 20% fallback but must not claim an A-kill panic clear. This prevents the system from turning an unavailable microstructure field into a false high-confidence liquidation signal.

## Required future data

To make the panic branch and same-day rebound branch research-grade, add:

- daily sector constituent limit-down count and sealed-limit amount;
- sector-level intraday 30-minute OHLCV or breadth/repair fields;
- market-wide turnover and cross-sector relative-flow series;
- timestamped data availability metadata.
