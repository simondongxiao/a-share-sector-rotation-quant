# ETF universe and Shenwan mapping audit

Generated: 2026-09-23T22:09:32+08:00

## Universe

- Inventory source: Tonghuashun public ETF inventory endpoint.
- Captured listed ETF records: **1,725**.
- History files downloaded/cached: **1,673**.
- Empty history responses: **52**.
- Download errors: **0**.
- Daily observations saved: **1,445,141**.
- Observed history window: **2015-01-05 to 2026-09-22**.
- Current snapshot size available: **1,687**.
- Current snapshot passing RMB 100 million filter: **1,315**.
- Tracking index page field verified: **1,718**.
- Tracking index request errors: **7**.

The size filter is not yet historically point-in-time for every ETF. It is stored as a current/nearest-public snapshot and must not be used as a historical eligibility filter until a historical share/size series is added.

## Mapping draft

| Status | Count |
|---|---:|
| non_industry_or_broad | 1,264 |
| candidate | 297 |
| unmapped | 111 |
| manual_review | 53 |


The mapping is a keyword draft, not a final classification. Candidate mappings must be checked against each ETF's tracked index or fund contract. `manual_review` and `unmapped` rows are intentionally retained.

## Files

- `data/normalized/etf_universe.csv`
- `data/normalized/etf_universe_with_tracking_review.csv`
- `data/normalized/etf_tracking_index_review.csv`
- `data/normalized/etf_history_download_log.csv`
- `data/normalized/etf_shenwan_level1_mapping_draft.csv`
- `data/normalized/etf_shenwan_level1_mapping_manual_review.csv`
- `data/raw/etf_daily/<code>.csv.gz`
