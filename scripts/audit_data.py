from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NORMALIZED = ROOT / "data" / "normalized"
REPORTS = ROOT / "reports"


def main() -> None:
    universe_path = NORMALIZED / "etf_universe_with_tracking_review.csv"
    if not universe_path.exists():
        universe_path = NORMALIZED / "etf_universe.csv"
    universe = pd.read_csv(universe_path, dtype={"code": str})
    history = pd.read_csv(NORMALIZED / "etf_history_download_log.csv", dtype={"code": str})
    mapping = pd.read_csv(NORMALIZED / "etf_shenwan_level1_mapping_draft.csv", dtype={"code": str})
    history_rows = pd.to_numeric(history["rows"], errors="coerce").fillna(0).sum()
    downloaded = int(history["status"].isin(["downloaded", "cached"]).sum())
    empty = int((history["status"] == "empty").sum())
    errors = int((history["status"] == "error").sum())
    if "asset_size_cny" in universe.columns:
        size_snapshot = int(universe["asset_size_cny"].notna().sum())
        size_eligible = int((pd.to_numeric(universe["asset_size_cny"], errors="coerce") >= 100_000_000).sum())
    else:
        size_snapshot = int((universe["size_status"] == "point_in_time_snapshot_available").sum())
        size_eligible = int(universe["size_eligibility"].fillna(False).sum())
    tracking_counts = universe["tracking_status"].value_counts(dropna=False).to_dict() if "tracking_status" in universe.columns else {}
    report = f"""# ETF universe and Shenwan mapping audit

Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}

## Universe

- Inventory source: Tonghuashun public ETF inventory endpoint.
- Captured listed ETF records: **{len(universe):,}**.
- History files downloaded/cached: **{downloaded:,}**.
- Empty history responses: **{empty:,}**.
- Download errors: **{errors:,}**.
- Daily observations saved: **{int(history_rows):,}**.
- Observed history window: **{history['history_start'].min()} to {history['history_end'].max()}**.
- Current snapshot size available: **{size_snapshot:,}**.
- Current snapshot passing RMB 100 million filter: **{size_eligible:,}**.
- Tracking index page field verified: **{int(tracking_counts.get('verified_page_field', 0)):,}**.
- Tracking index request errors: **{int(tracking_counts.get('request_error', 0)):,}**.

The size filter is not yet historically point-in-time for every ETF. It is stored as a current/nearest-public snapshot and must not be used as a historical eligibility filter until a historical share/size series is added.

## Mapping draft

| Status | Count |
|---|---:|
"""
    status_counts = mapping["mapping_status"].value_counts(dropna=False)
    for status, count in status_counts.items():
        report += f"| {status} | {int(count):,} |\n"
    report += """
\nThe mapping is a keyword draft, not a final classification. Candidate mappings must be checked against each ETF's tracked index or fund contract. `manual_review` and `unmapped` rows are intentionally retained.

## Files

- `data/normalized/etf_universe.csv`
- `data/normalized/etf_universe_with_tracking_review.csv`
- `data/normalized/etf_tracking_index_review.csv`
- `data/normalized/etf_history_download_log.csv`
- `data/normalized/etf_shenwan_level1_mapping_draft.csv`
- `data/normalized/etf_shenwan_level1_mapping_manual_review.csv`
- `data/raw/etf_daily/<code>.csv.gz`
"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / "etf_universe_mapping_audit.md"
    path.write_text(report, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
