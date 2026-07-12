"""Export notebook result tables to dated CSV and Excel artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from make_worldcup26_interactive_html import load_notebook_tables


ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "outputs" / "tables"
MARKET_DIR = ROOT / "data" / "markets"
SOFASCORE_DIR = ROOT / "data" / "sofascore"


def clean_table(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    return clean.loc[:, ~clean.columns.astype(str).str.startswith("Unnamed")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--as-of",
        default=datetime.now().strftime("%Y%m%d"),
        help="Date suffix for exported files, formatted YYYYMMDD.",
    )
    args = parser.parse_args()
    if len(args.as_of) != 8 or not args.as_of.isdigit():
        raise ValueError("--as-of must use YYYYMMDD")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    live, market, recommendations, summary = (
        clean_table(frame) for frame in load_notebook_tables()
    )

    live_path = TABLES_DIR / f"worldcup_2026_projected_knockout_schedule_{args.as_of}.csv"
    market_path = TABLES_DIR / f"worldcup_2026_market_integrated_knockout_schedule_{args.as_of}.csv"
    recommendation_path = TABLES_DIR / f"worldcup_2026_match_recommendations_{args.as_of}.csv"
    summary_path = TABLES_DIR / f"worldcup_2026_live_summary_{args.as_of}.csv"
    workbook_path = TABLES_DIR / f"worldcup_2026_updated_tables_{args.as_of}.xlsx"

    live.to_csv(live_path, index=False, encoding="utf-8-sig")
    market.to_csv(market_path, index=False, encoding="utf-8-sig")
    recommendations.to_csv(recommendation_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # Stable aliases keep external links valid while dated files preserve the audit trail.
    live.to_csv(
        TABLES_DIR / "worldcup_2026_projected_knockout_schedule.csv",
        index=False,
        encoding="utf-8-sig",
    )
    live.to_excel(TABLES_DIR / "worldcup_2026_projected_knockout_schedule.xlsx", index=False)

    market_snapshot_path = MARKET_DIR / f"prediction_market_snapshot_{args.as_of}.csv"
    market_snapshot = (
        clean_table(pd.read_csv(market_snapshot_path))
        if market_snapshot_path.exists()
        else pd.DataFrame()
    )
    fetch_summary_path = SOFASCORE_DIR / "sofascore_fetch_summary.csv"
    fetch_summary = (
        clean_table(pd.read_csv(fetch_summary_path))
        if fetch_summary_path.exists()
        else pd.DataFrame()
    )

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        live.to_excel(writer, sheet_name="Live Bracket", index=False)
        market.to_excel(writer, sheet_name="Market Bracket", index=False)
        recommendations.to_excel(writer, sheet_name="Recommendations", index=False)
        if not market_snapshot.empty:
            market_snapshot.to_excel(writer, sheet_name="Market Snapshot", index=False)
        if not fetch_summary.empty:
            fetch_summary.to_excel(writer, sheet_name="SofaScore Coverage", index=False)

    print(live_path)
    print(market_path)
    print(recommendation_path)
    print(summary_path)
    print(workbook_path)


if __name__ == "__main__":
    main()
