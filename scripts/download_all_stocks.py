"""AlphaPulse v2.0 - Batch Stock Downloader & Synthetic Backfill Script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import get_all_stocks, get_base_config
from src.data.downloader import cache_raw_stock_data, fetch_stock_history, sanitize_ticker_filename
from src.data.indicators import add_technical_indicators
from src.data.synthetic import backfill_synthetic_history
from src.data.validator import calculate_history_coverage, validate_ohlcv
from src.utils.logger import setup_logger

logger = setup_logger("scripts.download_all")


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess 25-year stock datasets.")
    parser.add_argument("--period", type=str, default="max", help="History period (default: max)")
    parser.add_argument("--backfill", action="store_true", default=True, help="Enable synthetic backfill for < 25y")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stocks = get_all_stocks()
    logger.info(f"Targeting {len(stocks)} top Indian stocks...")

    summary_records = []

    for stock in stocks:
        ticker = stock.ticker
        logger.info(f"\n{'='*60}\nProcessing {stock.name} ({ticker}) | Sector: {stock.sector}\n{'='*60}")
        try:
            # 1. Download raw history
            raw_df = fetch_stock_history(ticker, period=args.period)
            cache_raw_stock_data(raw_df, ticker, output_dir="data/raw")

            # 2. Check coverage and backfill if necessary
            coverage = calculate_history_coverage(raw_df, target_years=25)
            logger.info(f"Initial 25-year coverage: {coverage:.1f}%")

            processed_df = raw_df
            if args.backfill and stock.needs_synthetic:
                processed_df = backfill_synthetic_history(
                    raw_df,
                    target_start_date=stock.start_date,
                    method=stock.synthetic_method,
                )
                coverage = calculate_history_coverage(processed_df, target_years=25)
                logger.info(f"Post-backfill 25-year coverage: {coverage:.1f}%")

            # 3. Add indicators
            featured_df = add_technical_indicators(processed_df)

            # 4. Save processed parquet
            clean_name = sanitize_ticker_filename(ticker)
            parquet_path = out_dir / f"{clean_name}.parquet"
            featured_df.to_parquet(parquet_path)
            logger.info(f"Saved processed dataset ({len(featured_df):,} rows) to {parquet_path}")

            summary_records.append(
                {
                    "Ticker": ticker,
                    "Name": stock.name,
                    "Sector": stock.sector,
                    "Start Date": featured_df.index[0].strftime("%Y-%m-%d"),
                    "End Date": featured_df.index[-1].strftime("%Y-%m-%d"),
                    "Total Rows": len(featured_df),
                    "Synthetic Rows": int(featured_df["is_synthetic"].sum()) if "is_synthetic" in featured_df.columns else 0,
                    "25Y Coverage": f"{coverage:.1f}%",
                }
            )

        except Exception as exc:
            logger.error(f"Failed to process {ticker}: {exc}")
            summary_records.append(
                {
                    "Ticker": ticker,
                    "Name": stock.name,
                    "Sector": stock.sector,
                    "Start Date": "FAILED",
                    "End Date": "FAILED",
                    "Total Rows": 0,
                    "Synthetic Rows": 0,
                    "25Y Coverage": "0.0%",
                }
            )

    print("\n" + "=" * 80)
    print("ALPHAPULSE v2.0 - DATASET PIPELINE SUMMARY")
    print("=" * 80)
    summary_df = pd.DataFrame(summary_records)
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
