"""AlphaPulse v2.0 - Stock Data Downloader & Caching Module."""

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def sanitize_ticker_filename(ticker: str) -> str:
    """Sanitizes ticker symbols into filesystem-safe filenames (e.g. RELIANCE.NS -> RELIANCE_NS)."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", ticker.strip())


def fetch_stock_history(
    ticker: str,
    period: str = "max",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Fetches historical OHLCV data using yfinance with offline dataset fallbacks."""
    clean_name = sanitize_ticker_filename(ticker)
    symbol_base = ticker.split(".")[0].strip().upper()
    
    # Candidate local files
    candidate_files = [
        PROJECT_ROOT / "datasets" / f"{symbol_base}_25Y.csv",
        PROJECT_ROOT / "datasets" / f"{clean_name}_25Y.csv",
        PROJECT_ROOT / "data" / "raw" / f"{clean_name}.csv",
    ]

    def _try_load_local() -> pd.DataFrame | None:
        for f in candidate_files:
            if f.exists():
                try:
                    df = pd.read_csv(f, index_col=0, parse_dates=True)
                    if not df.empty and {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
                        df.index.name = "Date"
                        return df.sort_index()
                except Exception:
                    continue
        return None

    # Check offline pre-downloaded 25Y datasets first when no custom date slice is asked
    if not start and not end:
        local_df = _try_load_local()
        if local_df is not None:
            return local_df

    # Download via yfinance
    kwargs = {"auto_adjust": True, "progress": False}
    if start or end:
        if start:
            kwargs["start"] = start
        if end:
            kwargs["end"] = end
    else:
        kwargs["period"] = period

    try:
        data = yf.download(ticker, **kwargs)
    except Exception:
        data = pd.DataFrame()

    # Fallback to offline dataset if network download failed or empty
    if data.empty:
        local_df = _try_load_local()
        if local_df is not None:
            return local_df
        raise ValueError(f"No market data returned for ticker '{ticker}'. Please check connectivity or symbol.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename(columns=str.title)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Downloaded market data missing required OHLCV columns: {missing}")

    data = data[required].dropna()
    data.index.name = "Date"
    return data.sort_index()


def cache_raw_stock_data(
    df: pd.DataFrame,
    ticker: str,
    output_dir: str | Path = "data/raw",
) -> Path:
    """Caches raw downloaded data into a CSV file."""
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{sanitize_ticker_filename(ticker)}.csv"
    df.to_csv(out_file)
    return out_file
