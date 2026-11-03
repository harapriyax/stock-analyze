"""AlphaPulse v2.0 - Data Validation & Quality Assurance."""

from __future__ import annotations

import pandas as pd


def calculate_history_coverage(df: pd.DataFrame, target_years: int = 25) -> float:
    """Calculates coverage percentage of target historical depth (e.g., 25 years)."""
    if df.empty or len(df) < 2:
        return 0.0

    start_date = pd.to_datetime(df.index[0])
    end_date = pd.to_datetime(df.index[-1])
    actual_years = (end_date - start_date).days / 365.25
    coverage = (actual_years / target_years) * 100.0
    return float(min(100.0, max(0.0, coverage)))


def validate_ohlcv(df: pd.DataFrame) -> bool:
    """Validates structural hygiene and numerical integrity of market data."""
    if df.empty:
        return False

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        return False

    # Verify strictly positive price values
    if (df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        return False

    # Verify chronological monotonic ordering
    if not df.index.is_monotonic_increasing:
        return False

    return True
