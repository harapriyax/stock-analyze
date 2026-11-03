"""AlphaPulse v2.0 - Unit Tests for Synthetic Data Generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.synthetic import backfill_synthetic_history, generate_gbm_path


def test_gbm_trajectory():
    prices = generate_gbm_path(s0=100.0, n_days=50, mu=0.1, sigma=0.2, seed=42)
    assert len(prices) == 50
    assert np.all(prices > 0)  # Prices must remain strictly positive


def test_synthetic_backfill():
    # Construct a sample real dataset starting in 2010
    dates = pd.bdate_range("2010-01-01", "2010-06-01")
    n = len(dates)
    real_df = pd.DataFrame(
        {
            "Open": np.linspace(100, 150, n),
            "High": np.linspace(102, 153, n),
            "Low": np.linspace(98, 148, n),
            "Close": np.linspace(101, 151, n),
            "Volume": np.full(n, 1000000.0),
        },
        index=dates,
    )

    # Backfill to 2008
    combined = backfill_synthetic_history(real_df, target_start_date="2008-01-01", seed=42)

    assert len(combined) > len(real_df)
    assert "is_synthetic" in combined.columns
    assert combined["is_synthetic"].sum() > 0
    assert combined.index[0] <= pd.to_datetime("2008-01-05")
    # All prices must be positive
    assert (combined[["Open", "High", "Low", "Close"]] > 0).all().all()

