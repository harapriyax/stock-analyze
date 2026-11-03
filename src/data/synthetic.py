"""AlphaPulse v2.0 - Calibrated Synthetic Data Generation Module."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_gbm_path(
    s0: float,
    n_days: int,
    mu: float = 0.10,
    sigma: float = 0.20,
    seed: int | None = 42,
) -> np.ndarray:
    """Generates a Geometric Brownian Motion (GBM) price trajectory."""
    if seed is not None:
        np.random.seed(seed)

    dt = 1.0 / 252.0
    drift = (mu - 0.5 * (sigma**2)) * dt
    diffusion = sigma * np.sqrt(dt)

    shocks = np.random.normal(0, 1, size=n_days)
    log_returns = drift + diffusion * shocks

    prices = np.zeros(n_days, dtype=float)
    prices[0] = s0 * np.exp(log_returns[0])
    for t in range(1, n_days):
        prices[t] = prices[t - 1] * np.exp(log_returns[t])

    return np.maximum(prices, 1e-4)


def backfill_synthetic_history(
    real_df: pd.DataFrame,
    target_start_date: str = "2000-01-01",
    method: str = "gbm",
    seed: int | None = 42,
) -> pd.DataFrame:
    """Seamlessly backfills historical listing gaps to achieve a 25-year modeling horizon."""
    real_sorted = real_df.sort_index().copy()
    target_start = pd.to_datetime(target_start_date)

    if "is_synthetic" not in real_sorted.columns:
        real_sorted["is_synthetic"] = 0

    first_real_date = pd.to_datetime(real_sorted.index[0])
    if first_real_date <= target_start:
        return real_sorted

    # Construct business day calendar for missing history
    synthetic_dates = pd.bdate_range(start=target_start, end=first_real_date - pd.Timedelta(days=1))
    n_days = len(synthetic_dates)
    if n_days == 0:
        return real_sorted

    first_real_close = float(real_sorted["Close"].iloc[0])
    first_real_volume = float(real_sorted["Volume"].iloc[:30].mean()) if len(real_sorted) >= 30 else float(real_sorted["Volume"].iloc[0])

    # Calibrate parameters from earliest real trading window
    lookback_window = min(len(real_sorted), 252)
    sample_returns = real_sorted["Close"].pct_change().dropna().iloc[:lookback_window]
    sigma = float(sample_returns.std() * np.sqrt(252)) if len(sample_returns) > 5 else 0.25
    mu = float(sample_returns.mean() * 252) if len(sample_returns) > 5 else 0.12
    sigma = max(0.10, min(sigma, 0.60))

    # Generate path backwards to match first_real_close at the boundary
    raw_path = generate_gbm_path(s0=100.0, n_days=n_days, mu=mu, sigma=sigma, seed=seed)
    scale_factor = first_real_close / raw_path[-1]
    synth_close = raw_path * scale_factor

    # Synthesize consistent OHLCV relations
    if seed is not None:
        np.random.seed(seed + 1)
    noise_high = np.random.uniform(1.002, 1.018, size=n_days)
    noise_low = np.random.uniform(0.982, 0.998, size=n_days)
    noise_open = np.random.uniform(0.995, 1.005, size=n_days)

    synth_open = synth_close * noise_open
    synth_high = np.maximum(synth_close, synth_open) * noise_high
    synth_low = np.minimum(synth_close, synth_open) * noise_low
    synth_volume = np.maximum(100.0, np.random.normal(first_real_volume, first_real_volume * 0.20, size=n_days))

    synth_df = pd.DataFrame(
        {
            "Open": synth_open,
            "High": synth_high,
            "Low": synth_low,
            "Close": synth_close,
            "Volume": synth_volume,
            "is_synthetic": 1,
        },
        index=synthetic_dates,
    )
    synth_df.index.name = "Date"

    combined = pd.concat([synth_df, real_sorted]).sort_index()
    return combined
