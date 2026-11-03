"""AlphaPulse v2.0 - Quantitative Technical Indicator Suite."""

from __future__ import annotations

import pandas as pd
import ta

FEATURE_COLUMNS = [
    "Close",
    "Volume",
    "sma_20",
    "sma_50",
    "ema_12",
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_high",
    "bb_low",
    "atr_14",
    "obv",
]


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Computes the 12-factor quantitative technical indicator suite."""
    out = df.copy()

    close = out["Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    volume = out["Volume"].astype(float)

    # Moving Averages
    out["sma_20"] = ta.trend.sma_indicator(close, window=20)
    out["sma_50"] = ta.trend.sma_indicator(close, window=50)
    out["ema_12"] = ta.trend.ema_indicator(close, window=12)

    # Momentum
    out["rsi_14"] = ta.momentum.rsi(close, window=14)
    macd = ta.trend.MACD(close)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()

    # Volatility
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()
    out["atr_14"] = ta.volatility.average_true_range(high, low, close, window=14)

    # Volume
    out["obv"] = ta.volume.on_balance_volume(close, volume)

    # Clean initial NaN values from rolling windows
    out = out.dropna(subset=FEATURE_COLUMNS)
    return out
