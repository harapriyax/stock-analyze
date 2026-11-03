"""AlphaPulse v2.0 - Normalization & Sliding Window Sequencing Module."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data.indicators import FEATURE_COLUMNS


def prepare_sequences(
    featured: pd.DataFrame,
    lookback: int = 60,
    test_ratio: float = 0.15,
    val_ratio: float = 0.10,
) -> dict:
    """Prepares normalized sliding-window tensor sequences with strict chronological splits."""
    values = featured[FEATURE_COLUMNS].astype(float).values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    x, y = [], []
    for i in range(lookback, len(scaled)):
        x.append(scaled[i - lookback : i])
        y.append(scaled[i, 0])  # Close price is index 0 in FEATURE_COLUMNS

    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    total_len = len(x)
    test_split = int(total_len * (1 - test_ratio))
    val_split = int(test_split * (1 - val_ratio))

    x_train, y_train = x[:val_split], y[:val_split]
    x_val, y_val = x[val_split:test_split], y[val_split:test_split]
    x_test, y_test = x[test_split:], y[test_split:]

    dates = featured.index[lookback:]
    train_dates = dates[:val_split]
    val_dates = dates[val_split:test_split]
    test_dates = dates[test_split:]

    return {
        "scaler": scaler,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "train_dates": train_dates,
        "val_dates": val_dates,
        "test_dates": test_dates,
    }
