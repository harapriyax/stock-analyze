"""AlphaPulse v2.0 - Multi-Step Autoregressive Forecaster."""

from __future__ import annotations

import numpy as np
import torch

from src.models.base import BaseStockModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def invert_scaled_close(scaler, close_scaled: np.ndarray, n_features: int = 12) -> np.ndarray:
    """Inverts scaled 1D close prices back to original rupee (INR) currency scale."""
    close_flat = np.asarray(close_scaled).reshape(-1)
    dummy = np.zeros((len(close_flat), n_features))
    dummy[:, 0] = close_flat
    return scaler.inverse_transform(dummy)[:, 0]


@torch.no_grad()
def forecast_next_days(
    model: BaseStockModel,
    scaled_window: np.ndarray,
    scaler,
    days: int = 7,
    n_features: int = 12,
    device: torch.device | None = None,
) -> np.ndarray:
    """Executes recursive autoregressive forecasting across forward trading horizon."""
    dev = device or DEVICE
    model.to(dev)
    model.eval()

    window = scaled_window.copy()
    preds_scaled = []

    for _ in range(days):
        # Shape: (1, lookback, n_features)
        xt = torch.tensor(window[np.newaxis, ...], dtype=torch.float32, device=dev)
        next_close_scaled = float(model(xt).cpu().numpy()[0, 0])
        preds_scaled.append(next_close_scaled)

        # Autoregressive sequence update: roll window forward
        next_row = window[-1].copy()
        next_row[0] = next_close_scaled  # Close price is index 0
        window = np.vstack([window[1:], next_row])

    return invert_scaled_close(scaler, np.array(preds_scaled), n_features=n_features)

