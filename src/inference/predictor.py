"""AlphaPulse v2.0 - Batch & Uncertainty Inference Engine."""

from __future__ import annotations

import numpy as np
import torch

from src.models.base import BaseStockModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_array(
    model: BaseStockModel,
    x: np.ndarray,
    device: torch.device | None = None,
) -> np.ndarray:
    """Computes deterministic model predictions for a batch of sequential inputs."""
    dev = device or DEVICE
    model.to(dev)
    model.eval()

    xt = torch.tensor(x, dtype=torch.float32, device=dev)
    preds = model(xt).cpu().numpy().reshape(-1)
    return preds


def predict_with_uncertainty(
    model: BaseStockModel,
    x: np.ndarray,
    n_samples: int = 50,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimates epistemic uncertainty using Monte Carlo Dropout rollouts."""
    dev = device or DEVICE
    model.to(dev)

    # Enable dropout during inference for MC sampling
    def enable_dropout(m: torch.nn.Module):
        if isinstance(m, torch.nn.Dropout):
            m.train()

    model.apply(enable_dropout)
    xt = torch.tensor(x, dtype=torch.float32, device=dev)

    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(xt).cpu().numpy().reshape(-1)
            samples.append(pred)

    model.eval()
    samples_arr = np.array(samples)
    mean_preds = np.mean(samples_arr, axis=0)
    std_preds = np.std(samples_arr, axis=0)
    return mean_preds, std_preds

