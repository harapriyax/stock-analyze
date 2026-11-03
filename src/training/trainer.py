"""AlphaPulse v2.0 - Unified Model Training Engine."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseStockModel
from src.training.callbacks import EarlyStopping
from src.utils.logger import setup_logger

logger = setup_logger("alphapulse.trainer")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Converts numpy arrays to PyTorch Tensor DataLoader."""
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: BaseStockModel,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    patience: int = 6,
    grad_clip: float = 1.0,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 3,
    min_lr: float = 1e-5,
    device: torch.device | None = None,
) -> tuple[BaseStockModel, dict[str, list[float]]]:
    """Unified training loop supporting any registered model architecture."""
    dev = device or DEVICE
    model = model.to(dev)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=scheduler_factor,
        patience=scheduler_patience,
        min_lr=min_lr,
    )
    early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)

    train_loader = _to_tensor_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = _to_tensor_loader(x_val, y_val, batch_size=batch_size, shuffle=False)

    history: dict[str, list[float]] = {"loss": [], "val_loss": []}

    logger.info(
        f"Training {model.model_type().upper()} on {dev} | "
        f"Train: {len(x_train):,} samples | Val: {len(x_val):,} samples | Max Epochs: {epochs}"
    )

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_batch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()

            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            train_batch_losses.append(loss.item())

        epoch_train_loss = float(np.mean(train_batch_losses))

        # 2. Validation Phase
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                val_batch_losses.append(loss_fn(model(xb), yb).item())

        epoch_val_loss = float(np.mean(val_batch_losses)) if val_batch_losses else epoch_train_loss

        history["loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

        # 3. Learning Rate Scheduling
        scheduler.step(epoch_val_loss)

        # 4. Check Early Stopping
        if early_stopping(epoch_val_loss, model):
            logger.info(f"Early stopping triggered at epoch {epoch} (Best Val Loss: {early_stopping.best_loss:.6f})")
            break

    # Restore best checkpoint
    early_stopping.restore(model)
    model.eval()
    return model, history

