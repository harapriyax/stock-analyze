"""AlphaPulse v2.0 - Training Callbacks."""

from __future__ import annotations

import copy
from typing import Any

import torch


class EarlyStopping:
    """Early stops training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience: int = 6,
        min_delta: float = 1e-6,
        restore_best_weights: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.best_state_dict: dict[str, Any] | None = None

    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def restore(self, model: torch.nn.Module) -> None:
        """Restores model to best recorded state dict."""
        if self.restore_best_weights and self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)

