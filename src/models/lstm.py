"""AlphaPulse v2.0 - Standard Recurrent LSTM Architecture."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import BaseStockModel


class StockLSTM(BaseStockModel):
    """Multi-layer Recurrent LSTM forecasting network with dropout regularization."""

    def __init__(
        self,
        n_features: int,
        units: int = 96,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self._units = units
        self._num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(units, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def model_type(self) -> str:
        return "lstm"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = self.drop(out[:, -1, :])
        return self.head(last_hidden)
