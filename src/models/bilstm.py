"""AlphaPulse v2.0 - Bidirectional LSTM Architecture."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import BaseStockModel


class StockBiLSTM(BaseStockModel):
    """Bidirectional LSTM network capturing dual-horizon trend and retrospective momentum context."""

    def __init__(
        self,
        n_features: int,
        units: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self._units = units
        self._num_layers = num_layers
        self.bilstm = nn.LSTM(
            input_size=n_features,
            hidden_size=units,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(units * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def model_type(self) -> str:
        return "bilstm"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.bilstm(x)
        last_hidden = self.drop(out[:, -1, :])
        return self.head(last_hidden)
