"""AlphaPulse v2.0 - 1D Temporal Convolutional Architecture."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.base import BaseStockModel


class StockCNN(BaseStockModel):
    """1D Temporal Convolutional Network extracting multi-scale sequential price patterns."""

    def __init__(
        self,
        n_features: int,
        filters: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        filters = filters or [64, 128, 64]
        self._filters = filters
        self._kernel_size = kernel_size

        layers = []
        in_c = n_features
        for out_c in filters:
            layers.append(nn.Conv1d(in_c, out_c, kernel_size=kernel_size, padding=kernel_size // 2))
            layers.append(nn.BatchNorm1d(out_c))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_c = out_c

        self.conv_net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(filters[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def model_type(self) -> str:
        return "cnn"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transpose (batch, lookback, features) -> (batch, features, lookback)
        feat = x.transpose(1, 2)
        out = self.conv_net(feat)
        pooled = self.pool(out).squeeze(-1)
        return self.head(pooled)
