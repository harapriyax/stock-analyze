"""AlphaPulse v2.0 - Hybrid Transformer-LSTM Architecture."""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from src.models.base import BaseStockModel


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for attention sequence modeling."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class StockTransformerLSTM(BaseStockModel):
    """Hybrid architecture pairing Multi-Head Self-Attention with Recurrent LSTM decoding."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        lstm_units: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=lstm_units, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(lstm_units, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def model_type(self) -> str:
        return "transformer_lstm"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project features to d_model space and add positional encoding
        proj = self.pos_encoder(self.input_proj(x))
        # Self-attention encoding
        encoded = self.transformer_encoder(proj)
        # Recurrent temporal decoding
        lstm_out, _ = self.lstm(encoded)
        last_hidden = self.drop(lstm_out[:, -1, :])
        return self.head(last_hidden)
