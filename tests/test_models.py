"""AlphaPulse v2.0 - Unit Tests for Deep Learning Architectures."""

from __future__ import annotations

import pytest
import torch

from src.models.bilstm import StockBiLSTM
from src.models.cnn import StockCNN
from src.models.lstm import StockLSTM
from src.models.registry import get_model, list_available_models
from src.models.transformer_lstm import StockTransformerLSTM


@pytest.fixture
def dummy_batch():
    batch_size = 8
    lookback = 30
    n_features = 12
    return torch.randn(batch_size, lookback, n_features)


def test_stock_lstm_forward(dummy_batch):
    model = StockLSTM(n_features=12, units=32, num_layers=2)
    out = model(dummy_batch)
    assert out.shape == (8, 1)
    assert model.count_parameters() > 0
    assert model.model_type() == "lstm"


def test_stock_bilstm_forward(dummy_batch):
    model = StockBiLSTM(n_features=12, units=32, num_layers=2)
    out = model(dummy_batch)
    assert out.shape == (8, 1)
    assert model.count_parameters() > 0
    assert model.model_type() == "bilstm"


def test_stock_cnn_forward(dummy_batch):
    model = StockCNN(n_features=12, filters=[32, 64, 32], kernel_size=3)
    out = model(dummy_batch)
    assert out.shape == (8, 1)
    assert model.count_parameters() > 0
    assert model.model_type() == "cnn"


def test_stock_transformer_lstm_forward(dummy_batch):
    model = StockTransformerLSTM(
        n_features=12,
        d_model=32,
        nhead=2,
        num_transformer_layers=1,
        dim_feedforward=64,
        lstm_units=32,
    )
    out = model(dummy_batch)
    assert out.shape == (8, 1)
    assert model.count_parameters() > 0
    assert model.model_type() == "transformer_lstm"


def test_model_registry():
    models = list_available_models()
    assert "lstm" in models
    assert "bilstm" in models
    assert "cnn" in models
    assert "transformer_lstm" in models

    # Test factory instantiation
    for m_name in models:
        m = get_model(m_name, n_features=12)
        assert m.count_parameters() > 0

