"""AlphaPulse v2.0 - Dynamic Model Zoo Factory & Registry."""

from __future__ import annotations

from typing import Type

from src.models.base import BaseStockModel
from src.models.bilstm import StockBiLSTM
from src.models.cnn import StockCNN
from src.models.lstm import StockLSTM
from src.models.transformer_lstm import StockTransformerLSTM

MODEL_REGISTRY: dict[str, Type[BaseStockModel]] = {
    "lstm": StockLSTM,
    "bilstm": StockBiLSTM,
    "cnn": StockCNN,
    "transformer_lstm": StockTransformerLSTM,
}


def list_available_models() -> list[str]:
    """Returns canonical list of registered model architectures."""
    return list(MODEL_REGISTRY.keys())


def get_model(name: str, n_features: int, **kwargs) -> BaseStockModel:
    """Instantiates a neural forecasting model architecture by name."""
    key = name.lower().strip()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model architecture '{name}'. Available: {list_available_models()}"
        )
    model_cls = MODEL_REGISTRY[key]
    return model_cls(n_features=n_features, **kwargs)
