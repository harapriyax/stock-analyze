"""AlphaPulse v2.0 - Configuration Management Module.

Loads and validates YAML configuration files for data pipelines, model architectures,
and training parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


@dataclass
class DataConfig:
    lookback: int = 60
    test_ratio: float = 0.15
    val_ratio: float = 0.10
    feature_columns: list[str] = field(
        default_factory=lambda: [
            "Close",
            "Volume",
            "sma_20",
            "sma_50",
            "ema_12",
            "rsi_14",
            "macd",
            "macd_signal",
            "bb_high",
            "bb_low",
            "atr_14",
            "obv",
        ]
    )
    auto_adjust: bool = True
    history_years: int = 25


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    patience: int = 6
    grad_clip: float = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    min_lr: float = 1e-5
    loss_fn: str = "mse"


@dataclass
class StockConfig:
    ticker: str
    name: str
    sector: str
    start_date: str = "2001-11-01"
    needs_synthetic: bool = False
    synthetic_method: str = "gbm"


def load_yaml(file_path: Path | str) -> dict[str, Any]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def get_base_config(configs_dir: Path | None = None) -> dict[str, Any]:
    c_dir = configs_dir or CONFIGS_DIR
    return load_yaml(c_dir / "base.yaml")


def get_all_stocks(configs_dir: Path | None = None) -> list[StockConfig]:
    c_dir = configs_dir or CONFIGS_DIR
    data = load_yaml(c_dir / "stocks.yaml")
    stocks_raw = data.get("stocks", [])
    return [
        StockConfig(
            ticker=s["ticker"],
            name=s["name"],
            sector=s["sector"],
            start_date=s.get("start_date", "2001-11-01"),
            needs_synthetic=s.get("needs_synthetic", False),
            synthetic_method=s.get("synthetic_method", "gbm"),
        )
        for s in stocks_raw
    ]


def get_stock_config(ticker: str, configs_dir: Path | None = None) -> StockConfig:
    stocks = get_all_stocks(configs_dir)
    for stock in stocks:
        if stock.ticker.upper() == ticker.upper():
            return stock
    # If not in curated list, return default
    return StockConfig(ticker=ticker, name=ticker, sector="General")


def get_model_config(model_name: str, configs_dir: Path | None = None) -> dict[str, Any]:
    c_dir = configs_dir or CONFIGS_DIR
    model_file = c_dir / "models" / f"{model_name.lower()}.yaml"
    if not model_file.exists():
        raise ValueError(
            f"Unsupported model '{model_name}'. No configuration found at {model_file}."
        )
    return load_yaml(model_file)
