# 🚀 AlphaPulse v2.0 — Implementation Plan

> Multi-Model × Multi-Stock Production-Grade Stock Prediction Engine

---

## 1. Executive Summary

Transform the existing single-model (LSTM) single-stock (Reliance) app into a **production-grade
multi-model (LSTM, BiLSTM, CNN, Transformer+LSTM) × multi-stock (12 top Indian companies)** stock
prediction engine with **25-year historical data** and **synthetic data generation**.

---

## 2. Existing Codebase Analysis

### Current Architecture

```
stock-Prediction/
├── app.py             (1183 lines) — Streamlit UI, monolithic
├── train.py           (49 lines)   — CLI train script, Reliance-only
├── src/
│   ├── __init__.py
│   ├── data.py        (117 lines)  — yfinance fetch, 12 indicators, sequencing
│   └── model.py       (167 lines)  — StockLSTM class, train loop, save/load
├── requirements.txt
├── Dockerfile
└── .streamlit/config.toml
```

### Strengths ✅
- Clean 2-layer PyTorch LSTM with early stopping & ReduceLROnPlateau
- 12 solid technical indicators (SMA-20/50, EMA-12, RSI-14, MACD, BB, ATR-14, OBV)
- Professional dark-theme Streamlit dashboard (candles, KPIs, forecast)
- Chronological train/val/test split with MinMaxScaler normalization
- Gradient clipping, dropout regularization, best-model checkpointing

### Weaknesses ❌
- **Single model** — only vanilla 2-layer LSTM, no architecture comparison
- **Single stock** — hardcoded Reliance, function named `fetch_reliance()`
- **No synthetic data** — relies entirely on yfinance data availability
- **Flat structure** — all logic in 3 files, no separation of concerns
- **No config system** — hyperparameters scattered, no YAML/CLI overrides
- **No experiment tracking** — no benchmark storage, no model comparison
- **No tests** — zero unit tests

---

## 3. Target Production Folder Structure

```
stock-Prediction/
│
├── configs/                          # YAML configuration files
│   ├── base.yaml                     # Default hyperparams & shared settings
│   ├── stocks.yaml                   # 12 stock tickers, sectors, date ranges
│   └── models/
│       ├── lstm.yaml                 # LSTM-specific hyperparams
│       ├── bilstm.yaml              # BiLSTM-specific hyperparams
│       ├── cnn.yaml                  # CNN-specific hyperparams
│       └── transformer_lstm.yaml     # Transformer+LSTM hyperparams
│
├── data/                             # Data directory (gitignored)
│   ├── raw/                          # Raw downloaded CSVs per stock
│   ├── processed/                    # Feature-engineered .parquet files
│   ├── synthetic/                    # Synthetically generated backfill data
│   └── combined/                     # Final merged (real+synthetic) datasets
│
├── src/                              # Core Python package
│   ├── __init__.py
│   ├── config.py                     # YAML loader + dataclass configs
│   ├── data/
│   │   ├── __init__.py
│   │   ├── downloader.py             # Multi-stock yfinance downloader
│   │   ├── indicators.py             # Technical indicator computation
│   │   ├── preprocessor.py           # Scaling, windowing, train/test split
│   │   ├── synthetic.py              # GBM, GARCH, block bootstrap generators
│   │   └── validator.py              # Data quality checks & gap detection
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                   # Abstract BaseStockModel class
│   │   ├── lstm.py                   # Vanilla 2-layer LSTM
│   │   ├── bilstm.py                 # Bidirectional LSTM
│   │   ├── cnn.py                    # 1D Temporal CNN
│   │   ├── transformer_lstm.py       # Transformer Encoder + LSTM Decoder
│   │   └── registry.py              # Model factory & auto-registration
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                # Unified training loop for all models
│   │   ├── callbacks.py              # EarlyStopping, Checkpointing, Logging
│   │   └── metrics.py                # RMSE, MAE, MAPE, R², Directional Acc
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py              # Unified prediction interface
│   │   └── forecaster.py             # Multi-step recursive forecasting
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                 # Structured logging
│       ├── io.py                     # Model bundle save/load (JSON meta)
│       └── visualization.py          # Reusable Plotly chart builders
│
├── scripts/                          # CLI scripts
│   ├── download_all_stocks.py
│   ├── generate_synthetic.py
│   ├── train_all_models.py
│   └── benchmark.py
│
├── models/                           # Saved model artifacts (gitignored)
│   ├── {TICKER}/{MODEL_NAME}/
│   │   ├── model.pt
│   │   ├── scaler.pkl
│   │   ├── meta.json
│   │   └── history.json
│   └── benchmarks/
│       └── comparison_results.json
│
├── results/                          # Experiment outputs (gitignored)
│   ├── plots/
│   ├── reports/
│   └── metrics/
│
├── tests/                            # pytest test suite
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_training.py
│   └── test_synthetic.py
│
├── notebooks/                        # Jupyter EDA notebooks
│
├── app.py                            # Streamlit dashboard (refactored)
├── train.py                          # Unified CLI entry point
├── requirements.txt                  # All dependencies
├── pyproject.toml                    # Project metadata
├── Makefile                          # Common commands
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── README.md
└── TASK.md
```

---

## 4. Target Stocks (12 Top Indian Companies)

| # | Company | yfinance | Sector | 25yr Data |
|---|---------|----------|--------|-----------|
| 1 | Reliance Industries | `RELIANCE.NS` | Energy/Conglomerate | ✅ Full |
| 2 | HDFC Bank | `HDFCBANK.NS` | Banking | ✅ Full |
| 3 | TCS | `TCS.NS` | IT Services | ⚠️ ~22yr (need ~3yr synthetic) |
| 4 | Infosys | `INFY.NS` | IT Services | ✅ Full |
| 5 | ICICI Bank | `ICICIBANK.NS` | Banking | ✅ Full |
| 6 | Hindustan Unilever | `HINDUNILVR.NS` | FMCG | ✅ Full |
| 7 | ITC | `ITC.NS` | FMCG/Conglomerate | ✅ Full |
| 8 | State Bank of India | `SBIN.NS` | Banking (PSU) | ✅ Full |
| 9 | Bharti Airtel | `BHARTIARTL.NS` | Telecom | ⚠️ ~24yr (need ~1yr synthetic) |
| 10 | Bajaj Finance | `BAJFINANCE.NS` | NBFC | ✅ Full |
| 11 | Larsen & Toubro | `LT.NS` | Infrastructure | ✅ Full |
| 12 | Sun Pharma | `SUNPHARMA.NS` | Pharmaceuticals | ✅ Full |

**Data Source:** Yahoo Finance via `yfinance` library (`period="max"`)
**Fields:** Open, High, Low, Close, Volume (auto-adjusted)

---

## 5. Model Architectures

### 5.1 Vanilla LSTM (existing — refactored)
```
Input(lookback, F) → LSTM(96) → Dropout → LSTM(48) → Dropout → Dense(32) → ReLU → Dense(1)
```
- Sequential memory, captures momentum & trend continuation

### 5.2 Bidirectional LSTM (BiLSTM) — NEW
```
Input(lookback, F) → BiLSTM(64×2) → Dropout → BiLSTM(32×2) → Dropout → Dense(32) → ReLU → Dense(1)
```
- Forward + backward context, captures both momentum and mean-reversion

### 5.3 1D Temporal CNN — NEW
```
Input(lookback, F) → Conv1D(64,k=3) → BN → ReLU → Conv1D(128,k=3) → BN → ReLU
                   → Conv1D(64,k=3) → BN → ReLU → GlobalAvgPool → Dense(64) → ReLU → Dropout → Dense(1)
```
- Local pattern extraction (candlestick patterns, short-term signals)

### 5.4 Transformer + LSTM Hybrid — NEW
```
Input(lookback, F) → Positional Encoding → TransformerEncoder(4 heads, 2 layers, d=64)
                   → LSTM(64) → Dropout → Dense(32) → ReLU → Dense(1)
```
- Self-attention for long-range dependencies + LSTM for sequential decoding

---

## 6. Synthetic Data Strategy

For stocks with < 25 years of real data:

| Method | Use Case |
|--------|----------|
| **GBM** (Geometric Brownian Motion) | Calibrate μ/σ from earliest real data, generate backward |
| **GARCH(1,1)** | Realistic volatility clustering in synthetic returns |
| **Block Bootstrap** | Resample blocks of real returns preserving autocorrelation |

All synthetic data is tagged with `is_synthetic=True` column for transparency.

---

## 7. Benchmark Plan

**48 total runs** = 12 stocks × 4 models

| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Square Error (₹) |
| MAE | Mean Absolute Error (₹) |
| MAPE | Mean Absolute Percentage Error (%) |
| R² | Coefficient of Determination |
| Directional Accuracy | % correct up/down predictions |
| Training Time | Seconds to convergence |

---

## 8. Tech Stack

| Component | Package | Version |
|-----------|---------|---------|
| Deep Learning | `torch` | 2.15.0+cpu |
| Model Info | `torchinfo` | 1.8+ |
| Arrays | `numpy` | 2.5+ |
| DataFrames | `pandas` | 3.1+ |
| ML Utils | `scikit-learn` | 1.9+ |
| Financial Data | `yfinance` | 1.7+ |
| Indicators | `ta` | 0.11+ |
| GARCH Models | `arch` | 8.0+ |
| Statistics | `scipy` | 1.18+ |
| Config | `pyyaml` | 6.0+ |
| Dashboard | `streamlit` | 1.66+ |
| Charts | `plotly` | 7.1+ |
| Progress | `tqdm` | 4.70+ |
| Parquet I/O | `pyarrow` | 15+ |
| Excel Export | `openpyxl` | 3.1+ |
| Testing | `pytest` + `pytest-cov` | 8+ |
| Linting | `ruff` | 0.3+ |
| Serialization | `joblib` | 1.6+ |

---

## 9. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     configs/ (YAML)                             │
│  stocks.yaml ─── base.yaml ─── models/{lstm,bilstm,cnn,tf}.yaml│
└───────┬─────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────── DATA PIPELINE ─────────────────────────────┐
│  downloader.py → validator.py → synthetic.py → indicators.py   │
│       │               │              │              │           │
│  data/raw/       gap reports    data/synthetic/  data/processed/│
└───────────────────────────────────────────────┬─────────────────┘
                                                │
                                    preprocessor.py (scale + window)
                                                │
                                                ▼
┌───────────────────── MODEL ZOO ─────────────────────────────────┐
│  base.py ← lstm.py │ bilstm.py │ cnn.py │ transformer_lstm.py  │
│                     registry.py (factory pattern)               │
└───────────────────────────────────────────────┬─────────────────┘
                                                │
                                                ▼
┌───────────────────── TRAINING ENGINE ───────────────────────────┐
│  trainer.py (unified) + callbacks.py + metrics.py               │
│  → models/{TICKER}/{MODEL}/model.pt + meta.json + history.json  │
└───────────────────────────────────────────────┬─────────────────┘
                                                │
                                                ▼
┌───────────────────── INFERENCE ──────────────────────────────────┐
│  predictor.py + forecaster.py → results/benchmarks/              │
└───────────────────────────────────────────────┬──────────────────┘
                                                │
                                                ▼
┌───────────────────── STREAMLIT UI ──────────────────────────────┐
│  app.py — Stock selector, Model selector, Train, Compare,       │
│           Forecast overlay, Benchmark leaderboard               │
└─────────────────────────────────────────────────────────────────┘
```
