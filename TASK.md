# 📋 TASK.md — AlphaPulse v2.0 Complete Task Tracker

> **Goal:** Multi-model (LSTM, BiLSTM, CNN, Transformer+LSTM) × multi-stock (12 top Indian
> companies) stock prediction engine with 25-year data + synthetic backfill.
>
> **Python:** 3.13.5 (Anaconda) | **Venv:** `.venv/` | **Framework:** PyTorch 2.15.0+cpu

---

## ✅ Phase 0: Project Setup & Dependencies
**Status: COMPLETED**

- [x] **T0.1** Analyze existing codebase (`app.py`, `src/data.py`, `src/model.py`, `train.py`)
- [x] **T0.2** Create Python 3.13 virtual environment (`.venv/`)
- [x] **T0.3** Upgrade pip, setuptools, wheel
- [x] **T0.4** Install PyTorch 2.15.0+cpu from official PyTorch index
- [x] **T0.5** Install all remaining production dependencies:
  - `numpy`, `pandas`, `scikit-learn`, `joblib`, `yfinance`, `ta`, `arch`, `scipy`,
    `streamlit`, `plotly`, `pyyaml`, `tqdm`, `torchinfo`, `pytest`, `pytest-cov`,
    `ruff`, `pyarrow`, `openpyxl`
- [x] **T0.6** Create production folder structure:
  - `configs/`, `configs/models/`
  - `src/data/`, `src/models/`, `src/training/`, `src/inference/`, `src/utils/`
  - `scripts/`, `notebooks/`, `tests/`
  - `data/raw/`, `data/processed/`, `data/synthetic/`
  - `models/benchmarks/`, `results/benchmarks/`
- [x] **T0.7** Update `.gitignore` (data/, models/, results/, *.pt, *.pkl, *.parquet, .env, __pycache__/)
- [x] **T0.8** Create `pyproject.toml` with project metadata and tool configs
- [x] **T0.9** Verify imports and test suite

---

## ✅ Phase 1: Configuration System
**Status: COMPLETED**

- [x] **T1.1** Create `configs/base.yaml` (Lookback: 60, Test: 0.15, Val: 0.10, 12 features, Adam params)
- [x] **T1.2** Create `configs/stocks.yaml` (12 curated top Indian equities with 25-year horizons)
- [x] **T1.3** Create model-specific YAML configurations:
  - [x] `configs/models/lstm.yaml`
  - [x] `configs/models/bilstm.yaml`
  - [x] `configs/models/cnn.yaml`
  - [x] `configs/models/transformer_lstm.yaml`
- [x] **T1.4** Create `src/config.py` (dataclasses `DataConfig`, `TrainingConfig`, `StockConfig`, YAML loaders)

---

## ✅ Phase 2: Data Pipeline
**Status: COMPLETED**

- [x] **T2.1** Create `src/data/__init__.py`
- [x] **T2.2** Create `src/data/downloader.py` (multi-stock fetcher with retry logic and caching)
- [x] **T2.3** Create `scripts/download_all_stocks.py` (CLI batch downloader for all 12 stocks)
- [x] **T2.4** Create `src/data/validator.py` (OHLCV hygiene, gap detection, 25-year coverage calculation)
- [x] **T2.5** Create `src/data/indicators.py` (12-factor quantitative technical indicators)
- [x] **T2.6** Create `src/data/synthetic.py` (calibrated GBM & GARCH backfill generators with continuity)
- [x] **T2.7** Create `src/data/preprocessor.py` (sliding window sequences, MinMaxScaler without lookahead leakage)

---

## ✅ Phase 3: Model Zoo
**Status: COMPLETED**

- [x] **T3.1** Create `src/models/__init__.py`
- [x] **T3.2** Create `src/models/base.py` (`BaseStockModel` abstract class with parameter counting)
- [x] **T3.3** Create `src/models/lstm.py` (`StockLSTM` dual-layer recurrent network)
- [x] **T3.4** Create `src/models/bilstm.py` (`StockBiLSTM` bidirectional recurrent context network)
- [x] **T3.5** Create `src/models/cnn.py` (`StockCNN` 1D temporal convolutional pattern extractor)
- [x] **T3.6** Create `src/models/transformer_lstm.py` (`StockTransformerLSTM` self-attention + recurrent decoder)
- [x] **T3.7** Create `src/models/registry.py` (dynamic factory & registry lookup)

---

## ✅ Phase 4: Training Engine
**Status: COMPLETED**

- [x] **T4.1** Create `src/training/__init__.py`
- [x] **T4.2** Create `src/training/metrics.py` (RMSE, MAE, MAPE, R², Directional Accuracy)
- [x] **T4.3** Create `src/training/callbacks.py` (EarlyStopping with best weights restoration)
- [x] **T4.4** Create `src/training/trainer.py` (unified PyTorch training loop supporting any registered model)

---

## ✅ Phase 5: Inference & Forecasting
**Status: COMPLETED**

- [x] **T5.1** Create `src/inference/__init__.py`
- [x] **T5.2** Create `src/inference/predictor.py` (deterministic & Monte Carlo Dropout uncertainty predictions)
- [x] **T5.3** Create `src/inference/forecaster.py` (recursive autoregressive multi-day forecasting)

---

## ✅ Phase 6: Utility Modules & CLI Scripts
**Status: COMPLETED**

- [x] **T6.1** Create `src/utils/__init__.py`
- [x] **T6.2** Create `src/utils/logger.py` (standardized structured logger)
- [x] **T6.3** Create `src/utils/io.py` (JSON metadata, weights state_dict, scaler persistence)
- [x] **T6.4** Create `src/utils/visualization.py` (Plotly dark-fintech charts)
- [x] **T6.5** Create `train.py` (unified CLI for individual runs and full benchmark suites)

---

## ✅ Phase 7: Streamlit Dashboard Overhaul
**Status: COMPLETED**

- [x] **T7.1** Upgrade `app.py` with multi-stock selector (12 curated equities + Custom symbol)
- [x] **T7.2** Add multi-model selector (LSTM, BiLSTM, 1D CNN, Transformer+LSTM)
- [x] **T7.3** Implement 5-tab fintech UI:
  - 📊 **Market Terminal** (Candlesticks, technical overlays, volume, 12 features)
  - 🧠 **Neural Network & Training** (Loss curves, architecture breakdown, live retraining)
  - 🎯 **Hold-Out Validation** (Actual vs Predicted, error corridor, 5 KPIs, prediction journal)
  - 🔮 **Forward Forecast** (Autoregressive projection, confidence intervals, daily targets)
  - 🏆 **Model Zoo Benchmark** (Side-by-side leaderboard across architectures)

---

## ✅ Phase 8: Testing & Verification
**Status: COMPLETED**

- [x] **T8.1** Create `tests/test_models.py` (Forward pass & shapes for all 4 models + registry)
- [x] **T8.2** Create `tests/test_metrics.py` (RMSE, MAE, MAPE, R², Directional Accuracy tests)
- [x] **T8.3** Create `tests/test_synthetic.py` (GBM positive path & backfill coverage tests)
- [x] **T8.4** Run pytest suite: **10/10 tests passed in 39.04s!**

---

## ✅ Phase 9: Documentation & Deployment
**Status: COMPLETED**

- [x] **T9.1** Comprehensive `README.md` with architecture diagrams and quickstart
- [x] **T9.2** Automation `Makefile` with targets (`make download`, `make train`, `make app`, `make test`)
- [x] **T9.3** `IMPLEMENTATION_PLAN.md` full architectural design document

---

## 📊 Curated 12 Indian Stocks Reference

| # | Company | yfinance Ticker | Sector | 25y History Coverage |
|---|---------|----------------|--------|----------------------|
| 1 | Reliance Industries | `RELIANCE.NS` | Energy/Conglomerate | 100% Real |
| 2 | HDFC Bank | `HDFCBANK.NS` | Banking | 100% Real |
| 3 | Tata Consultancy Services | `TCS.NS` | IT Services | ~22y Real + ~3y Synthetic |
| 4 | Infosys | `INFY.NS` | IT Services | 100% Real |
| 5 | ICICI Bank | `ICICIBANK.NS` | Banking | 100% Real |
| 6 | Hindustan Unilever | `HINDUNILVR.NS` | FMCG | 100% Real |
| 7 | ITC Ltd | `ITC.NS` | FMCG/Conglomerate | 100% Real |
| 8 | State Bank of India | `SBIN.NS` | Banking (PSU) | 100% Real |
| 9 | Bharti Airtel | `BHARTIARTL.NS` | Telecom | ~24y Real + ~1y Synthetic |
| 10 | Bajaj Finance | `BAJFINANCE.NS` | NBFC | 100% Real |
| 11 | Larsen & Toubro | `LT.NS` | Infrastructure | 100% Real |
| 12 | Sun Pharma | `SUNPHARMA.NS` | Pharmaceuticals | 100% Real |
