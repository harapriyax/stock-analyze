# 📈 AlphaPulse v2.0 — Multi-Model Stock Intelligence Engine

Enterprise-grade deep learning and quantitative forecasting platform for Indian equity markets (NSE/BSE). Powered by **PyTorch**, **Streamlit**, and **Plotly**, AlphaPulse provides end-to-end capabilities spanning 25-year historical data ingestion, calibrated synthetic data backfilling, multi-factor indicator extraction, and a comparative Model Zoo featuring 4 neural network architectures.

---

## 🌟 Key Highlights

- **🏛️ 12 Top Indian Equities**: Curated coverage across major sectors (Reliance, HDFC Bank, TCS, Infosys, ICICI Bank, HUL, ITC, SBI, Bharti Airtel, Bajaj Finance, L&T, Sun Pharma) plus Custom Symbol search.
- **📅 25-Year Historical Depth**: Long-horizon time-series modeling reaching back across multiple macroeconomic cycles.
- **🎲 Calibrated Synthetic Backfilling**: Seamlessly fills historical listing gaps using Geometric Brownian Motion (GBM) and GARCH(1,1) volatility models with statistical continuity and transparency flags (`is_synthetic`).
- **🧠 Model Zoo**:
  - ⚡ **StockLSTM**: Standard dual-layer recurrent LSTM with intermediate dropout gates.
  - 🔄 **StockBiLSTM**: Bidirectional LSTM network capturing forward momentum and retrospective trend context simultaneously.
  - 🌊 **StockCNN**: 1D Temporal Convolutional Network extracting localized multi-channel indicator features with Batch Normalization.
  - 🧬 **StockTransformerLSTM**: Hybrid architecture pairing Multi-Head Self-Attention (Sinusoidal Positional Encoding) with Recurrent LSTM decoding.
- **📊 12 Technical Indicator Channels**: Moving Averages (SMA 20/50, EMA 12), Momentum (RSI 14, MACD, MACD Signal), Volatility (Bollinger Bands, ATR 14), and Volume (OBV).
- **🎯 Rigorous Evaluation**: Out-of-sample hold-out validation tracking RMSE, MAE, MAPE, $R^2$, and Directional Accuracy.
- **🔮 Autoregressive Forward Forecast**: Multi-step recursive projections with Monte Carlo volatility confidence corridors.
- **🏆 Live Benchmark Leaderboard**: Cross-architecture comparison matrix within the dashboard.

---

## 🏗️ Production Directory Architecture

```
stock-Prediction/
├── configs/                          # Modular YAML configurations
│   ├── base.yaml                     # Shared dataset & training hyperparameters
│   ├── stocks.yaml                   # 12 target equities metadata & backfill settings
│   └── models/                       # Model-specific parameters
│       ├── lstm.yaml
│       ├── bilstm.yaml
│       ├── cnn.yaml
│       └── transformer_lstm.yaml
│
├── data/                             # Data storage (gitignored)
│   ├── raw/                          # Raw OHLCV CSV downloads
│   ├── processed/                    # Feature-engineered Parquet files
│   └── synthetic/                    # Generated synthetic records
│
├── src/                              # Core Python package
│   ├── config.py                     # YAML loader & validation dataclasses
│   ├── data/
│   │   ├── downloader.py             # Resilient yfinance multi-stock fetcher
│   │   ├── validator.py              # Data hygiene & coverage checks
│   │   ├── indicators.py             # 12-factor quantitative indicator suite
│   │   ├── synthetic.py              # Calibrated GBM & GARCH backfill generators
│   │   └── preprocessor.py           # MinMaxScaler & sliding window sequence generator
│   ├── models/
│   │   ├── base.py                   # Abstract BaseStockModel class
│   │   ├── lstm.py                   # 2-layer LSTM
│   │   ├── bilstm.py                 # Bidirectional LSTM
│   │   ├── cnn.py                    # 1D Temporal CNN
│   │   ├── transformer_lstm.py       # Transformer Encoder + LSTM Decoder
│   │   └── registry.py              # Dynamic model factory & registry
│   ├── training/
│   │   ├── trainer.py                # Unified PyTorch training loop
│   │   ├── callbacks.py              # EarlyStopping & ModelCheckpoint
│   │   └── metrics.py                # Quantitative evaluation metrics
│   ├── inference/
│   │   ├── predictor.py              # Batch & Monte Carlo uncertainty inference
│   │   └── forecaster.py             # Autoregressive forward horizon projector
│   └── utils/
│       ├── logger.py                 # Structured logging utility
│       ├── io.py                     # JSON metadata & model bundle serialization
│       └── visualization.py          # Reusable dark-mode Plotly components
│
├── scripts/
│   └── download_all_stocks.py        # CLI batch downloader & backfill runner
│
├── tests/                            # Comprehensive pytest test suite
│   ├── test_models.py
│   ├── test_metrics.py
│   └── test_synthetic.py
│
├── app.py                            # Streamlit fintech intelligence dashboard
├── train.py                          # Unified CLI training & benchmarking entry point
├── Makefile                          # Task automation commands
├── requirements.txt                  # Locked production dependencies
└── pyproject.toml                    # Modern project metadata & tool configurations
```

---

## ⚡ Quick Start

### 1. Environment Setup
```bash
# Python 3.13 Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

### 2. Ingest 25-Year Datasets
Download all 12 target stocks with automated synthetic backfill:
```bash
python scripts/download_all_stocks.py
```

### 3. Model Training & Benchmarking CLI
Train individual models or entire suites from the command line:

```bash
# Train standard LSTM on Reliance
python train.py --ticker RELIANCE.NS --model lstm --epochs 35

# Train BiLSTM on HDFC Bank
python train.py --ticker HDFCBANK.NS --model bilstm --epochs 40

# Train Transformer+LSTM on TCS
python train.py --ticker TCS.NS --model transformer_lstm --epochs 50

# Benchmark all 4 models on a stock
python train.py --ticker RELIANCE.NS --model all

# Full benchmark matrix (All 12 stocks × 4 models = 48 runs)
python train.py --ticker all --model all
```

### 4. Launch Interactive Intelligence Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 🧪 Running Tests
```bash
pytest tests/ -v
```

---

## ⚖️ Disclaimer
This project is developed for quantitative research and educational demonstration purposes only. Financial markets are subject to macroeconomic volatility and regime shifts. Predictions should not be construed as investment or trading advice.

