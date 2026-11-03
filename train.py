"""AlphaPulse v2.0 - Unified CLI Training & Benchmarking Entry Point.

Examples:
    # Train BiLSTM on Reliance
    python train.py --ticker RELIANCE.NS --model bilstm --epochs 40

    # Train Transformer+LSTM on HDFC Bank
    python train.py --ticker HDFCBANK.NS --model transformer_lstm --epochs 50

    # Train all 4 models on TCS
    python train.py --ticker TCS.NS --model all
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import get_all_stocks, get_base_config, get_model_config, get_stock_config
from src.data.downloader import fetch_stock_history, sanitize_ticker_filename
from src.data.indicators import FEATURE_COLUMNS, add_technical_indicators
from src.data.preprocessor import prepare_sequences
from src.data.synthetic import backfill_synthetic_history
from src.inference.forecaster import invert_scaled_close
from src.inference.predictor import predict_array
from src.models.registry import get_model, list_available_models
from src.training.metrics import evaluate_all
from src.training.trainer import train_model
from src.utils.io import save_model_bundle
from src.utils.logger import setup_logger

logger = setup_logger("alphapulse.train")


def train_single_run(
    ticker: str,
    model_name: str,
    lookback: int = 60,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 0.001,
    models_root: Path = Path("models"),
) -> dict[str, any]:
    """Executes a single end-to-end training and evaluation experiment."""
    stock_cfg = get_stock_config(ticker)
    clean_ticker = sanitize_ticker_filename(ticker)
    logger.info(f"\n{'='*70}\n[START] Training {model_name.upper()} on {ticker} ({stock_cfg.name})\n{'='*70}")

    # 1. Fetch / Load historical data
    processed_file = Path("data/processed") / f"{clean_ticker}.parquet"
    if processed_file.exists():
        logger.info(f"Loading existing cached dataset: {processed_file}")
        df = pd.read_parquet(processed_file)
    else:
        logger.info(f"Downloading real-time history for {ticker}...")
        raw_df = fetch_stock_history(ticker, period="max")
        if stock_cfg.needs_synthetic:
            raw_df = backfill_synthetic_history(raw_df, target_start_date=stock_cfg.start_date)
        df = add_technical_indicators(raw_df)

    # 2. Prepare Sequences
    data_bundle = prepare_sequences(df, lookback=lookback)
    x_train, y_train = data_bundle["x_train"], data_bundle["y_train"]
    x_val, y_val = data_bundle["x_val"], data_bundle["y_val"]
    x_test, y_test = data_bundle["x_test"], data_bundle["y_test"]
    scaler = data_bundle["scaler"]

    # 3. Model Config & Instantiation
    m_cfg = get_model_config(model_name).get("model", {})
    m_cfg_clean = {k: v for k, v in m_cfg.items() if k not in ["name", "architecture"]}

    model = get_model(
        name=model_name,
        n_features=len(FEATURE_COLUMNS),
        **m_cfg_clean,
    )
    logger.info(f"Instantiated {model_name.upper()} | Trainable Parameters: {model.count_parameters():,}")

    # 4. Train Model
    trained_model, history = train_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
    )

    # 5. Evaluate on Hold-Out Test Set
    pred_scaled = predict_array(trained_model, x_test)
    y_test_real = invert_scaled_close(scaler, y_test, n_features=len(FEATURE_COLUMNS))
    pred_real = invert_scaled_close(scaler, pred_scaled, n_features=len(FEATURE_COLUMNS))
    metrics = evaluate_all(y_test_real, pred_real)

    logger.info(
        f"Test Results for {ticker} [{model_name.upper()}]: "
        f"RMSE: INR {metrics['RMSE']:.2f} | MAE: INR {metrics['MAE']:.2f} | "
        f"MAPE: {metrics['MAPE']:.2f}% | R2: {metrics['R2']:.3f} | "
        f"Dir Acc: {metrics['Directional_Accuracy']:.1f}%"
    )

    # 6. Save Model Bundle
    save_path = models_root / clean_ticker / model_name.lower()
    meta = {
        "ticker": ticker,
        "company_name": stock_cfg.name,
        "sector": stock_cfg.sector,
        "model_name": model_name,
        "parameters": model.count_parameters(),
        "lookback": lookback,
        "feature_columns": FEATURE_COLUMNS,
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "metrics": metrics,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_model_bundle(save_path, trained_model, scaler, meta, history)
    logger.info(f"Model bundle saved to: {save_path}")

    return {
        "ticker": ticker,
        "model": model_name,
        **metrics,
        "parameters": model.count_parameters(),
        "epochs_trained": len(history["loss"]),
    }


def main():
    parser = argparse.ArgumentParser(description="AlphaPulse v2.0 - Model Training & Benchmarking CLI")
    parser.add_argument("--ticker", type=str, default="RELIANCE.NS", help="Stock ticker (e.g. RELIANCE.NS, or 'all')")
    parser.add_argument("--model", type=str, default="lstm", help="Model type: lstm, bilstm, cnn, transformer_lstm, or all")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--lookback", type=int, default=60, help="Lookback window length")
    args = parser.parse_args()

    # Determine tickers to run
    if args.ticker.lower() == "all":
        tickers = [s.ticker for s in get_all_stocks()]
    else:
        tickers = [args.ticker]

    # Determine models to run
    available = list_available_models()
    if args.model.lower() == "all":
        models = available
    else:
        if args.model.lower() not in available:
            raise ValueError(f"Invalid model '{args.model}'. Choose from: {available}")
        models = [args.model.lower()]

    benchmark_records = []
    for ticker in tickers:
        for m in models:
            try:
                res = train_single_run(
                    ticker=ticker,
                    model_name=m,
                    lookback=args.lookback,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                )
                benchmark_records.append(res)
            except Exception as exc:
                logger.error(f"Error training {m} on {ticker}: {exc}")

    if benchmark_records:
        bench_df = pd.DataFrame(benchmark_records)
        print("\n" + "=" * 90)
        print("ALPHAPULSE v2.0 - TRAINING BENCHMARK SUMMARY")
        print("=" * 90)
        print(bench_df.to_string(index=False))
        print("=" * 90 + "\n")

        # Save benchmark to results directory
        results_dir = Path("results/benchmarks")
        results_dir.mkdir(parents=True, exist_ok=True)
        bench_df.to_csv(results_dir / "latest_benchmark.csv", index=False)


if __name__ == "__main__":
    main()
