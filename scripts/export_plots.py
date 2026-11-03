"""AlphaPulse v2.0 - Automated Analytical Plot Exporter.

Generates and saves standalone interactive Plotly HTML figures to `results/plots/`.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import get_stock_config
from src.data.downloader import fetch_stock_history, sanitize_ticker_filename
from src.data.indicators import FEATURE_COLUMNS, add_technical_indicators
from src.data.preprocessor import prepare_sequences
from src.data.synthetic import backfill_synthetic_history
from src.inference.forecaster import forecast_next_days, invert_scaled_close
from src.inference.predictor import predict_array
from src.models.registry import get_model
from src.utils.io import load_model_bundle
from src.utils.logger import setup_logger
from src.utils.visualization import (
    plot_candlesticks_with_indicators,
    plot_forward_forecast,
    plot_loss_curves,
    plot_prediction_comparison,
)

logger = setup_logger("scripts.export_plots")


def export_all_plots(ticker: str = "RELIANCE.NS", model_type: str = "lstm", days_forecast: int = 7):
    """Exports the core visual analytical plots to results/plots/."""
    clean_ticker = sanitize_ticker_filename(ticker)
    output_dir = PROJECT_ROOT / "results" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating result plots for {ticker} [{model_type.upper()}]...")

    # 1. Fetch & preprocess data
    raw_df = fetch_stock_history(ticker)
    stock_cfg = get_stock_config(ticker)
    if stock_cfg.needs_synthetic:
        raw_df = backfill_synthetic_history(raw_df, target_start_date=stock_cfg.start_date)
    featured_df = add_technical_indicators(raw_df)

    # Plot 1: Financial Candlesticks & Technical Overlays
    candle_fig = plot_candlesticks_with_indicators(featured_df.iloc[-250:], title=f"{ticker} - 1-Year Market Price Action")
    candle_path = output_dir / f"{clean_ticker}_candlesticks.html"
    candle_fig.write_html(candle_path)
    logger.info(f"Saved: {candle_path}")

    # Prepare sequences
    seq_data = prepare_sequences(featured_df, lookback=60)
    scaler = seq_data["scaler"]
    x_test, y_test = seq_data["x_test"], seq_data["y_test"]
    test_dates = seq_data["test_dates"]

    # Check for trained model bundle
    bundle_dir = PROJECT_ROOT / "models" / clean_ticker / model_type.lower()
    if not (bundle_dir / "model.pt").exists():
        logger.warning(f"No trained model bundle found at {bundle_dir}. Please run 'python train.py --ticker {ticker} --model {model_type}' first to generate model result plots.")
        return

    meta, state_dict, scaler, history = load_model_bundle(bundle_dir)
    m_cfg = get_stock_config(ticker)
    from src.config import get_model_config
    m_dict = get_model_config(model_type).get("model", {})
    m_clean = {k: v for k, v in m_dict.items() if k not in ["name", "architecture"]}
    model = get_model(name=model_type, n_features=len(FEATURE_COLUMNS), **m_clean)
    model.load_state_dict(state_dict)

    # Plot 2: Test Set Prediction Comparison
    preds_scaled = predict_array(model, x_test)
    preds_inr = invert_scaled_close(scaler, preds_scaled)
    actual_inr = invert_scaled_close(scaler, y_test)

    pred_fig = plot_prediction_comparison(test_dates, actual_inr, preds_inr, model_name=model_type)
    pred_path = output_dir / f"{clean_ticker}_{model_type}_predictions.html"
    pred_fig.write_html(pred_path)
    logger.info(f"Saved: {pred_path}")

    # Plot 3: Loss Convergence Curves
    if history and "loss" in history:
        loss_fig = plot_loss_curves(history, title=f"{ticker} [{model_type.upper()}] - Training Loss Convergence")
        loss_path = output_dir / f"{clean_ticker}_{model_type}_loss.html"
        loss_fig.write_html(loss_path)
        logger.info(f"Saved: {loss_path}")

    # Plot 4: Autoregressive Forward Forecast
    last_window = seq_data["x_test"][-1]
    future_preds = forecast_next_days(model, last_window, scaler, days=days_forecast)
    last_date = test_dates[-1]
    future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=days_forecast)

    hist_tail = pd.Series(actual_inr[-60:], index=test_dates[-60:])
    forecast_fig = plot_forward_forecast(hist_tail, future_dates, future_preds, model_name=model_type)
    forecast_path = output_dir / f"{clean_ticker}_{model_type}_forecast.html"
    forecast_fig.write_html(forecast_path)
    logger.info(f"Saved: {forecast_path}")

    # Plot 5: Residual Error Distribution
    residuals = actual_inr - preds_inr
    resid_fig = go.Figure()
    resid_fig.add_trace(go.Histogram(x=residuals, nbinsx=40, marker_color="#2563eb", opacity=0.8, name="Residuals"))
    resid_fig.update_layout(
        title=dict(text=f"{ticker} [{model_type.upper()}] - Prediction Residual Distribution", font=dict(color="#0f172a", size=15)),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        height=380,
        xaxis=dict(title="Residual Error (INR = Actual - Predicted)", gridcolor="#f1f5f9"),
        yaxis=dict(title="Frequency", gridcolor="#f1f5f9"),
    )
    resid_path = output_dir / f"{clean_ticker}_{model_type}_residuals.html"
    resid_fig.write_html(resid_path)
    logger.info(f"Saved: {resid_path}")

    print(f"\nAll 5 analytical result plots saved successfully to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export interactive HTML result plots.")
    parser.add_argument("--ticker", type=str, default="RELIANCE.NS", help="Stock ticker symbol")
    parser.add_argument("--model", type=str, default="lstm", help="Model architecture (lstm, bilstm, cnn, transformer_lstm)")
    parser.add_argument("--days", type=int, default=14, help="Forecast horizon days")
    args = parser.parse_args()

    export_all_plots(ticker=args.ticker, model_type=args.model, days_forecast=args.days)
