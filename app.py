"""AlphaPulse v2.0 - Clean Light Fintech Stock Intelligence Terminal.

Enterprise-grade quantitative intelligence dashboard supporting 12 top Indian equities,
4 deep learning architectures (LSTM, BiLSTM, 1D CNN, Transformer+LSTM),
25-year historical market depth, and vector SVG iconography.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import get_all_stocks, get_model_config, get_stock_config
from src.data.downloader import fetch_stock_history, sanitize_ticker_filename
from src.data.indicators import FEATURE_COLUMNS, add_technical_indicators
from src.data.preprocessor import prepare_sequences
from src.data.synthetic import backfill_synthetic_history
from src.inference.forecaster import forecast_next_days, invert_scaled_close
from src.inference.predictor import predict_array
from src.models.registry import get_model, list_available_models
from src.training.metrics import evaluate_all
from src.training.trainer import train_model
from src.utils.io import load_model_bundle, save_model_bundle
from src.utils.visualization import (
    plot_candlesticks_with_indicators,
    plot_forward_forecast,
    plot_loss_curves,
    plot_prediction_comparison,
)

ROOT = Path(__file__).resolve().parent
MODELS_ROOT = ROOT / "models"
DATA_ROOT = ROOT / "data"

st.set_page_config(
    page_title="AlphaPulse v2.0 | Institutional Stock Intelligence",
    page_icon="https://cdn-icons-png.flaticon.com/512/2620/2620602.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# VECTOR SVG ICON DEFINITIONS (Replacing all emojis with real vector icons)
# ==============================================================================
ICON_CHART_UP = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>"""
ICON_CHART_DOWN = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 18 13.5 8.5 8.5 13.5 1 6"></polyline><polyline points="17 18 23 18 23 12"></polyline></svg>"""
ICON_ARROW_UP = """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"></line><polyline points="5 12 12 5 19 12"></polyline></svg>"""
ICON_ARROW_DOWN = """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><polyline points="19 12 12 19 5 12"></polyline></svg>"""
ICON_CHIP = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>"""
ICON_BUILDING = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>"""
ICON_SLIDERS = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>"""
ICON_TARGET = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>"""
ICON_LIGHTNING = """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>"""
ICON_CHECK = """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>"""
ICON_ALERT = """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#d97706" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>"""
ICON_CALENDAR = """<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>"""
ICON_AWARD = """<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="7"></circle><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"></polyline></svg>"""
ICON_INFO = """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>"""

# ==============================================================================
# PREMIUM LIGHT FINTECH CSS THEME
# ==============================================================================
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

      * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      }
      code, pre, .mono {
        font-family: 'JetBrains Mono', monospace !important;
      }

      /* Base App Layout - Pure Crisp Light */
      .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
      }

      /* Global Typography */
      p, span, div, li {
        color: #334155;
      }
      h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
      }

      /* Header Bar */
      header[data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid #e2e8f0 !important;
      }
      header[data-testid="stHeader"] * {
        color: #475569 !important;
      }

      /* Form Labels */
      label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] span {
        color: #1e293b !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
      }

      /* Sidebar Design */
      [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
      }
      [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #0f172a !important;
      }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #475569 !important;
      }

      /* Form Input Elements */
      [data-testid="stSidebar"] input,
      [data-testid="stSidebar"] select,
      [data-testid="stSidebar"] [data-baseweb="input"],
      [data-testid="stSidebar"] [data-baseweb="select"] {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
      }
      [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        border: none !important;
      }

      /* Primary Action Button */
      button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.22) !important;
        transition: all 0.2s ease !important;
      }
      button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35) !important;
      }

      /* Tabs Navigation */
      [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 6px;
        background: #ffffff;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 22px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
      }
      [data-testid="stTabs"] [data-baseweb="tab"] {
        padding: 8px 18px !important;
        border-radius: 8px !important;
        color: #64748b !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        border: none !important;
        background: transparent !important;
      }
      [data-testid="stTabs"] [aria-selected="true"] {
        background: #eff6ff !important;
        color: #2563eb !important;
        font-weight: 700 !important;
        border: 1px solid #bfdbfe !important;
      }

      /* Hero Section Box */
      .hero-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 22px;
        box-shadow: 0 2px 8px -2px rgba(0, 0, 0, 0.05);
      }
      .hero-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #0f172a;
        margin: 0 0 6px 0;
      }
      .hero-subtitle {
        color: #475569 !important;
        font-size: 0.95rem;
        margin: 0;
        max-width: 860px;
        line-height: 1.55;
      }

      /* Status Pill Badges */
      .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
      }
      .badge-live {
        background: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
      }
      .badge-ai {
        background: #eff6ff;
        color: #2563eb;
        border: 1px solid #bfdbfe;
      }
      .pulse-dot {
        width: 7px;
        height: 7px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 6px #10b981;
      }

      /* KPI Metric Cards */
      .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.15s ease, border-color 0.15s ease;
      }
      .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #cbd5e1;
        box-shadow: 0 4px 8px -2px rgba(0, 0, 0, 0.06);
      }
      .kpi-label {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b !important;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
      }
      .kpi-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0f172a !important;
        letter-spacing: -0.02em;
        line-height: 1.2;
      }
      .kpi-delta-up {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        color: #16a34a !important;
        font-size: 0.84rem;
        font-weight: 700;
        margin-top: 4px;
      }
      .kpi-delta-down {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        color: #dc2626 !important;
        font-size: 0.84rem;
        font-weight: 700;
        margin-top: 4px;
      }
      .kpi-sub {
        color: #64748b !important;
        font-size: 0.8rem;
        margin-top: 4px;
      }

      /* Technical Indicator Feature Badges */
      .feature-pill {
        display: inline-block;
        padding: 5px 12px;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #334155 !important;
        margin: 3px;
      }

      /* Dataframe & Table Styling */
      [data-testid="stDataFrame"] {
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        overflow: hidden;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, ttl=3600)
def load_or_fetch_stock_data(ticker: str, period: str = "max") -> pd.DataFrame:
    """Fetches market data with fallback to processed local parquets."""
    stock_cfg = get_stock_config(ticker)
    clean_name = sanitize_ticker_filename(ticker)

    processed_file = DATA_ROOT / "processed" / f"{clean_name}.parquet"
    if processed_file.exists():
        return pd.read_parquet(processed_file)

    raw_df = fetch_stock_history(ticker, period=period)
    if stock_cfg.needs_synthetic:
        raw_df = backfill_synthetic_history(raw_df, target_start_date=stock_cfg.start_date)

    featured_df = add_technical_indicators(raw_df)
    (DATA_ROOT / "processed").mkdir(parents=True, exist_ok=True)
    featured_df.to_parquet(processed_file)
    return featured_df


def model_bundle_exists(ticker: str, model_name: str) -> bool:
    clean_ticker = sanitize_ticker_filename(ticker)
    target_dir = MODELS_ROOT / clean_ticker / model_name.lower()
    return (target_dir / "model.pt").exists() and (target_dir / "meta.json").exists()


# ------------------------------------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px; padding-bottom: 14px; border-bottom: 1px solid #e2e8f0;">
          <div style="width: 36px; height: 36px; border-radius: 8px; background: #eff6ff; border: 1px solid #bfdbfe; color: #2563eb; display: flex; align-items: center; justify-content: center;">
            {ICON_CHART_UP}
          </div>
          <div>
            <div style="font-weight: 800; font-size: 1.12rem; color: #0f172a; line-height: 1.2;">AlphaPulse v2.0</div>
            <div style="font-size: 0.72rem; color: #2563eb; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">Quantitative Deep Learning</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"#### {ICON_BUILDING} &nbsp;Target Equity Asset", unsafe_allow_html=True)
    available_stocks = get_all_stocks()
    stock_options = [f"{s.ticker} — {s.name}" for s in available_stocks] + ["Custom Symbol..."]

    selected_stock_label = st.selectbox("Select Stock", stock_options, index=0)
    if selected_stock_label == "Custom Symbol...":
        selected_ticker = st.text_input("Enter NSE/BSE Symbol (e.g. INFY.NS)", value="RELIANCE.NS").strip().upper()
    else:
        selected_ticker = selected_stock_label.split(" — ")[0]

    current_stock_cfg = get_stock_config(selected_ticker)
    st.caption(f"Sector: **{current_stock_cfg.sector}** | Target Horizon: **25 Years**")

    st.markdown("---")
    st.markdown(f"#### {ICON_CHIP} &nbsp;Neural Architecture", unsafe_allow_html=True)
    model_choices = {
        "lstm": "LSTM (Dual-Layer Recurrent)",
        "bilstm": "BiLSTM (Bidirectional Context)",
        "cnn": "1D CNN (Temporal Feature Extractor)",
        "transformer_lstm": "Transformer + LSTM (Attention Hybrid)",
    }
    selected_model_type = st.selectbox(
        "Model Architecture",
        options=list(model_choices.keys()),
        format_func=lambda x: model_choices[x],
        index=0,
    )

    st.markdown("---")
    st.markdown(f"#### {ICON_SLIDERS} &nbsp;Hyperparameter Tuning", unsafe_allow_html=True)
    lookback = st.slider("Lookback Window (Days)", 20, 120, 60, 5)
    epochs = st.slider("Training Epochs", 5, 80, 35, 5)
    batch_size = st.select_slider("Mini-Batch Size", options=[16, 32, 64], value=32)
    learning_rate = st.select_slider(
        "Learning Rate",
        options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        value=1e-3,
        format_func=lambda x: f"{x:g}",
    )

    st.markdown("---")
    st.markdown(f"#### {ICON_TARGET} &nbsp;Inference Horizon", unsafe_allow_html=True)
    forecast_days = st.slider("Forward Prediction Horizon (Days)", 1, 30, 7)

    train_btn_clicked = st.button("Retrain Neural Model", type="primary", width="stretch")

    clean_ticker_str = sanitize_ticker_filename(selected_ticker)
    model_ready = model_bundle_exists(selected_ticker, selected_model_type)

    if model_ready:
        st.markdown(
            f"""
            <div style="margin-top: 14px; padding: 10px 14px; border-radius: 8px; background: #ecfdf5; border: 1px solid #a7f3d0; display: flex; align-items: center; gap: 8px;">
              {ICON_CHECK}
              <span style="font-size: 0.82rem; color: #059669; font-weight: 700;">{selected_model_type.upper()} Model Active</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="margin-top: 14px; padding: 10px 14px; border-radius: 8px; background: #fffbeb; border: 1px solid #fde68a; display: flex; align-items: center; gap: 8px;">
              {ICON_ALERT}
              <span style="color: #b45309; font-size: 0.82rem; font-weight: 700;">{selected_model_type.upper()} Not Trained Yet</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ------------------------------------------------------------------------------
# DATA INGESTION & INDICATORS
# ------------------------------------------------------------------------------
try:
    with st.spinner(f"Ingesting trading records for {selected_ticker}..."):
        featured_df = load_or_fetch_stock_data(selected_ticker)
except Exception as exc:
    st.error(f"Failed to load dataset for {selected_ticker}: {exc}")
    st.stop()

latest_session = featured_df.iloc[-1]
prev_session = featured_df.iloc[-2]
price_delta = float(latest_session["Close"] - prev_session["Close"])
price_delta_pct = (price_delta / float(prev_session["Close"])) * 100
is_bullish = price_delta >= 0

synth_count = int(featured_df["is_synthetic"].sum()) if "is_synthetic" in featured_df.columns else 0
synth_ratio = (synth_count / len(featured_df)) * 100

# ------------------------------------------------------------------------------
# HERO BANNER (CLEAN LIGHT)
# ------------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-box">
      <div style="display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap;">
        <span class="badge badge-live"><span class="pulse-dot"></span> NSE Market Live</span>
        <span class="badge badge-ai">Model: {selected_model_type.upper()}</span>
        <span class="badge badge-ai">Depth: {len(featured_df):,} Sessions (~25y)</span>
        {f'<span class="badge" style="background: #fffbeb; color: #b45309; border: 1px solid #fde68a;">Synthetic Backfill: {synth_count:,} Sessions ({synth_ratio:.1f}%)</span>' if synth_count > 0 else '<span class="badge" style="background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0;">100% Real Historical Records</span>'}
      </div>
      <h1 class="hero-title">{selected_ticker} — {current_stock_cfg.name}</h1>
      <p class="hero-subtitle">
        Enterprise-grade temporal sequence forecasting using {model_choices[selected_model_type]},
        integrating 12 technical momentum, volatility, and trend channels with adaptive optimization.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# TOP METRICS ROW
# ------------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    delta_class = "kpi-delta-up" if is_bullish else "kpi-delta-down"
    delta_icon = ICON_ARROW_UP if is_bullish else ICON_ARROW_DOWN
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{ICON_CHART_UP} Latest Market Close</div>
          <div class="kpi-value">₹{latest_session['Close']:,.2f}</div>
          <div class="{delta_class}">{delta_icon} {price_delta:+.2f} ({price_delta_pct:+.2f}%)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi2:
    rsi_val = float(latest_session["rsi_14"])
    rsi_badge = "<span style='color: #dc2626; font-weight: 700;'>Overbought (>70)</span>" if rsi_val > 70 else (
        "<span style='color: #16a34a; font-weight: 700;'>Oversold (<30)</span>" if rsi_val < 30 else
        "<span style='color: #64748b; font-weight: 700;'>Neutral Momentum</span>"
    )
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{ICON_TARGET} RSI (14-Session)</div>
          <div class="kpi-value">{rsi_val:.1f}</div>
          <div class="kpi-sub">{rsi_badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi3:
    trend_state = "Bullish Alignment" if latest_session["sma_20"] > latest_session["sma_50"] else "Bearish Alignment"
    trend_color = "#16a34a" if latest_session["sma_20"] > latest_session["sma_50"] else "#dc2626"
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{ICON_CHART_UP} SMA 20 vs SMA 50</div>
          <div class="kpi-value">₹{latest_session['sma_20']:,.2f}</div>
          <div class="kpi-sub" style="color: {trend_color}; font-weight: 700;">{trend_state}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi4:
    bb_dist = ((latest_session["Close"] - latest_session["bb_low"]) / (latest_session["bb_high"] - latest_session["bb_low"])) * 100
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{ICON_SLIDERS} Bollinger %B</div>
          <div class="kpi-value">{bb_dist:.1f}%</div>
          <div class="kpi-sub">₹{latest_session['bb_low']:,.0f} — ₹{latest_session['bb_high']:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi5:
    vol_m = latest_session["Volume"] / 1e6
    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{ICON_BUILDING} Trading Volume</div>
          <div class="kpi-value">{vol_m:.2f}M</div>
          <div class="kpi-sub">{latest_session['Volume']:,.0f} shares</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# DATA PREPARATION & TRAINING HANDLER
# ------------------------------------------------------------------------------
data_bundle = prepare_sequences(featured_df, lookback=lookback)
x_train, y_train = data_bundle["x_train"], data_bundle["y_train"]
x_val, y_val = data_bundle["x_val"], data_bundle["y_val"]
x_test, y_test = data_bundle["x_test"], data_bundle["y_test"]

model_save_dir = MODELS_ROOT / clean_ticker_str / selected_model_type.lower()

if train_btn_clicked:
    with st.status(
        f"Training {selected_model_type.upper()} on {len(x_train):,} sequences...",
        expanded=True,
    ) as status:
        st.write(f"Configuring {selected_model_type.upper()} architecture...")
        m_cfg = get_model_config(selected_model_type).get("model", {})
        m_cfg_clean = {k: v for k, v in m_cfg.items() if k not in ["name", "architecture"]}

        model_inst = get_model(
            name=selected_model_type,
            n_features=len(FEATURE_COLUMNS),
            **m_cfg_clean,
        )

        st.write("Running Adam optimization with ReduceLROnPlateau and EarlyStopping...")
        trained_model, history = train_model(
            model=model_inst,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        st.write("Evaluating metrics on hold-out test set...")
        pred_scaled = predict_array(trained_model, x_test)
        y_test_real = invert_scaled_close(data_bundle["scaler"], y_test, n_features=len(FEATURE_COLUMNS))
        pred_real = invert_scaled_close(data_bundle["scaler"], pred_scaled, n_features=len(FEATURE_COLUMNS))
        metrics = evaluate_all(y_test_real, pred_real)

        st.write("Saving model weights and scaling parameters...")
        meta = {
            "ticker": selected_ticker,
            "company_name": current_stock_cfg.name,
            "sector": current_stock_cfg.sector,
            "model_name": selected_model_type,
            "parameters": trained_model.count_parameters(),
            "lookback": lookback,
            "feature_columns": FEATURE_COLUMNS,
            "train_samples": len(x_train),
            "test_samples": len(x_test),
            "metrics": metrics,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_model_bundle(model_save_dir, trained_model, data_bundle["scaler"], meta, history)
        st.session_state["history"] = history
        status.update(label=f"Training complete! {selected_model_type.upper()} saved successfully.", state="complete")
    st.success(f"{selected_model_type.upper()} weights synchronized for {selected_ticker}.")

# Load active model bundle if present
active_bundle = None
if model_save_dir.exists() and (model_save_dir / "model.pt").exists():
    try:
        active_bundle = load_model_bundle(model_save_dir)
    except Exception as exc:
        st.error(f"Error loading {selected_model_type.upper()} bundle: {exc}")

# ------------------------------------------------------------------------------
# TABS INTERFACE (CLEAN VECTOR ICONS)
# ------------------------------------------------------------------------------
tab_market, tab_train, tab_test, tab_forecast, tab_benchmark = st.tabs(
    [
        "Market Terminal",
        "Neural Training",
        "Hold-Out Validation",
        "Forward Horizon Forecast",
        "Model Zoo Benchmark",
    ]
)

# TAB 1: MARKET TERMINAL
with tab_market:
    chart_col, control_col = st.columns([4, 1])
    with chart_col:
        st.markdown("<h3 style='margin-bottom: 2px;'>Candlestick Price Action & Technical Overlays</h3>", unsafe_allow_html=True)
        st.caption("Interactive Candlesticks with SMA 20, SMA 50, Bollinger Bands and Volume Dynamics")
    with control_col:
        show_overlays = st.checkbox("Technical Overlays", value=True)
        candle_count = st.selectbox("Show Sessions", [100, 250, 500, "All (~25y)"], index=1)

    slice_df = featured_df if candle_count == "All (~25y)" else featured_df.tail(int(candle_count))
    st.plotly_chart(
        plot_candlesticks_with_indicators(slice_df, show_overlays=show_overlays),
        width="stretch",
    )

    col_recent, col_features = st.columns([1.2, 1])
    with col_recent:
        st.markdown("### Recent Trading Sessions")
        recent_view = featured_df[["Open", "High", "Low", "Close", "Volume", "rsi_14", "sma_20"]].tail(10)
        st.dataframe(
            recent_view.style.format(
                {
                    "Open": "₹{:.2f}",
                    "High": "₹{:.2f}",
                    "Low": "₹{:.2f}",
                    "Close": "₹{:.2f}",
                    "Volume": "{:,.0f}",
                    "rsi_14": "{:.1f}",
                    "sma_20": "₹{:.2f}",
                }
            ),
            width="stretch",
            height=340,
        )

    with col_features:
        st.markdown("### Multi-Factor Feature Matrix")
        st.markdown(
            """
            <div style="color: #475569; font-size: 0.9rem; line-height: 1.5; margin-bottom: 12px;">
              The engine processes <b>12 multidimensional feature channels</b> simultaneously to model non-linear temporal dynamics:
            </div>
            """,
            unsafe_allow_html=True,
        )
        pills_html = "".join([f"<span class='feature-pill'>{feat}</span>" for feat in FEATURE_COLUMNS])
        st.markdown(f"<div>{pills_html}</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="margin-top: 18px; padding: 14px 18px; border-radius: 10px; background: #ffffff; border: 1px solid #e2e8f0; font-size: 0.88rem; color: #334155;">
              ⚡ <b>Dataset Scale:</b> <span style="color: #2563eb; font-weight: 700;">{len(featured_df):,}</span> total trading sessions processed with MinMaxScaler standardization.
            </div>
            """,
            unsafe_allow_html=True,
        )

# TAB 2: NEURAL NETWORK & TRAINING
with tab_train:
    st.markdown(f"### {selected_model_type.upper()} Deep Learning Architecture")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">{ICON_CHIP} Training Sequences</div>
              <div class="kpi-value">{len(x_train):,}</div>
              <div class="kpi-sub">Chronological Train Split</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">{ICON_TARGET} Validation Sequences</div>
              <div class="kpi-value">{len(x_val):,}</div>
              <div class="kpi-sub">Early Stopping Monitor</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t3:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">{ICON_BUILDING} Test Sequences</div>
              <div class="kpi-value">{len(x_test):,}</div>
              <div class="kpi-sub">Hold-Out Evaluation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t4:
        st.markdown(
            f"""
            <div class="kpi-card">
              <div class="kpi-label">{ICON_CALENDAR} Sequence Lookback</div>
              <div class="kpi-value">{lookback} <span style="font-size: 1rem; color: #2563eb;">Days</span></div>
              <div class="kpi-sub">{len(FEATURE_COLUMNS)} Input Channels</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    if "history" in st.session_state:
        st.plotly_chart(
            plot_loss_curves(st.session_state["history"], title=f"{selected_model_type.upper()} Loss Convergence Curve"),
            width="stretch",
        )
    elif active_bundle and active_bundle[3] is not None:
        st.plotly_chart(
            plot_loss_curves(active_bundle[3], title=f"{selected_model_type.upper()} Saved Training History"),
            width="stretch",
        )
    else:
        st.info(f"No active training history found for {selected_model_type.upper()}. Click 'Retrain Neural Model' in the sidebar.")

# TAB 3: HOLD-OUT TEST PREDICTIONS
with tab_test:
    if active_bundle is None:
        st.warning(f"No trained {selected_model_type.upper()} weights found for {selected_ticker}. Please retrain from the sidebar.")
    else:
        meta, state_dict, scaler, _ = active_bundle
        m_cfg = get_model_config(selected_model_type).get("model", {})
        m_cfg_clean = {k: v for k, v in m_cfg.items() if k not in ["name", "architecture"]}

        model_eval = get_model(name=selected_model_type, n_features=len(FEATURE_COLUMNS), **m_cfg_clean)
        model_eval.load_state_dict(state_dict)

        pred_scaled = predict_array(model_eval, x_test)
        actual = invert_scaled_close(scaler, y_test, n_features=len(FEATURE_COLUMNS))
        predicted = invert_scaled_close(scaler, pred_scaled, n_features=len(FEATURE_COLUMNS))
        metrics = evaluate_all(actual, predicted)

        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">{ICON_TARGET} Root Mean Sq Error</div>
                  <div class="kpi-value">₹{metrics['RMSE']:.2f}</div>
                  <div class="kpi-sub">Standard Deviation Error</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">{ICON_SLIDERS} Mean Absolute Error</div>
                  <div class="kpi-value">₹{metrics['MAE']:.2f}</div>
                  <div class="kpi-sub">Average Price Drift</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            mape_badge = "<span style='color: #16a34a; font-weight: 700;'>Elite Precision</span>" if metrics['MAPE'] < 3.0 else "Good Precision"
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">{ICON_CHECK} Mean Abs % Error</div>
                  <div class="kpi-value">{metrics['MAPE']:.2f}%</div>
                  <div class="kpi-sub">{mape_badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m4:
            r2_badge = "<span style='color: #2563eb; font-weight: 700;'>High Correlation</span>" if metrics['R2'] > 0.85 else "Moderate Fit"
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">{ICON_AWARD} R² Fit Score</div>
                  <div class="kpi-value">{metrics['R2']:.3f}</div>
                  <div class="kpi-sub">{r2_badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m5:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">{ICON_CHART_UP} Directional Accuracy</div>
                  <div class="kpi-value">{metrics['Directional_Accuracy']:.1f}%</div>
                  <div class="kpi-sub">Sign Concordance</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown("#### Quantitative Financial Performance Indexes")
        f1, f2, f3, f4, f5 = st.columns(5)
        with f1:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">{ICON_CHART_UP} Sharpe Ratio</div>
                  <div class="kpi-value">{metrics.get('Sharpe_Ratio', 0.0):.2f}</div>
                  <div class="kpi-sub">Annualized Excess Index</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with f2:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">{ICON_TARGET} Sortino Ratio</div>
                  <div class="kpi-value">{metrics.get('Sortino_Ratio', 0.0):.2f}</div>
                  <div class="kpi-sub">Downside Volatility Index</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with f3:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">{ICON_ALERT} Max Drawdown</div>
                  <div class="kpi-value">{metrics.get('Max_Drawdown', 0.0):.1f}%</div>
                  <div class="kpi-sub">Peak-to-Trough Risk</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with f4:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">{ICON_AWARD} Trading Win Rate</div>
                  <div class="kpi-value">{metrics.get('Win_Rate', 0.0):.1f}%</div>
                  <div class="kpi-sub">Profitable Sessions Index</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with f5:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-label">{ICON_BUILDING} Profit Factor</div>
                  <div class="kpi-value">{metrics.get('Profit_Factor', 1.0):.2f}</div>
                  <div class="kpi-sub">Gross Gains / Losses</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        st.plotly_chart(
            plot_prediction_comparison(data_bundle["test_dates"], actual, predicted, model_name=selected_model_type),
            width="stretch",
        )

        st.markdown("### Granular Prediction Journal")
        journal_df = pd.DataFrame(
            {
                "Actual Close": actual,
                "Predicted Close": predicted,
                "Abs Error (INR)": np.abs(predicted - actual),
                "Error (%)": (np.abs(predicted - actual) / actual) * 100,
            },
            index=data_bundle["test_dates"],
        )
        st.dataframe(
            journal_df.tail(15).style.format(
                {
                    "Actual Close": "₹{:.2f}",
                    "Predicted Close": "₹{:.2f}",
                    "Abs Error (INR)": "₹{:.2f}",
                    "Error (%)": "{:.2f}%",
                }
            ),
            width="stretch",
        )

# TAB 4: FORWARD FORECAST
with tab_forecast:
    if active_bundle is None:
        st.warning(f"No trained {selected_model_type.upper()} weights found for {selected_ticker}. Please retrain from the sidebar.")
    else:
        meta, state_dict, scaler, _ = active_bundle
        m_cfg = get_model_config(selected_model_type).get("model", {})
        m_cfg_clean = {k: v for k, v in m_cfg.items() if k not in ["name", "architecture"]}

        model_fc = get_model(name=selected_model_type, n_features=len(FEATURE_COLUMNS), **m_cfg_clean)
        model_fc.load_state_dict(state_dict)

        last_window = data_bundle["scaled"][-lookback:]
        future_prices = forecast_next_days(
            model_fc,
            last_window,
            scaler,
            days=forecast_days,
            n_features=len(FEATURE_COLUMNS),
        )

        last_date = featured_df.index[-1]
        future_dates = pd.bdate_range(last_date + timedelta(days=1), periods=forecast_days)

        current_close = float(featured_df["Close"].iloc[-1])
        next_day_target = float(future_prices[0])
        next_day_delta = next_day_target - current_close
        next_day_delta_pct = (next_day_delta / current_close) * 100

        terminal_target = float(future_prices[-1])
        total_delta = terminal_target - current_close
        total_delta_pct = (total_delta / current_close) * 100

        sig_color = "#16a34a" if total_delta >= 0 else "#dc2626"
        sig_dir = "BULLISH" if total_delta >= 0 else "BEARISH"
        sig_icon = ICON_ARROW_UP if total_delta >= 0 else ICON_ARROW_DOWN

        st.markdown(
            f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 22px 26px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
              <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
                <div>
                  <div style="color: #64748b; font-size: 0.8rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Forward Forecast Trajectory ({selected_model_type.upper()})</div>
                  <div style="font-size: 1.75rem; font-weight: 800; color: {sig_color}; margin-top: 4px; display: flex; align-items: center; gap: 8px;">
                    {sig_icon} {sig_dir} ({total_delta_pct:+.2f}%)
                  </div>
                  <div style="color: #475569; font-size: 0.92rem; margin-top: 2px;">
                    Projected terminal target price of <b>₹{terminal_target:,.2f}</b> by {future_dates[-1].strftime('%d %b %Y')}.
                  </div>
                </div>
                <div style="display: flex; gap: 26px; border-left: 1px solid #e2e8f0; padding-left: 26px;">
                  <div>
                    <div style="color: #64748b; font-size: 0.78rem; text-transform: uppercase; font-weight: 700;">Next Session Target</div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #0f172a; margin-top: 2px;">₹{next_day_target:,.2f}</div>
                    <div style="font-size: 0.84rem; font-weight: 700; color: {'#16a34a' if next_day_delta >= 0 else '#dc2626'};">
                      {next_day_delta:+.2f} ({next_day_delta_pct:+.2f}%)
                    </div>
                  </div>
                  <div>
                    <div style="color: #64748b; font-size: 0.78rem; text-transform: uppercase; font-weight: 700;">Base Market Close</div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #0f172a; margin-top: 2px;">₹{current_close:,.2f}</div>
                    <div style="font-size: 0.82rem; color: #64748b;">Baseline Reference</div>
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            plot_forward_forecast(
                featured_df["Close"].tail(90),
                future_dates,
                future_prices,
                model_name=selected_model_type,
            ),
            width="stretch",
        )

        st.markdown("### Day-by-Day Forecast Schedule")
        day_names = [d.strftime("%A") for d in future_dates]
        schedule_df = pd.DataFrame(
            {
                "Day": day_names,
                "Forecast Close": future_prices,
                "Expected Delta (INR)": [p - current_close for p in future_prices],
                "Expected Delta (%)": [(p - current_close) / current_close * 100 for p in future_prices],
            },
            index=[d.strftime("%Y-%m-%d") for d in future_dates],
        )
        st.dataframe(
            schedule_df.style.format(
                {
                    "Forecast Close": "₹{:.2f}",
                    "Expected Delta (INR)": "₹{:+.2f}",
                    "Expected Delta (%)": "{:+.2f}%",
                }
            ),
            width="stretch",
        )

# TAB 5: MODEL ZOO BENCHMARK
with tab_benchmark:
    st.markdown("### Model Architecture Benchmark Leaderboard")
    st.caption(f"Cross-architecture hold-out validation comparison on **{selected_ticker}**")

    records = []
    models_dir_stock = MODELS_ROOT / clean_ticker_str
    if models_dir_stock.exists():
        for m_name in list_available_models():
            m_path = models_dir_stock / m_name
            if (m_path / "meta.json").exists():
                try:
                    loaded_meta, _, _, _ = load_model_bundle(m_path)
                    metrics_d = loaded_meta.get("metrics", {})
                    records.append(
                        {
                            "Model": m_name.upper(),
                            "RMSE (₹)": metrics_d.get("RMSE", 0.0),
                            "MAE (₹)": metrics_d.get("MAE", 0.0),
                            "MAPE (%)": metrics_d.get("MAPE", 0.0),
                            "R² Score": metrics_d.get("R2", 0.0),
                            "Dir Accuracy (%)": metrics_d.get("Directional_Accuracy", 0.0),
                            "Sharpe Ratio": metrics_d.get("Sharpe_Ratio", 0.0),
                            "Max DD (%)": metrics_d.get("Max_Drawdown", 0.0),
                            "Win Rate (%)": metrics_d.get("Win_Rate", 0.0),
                            "Profit Factor": metrics_d.get("Profit_Factor", 1.0),
                            "Parameters": f"{loaded_meta.get('parameters', 0):,}",
                            "Trained At": loaded_meta.get("trained_at", "N/A"),
                        }
                    )
                except Exception:
                    pass

    if records:
        b_df = pd.DataFrame(records).sort_values(by="RMSE (₹)", ascending=True)
        st.dataframe(
            b_df.style.format(
                {
                    "RMSE (₹)": "₹{:.2f}",
                    "MAE (₹)": "₹{:.2f}",
                    "MAPE (%)": "{:.2f}%",
                    "R² Score": "{:.3f}",
                    "Dir Accuracy (%)": "{:.1f}%",
                    "Sharpe Ratio": "{:.2f}",
                    "Max DD (%)": "{:.1f}%",
                    "Win Rate (%)": "{:.1f}%",
                    "Profit Factor": "{:.2f}",
                }
            ),
            width="stretch",
        )

        best_m = b_df.iloc[0]["Model"]
        st.markdown(
            f"""
            <div style="margin-top: 14px; padding: 14px 18px; border-radius: 10px; background: #ecfdf5; border: 1px solid #a7f3d0; color: #065f46; display: flex; align-items: center; gap: 8px;">
              {ICON_AWARD}
              <span><b>Top Architecture:</b> <b>{best_m}</b> currently achieves lowest RMSE on hold-out validation.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info(f"No trained models found for {selected_ticker} yet. Train each model architecture using the sidebar to populate the comparative leaderboard.")
