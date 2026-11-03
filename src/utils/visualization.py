"""AlphaPulse v2.0 - Clean Light Fintech Visualization Components."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_candlesticks_with_indicators(
    df: pd.DataFrame,
    title: str = "Market Price Action",
    show_overlays: bool = True,
) -> go.Figure:
    """Creates a crisp light financial candlestick chart with technical indicator overlays."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # 1. Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color="#16a34a",
            increasing_fillcolor="#16a34a",
            decreasing_line_color="#ef4444",
            decreasing_fillcolor="#ef4444",
            whiskerwidth=0.6,
        ),
        row=1,
        col=1,
    )

    if show_overlays:
        if "sma_20" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["sma_20"],
                    name="SMA 20",
                    line=dict(color="#2563eb", width=1.75),
                ),
                row=1,
                col=1,
            )
        if "sma_50" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["sma_50"],
                    name="SMA 50",
                    line=dict(color="#ea580c", width=1.75),
                ),
                row=1,
                col=1,
            )
        if "bb_high" in df.columns and "bb_low" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["bb_high"],
                    name="Bollinger High",
                    line=dict(color="rgba(124, 58, 237, 0.4)", width=1, dash="dot"),
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["bb_low"],
                    name="Bollinger Low",
                    line=dict(color="rgba(124, 58, 237, 0.4)", width=1, dash="dot"),
                    fill="tonexty",
                    fillcolor="rgba(124, 58, 237, 0.06)",
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )

    # 2. Volume Bars
    vol_colors = [
        "#16a34a" if c >= o else "#ef4444"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="Volume",
            marker_color=vol_colors,
            opacity=0.65,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        height=560,
        font=dict(color="#1e293b", family="Plus Jakarta Sans, sans-serif"),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#f1f5f9", linecolor="#cbd5e1")
    fig.update_yaxes(gridcolor="#f1f5f9", linecolor="#cbd5e1")
    return fig


def plot_prediction_comparison(
    dates: pd.Index | np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str = "Model",
) -> go.Figure:
    """Plots actual vs predicted hold-out prices with shaded residual error band."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=actual,
            name="Actual Close",
            line=dict(color="#2563eb", width=2),
            mode="lines",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=predicted,
            name=f"{model_name.upper()} Prediction",
            line=dict(color="#d97706", width=2),
            mode="lines",
        )
    )

    # Shaded error band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([dates, dates[::-1]]),
            y=np.concatenate([predicted, actual[::-1]]),
            fill="toself",
            fillcolor="rgba(217, 119, 6, 0.08)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        height=460,
        font=dict(color="#1e293b", family="Plus Jakarta Sans, sans-serif"),
        yaxis=dict(title="Price (INR)", gridcolor="#f1f5f9", linecolor="#cbd5e1"),
        xaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1"),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="#e2e8f0",
            borderwidth=1,
        ),
        hovermode="x unified",
    )
    return fig


def plot_loss_curves(history: dict[str, list[float]], title: str = "Loss Convergence") -> go.Figure:
    """Plots train vs validation loss convergence across training epochs."""
    fig = go.Figure()
    epochs = list(range(1, len(history["loss"]) + 1))

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history["loss"],
            name="Train Loss (MSE)",
            line=dict(color="#2563eb", width=2.5),
        )
    )
    if "val_loss" in history and history["val_loss"]:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=history["val_loss"],
                name="Validation Loss (MSE)",
                line=dict(color="#ea580c", width=2.5, dash="dash"),
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#0f172a", size=15)),
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        height=380,
        font=dict(color="#1e293b", family="Plus Jakarta Sans, sans-serif"),
        xaxis=dict(title="Epoch", gridcolor="#f1f5f9", linecolor="#cbd5e1"),
        yaxis=dict(title="MSE Loss (Normalized)", gridcolor="#f1f5f9", linecolor="#cbd5e1"),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )
    return fig


def plot_forward_forecast(
    historical_tail: pd.Series,
    future_dates: pd.DatetimeIndex,
    future_prices: np.ndarray,
    model_name: str = "Model",
) -> go.Figure:
    """Plots historical trend connected to multi-day forecast trajectory with confidence corridor."""
    fig = go.Figure()

    # Historical
    fig.add_trace(
        go.Scatter(
            x=historical_tail.index,
            y=historical_tail.values,
            name="Historical Close",
            line=dict(color="#2563eb", width=2),
        )
    )

    # Connector
    conn_dates = [historical_tail.index[-1]] + list(future_dates)
    conn_prices = [historical_tail.values[-1]] + list(future_prices)

    # Confidence corridor estimation
    base_price = historical_tail.values[-1]
    corridor_growth = np.array([np.sqrt(i + 1) * 0.012 * base_price for i in range(len(conn_prices))])
    upper_bound = conn_prices + corridor_growth
    lower_bound = conn_prices - corridor_growth

    fig.add_trace(
        go.Scatter(
            x=conn_dates + conn_dates[::-1],
            y=list(upper_bound) + list(lower_bound[::-1]),
            fill="toself",
            fillcolor="rgba(22, 163, 74, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="Confidence Corridor",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=conn_dates,
            y=conn_prices,
            name=f"{model_name.upper()} Autoregressive Forecast",
            line=dict(color="#16a34a", width=3),
            marker=dict(size=6, color="#16a34a"),
        )
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        height=480,
        font=dict(color="#1e293b", family="Plus Jakarta Sans, sans-serif"),
        yaxis=dict(title="Price (INR)", gridcolor="#f1f5f9", linecolor="#cbd5e1"),
        xaxis=dict(gridcolor="#f1f5f9", linecolor="#cbd5e1"),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        hovermode="x unified",
    )
    return fig
