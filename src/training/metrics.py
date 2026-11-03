"""AlphaPulse v2.0 - Quantitative Evaluation Metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Root Mean Squared Error in real price units (INR)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Absolute Error in real price units (INR)."""
    return float(mean_absolute_error(y_true, y_pred))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Absolute Percentage Error (MAPE)."""
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes R-squared determination score."""
    return float(r2_score(y_true, y_pred))


def compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes the percentage of correct upward/downward movement predictions."""
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0
    actual_diff = np.diff(y_true)
    pred_diff = np.diff(y_pred)
    concordant = np.sign(actual_diff) == np.sign(pred_diff)
    return float(np.mean(concordant) * 100)


def compute_financial_indexes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    risk_free_rate: float = 0.06,
) -> dict[str, float]:
    """Computes quantitative financial performance indexes based on simulated trading signals.

    Includes:
      - Sharpe Ratio (Annualized risk-adjusted return)
      - Sortino Ratio (Downside deviation risk-adjusted return)
      - Max Drawdown (Maximum peak-to-trough drop in %)
      - Win Rate (% of profitable trading days)
      - Profit Factor (Gross profits / gross losses)
    """
    y_t = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_p = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if len(y_t) < 3:
        return {
            "Sharpe_Ratio": 0.0,
            "Sortino_Ratio": 0.0,
            "Max_Drawdown": 0.0,
            "Win_Rate": 0.0,
            "Profit_Factor": 1.0,
        }

    # Actual daily returns
    r_act = np.diff(y_t) / np.clip(y_t[:-1], 1e-8, None)
    # Signal derived from prediction relative to previous day
    signals = np.where(y_p[1:] >= y_t[:-1], 1.0, -1.0)
    strat_returns = signals * r_act

    # Annualized Sharpe ratio (252 trading sessions)
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
    excess_returns = strat_returns - daily_rf
    mean_exc = float(np.mean(excess_returns))
    std_strat = float(np.std(strat_returns))
    sharpe = float((mean_exc / std_strat) * np.sqrt(252)) if std_strat > 1e-8 else 0.0

    # Sortino ratio (downside standard deviation)
    downside = strat_returns[strat_returns < 0]
    downside_std = float(np.std(downside)) if len(downside) > 0 else 1e-8
    sortino = float((mean_exc / downside_std) * np.sqrt(252)) if downside_std > 1e-8 else 0.0

    # Cumulative wealth equity curve & Max Drawdown
    equity = np.cumprod(1.0 + strat_returns)
    peak = np.maximum.accumulate(equity)
    drawdowns = (peak - equity) / peak
    max_dd = float(np.max(drawdowns) * 100) if len(drawdowns) > 0 else 0.0

    # Win rate and profit factor
    wins = strat_returns[strat_returns > 0]
    losses = strat_returns[strat_returns < 0]
    win_rate = float(len(wins) / len(strat_returns) * 100) if len(strat_returns) > 0 else 0.0
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-8
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 1.0

    return {
        "Sharpe_Ratio": round(sharpe, 3),
        "Sortino_Ratio": round(sortino, 3),
        "Max_Drawdown": round(max_dd, 2),
        "Win_Rate": round(win_rate, 2),
        "Profit_Factor": round(min(profit_factor, 999.9), 2),
    }


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Aggregates all evaluation metrics and financial indexes into a standardized dictionary."""
    y_t = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_p = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    fin_indexes = compute_financial_indexes(y_t, y_p)

    return {
        "RMSE": compute_rmse(y_t, y_p),
        "MAE": compute_mae(y_t, y_p),
        "MAPE": compute_mape(y_t, y_p),
        "R2": compute_r2(y_t, y_p),
        "Directional_Accuracy": compute_directional_accuracy(y_t, y_p),
        **fin_indexes,
    }

