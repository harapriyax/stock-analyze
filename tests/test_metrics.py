"""AlphaPulse v2.0 - Unit Tests for Metrics."""

from __future__ import annotations

import numpy as np

from src.training.metrics import (
    compute_directional_accuracy,
    compute_mae,
    compute_mape,
    compute_r2,
    compute_rmse,
    evaluate_all,
)


def test_metrics_perfect_prediction():
    y_true = np.array([100.0, 105.0, 102.0, 108.0])
    y_pred = np.array([100.0, 105.0, 102.0, 108.0])

    assert compute_rmse(y_true, y_pred) == 0.0
    assert compute_mae(y_true, y_pred) == 0.0
    assert compute_mape(y_true, y_pred) == 0.0
    assert compute_r2(y_true, y_pred) == 1.0
    assert compute_directional_accuracy(y_true, y_pred) == 100.0


def test_directional_accuracy():
    y_true = np.array([100.0, 110.0, 105.0, 115.0])  # diffs: +10, -5, +10
    y_pred = np.array([100.0, 112.0, 103.0, 118.0])  # diffs: +12, -9, +15 (same signs)

    acc = compute_directional_accuracy(y_true, y_pred)
    assert acc == 100.0

    # Opposite direction
    y_pred_bad = np.array([100.0, 95.0, 110.0, 100.0])  # diffs: -5, +15, -10 (all opposite)
    assert compute_directional_accuracy(y_true, y_pred_bad) == 0.0


def test_evaluate_all():
    y_true = np.array([100.0, 110.0, 105.0])
    y_pred = np.array([102.0, 108.0, 106.0])

    res = evaluate_all(y_true, y_pred)
    assert "RMSE" in res
    assert "MAE" in res
    assert "MAPE" in res
    assert "R2" in res
    assert "Directional_Accuracy" in res

