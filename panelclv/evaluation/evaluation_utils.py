"""Evaluation metrics for the multinomial baselines.

Pure NumPy. Operates on aggregate arrays (per-step totals across customers).
The Monte Carlo forecasting loop lives in
`Models/monte_carlo_forecasting.py`.

Convention note (read this before using these helpers in a new place)
---------------------------------------------------------------------
The package's notebooks and the plot helper (`plot_utils.metrics_table`) all
report three numbers — `rmse`, `bias_percent`, `mape_aggregate_style` — via
`monte_carlo_forecasting.compute_forecast_metrics`, in **percent** scale and
on per-customer per-week arrays of shape (N, T_HOLD).

The helpers in THIS module:
    - operate on aggregate (T,) vectors (sums-across-customers),
    - return MAPE/bias-fraction as **fractions** (0.05 = 5 %), not percent,
    - return MAE alongside RMSE.

They are kept because some downstream notebooks still import them, but for
new code prefer `compute_forecast_metrics`. Mixing the two scales in the
same table will silently produce mismatched-looking numbers (5 vs 0.05).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Metrics (operate on the original count scale; no target transform applied)
# ---------------------------------------------------------------------------


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean(diff ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(
        np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    )))


def mape_positive(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE computed only on positive actual values (skips zero-count steps)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = y_true > 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def aggregate_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sum(y_pred) - sum(y_true). Positive = over-forecast, negative = under-forecast."""
    return float(np.sum(y_pred) - np.sum(y_true))


def aggregate_bias_fraction(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Aggregate bias normalised by total actual.

    Returns a FRACTION (0.05 = +5% over-forecast), matching the scale of
    `mape_positive` and `cumulative_mape`. Multiply by 100 when displaying
    as a percentage. Returns NaN if `sum(y_true) == 0`.
    """
    total_true = float(np.sum(np.asarray(y_true, dtype=np.float64)))
    if total_true == 0:
        return float("nan")
    return float((np.sum(y_pred) - total_true) / total_true)


def cumulative_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE on the cumulative purchase curve C_t = sum_{s <= t} y_s.

    Smoother than per-step MAPE; matches the cumulative-curve metric used in
    the Valendin paper. Steps where the cumulative actual is still 0 are
    skipped.
    """
    c_true = np.cumsum(np.asarray(y_true, dtype=np.float64))
    c_pred = np.cumsum(np.asarray(y_pred, dtype=np.float64))
    mask = c_true > 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((c_true[mask] - c_pred[mask]) / c_true[mask])))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse":                    rmse(y_true, y_pred),
        "mae":                     mae(y_true, y_pred),
        "mape_positive":           mape_positive(y_true, y_pred),
        "cumulative_mape":         cumulative_mape(y_true, y_pred),
        "aggregate_bias":          aggregate_bias(y_true, y_pred),
        "aggregate_bias_fraction": aggregate_bias_fraction(y_true, y_pred),
    }


