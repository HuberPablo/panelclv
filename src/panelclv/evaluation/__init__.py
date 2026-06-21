"""Evaluation: metrics and forecast diagnostics / plotting.

Scoring (``evaluation_utils``) and the weekly-aggregate plotting / metrics-table /
alignment / prediction-CSV-I/O helpers (``plot_utils``) live here. These consume a
forecast that the model + Monte Carlo simulator (in ``panelclv.models``) already
produced; they do not define the model, so they sit in their own subpackage.
"""

from .evaluation_utils import (
    compute_metrics,
    rmse,
    mae,
    mape_positive,
    aggregate_bias,
)
from .plot_utils import (
    weekly_actuals,
    holdout_actuals_NT,
    weekly_aggregate_predictions,
    plot_weekly_aggregated,
    metrics_table,
    alignment_check,
    forecast_from_checkpoint,
    pareto_forecast,
    save_predictions_to_csv,
    load_predictions_from_csv,
)

__all__ = [
    "compute_metrics",
    "rmse",
    "mae",
    "mape_positive",
    "aggregate_bias",
    "weekly_actuals",
    "holdout_actuals_NT",
    "weekly_aggregate_predictions",
    "plot_weekly_aggregated",
    "metrics_table",
    "alignment_check",
    "forecast_from_checkpoint",
    "pareto_forecast",
    "save_predictions_to_csv",
    "load_predictions_from_csv",
]
