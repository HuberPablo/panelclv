"""Evaluation: metrics and forecast diagnostics / plotting.

The weekly-aggregate plotting / metrics-table / alignment / prediction-CSV-I/O
helpers (``plot_utils``) and the per-group tables (``segment_analysis``) live here.
These consume a forecast that the model + Monte Carlo simulator (in
``panelclv.models``) already produced; they do not define the model, so they sit in
their own subpackage.

``models.monte_carlo_forecasting.compute_forecast_metrics`` is the single authority for
``rmse`` / ``bias_percent`` / ``mape_aggregate_style``; everything here delegates to it
rather than defining its own. The one number it does not return is ``aggregate_bias``
(raw-count bias), which the per-group table needs because percentage bias is
uninformative for a group whose actual total is near zero.
"""

from .plot_utils import (
    weekly_actuals,
    weekly_aggregate_predictions,
    plot_weekly_aggregated,
    metrics_table,
    alignment_check,
    pareto_forecast,
    save_predictions_to_csv,
    load_predictions_from_csv,
)
from .segment_analysis import (
    assign_customer_groups,
    group_metrics_table,
    aggregate_bias,
)
from .forecast_run import ForecastRun

__all__ = [
    "weekly_actuals",
    "weekly_aggregate_predictions",
    "plot_weekly_aggregated",
    "metrics_table",
    "alignment_check",
    "pareto_forecast",
    "save_predictions_to_csv",
    "load_predictions_from_csv",
    "assign_customer_groups",
    "group_metrics_table",
    "aggregate_bias",
    "ForecastRun",
]
