"""Evaluation: metrics and forecast diagnostics / plotting.

The weekly-aggregate plot and the metrics table (``plots``) and the per-group
tables (``segment_analysis``) live here. These consume a forecast that the model +
Monte Carlo simulator (in ``panelclv.models``) already produced; they do not define
the model, so they sit in their own subpackage. The forecast's on-disk format is
not evaluation either — that is ``panelclv.predictions``, which both this
subpackage and the model layer read from.

``models.monte_carlo_forecasting.compute_forecast_metrics`` is the single authority for
``rmse`` / ``rmse_customer_total`` / ``bias_percent`` / ``mape_aggregate`` — the only
place in the package that computes them. The two RMSEs differ in what they aggregate
over before squaring: a customer-week cell, and a customer's whole holdout total (the
"individual-level RMSE" the published benchmark reports). Everything here delegates to it rather than defining its own.
The one number it does not return is ``aggregate_bias`` (raw-count bias), which the
per-group table needs because percentage bias is uninformative for a group whose
actual total is near zero.

``effects`` is the same kind of authority one level up: the only implementation of a
comparative claim (delta of means, 95% percentile bootstrap, independent or paired), as
``docs/statistical-protocol.md`` defines it.
"""

from .plots import (
    plot_weekly_aggregated,
    metrics_table,
)
from .effects import Effect, effect, table as effects_table
from .segment_analysis import (
    CUSTOMER_GROUPS,
    assign_customer_groups,
    group_metrics_table,
    aggregate_bias,
)

__all__ = [
    "Effect",
    "effect",
    "effects_table",
    "plot_weekly_aggregated",
    "metrics_table",
    "CUSTOMER_GROUPS",
    "assign_customer_groups",
    "group_metrics_table",
    "aggregate_bias",
]
