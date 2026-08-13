"""Weekly-aggregate forecast plot and the per-model metrics table.

Both take a forecast the model + Monte Carlo simulator already produced and say
something about it to a reader: one draws it against the actuals, the other prints
the three numbers the thesis reports.

# Metric convention
# -----------------
# The thesis pipeline reports three numbers everywhere — `rmse`, `bias_percent`,
# `mape_aggregate` — all in **percent scale** and all computed on per-customer
# per-week arrays of shape (N, T_HOLD). `metrics_table` below delegates to
# `models.monte_carlo_forecasting.compute_forecast_metrics` so the notebook
# printouts and the plot helper agree to the last decimal. That function is the
# package's single scoring authority — there is no second definition of these
# numbers to drift from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Both cross-package imports point DOWN the altitude split: the metric authority
# scores the model's own forecast and stays in `panelclv.models`; the Pareto/NBD
# fit lives with the benchmark it belongs to; the shape reducer is prediction I/O.
from panelclv.benchmarks.pareto_nbd import pareto_from_data
from panelclv.models.monte_carlo_forecasting import compute_forecast_metrics
from panelclv.predictions import reduce_to_customer_period


def _aggregate_across_customers(
    predictions: np.ndarray,
    ci: tuple[float, float] = (0.025, 0.975),
) -> dict[str, np.ndarray]:
    """Sum a forecast across customers into one curve per holdout week.

    Accepts:
        - (n_simulations, n_customers, T, 1) — Monte Carlo output.
        - (n_customers, T) or (n_customers, T, 1) — deterministic prediction
          (e.g. Pareto/NBD expected counts).

    Returns a dict with key "mean" (always) and "low_ci" / "high_ci" when the
    prediction is a Monte Carlo array — the quantiles are taken across the
    simulated paths, which is what the plot draws as its ribbon. A deterministic
    prediction has no paths to spread, so it gets no ribbon.
    """
    arr = np.asarray(predictions, dtype=np.float64)

    if arr.ndim == 4:
        # (S, N, T, 1) -> sum over customers -> (S, T). Summed BEFORE averaging over
        # paths, because the ribbon is the spread of whole simulated aggregates —
        # which is exactly what `reduce_to_customer_period` would average away.
        per_sim_per_period = arr.squeeze(-1).sum(axis=1)
        lo, hi = ci
        return {
            "mean": per_sim_per_period.mean(axis=0),
            "low_ci": np.quantile(per_sim_per_period, lo, axis=0),
            "high_ci": np.quantile(per_sim_per_period, hi, axis=0),
        }

    # The other two shapes carry no paths, so the reducer's own dispatch (and its
    # error message) is the one definition of what a prediction may look like.
    return {"mean": reduce_to_customer_period(arr).sum(axis=0)}


def plot_weekly_aggregated(
    actuals: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
    train_actuals: np.ndarray | None = None,
    title: str = "Weekly aggregated transactions",
    show_ci: bool = True,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    *,
    pareto_benchmark: bool = False,
    data: dict[str, Any] | None = None,
):
    """Plot weekly-aggregate actuals vs each model's weekly-aggregate forecast.

    Parameters
    ----------
    actuals : (T_HOLD,) ndarray
        Aggregated actuals for the holdout window.
    predictions_by_model : dict[str, ndarray]
        Each prediction is plotted on the holdout x-axis.
    train_actuals : (T_CAL,) ndarray, optional
        If provided, the training-window aggregate is plotted to the left of
        the holdout, with a dashed vertical boundary at `T_CAL - 0.5`.
    show_ci : bool
        Draw the 95% MC ribbon for any prediction supplied as a Monte Carlo
        array.
    pareto_benchmark : bool
        When True, fit the Pareto/NBD benchmark and add it to the plot as one
        more (no-CI) line. Requires `data`; everything the model needs
        (train_panel, T_HOLD, cohort ids, target_col, id_col, frequency) is read
        from it, so no Pareto-specific arguments are taken — it is fit and
        aligned on exactly the cohort `data` describes.
    data : dict, optional
        A `prepare_dataset` output. Only used when `pareto_benchmark=True`.

    A 95% MC confidence ribbon is drawn for any model whose predictions are a
    Monte Carlo array. Returns `(fig, ax)`.
    """
    import matplotlib.pyplot as plt

    # Optionally fit + append the Pareto/NBD benchmark as one more line. The
    # caller's dict is copied so it is never mutated.
    models = dict(predictions_by_model)
    if pareto_benchmark:
        models["Pareto/NBD"] = pareto_from_data(data)             # (N, T_HOLD) line, no CI

    if figsize is None:
        figsize = (15, 4.5) if train_actuals is not None else (10, 5)

    fig, ax = plt.subplots(figsize=figsize)

    if train_actuals is not None:
        t_cal = len(train_actuals)
        train_x = np.arange(t_cal)
        hold_x  = np.arange(t_cal, t_cal + len(actuals))
        ax.plot(train_x, train_actuals,
                label="Actual (training)", color="grey", linewidth=1.3, alpha=0.8)
        ax.axvline(t_cal - 0.5, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    else:
        hold_x = np.arange(len(actuals))

    ax.plot(hold_x, actuals, label="Actual (holdout)", color="black", linewidth=2.0)

    for name, preds in models.items():
        agg = _aggregate_across_customers(preds)
        (line,) = ax.plot(hold_x, agg["mean"], label=name, linewidth=1.5)
        if show_ci and "low_ci" in agg:
            ax.fill_between(
                hold_x, agg["low_ci"], agg["high_ci"],
                alpha=0.15, color=line.get_color(),
            )

    ax.set_xlabel("Week" if train_actuals is not None else "Holdout week")
    ax.set_ylabel("Aggregate transactions")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    return fig, ax


def metrics_table(
    actuals: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
    *,
    pareto_benchmark: bool = False,
    data: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Per-model evaluation table — same three numbers as the notebooks print.

    Returns one row per model with columns
    `rmse`, `bias_percent`, `mape_aggregate` — exactly the keys
    `monte_carlo_forecasting.compute_forecast_metrics` returns. This is the
    package's single metric convention; everything (this table, the LSTM
    notebook printout, the demo notebook printout) flows through the same
    helper so the numbers reconcile to the last decimal.

    Parameters
    ----------
    actuals : np.ndarray, shape (N, T_HOLD)
        Per-customer per-week actuals — the natural output of
        `forecast["actual"]` from the Monte Carlo simulator. NOT the aggregated
        (T_HOLD,) vector; the metric definitions need the per-customer
        granularity to compute individual RMSE.
    predictions_by_model : dict[str, np.ndarray]
        Each prediction may be:
          - (S, N, T_HOLD, 1)  full Monte Carlo array → reduced via mean over S,
          - (N, T_HOLD, 1)     deterministic prediction with trailing channel,
          - (N, T_HOLD)        already a per-customer mean (e.g. Pareto/NBD).
        `reduce_to_customer_period` normalises all three to (N, T_HOLD)
        before scoring, so the function is shape-polymorphic at the input.
    pareto_benchmark : bool
        When True, fit the Pareto/NBD benchmark on `data` and add it as a
        `"Pareto/NBD"` row, so the LSTM and the benchmark land in one table on
        the same actuals. Requires `data` (a `prepare_dataset` output); nothing
        else is needed — it is fit + aligned on exactly that cohort.
    data : dict, optional
        A `prepare_dataset` output. Only used when `pareto_benchmark=True`.

    Notes
    -----
    Why NOT pre-aggregate actuals/predictions to (T_HOLD,) and score those?
    The aggregate vector would still give a correct `mape_aggregate`,
    but `rmse` on the aggregate is a different quantity (lower bound on the
    individual RMSE thanks to error cancellation across customers). The thesis
    reports individual RMSE, so we score on per-customer arrays.
    """
    actuals = np.asarray(actuals, dtype=np.float64)
    if actuals.ndim != 2:
        raise ValueError(
            f"actuals must be (N, T_HOLD); got shape {actuals.shape}. "
            f"If you have an aggregated (T_HOLD,) vector, score the "
            f"per-customer array instead (`forecast[\"actual\"]`)."
        )

    # Optionally fit + append the Pareto/NBD benchmark as one more row (copy so
    # the caller's dict is never mutated). Same primitive as the plot helper.
    models = dict(predictions_by_model)
    if pareto_benchmark:
        models["Pareto/NBD"] = pareto_from_data(data)

    rows = []
    for name, preds in models.items():
        # Normalize predictions to (N, T_HOLD); `reduce_to_customer_period`
        # already handles the three accepted shapes and means over MC sims.
        pred_NT = reduce_to_customer_period(preds)
        if pred_NT.shape != actuals.shape:
            raise ValueError(
                f"model {name!r}: prediction shape {pred_NT.shape} does not "
                f"match actuals shape {actuals.shape}"
            )
        m = compute_forecast_metrics(actuals, pred_NT)
        m["model"] = name
        rows.append(m)

    df = pd.DataFrame(rows).set_index("model")
    return df[["rmse", "bias_percent", "mape_aggregate"]]
