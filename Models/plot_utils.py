"""Plotting and aggregation helpers for the multinomial baselines.

Given Monte Carlo forecast arrays from `monte_carlo_forecasting.mc_forecast`
(or any deterministic per-customer prediction matrix), these helpers:

    - aggregate transaction counts across customers per holdout week,
    - draw a weekly-aggregate "actual vs predicted" plot for several models
      in one figure (with optional MC confidence ribbon),
    - report a RMSE / bias% / MAPE table using the SAME definitions as the
      Monte Carlo simulator (`compute_forecast_metrics`).

# Metric convention
# -----------------
# The thesis pipeline reports three numbers everywhere — `rmse`, `bias_percent`,
# `mape_aggregate_style` — all in **percent scale** and all computed on
# per-customer per-week arrays of shape (N, T_HOLD). `metrics_table` below
# delegates to `monte_carlo_forecasting.compute_forecast_metrics` so the
# notebook printouts and the plot helper agree to the last decimal.
# The older `evaluation_utils.compute_metrics` keys (mae, mape_positive,
# cumulative_mape, ...) are kept for back-compat but are NOT the convention
# used by the package's notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from .evaluation_utils import compute_metrics
from .monte_carlo_forecasting import (
    compute_forecast_metrics,
    run_monte_carlo_forecast as _mc_forecast,
)


# ---------------------------------------------------------------------------
# CSV I/O for predictions
# ---------------------------------------------------------------------------


def _reduce_to_customer_week(predictions: np.ndarray) -> np.ndarray:
    """Collapse MC predictions to a 2-D (n_customers, T) array of means."""
    arr = np.asarray(predictions, dtype=np.float64)
    if arr.ndim == 4:                # (S, N, T, 1)  -> mean over S, drop channel
        return arr.squeeze(-1).mean(axis=0)
    if arr.ndim == 3 and arr.shape[-1] == 1:   # (N, T, 1)
        return arr.squeeze(-1)
    if arr.ndim == 2:                # (N, T)
        return arr
    raise ValueError(
        f"Expected predictions of shape (S, N, T, 1), (N, T, 1), or (N, T); "
        f"got {predictions.shape}"
    )


def save_predictions_to_csv(
    predictions: np.ndarray,
    path: str | Path,
    customer_ids: Sequence | None = None,
    week_offset: int = 0,
    id_col: str = "customer_id",
) -> Path:
    """Save predictions to a wide CSV: `id_col` + `week_0..week_{T-1}`.

    For Monte Carlo arrays of shape (S, N, T, 1), the saved values are the mean
    across simulations. Deterministic predictions (N, T) or (N, T, 1) are saved
    as-is. The parent folder is created if it doesn't exist.
    """
    arr = _reduce_to_customer_week(predictions)
    n_customers, n_weeks = arr.shape

    if customer_ids is None:
        customer_ids = np.arange(n_customers)
    else:
        customer_ids = np.asarray(customer_ids)
        if customer_ids.shape[0] != n_customers:
            raise ValueError(
                f"customer_ids has {customer_ids.shape[0]} rows but predictions "
                f"have {n_customers} customers"
            )

    columns = [f"week_{i + week_offset}" for i in range(n_weeks)]
    df = pd.DataFrame(arr, columns=columns)
    df.insert(0, id_col, customer_ids)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_predictions_from_csv(
    path: str | Path,
    id_col_candidates: Sequence[str] = ("customer_id", "id", "Id", "ID"),
    holdout_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load wide-CSV predictions back as a (n_customers, T) array.

    Returns `(values, ids)`. `ids` is `None` when no id column is found.
    If `holdout_length` is given, trailing/extra week columns are truncated.
    """
    df = pd.read_csv(path)
    ids = None
    for col in id_col_candidates:
        if col in df.columns:
            ids = df[col].to_numpy()
            df = df.drop(columns=[col])
            break
    arr = df.to_numpy(dtype=np.float64)
    if holdout_length is not None:
        arr = arr[:, :holdout_length]
    return arr, ids


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def holdout_actuals_NT(
    holdout: Sequence[pd.DataFrame],
    *,
    target_col: str | None = None,
    count_col: str | int | None = None,
) -> np.ndarray:
    """Return per-customer per-week actuals over the holdout as `(N, T_HOLD)`.

    `holdout` is a list of per-customer dataframes (each shape `(T, num_features)`),
    matching the layout `run_monte_carlo_forecast` consumes. Exactly the same
    array shape the LSTM / Transformer notebooks pass to
    `compute_forecast_metrics`, so callers can score directly without
    aggregating first.

    Parameters
    ----------
    target_col : str, optional (recommended)
        Name of the transaction-count column. Resolved via `df[target_col]`.
        Pass this when you have a `PanelConfig` — it's the schema-correct way
        to name the column and survives any reordering of `seq_cols`.
    count_col : str | int, optional (legacy)
        Older positional fallback. Accepts a name OR an integer column index.
        Kept for back-compat with notebooks that hardcoded the old default
        (`count_col=3`); prefer `target_col=` for new code.

    Exactly one of `target_col` / `count_col` must be supplied — there is no
    silent positional default anymore, because index 3 was tied to a single
    legacy schema and silently produced wrong arrays on any other layout.
    """
    if (target_col is None) == (count_col is None):
        raise ValueError(
            "Pass exactly one of target_col= (recommended) or count_col= "
            "(legacy). The old magic default count_col=3 was removed because "
            "it produced silently-wrong arrays on any non-default schema."
        )
    arrs = []
    for df in holdout:
        if target_col is not None:
            arrs.append(df[target_col].to_numpy(dtype=np.float64))
        elif isinstance(count_col, int):
            arrs.append(df.iloc[:, count_col].to_numpy(dtype=np.float64))
        else:
            arrs.append(df[count_col].to_numpy(dtype=np.float64))
    # Stack along customer axis -> (N, T_HOLD). Callers that want the
    # weekly aggregate (T_HOLD,) for the plot should sum over axis 0.
    return np.stack(arrs, axis=0)


def weekly_actuals(
    holdout: Sequence[pd.DataFrame],
    *,
    target_col: str | None = None,
    count_col: str | int | None = None,
) -> np.ndarray:
    """Sum transaction counts across customers per holdout week, shape `(T_HOLD,)`.

    Thin wrapper over `holdout_actuals_NT` that sums along the customer axis —
    use this for the plot overlay (`plot_weekly_aggregated`). For metrics, use
    `holdout_actuals_NT` directly and pass the `(N, T_HOLD)` array to
    `metrics_table` / `compute_forecast_metrics`.

    See `holdout_actuals_NT` for the `target_col` / `count_col` contract;
    the legacy positional default (`count_col=3`) is no longer accepted.
    """
    return holdout_actuals_NT(
        holdout, target_col=target_col, count_col=count_col,
    ).sum(axis=0)


def weekly_aggregate_predictions(
    predictions: np.ndarray,
    ci: tuple[float, float] = (0.025, 0.975),
) -> dict[str, np.ndarray]:
    """Aggregate predictions across customers per holdout week.

    Accepts:
        - (n_simulations, n_customers, T, 1) — Monte Carlo output.
        - (n_customers, T) or (n_customers, T, 1) — deterministic prediction
          (e.g. Pareto/NBD expected counts).

    Returns a dict with key "mean" (always) and "low_ci" / "high_ci" when the
    prediction is a Monte Carlo array.
    """
    arr = np.asarray(predictions, dtype=np.float64)

    if arr.ndim == 4:
        # (S, N, T, 1) -> sum over customers -> (S, T)
        per_sim_per_week = arr.squeeze(-1).sum(axis=1)
        lo, hi = ci
        return {
            "mean": per_sim_per_week.mean(axis=0),
            "low_ci": np.quantile(per_sim_per_week, lo, axis=0),
            "high_ci": np.quantile(per_sim_per_week, hi, axis=0),
        }

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected predictions of shape (S, N, T, 1), (N, T, 1) or (N, T); "
            f"got {predictions.shape}"
        )
    return {"mean": arr.sum(axis=0)}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


# Calendar days per period, used to fit Pareto/NBD on the right time scale.
_PERIOD_IN_DAYS: dict[str, float] = {"weekly": 7.0, "monthly": 30.4368, "daily": 1.0}


def _pareto_from_data(data: dict[str, Any] | None, variant: str = "mle") -> np.ndarray:
    """Fit a Pareto/NBD benchmark on a `prepare_dataset` output.

    Shared by `plot_weekly_aggregated` and `metrics_table` so the benchmark is
    fit ONE way. Everything is read from `data` (train_panel, T_HOLD, cohort
    ids, target_col, id_col, frequency) — the dict is self-describing, so no
    Pareto-specific arguments are needed and it is fit + aligned on exactly the
    cohort `data` describes. The heavy fitter is imported lazily so plot_utils
    stays importable without it. Returns an (N, T_HOLD) per-customer prediction.

    `variant` selects the estimator:
        "mle"   — `pareto_nbd.compute_pareto_predictions` (lifetimes MLE; fast).
        "paper" — `pareto_paper.compute_pareto_paper_predictions` (hierarchical-Bayes
                  MCMC, BTYDplus-faithful; the estimator Valendin et al. actually use).
    """
    if data is None:
        raise ValueError("a Pareto/NBD benchmark requires data=<prepare_dataset output>.")
    missing = [k for k in ("train_panel", "T_HOLD", "ids", "target_col",
                           "id_col", "frequency") if k not in data]
    if missing:
        raise ValueError(
            f"data is missing keys {missing} needed for the Pareto/NBD benchmark; "
            f"re-run prepare_dataset (older runs predate id_col/frequency)."
        )
    period_in_days = _PERIOD_IN_DAYS.get(data["frequency"])
    if period_in_days is None:
        raise ValueError(
            f"cannot map frequency {data['frequency']!r} to a period length; "
            f"known frequencies: {sorted(_PERIOD_IN_DAYS)}."
        )
    if variant == "paper":
        from .pareto_paper import compute_pareto_paper_predictions
        pred, _ = compute_pareto_paper_predictions(
            data["train_panel"],
            holdout_length=data["T_HOLD"],
            id_col=data["id_col"],
            target_col=data["target_col"],
            period_in_days=period_in_days,
            customer_ids=data["ids"],
        )
        return pred
    if variant != "mle":
        raise ValueError(f"unknown Pareto variant {variant!r}; use 'mle' or 'paper'.")
    from .pareto_nbd import compute_pareto_predictions
    pareto_pred, _ = compute_pareto_predictions(
        data["train_panel"],
        holdout_length=data["T_HOLD"],
        id_col=data["id_col"],
        target_col=data["target_col"],
        period_in_days=period_in_days,
        customer_ids=data["ids"],
    )
    return pareto_pred


def plot_weekly_aggregated(
    actuals: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
    train_actuals: np.ndarray | None = None,
    title: str = "Weekly aggregated transactions",
    show_ci: bool = True,
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    *,
    pareto_nbd_benchmark: bool = False,
    pareto_paper_benchmark: bool = False,
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
    pareto_nbd_benchmark : bool
        When True, fit the Pareto/NBD benchmark and add it to the plot as one
        more (no-CI) line. Requires `data`; everything the model needs
        (train_panel, T_HOLD, cohort ids, target_col, id_col, frequency) is read
        from it, so no Pareto-specific arguments are taken — it is fit and
        aligned on exactly the cohort `data` describes.
    data : dict, optional
        A `prepare_dataset` output. Only used when `pareto_nbd_benchmark=True`.

    A 95% MC confidence ribbon is drawn for any model whose predictions are a
    Monte Carlo array. Returns `(fig, ax)`.
    """
    import matplotlib.pyplot as plt

    # Optionally fit + append the Pareto/NBD benchmark as one more line. The
    # caller's dict is copied so it is never mutated.
    models = dict(predictions_by_model)
    if pareto_nbd_benchmark:
        models["Pareto/NBD"] = _pareto_from_data(data, "mle")     # (N, T_HOLD) line, no CI
    if pareto_paper_benchmark:
        models["Pareto/NBD (HB)"] = _pareto_from_data(data, "paper")

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
        agg = weekly_aggregate_predictions(preds)
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metrics_table(
    actuals: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
    *,
    pareto_nbd_benchmark: bool = False,
    pareto_paper_benchmark: bool = False,
    data: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Per-model evaluation table — same three numbers as the notebooks print.

    Returns one row per model with columns
    `rmse`, `bias_percent`, `mape_aggregate_style` — exactly the keys
    `monte_carlo_forecasting.compute_forecast_metrics` returns. This is the
    package's single metric convention; everything (this table, the LSTM
    notebook printout, the demo notebook printout) flows through the same
    helper so the numbers reconcile to the last decimal.

    Parameters
    ----------
    actuals : np.ndarray, shape (N, T_HOLD)
        Per-customer per-week actuals — the natural output of
        `holdout_actuals_NT` or `forecast["actual"]`. NOT the aggregated
        (T_HOLD,) vector; the metric definitions need the per-customer
        granularity to compute individual RMSE.
    predictions_by_model : dict[str, np.ndarray]
        Each prediction may be:
          - (S, N, T_HOLD, 1)  full Monte Carlo array → reduced via mean over S,
          - (N, T_HOLD, 1)     deterministic prediction with trailing channel,
          - (N, T_HOLD)        already a per-customer mean (e.g. Pareto/NBD).
        `_reduce_to_customer_week` normalises all three to (N, T_HOLD)
        before scoring, so the function is shape-polymorphic at the input.
    pareto_nbd_benchmark : bool
        When True, fit the Pareto/NBD benchmark on `data` and add it as a
        `"Pareto/NBD"` row, so the LSTM and the benchmark land in one table on
        the same actuals. Requires `data` (a `prepare_dataset` output); nothing
        else is needed — it is fit + aligned on exactly that cohort.
    data : dict, optional
        A `prepare_dataset` output. Only used when `pareto_nbd_benchmark=True`.

    Notes
    -----
    Why NOT pre-aggregate actuals/predictions to (T_HOLD,) and score those?
    The aggregate vector would still give a correct `mape_aggregate_style`,
    but `rmse` on the aggregate is a different quantity (lower bound on the
    individual RMSE thanks to error cancellation across customers). The thesis
    reports individual RMSE, so we score on per-customer arrays.
    """
    actuals = np.asarray(actuals, dtype=np.float64)
    if actuals.ndim != 2:
        raise ValueError(
            f"actuals must be (N, T_HOLD); got shape {actuals.shape}. "
            f"If you have the aggregated (T_HOLD,) vector, use "
            f"holdout_actuals_NT(...) instead of weekly_actuals(...)."
        )

    # Optionally fit + append the Pareto/NBD benchmark as one more row (copy so
    # the caller's dict is never mutated). Same primitive as the plot helper.
    models = dict(predictions_by_model)
    if pareto_nbd_benchmark:
        models["Pareto/NBD"] = _pareto_from_data(data, "mle")
    if pareto_paper_benchmark:
        models["Pareto/NBD (HB)"] = _pareto_from_data(data, "paper")

    rows = []
    for name, preds in models.items():
        # Normalize predictions to (N, T_HOLD); `_reduce_to_customer_week`
        # already handles the three accepted shapes and means over MC sims.
        pred_NT = _reduce_to_customer_week(preds)
        if pred_NT.shape != actuals.shape:
            raise ValueError(
                f"model {name!r}: prediction shape {pred_NT.shape} does not "
                f"match actuals shape {actuals.shape}"
            )
        m = compute_forecast_metrics(actuals, pred_NT)
        m["model"] = name
        rows.append(m)

    df = pd.DataFrame(rows).set_index("model")
    return df[["rmse", "bias_percent", "mape_aggregate_style"]]


# ---------------------------------------------------------------------------
# Alignment diagnostic
# ---------------------------------------------------------------------------


def alignment_check(
    actuals: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
    max_lag: int = 3,
) -> pd.DataFrame:
    """Detect an involuntary time shift between actuals and predictions.

    For each model, the weekly-aggregate prediction series is shifted by
    every integer lag k in [-max_lag, +max_lag] and its Pearson correlation
    with the aggregate actuals is computed. The lag at which correlation
    peaks is reported as `best_lag`:

        best_lag = 0     no shift — predictions and actuals align
        best_lag > 0     predictions are LATE by `best_lag` steps
                         (week i of predictions matches week i+best_lag of actuals)
        best_lag < 0     predictions are EARLY by `|best_lag|` steps

    `total_actual` / `total_pred` give a level sanity check; large gaps
    point to a bias issue rather than a timing issue.
    """
    actuals_arr = np.asarray(actuals, dtype=np.float64)
    T = len(actuals_arr)

    rows = []
    for name, preds in predictions_by_model.items():
        agg = weekly_aggregate_predictions(preds)["mean"]
        if len(agg) != T:
            raise ValueError(
                f"Model {name!r}: predictions length {len(agg)} != actuals length {T}"
            )

        row: dict = {"model": name}
        corrs: dict[int, float] = {}
        for k in range(-max_lag, max_lag + 1):
            if k > 0:
                a, b = actuals_arr[k:], agg[:-k]
            elif k < 0:
                a, b = actuals_arr[:k], agg[-k:]
            else:
                a, b = actuals_arr, agg
            if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
                corrs[k] = float("nan")
            else:
                corrs[k] = float(np.corrcoef(a, b)[0, 1])
            row[f"lag_{k:+d}"] = corrs[k]

        valid = {k: v for k, v in corrs.items() if not np.isnan(v)}
        row["best_lag"] = int(max(valid, key=valid.get)) if valid else 0
        row["total_actual"] = float(actuals_arr.sum())
        row["total_pred"] = float(agg.sum())
        rows.append(row)

    return pd.DataFrame(rows).set_index("model")


# ---------------------------------------------------------------------------
# Convenience: run MC directly from a checkpoint
# ---------------------------------------------------------------------------


def forecast_from_checkpoint(
    checkpoint_path: str | Path,
    inference_model_factory,                # callable: () -> nn.Module
    data: dict[str, Any],
    n_simulations: int = 30,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Load an inference model from disk and run the Valendin-style MC forecast.

    `inference_model_factory` is a zero-arg callable that constructs the
    inference model with the exact architecture used during training (so the
    checkpoint loads cleanly). `data` is the dict returned by
    `dynamic_panel_dataset.prepare_dataset`.

    Returns the dict produced by `mc_forecast`: `prediction_mean`, `actual`,
    `simulations`, `target_col`, `target_idx`, `n_simulations`.
    """
    model = inference_model_factory()
    state = torch.load(checkpoint_path, map_location="cpu")
    # The training-mode Transformer registers a `_cached_mask` buffer the
    # inference model doesn't have. Drop it; no-op for the LSTM.
    state.pop("_cached_mask", None)
    model.load_state_dict(state)
    return _mc_forecast(model, data, n_simulations=n_simulations, device=device)
