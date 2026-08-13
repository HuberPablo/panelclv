"""The headline suite figure: every model's forecast over training + holdout.

Reads a finished suite through `suite_reader` — the models it holds, their stored
forecasts and the dataset it ran on — and overlays them on one weekly-aggregate
figure. Optionally restricted to a behavioural customer group or an explicit set of
customer ids, and optionally shaded with each model's spread ACROSS the suite's
independent studies. That band is a Student-t interval on the mean, computed by the
same helper `suite_metrics` prints in its tables, so the shading and the numbers
cannot disagree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.patches import Patch

from panelclv.benchmarks import pareto_forecast
from panelclv.data_preparation.target_channel import (
    calibration_counts,
    holdout_actuals,
)
from panelclv.evaluation import (
    CUSTOMER_GROUPS,
    assign_customer_groups,
    plot_weekly_aggregated,
)
from panelclv.predictions import load_predictions_from_csv

from .suite_metrics import t_interval_half_width
from .suite_reader import (
    _actuals_from_panel,
    _discover_models,
    _prediction_index,
    aggregate_suite_predictions,
    load_model_predictions,
)


# ---------------------------------------------------------------------------
# Actuals — the weekly curves the forecasts are overlaid on
# ---------------------------------------------------------------------------


def _aggregate_actuals(
    data: dict[str, Any], row_idx: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Sum the target channel across customers → (train_curve, holdout_curve).

    The count channel of the ``(N, T, F)`` calibration/holdout tensors, read through
    ``data_preparation.target_channel``. Summing over customers gives the
    weekly-aggregate curves the plot overlays: ``(T_CAL,)`` for training and
    ``(T_HOLD,)`` for holdout. If ``row_idx`` is given, only those customer rows are
    summed (a group / customer-id subset).
    """
    calibration = calibration_counts(data)                 # (N, T_CAL)
    holdout = holdout_actuals(data)                        # (N, T_HOLD)
    if row_idx is not None:
        calibration = calibration[row_idx]
        holdout = holdout[row_idx]
    return calibration.sum(axis=0), holdout.sum(axis=0)


# ---------------------------------------------------------------------------
# Customer selection — restrict a plot to a behavioural group or explicit ids
# ---------------------------------------------------------------------------


def _rows_for_ids(data: dict[str, Any], ids: Any) -> np.ndarray:
    """Map customer ids to row indices into the ``(N, ...)`` arrays (dedup, in order).

    Ids are matched as ``str`` so int/str id types interoperate; a repeated id is kept
    once. Raises ``ValueError`` naming ids that are not in the cohort.
    """
    id_to_row = {str(cid): i for i, cid in enumerate(data["ids"])}
    rows: list[int] = []
    missing: list[Any] = []
    seen: set[int] = set()
    for cid in ids:
        r = id_to_row.get(str(cid))
        if r is None:
            missing.append(cid)
        elif r not in seen:
            seen.add(r)
            rows.append(r)
    if missing:
        raise ValueError(
            f"{len(missing)} customer id(s) not in the cohort (first few: {missing[:5]})"
        )
    return np.asarray(rows, dtype=int)


def _select_rows(
    data: dict[str, Any],
    group: Any = None,
    customer_ids: Any = None,
) -> tuple[np.ndarray | None, str | None]:
    """Resolve a subset selector to ``(row_idx, label)`` for plotting.

    ``group``       — a behavioural group name or list of names from
                      ``CUSTOMER_GROUPS`` plus the derived ``"Other"`` (membership is
                      the same grouping the metrics table uses; "Other" = matched by
                      no group predicate).
    ``customer_ids``— a single customer id or an iterable of ids.
    Pass at most one. With neither, returns ``(None, None)`` — the whole cohort, so the
    caller keeps its original full-cohort behaviour untouched.
    """
    if group is not None and customer_ids is not None:
        raise ValueError("pass at most one of group= or customer_ids=")

    if customer_ids is not None:
        # A lone id (int/str) vs an iterable of ids.
        if isinstance(customer_ids, (str, bytes)) or np.isscalar(customer_ids):
            sel = [customer_ids]
        else:
            sel = list(customer_ids)
        row_idx = _rows_for_ids(data, sel)
        label = (
            f"customer {sel[0]}" if len(sel) == 1
            else f"{len(row_idx)} customers"
        )
        return row_idx, label

    if group is not None:
        names = [group] if isinstance(group, str) else list(group)
        # Every defined group plus the derived "Other", so group="Other" is well-defined.
        grouping = assign_customer_groups(data, groups=CUSTOMER_GROUPS, with_other=True)

        picked: list[Any] = []
        for name in names:
            if name not in grouping:
                raise ValueError(
                    f"unknown group {name!r}; available: {list(grouping)}"
                )
            picked.extend(grouping[name].tolist())
        row_idx = _rows_for_ids(data, picked)
        label = " + ".join(names)
        return row_idx, label

    return None, None


def _across_study_band(
    model_dir: Path, row_idx: np.ndarray | None, ci: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Per-week across-studies ``(mean, low, high, n_studies)`` of the aggregate forecast.

    Sums each study's ``Prediction_i.csv`` over the selected customers into a weekly
    curve, then summarises the spread of those curves ACROSS the independent studies with
    a Student-t interval ``mean ± t_{1-a/2, n-1} · s/√n``. Returns ``None`` for a model
    with fewer than two studies (a deterministic benchmark has a single fit, so there is
    no study-to-study spread to shade).
    """
    preds_dir = Path(model_dir) / "Predictions"
    curves: list[np.ndarray] = []
    for path in sorted(preds_dir.glob("Prediction_*.csv"), key=_prediction_index):
        values, _ = load_predictions_from_csv(path)          # (N, T_HOLD)
        if row_idx is not None:
            values = values[row_idx]
        curves.append(values.sum(axis=0))                    # aggregate over customers
    n = len(curves)
    if n < 2:
        return None

    stack = np.stack(curves, axis=0)                         # (n_studies, T_HOLD)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0, ddof=1)                          # sample spread across studies
    half = t_interval_half_width(std, n, ci)
    return mean, mean - half, mean + half, n


# ---------------------------------------------------------------------------
# The headline: plot every model's forecast over training + holdout
# ---------------------------------------------------------------------------


def plot_suite_forecast(
    root: str | Path,
    panel_path: str | Path | None = None,
    *,
    study: int | None = None,
    data: dict[str, Any] | None = None,
    group: Any = None,
    customer_ids: Any = None,
    confidence_interval: bool = False,
    ci: float = 0.95,
    save_path: str | Path | None = None,
    title: str | None = None,
    pareto_benchmark: bool = False,
    **plot_kwargs: Any,
):
    """Overlay every model's holdout forecast on the training+holdout actuals.

    Parameters
    ----------
    root
        The suite folder ``Studies/<name>/``.
    panel_path
        Path to the customer-period panel CSV the suite was built from. Used to
        rebuild the actual transaction curves (see module docstring). Required
        unless ``data`` is given.
    study
        ``int`` → plot that single study's forecast per model. ``None`` (default)
        → plot the across-studies mean per model, and write the aggregated CSVs
        via :func:`aggregate_suite_predictions`.
    data
        In-session escape hatch: pass the live ``prepare_dataset`` dict to skip the
        panel rebuild. Exactly one of ``panel_path`` / ``data`` must be supplied.
    group
        Restrict the plot to a behavioural group (or list of groups) — any of
        ``CUSTOMER_GROUPS`` or the derived ``"Other"``, the same membership the metrics
        table uses. Actuals and every model's forecast are then aggregated over only
        those customers.
    customer_ids
        Restrict the plot to a single customer id or an iterable of ids. Aggregates
        (sums) over exactly those customers — for one id, that customer's own curve.
        Pass at most one of ``group`` / ``customer_ids``; with neither, the whole
        cohort is plotted (unchanged behaviour).
    confidence_interval
        When ``True`` (requires the default ``study=None``, the across-studies mode),
        shade each model's forecast with a ``ci``-level band summarising its spread ACROSS
        the suite's independent studies (mean ± a Student-t interval), and annotate the
        title with e.g. ``"95% CI across 20 studies"``. A model with a single study (a
        deterministic benchmark) draws no band. This is a different ribbon from the MC one
        ``plot_weekly_aggregated``'s ``show_ci`` draws — that needs per-simulation arrays,
        which the disk-loaded per-customer means are not.
    ci
        Confidence level for that band (default ``0.95``). Only used when
        ``confidence_interval=True``.
    save_path
        If given, the figure is written here (PNG). With ``confidence_interval=True`` the
        file is written after the bands are drawn, so they are included.
    title
        Overrides the auto-generated, mode-aware title.
    pareto_benchmark
        Add the live Pareto/NBD benchmark. It is fit on the **full** cohort and then
        restricted to the same customer selection, so it is the identical model shown
        in the full-cohort plot — only the customers aggregated differ. Requires
        ``data`` (rebuilt from ``panel_path`` if needed).
    **plot_kwargs
        Forwarded verbatim to ``plot_weekly_aggregated`` (e.g. ``figsize=...``,
        ``show_ci=...``). Note MC confidence ribbons will not appear for disk-loaded
        predictions: the stored values are already per-customer means, not
        per-simulation arrays.

    Returns ``(fig, ax)``.
    """
    root = Path(root)

    if (panel_path is None) == (data is None):
        raise ValueError("pass exactly one of panel_path= or data=")
    if confidence_interval and study is not None:
        raise ValueError(
            "confidence_interval=True summarises spread ACROSS studies, so it needs the "
            "default study=None (across-studies mode), not a single study."
        )
    if data is None:
        data = _actuals_from_panel(root, panel_path)

    # Resolve the optional customer selection to row indices once; None = full cohort.
    row_idx, sel_label = _select_rows(data, group=group, customer_ids=customer_ids)

    train_curve, holdout_curve = _aggregate_actuals(data, row_idx)
    ref_ids = np.asarray(data["ids"])

    def _subset(values: np.ndarray) -> np.ndarray:
        """Restrict a full-cohort (N, T) forecast to the current selection."""
        return values if row_idx is None else values[row_idx]

    # Gather each model's forecast (single study or across-studies mean). Assert
    # cohort alignment against the actuals so prediction row i and actual row i are
    # the same customer — the linchpin of a correct overlay — then subset.
    predictions_by_model: dict[str, np.ndarray] = {}
    for name, model_dir in _discover_models(root):
        values, ids = load_model_predictions(model_dir, study=study)
        if ids is not None and not np.array_equal(np.asarray(ids), ref_ids):
            raise ValueError(
                f"model {name!r}: prediction customer ids do not match the "
                f"rebuilt dataset cohort — is panel_path the panel this suite was "
                f"built from?"
            )
        if values.shape[0] != ref_ids.shape[0]:
            raise ValueError(
                f"model {name!r}: {values.shape[0]} customers in predictions but "
                f"{ref_ids.shape[0]} in the dataset"
            )
        if values.shape[1] != holdout_curve.shape[0]:
            raise ValueError(
                f"model {name!r}: forecast horizon {values.shape[1]} != holdout "
                f"length {holdout_curve.shape[0]}"
            )
        predictions_by_model[name] = _subset(values)

    # Live Pareto/NBD benchmark(s): fit ONCE on the full cohort, then subset to the
    # same customers — so a group plot shows the same model as the full-cohort plot.
    # Pre-computing here (rather than letting plot_weekly_aggregated fit it) is what
    # lets the benchmark honour the selection.
    if pareto_benchmark:
        predictions_by_model["Pareto/NBD"] = _subset(
            pareto_forecast(data)["prediction_mean"]
        )

    # When averaging, also persist the (full-cohort) aggregated datasets on disk.
    if study is None:
        aggregate_suite_predictions(root)

    if title is None:
        if study is None:
            # Report how many studies were averaged (from the first model's files).
            _, first_dir = _discover_models(root)[0]
            n = len(list((first_dir / "Predictions").glob("Prediction_*.csv")))
            title = f"Weekly aggregated transactions — averaged over {n} studies"
        else:
            title = f"Weekly aggregated transactions — study {study}"
        if sel_label is not None:
            title = f"{title} — {sel_label}"

    # Across-studies confidence band: for each model with >1 study, shade the spread of
    # its weekly-aggregate forecast over the independent studies. Computed BEFORE the plot
    # so the study count can go in the title; drawn AFTER (matched to each line's colour).
    bands: dict[str, tuple] = {}
    if confidence_interval:
        pct = f"{ci * 100:g}%"
        bands = {
            name: band
            for name, model_dir in _discover_models(root)
            if (band := _across_study_band(model_dir, row_idx, ci)) is not None
        }
        if not bands:
            raise ValueError(
                "confidence_interval=True but no model has >1 study to form an interval "
                "(a deterministic benchmark is a single fit)."
            )
        n_studies = next(iter(bands.values()))[3]
        title = f"{title} — {pct} CI across {n_studies} studies"

    fig, ax = plot_weekly_aggregated(
        actuals=holdout_curve,
        predictions_by_model=predictions_by_model,
        train_actuals=train_curve,
        title=title,
        # Defer saving to after the bands are drawn, else the PNG misses them.
        save_path=None if confidence_interval else save_path,
        data=data,
        **plot_kwargs,
    )

    if confidence_interval:
        # Match each band to its model line by colour (the line labels are the model names).
        color_by_label = {ln.get_label(): ln.get_color() for ln in ax.get_lines()}
        t_cal = len(train_curve)
        hold_x = np.arange(t_cal, t_cal + holdout_curve.shape[0])
        for name, (_, low, high, _) in bands.items():
            ax.fill_between(
                hold_x, low, high,
                color=color_by_label.get(name), alpha=0.2, linewidth=0, zorder=1,
            )
        # One neutral legend entry explaining the shading, appended to the model legend.
        handles, labels = ax.get_legend_handles_labels()
        proxy = Patch(facecolor="grey", alpha=0.25, label=f"{pct} CI (across studies)")
        ax.legend(handles + [proxy], labels + [proxy.get_label()], loc="best")
        fig.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)

    return fig, ax
