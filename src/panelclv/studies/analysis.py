"""Read a finished study suite back off disk and visualize its forecasts.

The runner (`panelclv.studies.runner`) leaves a tree like::

    Studies/<name>/
        config.json                     # whole-suite record (carries panel_config)
        <ModelName>/Predictions/Prediction_{i}.csv   # per-study MC-mean forecasts

This module is the read-side companion. It never touches checkpoints, Optuna DBs
or `results.csv`; it only reads `config.json` + the `Prediction_*.csv` files, and
it writes the aggregated per-model CSVs (and, optionally, a figure). Three jobs:

1. ``load_model_predictions`` — one study's forecast, or the per-customer mean
   across every study, for a single model.
2. ``aggregate_suite_predictions`` — write that across-studies mean to
   ``Studies/<name>/aggregated_<ModelName>.csv`` (flat at the suite root, one wide
   CSV per model, same column format as the per-study files).
3. ``plot_suite_forecast`` — the headline: rebuild the training/holdout *actuals*
   from the dataset path (via the persisted ``panel_config``) and overlay every
   model's holdout forecast on one figure.

**Why a dataset path.** The suite archives the *recipe* (`panel_config`) but not
the dataset arrays, so the actual transaction curves are not on disk. Given the
customer-period panel CSV, we rebuild the exact `calibration`/`holdout` tensors
the suite used by re-running `prepare_dataset` with the persisted config — same
code path, same cohort order, so predictions and actuals line up by construction.
`prepare_dataset` needs only numpy + pandas, so this stays torch-free until the
plot itself is drawn.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from panelclv.benchmarks import pareto_forecast
from panelclv.evaluation.plots import plot_weekly_aggregated
from panelclv.predictions import load_predictions_from_csv, save_predictions_to_csv
from panelclv.registry import MODEL_TYPES, is_neural


# ---------------------------------------------------------------------------
# Suite discovery
# ---------------------------------------------------------------------------


def _read_suite_config(root: Path) -> dict[str, Any] | None:
    """Return the suite's ``config.json`` as a dict, or ``None`` if absent."""
    cfg_path = Path(root) / "config.json"
    if not cfg_path.is_file():
        return None
    with open(cfg_path) as f:
        return json.load(f)


def _discover_models(root: Path) -> list[tuple[str, Path]]:
    """List ``(model_name, model_dir)`` for every model with predictions.

    Order comes from ``config.json``'s ``models`` list (so the plot legend is
    reproducible, not filesystem-order); models whose ``Predictions/`` folder is
    missing or empty are skipped. If ``config.json`` has no model list (a hand-made
    or partial tree), fall back to scanning for any subdirectory that contains a
    non-empty ``Predictions/`` folder, alphabetically.
    """
    root = Path(root)
    cfg = _read_suite_config(root)

    names: list[str]
    if cfg and cfg.get("models"):
        names = [m["name"] for m in cfg["models"]]
    else:
        names = sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and (p / "Predictions").is_dir()
        )

    found: list[tuple[str, Path]] = []
    for name in names:
        model_dir = root / name
        preds = model_dir / "Predictions"
        if preds.is_dir() and any(preds.glob("Prediction_*.csv")):
            found.append((name, model_dir))
    if not found:
        raise FileNotFoundError(
            f"no models with a non-empty Predictions/ folder under {root}"
        )
    return found


def _id_col(root: Path) -> str:
    """The customer-id column name for saved CSVs, read from the suite config.

    Falls back to ``save_predictions_to_csv``'s own default so a config-less tree
    still works.
    """
    cfg = _read_suite_config(root) or {}
    pc = cfg.get("panel_config") or {}
    summary = cfg.get("data_summary") or {}
    return pc.get("id_col") or summary.get("id_col") or "customer_id"


# ---------------------------------------------------------------------------
# Loading / aggregating predictions
# ---------------------------------------------------------------------------


def _prediction_index(path: Path) -> int:
    """Sort key for ``Prediction_{i}.csv`` — the integer ``i`` (10 sorts after 2)."""
    m = re.search(r"Prediction_(\d+)\.csv$", path.name)
    return int(m.group(1)) if m else -1


# Which families the runner runs per study (one Prediction_i.csv each) is read off the
# registry rather than restated here: a local copy of the neural list is exactly what
# drifted once, and a stale one misreads a whole model's archive as a single
# deterministic fit (ADR-0006).


def _is_deterministic_model(model_dir: Path) -> bool:
    """True for a single-fit benchmark (e.g. Pareto/NBD), read from its config.json."""
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.is_file():
        return False
    with open(cfg_path) as f:
        mt = json.load(f).get("model_type")
    if mt is None:
        return False
    # An archive may name a model type this build no longer registers; the reader's
    # job is to locate files, so an unknown type reads as a single fit rather than
    # refusing to read the folder at all.
    runs_once_per_study = mt in MODEL_TYPES and is_neural(mt)
    return not runs_once_per_study


def load_model_predictions(
    model_dir: str | Path, study: int | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load one model's holdout forecast as ``(values (N, T_HOLD), ids)``.

    Parameters
    ----------
    model_dir
        ``Studies/<name>/<ModelName>`` — the folder holding ``Predictions/``.
    study
        ``int`` → load exactly that study's ``Prediction_{study}.csv``.
        ``None`` → average across every ``Prediction_*.csv`` (the per-customer,
        per-timestep mean over studies). A single-study model averages to itself.

    In average mode all studies must describe the same cohort in the same order
    and the same horizon; a mismatch raises ``ValueError`` naming the offenders,
    because a silent misalignment would corrupt the mean.
    """
    preds_dir = Path(model_dir) / "Predictions"

    if study is not None:
        # A deterministic benchmark (Pareto/NBD) has no per-study variation, so the
        # runner writes only Prediction_1.csv. Always load that, whatever study index
        # was requested; every other model honors the requested index as before.
        if _is_deterministic_model(preds_dir.parent):
            study = 1
        path = preds_dir / f"Prediction_{study}.csv"
        if not path.is_file():
            available = sorted(
                p.name for p in preds_dir.glob("Prediction_*.csv")
            )
            raise FileNotFoundError(
                f"{path} not found; available predictions: {available or 'none'}"
            )
        return load_predictions_from_csv(path)

    paths = sorted(preds_dir.glob("Prediction_*.csv"), key=_prediction_index)
    if not paths:
        raise FileNotFoundError(f"no Prediction_*.csv under {preds_dir}")

    values_stack: list[np.ndarray] = []
    ref_ids: np.ndarray | None = None
    ref_path: Path | None = None
    for path in paths:
        vals, ids = load_predictions_from_csv(path)
        if ref_ids is None:
            ref_ids, ref_path = ids, path
        else:
            if vals.shape != values_stack[0].shape:
                raise ValueError(
                    f"prediction shape mismatch: {ref_path.name} is "
                    f"{values_stack[0].shape} but {path.name} is {vals.shape} — "
                    f"cannot average studies over different cohorts/horizons"
                )
            if ref_ids is not None and ids is not None and not np.array_equal(ids, ref_ids):
                raise ValueError(
                    f"customer ids differ between {ref_path.name} and {path.name} — "
                    f"studies must forecast the same customers in the same order to average"
                )
        values_stack.append(vals)

    # (n_studies, N, T) -> mean over studies -> (N, T).
    mean_values = np.stack(values_stack, axis=0).mean(axis=0)
    return mean_values, ref_ids


def aggregate_suite_predictions(root: str | Path) -> list[Path]:
    """Write each model's across-studies mean forecast to the suite root.

    For every model, average its ``Prediction_*.csv`` files (per customer, per
    timestep) and write the result as ``Studies/<name>/aggregated_<ModelName>.csv``
    — flat at the suite root, next to the model folders, one wide CSV per model in
    the same ``id_col + week_0..week_{T-1}`` format as the per-study files (so it
    round-trips through ``load_predictions_from_csv``). Overwrites in place on
    re-run (the aggregate is a pure function of the predictions on disk). Returns
    the list of files written.
    """
    root = Path(root)
    id_col = _id_col(root)
    written: list[Path] = []
    for name, model_dir in _discover_models(root):
        mean_values, ids = load_model_predictions(model_dir, study=None)
        out = root / f"aggregated_{name}.csv"
        save_predictions_to_csv(mean_values, out, customer_ids=ids, id_col=id_col)
        written.append(out)
    return written


# ---------------------------------------------------------------------------
# Actuals — rebuilt from the dataset path via the persisted recipe
# ---------------------------------------------------------------------------


def _actuals_from_panel(root: Path, panel_path: str | Path) -> dict[str, Any]:
    """Rebuild the ``prepare_dataset`` dict from the panel CSV + persisted config.

    Reads ``config.json``'s ``panel_config`` (the recipe the suite ran with),
    reconstructs the ``PanelConfig``, and re-runs ``prepare_dataset`` on the panel
    at ``panel_path``. Because it is the same function and config the suite used,
    the returned ``calibration``/``holdout`` tensors, ``target_idx`` and cohort
    ``ids`` match the run's exactly — so the actuals align with the stored
    predictions. numpy + pandas only (no torch).
    """
    import pandas as pd  # local: keep module import cheap
    from panelclv.configs.panel_config import PanelConfig
    from panelclv.data_preparation.dynamic_panel_dataset import prepare_dataset

    cfg = _read_suite_config(root)
    if not cfg or cfg.get("panel_config") is None:
        raise ValueError(
            f"{root}/config.json has no panel_config, so the actuals cannot be "
            f"rebuilt from a panel path (the suite predates the carried config). "
            f"Pass data=<prepare_dataset output> to plot_suite_forecast instead."
        )
    panel_path = Path(panel_path)
    if not panel_path.is_file():
        raise FileNotFoundError(f"panel not found: {panel_path}")

    panel_config = PanelConfig.from_dict(cfg["panel_config"])
    panel = pd.read_csv(panel_path)
    return prepare_dataset(panel, panel_config, verbose=False)


def _aggregate_actuals(
    data: dict[str, Any], row_idx: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Sum the target channel across customers → (train_curve, holdout_curve).

    ``calibration``/``holdout`` are ``(N, T, F)``; ``target_idx`` is the count
    channel. Summing over customers gives the weekly-aggregate curves the plot
    overlays: ``(T_CAL,)`` for training and ``(T_HOLD,)`` for holdout. If ``row_idx``
    is given, only those customer rows are summed (a group / customer-id subset).
    """
    ti = int(data["target_idx"])
    calibration = np.asarray(data["calibration"], dtype=np.float64)
    holdout = np.asarray(data["holdout"], dtype=np.float64)
    if row_idx is not None:
        calibration = calibration[row_idx]
        holdout = holdout[row_idx]
    train_curve = calibration[:, :, ti].sum(axis=0)
    holdout_curve = holdout[:, :, ti].sum(axis=0)
    return train_curve, holdout_curve


# ---------------------------------------------------------------------------
# Customer selection — restrict a plot to a behavioural group or explicit ids
# ---------------------------------------------------------------------------

# "Other" is derived; these are the predicate groups assign_customer_groups defines.
_CANONICAL_GROUPS = ("At Risk", "Opportunity")


def _other_ids(data: dict[str, Any], group_ids: dict[str, np.ndarray]) -> np.ndarray:
    """Customer ids matched by none of ``group_ids`` (the 'Other' catch-all).

    Compared as ``str`` (matching how the metric helpers key ids) and returned in the
    cohort's own order, so the group counts sum to N when the groups are disjoint.
    """
    assigned = {str(cid) for ids in group_ids.values() for cid in ids}
    all_ids = np.asarray(data["ids"])
    return np.asarray([cid for cid in all_ids if str(cid) not in assigned])


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
                      ``{"At Risk", "Opportunity", "Other"}`` (membership is the same
                      canonical grouping the metrics table uses; "Other" = in neither
                      canonical group).
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
        from panelclv.evaluation import assign_customer_groups

        names = [group] if isinstance(group, str) else list(group)
        # Canonical grouping + derived "Other", so group="Other" is well-defined.
        grouping = assign_customer_groups(data, groups=_CANONICAL_GROUPS)
        grouping["Other"] = _other_ids(data, grouping)

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
    from scipy import stats

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
    half = stats.t.ppf(1 - (1 - ci) / 2, df=n - 1) * std / np.sqrt(n)
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
        Restrict the plot to a behavioural group (or list of groups) from
        ``{"At Risk", "Opportunity", "Other"}`` — the same membership the metrics
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
        from matplotlib.patches import Patch

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


# ---------------------------------------------------------------------------
# Per-customer-group metrics — the segment-analysis companion to the plot
# ---------------------------------------------------------------------------


def _suite_prediction_paths(root: Path, study: int | None) -> dict[str, Path]:
    """Map each model to the prediction CSV to score, mirroring the plot's choice.

    ``study=None`` → the across-studies mean at the suite root
    (``aggregated_<Model>.csv``), (re)written first so the file is current.
    ``study=<i>`` → that study's ``Prediction_{i}.csv`` per model, except a
    deterministic benchmark (Pareto/NBD), which has only ``Prediction_1.csv`` and so
    always resolves to it — the same rule ``load_model_predictions`` applies.
    """
    root = Path(root)
    if study is None:
        # Ensure the aggregated CSVs exist / are up to date, then point at them.
        aggregate_suite_predictions(root)
        return {name: root / f"aggregated_{name}.csv" for name, _ in _discover_models(root)}

    paths: dict[str, Path] = {}
    for name, model_dir in _discover_models(root):
        idx = 1 if _is_deterministic_model(model_dir) else study
        paths[name] = model_dir / "Predictions" / f"Prediction_{idx}.csv"
    return paths


def group_metrics_suite_table(
    root: str | Path,
    panel_path: str | Path | None = None,
    *,
    study: int | None = None,
    data: dict[str, Any] | None = None,
    groups: Any = ("At Risk", "Opportunity"),
    save_path: str | Path | None = None,
):
    """Per-(group, model) RMSE / MAPE / bias for a finished suite, from ROOT + panel.

    The segment-analysis companion to :func:`plot_suite_forecast`: give it the suite
    folder and the panel it was built from, and it rebuilds the exact dataset (via the
    persisted ``panel_config``), assigns each customer to a behavioural group, and
    scores every model's stored forecast per group.

    Parameters mirror :func:`plot_suite_forecast`:

    root
        The suite folder ``Studies/<name>/``.
    panel_path
        The customer-period panel CSV the suite was built from (rebuilds the actuals
        and the cohort). Required unless ``data`` is given.
    study
        ``None`` (default) → score the across-studies mean per model; ``int`` → score
        that single study (Pareto/NBD always uses its sole ``Prediction_1.csv``).
    data
        In-session escape hatch: pass a live ``prepare_dataset`` dict to skip the panel
        rebuild. Exactly one of ``panel_path`` / ``data`` must be supplied.
    groups
        Which behavioural groups to break the metrics out by (see
        ``assign_customer_groups``). An ``"Other"`` group — every customer matched by
        none of these — is **always** appended, so the table covers the whole cohort.
    save_path
        If given, the table is also written to this CSV.

    Returns the MultiIndex ``(group, model)`` DataFrame from
    :func:`panelclv.evaluation.group_metrics_table`, always including an ``"Other"`` row
    block for the unmatched customers.
    """
    # Lazy import: segment_analysis pulls in torch via the metric helpers, so we only
    # pay that cost when actually computing metrics (keeps the module import cheap).
    from panelclv.evaluation import assign_customer_groups, group_metrics_table

    root = Path(root)
    if (panel_path is None) == (data is None):
        raise ValueError("pass exactly one of panel_path= or data=")
    if data is None:
        data = _actuals_from_panel(root, panel_path)

    model_predictions = _suite_prediction_paths(root, study)
    group_ids = assign_customer_groups(data, groups=groups)

    # Always add an "Other" catch-all so the table accounts for the whole cohort, not
    # just the flagged segments: every customer matched by none of the requested
    # groups. This makes the group counts sum to N whenever the groups are disjoint
    # (At Risk / Opportunity are), so nothing is silently dropped.
    group_ids = {**group_ids, "Other": _other_ids(data, group_ids)}

    return group_metrics_table(data, model_predictions, group_ids, save_path=save_path)


# ---------------------------------------------------------------------------
# Whole-cohort study metrics — the headline RMSE / bias / MAPE per model,
# optionally with a confidence interval across the suite's independent studies
# ---------------------------------------------------------------------------


# The three whole-cohort metrics `compute_forecast_metrics` returns — the exact
# numbers the runner writes to results.csv, per study.
_STUDY_METRIC_COLS = ["rmse", "bias_percent", "mape_aggregate"]


def study_metrics(
    root: str | Path,
    panel_path: str | Path,
    confidence_interval: bool = False,
    standard_deviation: bool = False,
    ci: float = 0.95,
    display: bool = False,
    decimals: int = 3,
):
    """Whole-cohort forecast metrics per model for a finished suite, from ROOT + panel.

    Give it the suite folder and the dataset (panel) CSV it was built from; it rebuilds
    the exact dataset (via the persisted ``panel_config``), then scores **every** study's
    stored forecast with the same function the runner used (``compute_forecast_metrics``:
    whole-cohort RMSE, aggregate %-bias, aggregate MAPE). Each neural model ran
    ``n_studies_per_model`` independent studies (own seed), so the summary is the mean
    over those studies; a deterministic benchmark (Pareto/NBD) has a single study, which
    averages to itself.

    ``standard_deviation`` and ``confidence_interval`` describe the SAME across-studies
    spread but answer different questions, and are independent toggles you can combine:

    - ``standard_deviation`` → ``std``, the study-to-study spread. This is "where a single
      re-run lands" — the usual ``mean ± SD`` reported in ML papers over N seeds.
    - ``confidence_interval`` → a Student-t interval on the **mean**,
      ``mean ± t_{1-a/2, n-1} · s/√n`` (``ci_low`` / ``ci_high``). This is "how precisely
      the *expected* metric is estimated"; it is ~√n narrower than the SD.

    Parameters
    ----------
    root
        The suite folder ``Studies/<name>/``.
    panel_path
        The customer-period panel (dataset) CSV the suite was built from — used to rebuild
        the actuals and the cohort.
    confidence_interval
        Add the across-studies Student-t interval on the mean (``ci_low`` / ``ci_high``).
        A single-study model (Pareto/NBD) has no interval (``NaN``).
    standard_deviation
        Add the across-studies sample standard deviation (``std``, ``ddof=1``). ``NaN`` for
        a single-study model.
    ci
        Confidence level for the interval (default ``0.95``). Only used when
        ``confidence_interval=True``.
    display
        Render a paper-ready ``mean ± …`` table of strings (one column per metric) instead
        of the numeric columns. The ``±`` term is the SD when ``standard_deviation=True``
        and the CI half-width (``t·s/√n``, symmetric) when ``confidence_interval=True``;
        with both, each term is shown and labelled ``(SD)`` / ``(CI)``. Requires at least
        one of the two flags (there is no ``±`` term otherwise). A single-study model shows
        just its mean. Numbers are rounded to ``decimals``.
    decimals
        Decimal places for the ``display`` strings (default ``3``). Ignored otherwise.

    Returns a ``pandas.DataFrame`` indexed by model. With neither spread flag (and no
    ``display``): flat columns ``[rmse, bias_percent, mape_aggregate, n_studies]``
    (the means). With a spread flag and ``display=False``: a MultiIndex column
    ``(metric, stat)`` whose stats are ``mean`` plus whichever of ``std`` / ``ci_low`` /
    ``ci_high`` were requested, then ``n``. With ``display=True``: one string column per
    metric, ``"mean ± …"``.
    """
    root = Path(root)
    data = _actuals_from_panel(root, panel_path)
    return _study_metrics_from_data(
        data, root,
        confidence_interval=confidence_interval,
        standard_deviation=standard_deviation,
        ci=ci, display=display, decimals=decimals,
    )


def _study_metrics_from_data(
    data: dict[str, Any],
    root: Path,
    *,
    confidence_interval: bool,
    standard_deviation: bool,
    ci: float,
    display: bool,
    decimals: int,
):
    """Core of :func:`study_metrics` given an already-rebuilt ``prepare_dataset`` dict.

    Split out so :func:`compare_study_metrics` can reuse the actuals it rebuilt (for its
    holdout/cohort consistency check) without rebuilding them a second time. See
    :func:`study_metrics` for the parameter semantics and return shapes.
    """
    # Lazy import: compute_forecast_metrics pulls in torch via the models package, so
    # we only pay that cost when actually scoring (keeps the module import cheap).
    import pandas as pd
    from panelclv.models import compute_forecast_metrics

    # Holdout actuals as (N, T_HOLD): the target channel of the (N, T, F) holdout tensor,
    # in the cohort's own order — the same order the prediction CSVs are saved in.
    actual = np.asarray(data["holdout"], dtype=np.float64)[:, :, int(data["target_idx"])]
    ref_ids = np.asarray(data["ids"])

    # Score each (model, study) forecast on its own — one row per study. Alignment is
    # asserted so prediction row i and actual row i are the same customer.
    rows: list[dict[str, Any]] = []
    for name, model_dir in _discover_models(root):
        preds = model_dir / "Predictions"
        for i in sorted(_prediction_index(p) for p in preds.glob("Prediction_*.csv")):
            values, ids = load_model_predictions(model_dir, study=i)
            if ids is not None and not np.array_equal(np.asarray(ids), ref_ids):
                raise ValueError(
                    f"model {name!r} study {i}: prediction customer ids do not match the "
                    f"rebuilt cohort — is panel_path the panel this suite was built from?"
                )
            if values.shape != actual.shape:
                raise ValueError(
                    f"model {name!r} study {i}: forecast shape {values.shape} != actual "
                    f"{actual.shape}"
                )
            rows.append(
                {"model": name, "study": i, **compute_forecast_metrics(actual, values)}
            )

    per_study = pd.DataFrame(rows)
    grouped = per_study.groupby("model", sort=False)[_STUDY_METRIC_COLS]

    n = grouped.size()
    mean = grouped.mean()

    # Plain means (flat table) when nothing extra was asked for.
    if not (confidence_interval or standard_deviation or display):
        table = mean.copy()
        table["n_studies"] = n
        return table

    std = grouped.std(ddof=1)                        # sample spread across studies; NaN if n<2

    # CI half-width (t · s/√n), symmetric, per model×metric. Only needed for the CI columns
    # or the ± display; NaN where a model has < 2 studies.
    half = None
    if confidence_interval:
        from scipy import stats  # local: SciPy only needed for the t-interval

        alpha = 1.0 - ci
        sem = std.div(np.sqrt(n), axis=0)
        tcrit = pd.Series(
            {m: stats.t.ppf(1 - alpha / 2, df=int(k) - 1) if k > 1 else np.nan
             for m, k in n.items()}
        )
        half = sem.mul(tcrit, axis=0)

    # ---- paper-ready "mean ± …" strings -----------------------------------------------
    if display:
        if not (confidence_interval or standard_deviation):
            raise ValueError(
                "display=True needs standard_deviation=True and/or confidence_interval=True "
                "to have a ± term to show."
            )

        def cell(model: Any, metric: str) -> str:
            mu = f"{mean.loc[model, metric]:.{decimals}f}"
            terms: list[str] = []
            if standard_deviation:
                s = std.loc[model, metric]
                if not pd.isna(s):  # single-study models have no spread
                    label = " (SD)" if confidence_interval else ""
                    terms.append(f"± {s:.{decimals}f}{label}")
            if confidence_interval:
                h = half.loc[model, metric]
                if not pd.isna(h):
                    label = " (CI)" if standard_deviation else ""
                    terms.append(f"± {h:.{decimals}f}{label}")
            return f"{mu} {' '.join(terms)}".rstrip()

        disp = pd.DataFrame(
            {metric: {m: cell(m, metric) for m in mean.index} for metric in _STUDY_METRIC_COLS}
        )
        disp.index.name = mean.index.name  # "model"
        return disp

    # ---- numeric (metric, stat) table -------------------------------------------------
    # Assemble only the requested stats, in a stable order: mean, [std], [ci], n.
    parts: dict[str, Any] = {"mean": mean}
    if standard_deviation:
        parts["std"] = std
    if confidence_interval:
        parts["ci_low"] = mean.sub(half)
        parts["ci_high"] = mean.add(half)
    # broadcast the per-model study count across the metric columns
    parts["n"] = pd.DataFrame({c: n for c in _STUDY_METRIC_COLS})

    # keys=... fixes the stat order; swaplevel puts it as (metric, stat) so each metric's
    # stats sit together, and we keep the metric order from _STUDY_METRIC_COLS.
    table = (
        pd.concat(parts.values(), axis=1, keys=parts.keys())
        .swaplevel(axis=1)
        .reindex(columns=_STUDY_METRIC_COLS, level=0)
    )
    table.columns.names = ["metric", "stat"]
    return table


def compare_study_metrics(
    suites: dict[str, str | Path],
    panel_path: str | Path,
    confidence_interval: bool = False,
    standard_deviation: bool = False,
    ci: float = 0.95,
    display: bool = False,
    decimals: int = 3,
):
    """Stack :func:`study_metrics` for several suites into one comparison table.

    Compares up to four finished suites — e.g. the same models under different validation
    windows, losses or feature sets — scored against one shared dataset. Each ``ROOT`` is
    summarised exactly as :func:`study_metrics` would (same ``confidence_interval`` /
    ``standard_deviation`` / ``ci`` / ``display`` / ``decimals`` options, all forwarded),
    then the per-suite tables are stacked with a ``(model, suite)`` row index so each
    model's variants sit together.

    Parameters
    ----------
    suites
        ``{name: ROOT}`` — a display name mapped to each suite folder, at most four. The
        names label the rows and set the suite order; dict keys keep them unique.
    panel_path
        The one dataset (panel) CSV shared by every suite. Note each suite still rebuilds
        its *own* actuals from its *own* persisted ``panel_config``, so if two suites used
        different holdout windows or cohorts their metrics are not comparable — a warning
        is emitted when the rebuilt holdout length or cohort size differs across suites.
    confidence_interval, standard_deviation, ci, display, decimals
        Forwarded verbatim to :func:`study_metrics`; see it for their meaning and the
        resulting column shapes.

    Returns a ``pandas.DataFrame`` with the same columns :func:`study_metrics` produces for
    the chosen options, indexed by ``(model, suite)`` — models in discovery order, suites
    in the order given.
    """
    import warnings

    import pandas as pd

    if not isinstance(suites, dict):
        raise TypeError("suites must be a dict {name: ROOT}")
    if not 1 <= len(suites) <= 4:
        raise ValueError(f"pass between 1 and 4 named suites, got {len(suites)}")

    tables: dict[str, Any] = {}
    holdout_lengths: dict[str, int] = {}
    cohort_sizes: dict[str, int] = {}
    for name, root in suites.items():
        root = Path(root)
        data = _actuals_from_panel(root, panel_path)          # rebuilt once, reused below
        holdout_lengths[name] = int(np.asarray(data["holdout"]).shape[1])
        cohort_sizes[name] = int(len(np.asarray(data["ids"])))
        tables[name] = _study_metrics_from_data(
            data, root,
            confidence_interval=confidence_interval,
            standard_deviation=standard_deviation,
            ci=ci, display=display, decimals=decimals,
        )

    # Metrics are only comparable across suites that share a holdout window and cohort;
    # each ROOT rebuilds its own actuals, so flag a mismatch rather than silently combining
    # apples and oranges. Warn (not raise) so an intentional cross-window view is allowed.
    for label, sizes in (("holdout length", holdout_lengths), ("cohort size", cohort_sizes)):
        if len(set(sizes.values())) > 1:
            detail = ", ".join(f"{k}: {v}" for k, v in sizes.items())
            warnings.warn(
                f"suites differ in {label} ({detail}) — metrics may not be comparable.",
                stacklevel=2,
            )

    # Stack per-suite tables → (suite, model), then flip to (model, suite) so each model's
    # variants group together. Reindex explicitly to keep models in discovery order and
    # suites in the given order (sort_index would alphabetise both and lose that).
    combined = pd.concat(tables.values(), keys=tables.keys(), names=["suite", "model"])
    combined = combined.swaplevel(0, 1)
    model_order = list(dict.fromkeys(combined.index.get_level_values("model")))
    ordered = [(m, s) for m in model_order for s in suites if (m, s) in combined.index]
    return combined.reindex(ordered)


# ---------------------------------------------------------------------------
# Dataset description — global characteristics over the selected period
# ---------------------------------------------------------------------------


def describe_dataset(data: dict[str, Any], *, name: str | None = None):
    """Global characteristics of a prepared dataset over its selected period.

    The ``prepare_dataset`` dict already encodes the period chosen in the study's
    ``PanelConfig`` (training / validation / holdout dates): ``calibration`` covers
    ``[training_start, training_end]`` and ``holdout`` covers ``[holdout_start,
    holdout_end]``. Counts are read from the target channel of those dense
    ``(N, T, F)`` tensors — the same source the metrics table and plots use, so the
    numbers are cohort-aligned with the rest of the analysis. "Overall" means
    calibration + holdout combined.

    Note the temporal split (see CLAUDE.md): the validation window is the **tail** of
    the calibration window, so ``calibration_length`` already includes it and
    ``training_length`` is the pre-validation prefix (``calibration_length -
    validation_length``). Counts reflect the target as prepared; if the config set
    ``clip_target_upper``, per-period counts are capped at that value.

    Parameters
    ----------
    data
        A ``prepare_dataset`` output.
    name
        Name for the returned Series (handy when concatenating several datasets into
        one comparison table via ``pd.concat([...], axis=1)``).

    Returns a ``pandas.Series`` of labelled characteristics.
    """
    import pandas as pd  # local: keep the module import cheap / torch-free

    ti = int(data["target_idx"])
    ids = np.asarray(data["ids"])
    N = len(ids)
    calibration = np.asarray(data["calibration"], dtype=np.float64)
    holdout = np.asarray(data["holdout"], dtype=np.float64)

    T_CAL = int(data.get("T_CAL", calibration.shape[1]))
    T_HOLD = int(data.get("T_HOLD", holdout.shape[1]))
    n_val = int(data.get("n_val_periods", 0))
    # Pre-validation prefix length: weights train on [0, val_start_idx).
    val_start = int(data.get("val_start_idx", T_CAL - n_val))

    # Per (customer, period) integer counts over each window. rint guards the rare
    # float-dtype tensor so exact-count comparisons (== 1) are safe.
    calib_cells = np.rint(calibration[:, :, ti]).astype(np.int64)   # (N, T_CAL)
    hold_cells = np.rint(holdout[:, :, ti]).astype(np.int64)        # (N, T_HOLD)
    calib_counts = calib_cells.sum(axis=1)                          # (N,) per customer
    hold_counts = hold_cells.sum(axis=1)
    total_counts = calib_counts + hold_counts                      # "overall" per customer

    def pct(count: Any) -> float:
        """Share of the cohort, as a percentage (0-100)."""
        return round(100.0 * float(count) / N, 2) if N else float("nan")

    # Prefer the carried PanelConfig for the period dates; fall back to loose data keys.
    pc = data.get("panel_config")

    def cfg(attr: str) -> Any:
        val = getattr(pc, attr, None) if pc is not None else None
        return val if val is not None else data.get(attr)

    zero_cells = np.concatenate([calib_cells, hold_cells], axis=1) == 0

    metrics: dict[str, Any] = {
        # --- the selected period (straight from the study's PanelConfig) ---
        "frequency": data.get("frequency"),
        "training_start": cfg("training_start"),
        "training_end": cfg("training_end"),
        "validation_start": cfg("validation_start"),
        "holdout_start": cfg("holdout_start"),
        "holdout_end": cfg("holdout_end"),
        # --- window sizes (customers / periods) ---
        "cohort_size": N,
        "calibration_length": T_CAL,          # incl. the validation tail
        "training_length": val_start,         # pre-validation prefix
        "validation_length": n_val,
        "holdout_length": T_HOLD,
        # --- transaction volume ---
        "total_transactions_overall": int(total_counts.sum()),
        "total_transactions_calibration": int(calib_counts.sum()),
        "total_transactions_holdout": int(hold_counts.sum()),
        "avg_transactions_per_customer": round(float(total_counts.mean()), 3),
        "median_transactions_per_customer": float(np.median(total_counts)),
        "avg_transactions_per_customer_calibration": round(float(calib_counts.mean()), 3),
        "avg_transactions_per_customer_holdout": round(float(hold_counts.mean()), 3),
        "transactions_per_customer_per_period": round(
            float(total_counts.sum() / (N * (T_CAL + T_HOLD))), 4
        ),
        # --- per-customer activity distribution (overall) ---
        "customers_with_0_transactions": int((total_counts == 0).sum()),
        "pct_customers_with_0_transactions": pct((total_counts == 0).sum()),
        "customers_with_1_transaction": int((total_counts == 1).sum()),
        "customers_with_lt5_transactions": int((total_counts < 5).sum()),
        "pct_customers_with_lt5_transactions": pct((total_counts < 5).sum()),
        "pct_repeat_customers_ge2": pct((total_counts >= 2).sum()),
        "p90_transactions_per_customer": float(np.percentile(total_counts, 90)),
        "p99_transactions_per_customer": float(np.percentile(total_counts, 99)),
        "max_transactions_per_customer": int(total_counts.max()),
        # --- churn / sparsity ---
        "pct_customers_inactive_in_holdout": pct((hold_counts == 0).sum()),
        "pct_customers_inactive_in_calibration": pct((calib_counts == 0).sum()),
        "panel_sparsity_pct_zero_cells": round(100.0 * float(zero_cells.mean()), 2),
    }
    return pd.Series(metrics, name=name or "dataset")


def describe_suite_dataset(
    root: str | Path,
    panel_path: str | Path | None = None,
    *,
    data: dict[str, Any] | None = None,
    name: str | None = None,
):
    """Describe a finished suite's dataset from ROOT + panel — companion to the plot.

    Rebuilds the exact dataset the suite ran on (via the persisted ``panel_config``)
    and returns :func:`describe_dataset` for it, so the characteristics match the
    period the study actually used. Exactly one of ``panel_path`` / ``data`` must be
    given (``data`` skips the rebuild). ``name`` defaults to the suite folder name.
    """
    root = Path(root)
    if (panel_path is None) == (data is None):
        raise ValueError("pass exactly one of panel_path= or data=")
    if data is None:
        data = _actuals_from_panel(root, panel_path)
    return describe_dataset(data, name=name or root.name)
