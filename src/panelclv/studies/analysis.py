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


# Model families the runner runs per study (one Prediction_i.csv each). Any other
# model_type (e.g. Pareto/NBD) is a single deterministic fit -> only Prediction_1.csv.
_NEURAL_TYPES = {"lstm", "transformer"}


def _is_deterministic_model(model_dir: Path) -> bool:
    """True for a single-fit benchmark (e.g. Pareto/NBD), read from its config.json."""
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.is_file():
        return False
    with open(cfg_path) as f:
        mt = json.load(f).get("model_type")
    return mt is not None and mt not in _NEURAL_TYPES


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
    # Lazy import: plot_utils pulls in torch at module load, so we only pay that
    # cost when actually touching prediction CSVs (keeps discovery torch-free).
    from panelclv.evaluation.plot_utils import load_predictions_from_csv

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
    from panelclv.evaluation.plot_utils import save_predictions_to_csv

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
    save_path: str | Path | None = None,
    title: str | None = None,
    pareto_nbd_benchmark: bool = False,
    pareto_paper_benchmark: bool = False,
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
    save_path
        If given, the figure is written here (PNG).
    title
        Overrides the auto-generated, mode-aware title.
    pareto_nbd_benchmark, pareto_paper_benchmark
        Add the live Pareto/NBD benchmark (MLE and/or hierarchical-Bayes). It is fit
        on the **full** cohort and then restricted to the same customer selection, so
        it is the identical model shown in the full-cohort plot — only the customers
        aggregated differ. Requires ``data`` (rebuilt from ``panel_path`` if needed).
    **plot_kwargs
        Forwarded verbatim to ``plot_weekly_aggregated`` (e.g. ``figsize=...``,
        ``show_ci=...``). Note MC confidence ribbons will not appear for disk-loaded
        predictions: the stored values are already per-customer means, not
        per-simulation arrays.

    Returns ``(fig, ax)``.
    """
    from panelclv.evaluation.plot_utils import plot_weekly_aggregated, pareto_forecast

    root = Path(root)

    if (panel_path is None) == (data is None):
        raise ValueError("pass exactly one of panel_path= or data=")
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
    if pareto_nbd_benchmark:
        predictions_by_model["Pareto/NBD"] = _subset(
            pareto_forecast(data, "mle")["prediction_mean"]
        )
    if pareto_paper_benchmark:
        predictions_by_model["Pareto/NBD (HB)"] = _subset(
            pareto_forecast(data, "paper")["prediction_mean"]
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

    return plot_weekly_aggregated(
        actuals=holdout_curve,
        predictions_by_model=predictions_by_model,
        train_actuals=train_curve,
        title=title,
        save_path=save_path,
        data=data,
        **plot_kwargs,
    )


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


def group_metrics_suite_distribution(
    root: str | Path,
    panel_path: str | Path | None = None,
    *,
    data: dict[str, Any] | None = None,
    groups: Any = ("At Risk", "Opportunity"),
    stats: Any = ("mean", "std", "min", "max"),
    return_per_study: bool = False,
    save_path: str | Path | None = None,
):
    """Distribution of each per-(group, model) metric ACROSS the suite's studies.

    Where :func:`group_metrics_suite_table` scores the across-studies **mean** forecast
    (one value per metric), this scores **every study's** forecast on its own and reports
    the spread of each metric across studies — so a thesis reader sees study-to-study
    variability, not just a point estimate. Two studies with the same mean but very
    different variance look identical in the mean table and different here.

    A deterministic model (Pareto/NBD) has a single study, so its ``std`` is ``NaN`` and
    its ``min``/``max``/``mean`` all equal that one value.

    Parameters mirror :func:`group_metrics_suite_table`, plus:

    stats
        Which summaries to report per metric (any pandas agg names), default
        ``("mean", "std", "min", "max")``. Add e.g. ``"median"`` for robustness.
    return_per_study
        If True, return the raw long DataFrame (one row per group × model × study) instead
        of the aggregated distribution — handy for boxplots / histograms of a metric.

    Returns a DataFrame indexed by ``(group, model)`` with a MultiIndex column
    ``(metric, stat)`` over metrics ``{rmse, mape, bias, bias_percent}`` — unless
    ``return_per_study=True``.
    """
    import pandas as pd  # local: keep the module import cheap
    from panelclv.evaluation import assign_customer_groups, group_metrics_table

    root = Path(root)
    if (panel_path is None) == (data is None):
        raise ValueError("pass exactly one of panel_path= or data=")
    if data is None:
        data = _actuals_from_panel(root, panel_path)

    group_ids = assign_customer_groups(data, groups=groups)
    group_ids = {**group_ids, "Other": _other_ids(data, group_ids)}

    metric_cols = ["rmse", "mape", "bias", "bias_percent"]

    # Score each (model, study) on its own Prediction_i.csv and stack the per-group rows.
    # Reusing group_metrics_table per single-model dict keeps the metric + id-alignment
    # code in one place; iterating each model's own indices handles models that ran a
    # different number of studies (e.g. the deterministic Pareto/NBD has just one).
    per_study: list[pd.DataFrame] = []
    for name, model_dir in _discover_models(root):
        preds = model_dir / "Predictions"
        for i in sorted(_prediction_index(p) for p in preds.glob("Prediction_*.csv")):
            tbl = group_metrics_table(
                data, {name: preds / f"Prediction_{i}.csv"}, group_ids
            ).reset_index()                              # -> group, model, n_customers, metrics
            tbl["study"] = i
            per_study.append(tbl)

    long_df = pd.concat(per_study, ignore_index=True)
    if return_per_study:
        return long_df

    dist = (
        long_df.groupby(["group", "model"])[metric_cols]
        .agg(list(stats))
        .sort_index()
    )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dist.to_csv(save_path)
    return dist


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
