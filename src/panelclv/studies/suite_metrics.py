"""Score a finished suite: whole-cohort metrics, per-group metrics, and their spread.

Reads a suite through `suite_reader` and re-scores every stored forecast with
`compute_forecast_metrics`, the package's single scoring authority — the numbers are
recomputed from the prediction CSVs rather than read back from `results.csv`. Three
public tables:

1. ``study_metrics`` — whole-cohort RMSE / bias / MAPE per model, averaged over the
   suite's independent studies, optionally with their spread.
2. ``compare_study_metrics`` — several suites stacked into one comparison table.
3. ``group_metrics_suite_table`` — the same scoring broken out by customer group.

Spread across studies is reported two ways, and both are the *same* arithmetic:
``t_interval_half_width`` is the package's one Student-t interval, used here for the
``ci_low`` / ``ci_high`` columns, by `suite_plots` for the band it shades, and by the
Pareto grid for its per-cell interval. A plot's band and a table's confidence interval
are the same number because they are the same code.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from panelclv.evaluation import (
    CUSTOMER_GROUPS,
    assign_customer_groups,
    group_metrics_table,
)
from panelclv.models import compute_forecast_metrics

from .suite_reader import (
    _actuals_from_panel,
    _discover_models,
    _is_deterministic_model,
    _prediction_index,
    aggregate_suite_predictions,
    load_model_predictions,
)


# ---------------------------------------------------------------------------
# The one Student-t interval
# ---------------------------------------------------------------------------


def t_interval_half_width(std: Any, n: int, ci: float) -> Any:
    """Half-width of the Student-t confidence interval on a mean: ``t_(1-a/2, n-1)·s/√n``.

    The package's only implementation of that arithmetic. ``std`` is the sample standard
    deviation (``ddof=1``) of ``n`` independent values — a scalar, an array or a pandas
    object, whatever the caller is summarising — and the result has the same shape, so the
    interval is ``mean ± half``. Fewer than two values have no spread, so the half-width is
    ``NaN`` rather than an error: a deterministic benchmark is a single fit, and its row
    belongs in the table without an interval.
    """
    if n < 2:
        return std * np.nan
    return float(stats.t.ppf(0.5 + ci / 2.0, n - 1)) * std / np.sqrt(n)


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
    groups: Any = CUSTOMER_GROUPS,
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
        Which behavioural groups to break the metrics out by. Defaults to every group
        ``assign_customer_groups`` defines and can only narrow that set. An ``"Other"``
        group — every customer matched by none of these — is **always** appended, so
        the table covers the whole cohort.
    save_path
        If given, the table is also written to this CSV.

    Returns the MultiIndex ``(group, model)`` DataFrame from
    :func:`panelclv.evaluation.group_metrics_table`, always including an ``"Other"`` row
    block for the unmatched customers.
    """
    root = Path(root)
    if (panel_path is None) == (data is None):
        raise ValueError("pass exactly one of panel_path= or data=")
    if data is None:
        data = _actuals_from_panel(root, panel_path)

    model_predictions = _suite_prediction_paths(root, study)

    # `with_other=True` adds the catch-all so the table accounts for the whole cohort,
    # not just the flagged segments: every customer matched by none of the requested
    # groups. This makes the group counts sum to N whenever the groups are disjoint
    # (At Risk / Opportunity are), so nothing is silently dropped.
    group_ids = assign_customer_groups(data, groups=groups, with_other=True)

    return group_metrics_table(data, model_predictions, group_ids, save_path=save_path)


# ---------------------------------------------------------------------------
# Whole-cohort study metrics — RMSE / bias / MAPE per model, and their spread
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
        # One row per model: each has its own study count, so the interval is taken
        # per model over that model's three metrics.
        half = pd.DataFrame(
            {m: t_interval_half_width(std.loc[m], int(k), ci) for m, k in n.items()}
        ).T

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
