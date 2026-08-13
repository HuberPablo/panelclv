"""Cross-grid analysis of a Pareto/NBD generation study trained per-dataset.

``data_preparation.pareto_nbd_simulation.generate_pnbd_study`` lays out a grid of
synthetic datasets — ``mean_transaction_rate`` x ``churn_rate``, with
``n_datasets`` replicate panels per cell. Training one study suite per dataset
(``run_study_suite`` with ``n_studies_per_model=1``) produces a parallel tree

    <train_base>/<combo>__<dataset>/results.csv        # one row per (model, study)

This module joins the two halves: it reads every dataset's model metrics, tags
each with the dataset's grid coordinates, averages the replicate datasets within
each ``(rate, churn)`` **cell** with a confidence interval, and plots how each
model performs across the grid — so you can compare models per dataset and spot
performance patterns (e.g. does the LSTM's error track the Pareto/NBD benchmark's
as churn rises, or diverge in the sparse low-rate corner?).

**Everything here reads only what a trained suite stored**, so it runs on any grid of
suites — the metrics come out of ``results.csv`` and the grid coordinates out of the
generation study's manifest. Scoring a forecast against the *generator's* latent truth
(the true death week, the true seasonal curve) is the sibling module,
``synthetic_grid``, and the split is the point: which of the two a caller imports says
whether the analysis could ever run on a real panel. The dependency runs one way —
that module reads this one's grid axes, never the reverse.

Typical use::

    from panelclv.studies import (
        collect_grid_results, cell_summary, compare_models_table,
        plot_pattern, plot_diff_grid,
    )

    results = collect_grid_results(dataset_dir, train_base)  # long: one row / (model, dataset)
    summary = cell_summary(results)                          # mean + 95% CI per (model, cell)
    compare_models_table(summary, "mape")                    # side-by-side per cell
    plot_pattern(summary, "mape")                            # metric vs churn, panel per rate
    plot_diff_grid(results, "mape")                          # LSTM - ParetoNBD heatmap
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from panelclv.data_preparation.pareto_nbd_simulation import list_pnbd_datasets

from .suite_metrics import t_interval_half_width

# Metric name (as we expose it) -> column name in each suite's results.csv. The
# aggregate MAPE is stored under the scoring authority's own key; we surface it as
# "mape". Suites written before that key was renamed carry `mape_aggregate_style`
# instead, and this reader no longer finds it: staying readable by older archives was
# deliberately dropped as a requirement (ADR-0003 records the same trade). Pass
# `metrics=` to ask only for the columns such a suite does carry.
_METRIC_SOURCE = {
    "rmse": "rmse",
    "bias_percent": "bias_percent",
    "mape": "mape_aggregate",
}
DEFAULT_METRICS = ("rmse", "bias_percent", "mape")

# The two grid axes carried through from the generation study.
_AXES = ["mean_transaction_rate", "churn_rate"]


def _resolve_grid(
    dataset_dir: str | Path, train_base: str | Path | None
) -> tuple[Path, Path]:
    """The dataset directory and the trained suites beside it, as paths.

    ``train_base`` defaults to ``Studies/<dataset_dir name>`` — the convention the
    training loop uses — resolved relative to the current working directory. Written
    once here and reused by ``synthetic_grid``, so the two halves of the grid surface
    cannot come to disagree about where a grid's suites live.
    """
    dataset_dir = Path(dataset_dir)
    if train_base is None:
        train_base = Path("Studies") / dataset_dir.name
    return dataset_dir, Path(train_base)


def _suite_dir(train_base: Path, combo: str, dataset: str) -> Path:
    """The ordinary study suite trained on one generated dataset.

    ``<train_base>/<combo>__<dataset>/`` — the one place the join between the
    generation tree and the trained tree is spelled.
    """
    return train_base / f"{combo}__{dataset}"


# ---------------------------------------------------------------------------
# 1. Collect — join per-dataset model metrics with their grid coordinates
# ---------------------------------------------------------------------------


def collect_grid_results(
    dataset_dir: str | Path,
    train_base: str | Path | None = None,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Read every trained dataset's metrics into one long table tagged by grid cell.

    Parameters
    ----------
    dataset_dir
        The dataset directory (what ``generate_pnbd_study`` returned) — used
        to enumerate the datasets and their ``(rate, churn)`` coordinates.
    train_base
        The folder holding the trained suites (``<combo>__<dataset>/`` subfolders).
        Defaults to ``Studies/<dataset_dir name>``, the convention the training loop uses.
    metrics
        Which metrics to pull from each suite's ``results.csv``.

    Returns
    -------
    DataFrame with one row per (model, dataset): the grid coordinates
    (``mean_transaction_rate``, ``churn_rate``), the ``combo`` / ``dataset`` labels,
    the ``model`` name, and one column per requested metric. Datasets with no
    ``results.csv`` yet (not trained) are skipped.
    """
    dataset_dir, train_base = _resolve_grid(dataset_dir, train_base)

    unknown = [m for m in metrics if m not in _METRIC_SOURCE]
    if unknown:
        raise ValueError(f"unknown metrics {unknown}; known: {list(_METRIC_SOURCE)}")

    grid = list_pnbd_datasets(dataset_dir)
    rows: list[dict] = []
    for g in grid.itertuples(index=False):
        res_path = _suite_dir(train_base, g.combo, g.dataset) / "results.csv"
        if not res_path.exists():
            continue                                 # dataset not trained yet — skip
        res = pd.read_csv(res_path)                  # one row per (model, study)
        for r in res.itertuples(index=False):
            row = {
                "mean_transaction_rate": g.mean_transaction_rate,
                "churn_rate": g.churn_rate,
                "combo": g.combo,
                "dataset": g.dataset,
                "model": r.model,
            }
            for m in metrics:
                row[m] = getattr(r, _METRIC_SOURCE[m])
            rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"no results.csv found under {train_base}; has the training loop run?"
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Aggregate — average the replicate datasets within each grid cell + CI
# ---------------------------------------------------------------------------


def _mean_ci(values: np.ndarray, ci: float) -> dict:
    """Mean + Student-t confidence interval on the mean over a cell's replicates.

    The interval itself is `suite_metrics.t_interval_half_width` — the package's one
    implementation — so a grid cell's CI and the suite tables' are the same arithmetic.
    A single-study model (e.g. the Pareto/NBD baseline) has no spread, so it reports
    `NaN` for the std and both bounds.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = int(x.size)
    mean = float(x.mean()) if n else float("nan")
    std = float(x.std(ddof=1)) if n > 1 else float("nan")
    half = float(t_interval_half_width(std, n, ci))
    return {"mean": mean, "std": std, "n": n, "ci_low": mean - half, "ci_high": mean + half}


def cell_summary(
    results: pd.DataFrame,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    ci: float = 0.95,
) -> pd.DataFrame:
    """Average each ``(model, rate, churn)`` cell over its replicate datasets.

    A *cell* is the set of replicate datasets sharing a grid coordinate (and, if a
    model ran several studies, those too) — i.e. all rows in ``results`` with the same
    ``(model, mean_transaction_rate, churn_rate)``. For each cell and metric it
    reports the mean, sample std, count, and a Student-t ``ci`` interval on the
    mean (the replicate-to-replicate uncertainty — "how does this model do on *this
    kind* of dataset").

    Returns a tidy (long) DataFrame with columns
    ``[model, mean_transaction_rate, churn_rate, metric, mean, std, n, ci_low, ci_high]``.
    """
    records: list[dict] = []
    for (model, rate, churn), cell in results.groupby(["model", *_AXES]):
        for metric in metrics:
            records.append({
                "model": model,
                "mean_transaction_rate": rate,
                "churn_rate": churn,
                "metric": metric,
                **_mean_ci(cell[metric].to_numpy(), ci),
            })
    return pd.DataFrame(records)


def compare_models_table(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Side-by-side ``mean [ci_low, ci_high]`` per grid cell, models as columns.

    Reads the long table from :func:`cell_summary` and pivots one ``metric`` into a
    readable comparison: rows are ``(rate, churn)`` cells, columns are models.
    """
    sub = summary[summary["metric"] == metric].copy()
    if sub.empty:
        raise ValueError(f"metric {metric!r} not in summary")
    sub["val"] = sub.apply(
        lambda r: f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
        if np.isfinite(r["ci_low"]) else f"{r['mean']:.3f}",
        axis=1,
    )
    return sub.pivot_table(index=_AXES, columns="model", values="val", aggfunc="first")


# ---------------------------------------------------------------------------
# 3. Plots — the two pattern views
# ---------------------------------------------------------------------------


def plot_pattern(summary: pd.DataFrame, metric: str):
    """Metric vs churn, one panel per transaction rate, one line per model, 95% CI.

    The clearest "find patterns across datasets" view: each panel fixes a rate so
    the churn trend is clean, and the CI bars say whether a model gap is real or
    replicate noise. Returns the matplotlib ``Figure``.
    """
    sub = summary[summary["metric"] == metric]
    if sub.empty:
        raise ValueError(f"metric {metric!r} not in summary")
    rates = sorted(sub["mean_transaction_rate"].unique())

    fig, axes = plt.subplots(1, len(rates), figsize=(4.2 * len(rates), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, rate in zip(axes, rates):
        panel = sub[sub["mean_transaction_rate"] == rate]
        for model, gm in panel.groupby("model"):
            gm = gm.sort_values("churn_rate")
            # Asymmetric error bars from the CI (falls back to 0 for single-study models).
            lo = (gm["mean"] - gm["ci_low"]).fillna(0.0)
            hi = (gm["ci_high"] - gm["mean"]).fillna(0.0)
            ax.errorbar(gm["churn_rate"], gm["mean"], yerr=[lo, hi],
                        marker="o", capsize=3, label=model)
        ax.set_title(f"rate = {rate}")
        ax.set_xlabel("churn_rate")
    axes[0].set_ylabel(metric)
    axes[0].legend()
    fig.suptitle(f"{metric} — model comparison across grid (mean ± 95% CI over replicates)")
    fig.tight_layout()
    return fig


def plot_diff_grid(
    results: pd.DataFrame,
    metric: str,
    *,
    model_a: str = "LSTM",
    model_b: str = "ParetoNBD",
):
    """Heatmap of ``model_a - model_b`` mean metric over the ``rate x churn`` grid.

    ``model_b`` defaults to the live benchmark name. Suites archived before the MLE
    Pareto/NBD was retired store theirs as ``"ParetoNBD_MLE"``; pass that explicitly
    for those. A name absent from ``results`` raises, listing what is available.

    Collapses the comparison to one glance: near-zero (white) cells are where
    ``model_a`` matches ``model_b``; strong colour is where they diverge. On error
    metrics (rmse/mape) blue = ``model_a`` lower (better). Both models are scored on
    the same dataset within a cell, so the difference is scale-consistent even for
    RMSE. Returns the matplotlib ``Figure``.
    """
    means = results.groupby(["model", *_AXES])[metric].mean()
    for m in (model_a, model_b):
        if m not in means.index.get_level_values("model"):
            raise ValueError(f"model {m!r} not in results (have "
                             f"{sorted(results['model'].unique())})")
    diff = (means.xs(model_a) - means.xs(model_b)).unstack("churn_rate")

    fig, ax = plt.subplots(figsize=(6, 4))
    lim = float(np.nanmax(np.abs(diff.values))) or 1.0
    im = ax.imshow(diff.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(diff.shape[1])); ax.set_xticklabels(diff.columns)
    ax.set_yticks(range(diff.shape[0])); ax.set_yticklabels(diff.index)
    ax.set_xlabel("churn_rate"); ax.set_ylabel("mean transaction rate")
    ax.set_title(f"{metric}: {model_a} − {model_b}   (blue = {model_a} lower)")
    for (i, j), v in np.ndenumerate(diff.values):
        if np.isfinite(v):
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig
