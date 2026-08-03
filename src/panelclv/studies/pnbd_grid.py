"""Cross-grid analysis for a Pareto/NBD generation study trained per-dataset.

``data_preparation.pareto_simulation.generate_pnbd_study`` lays out a grid of
synthetic datasets — ``mean_transaction_rate`` x ``churn_rate``, with
``n_datasets`` replicate panels per cell. Training one study suite per dataset
(``run_study_suite`` with ``n_studies_per_model=1``) produces a parallel tree

    <train_base>/<combo>__<dataset>/results.csv        # one row per (model, study)

This module joins the two halves: it reads every dataset's model metrics, tags
each with the dataset's grid coordinates, averages the replicate datasets within
each ``(rate, churn)`` **group** with a confidence interval, and plots how each
model performs across the grid — so you can compare models per dataset and spot
performance patterns (e.g. does the LSTM's error track the Pareto/NBD benchmark's
as churn rises, or diverge in the sparse low-rate corner?).

Typical use::

    from panelclv.data_preparation import pareto_simulation as ps
    from panelclv.studies import (
        collect_grid_results, group_summary, compare_models_table,
        plot_pattern, plot_diff_grid,
    )

    results = collect_grid_results(study_dir, train_base)   # long: one row / (model, dataset)
    summary = group_summary(results)                        # mean + 95% CI per (model, cell)
    compare_models_table(summary, "mape")                   # side-by-side per group
    plot_pattern(summary, "mape")                           # metric vs churn, panel per rate
    plot_diff_grid(results, "mape")                         # LSTM - ParetoNBD heatmap
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from panelclv.data_preparation.pareto_simulation import (
    list_pnbd_datasets,
    load_pnbd_dataset,
    _seasonal_weekly_multiplier,
    WEEKS_PER_YEAR,
)

# Metric name (as we expose it) -> column name in each suite's results.csv. The
# aggregate-style MAPE is stored under a longer key; we surface it as "mape".
_METRIC_SOURCE = {
    "rmse": "rmse",
    "bias_percent": "bias_percent",
    "mape": "mape_aggregate_style",
}
DEFAULT_METRICS = ("rmse", "bias_percent", "mape")

# The two grid axes carried through from the generation study.
_AXES = ["mean_transaction_rate", "churn_rate"]


# ---------------------------------------------------------------------------
# 1. Collect — join per-dataset model metrics with their grid coordinates
# ---------------------------------------------------------------------------


def collect_grid_results(
    study_dir: str | Path,
    train_base: str | Path | None = None,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Read every trained dataset's metrics into one long table tagged by grid cell.

    Parameters
    ----------
    study_dir
        The generation study folder (what ``generate_pnbd_study`` returned) — used
        to enumerate the datasets and their ``(rate, churn)`` coordinates.
    train_base
        The folder holding the trained suites (``<combo>__<dataset>/`` subfolders).
        Defaults to ``Studies/<study_dir name>`` — the convention the training loop
        uses — resolved relative to the current working directory.
    metrics
        Which metrics to pull from each suite's ``results.csv``.

    Returns
    -------
    DataFrame with one row per (model, dataset): the grid coordinates
    (``mean_transaction_rate``, ``churn_rate``), the ``combo`` / ``dataset`` labels,
    the ``model`` name, and one column per requested metric. Datasets with no
    ``results.csv`` yet (not trained) are skipped.
    """
    study_dir = Path(study_dir)
    if train_base is None:
        train_base = Path("Studies") / study_dir.name
    train_base = Path(train_base)

    unknown = [m for m in metrics if m not in _METRIC_SOURCE]
    if unknown:
        raise ValueError(f"unknown metrics {unknown}; known: {list(_METRIC_SOURCE)}")

    grid = list_pnbd_datasets(study_dir)
    rows: list[dict] = []
    for g in grid.itertuples(index=False):
        res_path = train_base / f"{g.combo}__{g.dataset}" / "results.csv"
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
# 1b. Seasonality — how well each model recovers the KNOWN seasonal curve
# ---------------------------------------------------------------------------
#
# This is a per-dataset diagnostic that results.csv cannot supply (it stores only
# RMSE / bias / MAPE), so it re-reads the panels and the per-model prediction files.
# The metric is deliberately the *strong* one: a naive correlation of predicted vs
# actual weekly totals is contaminated by trend (a model that merely tracks volume
# declining as customers die scores well without any seasonal ability). We instead
# detrend the prediction and correlate it against the seasonal multiplier the
# generator actually used — a curve no trend-fitting can fake.

# Quarter-year window: long enough to be the "trend" that seasonality rides on,
# short enough to leave the within-year seasonal wiggle intact after subtraction.
_DETREND_WINDOW = 13


def _holdout_weeks(cfg: dict, horizon: int) -> np.ndarray:
    """Absolute (0-indexed) week numbers of the ``horizon``-week holdout.

    The panel runs weeks ``0 .. n_weeks-1`` on the generator's clock; the holdout is
    always the final ``horizon`` weeks (``horizon`` = the number of ``week_*`` columns
    a model forecast), so no calendar year is assumed.
    """
    n_weeks = int(cfg["n_weeks"])
    return np.arange(n_weeks - horizon, n_weeks)


def _holdout_season(cfg: dict, holdout_weeks: np.ndarray) -> np.ndarray:
    """The generator's true seasonal multiplier over the holdout weeks.

    Reuses ``_seasonal_weekly_multiplier`` — the *same* function that produced the
    data — rather than reimplementing it, so the reference curve is exact (0-indexed
    week-of-year, amplitude, and the mean-1 normalisation all match the simulator).
    Returns all-ones when the study has no seasonality.
    """
    season_year = _seasonal_weekly_multiplier(
        cfg.get("seasonal_peaks", ()),
        cfg.get("seasonal_amplitude", 0.0),
        cfg.get("seasonal_width", 1.0),
    )
    return season_year[holdout_weeks % WEEKS_PER_YEAR]


def _detrend(series: np.ndarray) -> np.ndarray:
    """Subtract a centred rolling mean, leaving the seasonal residual."""
    s = pd.Series(series, dtype=float)
    return (s - s.rolling(_DETREND_WINDOW, center=True, min_periods=1).mean()).to_numpy()


def _weekly_prediction(pred_path: Path) -> np.ndarray:
    """Sum a model's per-customer prediction file into weekly holdout totals."""
    pred = pd.read_csv(pred_path)
    return pred.drop(columns=["Id"]).sum(axis=0).to_numpy()


def seasonality_grid(
    study_dir: str | Path,
    train_base: str | Path | None = None,
) -> pd.DataFrame:
    """Per-``(rate, churn)`` seasonal-detection ability of each model.

    For every trained dataset, each model's predicted weekly holdout totals are
    detrended and correlated against the generator's true seasonal curve; the
    correlations are then averaged over the replicate datasets in each grid cell.
    A value near 1 means the model recovers the seasonal shape; near 0 (Pareto/NBD,
    by construction) means no seasonal component.

    Parameters
    ----------
    study_dir
        The generation study folder (what ``generate_pnbd_study`` returned) — supplies
        the panels, the holdout actuals, and the true seasonal curve per dataset.
    train_base
        The trained-suites folder (``<combo>__<dataset>/`` subfolders). Defaults to
        ``Studies/<study_dir name>``, the convention the training loop uses.

    Returns
    -------
    DataFrame indexed by ``(mean_transaction_rate, churn_rate)`` with one column per
    model, holding the mean seasonal correlation across that cell's replicates — a
    matrix you can read down (churn) and across (rate) to see how the ability evolves.

    Raises
    ------
    ValueError
        If the study has no seasonal component (``seasonal_peaks`` absent), so the
        metric would be undefined.
    """
    study_dir = Path(study_dir)
    if train_base is None:
        train_base = Path("Studies") / study_dir.name
    train_base = Path(train_base)

    grid = list_pnbd_datasets(study_dir)
    rows: list[dict] = []
    for g in grid.itertuples(index=False):
        suite = train_base / f"{g.combo}__{g.dataset}"
        if not suite.is_dir():
            continue                                       # dataset not trained yet — skip
        _, _, cfg = load_pnbd_dataset(study_dir, g.combo, g.dataset)
        if not cfg.get("seasonal_peaks"):
            raise ValueError(
                f"dataset {g.combo}/{g.dataset} has no seasonal_peaks; "
                "seasonality_grid needs a study generated with seasonality"
            )

        for model_dir in sorted(p for p in suite.iterdir() if p.is_dir()):
            pred_path = model_dir / "Predictions" / "Prediction_1.csv"
            if not pred_path.exists():
                continue
            weekly_pred = _weekly_prediction(pred_path)

            # Holdout length is the forecast horizon; the true seasonal curve over
            # those exact weeks is the reference (detrended on both sides so only
            # within-year shape is compared, never the slow volume trend).
            weeks = _holdout_weeks(cfg, len(weekly_pred))
            season_resid = _detrend(_holdout_season(cfg, weeks))
            pred_resid = _detrend(weekly_pred)

            corr = (
                np.corrcoef(season_resid, pred_resid)[0, 1]
                if pred_resid.std() > 1e-12 else np.nan     # a flat forecast has no shape
            )
            rows.append({
                "mean_transaction_rate": g.mean_transaction_rate,
                "churn_rate": g.churn_rate,
                "model": model_dir.name,
                "seasonal_corr": corr,
            })

    if not rows:
        raise FileNotFoundError(
            f"no predictions found under {train_base}; has the training loop run?"
        )
    long = pd.DataFrame(rows)
    # Mean over the replicate datasets in each cell -> (rate, churn) x model matrix.
    return long.pivot_table(
        index=_AXES, columns="model", values="seasonal_corr", aggfunc="mean"
    )


def alive_volume_ratio_grid(
    study_dir: str | Path,
    train_base: str | Path | None = None,
) -> pd.DataFrame:
    """Per-``(rate, churn)`` alive-volume ratio of each model.

    One half of the customer-*week* volume decomposition (the companion is
    :func:`dead_volume_leakage_grid`). Every customer-week is classified by the true
    churn week ``tau``: week ``t`` (absolute index) is *alive* if ``t < tau``. Summing
    predicted volume ``y`` and oracle volume over the holdout gives

        P_A = sum of predicted volume over alive customer-weeks
        O_A = sum of oracle over alive customer-weeks   (all legitimate volume; the
              oracle is zero after death, so this is the total oracle)

    and the ratio is::

        alive_volume_ratio  R_A = P_A / O_A

    The oracle is the generator's own Poisson mean — each customer's rate ``lambda``
    times the fraction of the week it is alive times the true seasonal multiplier —
    so it is the correct target, not another estimate. Read the ratio as:

        = 1   alive periods receive exactly their expected volume
        < 1   the model **under**-serves alive periods (predicts too little)
        > 1   the model **over**-serves alive periods (predicts too much)

    Together with the leakage this is a clean split of the aggregate volume ratio::

        R_A + L_D = (P_A + P_D) / O_A = total predicted volume / total oracle volume

    — the legitimate half plus the leaked half. Only the customers a model actually
    forecast are scored (calibration-inactive customers were dropped before training),
    and ``P_A`` / ``O_A`` are summed over that same population.

    Parameters
    ----------
    study_dir
        The generation study folder — supplies the latent ground truth (``lambda``,
        ``tau``) and the true seasonal curve per dataset.
    train_base
        The trained-suites folder. Defaults to ``Studies/<study_dir name>``.

    Returns
    -------
    DataFrame indexed by ``(mean_transaction_rate, churn_rate)`` with one column per
    model, holding the mean alive-volume ratio across that cell's replicates.
    """
    study_dir = Path(study_dir)
    if train_base is None:
        train_base = Path("Studies") / study_dir.name
    train_base = Path(train_base)

    grid = list_pnbd_datasets(study_dir)
    rows: list[dict] = []
    for g in grid.itertuples(index=False):
        suite = train_base / f"{g.combo}__{g.dataset}"
        if not suite.is_dir():
            continue                                       # dataset not trained yet — skip
        _, gt, cfg = load_pnbd_dataset(study_dir, g.combo, g.dataset)
        latent = gt.set_index("Id")
        lam = latent["lambda"]
        tau = latent["tau"]

        for model_dir in sorted(p for p in suite.iterdir() if p.is_dir()):
            pred_path = model_dir / "Predictions" / "Prediction_1.csv"
            if not pred_path.exists():
                continue
            pred = pd.read_csv(pred_path).set_index("Id")
            horizon = pred.shape[1]                         # week_0 .. week_{H-1}, in order
            weeks = _holdout_weeks(cfg, horizon)            # absolute holdout week indices
            season = _holdout_season(cfg, weeks)            # true multiplier per holdout week

            lam_i = lam.reindex(pred.index).to_numpy()[:, None]        # (N, 1)
            tau_i = tau.reindex(pred.index).to_numpy()[:, None]        # (N, 1)
            y = pred.to_numpy()                                        # (N, H) predicted per week

            # O_A: total legitimate oracle volume. The oracle is zero on dead weeks
            # (alive_frac collapses to 0 once tau has passed), so summing over all
            # holdout weeks is the sum over alive customer-weeks.
            alive_frac = np.clip(np.minimum(tau_i, weeks + 1) - weeks, 0.0, 1.0)  # (N, H)
            oracle_alive = float((lam_i * alive_frac * season).sum())

            # P_A: predicted volume on alive customer-weeks (t < tau), the hard
            # week-level complement of the leakage mask.
            alive_week = weeks[None, :] < tau_i                        # (N, H) bool
            p_alive = float(y[alive_week].sum())

            rows.append({
                "mean_transaction_rate": g.mean_transaction_rate,
                "churn_rate": g.churn_rate,
                "model": model_dir.name,
                "alive_volume_ratio": p_alive / oracle_alive if oracle_alive > 0 else np.nan,
            })

    if not rows:
        raise FileNotFoundError(
            f"no predictions found under {train_base}; has the training loop run?"
        )
    long = pd.DataFrame(rows)
    return long.pivot_table(
        index=_AXES, columns="model", values="alive_volume_ratio", aggfunc="mean"
    )


def dead_volume_leakage_grid(
    study_dir: str | Path,
    train_base: str | Path | None = None,
) -> pd.DataFrame:
    """Per-``(rate, churn)`` dead-volume leakage of each model.

    The companion to :func:`alive_volume_ratio_grid`. Where that ratio classifies a
    *customer* once, this classifies every *customer-week* by the true churn week
    ``tau``: week ``t`` (absolute index) is an *alive* week if ``t < tau`` and a
    *dead* week otherwise. Summing predicted volume ``y`` and oracle volume over the
    holdout gives

        O_A = sum of oracle over alive customer-weeks   (all legitimate volume;
              oracle is zero after death by construction, so this is the total oracle)
        P_D = sum of predicted volume over dead customer-weeks   (pure error)

    and the leakage is that error normalised by the legitimate volume::

        dead_volume_leakage  L_D = P_D / O_A

    Because the oracle after death is exactly zero, every unit of ``P_D`` is spurious;
    dividing by ``O_A`` expresses it as a fraction of the volume that *should* have
    occurred. So ``L_D = 0`` means nothing predicted after death, ``L_D = 0.10`` means
    dead periods received erroneous volume equal to 10% of all legitimate volume, and
    ``L_D = 0.30`` is severe leakage. Lower is better. Unlike the aggregate
    ``share of predicted volume on dead customers``, this counts the dead *weeks* of
    customers who die mid-holdout, and normalises by the true volume rather than by
    the model's own (possibly inflated) total.

    Only the customers a model actually forecast are scored (calibration-inactive
    customers were dropped before training), and both ``P_D`` and ``O_A`` are summed
    over that same population.

    Parameters
    ----------
    study_dir
        The generation study folder — supplies ``lambda`` / ``tau`` and the true
        seasonal curve per dataset.
    train_base
        The trained-suites folder. Defaults to ``Studies/<study_dir name>``.

    Returns
    -------
    DataFrame indexed by ``(mean_transaction_rate, churn_rate)`` with one column per
    model, holding the mean dead-volume leakage across that cell's replicates.
    """
    study_dir = Path(study_dir)
    if train_base is None:
        train_base = Path("Studies") / study_dir.name
    train_base = Path(train_base)

    grid = list_pnbd_datasets(study_dir)
    rows: list[dict] = []
    for g in grid.itertuples(index=False):
        suite = train_base / f"{g.combo}__{g.dataset}"
        if not suite.is_dir():
            continue                                       # dataset not trained yet — skip
        _, gt, cfg = load_pnbd_dataset(study_dir, g.combo, g.dataset)
        latent = gt.set_index("Id")
        lam = latent["lambda"]
        tau = latent["tau"]

        for model_dir in sorted(p for p in suite.iterdir() if p.is_dir()):
            pred_path = model_dir / "Predictions" / "Prediction_1.csv"
            if not pred_path.exists():
                continue
            pred = pd.read_csv(pred_path).set_index("Id")
            horizon = pred.shape[1]                         # week_0 .. week_{H-1}, in order
            weeks = _holdout_weeks(cfg, horizon)            # absolute holdout week indices
            season = _holdout_season(cfg, weeks)            # true multiplier per holdout week

            lam_i = lam.reindex(pred.index).to_numpy()[:, None]        # (N, 1)
            tau_i = tau.reindex(pred.index).to_numpy()[:, None]        # (N, 1)
            y = pred.to_numpy()                                        # (N, H) predicted per week

            # O_A: total legitimate oracle volume. The oracle is zero on dead weeks
            # (alive_frac collapses to 0 once tau has passed), so summing over all
            # holdout weeks *is* the sum over alive customer-weeks.
            alive_frac = np.clip(np.minimum(tau_i, weeks + 1) - weeks, 0.0, 1.0)  # (N, H)
            oracle_alive = float((lam_i * alive_frac * season).sum())

            # P_D: predicted volume falling on dead customer-weeks (t >= tau). This
            # hard week-level mask catches the post-death tail of customers who churn
            # partway through the holdout, not just those already dead at its start.
            dead_week = weeks[None, :] >= tau_i                        # (N, H) bool
            leaked = float(y[dead_week].sum())

            rows.append({
                "mean_transaction_rate": g.mean_transaction_rate,
                "churn_rate": g.churn_rate,
                "model": model_dir.name,
                "dead_volume_leakage": leaked / oracle_alive if oracle_alive > 0 else np.nan,
            })

    if not rows:
        raise FileNotFoundError(
            f"no predictions found under {train_base}; has the training loop run?"
        )
    long = pd.DataFrame(rows)
    return long.pivot_table(
        index=_AXES, columns="model", values="dead_volume_leakage", aggfunc="mean"
    )


# ---------------------------------------------------------------------------
# 2. Aggregate — average the replicate datasets within each grid cell + CI
# ---------------------------------------------------------------------------


def _mean_ci(values: np.ndarray, ci: float) -> dict:
    """Mean + Student-t confidence interval on the mean over a group's replicates."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = int(x.size)
    mean = float(x.mean()) if n else float("nan")
    if n < 2:
        # A single-study model (e.g. Pareto/NBD baseline) has no spread to report.
        return {"mean": mean, "std": float("nan"), "n": n,
                "ci_low": float("nan"), "ci_high": float("nan")}
    std = float(x.std(ddof=1))
    half = float(stats.t.ppf(0.5 + ci / 2.0, n - 1) * std / np.sqrt(n))
    return {"mean": mean, "std": std, "n": n, "ci_low": mean - half, "ci_high": mean + half}


def group_summary(
    results: pd.DataFrame,
    *,
    metrics: Sequence[str] = DEFAULT_METRICS,
    ci: float = 0.95,
) -> pd.DataFrame:
    """Average each ``(model, rate, churn)`` group over its replicate datasets.

    A *group* is the set of replicate datasets sharing a grid cell (and, if a model
    ran several studies, those too) — i.e. all rows in ``results`` with the same
    ``(model, mean_transaction_rate, churn_rate)``. For each group and metric it
    reports the mean, sample std, count, and a Student-t ``ci`` interval on the
    mean (the replicate-to-replicate uncertainty — "how does this model do on *this
    kind* of dataset").

    Returns a tidy (long) DataFrame with columns
    ``[model, mean_transaction_rate, churn_rate, metric, mean, std, n, ci_low, ci_high]``.
    """
    records: list[dict] = []
    for (model, rate, churn), grp in results.groupby(["model", *_AXES]):
        for metric in metrics:
            records.append({
                "model": model,
                "mean_transaction_rate": rate,
                "churn_rate": churn,
                "metric": metric,
                **_mean_ci(grp[metric].to_numpy(), ci),
            })
    return pd.DataFrame(records)


def compare_models_table(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Side-by-side ``mean [ci_low, ci_high]`` per grid cell, models as columns.

    Reads the long table from :func:`group_summary` and pivots one ``metric`` into a
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
