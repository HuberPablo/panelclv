"""Grid measurements that need the generator — what a model got wrong, and why.

Every function here reaches into the **generation study** rather than into what a
trained suite stored: the manifest of generated datasets it is indexed by, the panels
it wrote, and — for the three ``*_grid`` tables — the latent ground truth the models
never see (each customer's true purchase rate ``lambda``, true death week ``tau``, and
the true seasonal multiplier). Nothing here can therefore run on a real panel, and
that is exactly the boundary this module exists to make readable: its sibling
``pareto_nbd_grid`` scores a forecast on stored results alone, so an import line
answers "could this analysis ever run on real data?" without opening a file.

The five measurements answer two questions the stored metrics cannot:

- **Did the model learn to stop?** ``dead_customer_mass`` (share of predicted volume
  spent on customers who never purchase again), and the customer-*week* decomposition
  ``alive_volume_ratio_grid`` / ``dead_volume_leakage_grid`` against the generator's
  own Poisson mean.
- **Did the model learn the season?** ``shape_correlation`` against the realised
  weekly totals, and ``seasonality_grid`` against the true multiplier — the stronger
  test, since no amount of trend-fitting can fake a curve the generator chose.

Two output shapes, deliberately:

- the three ``*_grid`` functions return a ``(rate, churn) x model`` matrix, already
  averaged over the replicate datasets in each cell — a table to read down and across;
- ``dead_customer_mass`` and ``shape_correlation`` stay at replicate granularity (one
  row per dataset and model, like ``collect_grid_results``), because the thesis figures
  they feed average across one axis at a time, which a cell matrix cannot do without
  silently weighting every cell equally.

Those last two are the arithmetic behind the thesis's two grid figures;
``scripts/make_grid_figures.py`` calls them rather than carrying a second copy.

Typical use::

    from panelclv.studies import seasonality_grid, dead_customer_mass

    seasonality_grid(study_dir)              # (rate, churn) x model, correlation
    dead_customer_mass(study_dir)            # long: one row per (dataset, model)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd

from panelclv.data_preparation.pareto_simulation import (
    list_pnbd_datasets,
    load_pnbd_dataset,
    seasonal_weekly_multiplier,
)
from panelclv.data_preparation.period_calendar import flat_week_index, year_and_week
from panelclv.predictions import load_predictions_from_csv

from . import layout

# The grid axes and the two path conventions are defined once, by the module that
# reads the stored results. The dependency runs this way only: the synthetic half may
# name the stored-results half, never the reverse, which is what keeps the real-panel
# boundary one-directional.
from .pareto_nbd_grid import _AXES, _resolve_grid, _suite_dir

# Quarter-year window: long enough to be the "trend" that seasonality rides on,
# short enough to leave the within-year seasonal wiggle intact after subtraction.
_DETREND_WINDOW = 13


# ---------------------------------------------------------------------------
# 1. The traversal — every trained (dataset, model) pair, exactly once
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Dataset:
    """One generated dataset: its grid coordinates and everything the generator wrote."""

    mean_transaction_rate: float
    churn_rate: float
    combo: str
    dataset: str
    suite: Path                    # the trained suite: <train_base>/<combo>__<dataset>
    panel: pd.DataFrame
    ground_truth: pd.DataFrame     # per-customer latent truth: lambda, mu, tau
    config: dict

    # The generator records the panel's column roles with every dataset, so they are
    # read rather than spelled here — the same reason model code names columns through
    # `PanelConfig`. `lambda` / `tau` below are the ground-truth file's own schema.
    @property
    def id_col(self) -> str:
        return self.config["schema"]["id_col"]

    @property
    def target_col(self) -> str:
        return self.config["schema"]["target_col"]

    @property
    def time_cols(self) -> tuple[str, str]:
        """``(year column, week-of-year column)`` of the panel."""
        year_col, week_col = self.config["schema"]["time_cols"]
        return year_col, week_col


@dataclass(frozen=True)
class _Forecast:
    """One model's stored forecast for one dataset, on the generator's week clock."""

    model: str
    predicted: np.ndarray          # (N, H) predicted volume per customer per week
    customers: np.ndarray          # (N,) customer ids, aligned to `predicted`'s rows
    weeks: np.ndarray              # (H,) absolute week index of each column
    season: np.ndarray             # (H,) the generator's true multiplier over those weeks


# A measurement is one number per (dataset, model) — the only thing that differs
# between the five functions below, which is why the traversal is written once.
_Measure = Callable[[_Dataset, _Forecast], float]


def _datasets(study_dir: Path, train_base: Path) -> Iterator[_Dataset]:
    """Every generated dataset that has a trained suite beside it."""
    for row in list_pnbd_datasets(study_dir).itertuples(index=False):
        suite = _suite_dir(train_base, row.combo, row.dataset)
        if not suite.is_dir():
            continue                                   # dataset not trained yet — skip
        panel, ground_truth, config = load_pnbd_dataset(study_dir, row.combo, row.dataset)
        yield _Dataset(
            mean_transaction_rate=row.mean_transaction_rate,
            churn_rate=row.churn_rate,
            combo=row.combo,
            dataset=row.dataset,
            suite=suite,
            panel=panel,
            ground_truth=ground_truth,
            config=config,
        )


def _forecasts(dataset: _Dataset) -> Iterator[_Forecast]:
    """Every model in a dataset's suite that has written a forecast.

    The forecast is read back through ``predictions.load_predictions_from_csv``, the
    one reader of that layout, and found at the path ``layout`` names — a suite the
    grid walks is an ordinary suite, so neither is re-spelled here.
    """
    for model_dir in sorted(p for p in dataset.suite.iterdir() if p.is_dir()):
        pred_path = layout.prediction_path(model_dir, 1)
        if not pred_path.exists():
            continue
        predicted, customers = load_predictions_from_csv(pred_path)
        if customers is None:
            raise ValueError(
                f"{pred_path} has no customer-id column, so it cannot be joined to "
                "the generator's per-customer ground truth"
            )
        # The holdout is the final `horizon` weeks of the panel, where the horizon is
        # the number of week columns the model forecast — so no calendar year is
        # assumed and a shorter or longer forecast lands on the right weeks.
        weeks = _holdout_weeks(dataset.config, predicted.shape[1])
        yield _Forecast(model_dir.name, predicted, customers, weeks,
                        _holdout_season(dataset, weeks))


def _measure_grid(
    study_dir: str | Path,
    train_base: str | Path | None,
    measure: _Measure,
    value: str,
) -> pd.DataFrame:
    """Apply ``measure`` to every trained (dataset, model) pair, into one long table."""
    study_dir, train_base = _resolve_grid(study_dir, train_base)

    rows: list[dict] = []
    for dataset in _datasets(study_dir, train_base):
        for forecast in _forecasts(dataset):
            rows.append({
                "mean_transaction_rate": dataset.mean_transaction_rate,
                "churn_rate": dataset.churn_rate,
                "combo": dataset.combo,
                "dataset": dataset.dataset,
                "model": forecast.model,
                value: measure(dataset, forecast),
            })

    if not rows:
        raise FileNotFoundError(
            f"no predictions found under {train_base}; has the training loop run?"
        )
    return pd.DataFrame(rows)


def _cell_matrix(long: pd.DataFrame, value: str) -> pd.DataFrame:
    """Mean over the replicate datasets in each cell -> (rate, churn) x model matrix."""
    return long.pivot_table(index=_AXES, columns="model", values=value, aggfunc="mean")


# ---------------------------------------------------------------------------
# 2. The generator's clock and curve — what "holdout" and "season" mean here
# ---------------------------------------------------------------------------


def _holdout_weeks(config: dict, horizon: int) -> np.ndarray:
    """Absolute (0-indexed) week numbers of the ``horizon``-week holdout.

    The panel runs weeks ``0 .. n_weeks-1`` on the generator's clock; the holdout is
    always the final ``horizon`` weeks.
    """
    n_weeks = int(config["n_weeks"])
    return np.arange(n_weeks - horizon, n_weeks)


def _holdout_season(dataset: _Dataset, weeks: np.ndarray) -> np.ndarray:
    """The generator's true seasonal multiplier over ``weeks``.

    Reuses ``seasonal_weekly_multiplier`` — the *same* function that produced the data
    — rather than reimplementing it, so the reference curve is exact (0-indexed
    week-of-year, amplitude, and the mean-1 normalisation all match the simulator).
    Returns all-ones when the study has no seasonality.
    """
    config = dataset.config
    season_year = seasonal_weekly_multiplier(
        config.get("seasonal_peaks", ()),
        config.get("seasonal_amplitude", 0.0),
        config.get("seasonal_width", 1.0),
    )
    _, woy = year_and_week(weeks, int(config["start_year"]))   # week-of-year 0..51
    return season_year[woy]


def _holdout_rows(dataset: _Dataset, weeks: np.ndarray) -> pd.DataFrame:
    """The panel rows falling in ``weeks``, tagged with their absolute week index.

    The panel carries ``(year, week-of-year)``; the generator rolled the year over at
    the package's week convention (``period_calendar``) from ``start_year``, so this
    inverts that back to the flat week index the forecast columns are on.
    """
    panel = dataset.panel
    year_col, week_col = dataset.time_cols
    week_index = flat_week_index(
        panel[year_col], panel[week_col], int(dataset.config["start_year"])
    )
    return panel.assign(week_index=week_index).loc[week_index.isin(weeks)]


def _detrend(series: np.ndarray) -> np.ndarray:
    """Subtract a centred rolling mean, leaving the seasonal residual."""
    s = pd.Series(series, dtype=float)
    return (s - s.rolling(_DETREND_WINDOW, center=True, min_periods=1).mean()).to_numpy()


def _oracle_split(dataset: _Dataset, forecast: _Forecast) -> tuple[float, np.ndarray]:
    """``(O_A, alive_week)`` — the legitimate volume, and which weeks were legitimate.

    The oracle is the generator's own Poisson mean — each customer's rate ``lambda``
    times the fraction of the week it is still alive times the true seasonal
    multiplier — so it is the correct target, not another estimate. It is zero on dead
    weeks by construction (``alive_frac`` collapses to 0 once ``tau`` has passed), so
    summing it over the whole holdout *is* the sum over alive customer-weeks.

    ``alive_week`` is the hard week-level companion: ``t < tau``. It catches the
    post-death tail of customers who churn partway through the holdout, which a
    customer-level split would miss. Only the customers a model actually forecast are
    scored (calibration-inactive customers were dropped before training), so both
    halves are summed over that same population.
    """
    latent = dataset.ground_truth.set_index(dataset.id_col)
    customers = forecast.customers
    lam = latent["lambda"].reindex(customers).to_numpy()[:, None]      # (N, 1)
    tau = latent["tau"].reindex(customers).to_numpy()[:, None]         # (N, 1)

    weeks = forecast.weeks
    alive_frac = np.clip(np.minimum(tau, weeks + 1) - weeks, 0.0, 1.0)  # (N, H)
    oracle_alive = float((lam * alive_frac * forecast.season).sum())
    alive_week = weeks[None, :] < tau                                   # (N, H) bool
    return oracle_alive, alive_week


# ---------------------------------------------------------------------------
# 3. The measurements — one number per (dataset, model)
# ---------------------------------------------------------------------------


def _seasonal_corr(dataset: _Dataset, forecast: _Forecast) -> float:
    """Detrended correlation of predicted weekly totals against the true curve.

    Deliberately the *strong* form: a naive correlation of predicted vs actual weekly
    totals is contaminated by trend (a model that merely tracks volume declining as
    customers die scores well without any seasonal ability). Both sides are detrended
    and the reference is the multiplier the generator actually used — a curve no
    trend-fitting can fake.
    """
    if not dataset.config.get("seasonal_peaks"):
        raise ValueError(
            f"dataset {dataset.combo}/{dataset.dataset} has no seasonal_peaks; "
            "seasonality_grid needs a study generated with seasonality"
        )
    pred_resid = _detrend(forecast.predicted.sum(axis=0))
    if pred_resid.std() <= 1e-12:
        return np.nan                                  # a flat forecast has no shape
    return float(np.corrcoef(_detrend(forecast.season), pred_resid)[0, 1])


def _alive_volume_ratio(dataset: _Dataset, forecast: _Forecast) -> float:
    """``R_A = P_A / O_A`` — predicted volume on alive customer-weeks, over the oracle."""
    oracle_alive, alive_week = _oracle_split(dataset, forecast)
    if oracle_alive <= 0:
        return np.nan
    return float(forecast.predicted[alive_week].sum() / oracle_alive)


def _dead_volume_leakage(dataset: _Dataset, forecast: _Forecast) -> float:
    """``L_D = P_D / O_A`` — predicted volume on dead customer-weeks, over the oracle."""
    oracle_alive, alive_week = _oracle_split(dataset, forecast)
    if oracle_alive <= 0:
        return np.nan
    return float(forecast.predicted[~alive_week].sum() / oracle_alive)


def _dead_customer_mass(dataset: _Dataset, forecast: _Forecast) -> float:
    """Share of predicted volume assigned to customers with no holdout purchase."""
    holdout = _holdout_rows(dataset, forecast.weeks)
    bought = holdout.groupby(dataset.id_col)[dataset.target_col].sum()
    silent = bought.index[bought.to_numpy() == 0]
    total = forecast.predicted.sum(axis=1)             # predicted volume per customer
    return float(total[np.isin(forecast.customers, silent)].sum() / total.sum())


def _shape_correlation(dataset: _Dataset, forecast: _Forecast) -> float:
    """Correlation of predicted against actual weekly holdout totals."""
    holdout = _holdout_rows(dataset, forecast.weeks)
    # Reindexed onto the forecast's own weeks, so the two series are aligned week for
    # week; a holdout week nobody purchased in is a real zero, not a missing row.
    actual = (holdout.groupby("week_index")[dataset.target_col].sum()
              .reindex(forecast.weeks, fill_value=0).to_numpy())
    predicted = forecast.predicted.sum(axis=0)
    if predicted.std() == 0 or actual.std() == 0:
        return np.nan                                  # correlation is undefined
    return float(np.corrcoef(actual, predicted)[0, 1])


# ---------------------------------------------------------------------------
# 4. The public tables
# ---------------------------------------------------------------------------


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
        the panels, the latent ground truth, and the true seasonal curve per dataset.
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
    long = _measure_grid(study_dir, train_base, _seasonal_corr, "seasonal_corr")
    return _cell_matrix(long, "seasonal_corr")


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

    Read it as:

        = 1   alive periods receive exactly their expected volume
        < 1   the model **under**-serves alive periods (predicts too little)
        > 1   the model **over**-serves alive periods (predicts too much)

    Together with the leakage this is a clean split of the aggregate volume ratio::

        R_A + L_D = (P_A + P_D) / O_A = total predicted volume / total oracle volume

    — the legitimate half plus the leaked half.

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
    long = _measure_grid(study_dir, train_base, _alive_volume_ratio, "alive_volume_ratio")
    return _cell_matrix(long, "alive_volume_ratio")


def dead_volume_leakage_grid(
    study_dir: str | Path,
    train_base: str | Path | None = None,
) -> pd.DataFrame:
    """Per-``(rate, churn)`` dead-volume leakage of each model.

    The companion to :func:`alive_volume_ratio_grid`, over the same customer-week
    classification by the true churn week ``tau``::

        dead_volume_leakage  L_D = P_D / O_A

    where ``P_D`` is predicted volume falling on dead customer-weeks (``t >= tau``) —
    pure error, since the oracle after death is exactly zero — and ``O_A`` is the
    legitimate volume. So ``L_D = 0`` means nothing predicted after death,
    ``L_D = 0.10`` means dead periods received erroneous volume equal to 10% of all
    legitimate volume, and ``L_D = 0.30`` is severe leakage. Lower is better.

    Unlike :func:`dead_customer_mass`, which classifies a *customer* once by what it
    actually purchased, this counts the dead *weeks* of customers who die mid-holdout
    and normalises by the true volume rather than by the model's own (possibly
    inflated) total.

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
    long = _measure_grid(study_dir, train_base, _dead_volume_leakage, "dead_volume_leakage")
    return _cell_matrix(long, "dead_volume_leakage")


def dead_customer_mass(
    study_dir: str | Path,
    train_base: str | Path | None = None,
) -> pd.DataFrame:
    """Share of each model's predicted holdout volume spent on silent customers.

    A *silent* customer made no purchase at all in the holdout window, so volume
    predicted for it is the direct signature of a missing death mechanism. Unlike the
    two ``*_volume_*_grid`` tables this needs no latent truth beyond the panel itself,
    which is what makes it the figure's measurement: it is the failure as an observer
    of the holdout would see it.

    Note this is a *relative* diagnostic, not an error rate: a customer who is still
    alive but has a low transaction rate can legitimately record zero purchases in a
    year, so the correct share is well above zero and is not observable. Pareto/NBD —
    near-unbiased in aggregate across this grid — is therefore the reference level the
    neural models are read against.

    Parameters
    ----------
    study_dir
        The generation study folder — supplies the panels the holdout actuals come from.
    train_base
        The trained-suites folder. Defaults to ``Studies/<study_dir name>``.

    Returns
    -------
    Long DataFrame, one row per (dataset, model): the grid coordinates, the ``combo`` /
    ``dataset`` labels, the ``model`` name and ``dead_customer_mass``. Replicate
    granularity, so a caller can average across whichever axis its figure fixes.
    """
    return _measure_grid(study_dir, train_base, _dead_customer_mass, "dead_customer_mass")


def shape_correlation(
    study_dir: str | Path,
    train_base: str | Path | None = None,
) -> pd.DataFrame:
    """Per-dataset correlation between predicted and actual weekly holdout totals.

    This isolates *shape* from *level*: correlation is invariant to a multiplicative
    over-prediction, so a model can score 1.0 here while being 300% biased — which is
    exactly the separation the seasonal figure needs to make. It is the weaker
    companion to :func:`seasonality_grid`: the reference here is the realised series
    (trend, noise and all), not the generator's curve.

    Parameters
    ----------
    study_dir
        The generation study folder — supplies the panels the holdout actuals come from.
    train_base
        The trained-suites folder. Defaults to ``Studies/<study_dir name>``.

    Returns
    -------
    Long DataFrame, one row per (dataset, model): the grid coordinates, the ``combo`` /
    ``dataset`` labels, the ``model`` name and ``shape_correlation``. Replicate
    granularity, so a caller can average across whichever axis its figure fixes.
    """
    return _measure_grid(study_dir, train_base, _shape_correlation, "shape_correlation")
