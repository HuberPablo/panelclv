"""Tests for the synthetic-grid measurements (`studies/synthetic_grid.py`).

The module measures what a model got right *given the generator's truth*, so every
test here builds a miniature generation study on disk — panels, latent ground truth
and per-dataset configs — plus the trained tree of prediction files beside it, and
checks the arithmetic against hand-computed references.

The fixture is deliberately tiny and irregular: eight weeks, a four-week holdout, one
customer that buys only in the calibration window (so a wrong holdout window shows up
as a wrong "silent customer" set) and one that dies half way through the holdout (so a
customer-level alive/dead split cannot pass for the customer-*week* one).

Run:  pytest -q tests/test_synthetic_grid.py
"""

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panelclv.data_preparation.pareto_nbd_simulation import seasonal_weekly_multiplier
from panelclv.studies.synthetic_grid import (
    alive_volume_ratio_grid,
    dead_customer_mass,
    dead_volume_leakage_grid,
    seasonality_grid,
    shape_correlation,
)

# --- the fixture, as constants so every expectation below is readable ------------

COMBO, RATE, CHURN = "Dataset_10_20", 0.1, 0.2
N_WEEKS, HORIZON, START_YEAR = 8, 4, 1999
HOLDOUT = np.arange(N_WEEKS - HORIZON, N_WEEKS)          # absolute weeks 4..7
SEASON_KWARGS = dict(seasonal_peaks=[5], seasonal_amplitude=1.0, seasonal_width=1.0)
SEASON = seasonal_weekly_multiplier(**{k.replace("seasonal_", ""): v
                                       for k, v in SEASON_KWARGS.items()})[HOLDOUT % 52]

# Weekly transactions per customer over the whole panel (weeks 0..7). Customer 2 buys
# in the calibration window only, so it is the holdout-silent one.
PANEL = {1: [0, 1, 0, 0, 2, 0, 1, 0],
         2: [3, 0, 0, 1, 0, 0, 0, 0]}
ACTUAL_WEEKLY = np.array([2, 0, 1, 0])                   # holdout totals over customers

# Latent truth: customer 1 survives the panel, customer 2 dies mid-holdout.
LAM = {1: 0.5, 2: 0.2}
TAU = {1: 10.0, 2: 5.5}

# Two replicate datasets in the one grid cell, each with the same panel but a
# different stored forecast, so the cell mean is not the same number as either.
PRED = {
    "Dataset_1": {"ModelA": {1: [1.0, 2.0, 1.0, 0.0], 2: [0.0, 1.0, 0.0, 1.0]}},
    "Dataset_2": {"ModelA": {1: [0.0, 1.0, 1.0, 1.0], 2: [2.0, 0.0, 1.0, 0.0]}},
}


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    """Write the generation study and its trained tree; return both roots."""
    root = tmp_path_factory.mktemp("pnbd")
    gen, train = root / "Datasets" / "study", root / "Studies" / "study"

    for dataset in PRED:
        ds_dir = gen / COMBO / dataset
        ds_dir.mkdir(parents=True)
        panel = pd.DataFrame([
            {"Id": cid, "year": START_YEAR + w // 52, "week": w % 52, "Transactions": n}
            for cid, counts in PANEL.items() for w, n in enumerate(counts)
        ])
        panel.to_csv(ds_dir / "panel.csv", index=False)
        pd.DataFrame([
            {"Id": cid, "lambda": LAM[cid], "mu": 0.1, "tau": TAU[cid],
             "alive_weeks": min(TAU[cid], N_WEEKS)}
            for cid in PANEL
        ]).to_csv(ds_dir / "ground_truth.csv", index=False)
        with open(ds_dir / "config.json", "w") as fh:
            json.dump({
                "combo": COMBO, "dataset": dataset,
                "mean_transaction_rate": RATE, "churn_rate": CHURN,
                "params": {"r": 2.0, "alpha": 20.0, "s": 2.0, "beta": 400.0},
                "n_customers": len(PANEL), "n_weeks": N_WEEKS,
                "start_year": START_YEAR, "seed": 1, **SEASON_KWARGS,
                "schema": {"id_col": "Id", "target_col": "Transactions",
                           "time_cols": ["year", "week"], "frequency": "weekly"},
                "files": {"panel": "panel.csv", "ground_truth": "ground_truth.csv"},
            }, fh)

        for model, per_customer in PRED[dataset].items():
            pred_dir = train / f"{COMBO}__{dataset}" / model / "Predictions"
            pred_dir.mkdir(parents=True)
            pd.DataFrame([
                {"Id": cid, **{f"week_{w}": v for w, v in enumerate(values)}}
                for cid, values in per_customer.items()
            ]).to_csv(pred_dir / "Prediction_1.csv", index=False)

    return gen, train


def _predicted(dataset, model):
    """The (customer x holdout week) forecast matrix, in customer-id order."""
    return np.array([PRED[dataset][model][cid] for cid in sorted(PANEL)])


def test_dead_customer_mass_scores_the_holdout_silent_customers(study):
    """Share of predicted volume landing on customers with no holdout purchase.

    Customer 2 buys three times in the calibration window and never in the holdout,
    so reading the wrong window would drop it from the silent set and halve the score.
    """
    gen, train = study
    table = dead_customer_mass(gen, train)

    assert list(table.columns) == [
        "mean_transaction_rate", "churn_rate", "combo", "dataset", "model",
        "dead_customer_mass",
    ]
    assert len(table) == len(PRED)                      # one row per (dataset, model)

    for dataset in PRED:
        y = _predicted(dataset, "ModelA")
        expected = y[1].sum() / y.sum()                 # customer 2 is the silent one
        got = table.loc[table["dataset"] == dataset, "dead_customer_mass"].iloc[0]
        assert got == pytest.approx(expected)


def test_shape_correlation_compares_holdout_weekly_totals(study):
    """Correlation of predicted against *actual* weekly totals — shape, not level."""
    gen, train = study
    table = shape_correlation(gen, train)

    for dataset in PRED:
        weekly = _predicted(dataset, "ModelA").sum(axis=0)
        expected = np.corrcoef(ACTUAL_WEEKLY, weekly)[0, 1]
        got = table.loc[table["dataset"] == dataset, "shape_correlation"].iloc[0]
        assert got == pytest.approx(expected)


def test_the_volume_split_is_by_customer_week_against_the_oracle(study):
    """`R_A` and `L_D` split the total predicted volume over the same oracle.

    Customer 2 dies at week 5.5, so its weeks 4-5 are alive and 6-7 are dead: a
    customer-level split would put all four on one side and both ratios would move.
    """
    gen, train = study
    alive = alive_volume_ratio_grid(gen, train)
    dead = dead_volume_leakage_grid(gen, train)

    # The generator's own Poisson mean over the holdout: rate x alive fraction x season.
    tau = np.array([TAU[cid] for cid in sorted(PANEL)])[:, None]
    lam = np.array([LAM[cid] for cid in sorted(PANEL)])[:, None]
    alive_frac = np.clip(np.minimum(tau, HOLDOUT + 1) - HOLDOUT, 0.0, 1.0)
    oracle = (lam * alive_frac * SEASON).sum()
    alive_week = HOLDOUT[None, :] < tau                 # (N, H) hard week-level mask

    ratios, leaks = [], []
    for dataset in PRED:
        y = _predicted(dataset, "ModelA")
        ratios.append(y[alive_week].sum() / oracle)
        leaks.append(y[~alive_week].sum() / oracle)

    assert alive.loc[(RATE, CHURN), "ModelA"] == pytest.approx(np.mean(ratios))
    assert dead.loc[(RATE, CHURN), "ModelA"] == pytest.approx(np.mean(leaks))
    # The two halves recompose into the aggregate volume ratio, which is the whole
    # point of reporting them as a pair.
    total = np.mean([_predicted(d, "ModelA").sum() / oracle for d in PRED])
    assert alive.loc[(RATE, CHURN), "ModelA"] + dead.loc[(RATE, CHURN), "ModelA"] == (
        pytest.approx(total)
    )


def test_seasonality_grid_correlates_against_the_generators_curve(study):
    """The reference is the true multiplier, detrended on both sides."""
    gen, train = study
    table = seasonality_grid(gen, train)

    def detrend(x):
        s = pd.Series(x, dtype=float)
        return (s - s.rolling(13, center=True, min_periods=1).mean()).to_numpy()

    expected = np.mean([
        np.corrcoef(detrend(SEASON), detrend(_predicted(d, "ModelA").sum(axis=0)))[0, 1]
        for d in PRED
    ])
    assert table.index.names == ["mean_transaction_rate", "churn_rate"]
    assert table.loc[(RATE, CHURN), "ModelA"] == pytest.approx(expected)


def test_a_study_without_seasonality_is_refused(study, tmp_path):
    """The seasonal correlation is undefined with no seasonal component."""
    gen, train = study
    flat = tmp_path / "flat"
    for cfg_path in sorted(gen.glob("*/*/config.json")):
        ds_dir = flat / cfg_path.parent.relative_to(gen)
        ds_dir.mkdir(parents=True)
        for name in ("panel.csv", "ground_truth.csv"):
            (ds_dir / name).write_text((cfg_path.parent / name).read_text())
        cfg = json.loads(cfg_path.read_text())
        cfg["seasonal_peaks"] = []
        (ds_dir / "config.json").write_text(json.dumps(cfg))

    with pytest.raises(ValueError, match="seasonal"):
        seasonality_grid(flat, train)


def test_the_real_panel_boundary_is_readable_from_an_import_line():
    """`pareto_nbd_grid` reads only what a suite stored, so it never names this module.

    The split exists so that "can this run on a real panel?" is answered by an import
    line. A one-way dependency is what makes that true: the synthetic half may reach
    for the stored-results half's grid axes, never the reverse.
    """
    stored = Path(__file__).resolve().parents[1] / "src" / "panelclv" / "studies"
    tree = ast.parse((stored / "pareto_nbd_grid.py").read_text())
    imported = {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        for name in (node.module, *(a.name for a in node.names))
    }
    assert not any("synthetic_grid" in name for name in imported)
