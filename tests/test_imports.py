"""Smoke tests for the panelclv package.

Verifies that (a) the package and every subpackage import cleanly, (b) the
public API names resolve from their new subpackage homes after the altitude
split, and (c) a couple of pure-Python helpers compute the right numbers. These
are deliberately light (no training, no GPU) so they can run in CI in seconds.

Run:  pytest -q            (from the repo root, with the package installed)
"""

import importlib

import numpy as np
import pytest

# Every importable subpackage created by the altitude split.
SUBPACKAGES = [
    "panelclv",
    "panelclv.models",
    "panelclv.training",
    "panelclv.tuning",
    "panelclv.evaluation",
    "panelclv.benchmarks",
    "panelclv.experiments",
    "panelclv.data_preparation",
    "panelclv.configs",
]


@pytest.mark.parametrize("module", SUBPACKAGES)
def test_subpackage_imports(module):
    """Each subpackage imports without error."""
    importlib.import_module(module)


def test_public_api_resolves_from_new_homes():
    """The headline entry points are importable from the subpackage they now live in."""
    from panelclv.models import (
        MultinomialLSTMModel,
        InferenceMultinomialLSTMModel,
        mc_forecast,
        run_monte_carlo_forecast,
    )
    from panelclv.training import fit_model
    from panelclv.tuning import run_optuna_study, select_features
    from panelclv.evaluation import compute_metrics, plot_weekly_aggregated
    from panelclv.benchmarks import (
        compute_pareto_predictions,
        compute_pareto_paper_predictions,
    )
    from panelclv.experiments import make_data_builder, build_inference_from_trial

    # `mc_forecast` is documented as an alias for `run_monte_carlo_forecast`.
    assert mc_forecast is run_monte_carlo_forecast


def test_rmse_matches_hand_computation():
    """rmse(y_true, y_pred) = sqrt(mean(squared error))."""
    from panelclv.evaluation import rmse

    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])  # errors 0, 0, 2 -> mse = 4/3
    assert rmse(y_true, y_pred) == pytest.approx((4.0 / 3.0) ** 0.5)


def test_compute_metrics_returns_expected_keys():
    """compute_metrics returns a dict that includes the primary metrics."""
    from panelclv.evaluation import compute_metrics

    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([0.0, 1.0, 2.0, 3.0])
    metrics = compute_metrics(y_true, y_pred)
    assert isinstance(metrics, dict)
    assert "rmse" in metrics
    # A perfect prediction has zero RMSE.
    assert metrics["rmse"] == pytest.approx(0.0)
