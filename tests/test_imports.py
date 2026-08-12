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
    from panelclv.evaluation import metrics_table, plot_weekly_aggregated
    from panelclv.benchmarks import compute_pareto_predictions
    from panelclv.experiments import make_data_builder, build_inference_from_trial

    # `mc_forecast` is documented as an alias for `run_monte_carlo_forecast`.
    assert mc_forecast is run_monte_carlo_forecast


# `compute_forecast_metrics` is the package's single scoring authority — plots, group
# tables and study results all delegate to it — so its three definitions are pinned
# here against hand-computed values rather than only exercised indirectly.


def test_forecast_metrics_match_hand_computation():
    """rmse, bias_percent and mape_aggregate on a worked (N, T_HOLD) example."""
    from panelclv.models import compute_forecast_metrics

    # 2 customers x 3 holdout periods. Errors are +1 in one cell and -2 in another,
    # so mse = (1 + 4) / 6 and the totals differ by -1 out of 9.
    actual = np.array([[1.0, 2.0, 3.0],
                       [0.0, 1.0, 2.0]])
    pred = np.array([[1.0, 3.0, 3.0],
                     [0.0, 1.0, 0.0]])

    m = compute_forecast_metrics(actual, pred)
    assert m["rmse"] == pytest.approx((5.0 / 6.0) ** 0.5)
    # bias is on the grand total: (8 - 9) / 9.
    assert m["bias_percent"] == pytest.approx(100.0 * -1.0 / 9.0)
    # MAPE is aggregate: per-period totals, summed abs error over total actual.
    # actual_t = [1, 3, 5], pred_t = [1, 4, 3] -> abs diff [0, 1, 2] = 3 of 9.
    assert m["mape_aggregate"] == pytest.approx(100.0 * 3.0 / 9.0)


def test_forecast_metrics_perfect_prediction_is_zero():
    """A perfect forecast scores zero on all three numbers."""
    from panelclv.models import compute_forecast_metrics

    actual = np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 1.0]])
    m = compute_forecast_metrics(actual, actual.copy())
    assert set(m) == {"rmse", "bias_percent", "mape_aggregate"}
    assert m["rmse"] == pytest.approx(0.0)
    assert m["bias_percent"] == pytest.approx(0.0)
    assert m["mape_aggregate"] == pytest.approx(0.0)


def test_retired_metric_helpers_are_gone():
    """`evaluation_utils` and its keys were retired; one scoring path remains.

    Guards the consolidation: a re-added `compute_metrics` would reintroduce a second
    set of definitions on a different scale (fractions vs percent).
    """
    import panelclv.evaluation as ev

    for name in ("compute_metrics", "mae", "mape_positive",
                 "cumulative_mape", "aggregate_bias_fraction"):
        assert not hasattr(ev, name), f"{name} should have been retired"
    with pytest.raises(ImportError):
        importlib.import_module("panelclv.evaluation.evaluation_utils")


def test_retired_dead_surface_is_gone():
    """The kill list stays killed — issue 03 of the package cleanup.

    Each name below was deleted for having no caller anywhere in `src/`, the live
    entry points, `tests/` or any notebook (the ledger at
    `.scratch/package-simplification/ledger.csv` carries the per-row evidence). The
    risk this guards is re-addition by habit: an `__init__` export is one line, and
    nothing else in the suite would notice a symbol coming back with no caller.
    """
    import panelclv.models as models
    import panelclv.evaluation as ev
    import panelclv.studies as st

    # The two `mc_*` per-path aliases. `models/__init__` advertised them as "used
    # throughout the notebooks"; no notebook, live or archived, ever imported either.
    for name in ("mc_simulate_one_path", "mc_simulate_transformer_path"):
        assert not hasattr(models, name), f"{name} should have been retired"

    # `ForecastRun` and its module: a fifth on-disk prediction layout
    # (`<root>/<config>/<n>/manifest.json`) with no writer and no reader.
    assert not hasattr(ev, "ForecastRun"), "ForecastRun should have been retired"
    with pytest.raises(ImportError):
        importlib.import_module("panelclv.evaluation.forecast_run")

    # The only `studies` export with zero importers.
    assert not hasattr(st, "group_metrics_suite_distribution"), \
        "group_metrics_suite_distribution should have been retired"


def test_rollout_composite_selection_is_gone():
    """Rollout-composite trial selection was deleted — issue 04 of the package cleanup.

    ADR-0003 is retired: trials are selected on validation loss, full stop. Guarded
    here because the deletion is what makes `compute_forecast_metrics` the package's
    only implementation of rmse / bias / MAPE, and re-adding
    `weekly_aggregate_rollout_metrics` would quietly make that claim false again
    (its RMSE was 62x the authority's on the same arrays, its MAPE a different
    estimator). `mc_compute_metrics` goes with it: the authority never had a
    per-path variant, so it never needed an `mc_*` alias to disambiguate.
    """
    import inspect

    import panelclv.models as models
    import panelclv.tuning as tuning
    from panelclv.tuning import optuna_tuning, run_optuna_study
    from panelclv.tuning.optuna_tuning import objective

    # Checked on the defining module, not only on the subpackage: `ROLLOUT_METRIC`
    # was never exported, so a subpackage-only assertion would pass vacuously.
    for name in ("weekly_aggregate_rollout_metrics", "_validation_rollout_score",
                 "ROLLOUT_METRIC"):
        assert not hasattr(optuna_tuning, name), f"{name} should have been retired"
    assert not hasattr(tuning, "weekly_aggregate_rollout_metrics"), \
        "weekly_aggregate_rollout_metrics should have been retired"
    assert not hasattr(models, "mc_compute_metrics"), \
        "mc_compute_metrics should have been retired"

    # `selection_metric` was a parameter with one legal value, and every `rollout_*`
    # knob existed only to configure the other one.
    for fn in (run_optuna_study, objective):
        params = inspect.signature(fn).parameters
        assert "selection_metric" not in params, fn.__name__
        assert not [p for p in params if p.startswith("rollout")], fn.__name__
