"""Smoke tests for the panelclv package.

Verifies that (a) the package and every subpackage import cleanly, (b) the
public API names resolve from their new subpackage homes after the altitude
split, and (c) a couple of pure-Python helpers compute the right numbers. These
are deliberately light (no training, no GPU) so they can run in CI in seconds.

Run:  pytest -q            (from the repo root, with the package installed)
"""

import importlib
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "panelclv"

# Every subpackage the altitude split created, read off the tree rather than listed.
# This used to be a hand-written list, and it had already dropped `studies` — the same
# "one set written out N times" drift the root docstring and the time-flag tables had.
SUBPACKAGE_NAMES = sorted(
    d.name for d in SRC.iterdir() if d.is_dir() and (d / "__init__.py").exists()
)
SUBPACKAGES = ["panelclv"] + [f"panelclv.{name}" for name in SUBPACKAGE_NAMES]


@pytest.mark.parametrize("module", SUBPACKAGES)
def test_subpackage_imports(module):
    """Each subpackage imports without error."""
    importlib.import_module(module)


def test_root_docstring_names_every_subpackage():
    """`panelclv/__init__.py` is the package's map, so it has to be a complete one.

    It is the first thing a reader opens and the only place the altitude split is
    described end to end; a subpackage missing from it is invisible to anyone who has
    not gone looking in the tree. It had drifted before — nine folders, eight listed.
    """
    import panelclv

    missing = [name for name in SUBPACKAGE_NAMES if f"panelclv.{name}" not in panelclv.__doc__]
    assert not missing, f"root docstring does not name: {missing}"


def test_public_api_resolves_from_new_homes():
    """The headline entry points are importable from the subpackage they now live in."""
    from panelclv.models import (  # noqa: F401
        MultinomialLSTMModel,
        RolloutMultinomialLSTMModel,
        forecast_attention,
        forecast_recurrent,
    )
    from panelclv.training import fit_model  # noqa: F401
    from panelclv.tuning import run_optuna_study, select_features  # noqa: F401
    from panelclv.evaluation import metrics_table, plot_weekly_aggregated  # noqa: F401
    from panelclv.benchmarks import (  # noqa: F401
        compute_pareto_predictions,
        pareto_forecast,
        pareto_from_data,
    )
    from panelclv.predictions import (  # noqa: F401
        load_predictions_from_csv,
        reduce_to_customer_period,
        save_predictions_to_csv,
    )
    from panelclv.trials import make_data_builder, refit_best_trial  # noqa: F401
    from panelclv.registry import MODEL_TYPES, build_model, is_neural, rollout_for  # noqa: F401


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


def test_retired_forecast_and_plot_surface_is_gone():
    """Prediction I/O moved out and the `mc_*` aliases went — issue 08 of the cleanup.

    Three deletions in one test, because they were one module. `plot_utils` held
    prediction I/O, two plots and a Pareto/NBD fit; splitting it three ways is what
    let the model layer stop importing `evaluation` (ADR-0002), so a re-added
    `evaluation.plot_utils` would quietly reopen the cycle
    `tests/test_import_graph.py` now guards.

    The `mc_*` aliases exported one function under two public names, twice over;
    the rollouts are named for their mechanism now, because there are three rollout
    model classes but only two rollout functions.
    """
    import panelclv.evaluation as ev
    import panelclv.models as models
    from panelclv.models import monte_carlo_forecasting as mcf

    with pytest.raises(ImportError):
        importlib.import_module("panelclv.evaluation.plot_utils")

    # The two aliases, and the old canonical names they aliased.
    for name in ("mc_forecast", "mc_forecast_transformer",
                 "run_monte_carlo_forecast", "run_monte_carlo_forecast_transformer"):
        assert not hasattr(models, name), f"{name} should have been retired"
        assert not hasattr(mcf, name), f"{name} should have been retired"

    # The two per-path steppers, renamed by mechanism on the same grounds.
    for name in ("simulate_one_path", "simulate_transformer_path"):
        assert not hasattr(mcf, name), f"{name} should have been retired"

    # Import-only in the notebooks, so nothing called them: a notebook import is
    # not a caller.
    for name in ("weekly_actuals", "weekly_aggregate_predictions", "alignment_check"):
        assert not hasattr(ev, name), f"{name} should have been retired"

    # Prediction I/O and the Pareto/NBD forecast left `evaluation` entirely.
    for name in ("save_predictions_to_csv", "load_predictions_from_csv",
                 "pareto_forecast"):
        assert not hasattr(ev, name), f"{name} now lives outside evaluation"


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


def test_refit_is_the_only_forecast_source():
    """`prediction_source` and the `experiments` subpackage are gone — issue 05.

    ADR-0008 makes the refit on the full calibration window the one way a forecast is
    produced, so `StudySuiteConfig` has no knob choosing between it and the tuning
    checkpoint; a re-added field would be a knob with one legal value again. The
    subpackage rename is guarded in the same place because the old name had no referent
    in `CONTEXT.md`'s vocabulary, and an `experiments` module re-appearing by habit is
    exactly what the rename was for.
    """
    import dataclasses

    from panelclv.studies import StudySuiteConfig
    import panelclv.studies.config as studies_config

    fields = {f.name for f in dataclasses.fields(StudySuiteConfig)}
    assert "prediction_source" not in fields
    assert not hasattr(studies_config, "VALID_PREDICTION_SOURCES")

    with pytest.raises(ImportError):
        importlib.import_module("panelclv.experiments")

    # The two halves the old catch-all split into, under their current names.
    from panelclv.trials import CalibrationSplit, refit_loader, split_calibration  # noqa: F401


def test_a_rollout_model_comes_from_its_trained_model():
    """The rebuild-beside-it path is gone — issue 07 of the package cleanup.

    ADR-0007: a rollout model is obtained from the trained model that holds the
    weights, so the two can never be built with different constructor arguments.
    The names below were the second construction — a tuner-side if-chain and the
    function that drove it from a stored study's parameters — and re-adding either
    would make a mismatch expressible again. The `Inference*` spelling goes with
    them: `CONTEXT.md` lists *inference* under `_Avoid_`.
    """
    import panelclv.benchmarks as benchmarks
    import panelclv.models as models
    import panelclv.trials as trials
    from panelclv.tuning import optuna_tuning

    assert not hasattr(optuna_tuning, "_build_inference_model_for"), \
        "the rollout model comes from the trained model, not a second dispatch site"
    assert not hasattr(trials, "build_inference_from_trial"), \
        "a forecast comes from the refit, which hands back its own rollout model"

    for module in (models, benchmarks):
        retired = [n for n in dir(module) if n.startswith("Inference")]
        assert not retired, f"{module.__name__} still exports {retired}"

    # Every neural family answers the handover, under the name CONTEXT.md defines.
    for trained_cls in (models.MultinomialLSTMModel,
                        models.MultinomialTransformerModel,
                        benchmarks.ValendinLSTMModel):
        assert hasattr(trained_cls, "to_rollout"), trained_cls.__name__


def test_the_model_set_is_enumerated_once():
    """The seven scattered model-type enumerations are gone — issue 06.

    ADR-0006: a model is one entry in `panelclv.registry`, and every list of model
    types derives from that table's keys. The names below were the copies — a
    valid-types list, a neural list, a per-model search-defaults map, a suggester
    map, a builder map, and the suite's own forecaster map — and each re-added one
    would be a second place to register a model, which is the failure that used to
    surface only after training completed.
    """
    from panelclv.registry import MODEL_TYPES, is_neural
    import panelclv.studies.config as studies_config
    import panelclv.studies.runner as runner
    import panelclv.tuning as tuning
    from panelclv.tuning import optuna_tuning

    for name in ("VALID_MODEL_TYPES", "NEURAL_MODEL_TYPES"):
        assert not hasattr(studies_config, name), f"{name} should derive from the registry"
    assert not hasattr(runner, "_FORECASTERS"), \
        "the rollout is declared by the registry entry, not a second map"
    for name in ("_SEARCH_DEFAULTS", "_SUGGESTERS", "_BUILDERS",
                 "LSTM_SEARCH_DEFAULTS", "TRANSFORMER_SEARCH_DEFAULTS",
                 "VALENDIN_SEARCH_DEFAULTS", "validate_data_info"):
        assert not hasattr(optuna_tuning, name), f"{name} should have been retired"
    assert not hasattr(tuning, "validate_data_info"), \
        "the search space is validated against the registry entry that declares it"

    # `pareto_nbd` is IN the table (declaratively), which is what lets both lists
    # derive from it rather than one of them carrying a hand-written addend.
    assert "pareto_nbd" in MODEL_TYPES and not is_neural("pareto_nbd")


def test_model_spec_separates_search_space_from_training():
    """`data_info` carried both and was policed by a hand-maintained allowlist — issue 06.

    Splitting it means a misplaced key lands in the wrong *field*, where the search
    space is validated against the registry entry that declares it, rather than
    against a list of non-search keys someone has to remember to extend.
    """
    import dataclasses

    from panelclv.studies import ModelSpec

    fields = {f.name for f in dataclasses.fields(ModelSpec)}
    assert "data_info" not in fields
    assert {"search_space", "training"} <= fields


def _calls_t_ppf(source: str) -> bool:
    """True if the module calls SciPy's `stats.t.ppf` — the Student-t quantile."""
    import ast

    return any(
        isinstance(node, ast.Attribute)
        and node.attr == "ppf"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "t"
        for node in ast.walk(ast.parse(source))
    )


def test_suite_analysis_is_three_modules_with_one_interval():
    """`studies/analysis.py` split three ways — issue 09 of the package cleanup.

    Size was not the reason: one 1195-line module is what let a hand-written copy of
    the customer-group set and a *second* Student-t interval sit in the same file,
    both found only by audit. The guards are therefore about what the split protects —
    the old module cannot come back, the three that replaced it exist, and the
    t-interval has exactly one implementation, which the plot band and the Pareto
    grid both call.
    """
    import panelclv
    import panelclv.studies as st

    with pytest.raises(ImportError):
        importlib.import_module("panelclv.studies.analysis")
    for name in ("suite_reader", "suite_plots", "suite_metrics"):
        importlib.import_module(f"panelclv.studies.{name}")

    # The public surface is unchanged by the split — notebooks import the subpackage.
    for name in ("load_model_predictions", "aggregate_suite_predictions",
                 "plot_suite_forecast", "group_metrics_suite_table", "study_metrics",
                 "compare_study_metrics", "describe_dataset", "describe_suite_dataset"):
        assert hasattr(st, name), f"{name} should still resolve from panelclv.studies"

    # One Student-t interval in the package: everything else calls it. Read off the
    # syntax tree rather than the text, so prose naming the call is not a false hit.
    src = Path(panelclv.__file__).parent
    callers = {p.name for p in src.rglob("*.py") if _calls_t_ppf(p.read_text())}
    assert callers == {"suite_metrics.py"}, \
        f"the t-interval is implemented in more than one place: {sorted(callers)}"


def test_the_suite_modules_defer_no_imports():
    """The deferrals that claimed to keep the read path torch-free are gone — issue 09.

    They never saved anything: `panelclv.studies` pulls torch at package import, and one
    of the three deferred `pandas`, which cannot affect torch at all. A function-body
    import now reads as what it is — a load-order workaround — rather than as a policy
    these modules follow.
    """
    import ast

    import panelclv

    src = Path(panelclv.__file__).parent / "studies"
    for name in ("suite_reader.py", "suite_plots.py", "suite_metrics.py"):
        tree = ast.parse((src / name).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            deferred = [
                n for n in ast.walk(node)
                if isinstance(n, (ast.Import, ast.ImportFrom))
            ]
            assert not deferred, \
                f"{name}:{node.name} defers an import into its body"


def test_torch_is_not_imported_lazily_anywhere():
    """The torch-free idea is gone, not just unenforced — issue 06.

    Torch is a hard dependency, so deferring an import never bought the ability to
    run without it; `panelclv.benchmarks` carried ~30 lines of PEP 562 lazy loader
    to protect a property `panelclv.studies` does not have anyway.
    """
    import panelclv.benchmarks as benchmarks

    assert not hasattr(benchmarks, "_LAZY")
    assert "__getattr__" not in vars(benchmarks)
    # The names it deferred resolve as ordinary module attributes.
    assert benchmarks.ValendinLSTMModel.__module__.endswith("valendin_lstm")
