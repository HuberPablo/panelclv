"""One study suite, run for real, from panel to archive and back.

`test_golden_end_to_end.py` pins what the pipeline *computes*, but it builds its models
by calling `fit_model` directly. That skips the entire orchestration layer — the Optuna
search, the refit ADR-0008 says every forecast comes from, and the runner that lays the
archive out — which is the code that produced the thesis's numbers and, until this file,
the least covered code in the package. A suite-level break therefore surfaced only when
someone ran a real suite, hours in.

What this file adds over the golden test:

    run_optuna_study -> refit_best_trial -> refit_full_calibration -> run_study_suite

plus the reader that opens the archive again. It is deliberately a *structural* test:
it asserts the shapes, the row counts, and one invariant (below), never a pinned metric.
Values belong to the golden test, which is set up to be re-baselined deliberately; a
second set of pinned numbers here would be a second thing to re-baseline and would fail
for the same reasons.

**The invariant worth the runtime**: `results.csv` is written during training, and
`study_metrics` re-scores the *stored forecasts* off disk afterwards. Both go through
`compute_forecast_metrics`, which `CLAUDE.md` names the single scoring authority. If the
prediction CSV round-trips losslessly and the authority really is single, the two agree
exactly. A second scoring path, or a lossy write, breaks this and nothing else catches it.

Budgets are small but not degenerate: 3 studies per neural model, so the across-studies
spread is a real distribution rather than a single point; 2 trials, so Optuna actually
selects between trials rather than keeping the only one; 5 epochs, so early stopping and
the refit's epoch count have something to act on. The panel is the golden 23-customer
cohort, reused so there is one synthetic panel definition in the suite rather than two.

Run:  pytest -q tests/test_study_suite_end_to_end.py
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from panelclv.data_preparation import panel_dataset
from panelclv.studies import (
    ModelSpec,
    StudySuiteConfig,
    aggregate_suite_predictions,
    run_study_suite,
    study_metrics,
)

# The golden panel and config, imported rather than re-declared. `pythonpath = ["src"]`
# plus pytest's rootdir insertion put `tests/` on the path; `scripts/
# trace_golden_reachability.py` imports from this same module for the same reason.
from test_golden_end_to_end import _golden_config, _golden_panel

N_STUDIES = 3          # > 1, so `study_metrics` reports a real std over replications
N_TRIALS = 2           # > 1, so the Optuna search has something to choose between
N_EPOCHS = 5
N_SIMULATIONS = 8      # rollout paths; the golden test's count, for the same reason

NEURAL_MODELS = ("LSTM", "Transformer")
ALL_MODELS = (*NEURAL_MODELS, "ParetoNBD")


@pytest.fixture(scope="module")
def suite(tmp_path_factory) -> dict:
    """One finished suite on disk, plus the panel CSV needed to read it back.

    Module-scoped: this is the expensive fixture in the file, and every test below is an
    assertion about the same finished archive rather than about a fresh run.
    """
    panel = _golden_panel()
    data = panel_dataset.prepare_dataset(panel, _golden_config(), verbose=False)

    base = tmp_path_factory.mktemp("studies")
    panel_csv = base / "panel.csv"
    panel.to_csv(panel_csv, index=False)

    # Training controls only. The search spaces are left alone so each model samples the
    # ranges its registry entry declares — the path a real run takes (ADR-0006).
    training = {"n_epochs": N_EPOCHS, "patience": 2, "verbose": False}
    config = StudySuiteConfig(
        studies_base_path=str(base),
        suite_name="suite",
        n_studies_per_model=N_STUDIES,
        n_simulations=N_SIMULATIONS,
        device="cpu",                      # CPU keeps the run reproducible and GPU-free
        data=data,
        models=[
            ModelSpec(name="LSTM", model_type="lstm",
                      n_trials=N_TRIALS, training=training),
            ModelSpec(name="Transformer", model_type="transformer",
                      n_trials=N_TRIALS, training=training),
            # Short chains: this records which code runs, not whether it has converged —
            # the same trade the golden test's Pareto arm makes.
            ModelSpec(name="ParetoNBD", model_type="pareto_nbd",
                      pareto_kwargs={"mcmc": 200, "burnin": 50,
                                     "thin": 10, "chains": 1}),
        ],
    )
    root = run_study_suite(config)
    return {"root": root, "data": data, "panel_csv": panel_csv}


def test_the_suite_writes_one_row_per_model_and_study(suite):
    """`results.csv` is the table every downstream reader starts from.

    The Pareto/NBD baseline is deterministic given its seed, so the runner coerces it to
    one study however many the config asks for — that coercion is part of the contract
    and is asserted here rather than assumed.
    """
    results = pd.read_csv(suite["root"] / "results.csv")

    counts = results.groupby("model").size().to_dict()
    assert counts == {"LSTM": N_STUDIES, "Transformer": N_STUDIES, "ParetoNBD": 1}
    assert set(results.columns) >= {"model", "model_type", "study", "seed",
                                    "rmse", "bias_percent", "mape_aggregate"}
    # Every metric is a real number: a silently-NaN row would sail through a shape check
    # and poison every average taken downstream.
    assert results[["rmse", "mape_aggregate"]].notna().all().all()


def test_every_study_gets_its_own_seed_and_checkpoint(suite):
    """Studies are independent replications, not the same run counted N times.

    `n_studies_per_model` only means something if the studies differ. The runner owns
    the seed and the checkpoint directory per study for exactly this reason, so a shared
    seed (or a shared checkpoint dir, which would have them overwrite each other) makes
    the across-studies spread a lie.
    """
    results = pd.read_csv(suite["root"] / "results.csv")

    for model in NEURAL_MODELS:
        seeds = results.loc[results["model"] == model, "seed"].tolist()
        assert len(set(seeds)) == N_STUDIES, f"{model} reused a seed: {seeds}"

        optuna_dir = suite["root"] / model / "Optuna_Studies"
        studies = sorted(p.name for p in optuna_dir.iterdir() if p.is_dir())
        assert studies == [f"study_{i:02d}" for i in range(1, N_STUDIES + 1)]


def test_each_study_produces_one_forecast_of_the_holdout_shape(suite):
    """A prediction file per study, one row per customer, one column per holdout period."""
    data = suite["data"]
    n_customers, t_holdout = len(data["ids"]), int(data["T_HOLD"])

    for model in ALL_MODELS:
        expected = N_STUDIES if model in NEURAL_MODELS else 1
        files = sorted((suite["root"] / model / "Predictions").glob("Prediction_*.csv"))
        assert len(files) == expected, f"{model} wrote {len(files)} forecasts"

        for path in files:
            forecast = pd.read_csv(path)
            # id column + one column per holdout period.
            assert forecast.shape == (n_customers, t_holdout + 1), path.name


def test_the_suite_records_the_recipe_it_ran_on(suite):
    """`config.json` carries the `PanelConfig`, which is what makes a suite re-scorable.

    `suite_reader` rebuilds the dataset from this to compute holdout actuals. Without it
    a finished suite is a folder of numbers no one can check.
    """
    record = json.loads((suite["root"] / "config.json").read_text())

    assert record["suite_name"] == "suite"
    assert record["panel_config"]["target_col"] == "Transactions"
    assert [m["name"] for m in record["models"]] == list(ALL_MODELS)


def test_rescoring_the_stored_forecasts_reproduces_the_training_time_metrics(suite):
    """The invariant this file is worth its runtime for — see the module docstring.

    `results.csv` was written while training; `study_metrics` reads the prediction CSVs
    back off disk and scores them again. One scoring authority plus a lossless write
    means the two agree, so this is asserted tightly rather than approximately.
    """
    stored = pd.read_csv(suite["root"] / "results.csv")
    rescored = study_metrics(suite["root"], suite["panel_csv"])

    # Indexed by model, one column per metric holding the across-studies mean. (The
    # columns become a (metric, statistic) MultiIndex only when `standard_deviation`
    # or `confidence_interval` is asked for — see the spread test below.)
    for model in ALL_MODELS:
        for metric in ("rmse", "bias_percent", "mape_aggregate"):
            expected = stored.loc[stored["model"] == model, metric].mean()
            actual = rescored.loc[model, metric]
            assert actual == pytest.approx(expected, rel=1e-9), (
                f"{model}/{metric}: the archive re-scores to {actual}, but training "
                f"wrote {expected} — two scoring paths, or a lossy prediction write"
            )


def test_the_across_studies_spread_is_reported_per_model(suite):
    """Three studies give a distribution; the reader must not collapse it to one number.

    A merged-ledger finding this guards: a drifted model-type list once collapsed the
    Valendin benchmark's spread to a single study, silently. `n` is the check — it says
    how many studies were actually folded in.
    """
    rescored = study_metrics(suite["root"], suite["panel_csv"], standard_deviation=True)

    for model in NEURAL_MODELS:
        assert rescored.loc[model, ("rmse", "n")] == N_STUDIES
        # Independent seeds on a 5-epoch model do not land on identical weights, so the
        # spread is non-zero. A zero std here means the studies were not independent.
        assert rescored.loc[model, ("rmse", "std")] > 0

    # The deterministic baseline has one study and therefore no spread to report.
    assert rescored.loc["ParetoNBD", ("rmse", "n")] == 1


def test_the_suite_aggregates_each_models_studies_into_one_forecast(suite):
    """`aggregate_suite_predictions` averages a model's studies into one wide CSV."""
    written = aggregate_suite_predictions(suite["root"])
    assert sorted(p.name for p in written) == sorted(
        f"aggregated_{m}.csv" for m in ALL_MODELS
    )

    data = suite["data"]
    for path in written:
        assert pd.read_csv(path).shape == (len(data["ids"]), int(data["T_HOLD"]) + 1)
