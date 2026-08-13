"""Where a run lands is derivable from its config and its seed.

Reproducibility — ``CLAUDE.md`` priority 2 — covers *where* a result is written, not
only what is in it. Three writers used to stamp ``datetime.now()`` into a folder name,
which cost two things at once: finding an earlier run meant knowing what minute it had
been started, and re-running the same config wrote a second folder beside the first
instead of over it.

These pin the property that replaced that. A folder name is built from the arguments
that decide what is inside it, so the same config and seed name the same folder; the
wall-clock time is still recorded, as a metadata field nobody has to know to find
anything. The three sites are the Monte Carlo prediction dump (``models``), the
Pareto/NBD prediction dump (``benchmarks``) and the synthetic study folder
(``data_preparation``).

The two deliberately-excluded sites are asserted here too, because "left alone" is a
decision and an untested decision is one a later sweep silently reverses.
"""

import json
import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from panelclv.benchmarks.pareto_nbd import pareto_forecast
from panelclv.data_preparation.pareto_nbd_simulation import (
    _auto_dataset_dir_name,
    generate_pnbd_study,
)
from panelclv.predictions import RUN_METADATA_FILE, create_run_directory

# Any run of six-plus digits is a date stamp; no derived name has a reason to hold one.
TIMESTAMP = re.compile(r"\d{6,}")


def created_at(run_dir):
    """The recorded wall-clock time of a run, parsed — proving it is a real timestamp."""
    metadata = json.loads((run_dir / RUN_METADATA_FILE).read_text())
    return datetime.fromisoformat(metadata["created_at"])


# --------------------------------------------------------------------------------------
# The shared run folder
# --------------------------------------------------------------------------------------


def test_the_run_folder_is_named_by_its_caller_and_the_clock_goes_in_the_sidecar(tmp_path):
    """`create_run_directory` adds nothing to the name it is given."""
    run_dir = create_run_directory(tmp_path, "lstm_n30_seed7")

    assert run_dir == tmp_path / "lstm_n30_seed7"
    assert created_at(run_dir).year >= 2026


def test_the_same_name_twice_is_the_same_folder(tmp_path):
    """Re-running a config writes over its own run, rather than beside it."""
    first = create_run_directory(tmp_path, "lstm_n30_seed7")
    (first / "predictions.csv").write_text("x\n")
    second = create_run_directory(tmp_path, "lstm_n30_seed7")

    assert first == second
    assert [p.name for p in tmp_path.iterdir()] == ["lstm_n30_seed7"]
    assert (second / "predictions.csv").exists()   # the earlier dump is still reachable


# --------------------------------------------------------------------------------------
# Site 1 — the Monte Carlo prediction dump
# --------------------------------------------------------------------------------------


def test_the_monte_carlo_dump_names_the_run_by_its_simulations_and_seed(tmp_path):
    """`{tag}_n{n_simulations}_seed{seed}` — every token an argument of the call.

    Exercised through the private writer rather than a full rollout: the naming is the
    property under test and a trained model would add minutes without adding evidence.
    """
    from panelclv.models.monte_carlo_forecasting import _save_predictions_run

    prediction_mean = np.array([[1.0, 2.0], [3.0, 4.0]])
    data = {"ids": ["a", "b"], "id_col": "Id"}
    write = lambda: _save_predictions_run(          # noqa: E731 — same call, twice
        prediction_mean, data,
        output_dir=tmp_path, file_name="predictions.csv", run_name=None,
        model_type="lstm", n_simulations=30, seed=7,
    )

    first = write()
    second = write()

    assert first.parent.name == "lstm_n30_seed7"
    assert second == first
    assert created_at(first.parent)


def test_an_unseeded_monte_carlo_run_says_so_in_the_name(tmp_path):
    """`seed=None` is fresh randomness every call — the name records that, not a clock."""
    from panelclv.models.monte_carlo_forecasting import _save_predictions_run

    path = _save_predictions_run(
        np.array([[1.0]]), {"ids": ["a"], "id_col": "Id"},
        output_dir=tmp_path, file_name="predictions.csv", run_name="study_a",
        model_type="transformer", n_simulations=5, seed=None,
    )

    assert path.parent.name == "study_a_n5_seedNone"


# --------------------------------------------------------------------------------------
# Site 2 — the Pareto/NBD prediction dump
# --------------------------------------------------------------------------------------


@pytest.fixture
def pareto_data():
    """A minimal `prepare_dataset`-shaped dict: 12 customers, 8 weeks, 3 held out.

    Hand-built rather than run through `prepare_dataset`, because the fit is only here
    to reach the writer — the benchmark's own numbers are pinned by the golden test.
    """
    rng = np.random.default_rng(20260813)
    n_customers, n_periods, t_holdout = 12, 8, 3
    dates = pd.date_range("2026-01-01", periods=n_periods, freq="7D")
    ids = [f"c{i}" for i in range(n_customers)]

    panel = pd.DataFrame({
        "Id": np.repeat(ids, n_periods),
        "period_start": np.tile(dates, n_customers),
        "Transactions": rng.poisson(1.0, n_customers * n_periods).astype(float),
    })
    calibration = panel[panel["period_start"] < dates[n_periods - t_holdout]]
    holdout_counts = (
        panel[panel["period_start"] >= dates[n_periods - t_holdout]]["Transactions"]
        .to_numpy().reshape(n_customers, t_holdout)
    )
    return {
        "train_panel": calibration,
        "T_HOLD": t_holdout,
        "ids": ids,
        "id_col": "Id",
        "target_col": "Transactions",
        "frequency": "weekly",
        # (N, T_HOLD, F=1): the target is the only channel, so target_idx is 0.
        "holdout": holdout_counts[:, :, None],
        "target_idx": 0,
    }


def test_the_pareto_dump_names_the_run_by_its_fit_seed(pareto_data, tmp_path):
    """`{tag}_seed{seed}`, with the seed the MCMC chain actually ran on."""
    chain = {"mcmc": 40, "burnin": 10, "thin": 10, "chains": 1}
    result = pareto_forecast(
        pareto_data, save_predictions=True, output_dir=tmp_path, seed=99, **chain
    )

    run_dir = result["predictions_path"].parent
    assert run_dir.name == "pareto_seed99"
    assert created_at(run_dir)


def test_the_pareto_dump_names_the_default_seed_the_fit_used(pareto_data, tmp_path):
    """Passing no seed still names one — the sampler's default, declared in one place.

    A folder that omitted the seed when the caller did would be claiming the fit was
    unseeded, and it never is: `compute_pareto_predictions` supplies its own.
    """
    result = pareto_forecast(
        pareto_data, save_predictions=True, output_dir=tmp_path,
        mcmc=40, burnin=10, thin=10, chains=1,
    )

    assert result["predictions_path"].parent.name == "pareto_seed42"


# --------------------------------------------------------------------------------------
# Site 3 — the synthetic dataset directory
# --------------------------------------------------------------------------------------


def test_the_dataset_dir_name_is_the_grid_shape_and_the_base_seed():
    """The grid and the seed decide every dataset in the directory, so they name it."""
    name = _auto_dataset_dir_name(6, 4, 5, base_seed=42)

    assert name == "pnbd_study_6x4x5_seed42"
    assert not TIMESTAMP.search(name)
    assert _auto_dataset_dir_name(6, 4, 5, base_seed=43) != name   # a different grid, told apart


def test_regenerating_a_study_writes_to_the_same_folder_and_keeps_its_creation_time(tmp_path):
    """The end-to-end property: same call twice, one folder, timestamp still recorded."""
    grid = dict(
        mean_transaction_rates=[0.5], churn_rates=[0.2],
        n_customers=8, n_weeks=12, n_datasets=1, out_path=tmp_path, base_seed=7,
    )

    first, _ = generate_pnbd_study(**grid)
    second, _ = generate_pnbd_study(**grid)

    assert first == second == tmp_path / "pnbd_study_1x1x1_seed7"
    assert [p.name for p in tmp_path.iterdir()] == ["pnbd_study_1x1x1_seed7"]
    study_config = json.loads((second / "study_config.json").read_text())
    assert datetime.fromisoformat(study_config["created_at"])


# --------------------------------------------------------------------------------------
# The two sites left alone, on purpose
# --------------------------------------------------------------------------------------


def test_the_suite_record_and_the_tuner_keep_their_timestamps():
    """Neither of these puts the clock in a path, so neither was swept.

    The suite runner stamps a `created` field into its root `config.json` — provenance
    in metadata, which is exactly the shape the three sites above were moved *to*. The
    tuner can append a timestamp to a run name but the runner already opts out, so no
    archived study path depends on the clock. Pinned because a later pattern-match over
    `datetime.now()` would otherwise read both as leftovers.
    """
    from panelclv.studies import runner
    from panelclv.tuning import optuna_tuning

    suite_record = runner._suite_record.__doc__
    assert "config.json" in suite_record

    import inspect
    assert "append_timestamp=False" in inspect.getsource(runner)
    assert "append_timestamp" in inspect.signature(optuna_tuning.run_optuna_study).parameters
