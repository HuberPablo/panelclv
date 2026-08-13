"""Read-path coverage for the study-suite layout, reader and metrics.

A finished suite is a directory tree: a `config.json` recording the recipe the run used,
one folder per model, and a wide `Prediction_{i}.csv` per study. Three modules address
that tree — `studies/layout.py` names its paths, `studies/suite_reader.py` discovers the
models and loads their forecasts back, and `studies/suite_metrics.py` rescores those
forecasts against actuals rebuilt from the stored recipe. Apart from `layout`'s writer
half, covered in `test_studies_layout.py`, this file is the only coverage any of them has
— and the only place a suite is read that the package did not write in the same breath.

**How the fixture is built is the point.** The suite below is assembled with
`Path.write_text` from literal text — never through `save_predictions_to_csv` /
`layout.write_json`. A fixture written by the package's own writers would still pass if
writer and reader drifted together; against literal bytes it cannot. One test then drives
the real writer (`layout.prediction_path` + `save_predictions_to_csv`) and compares its
output to that same literal header, which closes the other direction: the reader is
checked against text, the writer against what the reader expects.

`Studies/` and `Predictions/` are both gitignored, so the fixture is built into `tmp_path`
from the constants below rather than committed as a file tree, which would be silently
untracked. Nothing here reads the real archive under `Studies/`. Archived suites are no
longer required to stay re-readable, so what is under test is the path today's studies are
written and read through, not a frozen on-disk format.

**Torch.** `load_predictions_from_csv` is torch-free — it lives in the
`panelclv.predictions` leaf — but `panelclv.studies` pulls torch in at package import
through the model registry, so the import at the top of this file is what needs it, and
every test here needs it equally.

Run:  PYTHONPATH=src pytest -q tests/test_suite_read_path.py

To regenerate the fixture metrics after a *deliberate* change, run with
``PANELCLV_PRINT_FIXTURE_METRICS=1`` and paste the printed block over `FIXTURE_METRICS`.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panelclv.configs.panel_config import PanelConfig
from panelclv.studies import layout, suite_metrics, suite_reader


# --------------------------------------------------------------------------------------
# The format, as constants: what the fixture below is written to and what the
# assertions compare against.
# --------------------------------------------------------------------------------------

# `results.csv`'s fixed leading columns, in order, followed by a union of `param_*`
# hyperparameter columns (NaN where a column does not apply to a model).
RESULTS_LEADING_COLS = [
    "model",
    "model_type",
    "study",
    "seed",
    "objective",
    "rmse",
    "bias_percent",
    "mape_aggregate",
]
# The three metric names the package writes today — `compute_forecast_metrics`' own
# keys, which `suite_metrics._STUDY_METRIC_COLS` and `pareto_nbd_grid._METRIC_SOURCE` both name.
RESULTS_METRIC_COLS = ["rmse", "bias_percent", "mape_aggregate"]

# A prediction file: `Prediction_{i}.csv`, wide, `<id_col>` then `week_0..week_{T-1}`.
PREDICTION_FILE_RE = re.compile(r"^Prediction_(\d+)\.csv$")
FIXTURE_ID_COL = "Id"          # what the fixture's Prediction_*.csv files are headed


def prediction_header(n_weeks: int, id_col: str = FIXTURE_ID_COL) -> str:
    """The exact first line of a `Prediction_*.csv` with `n_weeks` holdout periods."""
    return ",".join([id_col] + [f"week_{i}" for i in range(n_weeks)])


# Keys a suite `config.json` carries in both generations of the format. The reader
# requires nothing beyond these, which is what lets it open a suite older than itself.
SUITE_CONFIG_KEYS = {
    "study_name",
    "created",
    "studies_base_path",
    "n_studies_per_model",
    "prediction_source",
    "n_simulations",
    "base_seed",
    "device",
    "refit_kwargs",
    "models",
    "data_summary",
}
# Keys only the current generation carries — the `legacy_suite` fixture below drops
# them, which is why the reader must not require them.
SUITE_CONFIG_KEYS_CURRENT = {"overwrite", "keep_only_best_checkpoint", "panel_config"}

DATA_SUMMARY_KEYS = {
    "n_customers", "T_CAL", "T_HOLD", "F", "seq_cols", "target_col", "validation_start",
}
# Keys a per-model `config.json` carries. `prediction_source` is among them even though
# the writer stopped emitting it when the refit became the only forecast source
# (ADR-0008): suites written before that still open, and nothing in the reader needs it
# gone.
MODEL_CONFIG_KEYS = {
    "name", "model_type", "prediction_source", "n_simulations", "base_seed", "device",
}

# --------------------------------------------------------------------------------------
# The fixture: one suite, written as literal bytes.
#
# Four customers, seven calibration weeks, four holdout weeks, two models — an `lstm` with
# two studies and a `pareto_nbd` benchmark with one, which is the shape a real suite has.
# The panel below is the miniature a suite's `panel_config` describes: an `Id` column,
# `year`/`week` time columns, integer
# `Transactions`. Prediction values are all exactly representable in binary, so the
# text -> float64 conversion is lossless and the averaging assertion is exact.
# --------------------------------------------------------------------------------------

FIXTURE_IDS = ["C01", "C02", "C03", "C04"]
FIXTURE_T_CAL = 7
FIXTURE_T_HOLD = 4

FIXTURE_PANEL_CSV = """Id,year,week,Transactions
C01,2000,1,1
C01,2000,2,0
C01,2000,3,2
C01,2000,4,1
C01,2000,5,0
C01,2000,6,1
C01,2000,7,3
C01,2000,8,0
C01,2000,9,1
C01,2000,10,2
C01,2000,11,0
C01,2000,12,1
C02,2000,1,0
C02,2000,2,1
C02,2000,3,0
C02,2000,4,0
C02,2000,5,2
C02,2000,6,0
C02,2000,7,1
C02,2000,8,1
C02,2000,9,0
C02,2000,10,1
C02,2000,11,1
C02,2000,12,0
C03,2000,1,2
C03,2000,2,2
C03,2000,3,1
C03,2000,4,3
C03,2000,5,0
C03,2000,6,1
C03,2000,7,0
C03,2000,8,2
C03,2000,9,2
C03,2000,10,0
C03,2000,11,1
C03,2000,12,3
C04,2000,1,0
C04,2000,2,0
C04,2000,3,1
C04,2000,4,0
C04,2000,5,1
C04,2000,6,0
C04,2000,7,0
C04,2000,8,1
C04,2000,9,1
C04,2000,10,0
C04,2000,11,0
C04,2000,12,0
"""

# `PanelConfig.to_dict()` as the runner persists it. Every key here is a constructor
# argument, which is what makes `PanelConfig.from_dict` the exact inverse; the field set
# and the value types match a real suite's `panel_config` one for one.
FIXTURE_PANEL_CONFIG = {
    "id_col": "Id",
    "target_col": "Transactions",
    "frequency": "weekly",
    "training_start": "2000-01-03",
    "training_end": "2000-02-20",
    "validation_start": "2000-02-07",
    "holdout_start": "2000-02-21",
    "holdout_end": "2000-03-19",
    "time_cols": ["year", "week"],
    "date_col": None,
    "periods_per_year": 52,
    "clip_target_upper": 3,
    "require_calibration_activity": True,
    "time": [],
    "known_future": [],
    "observed_past": [],
    "static": [],
    "time_features": {"add_year_idx": True, "add_week_sin_cos": True},
    "ar_features": [],
    "embedded_cols": {"Transactions": "auto"},
}

FIXTURE_SUITE_CONFIG = {
    "study_name": "fixture_suite",
    "created": "2026-01-01T00:00:00",
    "studies_base_path": "Studies",
    "n_studies_per_model": 2,
    "prediction_source": "refit",
    "n_simulations": 8,
    "base_seed": 42,
    "device": "cpu",
    "refit_kwargs": {},
    "overwrite": False,
    "keep_only_best_checkpoint": False,
    "panel_config": FIXTURE_PANEL_CONFIG,
    "models": [
        {
            "name": "LSTM",
            "model_type": "lstm",
            "n_trials": 1,
            "data_info": {"n_epochs": 1},
            "pareto_kwargs": {},
        },
        {
            "name": "ParetoNBD_MLE",
            "model_type": "pareto_nbd",
            "n_trials": 1,
            "data_info": {},
            "pareto_kwargs": {},
        },
    ],
    "data_summary": {
        "n_customers": 4,
        "T_CAL": FIXTURE_T_CAL,
        "T_HOLD": FIXTURE_T_HOLD,
        "F": 3,
        "seq_cols": ["Transactions", "week_sin", "week_cos"],
        "target_col": "Transactions",
        "validation_start": "2000-02-07",
    },
}

# Per-model `config.json`, the two shapes `runner._model_record` produced when these
# suites were written.
FIXTURE_MODEL_CONFIGS = {
    "LSTM": {
        "name": "LSTM",
        "model_type": "lstm",
        "prediction_source": "refit",
        "n_simulations": 8,
        "base_seed": 42,
        "device": "cpu",
        "n_trials": 1,
        "data_info": {"n_epochs": 1},
        "refit_kwargs": {},
        "seeds": [43, 44],
    },
    "ParetoNBD_MLE": {
        "name": "ParetoNBD_MLE",
        "model_type": "pareto_nbd",
        "prediction_source": "refit",
        "n_simulations": 8,
        "base_seed": 42,
        "device": "cpu",
        "pareto_kwargs": {},
    },
}

# `<Model>/Predictions/Prediction_{i}.csv`. LSTM study 2 is study 1 plus exactly 0.25 in
# every cell, so the across-studies mean is study 1 + 0.125 with no float slack.
FIXTURE_PREDICTIONS = {
    ("LSTM", 1): """Id,week_0,week_1,week_2,week_3
C01,0.5,1.25,1.75,0.25
C02,0.75,0.5,1.0,0.75
C03,1.5,1.75,0.5,1.25
C04,0.25,0.75,0.25,0.5
""",
    ("LSTM", 2): """Id,week_0,week_1,week_2,week_3
C01,0.75,1.5,2.0,0.5
C02,1.0,0.75,1.25,1.0
C03,1.75,2.0,0.75,1.5
C04,0.5,1.0,0.5,0.75
""",
    ("ParetoNBD_MLE", 1): """Id,week_0,week_1,week_2,week_3
C01,1.0,1.0,1.0,1.0
C02,0.5,0.5,0.5,0.5
C03,1.5,1.5,1.5,1.5
C04,0.25,0.25,0.25,0.25
""",
}

# The metrics the *writer* stored for those predictions — i.e. what the runner would have
# put in `results.csv` and `metrics.csv`. `test_fixture_study_metrics_reproduce_results_csv`
# recomputes them from the prediction files and requires agreement — i.e. that the two
# producers of a suite's numbers, the runner and the reader, still say the same thing.
FIXTURE_METRICS = {
    ("LSTM", 1): {
        "rmse": 0.385275875185561,
        "bias_percent": 3.8461538461538463,
        "mape_aggregate": 19.23076923076923,
    },
    ("LSTM", 2): {
        "rmse": 0.4759858191164943,
        "bias_percent": 34.61538461538461,
        "mape_aggregate": 34.61538461538461,
    },
    ("ParetoNBD_MLE", 1): {
        "rmse": 0.7180703308172536,
        "bias_percent": 0.0,
        "mape_aggregate": 23.076923076923077,
    },
}

# `results.csv` columns beyond the fixed leading ones: the union of every model's
# `param_*` hyperparameters, blank on the rows where a column does not apply. Only the
# LSTM has any; the benchmark's row leaves it empty, exactly as pandas' column union does.
FIXTURE_RESULTS_PARAM_COLS = {"LSTM": ["param_embedding_dim"], "ParetoNBD_MLE": []}
FIXTURE_RESULTS_PARAM_UNION = ["param_embedding_dim"]

# One row per (model, study): (model, model_type, study, seed, objective). The benchmark
# has no Optuna objective, which the writer stores as NaN — i.e. an empty CSV field.
FIXTURE_RESULTS_ROWS = [
    ("LSTM", "lstm", 1, "43.0", "0.7739147039560171"),
    ("LSTM", "lstm", 2, "44.0", "0.7612290123456789"),
    ("ParetoNBD_MLE", "pareto_nbd", 1, "42", ""),
]


def _results_row_text(name: str, mtype: str, study: int, seed: str, objective: str,
                      param_cols: list[str]) -> str:
    """One `results.csv` / `metrics.csv` line for `(name, study)`."""
    metrics = FIXTURE_METRICS[(name, study)]
    params = ["64.0" if c == "param_embedding_dim" and name == "LSTM" else "" for c in param_cols]
    return ",".join(
        [name, mtype, str(study), seed, objective]
        + [repr(metrics[c]) for c in RESULTS_METRIC_COLS]
        + params
    )


def _results_csv_text() -> str:
    """`results.csv` for the fixture, in `runner`'s one-row-per-(model, study) shape."""
    lines = [",".join(RESULTS_LEADING_COLS + FIXTURE_RESULTS_PARAM_UNION)]
    lines += [_results_row_text(*row, FIXTURE_RESULTS_PARAM_UNION) for row in FIXTURE_RESULTS_ROWS]
    return "\n".join(lines) + "\n"


def _metrics_csv_text(name: str) -> str:
    """`<Model>/metrics.csv` — the same rows, but only this model's own `param_*` columns."""
    param_cols = FIXTURE_RESULTS_PARAM_COLS[name]
    lines = [",".join(RESULTS_LEADING_COLS + param_cols)]
    lines += [
        _results_row_text(*row, param_cols)
        for row in FIXTURE_RESULTS_ROWS
        if row[0] == name
    ]
    return "\n".join(lines) + "\n"


def _write_suite(root: Path, suite_config: dict) -> Path:
    """Write the fixture tree under `root` using only `write_text` — no package writers.

    Mirrors `studies/layout`'s documented tree: a suite `config.json` + `results.csv`, then
    per model a `config.json`, a `metrics.csv`, a `Predictions/` folder and (neural models
    only) `Optuna_Studies/study_{i:02d}/`.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(suite_config, indent=2))
    (root / "results.csv").write_text(_results_csv_text())

    for spec in suite_config["models"]:
        name = spec["name"]
        model_dir = root / name
        (model_dir / "Predictions").mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(
            json.dumps(FIXTURE_MODEL_CONFIGS[name], indent=2)
        )
        # `metrics.csv` is this model's slice of `results.csv` (its rows, its own `param_*`
        # columns) — the runner writes it per model and nothing in `src/` reads it back.
        (model_dir / "metrics.csv").write_text(_metrics_csv_text(name))

        for (model_name, index), text in FIXTURE_PREDICTIONS.items():
            if model_name == name:
                (model_dir / "Predictions" / f"Prediction_{index}.csv").write_text(text)

        if spec["model_type"] != "pareto_nbd":
            for index in (1, 2):
                sdir = layout.study_dir(model_dir, index)
                (sdir / "checkpoints").mkdir(parents=True, exist_ok=True)
                name_i = f"study_{index:02d}"
                (sdir / f"{name_i}_best.json").write_text(
                    json.dumps({"number": 0, "value": 0.5, "params": {"embedding_dim": 64}})
                )
                (sdir / f"{name_i}_trials.csv").write_text(
                    "number,value,params_embedding_dim\n0,0.5,64\n"
                )
    return root


@pytest.fixture
def suite(tmp_path) -> Path:
    """The current-generation fixture suite: `config.json` carries a `panel_config`."""
    return _write_suite(tmp_path / "fixture_suite", FIXTURE_SUITE_CONFIG)


@pytest.fixture
def legacy_suite(tmp_path) -> Path:
    """A pre-`panel_config` suite — the shape suites written before that key have."""
    config = {
        k: v
        for k, v in FIXTURE_SUITE_CONFIG.items()
        if k not in SUITE_CONFIG_KEYS_CURRENT
    }
    return _write_suite(tmp_path / "legacy_suite", config)


@pytest.fixture
def panel_csv(tmp_path) -> Path:
    """The miniature panel the fixture's `panel_config` describes."""
    path = tmp_path / "panel.csv"
    path.write_text(FIXTURE_PANEL_CSV)
    return path


# --------------------------------------------------------------------------------------
# 1. The tree
# --------------------------------------------------------------------------------------


def test_layout_helpers_address_the_fixture_tree(suite):
    """`layout`'s path helpers point at files that exist in a suite it did not write.

    `layout` is the module that *defines* the tree; this asserts its definition still
    matches the bytes on disk rather than only being self-consistent.
    """
    model_dir = suite / "LSTM"
    assert (suite / "config.json").is_file()
    assert (suite / "results.csv").is_file()
    assert (model_dir / "config.json").is_file()
    assert (model_dir / "metrics.csv").is_file()
    assert layout.prediction_path(model_dir, 1).is_file()
    assert layout.prediction_path(model_dir, 2).is_file()
    assert layout.study_dir(model_dir, 1).is_dir()
    assert (layout.study_dir(model_dir, 1) / "study_01_best.json").is_file()
    assert (layout.study_dir(model_dir, 1) / "study_01_trials.csv").is_file()
    # The benchmark has one prediction and no Optuna stage.
    assert layout.prediction_path(suite / "ParetoNBD_MLE", 1).is_file()
    assert not (suite / "ParetoNBD_MLE" / "Optuna_Studies").exists()


def test_discover_models_follows_config_order(suite):
    """Model order comes from `config.json`, not the filesystem — plots stay reproducible."""
    found = suite_reader._discover_models(suite)
    assert [name for name, _ in found] == ["LSTM", "ParetoNBD_MLE"]
    assert [d for _, d in found] == [suite / "LSTM", suite / "ParetoNBD_MLE"]


def test_deterministic_model_is_read_from_its_model_type(suite):
    """`pareto_nbd` is a single-fit benchmark; the neural family is not."""
    assert suite_reader._is_deterministic_model(suite / "ParetoNBD_MLE")
    assert not suite_reader._is_deterministic_model(suite / "LSTM")


def test_prediction_file_names_and_headers(suite):
    """`Prediction_{i}.csv`, headed `Id,week_0..week_{T_HOLD-1}`, one row per customer."""
    expected = prediction_header(FIXTURE_T_HOLD)
    for name in ("LSTM", "ParetoNBD_MLE"):
        paths = sorted((suite / name / "Predictions").glob("*.csv"))
        assert paths, name
        for path in paths:
            assert PREDICTION_FILE_RE.match(path.name), path.name
            lines = path.read_text().splitlines()
            assert lines[0] == expected
            assert len(lines) == 1 + len(FIXTURE_IDS)
    # The sort key is numeric, so Prediction_10 would follow Prediction_2, not precede it.
    assert suite_reader._prediction_index(Path("Prediction_10.csv")) == 10


def test_results_csv_column_contract(suite):
    """`results.csv` leads with the fixed columns; the rest are `param_*` hyperparameters."""
    df = pd.read_csv(suite / "results.csv")
    assert list(df.columns) == RESULTS_LEADING_COLS + FIXTURE_RESULTS_PARAM_UNION
    assert all(c.startswith("param_") for c in df.columns[len(RESULTS_LEADING_COLS):])
    # The three metric names `suite_metrics._STUDY_METRIC_COLS` and
    # `pareto_nbd_grid._METRIC_SOURCE`
    # read out of a suite this package wrote.
    assert suite_metrics._STUDY_METRIC_COLS == RESULTS_METRIC_COLS
    from panelclv.studies.pareto_nbd_grid import _METRIC_SOURCE
    assert set(_METRIC_SOURCE.values()) <= set(RESULTS_METRIC_COLS)
    # One row per (model, study): two LSTM studies, one benchmark fit.
    assert list(df["model"]) == ["LSTM", "LSTM", "ParetoNBD_MLE"]
    assert list(df["study"]) == [1, 2, 1]


def test_model_metrics_csv_is_a_slice_of_results_csv(suite):
    """Each `<Model>/metrics.csv` carries that model's rows, columns ⊆ `results.csv`'s."""
    results = pd.read_csv(suite / "results.csv")
    for name in ("LSTM", "ParetoNBD_MLE"):
        mine = pd.read_csv(suite / name / "metrics.csv")
        assert set(mine.columns) <= set(results.columns)
        assert set(mine["model"]) == {name}
        assert list(mine.columns[: len(RESULTS_LEADING_COLS)]) == RESULTS_LEADING_COLS


def test_model_config_json_records_its_type(suite):
    """The per-model record carries the keys the suite reader needs."""
    for name in ("LSTM", "ParetoNBD_MLE"):
        record = json.loads((suite / name / "config.json").read_text())
        assert MODEL_CONFIG_KEYS <= set(record)
        assert record["name"] == name


def test_suite_config_json_round_trips_to_a_panel_config(suite):
    """`config.json` -> `PanelConfig` -> dict is the identity, which the reader depends on.

    `_actuals_from_panel` rebuilds the dataset from this stored recipe, so a finished
    suite is only rescorable while `PanelConfig.from_dict` accepts what was written.
    Asserted as an exact dict round-trip: a renamed or dropped field breaks it
    immediately.
    """
    stored = json.loads((suite / "config.json").read_text())
    assert SUITE_CONFIG_KEYS <= set(stored)
    assert SUITE_CONFIG_KEYS_CURRENT <= set(stored)
    assert DATA_SUMMARY_KEYS <= set(stored["data_summary"])

    panel_config = PanelConfig.from_dict(stored["panel_config"])
    assert panel_config.to_dict() == stored["panel_config"]
    assert panel_config.id_col == FIXTURE_ID_COL
    assert panel_config.target_col == "Transactions"
    # `clip_target_upper` sets the softmax head size, so it is part of the recipe.
    assert panel_config.clip_target_upper == 3


def test_legacy_suite_config_is_still_discoverable(legacy_suite):
    """A suite whose config predates `panel_config` still lists its models to the reader.

    `_discover_models` reads `config.json`'s `models` list, so it must not require the keys
    that arrived later.
    """
    stored = json.loads((legacy_suite / "config.json").read_text())
    assert SUITE_CONFIG_KEYS <= set(stored)
    assert not (SUITE_CONFIG_KEYS_CURRENT & set(stored))
    assert [name for name, _ in suite_reader._discover_models(legacy_suite)] == [
        "LSTM", "ParetoNBD_MLE",
    ]


def test_id_col_comes_from_the_stored_panel_config(suite):
    """The id column saved CSVs are headed with is read out of the suite's own recipe.

    Not hard-coded and not sniffed from the files: `aggregate_suite_predictions` heads
    what it writes with whatever `panel_config.id_col` the run recorded.
    """
    assert suite_reader._id_col(suite) == FIXTURE_ID_COL


# --------------------------------------------------------------------------------------
# 2. Reading predictions back
# --------------------------------------------------------------------------------------


def test_predictions_load_back_with_ids_and_values(suite):
    """One study's file becomes `(values (N, T_HOLD), ids)` with the ids in cohort order."""
    values, ids = suite_reader.load_model_predictions(suite / "LSTM", study=1)
    assert values.shape == (len(FIXTURE_IDS), FIXTURE_T_HOLD)
    assert list(ids) == FIXTURE_IDS
    # Exact: the literal text values are binary-representable.
    assert values[0].tolist() == [0.5, 1.25, 1.75, 0.25]
    assert values[3].tolist() == [0.25, 0.75, 0.25, 0.5]


def test_predictions_average_across_studies(suite):
    """`study=None` means the per-customer, per-period mean over every `Prediction_*.csv`."""
    one, _ = suite_reader.load_model_predictions(suite / "LSTM", study=1)
    mean, ids = suite_reader.load_model_predictions(suite / "LSTM", study=None)
    assert list(ids) == FIXTURE_IDS
    # study 2 == study 1 + 0.25 everywhere, so the mean is exactly + 0.125.
    assert np.array_equal(mean, one + 0.125)


def test_deterministic_benchmark_ignores_the_requested_study_index(suite):
    """A single-fit benchmark has only `Prediction_1.csv`; any index resolves to it."""
    first, _ = suite_reader.load_model_predictions(suite / "ParetoNBD_MLE", study=1)
    seventh, _ = suite_reader.load_model_predictions(suite / "ParetoNBD_MLE", study=7)
    assert np.array_equal(first, seventh)


def test_aggregate_writes_one_flat_csv_per_model(suite):
    """`aggregated_<Model>.csv` at the suite root, same wide format, `Id`-headed here."""
    written = suite_reader.aggregate_suite_predictions(suite)
    assert {p.name for p in written} == {
        "aggregated_LSTM.csv", "aggregated_ParetoNBD_MLE.csv",
    }
    text = (suite / "aggregated_LSTM.csv").read_text().splitlines()
    assert text[0] == prediction_header(FIXTURE_T_HOLD)
    assert len(text) == 1 + len(FIXTURE_IDS)
    # It round-trips through the same reader as the per-study files.
    from panelclv.predictions import load_predictions_from_csv
    values, ids = load_predictions_from_csv(suite / "aggregated_LSTM.csv")
    one, _ = suite_reader.load_model_predictions(suite / "LSTM", study=1)
    assert list(ids) == FIXTURE_IDS
    assert np.allclose(values, one + 0.125)


def test_writer_emits_the_format_the_reader_reads(tmp_path):
    """The writer half: `save_predictions_to_csv` produces the literal fixture header.

    Every test above reads literal text, so none of them can catch the writer drifting
    away from it. This one drives the real writer and compares.
    """
    from panelclv.predictions import save_predictions_to_csv

    model_dir = tmp_path / "LSTM"
    values = np.arange(len(FIXTURE_IDS) * FIXTURE_T_HOLD, dtype=float).reshape(
        len(FIXTURE_IDS), FIXTURE_T_HOLD
    )
    path = save_predictions_to_csv(
        values,
        layout.prediction_path(model_dir, 3),
        customer_ids=FIXTURE_IDS,
        id_col=FIXTURE_ID_COL,
    )
    assert path == model_dir / "Predictions" / "Prediction_3.csv"
    lines = path.read_text().splitlines()
    assert lines[0] == prediction_header(FIXTURE_T_HOLD)
    assert lines[1].startswith("C01,")


# --------------------------------------------------------------------------------------
# 3. The full read path: config.json -> PanelConfig -> prepare_dataset -> scoring
# --------------------------------------------------------------------------------------


def test_fixture_study_metrics_reproduce_results_csv(suite, panel_csv):
    """The whole reader path, and the runner/reader agreement, on the fixture.

    `study_metrics` rebuilds the cohort from the stored `panel_config` + the panel CSV,
    checks the rebuilt ids against every prediction file's, and rescores each study with
    `compute_forecast_metrics`. Requiring the result to equal the stored `results.csv`
    means a suite's two producers of the same numbers — the runner that wrote them and
    the reader that recomputes them — still agree.
    """
    table = suite_metrics.study_metrics(suite, panel_csv)
    assert list(table.index) == ["LSTM", "ParetoNBD_MLE"]
    assert list(table.columns) == RESULTS_METRIC_COLS + ["n_studies"]
    assert list(table["n_studies"]) == [2, 1]

    if os.environ.get("PANELCLV_PRINT_FIXTURE_METRICS"):
        _print_fixture_metrics(suite, panel_csv)

    stored = pd.read_csv(suite / "results.csv")
    for name in ("LSTM", "ParetoNBD_MLE"):
        rows = stored[stored["model"] == name]
        for metric in RESULTS_METRIC_COLS:
            assert table.loc[name, metric] == pytest.approx(
                rows[metric].mean(), rel=1e-12
            ), f"{name}/{metric} drifted from results.csv"


def test_fixture_panel_rebuilds_the_recorded_cohort(suite, panel_csv):
    """`data_summary` describes the dataset the stored recipe rebuilds."""
    data = suite_reader._actuals_from_panel(suite, panel_csv)
    summary = FIXTURE_SUITE_CONFIG["data_summary"]
    assert list(data["ids"]) == FIXTURE_IDS
    assert int(data["T_CAL"]) == summary["T_CAL"]
    assert int(data["T_HOLD"]) == summary["T_HOLD"]
    assert int(data["F"]) == summary["F"]
    assert list(data["seq_cols"]) == summary["seq_cols"]


def _print_fixture_metrics(suite: Path, panel_csv: Path) -> None:
    """Print a paste-ready `FIXTURE_METRICS` block (see the module docstring)."""
    from panelclv.models import compute_forecast_metrics

    data = suite_reader._actuals_from_panel(suite, panel_csv)
    actual = np.asarray(data["holdout"], dtype=np.float64)[:, :, int(data["target_idx"])]
    print("\nFIXTURE_METRICS = {")
    for name, model_dir in suite_reader._discover_models(suite):
        for path in sorted(
            (model_dir / "Predictions").glob("Prediction_*.csv"),
            key=suite_reader._prediction_index,
        ):
            index = suite_reader._prediction_index(path)
            values, _ = suite_reader.load_model_predictions(model_dir, study=index)
            metrics = compute_forecast_metrics(actual, values)
            print(f'    ("{name}", {index}): {{')
            for metric in RESULTS_METRIC_COLS:
                print(f'        "{metric}": {metrics[metric]!r},')
            print("    },")
    print("}")
