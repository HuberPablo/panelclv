"""The on-disk study-format floor, made executable.

`CLAUDE.md`'s priority #2 and the package-simplification map's third floor item say the
same thing: **archived suites under `Studies/` stay readable**. A suite costs GPU-hours to
produce, so a rename or a column reshuffle that orphans one destroys work that cannot be
cheaply recreated. Until this file existed, no test read an archived suite and
`studies/runner.py` — the writer — had no test at all, so the format real results are
stored in was pinned by no assertion anywhere.

What is pinned here is the *format*, in both directions:

- **The reader recovers a suite written as literal bytes.** The fixture below is written
  with `Path.write_text` from text transcribed out of a real archived suite — never through
  `save_predictions_to_csv` / `layout.write_json`. If writer and reader are changed together
  the round-trip would still pass; against literal text it cannot. This half always runs.
- **The writer still emits that same format.** One test drives the real writers
  (`layout.prediction_path`, `save_predictions_to_csv`) and compares their output to the
  literal header the archive uses.
- **The real archive still matches the literals** (skipped when `Studies/` is absent) —
  including the nine stored `results.csv` numbers of one suite, recomputed from its stored
  predictions.

**Why both a fixture and a real-archive test.** `Studies/` and `Datasets/` are both
gitignored and the suites run 7.4 MB to 554 MB, so no real suite can be committed and a
test that reads one unconditionally fails on every fresh clone. A skip-if-absent test alone
would therefore be vacuous almost everywhere. A fixture alone would pin only this file's
*reconstruction* of the format. So: the fixture always runs and carries the assertions, and
the real-archive tests verify the fixture is a faithful excerpt and add the numeric
regression. (Note also that `Predictions/` is itself gitignored — hence the fixture is
built into `tmp_path` from constants rather than committed as a file tree, which would be
silently untracked.)

**Current behaviour is pinned, warts included.** Two are deliberate and must not be
"fixed" here:

- `analysis._id_col` falls back to `"customer_id"` for a suite whose `config.json` carries
  no `panel_config`, so `aggregate_suite_predictions` writes `aggregated_*.csv` headed
  `customer_id` next to `Prediction_*.csv` headed `Id`. This has already happened on disk
  (`Studies/cross_entropy_cfg_2y_Train_1yPred_NoCov_V1_10Studies_100_simulations/`).
  `test_legacy_suite_aggregate_uses_the_customer_id_fallback` pins it as-is.
- `study_metrics` is unusable on a suite without `panel_config` (4 of the 5 archived
  electronics suites), raising rather than falling back.
  `test_legacy_suite_has_no_panel_config_to_rebuild_actuals` pins the raise.

**Torch.** The read path is not torch-free, despite `analysis.py` being written to stay so:
`load_predictions_from_csv` lives in `evaluation/plot_utils.py`, which imports torch at
module level. The structural half of this file (tree shape, headers, `PanelConfig`
round-trip) therefore runs without torch; everything that actually loads a prediction is
marked `needs_torch`.

Run:  PYTHONPATH=src pytest -q tests/test_archive_formats.py

To regenerate the pinned fixture metrics after a *deliberate* change, run with
``PANELCLV_PRINT_ARCHIVE_GATE=1`` and paste the printed block over `FIXTURE_METRICS`.
"""

import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panelclv.configs.panel_config import PanelConfig
from panelclv.studies import analysis, layout

# --------------------------------------------------------------------------------------
# Where the real archive lives, and the two skip gates
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = REPO_ROOT / "Studies"

# The suite this file pins numerically: the smallest neural suite (7.4 MB), and the only
# archived electronics suite that carries a `panel_config` — so the only one on which
# `study_metrics(root, panel_path)` can rebuild the actuals at all.
PINNED_SUITE = ARCHIVE_ROOT / "cross_entropy_cfg_2yTrain_1yPred_NoCov_TestDimanche"
PINNED_PANEL = REPO_ROOT / "Datasets" / "Dataset_clean" / "electronics_customer_week_panel.csv"

# A Pareto/NBD generation study's *trained* tree: one standard suite per generated dataset,
# nested one level deeper. Its shape is pinned separately (see the pnbd tests). The grid's
# reader, `pnbd_grid.collect_grid_results`, joins it against the *generation* study that
# produced the panels — two directories that have to stay in step, so both are named here.
PNBD_GRID = ARCHIVE_ROOT / "pnbd_study_4x4x10_20260802-160937__ar"
PNBD_GENERATION = REPO_ROOT / "Datasets" / "Synthetic" / "pnbd_study_4x4x10_20260802-160937"

needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="the prediction reader imports evaluation.plot_utils, which imports torch",
)
needs_archive = pytest.mark.skipif(
    not PINNED_SUITE.is_dir(),
    reason=f"no archived suite at {PINNED_SUITE} (Studies/ is gitignored)",
)
needs_panel = pytest.mark.skipif(
    not PINNED_PANEL.is_file(),
    reason=f"no panel at {PINNED_PANEL} (Datasets/ is gitignored)",
)

# --------------------------------------------------------------------------------------
# The format, as constants. Transcribed from the real archive; the real-archive tests
# below check the transcription is still accurate.
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
    "mape_aggregate_style",
]
# The three metric names are an on-disk contract: `analysis._STUDY_METRIC_COLS` and
# `pnbd_grid._METRIC_SOURCE` both name them, and renaming one breaks every archived read.
RESULTS_METRIC_COLS = ["rmse", "bias_percent", "mape_aggregate_style"]

# A prediction file: `Prediction_{i}.csv`, wide, `<id_col>` then `week_0..week_{T-1}`.
PREDICTION_FILE_RE = re.compile(r"^Prediction_(\d+)\.csv$")
ARCHIVE_ID_COL = "Id"          # what every archived suite's Prediction_*.csv is headed


def prediction_header(n_weeks: int, id_col: str = ARCHIVE_ID_COL) -> str:
    """The exact first line of a `Prediction_*.csv` with `n_weeks` holdout periods."""
    return ",".join([id_col] + [f"week_{i}" for i in range(n_weeks)])


# Keys every archived suite `config.json` carries, in both generations of the format.
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
# Keys only the current generation carries. The four legacy electronics suites predate
# them, which is exactly why `analysis` must not require them.
SUITE_CONFIG_KEYS_CURRENT = {"overwrite", "keep_only_best_checkpoint", "panel_config"}

DATA_SUMMARY_KEYS = {
    "n_customers", "T_CAL", "T_HOLD", "F", "seq_cols", "target_col", "validation_start",
}
MODEL_CONFIG_KEYS = {
    "name", "model_type", "prediction_source", "n_simulations", "base_seed", "device",
}

# --------------------------------------------------------------------------------------
# The fixture: one suite, written as literal bytes.
#
# Four customers, seven calibration weeks, four holdout weeks, two models — an `lstm` with
# two studies and a `pareto_nbd` benchmark with one, which is the shape every archived
# electronics suite has. The panel below reproduces, in miniature, the panel the archive's
# `panel_config` describes: an `Id` column, `year`/`week` time columns, integer
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
# and the value types match the archived suite's `panel_config` one for one.
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

# Per-model `config.json`, exactly the two shapes `runner._model_record` produces.
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
# recomputes them from the prediction files and requires agreement, which is the property
# audit 04 verified on the real archive, reproduced here so it runs everywhere.
FIXTURE_METRICS = {
    ("LSTM", 1): {
        "rmse": 0.385275875185561,
        "bias_percent": 3.8461538461538463,
        "mape_aggregate_style": 19.23076923076923,
    },
    ("LSTM", 2): {
        "rmse": 0.4759858191164943,
        "bias_percent": 34.61538461538461,
        "mape_aggregate_style": 34.61538461538461,
    },
    ("ParetoNBD_MLE", 1): {
        "rmse": 0.7180703308172536,
        "bias_percent": 0.0,
        "mape_aggregate_style": 23.076923076923077,
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
    """A pre-`panel_config` suite, the shape 4 of the 5 archived electronics suites have."""
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
# 1. The tree — no torch needed
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
    found = analysis._discover_models(suite)
    assert [name for name, _ in found] == ["LSTM", "ParetoNBD_MLE"]
    assert [d for _, d in found] == [suite / "LSTM", suite / "ParetoNBD_MLE"]


def test_deterministic_model_is_read_from_its_model_type(suite):
    """`pareto_nbd` is a single-fit benchmark; the neural family is not."""
    assert analysis._is_deterministic_model(suite / "ParetoNBD_MLE")
    assert not analysis._is_deterministic_model(suite / "LSTM")


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
    assert analysis._prediction_index(Path("Prediction_10.csv")) == 10


def test_results_csv_column_contract(suite):
    """`results.csv` leads with the fixed columns; the rest are `param_*` hyperparameters."""
    df = pd.read_csv(suite / "results.csv")
    assert list(df.columns) == RESULTS_LEADING_COLS + FIXTURE_RESULTS_PARAM_UNION
    assert all(c.startswith("param_") for c in df.columns[len(RESULTS_LEADING_COLS):])
    # The three metric names `analysis._STUDY_METRIC_COLS` and `pnbd_grid._METRIC_SOURCE`
    # read out of an archived file. Renaming one orphans every archive.
    assert analysis._STUDY_METRIC_COLS == RESULTS_METRIC_COLS
    from panelclv.studies.pnbd_grid import _METRIC_SOURCE
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
    """The per-model record carries the keys `analysis` and a reader need."""
    for name in ("LSTM", "ParetoNBD_MLE"):
        record = json.loads((suite / name / "config.json").read_text())
        assert MODEL_CONFIG_KEYS <= set(record)
        assert record["name"] == name


def test_suite_config_json_round_trips_to_a_panel_config(suite):
    """`config.json` -> `PanelConfig` -> dict is the identity, which `analysis` depends on.

    `_actuals_from_panel` rebuilds the dataset from this stored recipe, so an archived
    suite is only readable while `PanelConfig.from_dict` accepts what was written. Asserted
    as an exact dict round-trip: a renamed or dropped field breaks it immediately.
    """
    stored = json.loads((suite / "config.json").read_text())
    assert SUITE_CONFIG_KEYS <= set(stored)
    assert SUITE_CONFIG_KEYS_CURRENT <= set(stored)
    assert DATA_SUMMARY_KEYS <= set(stored["data_summary"])

    panel_config = PanelConfig.from_dict(stored["panel_config"])
    assert panel_config.to_dict() == stored["panel_config"]
    assert panel_config.id_col == ARCHIVE_ID_COL
    assert panel_config.target_col == "Transactions"
    # `clip_target_upper` sets the softmax head size, so it is part of the recipe.
    assert panel_config.clip_target_upper == 3


def test_legacy_suite_config_is_still_discoverable(legacy_suite):
    """A pre-`panel_config` suite keeps its models discoverable and its id column absent."""
    stored = json.loads((legacy_suite / "config.json").read_text())
    assert SUITE_CONFIG_KEYS <= set(stored)
    assert not (SUITE_CONFIG_KEYS_CURRENT & set(stored))
    assert [name for name, _ in analysis._discover_models(legacy_suite)] == [
        "LSTM", "ParetoNBD_MLE",
    ]


def test_id_col_resolution_including_the_legacy_fallback(suite, legacy_suite):
    """`_id_col` reads `panel_config`; without one it falls back — a pinned wart.

    The fallback is `"customer_id"`, while the prediction files it sits beside are headed
    `"Id"`. Recorded, not corrected: tickets 06/11 rule on it.
    """
    assert analysis._id_col(suite) == ARCHIVE_ID_COL
    assert analysis._id_col(legacy_suite) == "customer_id"


# --------------------------------------------------------------------------------------
# 2. Reading predictions back — needs torch (via evaluation.plot_utils)
# --------------------------------------------------------------------------------------


@needs_torch
def test_predictions_load_back_with_ids_and_values(suite):
    """One study's file becomes `(values (N, T_HOLD), ids)` with the ids in cohort order."""
    values, ids = analysis.load_model_predictions(suite / "LSTM", study=1)
    assert values.shape == (len(FIXTURE_IDS), FIXTURE_T_HOLD)
    assert list(ids) == FIXTURE_IDS
    # Exact: the literal text values are binary-representable.
    assert values[0].tolist() == [0.5, 1.25, 1.75, 0.25]
    assert values[3].tolist() == [0.25, 0.75, 0.25, 0.5]


@needs_torch
def test_predictions_average_across_studies(suite):
    """`study=None` means the per-customer, per-period mean over every `Prediction_*.csv`."""
    one, _ = analysis.load_model_predictions(suite / "LSTM", study=1)
    mean, ids = analysis.load_model_predictions(suite / "LSTM", study=None)
    assert list(ids) == FIXTURE_IDS
    # study 2 == study 1 + 0.25 everywhere, so the mean is exactly + 0.125.
    assert np.array_equal(mean, one + 0.125)


@needs_torch
def test_deterministic_benchmark_ignores_the_requested_study_index(suite):
    """A single-fit benchmark has only `Prediction_1.csv`; any index resolves to it."""
    first, _ = analysis.load_model_predictions(suite / "ParetoNBD_MLE", study=1)
    seventh, _ = analysis.load_model_predictions(suite / "ParetoNBD_MLE", study=7)
    assert np.array_equal(first, seventh)


@needs_torch
def test_aggregate_writes_one_flat_csv_per_model(suite):
    """`aggregated_<Model>.csv` at the suite root, same wide format, `Id`-headed here."""
    written = analysis.aggregate_suite_predictions(suite)
    assert {p.name for p in written} == {
        "aggregated_LSTM.csv", "aggregated_ParetoNBD_MLE.csv",
    }
    text = (suite / "aggregated_LSTM.csv").read_text().splitlines()
    assert text[0] == prediction_header(FIXTURE_T_HOLD)
    assert len(text) == 1 + len(FIXTURE_IDS)
    # It round-trips through the same reader as the per-study files.
    from panelclv.evaluation.plot_utils import load_predictions_from_csv
    values, ids = load_predictions_from_csv(suite / "aggregated_LSTM.csv")
    one, _ = analysis.load_model_predictions(suite / "LSTM", study=1)
    assert list(ids) == FIXTURE_IDS
    assert np.allclose(values, one + 0.125)


@needs_torch
def test_legacy_suite_aggregate_uses_the_customer_id_fallback(legacy_suite):
    """PINNED WART: the aggregate is headed `customer_id` beside `Id` prediction files.

    Audit 04 found this already on disk in
    `Studies/cross_entropy_cfg_2y_Train_1yPred_NoCov_V1_10Studies_100_simulations/`. It is
    harmless only because `load_predictions_from_csv` sniffs both names. Pinned as current
    behaviour so a fix is a deliberate, visible change.
    """
    written = analysis.aggregate_suite_predictions(legacy_suite)
    aggregated = next(p for p in written if p.name == "aggregated_LSTM.csv")
    assert aggregated.read_text().splitlines()[0] == prediction_header(
        FIXTURE_T_HOLD, id_col="customer_id"
    )
    source = (legacy_suite / "LSTM" / "Predictions" / "Prediction_1.csv").read_text()
    assert source.splitlines()[0].startswith(ARCHIVE_ID_COL + ",")


@needs_torch
def test_writer_still_emits_the_archived_prediction_format(tmp_path):
    """The writer half: `save_predictions_to_csv` produces the literal archived header.

    The tests above read literal text, so they cannot catch the writer drifting away from
    it. This one drives the real writer and compares.
    """
    from panelclv.evaluation.plot_utils import save_predictions_to_csv

    model_dir = tmp_path / "LSTM"
    values = np.arange(len(FIXTURE_IDS) * FIXTURE_T_HOLD, dtype=float).reshape(
        len(FIXTURE_IDS), FIXTURE_T_HOLD
    )
    path = save_predictions_to_csv(
        values,
        layout.prediction_path(model_dir, 3),
        customer_ids=FIXTURE_IDS,
        id_col=ARCHIVE_ID_COL,
    )
    assert path == model_dir / "Predictions" / "Prediction_3.csv"
    lines = path.read_text().splitlines()
    assert lines[0] == prediction_header(FIXTURE_T_HOLD)
    assert lines[1].startswith("C01,")


# --------------------------------------------------------------------------------------
# 3. The full read path: config.json -> PanelConfig -> prepare_dataset -> scoring
# --------------------------------------------------------------------------------------


@needs_torch
def test_fixture_study_metrics_reproduce_results_csv(suite, panel_csv):
    """The whole reader path, and the runner/reader agreement, on the fixture.

    `study_metrics` rebuilds the cohort from the stored `panel_config` + the panel CSV,
    checks the rebuilt ids against every prediction file's, and rescores each study with
    `compute_forecast_metrics`. Requiring the result to equal the stored `results.csv`
    means the archive's two producers still agree — the property audit 04 verified on the
    real suite, asserted here so it runs without the archive.
    """
    table = analysis.study_metrics(suite, panel_csv)
    assert list(table.index) == ["LSTM", "ParetoNBD_MLE"]
    assert list(table.columns) == RESULTS_METRIC_COLS + ["n_studies"]
    assert list(table["n_studies"]) == [2, 1]

    if os.environ.get("PANELCLV_PRINT_ARCHIVE_GATE"):
        _print_fixture_metrics(suite, panel_csv)

    stored = pd.read_csv(suite / "results.csv")
    for name in ("LSTM", "ParetoNBD_MLE"):
        rows = stored[stored["model"] == name]
        for metric in RESULTS_METRIC_COLS:
            assert table.loc[name, metric] == pytest.approx(
                rows[metric].mean(), rel=1e-12
            ), f"{name}/{metric} drifted from results.csv"


def test_fixture_panel_rebuilds_the_recorded_cohort(suite, panel_csv):
    """`data_summary` describes the dataset the stored recipe rebuilds. numpy/pandas only."""
    data = analysis._actuals_from_panel(suite, panel_csv)
    summary = FIXTURE_SUITE_CONFIG["data_summary"]
    assert list(data["ids"]) == FIXTURE_IDS
    assert int(data["T_CAL"]) == summary["T_CAL"]
    assert int(data["T_HOLD"]) == summary["T_HOLD"]
    assert int(data["F"]) == summary["F"]
    assert list(data["seq_cols"]) == summary["seq_cols"]


def test_legacy_suite_has_no_panel_config_to_rebuild_actuals(legacy_suite, panel_csv):
    """PINNED WART: `study_metrics` is unusable on a suite without a `panel_config`.

    Four of the five archived electronics suites are in that state, so this is the archive
    floor's sharpest edge: the read path raises rather than falling back to `data=`.
    Recorded, not corrected.
    """
    with pytest.raises(ValueError, match="no panel_config"):
        analysis._actuals_from_panel(legacy_suite, panel_csv)


# --------------------------------------------------------------------------------------
# 4. The real archive — skipped when Studies/ (and Datasets/) are absent
# --------------------------------------------------------------------------------------

# The nine numbers `Studies/cross_entropy_cfg_2yTrain_1yPred_NoCov_TestDimanche/results.csv`
# holds — three models x three metrics — copied from that file. The gate recomputes them
# from the stored prediction CSVs and requires agreement.
PINNED_SUITE_RESULTS = {
    "LSTM": {
        "rmse": 0.3766844079321802,
        "bias_percent": -53.40695337290179,
        "mape_aggregate_style": 58.60463556558676,
    },
    "Transformer": {
        "rmse": 0.3782859045756142,
        "bias_percent": 73.72835716429461,
        "mape_aggregate_style": 83.17552828690229,
    },
    "ParetoNBD_MLE": {
        "rmse": 0.3754635997751549,
        "bias_percent": -58.6238973833398,
        "mape_aggregate_style": 62.835018151643645,
    },
}


@needs_archive
def test_archived_suite_matches_the_pinned_format():
    """The literals above are a faithful excerpt of the real suite's format."""
    stored = json.loads((PINNED_SUITE / "config.json").read_text())
    assert SUITE_CONFIG_KEYS <= set(stored)
    assert SUITE_CONFIG_KEYS_CURRENT <= set(stored)
    assert DATA_SUMMARY_KEYS <= set(stored["data_summary"])

    # The round-trip the reader depends on, on the real recipe.
    panel_config = PanelConfig.from_dict(stored["panel_config"])
    assert panel_config.to_dict() == stored["panel_config"]
    assert panel_config.id_col == ARCHIVE_ID_COL

    results = pd.read_csv(PINNED_SUITE / "results.csv")
    assert list(results.columns[: len(RESULTS_LEADING_COLS)]) == RESULTS_LEADING_COLS
    assert all(c.startswith("param_") for c in results.columns[len(RESULTS_LEADING_COLS):])

    names = [m["name"] for m in stored["models"]]
    assert [n for n, _ in analysis._discover_models(PINNED_SUITE)] == names

    expected = prediction_header(int(stored["data_summary"]["T_HOLD"]))
    n_customers = int(stored["data_summary"]["n_customers"])
    for name in names:
        model_dir = PINNED_SUITE / name
        assert (model_dir / "config.json").is_file()
        assert (model_dir / "metrics.csv").is_file()
        paths = sorted((model_dir / "Predictions").glob("*.csv"))
        assert paths, name
        for path in paths:
            assert PREDICTION_FILE_RE.match(path.name), path.name
            # Header + row count only: the files are wide and there are many of them.
            with open(path) as fh:
                assert fh.readline().rstrip("\n") == expected, path
                assert sum(1 for _ in fh) == n_customers, path


@needs_archive
def test_every_archived_neural_suite_still_parses():
    """Every finished suite under `Studies/` (both config generations) reads as expected.

    Headers only, so this stays fast on the 554 MB and 2.6 GB suites. Suites with no
    `results.csv` are unfinished runs and are skipped.
    """
    suites = sorted(
        p for p in ARCHIVE_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("cross_entropy") and (p / "results.csv").is_file()
    )
    assert suites, "no finished neural suite found"
    for root in suites:
        _assert_suite_shape(root)


@pytest.mark.skipif(
    not PNBD_GRID.is_dir(), reason=f"no Pareto/NBD grid archive at {PNBD_GRID}"
)
def test_pnbd_grid_archive_is_a_directory_of_standard_suites():
    """The grid tree is one level deeper: `<grid>/<combo>__<dataset>/` is a normal suite.

    `pnbd_grid.collect_grid_results` reads each sub-suite's `results.csv` by that path and
    pulls `_METRIC_SOURCE`'s columns out of it. Only the first three sub-suites are checked
    — the grid holds 160 of them and they are written by one code path.
    """
    from panelclv.studies.pnbd_grid import _METRIC_SOURCE

    sub_suites = sorted(p for p in PNBD_GRID.iterdir() if p.is_dir())
    assert len(sub_suites) > 1
    assert all("__" in p.name for p in sub_suites)
    checked = 0
    for root in sub_suites:
        if not (root / "results.csv").is_file():
            continue
        _assert_suite_shape(root)
        results = pd.read_csv(root / "results.csv")
        assert set(_METRIC_SOURCE.values()) <= set(results.columns)
        checked += 1
        if checked == 3:
            break
    assert checked == 3, "fewer than three trained sub-suites in the grid archive"


@pytest.mark.skipif(
    not (PNBD_GRID.is_dir() and PNBD_GENERATION.is_dir()),
    reason="the Pareto/NBD grid archive or its generation study is absent",
)
def test_pnbd_grid_results_still_join_to_their_generation_study():
    """The grid's own reader still joins archive to generation study, losing no dataset.

    `collect_grid_results` is the entry point for the Pareto/NBD half of the archive, and it
    spans two on-disk formats: the trained suites under `Studies/` and the *generation*
    tree under `Datasets/Synthetic/`, whose `<combo>/<dataset>/config.json` supplies the
    grid coordinates. It silently `continue`s past any sub-suite whose `results.csv` is
    missing, so a path change would thin the frame rather than raise — hence the row count
    is checked against what is actually on disk, not merely asserted non-empty.
    """
    from panelclv.data_preparation.pareto_simulation import list_pnbd_datasets
    from panelclv.studies.pnbd_grid import DEFAULT_METRICS, collect_grid_results

    trained = [
        p for p in PNBD_GRID.iterdir() if p.is_dir() and (p / "results.csv").is_file()
    ]
    rows_on_disk = sum(len(pd.read_csv(p / "results.csv")) for p in trained)
    assert rows_on_disk > 100, "grid archive is too small to be the pinned one"

    grid = collect_grid_results(PNBD_GENERATION, PNBD_GRID)
    assert list(grid.columns) == [
        "mean_transaction_rate", "churn_rate", "combo", "dataset", "model",
        *DEFAULT_METRICS,
    ]
    # Every (model, study) row of every trained sub-suite made it through the join.
    assert len(grid) == rows_on_disk
    # And every generated dataset is represented, so no coordinate was dropped.
    manifest = list_pnbd_datasets(PNBD_GENERATION)
    assert set(grid["combo"] + "__" + grid["dataset"]) == {p.name for p in trained}
    assert set(grid["combo"]) <= set(manifest["combo"])
    assert grid[["mean_transaction_rate", "churn_rate"]].drop_duplicates().shape[0] == (
        manifest[["mean_transaction_rate", "churn_rate"]].drop_duplicates().shape[0]
    )


def _assert_suite_shape(root: Path) -> None:
    """Assert one archived suite matches the pinned format, reading headers only."""
    stored = json.loads((root / "config.json").read_text())
    assert SUITE_CONFIG_KEYS <= set(stored), root
    assert DATA_SUMMARY_KEYS <= set(stored["data_summary"]), root
    if stored.get("panel_config") is not None:
        panel_config = PanelConfig.from_dict(stored["panel_config"])
        assert panel_config.to_dict() == stored["panel_config"], root

    results = pd.read_csv(root / "results.csv")
    assert list(results.columns[: len(RESULTS_LEADING_COLS)]) == RESULTS_LEADING_COLS, root
    assert all(
        c.startswith("param_") for c in results.columns[len(RESULTS_LEADING_COLS):]
    ), root

    t_hold = int(stored["data_summary"]["T_HOLD"])
    n_customers = int(stored["data_summary"]["n_customers"])
    for name in [m["name"] for m in stored["models"]]:
        preds = root / name / "Predictions"
        if not preds.is_dir():
            continue
        for path in sorted(preds.glob("*.csv")):
            assert PREDICTION_FILE_RE.match(path.name), path
            with open(path) as fh:
                assert fh.readline().rstrip("\n") == prediction_header(t_hold), path
                assert sum(1 for _ in fh) == n_customers, path


@needs_archive
@needs_panel
@needs_torch
def test_archived_suite_metrics_reproduce_its_stored_results_csv():
    """The numeric regression: the nine stored `results.csv` values, recomputed.

    `study_metrics` rebuilds the cohort from the suite's `panel_config` + the panel CSV and
    rescores every stored prediction file. Agreement means the archive's metrics are still
    recoverable from its predictions — the strongest single statement of the on-disk floor.

    Tolerance: `rel=1e-12`. The two paths agree to ~13 significant digits, not bitwise —
    audit 04's "full float precision" was a shade optimistic. The residual comes from
    float reduction order, not from the format.
    """
    stored = pd.read_csv(PINNED_SUITE / "results.csv")
    table = analysis.study_metrics(PINNED_SUITE, PINNED_PANEL)

    assert set(table.index) == set(PINNED_SUITE_RESULTS)
    for model, metrics in PINNED_SUITE_RESULTS.items():
        row = stored[stored["model"] == model]
        assert len(row) == 1, model
        for metric, value in metrics.items():
            # (a) the file still holds the pinned number ...
            assert float(row.iloc[0][metric]) == value, f"{model}/{metric} in results.csv"
            # ... and (b) the reader still recomputes it from the stored predictions.
            assert table.loc[model, metric] == pytest.approx(value, rel=1e-12), (
                f"{model}/{metric} recomputed from Prediction_*.csv"
            )


def _print_fixture_metrics(suite: Path, panel_csv: Path) -> None:
    """Print a paste-ready `FIXTURE_METRICS` block (see the module docstring)."""
    from panelclv.models import mc_compute_metrics

    data = analysis._actuals_from_panel(suite, panel_csv)
    actual = np.asarray(data["holdout"], dtype=np.float64)[:, :, int(data["target_idx"])]
    print("\nFIXTURE_METRICS = {")
    for name, model_dir in analysis._discover_models(suite):
        for path in sorted(
            (model_dir / "Predictions").glob("Prediction_*.csv"),
            key=analysis._prediction_index,
        ):
            index = analysis._prediction_index(path)
            values, _ = analysis.load_model_predictions(model_dir, study=index)
            metrics = mc_compute_metrics(actual, values)
            print(f'    ("{name}", {index}): {{')
            for metric in RESULTS_METRIC_COLS:
                print(f'        "{metric}": {metrics[metric]!r},')
            print("    },")
    print("}")
