"""Tests for the study-suite read/aggregate/plot layer (`studies.analysis`).

These build a tiny fake suite tree on disk (a couple of hand-made
``Prediction_*.csv`` per model) and exercise the averaging, the aggregated-CSV
output, the alignment guards, and the plot entry point. They use the ``data=``
escape hatch for plotting so no real panel / ``prepare_dataset`` run is needed.

Run:  pytest -q tests/test_studies_analysis.py
"""

import json

import numpy as np
import pytest

from panelclv.studies import (
    aggregate_suite_predictions,
    load_model_predictions,
    plot_suite_forecast,
)
from panelclv.evaluation.plot_utils import (
    load_predictions_from_csv,
    save_predictions_to_csv,
)

IDS = np.array([10, 20, 30])          # 3 customers
N, T = 3, 4                            # 3 customers, 4 holdout weeks


def _write_prediction(model_dir, index, values, ids=IDS):
    """Write one Prediction_{index}.csv in the wide format the suite uses."""
    path = model_dir / "Predictions" / f"Prediction_{index}.csv"
    save_predictions_to_csv(values, path, customer_ids=ids, id_col="customer_id")
    return path


@pytest.fixture
def suite(tmp_path):
    """A fake two-model suite. ModelA has 2 studies, ModelB has 1."""
    root = tmp_path / "suite"
    (root / "ModelA" / "Predictions").mkdir(parents=True)
    (root / "ModelB" / "Predictions").mkdir(parents=True)

    # ModelA: two studies whose mean is easy to reason about.
    a1 = np.arange(N * T, dtype=float).reshape(N, T)
    a2 = a1 + 2.0                       # so the per-cell mean is a1 + 1.0
    _write_prediction(root / "ModelA", 1, a1)
    _write_prediction(root / "ModelA", 2, a2)

    # ModelB: single study.
    b1 = np.ones((N, T)) * 5.0
    _write_prediction(root / "ModelB", 1, b1)

    # Minimal suite config.json: model order + id column.
    config = {
        "models": [{"name": "ModelA"}, {"name": "ModelB"}],
        "data_summary": {"id_col": "customer_id"},
    }
    with open(root / "config.json", "w") as f:
        json.dump(config, f)

    return root, a1, a2, b1


def _fake_data():
    """A stand-in prepare_dataset dict for the plot path (data= escape hatch).

    Only the keys plot_suite_forecast reads: calibration/holdout (N, T, F) tensors,
    target_idx and ids. F=1, target channel 0.
    """
    calibration = np.ones((N, 6, 1))           # 6 training weeks
    holdout = np.ones((N, T, 1)) * 2.0          # T holdout weeks
    return {
        "calibration": calibration,
        "holdout": holdout,
        "target_idx": 0,
        "ids": IDS,
    }


# --- loading / averaging -----------------------------------------------------


def test_single_study_load(suite):
    root, a1, _, _ = suite
    values, ids = load_model_predictions(root / "ModelA", study=1)
    assert np.array_equal(values, a1)
    assert np.array_equal(ids, IDS)


def test_average_across_studies_is_elementwise_mean(suite):
    root, a1, a2, _ = suite
    values, ids = load_model_predictions(root / "ModelA", study=None)
    assert np.allclose(values, (a1 + a2) / 2.0)          # == a1 + 1.0
    assert np.array_equal(ids, IDS)


def test_single_study_model_averages_to_itself(suite):
    root, _, _, b1 = suite
    values, _ = load_model_predictions(root / "ModelB", study=None)
    assert np.allclose(values, b1)


def test_missing_study_index_raises(suite):
    root, *_ = suite
    with pytest.raises(FileNotFoundError):
        load_model_predictions(root / "ModelA", study=99)


def test_id_mismatch_across_studies_raises(suite):
    root, *_ = suite
    # Overwrite ModelA study 2 with a different id vector.
    _write_prediction(root / "ModelA", 2, np.zeros((N, T)), ids=np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="customer ids differ"):
        load_model_predictions(root / "ModelA", study=None)


# --- aggregated CSV output ---------------------------------------------------


def test_aggregate_writes_flat_csvs_at_root(suite):
    root, a1, a2, b1 = suite
    written = aggregate_suite_predictions(root)

    names = {p.name for p in written}
    assert names == {"aggregated_ModelA.csv", "aggregated_ModelB.csv"}
    assert all(p.parent == root for p in written)          # flat at the suite root

    # Round-trips through the standard loader and holds the mean.
    vals, ids = load_predictions_from_csv(root / "aggregated_ModelA.csv")
    assert np.allclose(vals, (a1 + a2) / 2.0)
    assert np.array_equal(ids, IDS)


# --- plotting ----------------------------------------------------------------


def test_plot_returns_fig_with_one_line_per_model(suite):
    root, *_ = suite
    fig, ax = plot_suite_forecast(root, data=_fake_data(), study=1)

    labels = [ln.get_label() for ln in ax.get_lines()]
    # training actual + holdout actual + one line per model.
    assert "ModelA" in labels and "ModelB" in labels
    assert any("training" in l.lower() for l in labels)
    assert any("holdout" in l.lower() for l in labels)


def test_plot_average_mode_writes_aggregates_and_titles(suite):
    root, *_ = suite
    fig, ax = plot_suite_forecast(root, data=_fake_data(), study=None)
    assert (root / "aggregated_ModelA.csv").is_file()
    assert "averaged over 2 studies" in ax.get_title()


def test_plot_requires_exactly_one_source(suite):
    root, *_ = suite
    with pytest.raises(ValueError, match="exactly one"):
        plot_suite_forecast(root, panel_path="x.csv", data=_fake_data())


# --- PanelConfig round-trip (used by the panel-path actuals rebuild) ----------


def test_panelconfig_from_dict_roundtrip():
    from panelclv.configs.panel_config import PanelConfig

    cfg = PanelConfig(
        id_col="Id",
        target_col="Transactions",
        frequency="weekly",
        training_start="2017-01-01",
        training_end="2018-12-31",
        validation_start="2018-07-01",
        holdout_start="2019-01-01",
        holdout_end="2019-12-31",
        time_cols=("year", "week"),
    )
    rebuilt = PanelConfig.from_dict(cfg.to_dict())
    assert rebuilt.to_dict() == cfg.to_dict()
    # Extra unknown keys are ignored, not fatal.
    payload = {**cfg.to_dict(), "some_future_field": 123}
    assert PanelConfig.from_dict(payload).to_dict() == cfg.to_dict()
