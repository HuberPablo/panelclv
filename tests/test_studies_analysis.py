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
    compare_study_metrics,
    load_model_predictions,
    plot_suite_forecast,
    study_metrics,
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


def _fake_data(t=T):
    """A stand-in prepare_dataset dict for the plot path (data= escape hatch).

    Only the keys plot_suite_forecast reads: calibration/holdout (N, T, F) tensors,
    target_idx and ids. F=1, target channel 0. ``t`` sets the holdout length (used by the
    compare-suites mismatch test).
    """
    calibration = np.ones((N, 6, 1))           # 6 training weeks
    holdout = np.ones((N, t, 1)) * 2.0          # t holdout weeks
    return {
        "calibration": calibration,
        "holdout": holdout,
        "target_idx": 0,
        "ids": IDS,
    }


def _make_suite(root, t=T, offset=0.0):
    """Write a two-model suite (ModelA: 2 studies, ModelB: 1) with t-week forecasts."""
    (root / "ModelA" / "Predictions").mkdir(parents=True)
    (root / "ModelB" / "Predictions").mkdir(parents=True)
    a1 = np.arange(N * t, dtype=float).reshape(N, t) + offset
    _write_prediction(root / "ModelA", 1, a1)
    _write_prediction(root / "ModelA", 2, a1 + 2.0)
    _write_prediction(root / "ModelB", 1, np.ones((N, t)) * 5.0)
    with open(root / "config.json", "w") as f:
        json.dump(
            {"models": [{"name": "ModelA"}, {"name": "ModelB"}],
             "data_summary": {"id_col": "customer_id"}},
            f,
        )


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


# --- study_metrics (whole-cohort metrics, with SD / CI / display) -------------

METRIC_COLS = ["rmse", "bias_percent", "mape_aggregate"]


@pytest.fixture
def patched_actuals(suite, monkeypatch):
    """Point study_metrics' actuals rebuild at a controlled holdout — no panel needed.

    Monkeypatches ``_actuals_from_panel`` so ``panel_path`` is irrelevant; the metrics
    are then scored against ``_fake_data()``'s holdout (all 2.0, so no zero-denominator
    NaNs in bias/MAPE). Returns ``(suite_tuple, actual_2d)``.
    """
    from panelclv.studies import analysis

    data = _fake_data()
    monkeypatch.setattr(analysis, "_actuals_from_panel", lambda root, panel_path: data)
    actual = data["holdout"][:, :, data["target_idx"]]      # (N, T) the metric fn sees
    return suite, actual


def _expected_per_study(actual, preds_by_model):
    """{model: {metric: [per-study values]}} via the same metric fn study_metrics uses."""
    from panelclv.models import compute_forecast_metrics

    out = {}
    for name, preds in preds_by_model.items():
        vals = {m: [] for m in METRIC_COLS}
        for p in preds:
            r = compute_forecast_metrics(actual, p)
            for m in METRIC_COLS:
                vals[m].append(r[m])
        out[name] = vals
    return out


def test_study_metrics_default_returns_means(patched_actuals):
    (root, a1, a2, b1), actual = patched_actuals
    tbl = study_metrics(root, "ignored.csv")

    assert list(tbl.columns) == METRIC_COLS + ["n_studies"]
    assert tbl.loc["ModelA", "n_studies"] == 2       # 2 studies averaged
    assert tbl.loc["ModelB", "n_studies"] == 1

    exp = _expected_per_study(actual, {"ModelA": [a1, a2], "ModelB": [b1]})
    for m in METRIC_COLS:
        assert tbl.loc["ModelA", m] == pytest.approx(np.mean(exp["ModelA"][m]))
        assert tbl.loc["ModelB", m] == pytest.approx(exp["ModelB"][m][0])


def test_study_metrics_standard_deviation(patched_actuals):
    (root, a1, a2, b1), actual = patched_actuals
    tbl = study_metrics(root, "x", standard_deviation=True)

    # stat columns are exactly mean/std/n — no CI when only SD is asked for.
    assert set(tbl.columns.get_level_values("stat")) == {"mean", "std", "n"}

    exp = _expected_per_study(actual, {"ModelA": [a1, a2]})
    for m in METRIC_COLS:
        assert tbl.loc["ModelA", (m, "std")] == pytest.approx(
            np.std(exp["ModelA"][m], ddof=1)
        )
    # A single-study model has no spread.
    assert np.isnan(tbl.loc["ModelB", ("rmse", "std")])
    assert tbl.loc["ModelB", ("rmse", "n")] == 1


def test_study_metrics_ci_is_symmetric_t_interval(patched_actuals):
    from scipy import stats as sstats

    (root, a1, a2, b1), actual = patched_actuals
    tbl = study_metrics(root, "x", confidence_interval=True, ci=0.95)

    assert set(tbl.columns.get_level_values("stat")) == {"mean", "ci_low", "ci_high", "n"}

    exp = _expected_per_study(actual, {"ModelA": [a1, a2]})
    n = 2
    tcrit = sstats.t.ppf(0.975, n - 1)
    for m in METRIC_COLS:
        vals = exp["ModelA"][m]
        mean, sd = np.mean(vals), np.std(vals, ddof=1)
        half = tcrit * sd / np.sqrt(n)
        lo = tbl.loc["ModelA", (m, "ci_low")]
        hi = tbl.loc["ModelA", (m, "ci_high")]
        assert lo == pytest.approx(mean - half)
        assert hi == pytest.approx(mean + half)
        assert (lo + hi) / 2 == pytest.approx(mean)        # symmetric about the mean
    # Single-study model: interval undefined.
    assert np.isnan(tbl.loc["ModelB", ("rmse", "ci_low")])


def test_study_metrics_display_sd_strings(patched_actuals):
    (root, a1, a2, b1), actual = patched_actuals
    tbl = study_metrics(root, "x", standard_deviation=True, display=True, decimals=3)

    exp = _expected_per_study(actual, {"ModelA": [a1, a2]})
    mean = np.mean(exp["ModelA"]["rmse"])
    sd = np.std(exp["ModelA"]["rmse"], ddof=1)
    assert tbl.loc["ModelA", "rmse"] == f"{mean:.3f} ± {sd:.3f}"
    # Single-study model shows just the mean — no ± term.
    assert "±" not in tbl.loc["ModelB", "rmse"]


def test_study_metrics_display_both_labels_terms(patched_actuals):
    (root, *_), _ = patched_actuals
    tbl = study_metrics(
        root, "x", standard_deviation=True, confidence_interval=True, display=True
    )
    cell = tbl.loc["ModelA", "rmse"]
    assert "(SD)" in cell and "(CI)" in cell            # both terms labelled, unambiguous


def test_study_metrics_display_needs_a_spread_flag(patched_actuals):
    (root, *_), _ = patched_actuals
    with pytest.raises(ValueError, match="display=True needs"):
        study_metrics(root, "x", display=True)


# --- compare_study_metrics (several suites, one panel) ------------------------


@pytest.fixture
def two_suites(tmp_path, monkeypatch):
    """Two consistent suites (same holdout) with actuals rebuild monkeypatched away."""
    from panelclv.studies import analysis

    root_a, root_b = tmp_path / "A", tmp_path / "B"
    _make_suite(root_a)
    _make_suite(root_b)
    monkeypatch.setattr(analysis, "_actuals_from_panel", lambda root, panel_path: _fake_data())
    return root_a, root_b


def test_compare_stacks_with_model_suite_index(two_suites):
    root_a, root_b = two_suites
    out = compare_study_metrics({"A": root_a, "B": root_b}, "ignored.csv")

    # (model, suite): models in discovery order, suites in the dict order.
    assert out.index.names == ["model", "suite"]
    assert out.index.tolist() == [
        ("ModelA", "A"), ("ModelA", "B"),
        ("ModelB", "A"), ("ModelB", "B"),
    ]
    # Each block equals the single-suite study_metrics for that root.
    single = study_metrics(root_a, "ignored.csv")
    assert out.loc[("ModelA", "A"), "rmse"] == pytest.approx(single.loc["ModelA", "rmse"])
    assert out.loc[("ModelA", "A"), "n_studies"] == 2


def test_compare_display_and_ci_pass_through(two_suites):
    root_a, root_b = two_suites
    out = compare_study_metrics(
        {"A": root_a, "B": root_b}, "x", standard_deviation=True, display=True
    )
    assert out.index.names == ["model", "suite"]
    assert "±" in out.loc[("ModelA", "A"), "rmse"]     # display strings survive stacking


def test_compare_rejects_more_than_four(two_suites):
    root_a, _ = two_suites
    with pytest.raises(ValueError, match="between 1 and 4"):
        compare_study_metrics({f"s{i}": root_a for i in range(5)}, "x")


def test_compare_rejects_empty():
    with pytest.raises(ValueError, match="between 1 and 4"):
        compare_study_metrics({}, "x")


def test_compare_warns_on_mismatched_holdout(tmp_path, monkeypatch):
    from panelclv.studies import analysis

    root_a, root_b = tmp_path / "A", tmp_path / "B"
    _make_suite(root_a, t=T)          # 4-week holdout
    _make_suite(root_b, t=T + 2)      # 6-week holdout — internally consistent, but differs

    # Return actuals matching each suite's own forecast width, so the only inconsistency is
    # the across-suite holdout mismatch the warning is meant to catch.
    monkeypatch.setattr(
        analysis, "_actuals_from_panel",
        lambda root, panel_path: _fake_data(T if root.name == "A" else T + 2),
    )
    with pytest.warns(UserWarning, match="holdout length"):
        compare_study_metrics({"A": root_a, "B": root_b}, "x")


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
