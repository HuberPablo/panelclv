"""Pure-Python tests for the study-suite layout + config schema.

These exercise the on-disk folder contract, the path naming, the JSON coercion of
exotic record values, and the up-front config validation — none of which need
torch/optuna, so they run in CI in milliseconds.

Run:  pytest -q tests/test_studies_layout.py
"""

from pathlib import Path

import pytest

from panelclv.studies import ModelSpec, StudySuiteConfig
from panelclv.studies import layout


# --- layout: folder creation -------------------------------------------------


def test_create_suite_root_creates_folder(tmp_path):
    root = layout.create_suite_root(tmp_path, "suite_a")
    assert root == tmp_path / "suite_a"
    assert root.is_dir()


def test_create_suite_root_missing_base_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        layout.create_suite_root(tmp_path / "nope", "suite_a")


def test_create_suite_root_existing_needs_overwrite(tmp_path):
    layout.create_suite_root(tmp_path, "suite_a")
    with pytest.raises(FileExistsError):
        layout.create_suite_root(tmp_path, "suite_a")
    # overwrite=True reuses the folder without error
    again = layout.create_suite_root(tmp_path, "suite_a", overwrite=True)
    assert again.is_dir()


def test_model_dirs_neural_makes_optuna_and_predictions(tmp_path):
    root = layout.create_suite_root(tmp_path, "suite")
    dirs = layout.model_dirs(root, "LSTM", make_optuna=True)
    assert (root / "LSTM" / "Optuna_Studies").is_dir()
    assert (root / "LSTM" / "Predictions").is_dir()
    assert dirs["optuna_dir"] == root / "LSTM" / "Optuna_Studies"


def test_model_dirs_pareto_omits_optuna(tmp_path):
    root = layout.create_suite_root(tmp_path, "suite")
    dirs = layout.model_dirs(root, "ParetoNBD", make_optuna=False)
    assert (root / "ParetoNBD" / "Predictions").is_dir()
    assert not (root / "ParetoNBD" / "Optuna_Studies").exists()
    assert "optuna_dir" not in dirs


def test_path_naming():
    model_dir = Path("/x/suite/LSTM")
    assert layout.study_dir(model_dir, 3) == model_dir / "Optuna_Studies" / "study_03"
    assert layout.prediction_path(model_dir, 2) == model_dir / "Predictions" / "Prediction_2.csv"


# --- layout: jsonify ---------------------------------------------------------


def test_jsonify_handles_sets_tuples_paths():
    raw = {
        "batch_size": {64, 128, 256},
        "lr": (1e-4, 1e-2, "log"),
        "ckpt": Path("/tmp/ckpt"),
        "nested": [{"a": (1, 2)}],
    }
    out = layout.jsonify(raw)
    assert out["batch_size"] == [64, 128, 256]      # set -> sorted list
    assert out["lr"] == [1e-4, 1e-2, "log"]          # tuple -> list
    assert out["ckpt"] == "/tmp/ckpt"                # Path -> str
    assert out["nested"] == [{"a": [1, 2]}]
    # The whole thing must now be JSON-serializable.
    import json
    json.dumps(out)


def test_jsonify_falls_back_to_repr_for_unknown():
    class Weird:
        def __repr__(self):
            return "<weird>"

    assert layout.jsonify(Weird()) == "<weird>"


def test_write_json_roundtrip(tmp_path):
    p = layout.write_json(tmp_path / "sub" / "config.json", {"a": {1, 2}})
    import json
    assert json.loads(p.read_text()) == {"a": [1, 2]}


# --- config validation -------------------------------------------------------


def _data_stub():
    return {"ids": [1], "holdout": [[[0]]], "target_idx": 0,
            "train_panel": object(), "T_HOLD": 1}


def test_validate_ok(tmp_path):
    StudySuiteConfig(
        studies_base_path=tmp_path, study_name="s", data=_data_stub(),
        models=[ModelSpec(name="LSTM", model_type="LSTM")],  # case-normalised
    ).validate()


def test_modelspec_normalises_model_type():
    assert ModelSpec(name="x", model_type="  Transformer ").model_type == "transformer"


def test_validate_rejects_bad_model_type(tmp_path):
    cfg = StudySuiteConfig(
        studies_base_path=tmp_path, study_name="s", data=_data_stub(),
        models=[ModelSpec(name="x", model_type="xgboost")],
    )
    with pytest.raises(ValueError):
        cfg.validate()


def test_validate_rejects_duplicate_names(tmp_path):
    cfg = StudySuiteConfig(
        studies_base_path=tmp_path, study_name="s", data=_data_stub(),
        models=[ModelSpec(name="m", model_type="lstm"),
                ModelSpec(name="m", model_type="transformer")],
    )
    with pytest.raises(ValueError):
        cfg.validate()


def test_validate_rejects_bad_prediction_source(tmp_path):
    cfg = StudySuiteConfig(
        studies_base_path=tmp_path, study_name="s", data=_data_stub(),
        models=[ModelSpec(name="m", model_type="lstm")],
        prediction_source="warmstart",
    )
    with pytest.raises(ValueError):
        cfg.validate()


def test_validate_rejects_missing_data_keys(tmp_path):
    cfg = StudySuiteConfig(
        studies_base_path=tmp_path, study_name="s", data={"ids": [1]},
        models=[ModelSpec(name="m", model_type="lstm")],
    )
    with pytest.raises(KeyError):
        cfg.validate()
