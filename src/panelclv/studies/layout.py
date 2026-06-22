"""All path / folder logic for a study suite — pure, no modeling, easy to test.

The on-disk tree the runner produces:

    <studies_base_path>/<study_name>/
        config.json                 # whole-suite record
        results.csv                 # tidy one-row-per-(model, study) table
        <ModelName>/
            config.json             # this model's spec
            metrics.csv             # per-study metrics for this model
            Optuna_Studies/         # neural models only
                study_01/{<name>_best.json, <name>_trials.csv, checkpoints/}
                study_02/...
            Predictions/
                Prediction_1.csv    # MC forecast (or Pareto baseline) per study

Keeping this separate from the runner means the directory contract can be unit
tested without importing torch/optuna, and the folder naming lives in exactly one
place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def create_suite_root(
    studies_base_path: str | Path, study_name: str, overwrite: bool = False
) -> Path:
    """Create (and return) ``<studies_base_path>/<study_name>``.

    The base path must already exist (it is the user's ``Studies`` folder); the
    suite folder is created. If it already exists, refuse unless ``overwrite`` —
    silently writing into a previous suite's folder would mix results.
    """
    base = Path(studies_base_path)
    if not base.is_dir():
        raise FileNotFoundError(
            f"studies_base_path does not exist or is not a directory: {base}"
        )
    root = base / study_name
    if root.exists() and not overwrite:
        raise FileExistsError(
            f"study folder already exists: {root} (pass overwrite=True to reuse it)"
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def model_dirs(root: Path, model_name: str, make_optuna: bool = True) -> dict[str, Path]:
    """Create and return this model's ``model_dir`` / ``predictions_dir`` (+ optuna).

    ``make_optuna`` controls whether the ``Optuna_Studies`` folder is created — the
    Pareto/NBD baseline has no Optuna stage, so it passes ``False``.
    """
    model_dir = Path(root) / model_name
    predictions_dir = model_dir / "Predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    dirs = {"model_dir": model_dir, "predictions_dir": predictions_dir}
    if make_optuna:
        optuna_dir = model_dir / "Optuna_Studies"
        optuna_dir.mkdir(parents=True, exist_ok=True)
        dirs["optuna_dir"] = optuna_dir
    return dirs


def study_dir(model_dir: Path, index: int) -> Path:
    """``<model_dir>/Optuna_Studies/study_{index:02d}`` (zero-padded for sorting)."""
    return Path(model_dir) / "Optuna_Studies" / f"study_{index:02d}"


def prediction_path(model_dir: Path, index: int) -> Path:
    """``<model_dir>/Predictions/Prediction_{index}.csv``."""
    return Path(model_dir) / "Predictions" / f"Prediction_{index}.csv"


def jsonify(obj: Any) -> Any:
    """Recursively coerce an object into something ``json.dump`` can handle.

    The suite/model records contain the ``data_info`` dict, ``seq_cols`` lists,
    nested feature tuples and the occasional ``Path`` / numpy scalar. Sets become
    sorted lists, tuples become lists, ``Path`` becomes ``str``, numpy scalars
    become their Python counterparts, and anything still unknown falls back to
    ``repr`` so writing a record can never crash on an exotic value.
    """
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        # Natural order when the members are mutually comparable (e.g. a numeric
        # batch-size set), falling back to repr order for mixed-type sets.
        try:
            ordered = sorted(obj)
        except TypeError:
            ordered = sorted(obj, key=repr)
        return [jsonify(v) for v in ordered]
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return jsonify(obj.tolist())
    return repr(obj)


def write_json(path: str | Path, obj: Any) -> Path:
    """Write ``obj`` (after ``jsonify``) as pretty JSON; create parents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(jsonify(obj), f, indent=2)
    return path
