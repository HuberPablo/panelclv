"""Read a finished study suite back off disk: its models, forecasts and dataset.

The runner (`panelclv.studies.runner`) leaves a tree like::

    Studies/<name>/
        config.json                     # whole-suite record (carries panel_config)
        <ModelName>/Predictions/Prediction_{i}.csv   # per-study MC-mean forecasts

This module is the read side of that tree — the layer `suite_plots` and
`suite_metrics` are both built on. It never touches checkpoints, Optuna DBs or
`results.csv`; it only reads `config.json` + the `Prediction_*.csv` files, and it
writes the aggregated per-model CSVs. Four jobs:

1. **Discovery** — which models a suite holds, in the config's own order, and the
   column names its CSVs were written with.
2. ``load_model_predictions`` — one study's forecast, or the per-customer mean
   across every study, for a single model.
3. ``aggregate_suite_predictions`` — write that across-studies mean to
   ``Studies/<name>/aggregated_<ModelName>.csv`` (flat at the suite root, one wide
   CSV per model, same column format as the per-study files).
4. ``_actuals_from_panel`` / ``describe_dataset`` — rebuild the dataset the suite ran
   on and describe it.

**Why a dataset path.** The suite archives the *recipe* (`panel_config`) but not the
dataset arrays, so the actual transaction curves are not on disk. Given the
customer-period panel CSV, we rebuild the exact `calibration`/`holdout` tensors the
suite used by re-running `prepare_dataset` with the persisted config — same code path,
same cohort order, so predictions and actuals line up by construction.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from panelclv.configs.panel_config import PanelConfig
from panelclv.data_preparation.panel_dataset import prepare_dataset
from panelclv.data_preparation.target_channel import (
    calibration_counts,
    holdout_actuals,
)
from panelclv.predictions import (
    DEFAULT_ID_COL,
    load_predictions_from_csv,
    save_predictions_to_csv,
)
from panelclv.registry import MODEL_TYPES, is_neural


# ---------------------------------------------------------------------------
# Suite discovery
# ---------------------------------------------------------------------------


def _read_suite_config(root: Path) -> dict[str, Any] | None:
    """Return the suite's ``config.json`` as a dict, or ``None`` if absent."""
    cfg_path = Path(root) / "config.json"
    if not cfg_path.is_file():
        return None
    with open(cfg_path) as f:
        return json.load(f)


def _discover_models(root: Path) -> list[tuple[str, Path]]:
    """List ``(model_name, model_dir)`` for every model with predictions.

    Order comes from ``config.json``'s ``models`` list (so the plot legend is
    reproducible, not filesystem-order); models whose ``Predictions/`` folder is
    missing or empty are skipped. If ``config.json`` has no model list (a hand-made
    or partial tree), fall back to scanning for any subdirectory that contains a
    non-empty ``Predictions/`` folder, alphabetically.
    """
    root = Path(root)
    cfg = _read_suite_config(root)

    names: list[str]
    if cfg and cfg.get("models"):
        names = [m["name"] for m in cfg["models"]]
    else:
        names = sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and (p / "Predictions").is_dir()
        )

    found: list[tuple[str, Path]] = []
    for name in names:
        model_dir = root / name
        preds = model_dir / "Predictions"
        if preds.is_dir() and any(preds.glob("Prediction_*.csv")):
            found.append((name, model_dir))
    if not found:
        raise FileNotFoundError(
            f"no models with a non-empty Predictions/ folder under {root}"
        )
    return found


def _id_col(root: Path) -> str:
    """The customer-id column name for saved CSVs, read from the suite config.

    Falls back to ``save_predictions_to_csv``'s own default so a config-less tree
    still works.
    """
    cfg = _read_suite_config(root) or {}
    pc = cfg.get("panel_config") or {}
    summary = cfg.get("data_summary") or {}
    return pc.get("id_col") or summary.get("id_col") or DEFAULT_ID_COL


# ---------------------------------------------------------------------------
# Loading / aggregating predictions
# ---------------------------------------------------------------------------


def _prediction_index(path: Path) -> int:
    """Sort key for ``Prediction_{i}.csv`` — the integer ``i`` (10 sorts after 2)."""
    m = re.search(r"Prediction_(\d+)\.csv$", path.name)
    return int(m.group(1)) if m else -1


# Which families the runner runs per study (one Prediction_i.csv each) is read off the
# registry rather than restated here: a local copy of the neural list is exactly what
# drifted once, and a stale one misreads a whole model's archive as a single
# deterministic fit (ADR-0006).


def _is_deterministic_model(model_dir: Path) -> bool:
    """True for a single-fit benchmark (e.g. Pareto/NBD), read from its config.json."""
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.is_file():
        return False
    with open(cfg_path) as f:
        mt = json.load(f).get("model_type")
    if mt is None:
        return False
    # An archive may name a model type this build no longer registers; the reader's
    # job is to locate files, so an unknown type reads as a single fit rather than
    # refusing to read the folder at all.
    runs_once_per_study = mt in MODEL_TYPES and is_neural(mt)
    return not runs_once_per_study


def load_model_predictions(
    model_dir: str | Path, study: int | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load one model's holdout forecast as ``(values (N, T_HOLD), ids)``.

    Parameters
    ----------
    model_dir
        ``Studies/<name>/<ModelName>`` — the folder holding ``Predictions/``.
    study
        ``int`` → load exactly that study's ``Prediction_{study}.csv``.
        ``None`` → average across every ``Prediction_*.csv`` (the per-customer,
        per-timestep mean over studies). A single-study model averages to itself.

    In average mode all studies must describe the same cohort in the same order
    and the same horizon; a mismatch raises ``ValueError`` naming the offenders,
    because a silent misalignment would corrupt the mean.
    """
    preds_dir = Path(model_dir) / "Predictions"

    if study is not None:
        # A deterministic benchmark (Pareto/NBD) has no per-study variation, so the
        # runner writes only Prediction_1.csv. Always load that, whatever study index
        # was requested; every other model honors the requested index as before.
        if _is_deterministic_model(preds_dir.parent):
            study = 1
        path = preds_dir / f"Prediction_{study}.csv"
        if not path.is_file():
            available = sorted(
                p.name for p in preds_dir.glob("Prediction_*.csv")
            )
            raise FileNotFoundError(
                f"{path} not found; available predictions: {available or 'none'}"
            )
        return load_predictions_from_csv(path)

    paths = sorted(preds_dir.glob("Prediction_*.csv"), key=_prediction_index)
    if not paths:
        raise FileNotFoundError(f"no Prediction_*.csv under {preds_dir}")

    values_stack: list[np.ndarray] = []
    ref_ids: np.ndarray | None = None
    ref_path: Path | None = None
    for path in paths:
        vals, ids = load_predictions_from_csv(path)
        if ref_ids is None:
            ref_ids, ref_path = ids, path
        else:
            if vals.shape != values_stack[0].shape:
                raise ValueError(
                    f"prediction shape mismatch: {ref_path.name} is "
                    f"{values_stack[0].shape} but {path.name} is {vals.shape} — "
                    f"cannot average studies over different cohorts/horizons"
                )
            if ref_ids is not None and ids is not None and not np.array_equal(ids, ref_ids):
                raise ValueError(
                    f"customer ids differ between {ref_path.name} and {path.name} — "
                    f"studies must forecast the same customers in the same order to average"
                )
        values_stack.append(vals)

    # (n_studies, N, T) -> mean over studies -> (N, T).
    mean_values = np.stack(values_stack, axis=0).mean(axis=0)
    return mean_values, ref_ids


def aggregate_suite_predictions(root: str | Path) -> list[Path]:
    """Write each model's across-studies mean forecast to the suite root.

    For every model, average its ``Prediction_*.csv`` files (per customer, per
    timestep) and write the result as ``Studies/<name>/aggregated_<ModelName>.csv``
    — flat at the suite root, next to the model folders, one wide CSV per model in
    the same ``id_col + week_0..week_{T-1}`` format as the per-study files (so it
    round-trips through ``load_predictions_from_csv``). Overwrites in place on
    re-run (the aggregate is a pure function of the predictions on disk). Returns
    the list of files written.
    """
    root = Path(root)
    id_col = _id_col(root)
    written: list[Path] = []
    for name, model_dir in _discover_models(root):
        mean_values, ids = load_model_predictions(model_dir, study=None)
        out = root / f"aggregated_{name}.csv"
        save_predictions_to_csv(mean_values, out, customer_ids=ids, id_col=id_col)
        written.append(out)
    return written


# ---------------------------------------------------------------------------
# Actuals — rebuilt from the dataset path via the persisted recipe
# ---------------------------------------------------------------------------


def _actuals_from_panel(root: Path, panel_path: str | Path) -> dict[str, Any]:
    """Rebuild the ``prepare_dataset`` dict from the panel CSV + persisted config.

    Reads ``config.json``'s ``panel_config`` (the recipe the suite ran with),
    reconstructs the ``PanelConfig``, and re-runs ``prepare_dataset`` on the panel
    at ``panel_path``. Because it is the same function and config the suite used,
    the returned ``calibration``/``holdout`` tensors, ``target_idx`` and cohort
    ``ids`` match the run's exactly — so the actuals align with the stored
    predictions.
    """
    cfg = _read_suite_config(root)
    if not cfg or cfg.get("panel_config") is None:
        raise ValueError(
            f"{root}/config.json has no panel_config, so the actuals cannot be "
            f"rebuilt from a panel path (the suite predates the carried config). "
            f"Pass data=<prepare_dataset output> to plot_suite_forecast instead."
        )
    panel_path = Path(panel_path)
    if not panel_path.is_file():
        raise FileNotFoundError(f"panel not found: {panel_path}")

    panel_config = PanelConfig.from_dict(cfg["panel_config"])
    panel = pd.read_csv(panel_path)
    return prepare_dataset(panel, panel_config, verbose=False)


# ---------------------------------------------------------------------------
# Dataset description — global characteristics over the selected period
# ---------------------------------------------------------------------------


def describe_dataset(data: dict[str, Any], *, name: str | None = None):
    """Global characteristics of a prepared dataset over its selected period.

    The ``prepare_dataset`` dict already encodes the period chosen in the study's
    ``PanelConfig`` (training / validation / holdout dates): ``calibration`` covers
    ``[training_start, training_end]`` and ``holdout`` covers ``[holdout_start,
    holdout_end]``. Counts are read from the target channel of those dense
    ``(N, T, F)`` tensors — the same source the metrics table and plots use, so the
    numbers are cohort-aligned with the rest of the analysis. "Overall" means
    calibration + holdout combined.

    Note the temporal split (see CLAUDE.md): the validation window is the **tail** of
    the calibration window, so ``calibration_length`` already includes it and
    ``training_length`` is the pre-validation prefix (``calibration_length -
    validation_length``). Counts reflect the target as prepared; if the config set
    ``clip_target_upper``, per-period counts are capped at that value.

    Parameters
    ----------
    data
        A ``prepare_dataset`` output.
    name
        Name for the returned Series (handy when concatenating several datasets into
        one comparison table via ``pd.concat([...], axis=1)``).

    Returns a ``pandas.Series`` of labelled characteristics.
    """
    ids = np.asarray(data["ids"])
    N = len(ids)
    # The target channel of each window, (N, T) — the dataset's own count column.
    calibration = calibration_counts(data)
    holdout = holdout_actuals(data)

    T_CAL = int(data.get("T_CAL", calibration.shape[1]))
    T_HOLD = int(data.get("T_HOLD", holdout.shape[1]))
    n_val = int(data.get("n_val_periods", 0))
    # Pre-validation prefix length: weights train on [0, val_start_idx).
    val_start = int(data.get("val_start_idx", T_CAL - n_val))

    # Per (customer, period) integer counts over each window. rint guards the rare
    # float-dtype tensor so exact-count comparisons (== 1) are safe.
    calib_cells = np.rint(calibration).astype(np.int64)             # (N, T_CAL)
    hold_cells = np.rint(holdout).astype(np.int64)                  # (N, T_HOLD)
    calib_counts = calib_cells.sum(axis=1)                          # (N,) per customer
    hold_counts = hold_cells.sum(axis=1)
    total_counts = calib_counts + hold_counts                      # "overall" per customer

    def pct(count: Any) -> float:
        """Share of the cohort, as a percentage (0-100)."""
        return round(100.0 * float(count) / N, 2) if N else float("nan")

    # Prefer the carried PanelConfig for the period dates; fall back to loose data keys.
    pc = data.get("panel_config")

    def cfg(attr: str) -> Any:
        val = getattr(pc, attr, None) if pc is not None else None
        return val if val is not None else data.get(attr)

    zero_cells = np.concatenate([calib_cells, hold_cells], axis=1) == 0

    metrics: dict[str, Any] = {
        # --- the selected period (straight from the study's PanelConfig) ---
        "frequency": data.get("frequency"),
        "training_start": cfg("training_start"),
        "training_end": cfg("training_end"),
        "validation_start": cfg("validation_start"),
        "holdout_start": cfg("holdout_start"),
        "holdout_end": cfg("holdout_end"),
        # --- window sizes (customers / periods) ---
        "cohort_size": N,
        "calibration_length": T_CAL,          # incl. the validation tail
        "training_length": val_start,         # pre-validation prefix
        "validation_length": n_val,
        "holdout_length": T_HOLD,
        # --- transaction volume ---
        "total_transactions_overall": int(total_counts.sum()),
        "total_transactions_calibration": int(calib_counts.sum()),
        "total_transactions_holdout": int(hold_counts.sum()),
        "avg_transactions_per_customer": round(float(total_counts.mean()), 3),
        "median_transactions_per_customer": float(np.median(total_counts)),
        "avg_transactions_per_customer_calibration": round(float(calib_counts.mean()), 3),
        "avg_transactions_per_customer_holdout": round(float(hold_counts.mean()), 3),
        "transactions_per_customer_per_period": round(
            float(total_counts.sum() / (N * (T_CAL + T_HOLD))), 4
        ),
        # --- per-customer activity distribution (overall) ---
        "customers_with_0_transactions": int((total_counts == 0).sum()),
        "pct_customers_with_0_transactions": pct((total_counts == 0).sum()),
        "customers_with_1_transaction": int((total_counts == 1).sum()),
        "customers_with_lt5_transactions": int((total_counts < 5).sum()),
        "pct_customers_with_lt5_transactions": pct((total_counts < 5).sum()),
        "pct_repeat_customers_ge2": pct((total_counts >= 2).sum()),
        "p90_transactions_per_customer": float(np.percentile(total_counts, 90)),
        "p99_transactions_per_customer": float(np.percentile(total_counts, 99)),
        "max_transactions_per_customer": int(total_counts.max()),
        # --- churn / sparsity ---
        "pct_customers_inactive_in_holdout": pct((hold_counts == 0).sum()),
        "pct_customers_inactive_in_calibration": pct((calib_counts == 0).sum()),
        "panel_sparsity_pct_zero_cells": round(100.0 * float(zero_cells.mean()), 2),
    }
    return pd.Series(metrics, name=name or "dataset")


def describe_suite_dataset(
    root: str | Path,
    panel_path: str | Path | None = None,
    *,
    data: dict[str, Any] | None = None,
    name: str | None = None,
):
    """Describe a finished suite's dataset from ROOT + panel — companion to the plot.

    Rebuilds the exact dataset the suite ran on (via the persisted ``panel_config``)
    and returns :func:`describe_dataset` for it, so the characteristics match the
    period the study actually used. Exactly one of ``panel_path`` / ``data`` must be
    given (``data`` skips the rebuild). ``name`` defaults to the suite folder name.
    """
    root = Path(root)
    if (panel_path is None) == (data is None):
        raise ValueError("pass exactly one of panel_path= or data=")
    if data is None:
        data = _actuals_from_panel(root, panel_path)
    return describe_dataset(data, name=name or root.name)
