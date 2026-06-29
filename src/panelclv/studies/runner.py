"""Orchestrate a study suite: loop models x studies, archive everything.

This is pure orchestration — it adds no modeling logic, it only loops over the
config and calls the existing pieces:

    prepare_dataset (already done by the user, shared as config.data)
        -> make_data_builder                 (panelclv.experiments)
        -> run_optuna_study                  (panelclv.tuning)
        -> refit_best_trial / build_inference_from_trial (panelclv.experiments)
        -> mc_forecast / mc_forecast_transformer         (panelclv.models)
        -> save_predictions_to_csv           (panelclv.evaluation)
        -> mc_compute_metrics                (panelclv.models)

and, for the baseline, ``compute_pareto_predictions`` (panelclv.benchmarks).

For each model it writes a ``config.json`` record and a ``metrics.csv``; per study
it writes the Optuna summary (via ``run_optuna_study``'s own ``summary_dir``) and a
``Prediction_i.csv``. A suite-level ``config.json`` and a tidy ``results.csv``
(one row per ``(model, study)``) round it off for later analysis. See
``panelclv.studies.layout`` for the exact tree.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from panelclv.benchmarks import compute_pareto_predictions
from panelclv.evaluation.plot_utils import save_predictions_to_csv
from panelclv.experiments import (
    build_inference_from_trial,
    make_data_builder,
    refit_best_trial,
)
from panelclv.models import mc_compute_metrics, mc_forecast, mc_forecast_transformer
from panelclv.tuning import run_optuna_study

from .config import StudySuiteConfig, ModelSpec
from . import layout

# Map the LSTM/Transformer family to its Monte Carlo forecaster (the two rollouts
# differ because the architectures carry history differently — see
# monte_carlo_forecasting.py).
_FORECASTERS = {"lstm": mc_forecast, "transformer": mc_forecast_transformer}

# Period length (days) per PanelConfig.frequency, for the Pareto/NBD RFM summary.
_PERIOD_DAYS = {"daily": 1.0, "weekly": 7.0, "monthly": 30.0}


def run_study_suite(config: StudySuiteConfig) -> Path:
    """Run the whole suite and return the path to ``Studies/<study_name>/``.

    Validates the config, creates the suite root, then runs every model. Writes
    the suite-level ``config.json`` and the combined ``results.csv`` at the end.
    """
    config.validate()
    root = layout.create_suite_root(
        config.studies_base_path, config.study_name, overwrite=config.overwrite
    )
    layout.write_json(root / "config.json", _suite_record(config))

    all_rows: list[dict[str, Any]] = []
    for spec in config.models:
        if spec.is_neural:
            rows = _run_neural_model(spec, config, root)
        else:
            rows = _run_pareto_model(spec, config, root)
        all_rows.extend(rows)

    # Tidy, analysis-ready table across every model/study. Differing per-model
    # columns (e.g. neural param_* vs pareto param_penalizer_coef) are unioned by
    # pandas, with NaN where a column does not apply.
    pd.DataFrame(all_rows).to_csv(root / "results.csv", index=False)
    return root


def _run_neural_model(
    spec: ModelSpec, config: StudySuiteConfig, root: Path
) -> list[dict[str, Any]]:
    """Run ``n_studies_per_model`` Optuna studies for one neural model."""
    dirs = layout.model_dirs(root, spec.name, make_optuna=True)
    model_dir = dirs["model_dir"]
    layout.write_json(model_dir / "config.json", _model_record(spec, config))

    forecaster = _FORECASTERS[spec.model_type]
    rows: list[dict[str, Any]] = []

    for i in range(1, config.n_studies_per_model + 1):
        seed = config.base_seed + i
        sdir = layout.study_dir(model_dir, i)
        sdir.mkdir(parents=True, exist_ok=True)

        # The runner owns seed + checkpoint_dir (per study) so studies never share
        # weights; everything else is the user's data_info verbatim. A fresh
        # TPESampler(seed) makes the X studies genuine independent replications.
        data_info = {
            **spec.data_info,
            "seed": seed,
            "checkpoint_dir": str(sdir / "checkpoints"),
        }
        study = run_optuna_study(
            model_type=spec.model_type,
            data_builder=make_data_builder(config.data),
            data_info=data_info,
            n_trials=spec.n_trials,
            device=config.device,
            study_name=f"study_{i:02d}",
            append_timestamp=False,
            summary_dir=sdir,
            sampler=optuna.samplers.TPESampler(seed=seed),
            # Suite-wide disk policy: when True, drop every non-best trial's
            # checkpoint once this study finishes (the winning checkpoint, which
            # the refit/checkpoint rebuild below reloads, is preserved).
            keep_only_best_checkpoint=config.keep_only_best_checkpoint,
        )

        inference_model, data_best = _rebuild_winner(study, spec, config, sdir)

        forecast = forecaster(
            inference_model,
            data_best,
            n_simulations=config.n_simulations,
            seed=seed,
            device=config.device,
            return_simulations=False,
        )
        save_predictions_to_csv(
            forecast["prediction_mean"],
            layout.prediction_path(model_dir, i),
            customer_ids=config.data.get("ids"),
            id_col=config.data.get("id_col", "customer_id"),
        )
        metrics = mc_compute_metrics(forecast["actual"], forecast["prediction_mean"])
        rows.append(
            {
                "model": spec.name,
                "model_type": spec.model_type,
                "study": i,
                "seed": seed,
                "objective": float(study.best_value),
                **metrics,
                **{f"param_{k}": v for k, v in study.best_params.items()},
            }
        )

    pd.DataFrame(rows).to_csv(model_dir / "metrics.csv", index=False)
    return rows


def _rebuild_winner(
    study: "optuna.Study", spec: ModelSpec, config: StudySuiteConfig, sdir: Path
):
    """Rebuild the best trial's forecast-ready model per ``prediction_source``."""
    if config.prediction_source == "refit":
        # Explicit device/checkpoint_dir, then user refit_kwargs (which may override).
        refit_args = {
            "device": config.device,
            "checkpoint_dir": str(sdir / "refit_checkpoints"),
            **config.refit_kwargs,
        }
        return refit_best_trial(study, config.data, spec.model_type, **refit_args)
    return build_inference_from_trial(study, config.data, spec.model_type)


def _run_pareto_model(
    spec: ModelSpec, config: StudySuiteConfig, root: Path
) -> list[dict[str, Any]]:
    """Single deterministic Pareto/NBD fit — no Optuna, one prediction."""
    dirs = layout.model_dirs(root, spec.name, make_optuna=False)
    model_dir = dirs["model_dir"]
    layout.write_json(model_dir / "config.json", _model_record(spec, config))

    data = config.data
    freq = str(data.get("frequency", "weekly")).lower()
    pareto_kwargs = {
        "id_col": data.get("id_col", "Id"),
        "target_col": data.get("target_col", "Transactions"),
        "time_col": "period_start",
        "period_in_days": _PERIOD_DAYS.get(freq, 7.0),
        **spec.pareto_kwargs,
    }
    predictions, ids = compute_pareto_predictions(
        train_panel=data["train_panel"],
        holdout_length=int(data["T_HOLD"]),
        customer_ids=data.get("ids"),
        **pareto_kwargs,
    )
    save_predictions_to_csv(
        predictions,
        layout.prediction_path(model_dir, 1),
        customer_ids=ids,
        id_col=data.get("id_col", "customer_id"),
    )

    # predictions are ordered by data["ids"], and data["holdout"] is in the same
    # customer order, so the actuals line up row-for-row with the predictions.
    actual = np.asarray(data["holdout"])[:, :, int(data["target_idx"])]
    metrics = mc_compute_metrics(actual, predictions)
    row = {
        "model": spec.name,
        "model_type": spec.model_type,
        "study": 1,
        "seed": None,
        "objective": float("nan"),  # no Optuna objective for the baseline
        **metrics,
        "param_penalizer_coef": pareto_kwargs.get("penalizer_coef"),
    }
    pd.DataFrame([row]).to_csv(model_dir / "metrics.csv", index=False)
    return [row]


def _suite_record(config: StudySuiteConfig) -> dict[str, Any]:
    """Serializable record of the whole suite (written to the root config.json).

    Records every ``StudySuiteConfig`` field (bar the bulky ``data`` arrays, which
    are summarised) plus the full ``PanelConfig`` under ``panel_config`` — so the
    file alone documents exactly how the run was configured and how the dataset was
    built. ``panel_config`` is ``None`` only if ``data`` did not come from
    ``prepare_dataset`` (older dicts predate the carried config).
    """
    data = config.data
    panel_config = data.get("panel_config")
    return {
        "study_name": config.study_name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "studies_base_path": str(config.studies_base_path),
        "n_studies_per_model": config.n_studies_per_model,
        "prediction_source": config.prediction_source,
        "n_simulations": config.n_simulations,
        "base_seed": config.base_seed,
        "device": config.device,
        "refit_kwargs": config.refit_kwargs,
        "overwrite": config.overwrite,
        "keep_only_best_checkpoint": config.keep_only_best_checkpoint,
        # Full PanelConfig (all fields, post-normalization) — the dataset recipe.
        "panel_config": panel_config.to_dict() if panel_config is not None else None,
        "models": [
            {
                "name": m.name,
                "model_type": m.model_type,
                "n_trials": m.n_trials,
                "data_info": m.data_info,
                "pareto_kwargs": m.pareto_kwargs,
            }
            for m in config.models
        ],
        "data_summary": {
            "n_customers": len(data.get("ids", [])),
            "T_CAL": data.get("T_CAL"),
            "T_HOLD": data.get("T_HOLD"),
            "F": data.get("F"),
            "seq_cols": data.get("seq_cols"),
            "target_col": data.get("target_col"),
            "validation_start": data.get("validation_start"),
        },
    }


def _model_record(spec: ModelSpec, config: StudySuiteConfig) -> dict[str, Any]:
    """Serializable record of one model's spec (written to its config.json)."""
    record: dict[str, Any] = {
        "name": spec.name,
        "model_type": spec.model_type,
        "prediction_source": config.prediction_source,
        "n_simulations": config.n_simulations,
        "base_seed": config.base_seed,
        "device": config.device,
    }
    if spec.is_neural:
        record["n_trials"] = spec.n_trials
        record["data_info"] = spec.data_info
        record["refit_kwargs"] = config.refit_kwargs
        record["seeds"] = [
            config.base_seed + i for i in range(1, config.n_studies_per_model + 1)
        ]
    else:
        record["pareto_kwargs"] = spec.pareto_kwargs
    return record
