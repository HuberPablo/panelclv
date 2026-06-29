"""Declarative schema for a study suite — what the user fills in, nothing more.

A *study suite* runs ``n_studies_per_model`` independent Optuna studies for each
of several models over a single shared dataset, keeps the best trial of every
study, forecasts it, and archives everything under ``Studies/<study_name>/`` (see
``panelclv.studies.runner`` for the orchestration and ``layout`` for the on-disk
tree). The user only writes two dataclasses:

- ``ModelSpec`` — one per model: the same arguments already passed to
  ``run_optuna_study`` (``model_type``, the ``data_info`` training-knobs dict,
  ``n_trials``), plus ``pareto_kwargs`` for the non-Optuna Pareto/NBD baseline.
  The hyperparameter/feature search space lives inside ``run_optuna_study`` /
  ``make_data_builder``, so it is intentionally NOT restated here.
- ``StudySuiteConfig`` — the suite-wide settings: where to write, how many studies
  per model, the prediction source, and the single shared ``prepare_dataset``
  dict (``data``) used by every model.

Both are plain dataclasses with a ``validate()`` that fails loudly and early — a
typo'd ``model_type`` or a missing base path should raise before any training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Recognised model families. The two neural families go through Optuna; the
# Pareto/NBD baseline is a single deterministic fit (no tuning, one prediction).
NEURAL_MODEL_TYPES = ("lstm", "transformer")
VALID_MODEL_TYPES = NEURAL_MODEL_TYPES + ("pareto_nbd",)
VALID_PREDICTION_SOURCES = ("refit", "checkpoint")

# Keys the runner reads off the shared ``prepare_dataset`` dict. Checked up front
# so a dict that did not come from ``prepare_dataset`` fails clearly rather than
# midway through the first study.
REQUIRED_DATA_KEYS = ("ids", "holdout", "target_idx", "train_panel", "T_HOLD")


@dataclass
class ModelSpec:
    """One model in the suite — the same kwargs you pass to ``run_optuna_study``.

    Parameters
    ----------
    name
        Folder name for this model under the study root (e.g. ``"LSTM"``).
    model_type
        ``"lstm"``, ``"transformer"`` (Optuna-tuned) or ``"pareto_nbd"`` (baseline).
    data_info
        The training-knobs dict from the README quickstart (``n_epochs``,
        ``patience``, ``loss_type``, …). Ignored for ``pareto_nbd``. The runner
        adds ``seed`` and ``checkpoint_dir`` per study; do not set them here.
    n_trials
        Optuna trials per study. Ignored for ``pareto_nbd``.
    pareto_kwargs
        Extra arguments forwarded to ``compute_pareto_predictions`` (e.g.
        ``penalizer_coef``). Only used for ``pareto_nbd``.
    """

    name: str
    model_type: str
    data_info: dict[str, Any] = field(default_factory=dict)
    n_trials: int = 50
    pareto_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalise once so downstream comparisons can assume lower-case.
        self.model_type = str(self.model_type).strip().lower()

    @property
    def is_neural(self) -> bool:
        return self.model_type in NEURAL_MODEL_TYPES


@dataclass
class StudySuiteConfig:
    """Suite-wide settings plus the single shared dataset every model uses.

    Parameters
    ----------
    studies_base_path
        Existing ``Studies`` directory; the suite folder is created inside it.
    study_name
        New folder created under ``studies_base_path`` for this whole suite.
    data
        The ``prepare_dataset`` output, shared by every model.
    models
        The models to run.
    n_studies_per_model
        How many independent Optuna studies to run per neural model (each gets its
        own seed). Coerced to 1 for the deterministic Pareto/NBD baseline.
    prediction_source
        ``"refit"`` (warm-start retrain on the full calibration window — the
        paper's final step, the default) or ``"checkpoint"`` (the best trial's own
        tuning weights).
    n_simulations
        Monte Carlo paths per forecast.
    base_seed
        Study ``i`` uses ``base_seed + i`` for its sampler and training, so the
        studies are genuine independent replications.
    device, refit_kwargs, overwrite
        Passed through to the trainer / forecaster; ``overwrite`` allows reusing an
        existing study-name folder.
    keep_only_best_checkpoint
        Disk policy for the per-study Optuna search. ``False`` (default) keeps every
        trial's ``.pth``; these accumulate fast (``n_trials`` per study × every
        study). ``True`` forwards to ``run_optuna_study`` so that, once each study
        completes, all non-best trial checkpoints are deleted — only the study's
        winning checkpoint survives. That winner is exactly what ``prediction_source``
        (``"refit"`` warm-start or ``"checkpoint"``) rebuilds from, so the forecast
        is unaffected; you only lose the ability to inspect losing trials' weights.
    """

    studies_base_path: str | Path
    study_name: str
    data: dict[str, Any]
    models: list[ModelSpec]
    n_studies_per_model: int = 5
    prediction_source: str = "refit"
    n_simulations: int = 600
    base_seed: int = 42
    device: str | None = None
    refit_kwargs: dict[str, Any] = field(default_factory=dict)
    overwrite: bool = False
    keep_only_best_checkpoint: bool = False

    def validate(self) -> None:
        """Fail loudly before any training on a misconfigured suite."""
        base = Path(self.studies_base_path)
        if not base.is_dir():
            raise FileNotFoundError(
                f"studies_base_path does not exist or is not a directory: {base}"
            )
        if not self.study_name or "/" in self.study_name or "\\" in self.study_name:
            raise ValueError(
                f"study_name must be a single folder name, got {self.study_name!r}"
            )
        if not self.models:
            raise ValueError("models is empty — add at least one ModelSpec")

        names = [m.name for m in self.models]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"ModelSpec names must be unique; duplicates: {dupes}")

        for m in self.models:
            if m.model_type not in VALID_MODEL_TYPES:
                raise ValueError(
                    f"model {m.name!r}: model_type must be one of {VALID_MODEL_TYPES}, "
                    f"got {m.model_type!r}"
                )

        if self.prediction_source not in VALID_PREDICTION_SOURCES:
            raise ValueError(
                f"prediction_source must be one of {VALID_PREDICTION_SOURCES}, "
                f"got {self.prediction_source!r}"
            )
        if self.n_studies_per_model < 1:
            raise ValueError(
                f"n_studies_per_model must be >= 1, got {self.n_studies_per_model}"
            )
        if not isinstance(self.data, dict):
            raise TypeError("data must be the dict returned by prepare_dataset")
        missing = [k for k in REQUIRED_DATA_KEYS if k not in self.data]
        if missing:
            raise KeyError(
                f"data is missing keys {missing}; pass the dict returned by "
                f"prepare_dataset"
            )
