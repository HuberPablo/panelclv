"""Declarative schema for a study suite — what the user fills in, nothing more.

A *study suite* runs ``n_studies_per_model`` independent Optuna studies for each
of several models over a single shared dataset, keeps the best trial of every
study, forecasts it, and archives everything under ``Studies/<suite_name>/`` (see
``panelclv.studies.runner`` for the orchestration and ``layout`` for the on-disk
tree). The user only writes two dataclasses:

- ``ModelSpec`` — one per model: the same arguments already passed to
  ``run_optuna_study`` (``model_type``, the ``search_space`` overrides, the
  ``training`` knobs, ``n_trials``), plus ``pareto_kwargs`` for the non-Optuna
  Pareto/NBD baseline. Anything a spec leaves out of ``search_space`` keeps the
  range the model's registry entry declares.
- ``StudySuiteConfig`` — the suite-wide settings: where to write, how many studies
  per model, and the single shared ``prepare_dataset`` dict (``data``) used by
  every model.

Both are plain dataclasses with a ``validate()`` that fails loudly and early — a
typo'd ``model_type`` or a missing base path should raise before any training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Which model families exist is the registry's business, not this file's: the suite
# accepts exactly the types declared there (ADR-0006). Neural ones go through Optuna;
# a non-neural entry (Pareto/NBD) is a single deterministic fit — no tuning, one
# prediction — which ``is_neural`` reads off the entry rather than a second list.
from panelclv.registry import MODEL_TYPES, is_neural

# Keys the runner reads off the shared ``prepare_dataset`` dict. Checked up front
# so a dict that did not come from ``prepare_dataset`` fails clearly rather than
# midway through the first study.
REQUIRED_DATA_KEYS = (
    "ids", "holdout", "target_idx", "train_panel", "T_HOLD",
    # The panel's own column names. The Pareto/NBD baseline fits on them
    # directly and guessing either would fit the wrong column, so a suite that
    # lacks them has to fail here rather than after writing its config.json.
    "id_col", "target_col",
)


@dataclass
class ModelSpec:
    """One model in the suite — the same kwargs you pass to ``run_optuna_study``.

    Parameters
    ----------
    name
        Folder name for this model under the study root (e.g. ``"LSTM"``).
    model_type
        Any key of the model registry: ``"lstm"``, ``"transformer"``,
        ``"valendin_lstm"`` (Optuna-tuned) or ``"pareto_nbd"`` (baseline).
        ``"valendin_lstm"`` is the frozen published benchmark: its architecture is
        fixed, so only training hyperparameters (learning rate, weight decay, batch
        size) are searched.
    search_space
        Per-hyperparameter overrides of the registry entry's default range, in the
        ``registry.suggest_param`` mini-language (a ``{...}`` set is a categorical, a
        ``(lo, hi, "log"|"int")`` tuple a range, a scalar is pinned). Omitted
        parameters keep the entry's own range; a key the model does not have raises.
        Ignored for ``pareto_nbd``.
    training
        The controls that are not searched: ``n_epochs``, ``patience``,
        ``loss_type``, ``class_weights``, ``focal_gamma``, ``grad_clip``,
        ``verbose``, ``log_wandb``. Ignored for ``pareto_nbd``. The runner adds
        ``seed`` and ``checkpoint_dir`` per study; do not set them here.
    n_trials
        Optuna trials per study. Ignored for ``pareto_nbd``.
    pareto_kwargs
        Extra arguments forwarded to ``compute_pareto_predictions`` (the MCMC
        knobs ``mcmc``, ``burnin``, ``thin``, ``chains``, ``seed``,
        ``param_init``). ``seed`` defaults to the suite's ``base_seed``. Only used
        for ``pareto_nbd``.
    """

    name: str
    model_type: str
    search_space: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    n_trials: int = 50
    pareto_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalise once so downstream comparisons can assume lower-case.
        self.model_type = str(self.model_type).strip().lower()

    @property
    def is_neural(self) -> bool:
        """Whether this model trains, read off its registry entry (ADR-0006)."""
        return is_neural(self.model_type)


@dataclass
class StudySuiteConfig:
    """Suite-wide settings plus the single shared dataset every model uses.

    Parameters
    ----------
    studies_base_path
        Existing ``Studies`` directory; the suite folder is created inside it.
    suite_name
        New folder created under ``studies_base_path`` for this whole suite.
    data
        The ``prepare_dataset`` output, shared by every model.
    models
        The models to run.
    n_studies_per_model
        How many independent Optuna studies to run per neural model (each gets its
        own seed). Coerced to 1 for the deterministic Pareto/NBD baseline.
    n_simulations
        Monte Carlo paths per forecast.
    base_seed
        Study ``i`` uses ``base_seed + i`` for its sampler and training, so the
        studies are genuine independent replications.
    device, refit_kwargs, overwrite
        Passed through to the trainer / forecaster; ``overwrite`` allows reusing an
        existing suite folder.
    keep_only_best_checkpoint
        Disk policy for the per-study Optuna search. ``False`` (default) keeps every
        trial's ``.pth``; these accumulate fast (``n_trials`` per study × every
        study). ``True`` forwards to ``run_optuna_study`` so that, once each study
        completes, all non-best trial checkpoints are deleted — only the study's
        winning checkpoint survives. That winner is exactly what the refit warm-starts
        from (ADR-0008), so the forecast is unaffected; you only lose the ability to
        inspect losing trials' weights.
    """

    studies_base_path: str | Path
    suite_name: str
    data: dict[str, Any]
    models: list[ModelSpec]
    n_studies_per_model: int = 5
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
        if not self.suite_name or "/" in self.suite_name or "\\" in self.suite_name:
            raise ValueError(
                f"suite_name must be a single folder name, got {self.suite_name!r}"
            )
        if not self.models:
            raise ValueError("models is empty — add at least one ModelSpec")

        names = [m.name for m in self.models]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"ModelSpec names must be unique; duplicates: {dupes}")

        for m in self.models:
            if m.model_type not in MODEL_TYPES:
                raise ValueError(
                    f"model {m.name!r}: model_type must be one of {MODEL_TYPES}, "
                    f"got {m.model_type!r}"
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
