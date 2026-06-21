"""Hyper-parameter / covariate-subset search (Optuna) for the baselines.

Model selection sits a layer *above* both the model and the training loop: it
repeatedly builds, trains and scores candidate models to choose architecture and
feature subsets. It is model-aware (it rebuilds inference models and reuses the
Monte Carlo forecaster for the rollout objective) but is not part of the model
definition — hence its own subpackage.
"""

from .optuna_tuning import (
    run_optuna_study,
    select_features,
    select_features_for_trial,
    weekly_aggregate_rollout_metrics,
    validate_removable_features,
    validate_data_info,
)

__all__ = [
    "run_optuna_study",
    "select_features",
    "select_features_for_trial",
    "weekly_aggregate_rollout_metrics",
    "validate_removable_features",
    "validate_data_info",
]
