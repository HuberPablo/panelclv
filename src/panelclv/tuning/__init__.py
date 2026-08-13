"""Hyper-parameter / covariate-subset search (Optuna) for the baselines.

Model selection sits a layer *above* both the model and the training loop: it
repeatedly builds, trains and scores candidate models to choose architecture and
feature subsets. It is model-aware (it builds each candidate from the same
constructors the training path uses, through ``panelclv.registry``) but is not part
of the model definition — hence its own subpackage. Which model types exist, and what
each one searches, is declared in the registry (ADR-0006), not here.
"""

from .optuna_tuning import (
    run_optuna_study,
    select_features,
    select_features_for_trial,
    validate_removable_features,
)

__all__ = [
    "run_optuna_study",
    "select_features",
    "select_features_for_trial",
    "validate_removable_features",
]
