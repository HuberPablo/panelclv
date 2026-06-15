"""Experiment orchestration glue.

Thin helpers that tie ``prepare_dataset`` -> Optuna -> Monte Carlo forecast
together (DataLoader shaping, the Optuna ``data_builder`` closure, and the
"rebuild the winning trial and load its checkpoint" step). They hold no modeling
logic; they sit at the top of the dependency stack and import from
``panelclv.models`` and ``panelclv.tuning``. Centralising only this boilerplate
keeps the recurring notebook bugs (missing import, mis-cased ``model_type``,
forecasting on the wrong data slice, silent ``load_state_dict`` mismatch)
structurally impossible.
"""

from .experiment_utils import (
    make_loaders,
    make_refit_loader,
    make_data_builder,
    build_inference_from_trial,
    refit_best_trial,
)

__all__ = [
    "make_loaders",
    "make_refit_loader",
    "make_data_builder",
    "build_inference_from_trial",
    "refit_best_trial",
]
