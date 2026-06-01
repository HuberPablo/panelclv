"""Multinomial LSTM / Transformer model family for transaction-count forecasting.

This subpackage is scoped to the **model definition** only — the architectures, the
loss functions they optimise, and the autoregressive Monte Carlo simulator that
turns the categorical head into a forecast (per the Valendin design, the simulator
*is* the model's forecast mechanism, not a post-hoc step).

The surrounding concerns each have their own sibling subpackage under ``panelclv``:

- ``panelclv.training``    — the training loop (``fit_model``, ...).
- ``panelclv.tuning``      — Optuna architecture / covariate-subset search.
- ``panelclv.evaluation``  — metrics, plotting, forecast diagnostics, prediction CSV I/O.
- ``panelclv.benchmarks``  — the non-neural Pareto/NBD comparators.
- ``panelclv.experiments`` — thin prepare -> tune -> forecast orchestration glue.
"""

from .multinomial_lstm import (
    MultinomialLSTMModel,
    InferenceMultinomialLSTMModel,
)
from .multinomial_transformer import (
    MultinomialTransformerModel,
    InferenceMultinomialTransformerModel,
)
from .losses import (
    FocalLoss,
    SquaredEMDLoss,
    compute_class_weights,
    build_criterion,
)
from .monte_carlo_forecasting import (
    # Canonical names...
    run_monte_carlo_forecast,
    run_monte_carlo_forecast_transformer,
    # ...and the short ``mc_*`` aliases used throughout the notebooks.
    run_monte_carlo_forecast as mc_forecast,
    run_monte_carlo_forecast_transformer as mc_forecast_transformer,
    simulate_one_path as mc_simulate_one_path,
    simulate_transformer_path as mc_simulate_transformer_path,
    compute_forecast_metrics as mc_compute_metrics,
)

# `__all__` is the curated *headline* surface for the model family. Everything
# imported above stays importable by explicit name; only the advertised set
# (`from panelclv.models import *`, autocompletion, docs) is trimmed. Internals kept
# OFF this list but still importable: train-time loss classes/helpers
# (FocalLoss, SquaredEMDLoss, compute_class_weights, build_criterion) and the
# per-path simulator entry points (mc_simulate_one_path, mc_simulate_transformer_path).
__all__ = [
    # Model + inference wrappers, both families
    "MultinomialLSTMModel",
    "InferenceMultinomialLSTMModel",
    "MultinomialTransformerModel",
    "InferenceMultinomialTransformerModel",
    # Forecasting (autoregressive Monte Carlo simulator + its metrics)
    "mc_forecast",
    "mc_forecast_transformer",
    "mc_compute_metrics",
]
