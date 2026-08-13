"""Multinomial LSTM / Transformer model family for transaction-count forecasting.

This subpackage is scoped to the **model definition** only — the architectures, the
embedders that feed them, the loss functions they optimise, and the autoregressive
Monte Carlo simulator that turns the categorical head into a forecast (per the
Valendin design, the simulator *is* the model's forecast mechanism, not a post-hoc
step).

``embedders`` is shared infrastructure rather than architecture: how features become
a vector is a swappable component a model is given (ADR-0005), so the frozen
benchmark in ``panelclv.benchmarks`` draws its strategy from here too, exactly as it
draws the simulator.

The surrounding concerns each have their own sibling subpackage under ``panelclv``:

- ``panelclv.registry``    — the table declaring which of these models exist.
- ``panelclv.training``    — the training loop (``fit_model``, ...).
- ``panelclv.tuning``      — Optuna architecture / covariate-subset search.
- ``panelclv.evaluation``  — metrics, plotting, forecast diagnostics, prediction CSV I/O.
- ``panelclv.benchmarks``  — the non-neural Pareto/NBD comparator.
- ``panelclv.trials``      — assembling and refitting one trial.
"""

from .embedders import (
    Embedder,
    ProjectedEmbedder,
    ValendinEmbedder,
)
from .multinomial_lstm import (
    MultinomialLSTMModel,
    RolloutMultinomialLSTMModel,
)
from .multinomial_transformer import (
    MultinomialTransformerModel,
    RolloutMultinomialTransformerModel,
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
    # The scoring authority keeps the name CLAUDE.md uses for it; it never had a
    # per-path variant, so there is nothing for an ``mc_*`` alias to disambiguate.
    compute_forecast_metrics,
    # ...and the short ``mc_*`` aliases the notebooks call.
    run_monte_carlo_forecast as mc_forecast,
    run_monte_carlo_forecast_transformer as mc_forecast_transformer,
)

# `__all__` is the curated *headline* surface for the model family. Everything
# imported above stays importable by explicit name; only the advertised set
# (`from panelclv.models import *`, autocompletion, docs) is trimmed. Internals kept
# OFF this list but still importable: the train-time loss classes/helpers
# (FocalLoss, SquaredEMDLoss, compute_class_weights, build_criterion).
#
# The per-path steppers `simulate_one_path` / `simulate_transformer_path` are NOT
# re-exported here. They are internals of the two forecast entry points above; the
# `mc_simulate_*` aliases that once advertised them had no importer anywhere and were
# deleted (ledger `models/__init__` rows).
__all__ = [
    # The embedder seam: how features become a vector (ADR-0005). A model is given
    # one; swapping it is how the published architecture and ours differ.
    "Embedder",
    "ProjectedEmbedder",
    "ValendinEmbedder",
    # Trained model + the rollout model it hands over, both families
    "MultinomialLSTMModel",
    "RolloutMultinomialLSTMModel",
    "MultinomialTransformerModel",
    "RolloutMultinomialTransformerModel",
    # Forecasting (autoregressive Monte Carlo simulator + its metrics)
    "mc_forecast",
    "mc_forecast_transformer",
    "compute_forecast_metrics",
]
