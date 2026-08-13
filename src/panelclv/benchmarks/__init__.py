"""Frozen reference implementations — models we reproduce, not develop (ADR-0004).

Two benchmarks the thesis compares against, both sharing the infrastructure around
the architecture (data preparation, the embedder seam, the training loop, the Monte
Carlo simulator, evaluation) so a comparison isolates architecture:

- ``compute_pareto_predictions`` — the Pareto/NBD, a hierarchical-Bayes MCMC port of
                                   R's BTYDplus (the estimator Valendin et al. use).
- ``pareto_from_data`` /         — the same fit, driven from a ``prepare_dataset``
  ``pareto_forecast``              output: the array, and the dict shape the neural
                                   rollouts return. They live here because they
                                   build a Pareto/NBD forecast.
- ``ValendinLSTMModel`` /        — the Valendin et al. LSTM, transcribed layer for
  ``RolloutValendinLSTMModel``     layer from the reference notebook. The trained
                                   class hands over the rollout one (ADR-0007),
                                   declared inside the frozen file.

An earlier frequentist-MLE Pareto/NBD (via ``lifetimes``) was retired; it is kept for
provenance in the repo-root ``archive/``, outside the package.
"""

from .pareto_nbd import (
    compute_pareto_predictions,
    pareto_forecast,
    pareto_from_data,
)
from .valendin_lstm import RolloutValendinLSTMModel, ValendinLSTMModel

__all__ = [
    "compute_pareto_predictions",
    "pareto_from_data",
    "pareto_forecast",
    "ValendinLSTMModel",
    "RolloutValendinLSTMModel",
]
