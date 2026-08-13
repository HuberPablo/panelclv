"""Frozen reference implementations — models we reproduce, not develop (ADR-0004).

Two benchmarks the thesis compares against, both sharing the infrastructure around
the architecture (data preparation, the embedder seam, the training loop, the Monte
Carlo simulator, evaluation) so a comparison isolates architecture:

- ``compute_pareto_predictions`` — the Pareto/NBD, a hierarchical-Bayes MCMC port of
                                   R's BTYDplus (the estimator Valendin et al. use).
- ``ValendinLSTMModel`` /        — the Valendin et al. LSTM, transcribed layer for
  ``InferenceValendinLSTMModel``   layer from the reference notebook.

An earlier frequentist-MLE Pareto/NBD (via ``lifetimes``) was retired; it is kept for
provenance in the repo-root ``archive/``, outside the package.
"""

from .pareto_benchmark import compute_pareto_predictions
from .valendin_lstm import InferenceValendinLSTMModel, ValendinLSTMModel

__all__ = [
    "compute_pareto_predictions",
    "ValendinLSTMModel",
    "InferenceValendinLSTMModel",
]
