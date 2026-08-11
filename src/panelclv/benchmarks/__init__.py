"""Non-neural benchmark models (Pareto/NBD).

The frozen reference comparator for the neural model family in
``panelclv.models``. It shares their
``(train_panel, holdout_length, ...) -> (N, H)`` contract, so it drops into the
same plots, metrics tables and study runner:

- ``compute_pareto_predictions`` — hierarchical-Bayes MCMC port of R's BTYDplus
                                   (the estimator Valendin et al. use).

An earlier frequentist-MLE variant (via ``lifetimes``) is kept for provenance
under ``benchmarks/archive/`` and is not part of the public API.
"""

from .pareto_benchmark import compute_pareto_predictions

__all__ = [
    "compute_pareto_predictions",
]
