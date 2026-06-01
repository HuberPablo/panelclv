"""Non-neural benchmark models (Pareto/NBD family).

These are baseline comparators, not part of the neural model family in
``panelclv.models``. Both share the same
``(train_panel, holdout_length, ...) -> (N, H)`` contract and are drop-in
interchangeable:

- ``compute_pareto_predictions``       — frequentist MLE via ``lifetimes`` (default).
- ``compute_pareto_paper_predictions`` — hierarchical-Bayes MCMC port of R's
                                         BTYDplus (the estimator Valendin et al. use).
"""

from .pareto_nbd import compute_pareto_predictions
from .pareto_paper import compute_pareto_paper_predictions

__all__ = [
    "compute_pareto_predictions",
    "compute_pareto_paper_predictions",
]
