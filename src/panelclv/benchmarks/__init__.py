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

The neural benchmark is imported **lazily**: a caller who only wants Pareto/NBD does
not pay for torch. ``from panelclv.benchmarks import ValendinLSTMModel`` works as
usual, but merely importing this subpackage does not load torch.
"""

from typing import TYPE_CHECKING

from .pareto_benchmark import compute_pareto_predictions

if TYPE_CHECKING:  # for type checkers and IDEs only — never executed at runtime
    from .valendin_lstm import InferenceValendinLSTMModel, ValendinLSTMModel

# Attribute name -> the torch-importing module that defines it.
_LAZY = {
    "ValendinLSTMModel": ".valendin_lstm",
    "InferenceValendinLSTMModel": ".valendin_lstm",
}


def __getattr__(name: str):
    """Resolve the torch-backed names on first access (PEP 562)."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache, so the import cost is paid once
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])


__all__ = [
    "compute_pareto_predictions",
    "ValendinLSTMModel",
    "InferenceValendinLSTMModel",
]
