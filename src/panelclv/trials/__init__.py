"""Assembling and refitting one trial.

A *trial* is one trained model with one sampled set of hyperparameters and features
(``CONTEXT.md``); this subpackage is the altitude at which a single trial is built and
turned into a forecast-ready model. Above it, ``tuning`` searches over trials and
``studies`` runs many studies.

It holds two things a reader should not expect to find elsewhere:

- ``loaders`` — ``split_calibration``, the **sole enforcement point of ADR-0001**
  (training truncates before the validation window; validation keeps the full sequence
  and scores from a later index). That is modeling logic, not glue, and it lives here
  because ``data_preparation`` is deliberately numpy-only and this is where numpy
  becomes tensors.
- ``refit`` — the warm-start fine-tune over the full calibration window that produces
  every forecast in this package (ADR-0008).

It sits at the top of the dependency stack and imports from ``panelclv.tuning``,
``panelclv.training`` and ``panelclv.registry`` — never ``panelclv.models``, which it
reaches only through the registry's builders (ADR-0006).
"""

from .loaders import (
    CalibrationSplit,
    split_calibration,
    refit_loader,
    make_data_builder,
)
from .refit import refit_best_trial

__all__ = [
    "CalibrationSplit",
    "split_calibration",
    "refit_loader",
    "make_data_builder",
    "refit_best_trial",
]
