"""Training orchestration for the multinomial LSTM / Transformer baselines.

This subpackage holds the *training loop* — how a model is fit — as opposed to the
model definition itself (architectures, losses, the Monte Carlo simulator), which
lives in ``panelclv.models``. Keeping the loop separate from the model keeps each
subpackage at a single altitude: ``models`` is *what the model is*, ``training`` is
*how it is fit*.
"""

from .training_utils import (
    fit_model,
    refit_full_calibration,
    train_one_epoch,
    validate_one_epoch,
    FitResult,
)

__all__ = [
    "fit_model",
    "refit_full_calibration",
    "train_one_epoch",
    "validate_one_epoch",
    "FitResult",
]
