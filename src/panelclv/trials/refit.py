"""Turning a finished study into the model that forecasts the holdout (ADR-0008).

A study leaves behind a winning trial: an architecture, a feature subset and a
checkpoint. Getting from there to a forecast is a refit — the winning weights are
warm-started and fine-tuned for a few large-batch epochs over the **full** calibration
window, validation tail included, per Valendin et al. That is the only route to a
forecast in this package, and it ends by asking the refit model for its rollout model
(ADR-0007) rather than rebuilding one beside it from the study's stored parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

# Models are built through the registry rather than constructed here, so this module
# never has to know which architectures exist (ADR-0006).
from panelclv.registry import build_model
from panelclv.training.loop import refit_full_calibration
from panelclv.tuning.optuna_tuning import select_features_for_trial

from .loaders import refit_loader

if TYPE_CHECKING:  # optuna only needed for the type hint; avoid an import-time dep here
    import optuna

# Default epoch count for the refit when the caller passes n_epochs=None. The Valendin
# et al. paper describes "several fine-tuning epochs" of big-batch warm-start training,
# so this is a small fixed number rather than the (possibly large) number of epochs the
# tuning run took to converge.
DEFAULT_REFIT_EPOCHS = 5


def refit_best_trial(
    study: "optuna.Study",
    data_full: dict[str, Any],
    model_type: str,
    *,
    n_epochs: int | None = None,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    device: str | torch.device | None = None,
    checkpoint_dir: str | Path = "./checkpoints",
    loss_type: str = "cross_entropy",
    class_weights: "torch.Tensor | None" = None,
    focal_gamma: float = 2.0,
    emd_weight: float = 1.0,
    verbose: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Warm-start retrain the study's best model on the FULL calibration window.

    The paper's final step, and this package's only route to a forecast (ADR-0008):
    take the architecture / stopping epoch the Optuna study selected on the temporal
    validation window, then fine-tune the winning weights for a few epochs (big batch)
    on the full calibration window — validation tail included — so the model also
    learns the most recent dynamics. Returns ``(rollout_model, data_best)`` where
    ``data_best`` is ``data_full`` sliced to the best trial's feature subset. **Both**
    are needed downstream: the Monte Carlo forecaster must be fed ``data_best`` (so its
    ``seq_cols`` / ``target_idx`` match the trained weights), never ``data_full`` —
    returning it here removes that footgun.

    Steps: slice ``data_full`` to the best trial's feature set, build the TRAINING model
    from the best params (full-calibration ``seq_len``), warm-start it from the trial
    checkpoint, run ``training.refit_full_calibration`` over the full-calibration
    loader, then hand back the rollout model that model itself provides.

    ``n_epochs`` defaults to ``DEFAULT_REFIT_EPOCHS`` (the paper's "several" epochs);
    pass an explicit int to override. ``batch_size`` defaults to a large value (the
    paper's big-batch final step).
    """
    family = model_type.strip().lower()

    best = study.best_trial
    params = best.params
    warm_start_ckpt = best.user_attrs["checkpoint_path"]
    if n_epochs is None:
        # The paper's final step is "several" epochs of big-batch fine-tuning, so default
        # to a small fixed count (DEFAULT_REFIT_EPOCHS) rather than the tuning run's
        # epoch count, which can be large and would over-train the warm-started weights.
        n_epochs = DEFAULT_REFIT_EPOCHS

    # Slice to the winning feature set; build the TRAINING model at the FULL calibration
    # length (samples span all T-1 transitions here, not the truncated training prefix).
    data_best = select_features_for_trial(data_full, best)
    train_recipe = {
        "seq_cols":      data_best["seq_cols"],
        "embedded_cols": data_best["embedded_cols"],
        "target_col":    data_best["target_col"],
        "seq_len":       data_best["samples"].shape[1],
    }
    # Built through the registry, so a model type reaches refit only if it is wired
    # everywhere else too — and an unregistered one raises here rather than silently
    # refitting another architecture.
    model: torch.nn.Module = build_model(family, params, train_recipe)

    refit_full_calibration(
        model,
        refit_loader(data_best, batch_size),
        num_target_classes=model.num_target_classes,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
        checkpoint_dir=checkpoint_dir,
        model_name=f"{family}_refit_trial_{best.number}",
        warm_start_state=warm_start_ckpt,
        loss_type=loss_type,
        class_weights=class_weights,
        focal_gamma=focal_gamma,
        emd_weight=emd_weight,
        verbose=verbose,
    )

    # The refit trains a fixed number of epochs with no validation set, so the weights
    # it ends holding ARE the weights it saved. Asking the model for its rollout model
    # (ADR-0007) hands over that same backbone — no file round-trip, and no second
    # construction whose arguments could disagree with the trained one's.
    return model.to_rollout(), data_best
