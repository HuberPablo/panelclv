"""Thin orchestration helpers tying prepare_dataset -> Optuna -> forecast together.

These functions absorb the mechanical glue that otherwise gets hand-copied into
every experiment notebook -- DataLoader shaping, the Optuna ``data_builder``
closure, and the post-study "rebuild the winning model and load its checkpoint"
step. They deliberately hold **no** modeling logic and remove no tuning knobs: the
notebook still drives ``run_optuna_study`` and the Monte Carlo forecaster directly.
Centralising only the boilerplate makes the bugs that kept recurring there
(missing import, mis-cased ``model_type``, forecasting on ``data_full`` instead of
the trial's sliced ``data_best``, silent ``load_state_dict`` mismatch) structurally
impossible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# This orchestration glue sits above the model + tuning layers, so it imports
# them by absolute path: inference wrappers from `panelclv.models`, the feature
# selection helpers from `panelclv.tuning`.
from panelclv.models.multinomial_lstm import (
    MultinomialLSTMModel,
    InferenceMultinomialLSTMModel,
)
from panelclv.models.multinomial_transformer import (
    MultinomialTransformerModel,
    InferenceMultinomialTransformerModel,
)
from panelclv.training.training_utils import refit_full_calibration
from panelclv.tuning.optuna_tuning import select_features, select_features_for_trial

if TYPE_CHECKING:  # optuna only needed for the type hint; avoid an import-time dep here
    import optuna


# The closure signature run_optuna_study expects (see optuna_tuning module docstring):
# data_builder(feature_config, batch_size) -> (train_loader, val_loader, metadata).
DataBuilder = Callable[..., "tuple[DataLoader, DataLoader, dict[str, Any]]"]

# Default epoch count for the final full-calibration retrain when the caller passes
# n_epochs=None. The Valendin et al. paper describes "a few epochs" of big-batch
# warm-start fine-tuning, so this is a small fixed number rather than the (possibly
# large) number of epochs the tuning run took to converge.
DEFAULT_REFIT_EPOCHS = 5


def _require_val_start_idx(data: dict[str, Any]) -> int:
    """Read the temporal-split boundary from a prepare_dataset dict, or fail clearly.

    The customer-wise split was removed: every loader is built from the temporal
    validation window, which `prepare_dataset` records as `val_start_idx` (set from
    `PanelConfig.validation_start`). A dict missing it predates this change or was not
    produced by `prepare_dataset`.
    """
    s = data.get("val_start_idx")
    if s is None:
        raise KeyError(
            "data['val_start_idx'] is missing — build the dataset with prepare_dataset "
            "from a PanelConfig that sets validation_start (the temporal validation "
            "window). The customer-wise train_idx/val_idx split is no longer supported."
        )
    s = int(s)
    if s < 2:
        # Need >= 1 training transition (samples[:, :s-1] must be non-empty).
        raise ValueError(
            f"val_start_idx={s} leaves no training transitions; validation_start is too "
            f"close to training_start. Move validation_start later in the calibration window."
        )
    return s


def make_loaders(
    data: dict[str, Any],
    batch_size: int,
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """Shape one ``prepare_dataset`` / ``select_features`` dict into train+val loaders.

    The split is **temporal** (a time window over ALL customers), not customer-wise:
    the calibration window is cut at ``data["val_start_idx"]`` (= ``s``, the first
    validation PERIOD index, set by ``prepare_dataset`` from ``validation_start``). Over
    the AR axis ``samples[t]`` predicts calibration period ``t+1``, so:

      - **train** uses transitions whose target period is < ``s`` (the training prefix):
        ``X[:, :s-1]`` / ``y[:, :s-1]``. The model never consumes a validation period
        during training.
      - **val** uses the FULL sequence ``X`` / ``y`` so the recurrent/causal state warms
        up over the whole prefix; ``metadata["val_score_start"] = s-1`` then tells
        ``fit_model`` to score cross-entropy only on the validation suffix
        (periods ``s..T_CAL-1``).

    Tensor contract (same for every model in this package):
      - ``samples`` : (N, T-1, F) float32 inputs (already float32 out of data prep).
      - ``targets`` : (N, T-1) int64 class indices -- ``squeeze(-1)`` drops the trailing
                      singleton feature axis; the values index the softmax head.

    The returned ``metadata`` is the recipe ``run_optuna_study`` and the model
    constructors need to rebuild a matching network: ``seq_cols`` (feature-axis column
    names), ``embedded_cols`` (embedding cardinalities), ``target_col``, ``seq_len`` (the
    TRAIN sequence length ``s-1``, used by the Transformer's fixed-length mask cache; the
    longer val sequence simply rebuilds a mask on the fly, and the LSTM ignores it), and
    ``val_score_start`` (= ``s-1``).
    """
    s = _require_val_start_idx(data)

    X = data["samples"]                                 # (N, T-1, F) float32
    y = data["targets"].squeeze(-1).astype(np.int64)    # (N, T-1) class indices

    # Train on the prefix transitions only (targets at periods 1..s-1); validate on the
    # full sequence but score only the suffix (see metadata["val_score_start"]).
    X_train, y_train = X[:, : s - 1], y[:, : s - 1]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=shuffle_train,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=False,                                  # val order is irrelevant and must stay stable
    )

    metadata = {                                        # the recipe to build the matching model
        "seq_cols":        data["seq_cols"],
        "embedded_cols":   data["embedded_cols"],
        "target_col":      data["target_col"],
        "seq_len":         X_train.shape[1],            # train length s-1 (Transformer mask cache)
        "val_score_start": s - 1,                       # score CE on the validation suffix only
    }
    return train_loader, val_loader, metadata


def make_refit_loader(
    data: dict[str, Any],
    batch_size: int,
    shuffle_train: bool = True,
) -> DataLoader:
    """A single loader over the FULL calibration window for the final warm-start retrain.

    Unlike ``make_loaders`` (which truncates training to the pre-validation prefix), this
    yields every AR transition ``samples`` / ``targets`` (all T-1 steps), so the
    fine-tune in ``training.refit_full_calibration`` also learns from the validation-tail
    periods. ``batch_size`` is typically large (the paper's "big batch" final step).
    """
    X = data["samples"]                                 # (N, T-1, F) float32
    y = data["targets"].squeeze(-1).astype(np.int64)    # (N, T-1) class indices
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle_train,
    )


def make_data_builder(data_full: dict[str, Any]) -> DataBuilder:
    """Build the ``data_builder`` closure ``run_optuna_study`` calls once per trial.

    Optuna proposes a ``feature_config`` (which removable covariates to drop) and a
    ``batch_size``; the closure slices ``data_full`` to that feature subset with
    ``select_features`` and returns the matching temporal loaders + metadata. The
    train/val split is the same time boundary for every trial (carried in
    ``data_full["val_start_idx"]``), so trials differ only by hyperparameters and
    feature set.
    """

    def data_builder(feature_config: Sequence[str], batch_size: int):
        data = select_features(data_full, feature_config)   # drop chosen cols -> smaller F
        return make_loaders(data, batch_size)

    return data_builder


def build_inference_from_trial(
    study: "optuna.Study",
    data_full: dict[str, Any],
    model_type: str,
    checkpoint_path: str | Path | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild the study's winning model + load its checkpoint, ready to forecast.

    Returns ``(inference_model, data_best)`` where ``data_best`` is ``data_full``
    sliced to the best trial's feature subset. **Both** are needed downstream: the
    Monte Carlo forecaster must be fed ``data_best`` (so its ``seq_cols`` /
    ``target_idx`` match the trained weights), never ``data_full`` -- returning it
    here removes that footgun.

    The checkpoint was trained on the sliced layout, so the inference model is built
    from ``data_best["seq_cols"]`` / ``["embedded_cols"]`` and the best trial's
    architecture params. The inference model always samples a count class per step
    (see CLAUDE.md "Critical modeling distinction") — that is the forecast mechanism.

    ``checkpoint_path`` overrides which weights are loaded; pass the path returned by
    ``refit_best_trial`` to forecast with the full-calibration refit instead of the
    tuning checkpoint. Default ``None`` loads the best trial's own checkpoint.
    """
    family = model_type.strip().lower()
    if family not in ("lstm", "transformer"):
        raise ValueError(
            f"model_type must be 'lstm' or 'transformer', got {model_type!r}"
        )

    best = study.best_trial
    params = best.params
    if checkpoint_path is None:
        checkpoint_path = best.user_attrs["checkpoint_path"]
    # Slice to the winning feature set so seq_cols/embedded_cols line up with the weights.
    data_best = select_features_for_trial(data_full, best)

    common = dict(
        seq_cols=data_best["seq_cols"],
        embedded_cols=data_best["embedded_cols"],
        target_col=data_best["target_col"],
    )

    if family == "lstm":
        inference_model: torch.nn.Module = InferenceMultinomialLSTMModel(
            **common,
            embedding_dim=params["embedding_dim"],
            lstm_hidden_size=params["lstm_hidden_size"],
            dense_units=params["dense_units"],
            dropout=params["dropout"],
        )
    else:
        inference_model = InferenceMultinomialTransformerModel(
            **common,
            d_model=params["d_model"],
            nhead=params["nhead"],
            num_encoder_layers=params["num_encoder_layers"],
            dropout=params["dropout"],
        )

    state = torch.load(checkpoint_path, map_location="cpu")
    # The training Transformer may register a fixed-length "_cached_mask" buffer that
    # the inference model (which rebuilds the mask per call) has no slot for. It is
    # saved non-persistently now, but older checkpoints can still carry the key -- drop
    # it so the load stays strict over every real weight. No-op for the LSTM.
    state.pop("_cached_mask", None)
    inference_model.load_state_dict(state)
    # NB: no .eval() here -- the Monte Carlo forecaster calls model.eval() itself.

    return inference_model, data_best


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
    verbose: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Warm-start retrain the study's best model on the FULL calibration window.

    The Valendin et al. paper's final step: take the architecture / stopping epoch the
    Optuna study selected on the temporal validation window, then fine-tune the winning
    weights for a few epochs (big batch) on the full calibration window — validation tail
    included — so the model also learns the most recent dynamics. Returns the same
    ``(inference_model, data_best)`` pair as ``build_inference_from_trial``, but the
    inference model carries the **refit** weights, ready to forecast the real holdout.

    Steps: slice ``data_full`` to the best trial's feature set, rebuild the TRAINING
    model from the best params (full-calibration ``seq_len``), warm-start it from the
    trial checkpoint, run ``training.refit_full_calibration`` over the full-calibration
    loader, then rebuild the inference model and load the refit checkpoint via
    ``build_inference_from_trial(..., checkpoint_path=...)``.

    ``n_epochs`` defaults to ``DEFAULT_REFIT_EPOCHS`` (a small "few epochs" warm-start,
    per the paper); pass an explicit int to override. ``batch_size`` defaults to a large
    value (the paper's big-batch final step).
    """
    family = model_type.strip().lower()
    if family not in ("lstm", "transformer"):
        raise ValueError(
            f"model_type must be 'lstm' or 'transformer', got {model_type!r}"
        )

    best = study.best_trial
    params = best.params
    warm_start_ckpt = best.user_attrs["checkpoint_path"]
    if n_epochs is None:
        # The paper's final step is "a few epochs" of big-batch fine-tuning, so default
        # to a small fixed count (DEFAULT_REFIT_EPOCHS) rather than the tuning run's
        # epoch count, which can be large and would over-train the warm-started weights.
        n_epochs = DEFAULT_REFIT_EPOCHS

    # Slice to the winning feature set; build the TRAINING model at the FULL calibration
    # length (samples span all T-1 transitions here, not the truncated training prefix).
    data_best = select_features_for_trial(data_full, best)
    train_meta = {
        "seq_cols":      data_best["seq_cols"],
        "embedded_cols": data_best["embedded_cols"],
        "target_col":    data_best["target_col"],
        "seq_len":       data_best["samples"].shape[1],
    }
    if family == "lstm":
        model: torch.nn.Module = MultinomialLSTMModel(
            seq_cols=train_meta["seq_cols"],
            embedded_cols=train_meta["embedded_cols"],
            target_col=train_meta["target_col"],
            embedding_dim=params["embedding_dim"],
            lstm_hidden_size=params["lstm_hidden_size"],
            dense_units=params["dense_units"],
            dropout=params["dropout"],
        )
    else:
        model = MultinomialTransformerModel(
            seq_cols=train_meta["seq_cols"],
            embedded_cols=train_meta["embedded_cols"],
            target_col=train_meta["target_col"],
            seq_len=train_meta["seq_len"],
            d_model=params["d_model"],
            nhead=params["nhead"],
            num_encoder_layers=params["num_encoder_layers"],
            dropout=params["dropout"],
        )

    refit_loader = make_refit_loader(data_best, batch_size)
    result = refit_full_calibration(
        model,
        refit_loader,
        max_trans=model.num_target_classes,
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
        verbose=verbose,
    )

    # Rebuild the inference model and load the REFIT weights (not the tuning checkpoint).
    return build_inference_from_trial(
        study, data_full, model_type, checkpoint_path=result.checkpoint_path,
    )
