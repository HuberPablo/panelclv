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

from .multinomial_lstm import InferenceMultinomialLSTMModel
from .multinomial_transformer import InferenceMultinomialTransformerModel
from .optuna_tuning import select_features, select_features_for_trial

if TYPE_CHECKING:  # optuna only needed for the type hint; avoid an import-time dep here
    import optuna


# The closure signature run_optuna_study expects (see optuna_tuning module docstring):
# data_builder(feature_config, batch_size) -> (train_loader, val_loader, metadata).
DataBuilder = Callable[..., "tuple[DataLoader, DataLoader, dict[str, Any]]"]


def make_loaders(
    data: dict[str, Any],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    batch_size: int,
    shuffle_train: bool = True,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """Shape one ``prepare_dataset`` / ``select_features`` dict into train+val loaders.

    Mirrors the tensor contract every model in this package trains against:
      - ``samples`` : (N, T-1, F) float32 inputs (already float32 out of data prep).
      - ``targets`` : (N, T-1) int64 class indices -- ``squeeze(-1)`` drops the trailing
                      singleton feature axis; the values index the softmax head.
    Customers are split by row index (``train_idx`` / ``val_idx``), so the split is
    customer-wise and no customer leaks between train and val.

    The returned ``metadata`` is exactly the recipe ``run_optuna_study`` and the model
    constructors need to rebuild a matching network: ``seq_cols`` (feature-axis column
    names), ``input_spec`` (embedding cardinalities), ``target_col``, and ``seq_len``
    (the training sequence length T-1, used by the Transformer's fixed-length mask
    cache; the LSTM ignores it).
    """
    X = data["samples"]                                 # (N, T-1, F) float32
    y = data["targets"].squeeze(-1).astype(np.int64)    # (N, T-1) class indices

    train_idx = list(train_idx)
    val_idx = list(val_idx)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx])),
        batch_size=batch_size,
        shuffle=shuffle_train,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx])),
        batch_size=batch_size,
        shuffle=False,                                  # val order is irrelevant and must stay stable
    )

    metadata = {                                        # the recipe to build the matching model
        "seq_cols":   data["seq_cols"],
        "input_spec": data["input_spec"],
        "target_col": data["target_col"],
        "seq_len":    X.shape[1],
    }
    return train_loader, val_loader, metadata


def make_data_builder(
    data_full: dict[str, Any],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
) -> DataBuilder:
    """Build the ``data_builder`` closure ``run_optuna_study`` calls once per trial.

    Optuna proposes a ``feature_config`` (which removable covariates to drop) and a
    ``batch_size``; the closure slices ``data_full`` to that feature subset with
    ``select_features`` and returns the matching loaders + metadata. Splitting the
    customers once, here, keeps the train/val split identical across every trial, so
    trials differ only by their hyperparameters and feature set -- not by which
    customers they happened to see.
    """

    def data_builder(feature_config: Sequence[str], batch_size: int):
        data = select_features(data_full, feature_config)   # drop chosen cols -> smaller F
        return make_loaders(data, train_idx, val_idx, batch_size)

    return data_builder


def build_inference_from_trial(
    study: "optuna.Study",
    data_full: dict[str, Any],
    model_type: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Rebuild the study's winning model + load its checkpoint, ready to forecast.

    Returns ``(inference_model, data_best)`` where ``data_best`` is ``data_full``
    sliced to the best trial's feature subset. **Both** are needed downstream: the
    Monte Carlo forecaster must be fed ``data_best`` (so its ``seq_cols`` /
    ``target_idx`` match the trained weights), never ``data_full`` -- returning it
    here removes that footgun.

    The checkpoint was trained on the sliced layout, so the inference model is built
    from ``data_best["seq_cols"]`` / ``["input_spec"]`` and the best trial's
    architecture params. ``mode="sample"`` is the mode the simulator requires (sample
    a count class per step; see CLAUDE.md "Critical modeling distinction").
    """
    family = model_type.strip().lower()
    if family not in ("lstm", "transformer"):
        raise ValueError(
            f"model_type must be 'lstm' or 'transformer', got {model_type!r}"
        )

    best = study.best_trial
    params = best.params
    checkpoint_path = best.user_attrs["checkpoint_path"]
    # Slice to the winning feature set so seq_cols/input_spec line up with the weights.
    data_best = select_features_for_trial(data_full, best)

    common = dict(
        seq_cols=data_best["seq_cols"],
        input_spec=data_best["input_spec"],
        target_col=data_best["target_col"],
        mode="sample",
    )

    if family == "lstm":
        inference_model: torch.nn.Module = InferenceMultinomialLSTMModel(
            **common,
            hidden_dim=params["hidden_dim"],
            memory_units=params["memory_units"],
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
