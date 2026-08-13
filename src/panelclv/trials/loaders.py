"""Turning a ``prepare_dataset`` dict into the loaders a trial trains on.

This is where numpy becomes tensors: `data_preparation` is deliberately numpy-only,
so the crossing has to happen somewhere above it, and it happens here.

``split_calibration`` is the **sole enforcement point of ADR-0001**. Nothing else in
the package decides that training truncates before the validation window while
validation keeps the full sequence and scores from a later index — so if you want to
know what the temporal split does, this one function is the whole answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# `DataBuilder` is the closure signature `run_optuna_study` expects, so it is declared
# where that contract is — imported here rather than spelled out a second time.
from panelclv.tuning.optuna_tuning import DataBuilder, select_features


@dataclass(frozen=True)
class CalibrationSplit:
    """The two loaders ADR-0001's temporal split produces, plus the recipe behind them.

    ``recipe`` is not incidental bookkeeping: it is what every model constructor is
    rebuilt from, so it is a named field rather than a trailing dict. It carries
    ``seq_cols`` (feature-axis column names), ``embedded_cols`` (embedding
    cardinalities), ``target_col``, ``seq_len`` (the TRAIN sequence length, used by the
    Transformer's fixed-length mask cache; the longer val sequence simply rebuilds a
    mask on the fly, and the LSTM ignores it) and ``val_score_start`` (the AR index
    ``fit_model`` starts scoring cross-entropy from).
    """

    train_loader: DataLoader
    val_loader: DataLoader
    recipe: dict[str, Any]


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


def split_calibration(
    data: dict[str, Any],
    batch_size: int,
    shuffle_train: bool = True,
) -> CalibrationSplit:
    """Cut one ``prepare_dataset`` / ``select_features`` dict into train + val loaders.

    This function *is* ADR-0001. The split is **temporal** (a time window over ALL
    customers), not customer-wise: the calibration window is cut at
    ``data["val_start_idx"]`` (= ``s``, the first validation PERIOD index, set by
    ``prepare_dataset`` from ``validation_start``). Over the AR axis ``samples[t]``
    predicts calibration period ``t+1``, so:

      - **train** uses transitions whose target period is < ``s`` (the training prefix):
        ``X[:, :s-1]`` / ``y[:, :s-1]``. The model never consumes a validation period
        during training.
      - **val** uses the FULL sequence ``X`` / ``y`` so the recurrent/causal state warms
        up over the whole prefix; ``recipe["val_score_start"] = s-1`` then tells
        ``fit_model`` to score cross-entropy only on the validation suffix
        (periods ``s..T_CAL-1``).

    The two ``s-1``s below are one off-by-one seen twice, which is why they are written
    next to each other. ``seq_len = s-1`` is a *length*: the training slice covers
    transition indices ``0..s-2``. ``val_score_start = s-1`` is an *index*: the first
    transition scored on validation. They are equal because indices are 0-based, and the
    two ranges meet edge to edge — no transition is both trained on and scored, and none
    falls between them.

    Tensor contract (same for every model in this package):
      - ``samples`` : (N, T-1, F) float32 inputs (already float32 out of data prep).
      - ``targets`` : (N, T-1) int64 class indices -- ``squeeze(-1)`` drops the trailing
                      singleton feature axis; the values index the softmax head.
    """
    s = _require_val_start_idx(data)

    X = data["samples"]                                 # (N, T-1, F) float32
    y = data["targets"].squeeze(-1).astype(np.int64)    # (N, T-1) class indices

    # Train on the prefix transitions only (targets at periods 1..s-1); validate on the
    # full sequence but score only the suffix (see recipe["val_score_start"]).
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

    recipe = {                                          # the recipe to build the matching model
        "seq_cols":        data["seq_cols"],
        "embedded_cols":   data["embedded_cols"],
        "target_col":      data["target_col"],
        "seq_len":         X_train.shape[1],            # train length s-1 (Transformer mask cache)
        "val_score_start": s - 1,                       # score CE on the validation suffix only
    }
    return CalibrationSplit(train_loader, val_loader, recipe)


def refit_loader(
    data: dict[str, Any],
    batch_size: int,
    shuffle_train: bool = True,
) -> DataLoader:
    """A single loader over the FULL calibration window, for the refit (ADR-0008).

    Unlike ``split_calibration`` (which truncates training to the pre-validation
    prefix), this yields every AR transition ``samples`` / ``targets`` (all T-1 steps),
    so the fine-tune in ``training.refit_full_calibration`` also learns from the
    validation-tail periods. ``batch_size`` is typically large (the paper's "big batch"
    final step). There is no val loader because the refit has no validation set and
    therefore no early stopping.
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
    ``select_features`` and returns the matching temporal loaders + recipe, flattened
    to the ``(train_loader, val_loader, metadata)`` tuple the tuner's contract asks
    for. The train/val split is the same time boundary for every trial (carried in
    ``data_full["val_start_idx"]``), so trials differ only by hyperparameters and
    feature set.
    """

    def data_builder(feature_config: Sequence[str], batch_size: int):
        data = select_features(data_full, feature_config)   # drop chosen cols -> smaller F
        split = split_calibration(data, batch_size)
        return split.train_loader, split.val_loader, split.recipe

    return data_builder
