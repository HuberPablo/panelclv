"""Where the target sits on the feature axis, decided once.

`prepare_dataset` builds `(N, T, F)` tensors whose feature axis is `seq_cols`, and one
of those F channels is the count column the model classifies. Which one is not
re-negotiable downstream: the dataset records it as `target_idx`, and everything that
needs the counts reads that key through this module.

The rule is worth stating because breaking it is invisible. Every channel of the tensor
is a float array of the same shape, so indexing the wrong one yields plausible numbers
rather than an error: a forecast scored against a standardized covariate, or a rollout
fed a sine wave as its own count history. Six sites used to work the index out again
from `seq_cols` and `target_col`, and four of them spelled out the
`holdout[:, :, target_idx]` slice by hand.

A leaf: numpy only, so the model layer and the study layer can both read a dataset's
target without importing anything heavier than the dataset that produced it.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def target_index(seq_cols: Sequence[str], target_col: str) -> int:
    """Position of `target_col` on the feature axis — the package's one derivation.

    Used where there is no data dict to read from: the rollout simulators are handed a
    bare `(seq_cols, target_col)` pair because they step over tensors, not datasets.
    Everywhere a dataset is in hand, read its recorded `target_idx` instead — via
    :func:`calibration_counts` / :func:`holdout_actuals` — rather than calling this.
    """
    seq_cols = list(seq_cols)
    if target_col not in seq_cols:
        raise ValueError(
            f"target_col {target_col!r} not in seq_cols={seq_cols}"
        )
    return seq_cols.index(target_col)


def calibration_counts(data: dict[str, Any]) -> np.ndarray:
    """The calibration window's target channel as `(N, T_CAL)` float64.

    `data` is a `prepare_dataset` output (or the reduced-layout dict a tuned trial
    carries, whose `target_idx` describes its own narrower feature axis).
    """
    return _channel(data, "calibration")


def holdout_actuals(data: dict[str, Any]) -> np.ndarray:
    """The holdout window's target channel as `(N, T_HOLD)` float64 — the true counts.

    These are the actuals every forecast is scored against. They are read out of the
    holdout tensor for evaluation only; no rollout ever feeds them back to a model.
    """
    return _channel(data, "holdout")


def _channel(data: dict[str, Any], window: str) -> np.ndarray:
    """The target channel of `data[window]`, at the index the dataset recorded.

    Slices first and casts after, so only the one `(N, T)` channel is widened rather
    than the whole `(N, T, F)` tensor.
    """
    if "target_idx" not in data:
        raise KeyError(
            "data['target_idx'] is missing — it records which channel of the (N, T, F) "
            "tensors holds the counts, and nothing re-derives it. `prepare_dataset` and "
            "`tuning.select_features` both set it; a dict built by hand has to say it "
            "too, because no other key can be trusted to imply it."
        )
    tensor = np.asarray(data[window])                       # (N, T, F)
    return tensor[:, :, int(data["target_idx"])].astype(np.float64)
