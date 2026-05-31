"""Autoregressive, target-derived features (recency / activity).

These features are deterministic, **causal** functions of the target's own
past (a "transaction" = target value > 0), so they leak nothing and can be
recomputed on the fly during the Monte-Carlo holdout rollout from the *sampled*
target history. The SAME recurrence is used for the training-time precompute
(`compute_ar_feature_columns`) and the rollout (`ARFeatureState`), which is
what keeps training and inference consistent.

Supported names:
    period_since_last_transaction   periods since the last transaction (0 if it
                                    happened this period; counts up from the
                                    start of the series if it never has).
    has_transacted_before           1 if the customer has transacted at any
                                    period up to and including now, else 0.
    active_in_last_<K>_periods      1 if a transaction occurred within the last
                                    K periods (inclusive), else 0. K >= 1.

All three derive from a 2-value running state per customer:
    since  -- periods since the last transaction (0 this period)
    ever   -- whether any transaction has occurred yet
so:
    period_since_last_transaction = since
    has_transacted_before         = int(ever)
    active_in_last_K_periods      = int(ever and since < K)

Standard library + numpy only.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

_RECENCY = "period_since_last_transaction"
_HAS = "has_transacted_before"
_ACTIVE_RE = re.compile(r"^active_in_last_(\d+)_periods$")


def parse_ar_feature(name: str) -> tuple[str, int | None]:
    """Validate one AR-feature name → ('recency'|'has'|'active', K or None)."""
    if name == _RECENCY:
        return ("recency", None)
    if name == _HAS:
        return ("has", None)
    m = _ACTIVE_RE.match(name)
    if m:
        k = int(m.group(1))
        if k < 1:
            raise ValueError(f"active window must be >= 1 in {name!r}")
        return ("active", k)
    raise ValueError(
        f"unknown ar feature {name!r}; supported: {_RECENCY!r}, {_HAS!r}, "
        "'active_in_last_<K>_periods'"
    )


def validate_ar_features(names: Sequence[str]) -> None:
    """Raise ValueError if any name is not a supported AR feature."""
    for n in names:
        parse_ar_feature(n)


def _since_ever(target_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized (since, ever) over a (N, T) target array.

    since[i, t] = t - (last index <= t with target > 0), or t + 1 if none.
    ever[i, t]  = 1 if any target > 0 at positions <= t, else 0.
    """
    y = np.asarray(target_2d)
    if y.ndim != 2:
        raise ValueError(f"target_2d must be 2-D (N, T), got shape {y.shape}")
    txn = y > 0
    n, t = y.shape
    pos = np.broadcast_to(np.arange(t), (n, t))
    last_true = np.maximum.accumulate(np.where(txn, pos, -1), axis=1)
    since = pos - last_true                               # 0 at txn; t+1 if never
    ever = np.maximum.accumulate(txn.astype(np.int64), axis=1)
    return since, ever


def _render(name: str, since: np.ndarray, ever: np.ndarray) -> np.ndarray:
    kind, k = parse_ar_feature(name)
    if kind == "recency":
        return since.astype(np.float32)
    if kind == "has":
        return ever.astype(np.float32)
    return ((ever == 1) & (since < k)).astype(np.float32)   # active


def compute_ar_feature_columns(
    target_2d: np.ndarray, names: Sequence[str]
) -> dict[str, np.ndarray]:
    """Return ``{name: (N, T) float32 array}`` for the requested AR features."""
    validate_ar_features(names)
    since, ever = _since_ever(target_2d)
    return {name: _render(name, since, ever) for name in names}


class ARFeatureState:
    """Incremental per-customer (since, ever) for the holdout rollout.

    Initialize from the calibration target history (so the state matches the
    end of the warm-up), then call `.update(y)` once per holdout step *before*
    building that step's input. The returned ``{name: (N,) float32}`` is
    identical to the corresponding column from `compute_ar_feature_columns`.
    """

    def __init__(self, calibration_target_2d: np.ndarray, names: Sequence[str]):
        validate_ar_features(names)
        self.names = list(names)
        since, ever = _since_ever(calibration_target_2d)
        self.since = since[:, -1].astype(np.int64).copy()    # (N,)
        self.ever = ever[:, -1].astype(np.int64).copy()      # (N,)

    def update(self, y: np.ndarray) -> dict[str, np.ndarray]:
        """Advance one period with sampled target `y` (N,); return feature dict."""
        txn = np.asarray(y) > 0
        self.since = np.where(txn, 0, self.since + 1)
        self.ever = np.maximum(self.ever, txn.astype(np.int64))
        out: dict[str, np.ndarray] = {}
        for name in self.names:
            kind, k = parse_ar_feature(name)
            if kind == "recency":
                out[name] = self.since.astype(np.float32)
            elif kind == "has":
                out[name] = self.ever.astype(np.float32)
            else:
                out[name] = ((self.ever == 1) & (self.since < k)).astype(np.float32)
        return out
