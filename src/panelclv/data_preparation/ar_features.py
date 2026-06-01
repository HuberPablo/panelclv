"""Autoregressive, target-derived features (recency / activity / frequency).

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
    cumulative_transactions         running count of *active periods* (periods
                                    with target > 0) up to and including now --
                                    the "frequency" (x) of the RFM / Pareto-NBD
                                    / BG-NBD literature.
    cumulative_count                running sum of the target counts themselves
                                    up to and including now (>= cumulative_
                                    transactions; differs when a period holds
                                    more than one transaction).
    period_since_first_transaction  customer tenure: periods elapsed since the
                                    first transaction (0 at the first one, then
                                    counts up; 0 before any has occurred). The
                                    effective observation age "T" of the BTYD
                                    models, distinct from recency (gap since the
                                    *last* transaction).
    transaction_rate                cumulative_transactions / max(tenure, 1) --
                                    an empirical estimate of the per-period
                                    purchase rate (the Poisson rate lambda that
                                    Pareto/NBD shrinks toward); bounded and
                                    stationary where the two counters are not.

All features derive from a small per-customer running state:
    since     periods since the last transaction (0 this period)
    ever      whether any transaction has occurred yet
    cum_txn   number of active periods so far
    cum_cnt   sum of the target counts so far
    tenure    periods since the first transaction (0 before/at the first)
so:
    period_since_last_transaction  = since
    has_transacted_before          = int(ever)
    active_in_last_K_periods       = int(ever and since < K)
    cumulative_transactions        = cum_txn
    cumulative_count               = cum_cnt
    period_since_first_transaction = tenure
    transaction_rate               = cum_txn / max(tenure, 1)

Counts are treated as non-negative integers (the multinomial class indices), so
the cumulative state is integer-valued and the two compute paths agree exactly.

Standard library + numpy only.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

_RECENCY = "period_since_last_transaction"
_HAS = "has_transacted_before"
_CUM_TXN = "cumulative_transactions"
_CUM_CNT = "cumulative_count"
_TENURE = "period_since_first_transaction"
_RATE = "transaction_rate"
_ACTIVE_RE = re.compile(r"^active_in_last_(\d+)_periods$")


def parse_ar_feature(name: str) -> tuple[str, int | None]:
    """Validate one AR-feature name → (kind, K or None).

    kind is one of: 'recency', 'has', 'active', 'cum_txn', 'cum_cnt',
    'tenure', 'rate'. Only 'active' carries a window K; the rest return None.
    """
    if name == _RECENCY:
        return ("recency", None)
    if name == _HAS:
        return ("has", None)
    if name == _CUM_TXN:
        return ("cum_txn", None)
    if name == _CUM_CNT:
        return ("cum_cnt", None)
    if name == _TENURE:
        return ("tenure", None)
    if name == _RATE:
        return ("rate", None)
    m = _ACTIVE_RE.match(name)
    if m:
        k = int(m.group(1))
        if k < 1:
            raise ValueError(f"active window must be >= 1 in {name!r}")
        return ("active", k)
    raise ValueError(
        f"unknown ar feature {name!r}; supported: {_RECENCY!r}, {_HAS!r}, "
        f"{_CUM_TXN!r}, {_CUM_CNT!r}, {_TENURE!r}, {_RATE!r}, "
        "'active_in_last_<K>_periods'"
    )


def validate_ar_features(names: Sequence[str]) -> None:
    """Raise ValueError if any name is not a supported AR feature."""
    for n in names:
        parse_ar_feature(n)


def _base_states(target_2d: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorized running states over a (N, T) target array.

    Returns ``{since, ever, cum_txn, cum_cnt, tenure}``, each (N, T) int64,
    such that ``[:, t]`` is the state *after observing* period ``t``. These are
    exactly the quantities `ARFeatureState` maintains incrementally, so the
    precompute and the rollout produce identical feature columns.
    """
    y = np.asarray(target_2d)
    if y.ndim != 2:
        raise ValueError(f"target_2d must be 2-D (N, T), got shape {y.shape}")
    txn = y > 0
    n, t = y.shape
    pos = np.broadcast_to(np.arange(t), (n, t))

    # since[i, t] = t - (last index <= t with target > 0), or t + 1 if none.
    last_true = np.maximum.accumulate(np.where(txn, pos, -1), axis=1)
    since = pos - last_true
    # ever[i, t] = 1 once a transaction has occurred at any position <= t.
    ever = np.maximum.accumulate(txn.astype(np.int64), axis=1)

    # Cumulative frequency (active periods) and total count (sum of targets).
    cum_txn = np.cumsum(txn.astype(np.int64), axis=1)
    cum_cnt = np.cumsum(y.astype(np.int64), axis=1)

    # tenure[i, t] = t - first_txn_index, gated to 0 before the first txn. For
    # rows that never transact, `ever` masks the (meaningless) argmax to 0.
    first_idx = np.argmax(txn, axis=1)                    # 0 if the row is all-zero
    tenure = np.where(ever == 1, pos - first_idx[:, None], 0)

    return {
        "since": since,
        "ever": ever,
        "cum_txn": cum_txn,
        "cum_cnt": cum_cnt,
        "tenure": tenure,
    }


def _render(name: str, states: dict[str, np.ndarray]) -> np.ndarray:
    """Map one feature name onto a float32 array from the running `states`.

    Works elementwise, so the same code serves the 2-D precompute and the 1-D
    per-step rollout update.
    """
    kind, k = parse_ar_feature(name)
    if kind == "recency":
        return states["since"].astype(np.float32)
    if kind == "has":
        return states["ever"].astype(np.float32)
    if kind == "active":
        return ((states["ever"] == 1) & (states["since"] < k)).astype(np.float32)
    if kind == "cum_txn":
        return states["cum_txn"].astype(np.float32)
    if kind == "cum_cnt":
        return states["cum_cnt"].astype(np.float32)
    if kind == "tenure":
        return states["tenure"].astype(np.float32)
    # rate: per-period purchase intensity; max(tenure, 1) guards the first
    # period (tenure == 0) from a zero divide without distorting later steps.
    denom = np.maximum(states["tenure"], 1)
    return (states["cum_txn"] / denom).astype(np.float32)


def compute_ar_feature_columns(
    target_2d: np.ndarray, names: Sequence[str]
) -> dict[str, np.ndarray]:
    """Return ``{name: (N, T) float32 array}`` for the requested AR features."""
    validate_ar_features(names)
    states = _base_states(target_2d)
    return {name: _render(name, states) for name in names}


class ARFeatureState:
    """Incremental per-customer running state for the holdout rollout.

    Initialize from the calibration target history (so the state matches the
    end of the warm-up), then call `.update(y)` once per holdout step *before*
    building that step's input. The returned ``{name: (N,) float32}`` is
    identical to the corresponding column from `compute_ar_feature_columns`.
    """

    def __init__(self, calibration_target_2d: np.ndarray, names: Sequence[str]):
        validate_ar_features(names)
        self.names = list(names)
        states = _base_states(calibration_target_2d)
        # Carry only the final-period state into the rollout (each as a (N,) copy
        # so later in-place updates never alias the precompute arrays).
        self.since = states["since"][:, -1].astype(np.int64).copy()
        self.ever = states["ever"][:, -1].astype(np.int64).copy()
        self.cum_txn = states["cum_txn"][:, -1].astype(np.int64).copy()
        self.cum_cnt = states["cum_cnt"][:, -1].astype(np.int64).copy()
        self.tenure = states["tenure"][:, -1].astype(np.int64).copy()

    def update(self, y: np.ndarray) -> dict[str, np.ndarray]:
        """Advance one period with sampled target `y` (N,); return feature dict."""
        y = np.asarray(y)
        txn = y > 0
        ever_prev = self.ever                                  # tenure uses pre-update ever
        self.since = np.where(txn, 0, self.since + 1)
        # tenure increments once already-active; stays 0 through (and at) the
        # first transaction, matching `_base_states`' first-index convention.
        self.tenure = np.where(ever_prev == 1, self.tenure + 1, 0)
        self.ever = np.maximum(self.ever, txn.astype(np.int64))
        self.cum_txn = self.cum_txn + txn.astype(np.int64)
        self.cum_cnt = self.cum_cnt + y.astype(np.int64)
        states = {
            "since": self.since,
            "ever": self.ever,
            "cum_txn": self.cum_txn,
            "cum_cnt": self.cum_cnt,
            "tenure": self.tenure,
        }
        return {name: _render(name, states) for name in self.names}
