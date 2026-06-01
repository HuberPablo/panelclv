"""Tests for the autoregressive, target-derived features.

Covers the three features added on top of the original recency/activity set --
``cumulative_transactions`` (RFM/Pareto-NBD frequency x), ``cumulative_count``
(sum of counts), ``period_since_first_transaction`` (BTYD observation age T) and
``transaction_rate`` (empirical purchase rate lambda) -- with three kinds of
checks:

1. hand-computed values on a tiny series (the literature semantics are correct);
2. validation/parsing of the new names;
3. the consistency property that *makes the features leak-free*: stepping the
   incremental `ARFeatureState` over a holdout reproduces, column for column,
   what the vectorized `compute_ar_feature_columns` precompute would have
   produced on the full series. This is the invariant the Monte-Carlo rollout
   relies on, so it is exercised on randomized panels for every feature.

Run:  pytest -q tests/test_ar_features.py
"""

import numpy as np
import pytest

from panelclv.data_preparation.ar_features import (
    ARFeatureState,
    compute_ar_feature_columns,
    parse_ar_feature,
    validate_ar_features,
)

NEW_FEATURES = [
    "cumulative_transactions",
    "cumulative_count",
    "period_since_first_transaction",
    "transaction_rate",
]
ALL_FEATURES = [
    "period_since_last_transaction",
    "has_transacted_before",
    "active_in_last_3_periods",
    *NEW_FEATURES,
]


# --------------------------------------------------------------------------- #
# 1. Hand-computed semantics on a tiny, fully spelled-out series.
# --------------------------------------------------------------------------- #
# One customer, counts per period:        0   2   0   0   1   0
# active period (count > 0):              F   T   F   F   T   F
# cum_txn  (active periods so far):       0   1   1   1   2   2
# cum_cnt  (sum of counts so far):        0   2   2   2   3   3
# tenure   (periods since first txn):     0   0   1   2   3   4
# rate = cum_txn / max(tenure, 1):        0   1  1/1 1/2 2/3 2/4
TINY = np.array([[0, 2, 0, 0, 1, 0]], dtype=np.int64)


def test_cumulative_transactions_counts_active_periods():
    out = compute_ar_feature_columns(TINY, ["cumulative_transactions"])
    expected = np.array([[0, 1, 1, 1, 2, 2]], dtype=np.float32)
    np.testing.assert_array_equal(out["cumulative_transactions"], expected)


def test_cumulative_count_sums_the_raw_counts():
    out = compute_ar_feature_columns(TINY, ["cumulative_count"])
    expected = np.array([[0, 2, 2, 2, 3, 3]], dtype=np.float32)
    np.testing.assert_array_equal(out["cumulative_count"], expected)


def test_period_since_first_transaction_is_tenure():
    out = compute_ar_feature_columns(TINY, ["period_since_first_transaction"])
    expected = np.array([[0, 0, 1, 2, 3, 4]], dtype=np.float32)
    np.testing.assert_array_equal(out["period_since_first_transaction"], expected)


def test_transaction_rate_matches_cum_txn_over_tenure():
    out = compute_ar_feature_columns(TINY, ["transaction_rate"])
    expected = np.array([[0, 1, 1, 1 / 2, 2 / 3, 2 / 4]], dtype=np.float32)
    np.testing.assert_allclose(out["transaction_rate"], expected, rtol=1e-6)


def test_features_are_zero_for_a_customer_that_never_transacts():
    """Before any purchase every count/tenure/rate feature stays at 0."""
    silent = np.zeros((1, 5), dtype=np.int64)
    out = compute_ar_feature_columns(silent, NEW_FEATURES)
    for name in NEW_FEATURES:
        np.testing.assert_array_equal(out[name], np.zeros((1, 5), dtype=np.float32))


def test_count_greater_than_one_separates_txn_count_from_event_count():
    """A period with count 3 advances cum_count by 3 but cum_transactions by 1."""
    multi = np.array([[3, 0, 0]], dtype=np.int64)
    out = compute_ar_feature_columns(multi, ["cumulative_transactions", "cumulative_count"])
    np.testing.assert_array_equal(out["cumulative_transactions"], [[1, 1, 1]])
    np.testing.assert_array_equal(out["cumulative_count"], [[3, 3, 3]])


# --------------------------------------------------------------------------- #
# 2. Parsing / validation of the new names.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name,kind",
    [
        ("cumulative_transactions", "cum_txn"),
        ("cumulative_count", "cum_cnt"),
        ("period_since_first_transaction", "tenure"),
        ("transaction_rate", "rate"),
    ],
)
def test_parse_new_features(name, kind):
    assert parse_ar_feature(name) == (kind, None)


def test_validate_accepts_the_full_feature_set():
    validate_ar_features(ALL_FEATURES)  # must not raise


def test_unknown_feature_still_raises():
    with pytest.raises(ValueError):
        parse_ar_feature("cumulative_spend")


# --------------------------------------------------------------------------- #
# 3. The leak-free invariant: incremental rollout == vectorized precompute.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2, 7])
def test_rollout_state_matches_precompute(seed):
    """Stepping ARFeatureState over a holdout reproduces the precompute exactly.

    Build a full (N, T) panel, split it into a calibration warm-up and a
    holdout, then feed the holdout counts one period at a time through
    ARFeatureState. Each step's output must equal the matching holdout column of
    `compute_ar_feature_columns` run on the *full* series -- i.e. the rollout
    never needs the future, only the (sampled) targets it is fed.
    """
    rng = np.random.default_rng(seed)
    n, t_cal, t_hold = 6, 9, 7
    # Counts in 0..3 with plenty of zeros, so silent stretches and multi-count
    # periods both occur; force one customer to stay completely silent.
    full = rng.integers(0, 4, size=(n, t_cal + t_hold)).astype(np.int64)
    full[full == 1] = 0
    full[-1, :] = 0

    full_cols = compute_ar_feature_columns(full, ALL_FEATURES)

    state = ARFeatureState(full[:, :t_cal], ALL_FEATURES)
    for h in range(t_hold):
        step = state.update(full[:, t_cal + h])
        for name in ALL_FEATURES:
            np.testing.assert_allclose(
                step[name],
                full_cols[name][:, t_cal + h],
                rtol=1e-6,
                err_msg=f"{name} mismatch at holdout step {h}",
            )


def test_rollout_handles_first_transaction_inside_the_holdout():
    """Tenure/rate are correct when the first-ever purchase happens mid-rollout.

    The customer is silent through calibration and the first holdout step, then
    buys -- tenure must reset to 0 at that purchase and count up afterward,
    matching the precompute (the first-index convention is the subtle case).
    """
    full = np.array([[0, 0, 0, 0, 2, 0, 1]], dtype=np.int64)  # first txn at index 4
    t_cal = 3
    full_cols = compute_ar_feature_columns(full, NEW_FEATURES)

    state = ARFeatureState(full[:, :t_cal], NEW_FEATURES)
    for h in range(full.shape[1] - t_cal):
        step = state.update(full[:, t_cal + h])
        for name in NEW_FEATURES:
            np.testing.assert_allclose(step[name], full_cols[name][:, t_cal + h], rtol=1e-6)
