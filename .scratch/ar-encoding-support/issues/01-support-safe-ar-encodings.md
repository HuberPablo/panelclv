# 01 — Five support-safe encodings of the two window-capped clocks

Status: done

## Problem

`period_since_last_transaction` and `period_since_first_transaction` cannot exceed the
calibration window length while being fitted and keep counting through the holdout. The
only fix in the package is the nested `active_in_last_<K>_periods` flags, which work by
collapsing the whole tail onto one value — and therefore discard every distinction past
the deepest bin.

## What to build

Names in `configs/ar_feature_names.py`, branches in `data_preparation/ar_features.py`:

    log_period_since_last_transaction    log(1 + since)
    log_period_since_first_transaction   log(1 + tenure)
    saturating_recency_<C>_periods       since / (since + C)
    saturating_tenure_<C>_periods        tenure / (tenure + C)
    recency_over_tenure                  (T - t_x) / T, 0 before the first purchase

All five are pure functions of the states `_base_states` already keeps, so
`ARFeatureState` gains no fields and the precompute/rollout equality is unchanged.

## Comments

**2026-09-10 — done.**

Implemented as specified. `_WINDOWED_KINDS` replaces the single `_ACTIVE_RE` branch in
`parse_ar_feature`, so the three windowed families share one dispatch and a new one is a
row rather than a branch.

Tests in `tests/test_ar_features.py`: hand-computed values for all five, parse tests for
the two new windowed families, `C = 0` rejected, and the five added to `ALL_FEATURES` so
the existing randomised precompute-equals-rollout property covers them for free. Two
invariants pinned that the spec leans on — `recency_over_tenure` is exactly 1 for a
customer who bought once and then went silent, and the three bounded channels never
leave [0, 1] on any history. 34 tests pass.

One design note worth keeping: `recency_over_tenure` gates to 0 before the first
purchase, where tenure is 0 and `since` counts up from the start of the series, so the
ungated ratio would be unbounded — the exact failure the feature exists to avoid. The
gate collides with "transacted this period", which is why the docstring says to pair it
with `has_transacted_before`.
