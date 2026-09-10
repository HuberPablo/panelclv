# 03 — Ablation arms: do the compressed encodings beat the flags?

Status: ready-for-agent

Blocked by: 01

## Problem

Support is an input-side precondition, not a result. §4's own counter-example stands:
CDNOW's `ar_bounded_32` arm was nominally bounded and still lost, because only 3.5% of
calibration cells sat beyond its deepest bin against 68.9% of holdout cells. A short
distance outside the fitted range is not evidence that a trained model forecasts well
from it.

## What to build

Arms in `scripts/run_ar_encoding_ablation.py`, alongside the four that exist:

    ar_log        log_period_since_last_transaction
                  log_period_since_first_transaction
                  cumulative_transactions
    ar_saturating saturating_recency_<C>, saturating_tenure_<C>, transaction_rate
    ar_ratio      recency_over_tenure, transaction_rate, saturating_tenure_<C>,
                  has_transacted_before          # the bounded Pareto/NBD triple

C is chosen per panel the way `PANEL_DEPTHS` already chooses bin depth — it is a
half-saturation constant, not a bin edge, so the degeneracy `check_arm_depth` guards
does not apply, but the same "does calibration have mass where the holdout will sit"
judgement does.

`ar_log` deliberately keeps `cumulative_transactions` rather than swapping in
`transaction_rate`: it is the arm that tests the coordinate change alone, against
`ar_unbounded`, with the feature *set* held fixed.

## Acceptance

Both must hold, on both panels:

- `|bias|` at or below the no-AR baseline — 22.4% electronics, 14.6% CDNOW.
- Per-customer Spearman at or above the winning bounded flag arm — 0.267 electronics,
  0.438 CDNOW.

The second is what rejects an arm that fixes the level by going blind, which is the
failure this whole family is meant to avoid. Report the distribution across
replications, not a pooled number.

## Note on cost

Three arms x two panels x 40 replications is the same shape as the existing run, so the
sharding and machine-selection rules in `VastAI/Rules.md` apply unchanged. Run a
`--n-studies 1 --n-trials 5` probe on a fresh box before committing a shard to it.
