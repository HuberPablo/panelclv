# 02 — Rank encodings by distance outside the fitted range, not by escape fraction

Status: done

## Problem

`docs/feature_engineering.md` §4 ranks AR features by support-escape fraction. That
number **cannot rank two encodings of the same counter**: it is invariant to every
order-preserving transform, so `log1p(recency)` scores identically to `recency`.

The quantity that separates them is distance, in the calibration z-units the model
actually sees, because Theorem 1 of Xu et al. (2021) makes the out-of-range error grow
linearly with it.

## What to build

`scripts/measure_ar_support.py` — a documented tool, no checkpoint, seconds to run.
Report per encoding per panel: escape %, worst-case z beyond the ceiling, mean excess
over the escaping cells, and the calibration range. A `--linearity` mode fits the
empirical log-hazard in both recency coordinates and extrapolates each past the ceiling
against what the holdout actually did there.

## Comments

**2026-09-10 — done, and it turned up something the escape fraction hides.**

`cumulative_transactions` on CDNOW escapes on 0.176% of holdout cells and by **9.19 z**
at the worst cell, 3.87 z on average — further out than either window-capped clock. §4
files it under "unbounded in principle, in range in practice", which is true by count and
false by distance. Hence the `z avg` column: a channel can escape rarely and
catastrophically, and one number cannot say so.

`transaction_rate` reports a negative worst-case z (-10.6 electronics, -5.7 CDNOW),
meaning its holdout peak sits well inside the calibration ceiling. That is what a
genuinely stationary channel looks like and the sign is worth keeping rather than
clamping at zero.

Characterisation tests added to `tests/test_ar_feature_support.py` §4: the invariance
itself is asserted (four raw/re-encoded pairs escape on identical fractions), each
re-encoding is asserted closer to the ceiling than its raw counter, and
`recency_over_tenure` is asserted never to leave the range. 23 tests pass.
