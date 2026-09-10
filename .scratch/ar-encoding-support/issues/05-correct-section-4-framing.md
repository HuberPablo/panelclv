# 05 — §4 of the feature-engineering doc ranks encodings with a metric that cannot

Status: ready-for-agent

## Problem

Two statements in `docs/feature_engineering.md` §4 are now known to be incomplete.

1. The escape-fraction table is presented as the measure of "which ones to prefer". It
   is invariant to every monotone transform, so it can compare *features* and cannot
   compare *encodings of one feature*. Nothing in the section says so, and the natural
   reading — that a transform which lowers the escape fraction is what to look for —
   sends the reader after something no transform can deliver.
2. `cumulative_transactions` and `cumulative_count` are described as "unbounded in
   principle, but ... stays in range in practice (0.04%)". On CDNOW the same channel
   escapes on 0.176% of cells and by 9.19 z at the worst — further outside than either
   window-capped clock. True by count, false by distance.

## What to change

Add the distance column to the §4 table and state the invariance explicitly, so the two
ways an encoding can help (non-injective collapse, or shorter distance) are named. Point
at `scripts/measure_ar_support.py` as the way to regenerate it. Amend the cumulative
counter claim to carry both metrics.

Do not restate the spec in the doc — link `.scratch/ar-encoding-support/spec.md`.
