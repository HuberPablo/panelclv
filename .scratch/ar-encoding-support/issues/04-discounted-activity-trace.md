# 04 — Open question: is an EWMA activity trace worth a float running state?

Status: needs-triage

## The candidate

    a_t = rho * a_{t-1} + (1 - rho) * 1[y_t > 0]

Bounded in [0, 1], stationary, one scalar per customer per rho. A bank at several rho
values is a multi-scale encoding of the same silence the `active_in_last_<K>` flags
bin, keeping the resolution the flags discard.

## Measured

Escapes on 0% of electronics holdout cells at rho in {0.7, 0.9, 0.97} and on 0.002% to
0.105% of CDNOW cells. So it clears the support bar.

## Why it was not implemented

`_base_states` keeps integer-valued running states, and its docstring says why: "Counts
are treated as non-negative integers ... so the cumulative state is integer-valued and
the two compute paths agree exactly." An EWMA is the first candidate that needs a float
entry. Written as the same loop in both paths the two would agree bit-for-bit, but the
guarantee stops being structural and becomes a property of matching the order of
operations, which is a real thing to give up for a feature whose advantage over the
existing flags is not yet demonstrated.

Its measured z values are also very large (10 to 17 in calibration), meaning a heavy
right skew — a scaling question `standardize_covariates` does not answer, since it
centres and scales without reshaping.

## Decide after

Issue 03. If the flag arms lose to the compressed arms on Spearman, resolution past the
deepest bin is worth something and this is the way to keep more of it. If they do not,
close as `wontfix`.
