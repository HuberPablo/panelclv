# 12 — Derive the target column and the time flags once

**What to build:** the target column is produced in one place and read everywhere else, and
the time-flag set is written once. The bottom of the import stack stops naming something
above it.

**Blocked by:** 02

**Status:** ready-for-agent

Source: `.scratch/package-simplification/issues/05-reachability-ledger.md` (D13, D16, D17,
D1), `06-target-architecture.md` (decision 10, Q15)

## The target column

Produced once by the dataset preparation step, then **re-derived six more times**. A drift
between any two of those scores the wrong column — silently, because the shapes all still
match and the numbers all still look like counts.

## The time flags

Written **four times, and already drifted** — which is what produced an orphan day-of-year
column nobody reads. This is the half of the time-flag finding that ticket 06 left open when
it closed the customer-group half (issue 09 owns that one).

## The subpackage cycle

`configs` imports a validator from `data_preparation`, and `data_preparation` imports the
panel config back. Both at module level, neither deferred. It does not break today because
the cycle is at **subpackage granularity only** — the module graph underneath is acyclic,
which is why no audit caught it. But the panel config is meant to be the bottom of the stack
and it imports upward.

Fixed here rather than given its own issue: it is one upward import, and moving the validator
(or the name list it checks) settles it. Issue 08's acyclicity test will hold this closed.

## Folded in opportunistically

- The id-column two-fallback-string problem, ~9 sites, with the suite runner using both
  spellings in one function.
- The twice-defined data-builder alias — defined in two different subpackages, a duplication
  neither lane audit found.
- The root package docstring, which lists 8 of the 9 subpackages.

- [ ] Target column produced once; the six re-derivations read it
- [ ] Time-flag set written once; the orphan column gone
- [ ] `configs` no longer imports `data_preparation`; acyclicity test passes
- [ ] Id-column fallback resolved to one spelling
- [ ] Data-builder alias defined once
- [ ] Root docstring lists every subpackage
- [ ] Golden test green at rel=1e-6 — its feature-axis assertions are the net here
