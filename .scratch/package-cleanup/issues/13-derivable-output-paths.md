# 13 — Output paths derivable from config and seed

**What to build:** the same config and the same seed produce the same output paths. Finding
a previous run's output does not require knowing what time it was started.

**Blocked by:** 02

**Status:** ready-for-agent

Source: `.scratch/package-simplification/issues/05-reachability-ledger.md` (D19),
`06-target-architecture.md` (decision 8)

## Scope — three sites, not six

Three live sites put wall-clock time into a **folder name**, so an output path is not
derivable from config and seed. That is a direct hit on the project's second priority:
reproducibility means the same config and seed give the same result, and a path is part of
the result.

Derive those three names from config and seed. **Keep the timestamp as a metadata field** —
provenance is worth having, it just should not be load-bearing for lookup.

**The other three sites of the same pattern are deliberately left alone:** the suite runner
writes a creation-time metadata field, which is correct provenance; the tuner already opts
out of appending a timestamp; and the third dies with the module issue 03 deletes. Do not
sweep them by pattern-match.

- [ ] The three folder names derive from config and seed
- [ ] Timestamp preserved as metadata at each of those sites
- [ ] The three deliberately-excluded sites untouched
- [ ] Running the same config and seed twice writes to the same path
- [ ] Golden test green
