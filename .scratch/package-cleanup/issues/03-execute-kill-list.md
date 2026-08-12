# 03 — Execute the ledger's kill list

**What to build:** every symbol the ledger ruled dead is gone, and a test asserts it stays
gone. Nothing in the package has zero callers while reading as live surface.

**Blocked by:** 02

**Status:** ready-for-agent

Source: `.scratch/package-simplification/issues/05-reachability-ledger.md`,
`06-target-architecture.md` (decision 2)

## Scope

**11 rows, ~187 lines** — not the ledger's original 13 rows / 282 lines. The loss-variant
cluster is **kept whole**, including its tests: that reverses two of the ledger's kills and
is a settled decision, not an oversight.

The largest single unit is the forecast-run module — no callers at all, and it carries a
fifth on-disk prediction layout with no writer. The dead rollout aliases and the
suite-distribution function go with it.

**Every deletion in this issue cites its ledger row** in the commit, so the reasoning does
not have to be re-derived. The ledger is at
`.scratch/package-simplification/ledger.csv` / `.md` — 163 rows covering all 10,139 lines.

## Two rules that bind this issue

- **A notebook import is not a caller.** Only a call keeps a symbol alive. Deleting an
  import-only symbol **requires stripping the import line from the notebook in the same
  commit** — the notebook API test resolves every import and will go red otherwise.
- **The thesis carve-out overrides.** Anything that produced a figure or a number in the
  thesis is alive regardless of callers. Check before deleting.

## Test

Retired-name assertions append to the existing retired-symbols pattern in the import test.
This is the mechanism every later kill issue reuses.

- [ ] All 11 ledger rows deleted, each citing its row
- [ ] The loss-variant cluster and its tests untouched
- [ ] Any notebook import lines for deleted symbols removed in the same commit
- [ ] Retired names asserted gone in the import test
- [ ] Golden test green; notebook API test green
