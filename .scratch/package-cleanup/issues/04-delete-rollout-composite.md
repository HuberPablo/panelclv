# 04 — Delete rollout-composite selection

**What to build:** trials are selected on validation loss, full stop. The package computes
RMSE, bias and MAPE in exactly one place, which makes `CLAUDE.md`'s "single scoring
authority" claim true for the first time rather than aspirational.

**Blocked by:** 02

**Status:** done

Source: `.scratch/package-simplification/issues/06-target-architecture.md` (decision 1),
`08-reconcile-adrs-and-vocabulary.md`, `09-module-naming.md` (decisions 5, 6)

## Why it goes

The composite was never reachable from the production path — the suite runner passes no
selection metric and the suite config has no field for one. It was reachable from two
notebooks, and one stored study shows it really ran. But **all 1256 archived
selection-metric values read validation loss; zero composite.** Nothing stored needs
re-tuning, which is what makes the deletion free.

It was also not measuring what its ADR claimed. The rollout metric recomputation computes
RMSE over customer-summed totals where the authority uses per-cell values — **62x apart** —
and MAPE under a different estimator. Only bias agreed. So "aligning selection with the
metric actually reported" held for one metric of three.

**Accepted cost, recorded once:** selection on rollout quality is gone, so a model that
takes good next steps but drifts over a long horizon is unguarded again. Restoring it would
be a re-implementation, not a flag flip.

## Scope

The rollout metric recomputation, the validation rollout score, the rollout-metric
constant, the eleven rollout parameters on the objective and the study runner, and
`selection_metric` itself — with the composite gone it is a parameter with one legal value.
Two notebooks' tuning cells are edited in the same commit.

## Renames folded in

- `mape_aggregate_style` → `mape_aggregate`. "Style" is filler, and `CLAUDE.md` **already**
  calls the key `mape_aggregate` — the code is what drifted. One notebook site.
- The metrics-computation alias is deleted; the authority keeps the name `CLAUDE.md` uses.

## Docs

- **ADR-0003 is retired in place** with a `Retired` header — **not** `Superseded by`,
  since nothing replaces it and "superseded" sends a reader hunting for a successor. The
  header states the code still runs until this issue lands. A "Why it goes" section records
  the three facts worth keeping: never reachable from production, all 1256 archived values
  read validation loss, and the one-of-three metric alignment.
- **ADR-0001 Edit A** — its Consequences paragraph names two selection metrics; one
  remains.
- **The evaluation subpackage docstring rewording** — the authority claim is currently
  scoped narrowly because package-wide it was false. It is now true package-wide.

Full finished text for all three is in ticket 08 of the map. **Copy it; do not re-derive
it.**

- [x] Composite selection and `selection_metric` gone; the two notebook tuning cells updated
- [x] Exactly one implementation of RMSE, bias and MAPE remains in the package
- [x] `mape_aggregate` rename applied, including its one notebook site
- [x] ADR-0003 retired in place with its Why-it-goes section
      (its header said the code "still runs until those issues land"; that clause is
      now false, so the tense was updated — the only re-derivation of ticket 08's text)
- [x] ADR-0001 Edit A and the evaluation docstring rewording applied verbatim from ticket 08
      (Edit C's quoted key spelled `mape_aggregate`, since the rename below lands in the
      same commit)
- [x] Retired names appended to the import test's pattern
- [x] Golden test green at rel=1e-6; notebook API test green
