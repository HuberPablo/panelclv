# 10 — Split the Pareto grid module at the real-panel boundary

**What to build:** whether a function can run on a real panel is answerable by reading an
import line. The measurements behind the thesis's two grid figures exist once, not twice.

**Blocked by:** 02

**Status:** done

Source: `.scratch/package-simplification/issues/12-thesis-figure-code-home.md` (decisions 1,
2, 3, 6), `09-module-naming.md` (decision 3)

## Correction this issue is built on

**The figure script does not use the three grid functions.** It imports exactly one name
from the grid module and **re-implements** its own death-mass and shape-correlation
measurements. So those three functions are called from one notebook only — genuinely
*called*, so alive under the import-is-not-a-caller rule.

This inverts the obvious reading twice over: relocating them next to the script would put
them beside code that ignores them, and deleting them as script-superseded would remove the
notebook's only path to those tables.

## The boundary test

**"Can it run on a real panel?"** — a property of the code, readable from it. Not "did it
produce a thesis figure", which fails on the evidence: the suite forecast plot produced
thesis figures and is called 16 times from notebooks, and `scripts/` is not importable, so
relocating there makes code unreachable from the notebooks that *are* the thesis's analysis
surface.

The three grid functions read the **generator's ground truth** — the true death week and the
true seasonal multiplier — so by construction they can never run on a real panel. **Nothing
else in either module fails the test, so nothing moves to `scripts/` or a `thesis/`
directory.**

## The split

- The stored-results half keeps the grid collection, group summary, model comparison and the
  two plot functions — all read stored results and run on any grid suite. **This module also
  absorbs its rename** onto the unified Pareto/NBD spelling, so it drops out of issue 15's
  sweep.
- A **synthetic-grid module** takes the three ground-truth-reading functions and their
  helpers.

A boundary between two files is enforceable by reading an import line; a docstring marker
inside one file is not.

## The duplication collapse

Two implementations measure the same two things on the synthetic grid — the death-state
failure and the seasonal-tracking strength. **The script's arithmetic wins**, because its
output is what is in the thesis; the package functions become that arithmetic exposed for the
notebook, and the script becomes a thin caller.

Folded in: the three package functions share **51 of 55 byte-identical lines** with each
other plus a third copy of the same traversal scaffold. Factored once as part of the same
work.

## Promotion

The seasonal multiplier helper becomes **public**. The synthetic-grid module exists precisely
to reconstruct the pattern a stored study was generated with, so reaching across a subpackage
boundary for a private name is the wrong shape.

- [x] Two modules, split on the real-panel test; the stored-results half carries its new name
- [x] One implementation of the death-mass and shape-correlation measurements, script's arithmetic
- [x] The script is a thin caller; the traversal scaffold exists once
- [x] Seasonal multiplier helper public
- [x] The notebook's three grid calls still work
- [x] Golden test green; notebook API test green

## Comments

Landed 2026-08-13. Full suite green (225 passed), including the golden end-to-end test and
the notebook API test.

### The two modules

`studies/pnbd_grid.py` (630 lines) is now `studies/pareto_nbd_grid.py` (302) and
`studies/synthetic_grid.py` (533). The rename landed here, as this issue said it would, so
issue 15's file-rename list is down to four.

The split is enforced by an import line in both directions: `synthetic_grid` names
`pareto_nbd_grid` (for the grid axes and the two path conventions) and never the reverse,
and a test parses the stored-results module's imports to assert it.

### The boundary, stated as it actually holds

The ticket's test — "reads the generator's ground truth" — separates the three `*_grid`
tables cleanly, but it is *not* what pins the two new measurements to a generated grid:
`dead_customer_mass` and `shape_correlation` read only the panel and the stored forecast.
What makes them synthetic-only is the generation study's own layout — they are indexed by
`list_pnbd_datasets`, i.e. by a manifest only the generator writes. Both module docstrings
say that rather than the stronger claim, which two reviewers independently read as false.

### The duplication collapse, checked against the published output

`dead_customer_mass` and `shape_correlation` are now package functions in
`synthetic_grid`, and `make_grid_figures.py` calls them; its two local copies are gone
along with the 95 lines that held them.

The ticket says the script's arithmetic wins *because its output is what is in the
thesis*, so the arithmetic was moved but its traversal was not: the holdout window is
derived from the forecast horizon and the panel length where the script hardcoded
`HOLDOUT_YEAR = 2001`, and the datasets come from the manifest where the script hardcoded
`Dataset_1..10`. **Both coincide on the thesis grid, and that was verified rather than
assumed**: run against the real 4x4x10 study, the package functions reproduce every one of
the 160 x 3 numbers in `figures/dead_customer_mass.csv` and `figures/shape_correlation.csv`
to a maximum absolute difference of 1.1e-16. The three ground-truth grids were checked the
same way against the pre-split module: byte-identical tables.

One artefact change: the script now writes those two CSVs in the package's long layout
(one row per dataset and model, carrying the `combo` / `dataset` labels) rather than the
old wide one. Same numbers, plus the replicate labels the wide layout dropped.

### The traversal

The three grids shared 51 of 55 byte-identical lines. There is now one `_measure_grid`
walk over every trained (dataset, model) pair, parameterised by a measurement function, and
all five tables go through it. `alive_volume_ratio_grid` and `dead_volume_leakage_grid` are
two lines each over a shared `_oracle_split`.

Two things the rewrite picked up from the rest of the package rather than re-spelling:
the forecast is read back through `predictions.load_predictions_from_csv` (the charter's
one reader of that layout) at the path `layout.prediction_path` names, and the panel's
column roles are read from the schema the generator records with every dataset instead of
being hardcoded — which is what `PanelConfig` does for real panels.

### Promotion

`pareto_simulation.seasonal_weekly_multiplier` is public, with a docstring line saying why:
reconstructing the pattern a stored study was generated with is a supported operation, and
the reference curve has to be the *same* function that made the data.

### Two review findings deliberately not acted on

- **`(study_dir, train_base)` as a data clump.** A grid-study type would fit, but the pair
  is the existing signature of every function on this surface, including the four this
  issue did not touch. Splitting the convention across two shapes costs more than it saves.
- **Exporting the two new names from `panelclv.studies`.** Read as scope creep (no notebook
  calls them yet), but the subpackage's own docstring is that every entry point is exported
  so a caller imports the subpackage, never a module — and "exposed for the notebook" is the
  ticket's phrase. The script still imports by module path, which is what declares the
  boundary at the call site.
