# 09 — Split the suite analysis module, and collapse the customer-group set

**What to build:** the suite analysis surface is three modules named for what they do, and
the customer-group set is defined once by the predicates that define the groups. A plot's
band and a table's confidence interval are the same number because they are the same code.

**Blocked by:** 03

**Status:** done

Source: `.scratch/package-simplification/issues/12-thesis-figure-code-home.md` (decisions 4,
7, 8), `09-module-naming.md` (decision 12), `08-reconcile-adrs-and-vocabulary.md`

## The split

One 1195-line module becomes three: a **suite reader** (prediction loading, suite
aggregation, discovery and config helpers), **suite plots** (the suite forecast plot and its
helpers), and **suite metrics** (study metrics, cross-study comparison, group metrics table).

**The split is free at the call sites** — every one of these has live notebook callers and
all of them import the subpackage, not the module, so **zero notebook edits.**

**The justification is not size.** It is that a single 1195-line file is what let a hardcoded
group vocabulary and a *second copy of the Student-t interval* hide inside it, both found
only by audit.

## The customer-group set

The hardcoded group tuple is not "thesis segmentation vocabulary baked into package code",
as it was first described. It is **the third of five encodings of one set** — the predicates,
the assignment default, this tuple, and the defaults of both group-metrics functions, with
the catch-all re-derived at three call sites.

**Survivor: the predicates that define the groups.** Their keys *are* the set by
construction and cannot drift from it. The catch-all is derived once, in the same place.

## One Student-t interval

Implemented three times: the across-study band, the study-metrics computation — **both
inside this one file** — and the Pareto grid's own copy, whose confidence arithmetic is
algebraically identical under a different spelling. **One implementation, living in suite
metrics**; the band and the grid's helper both call it. That two of the three were in one
file is the strongest single argument for the split.

## Folded in

The three deferred imports in this module commented as keeping it torch-free were **already
inert** — the studies package pulls torch at package import, so they never saved anything,
and one of them defers pandas, which cannot affect torch at all. Remove them and the
module-docstring claim with them.

## Sizing note

If the whole set needs headroom, **this split is the designated thing to defer** — it is the
one decision justified by future legibility rather than by a wrong number or an unreachable
path. The group collapse and the single Student-t interval must land regardless.

- [x] Three modules, named for what they do; zero notebook edits required
- [x] Group set has one encoding — the predicates' keys — with the catch-all derived once
- [x] One Student-t implementation, called by the band and the grid helper
- [x] Torch-free deferrals and the docstring claim removed
- [x] Golden test green; notebook API test green

## Comments

Landed 2026-08-13. Full suite green (218 passed), including the golden end-to-end test and
the notebook API test.

### The three modules

`studies/analysis.py` (1119 lines by the time this ran — issue 03 had already cut 78) is
now `suite_reader.py` (386), `suite_plots.py` (369) and `suite_metrics.py` (424), split as
decision 7 ruled. **Zero notebook edits**: every entry point is still exported from
`panelclv.studies`, and the only occurrences of the old module path in `notebooks/` are
stored tracebacks in `notebooks/archive/`, which is frozen by its own README.

Three placements the decision's table did not name, because it assumed symbols that issue
03 did not in fact kill:

- **`describe_dataset` / `describe_suite_dataset` → `suite_reader`.** Decision 7 wrote its
  table "after losing `describe_dataset`", but the map's import-only rule never fired on it
  and issue 03 explicitly left it alive. It reads a suite's archived recipe and describes
  the dataset that comes back, which is the reader's job; putting it in `suite_metrics`
  would have mixed dataset characteristics with forecast scoring.
- **`_actuals_from_panel` → `suite_reader`**, its three callers spanning both other modules.
- **`_suite_prediction_paths` → `suite_metrics`**, its only caller.

Private helpers keep their underscores across the new module boundary rather than being
promoted: D20's "promote the private cross-boundary import" ruling was about crossing
*subpackages*, and these three modules are one subpackage's internals.

### The tests patch the caller, not the definition site

`_actuals_from_panel` is imported by name into `suite_metrics`, so `monkeypatch.setattr` on
`suite_reader` would leave the caller's binding pointing at the original. The fixtures in
`tests/test_suite_analysis.py` (renamed from `test_studies_analysis.py`) patch
`suite_metrics` instead, and the file's docstring says why.

### The group set

`evaluation.segment_analysis.CUSTOMER_GROUPS = tuple(_GROUP_PREDICATES)` — the keys, so the
set cannot drift from the definitions. `_CANONICAL_GROUPS` is gone; both remaining
`groups=` defaults now read the constant. The catch-all is derived once, in
`assign_customer_groups(..., with_other=True)`, which replaced the `{**group_ids, "Other":
_other_ids(...)}` composition at both surviving call sites (the third died with issue 03's
row 11). A flag on the existing function rather than a second public name: both call sites
want assignment-plus-catch-all in one step, and the notebook's existing narrowing call is
unchanged.

### One Student-t interval

`suite_metrics.t_interval_half_width(std, n, ci)` is the arithmetic, and it is now the only
`stats.t.ppf` call in the package — `tests/test_imports.py` asserts exactly that, by
scanning the installed source. `_across_study_band` and `pnbd_grid._mean_ci` both call it;
`_study_metrics_from_data` builds its per-model half-widths from it. The three spellings
were already algebraically identical (`1-(1-ci)/2` = `0.5+ci/2`), so no number moved —
verified against SciPy directly, and `_mean_ci`'s dict is unchanged for n=4, n=1 and n=0.
`test_plot_band_is_that_same_interval` pins the plot band to the same helper the tables
use, which is the property the split exists to protect.

### The deferrals are gone, and asserted gone

Every function-body import in the module — eight functions deferred one — went to the top
of the file it landed in, not only the three that claimed torch-freeness: the same argument
voids the "keep the module import cheap" ones, and `warnings` / `matplotlib.patches` never
had one. The module
docstring's "stays torch-free until the plot itself is drawn" claim went with them.
`test_the_suite_modules_defer_no_imports` walks the three modules' ASTs and fails on any
import inside a function body, so a deferral now has to be argued for rather than added.

### Two references to the old name left standing, deliberately

`docs/adr/0006` still names `studies/analysis.py` in its context paragraph — a record of
where the drifted model-type copy *was*, which was true when it was written; rewriting it
would rewrite history rather than a description. And `notebooks/Study.ipynb` still passes
`groups=("At Risk", "Opportunity")` explicitly. That is a call site narrowing the set, not a
sixth encoding of it, and this issue's "zero notebook edits" forbids touching it anyway.

### Not deferred

The sizing note nominated this split as the set's designated deferral. It was not needed:
the split, the group collapse and the interval all landed together.
