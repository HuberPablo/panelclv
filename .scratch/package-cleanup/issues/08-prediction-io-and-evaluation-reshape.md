# 08 — Prediction I/O gets its own module, and `evaluation/` is reshaped

**What to build:** the model layer stops reaching back into the evaluation layer, so the
rule ADR-0002 states becomes a fact. The `_utils` suffix leaves the package, and the
surviving plot module earns its name because by then it only plots.

**Blocked by:** 02, 04

**Status:** done

Source: `.scratch/package-simplification/issues/06-target-architecture.md` (decision 6, Q14,
closing findings), `12-thesis-figure-code-home.md` (decisions 5, 6),
`09-module-naming.md` (decisions 1, 5), `13-safety-net-scope.md` (decision 6)

## The cycle this closes

This is not merely an ADR-0002 breach. The evaluation plot-utilities module imports the
model layer at top level, and the model layer imports prediction saving back **from inside a
function**, with a comment stating a top-level import "would create a circular import at load
time". That is a genuine subpackage cycle closing at module level and surviving only on the
deferred import. Moving prediction I/O into its own module removes it.

## Three destinations

After every prior ruling, only four functions in the old plot-utilities module retain live
callers, and two of them are not evaluation at all:

| destination | contents |
|---|---|
| a prediction-I/O module in `evaluation/` | prediction save, prediction load, and the customer-week reducer |
| a plots module in `evaluation/` | the weekly-aggregate plot and the metrics table |
| the Pareto/NBD benchmark module | the Pareto forecast and its data-shaping helper — **out of `evaluation/` entirely**, because they build a Pareto/NBD forecast, which is what that file is for |
| *deleted* | the weekly-aggregate predictions helper and the alignment check — import-only in notebooks, already killed by the import-is-not-a-caller rule |

The two symbols the plot scripts kept alive are already gone with issue 02, so there is no
pending branch here.

## Two private cross-boundary imports promoted to public

Not relocated — promoted. A private name that two notebooks depend on is the worst of both:
no stability promise and no discoverability.

- The Pareto data-shaping helper becomes public in the benchmark module. Two live notebook
  call sites are a public surface whether or not the underscore admits it.
- (The seasonal multiplier helper is issue 10's half of the same finding.)

## Renames folded in

The rollout functions are named by **mechanism**, not model family, and every remaining
`mc_*` alias is deleted — the model `__init__` currently exports one function under two
public names, twice over.

**Why mechanism:** there are **three** rollout model classes but only **two** rollout
functions — the recurrent one is shared by two different models. A family-based name would
be false the day a recurrent non-LSTM lands. The registry declares which function each model
uses, so the function name's job is to say what it does, not who calls it.

Cost: four notebooks for the recurrent forecast alias, three for the attention one.

## Docs and test

- **ADR-0002 Edit B** — a line appended to Consequences recording that the rule held before
  it held as a fact. Verbatim in ticket 08 of the map.
- **A ~10-line subpackage-acyclicity test** rides this issue. Two subpackage cycles existed;
  this issue and issue 12 fix both, and nothing today would notice either being re-added.
  This asserts the import graph itself — it is not the torch-cost proxy that was dropped.

- [x] Prediction I/O in its own module; no deferred import remains in the model layer
- [x] Plot-utilities module no longer exists; the `_utils` suffix is gone from the package
- [x] Pareto forecast and its helper live in the benchmark module, public
- [x] Import-only symbols deleted, with their notebook import lines
- [x] Rollout functions renamed by mechanism; every `mc_*` alias gone; notebooks updated
- [x] ADR-0002 Edit B applied verbatim from ticket 08
- [x] Acyclicity test present and passing
- [x] Golden test green at rel=1e-6; notebook API test green

## Comments

Landed 2026-08-13. Full suite green (207 passed), including the golden end-to-end test at
`rel=1e-6` and the notebook API test. Both ADR-0004 gate scripts re-run because
`benchmarks/` was touched.

### The prediction-I/O module is `panelclv/predictions/`, not `evaluation/predictions.py`

The issue's placement cannot be built. `evaluation/plots.py` and
`evaluation/segment_analysis.py` import `models` at top level — the direction ADR-0002
mandates — so a top-level import of `evaluation.predictions` from `models/` re-creates the
subpackage cycle this issue exists to close, and fails at load time as well: `import
panelclv.models` runs `evaluation/__init__`, which reaches back into a
half-initialised `monte_carlo_forecasting` for `compute_forecast_metrics`. Making the
import non-deferred is what the issue asks for, and it is what breaks.

Three things were therefore true at once and only two could stay: the module in
`evaluation/`, no deferred import in the model layer, and a subpackage-granularity
acyclicity test. **Pablo chose a new leaf subpackage**, `panelclv/predictions/`, which
imports nothing from `panelclv` — so every arrow into it points down and all three
constraints hold. `benchmarks.pareto_forecast` writes through the same leaf, which is what
keeps `benchmarks -> evaluation` from replacing the cycle that was removed. The map's
`evaluation/predictions.py` (ticket 12 decision 5) is superseded on this point only;
`evaluation/plots.py` and the Pareto move are exactly as ruled.

### A third private cross-boundary import was promoted, and respelled

The issue names two (the Pareto fitter here, the seasonal multiplier in issue 10). The
customer-period reducer is a third: `evaluation/plots.metrics_table` needs it across a
subpackage boundary, which is the same finding — a private name another subpackage depends
on. It is public as **`reduce_to_customer_period`**, not `..._week`: `CONTEXT.md` lists
*week* under _Avoid_ for **Period**, and the argument that the `week_` CSV columns are an
on-disk floor covers the header, not a new identifier.

### `weekly_aggregate_predictions` survives as a private helper

The public name is gone, as ruled — but `plot_weekly_aggregated` calls it, so its body
lives on as `plots._aggregate_across_customers`. Only its Monte Carlo branch is real work
now: the two path-free shapes delegate to the reducer, so the shape contract has one
definition rather than two near-identical cascades in two subpackages.

### The acyclicity test asserts more than the pair that motivated it

Three deliberate widenings, all cheap: it counts **deferred imports too** (a lazy import is
still a dependency, and hiding one is how the last cycle survived); it detects cycles of
**any length**, not just mutual pairs; and `KNOWN_CYCLES` is asserted by **equality**, so
when issue 12 removes `configs ⇄ data_preparation` the test goes red until that line is
deleted with it. It is over the map's "~10-line" sketch; the excess is the documented
allowance, the walk, and the docstrings explaining both.

### Four deferred imports elsewhere lost their reason and went

`studies/analysis.py` deferred its prediction-I/O and plotting imports to keep suite
discovery torch-free. The reader is now a torch-free leaf, and `analysis.py` already pulls
torch through the registry at module load, so the comments were false and the laziness
bought nothing. Same for the stale skip reason in `tests/test_archive_formats.py`, whose
`needs_torch` gate now names what actually pulls torch. Retargeting that file is issue 14.

### Not done here, by scope

`src/panelclv/training/training_utils.py` still carries the `_utils` suffix the checkbox
mentions — that rename is issue 09 decision 3's table, which issue 15 executes. Nothing in
`evaluation/` carries it any more, which is the half this issue owns. `VastAI/vast_search.py`
names `run_monte_carlo_forecast` in a comment and was left alone: it has unrelated
uncommitted edits in the working tree.
