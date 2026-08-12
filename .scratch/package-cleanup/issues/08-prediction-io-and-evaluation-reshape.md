# 08 — Prediction I/O gets its own module, and `evaluation/` is reshaped

**What to build:** the model layer stops reaching back into the evaluation layer, so the
rule ADR-0002 states becomes a fact. The `_utils` suffix leaves the package, and the
surviving plot module earns its name because by then it only plots.

**Blocked by:** 02, 04

**Status:** ready-for-agent

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

- [ ] Prediction I/O in its own module; no deferred import remains in the model layer
- [ ] Plot-utilities module no longer exists; the `_utils` suffix is gone from the package
- [ ] Pareto forecast and its helper live in the benchmark module, public
- [ ] Import-only symbols deleted, with their notebook import lines
- [ ] Rollout functions renamed by mechanism; every `mc_*` alias gone; notebooks updated
- [ ] ADR-0002 Edit B applied verbatim from ticket 08
- [ ] Acyclicity test present and passing
- [ ] Golden test green at rel=1e-6; notebook API test green
