# Spec: package cleanup

Status: ready-for-agent

Source: `.scratch/package-simplification/` — the `/wayfinder` map that decided all of this.
Fourteen tickets, thirteen resolved; ticket 11 is the handoff that produced this spec.
Every decision below traces to a resolved ticket, named inline. **This spec decides nothing
new.** Where it is silent and a ticket is not, the ticket wins.

## Problem Statement

`panelclv` works. It trains the Transformer and both frozen benchmarks, runs study suites,
and produced the thesis's numbers. But it has accumulated the kind of rot that a thesis
examiner reads as carelessness and that a future session pays for repeatedly:

- **One idea is written out many times.** Three audits and a merged ledger covering all
  10,139 lines found 26 cross-lane duplications: the model set enumerated seven times, the
  target column re-derived six times, four week-numbering conventions, three period-length
  tables (two disagreeing on `monthly`), the time-flag set written four times and *already
  drifted*, five prediction layouts, the Student-t interval three times — twice inside one
  file. A fourth copy of the model-type list had already drifted silently and collapsed the
  Valendin benchmark's across-study spread to a single study.
- **`CLAUDE.md` documents footguns instead of removing them.** "Adding a model touches three
  places, and missing the second fails only after training completes" is a warning where
  structure should be. So is the whole "Invariants worth knowing before you hit them" section.
- **Dead and broken code reads as live.** Both `main_plot*.py` scripts raise on invocation,
  yet they are the only callers keeping three `evaluation/` symbols alive. Three one-shot
  scripts that have already done their job sit beside live entry points. `forecast_run.py`
  has no callers at all.
- **The docs lie in specific, checkable ways.** ADR-0003 documents a selection metric
  unreachable from the production path. ADR-0002's "never the other way round" was violated
  for as long as the ADR has existed. `experiments/__init__.py` claims to hold "no modeling
  logic" while holding the sole enforcement point of ADR-0001. `CONTEXT.md` lists *inference*
  under `_Avoid_` while the package ships three `Inference*` classes.
- **Two subpackage-level import cycles exist**, one surviving only on a deferred import that
  carries a comment admitting why.
- **The safety net has a hole exactly where the refactor lands.** The Transformer rollout —
  live production, routed by every Transformer suite — appears in zero test files, and the
  wrong model/stepper pairing fails *silently* rather than raising.

The cost is not that the code is wrong today. It is that every one of these makes the *next*
change riskier than it should be, and several are wrong-by-construction the day a monthly
panel runs.

## Solution

Execute the map's decisions as an ordered set of issues, each sized for one agent session,
each leaving the tree green.

The shape is **consolidate in place, do not re-partition** (ticket 06). All nine existing
subpackages keep their boundaries; one is renamed and one tenth is added. Under 3% of the
package is deletable — the work is overwhelmingly collapsing many copies of one idea into
one, not moving folders.

Concretely, when this lands:

- **Adding a model touches one place** — an entry in a new `registry/` subpackage — and every
  model-type list in the package derives from that table's keys.
- **A rollout model is obtained from its trained model** via `to_rollout()`, sharing the
  backbone, so a constructor mismatch is not expressible.
- **A forecast always comes from a refit**, matching the paper rather than the paper authors'
  published code.
- **`CLAUDE.md`'s entire invariants section is gone**, because all four warnings are closed —
  two by structure that already exists and was never noticed.
- **The golden end-to-end test covers four model families**, not one, and lands *before* any
  structural change.
- **The docs describe the code.** Three new ADRs, three amended, one retired; `CONTEXT.md`
  gains the vocabulary the redesign needs.

## User Stories

1. As a thesis author, I want adding a new model to require one registry entry, so that I
   never again discover a missing registration only after training completes.
2. As a thesis author, I want `VALID_MODEL_TYPES` and the neural/non-neural distinction
   derived from one table, so that the drift that silently collapsed the Valendin benchmark's
   spread cannot recur.
3. As a thesis author, I want Pareto/NBD to sit in the same registry table as the neural
   models, so that no model type is a hand-written addend to the enumerations.
4. As an agent adding a model, I want the registry's fields to be optional, so that a model
   with no search space and no rollout still registers in one place.
5. As a thesis author, I want a trained model to hand over its own rollout model, so that
   pairing the wrong two classes stops being expressible.
6. As a thesis author, I want the training loop to leave the best-by-validation weights in the
   model it returns, so that a forecast taken from the in-memory model matches its checkpoint.
7. As a thesis author, I want the rollout-model-from-stored-parameters path deleted, so that
   there is one way to get a rollout model rather than two that can disagree.
8. As a thesis author, I want a pinned scalar hyperparameter to reach `best_params` like every
   other key, so that pinning a value fails loudly at spec time rather than with a `KeyError`
   after tuning.
9. As a thesis author, I want every forecast to come from a refit on the full calibration
   window, so that the package follows the paper rather than the authors' published code.
10. As a thesis author, I want `rollout_composite` selection removed entirely, so that
    `compute_forecast_metrics` becomes the single scoring authority in fact and not only in
    the charter.
11. As a thesis reader, I want one implementation of RMSE, bias and MAPE in the package, so
    that two numbers reported for one run cannot be 62x apart.
12. As a thesis author, I want prediction I/O in its own module, so that the model layer stops
    reaching back into the evaluation layer through a deferred import.
13. As an agent, I want the `evaluation` and `models` subpackages to stop importing each other,
    so that the layering ADR-0002 describes is a fact rather than a rule.
14. As an agent, I want `configs` to stop importing `data_preparation`, so that the bottom of
    the stack does not name something above it.
15. As a thesis author, I want the two broken plot scripts deleted rather than repaired, so
    that no issue in the set has to keep dead code compiling against moving targets.
16. As a thesis author, I want applied one-shot migrations deleted, so that a script in
    `scripts/` reads as something I might still need to run.
17. As a thesis author, I want the fact explaining why the archive says `ParetoNBD_MLE`
    preserved when its migration script is deleted, so that deleting a script does not lose
    the reason it existed.
18. As a thesis author, I want `CLAUDE.md` to state what earns a slot in `scripts/`, so that
    the next three one-off checks do not accumulate the same way.
19. As a thesis author, I want the unreferenced config directory and the PyPI publishing guide
    deleted, so that the repo root holds only what the thesis needs.
20. As a thesis author, I want `.Rhistory` gitignored, so that R stops re-committing it.
21. As a thesis author, I want the GPU-rental scripts described in one README line, so that
    live tooling stops looking like clutter.
22. As an agent reading the package, I want `experiments/` renamed to `trials/`, so that every
    subpackage name has a referent in the project's own vocabulary.
23. As an agent reading the package, I want the altitude ladder to be legible — one trial,
    then a search over trials, then many studies — so that I can tell `trials/` from
    `tuning/` from `studies/` by name alone.
24. As an agent, I want the calibration split named for the decision it makes, so that the
    sole enforcement point of ADR-0001 is not buried in a catch-all module.
25. As an agent, I want the calibration split to return a named type rather than a trailing
    dict, so that the thing every model constructor is rebuilt from is first-class.
26. As an agent, I want the subpackage docstring's "no modeling logic" claim corrected, so
    that the off-by-one it hid is visible.
27. As an agent, I want the search space and the training controls to be two fields rather
    than one dict, so that a typo lands in the wrong field instead of relying on a
    hand-maintained allowlist.
28. As an agent, I want the rollout functions named by mechanism, so that the name stays true
    when a recurrent non-LSTM model lands.
29. As an agent, I want every `mc_*` alias deleted, so that one function has one public name.
30. As an agent, I want `Inference*` renamed to `Rollout*`, so that the code stops using a
    word `CONTEXT.md` lists under `_Avoid_`.
31. As an agent, I want `mape_aggregate_style` renamed to `mape_aggregate`, so that the code
    matches the name `CLAUDE.md` already uses.
32. As an agent, I want the three Pareto/NBD files spelling one concept three ways unified, so
    that the file name matches the model-type string and the glossary.
33. As an agent, I want `max_trans` killed in favour of the two names that mean the two real
    concepts, so that head size and the config knob that sets it stop sharing a third name.
34. As an agent, I want `study_name` on the suite config renamed to `suite_name`, so that it
    stops colliding with a term the glossary defines as something else.
35. As an agent, I want the Pareto grid's `(rate, churn)` point renamed to `cell`, so that
    `group` means customer segment only.
36. As a notebook analyst, I want the suite-forecast plot to stay in the package, so that the
    most-called function in the package remains importable from my notebooks.
37. As a notebook analyst, I want the private Pareto helper my notebooks import promoted to
    public, so that I am not depending on an underscore across a subpackage boundary.
38. As a notebook analyst, I want the seasonal multiplier helper promoted to public, so that
    reconstructing the pattern a stored study was generated with is a supported operation.
39. As a thesis author, I want the synthetic-grid functions that read the generator's ground
    truth separated into their own module, so that "can this run on a real panel?" is
    answerable by reading an import line.
40. As a thesis author, I want the two implementations of the death-state and seasonal-shape
    measurements collapsed to one, with the published script's arithmetic winning, so that the
    thesis figures and the notebook tables cannot disagree.
41. As a thesis author, I want the 1195-line suite analysis module split three ways, so that a
    second copy of the Student-t interval cannot hide inside it again.
42. As a thesis author, I want one Student-t interval implementation, so that the band a plot
    draws and the CI a table prints are the same number.
43. As a thesis author, I want the customer-group set defined once by the predicates that
    define the groups, so that its five encodings collapse to one that cannot drift.
44. As an agent, I want the target column produced once and read everywhere, so that a drift
    cannot silently score the wrong column.
45. As an agent, I want one week-numbering convention and one period-length table, so that a
    monthly panel is not wrong by construction.
46. As an agent, I want the time-flag set written once, so that no further orphan columns
    appear.
47. As a thesis author, I want output folder names derived from config and seed, so that the
    same config and seed give the same paths.
48. As a thesis author, I want the golden end-to-end test to cover the Transformer rollout, so
    that the seam this refactor reshapes is not the one seam with no net.
49. As a thesis author, I want the Valendin benchmark covered end-to-end by a test that runs on
    a fresh clone, so that its coverage does not depend on a gitignored dataset.
50. As a thesis author, I want the Pareto/NBD arm to assert shape, finiteness and determinism
    but pin no values, so that an unconverged MCMC chain does not pin noise.
51. As a thesis author, I want the reachability tracer to import its scenarios from the test,
    so that the second implementation of the golden pipeline it carries is deleted.
52. As a thesis author, I want the golden numbers to come out unchanged when `to_rollout()`
    lands, so that the unchanged numbers are themselves the proof the weights fix landed.
53. As an agent, I want a test asserting the subpackage import graph is acyclic, so that a
    re-added cycle is not silently reversible.
54. As an agent, I want retired names asserted gone, so that a deleted symbol does not quietly
    return.
55. As a thesis reader, I want ADR-0003 retired in place with its reasons recorded, so that
    nobody re-proposes rollout-composite selection without knowing why it went.
56. As a thesis reader, I want ADR-0004 to say frozen means the numbers rather than the
    surrounding code, so that a benchmark file can be maintained without breaking its own rule.
57. As a thesis reader, I want the registry, the rollout-model handover and the refit-only
    forecast each recorded as an ADR, so that three decisions with three different reversal
    costs are not bundled into one summary.
58. As a thesis reader, I want ADR-0002 to record when its rule became true, so that I do not
    assume it always was.
59. As an agent, I want `CONTEXT.md` to define registry, rollout model, refit, recipe and
    customer group, so that the code and the glossary use the same words.
60. As an agent, I want each `CLAUDE.md` edit to land in the same commit as the structure it
    describes, so that the highest-traffic doc in the repo is never half-true.
61. As a thesis author, I want the whole invariants section removed once its four warnings are
    closed, so that no empty heading is left behind.
62. As a thesis author, I want the reachability tracer kept, so that after moving, renaming and
    splitting across ten subpackages I can check that nothing was orphaned.

## Implementation Decisions

Naming below uses module and symbol names deliberately: in a refactor spec they *are* the
decision. Line numbers and code blocks are not reproduced — the resolved tickets hold those,
and several tickets hold **finished text meant to be copied verbatim** (see Further Notes).

### The shape

- **Consolidate in place.** All nine existing subpackages keep their boundaries. No
  re-partition. (Ticket 06)
- **`experiments/` is renamed `trials/`** and split into a loaders module and a refit module.
  It survives as a subpackage; this is a rename, not a merge. (Tickets 06, 09)
- **A tenth subpackage, `registry/`**, holding the model registry. Both `models/` and
  `studies/` are blocked as homes by real import cycles; a repository-root module was rejected
  as clutter. (Ticket 06)

### The registry

- One table, one entry per model, with **optional fields**: search space, builder, and the
  rollout function it forecasts through. `pareto_nbd` sits in it declaratively.
- `VALID_MODEL_TYPES` becomes the table's keys. `NEURAL_MODEL_TYPES` becomes the derived
  predicate "this entry has a training builder" — the copy that already drifted.
- Entries hold **direct** references, not lazy ones. The torch-free guarantee that motivated
  laziness is dropped, and it was protecting a property `panelclv.studies` does not have.
- The `suggest_*_params` functions move into the registry subpackage.
- `studies/runner.py` keeps its neural and deterministic paths separate — they differ in more
  than the forecaster.
- The model-to-rollout-function pairing is **declared through the registry, not enforced by
  sealing**. No `Stepper` abstraction is built for two implementations. (Tickets 06, 07)

### The rollout model

- `trained.to_rollout()` returns the paired rollout model, **sharing** the backbone object,
  not copying it.
- The registry gets **no** rollout-class field; nothing outside `models/` and `benchmarks/`
  names a rollout class.
- `benchmarks/valendin_lstm.py` declares its own pairing inside the frozen file. Editing a
  frozen benchmark's surrounding code is permitted; the validation script is the gate.
- **Deleted:** `_build_inference_model_for`, `build_inference_from_trial`, and the two
  notebook cells that call them.
- **Added:** the training loop must load the best-by-validation snapshot back into the model
  before returning. Without this, `to_rollout()` returns last-epoch weights — a silent wrong
  forecast. (Ticket 07)

### Refit-only

- `prediction_source` and the notebooks' `REFIT_ON_FULL_CALIBRATION` toggles both go; one
  legal value is not a choice. The two knobs expressed the same decision at two altitudes.
- Rationale is paper fidelity: the paper refits, the authors' published code does not, and
  this package follows the paper. (Tickets 06, 08)

### Deletions

- **`rollout_composite` and everything it drags:** the rollout-metric recomputation, the
  validation-rollout score, the `ROLLOUT_METRIC` constant, the eleven `rollout_*` parameters,
  and `selection_metric` itself. Two notebooks' tuning cells are edited in the same commit.
- **The kill list is 11 rows / ~187 lines**, not the ledger's 13 / 282: the loss-variant
  cluster is **kept whole**, including its tests. `forecast_run.py` and its orphan fifth
  prediction layout go; the suite-distribution function goes.
- **Both plot scripts** (`main_plot.py`, `main_plot_covar.py`) — neither runs, neither ever
  produced a tracked figure, and their job is done by a function called 16 times from
  notebooks. Their deletion kills `forecast_from_checkpoint` and `holdout_actuals_NT`.
- **Three one-shots**: the transformer verification check and both migrations.
- **Root**: the unreferenced config directory and the PyPI publishing guide.
- **Kept**: the reachability tracer (documented, and the check that nothing was orphaned),
  `figures/`, `archive/`, `VastAI/`, and the vendored skills directory. (Tickets 06, 10)

### `evaluation/` reshaped

- **`plot_utils.py` ceases to exist.** Three destinations: a prediction-I/O module (which is
  what closes the `evaluation`/`models` cycle), a plots module that earns its name because by
  then it only plots, and the two Pareto functions moved **out of `evaluation/`** into the
  Pareto benchmark module.
- The import-only notebook symbols die with it.
- Two private cross-boundary imports are **promoted to public**, not relocated. (Tickets 06, 12)

### `studies/` reshaped

- **The suite analysis module splits three ways**: a suite reader, suite plots, and suite
  metrics. Zero notebook cost — notebooks import the subpackage, not the module.
- **The Pareto grid module splits in two**, putting the real-panel boundary between two files:
  the stored-results readers stay, and the three ground-truth-reading grid functions move to a
  synthetic-grid module.
- **The boundary test is "can it run on a real panel?"**, a property readable from the code —
  not "did it produce a thesis figure", which would evacuate the package's most-called
  function into a directory that is not importable.
- **Nothing relocates to `scripts/`.** (Ticket 12)

### Duplications collapsed

Three because they can make a *number* wrong:

- **Week and period arithmetic** — four week conventions and three period tables, two
  disagreeing on `monthly`, both feeding the Pareto/NBD fit.
- **The target column** — produced once, re-derived six more times.
- **The time-flag set** — written four times and already drifted.

Folded in opportunistically: the id-column two-fallback problem, the twice-defined data-builder
alias, the **Student-t interval written three times** (one implementation, in suite metrics),
the **customer-group set's five encodings** (survivor: the predicates that define the groups,
so its keys *are* the set), the root docstring listing 8 of 9 subpackages, and the
`configs`/`data_preparation` cycle.

**Explicitly left alone:** the suite-traversal pattern, the 21 byte-identical lines shared by
the two rollouts, and the five on-disk prediction layouts — unifying those was for archive
compatibility, which was rescinded.

**Explicitly frozen:** the validation scripts' internal re-implementations are *deliberate
insulation*. A gate that imports the code it gates stops being a gate. No issue may dedupe
them. (Tickets 06, 12)

### Reproducibility

Wall-clock timestamps come out of **three** output folder names, so a path is derivable from
config and seed. The timestamp survives as a metadata field. Not all six sites — one writes a
correct provenance field, one already opts out, one dies with its module. (Ticket 06)

### Naming

Renames fold into the issue already touching each module. Full table in ticket 09; the
expensive one to know about is `max_trans`, at 13 notebook occurrences. Notebook cost is
**measured per symbol, never assumed** — it spans two orders of magnitude, and notebooks
import subpackages rather than modules with four exceptions.

### Docs

- **Three new ADRs** — one registry entry per model; a rollout model comes from a trained
  model; a forecast comes from a refit. Each lands with the issue that makes it true.
- **ADR-0003 retired in place**, with a `Retired` header rather than `Superseded by` — nothing
  replaces it — and a section recording the three facts worth keeping.
- **ADR-0001, -0002, -0004 amended**; **ADR-0005 untouched**, which is worth stating
  explicitly rather than by omission.
- **`CONTEXT.md`** gains *Registry*, *Rollout model*, *Refit*, *Recipe*, *Customer group*, and
  states there is deliberately **no term for "experiment"**.
- **`CLAUDE.md`**: the "three places" paragraph is rewritten to one place; the entire
  "Invariants worth knowing before you hit them" section is removed, heading included; one
  line is added on what earns a slot in `scripts/`. (Tickets 07, 08, 10)

## Testing Decisions

**What makes a good test here.** The failure mode this refactor risks is not a crash — it is a
slightly different forecast. So the tests that matter assert *numbers out of the public
pipeline*, at the highest seam available, with no knowledge of how the pipeline is wired
internally. A test that asserts a call happened, or that a private helper returns a shape, will
not catch the class of regression this work can introduce.

**The seams — all three already exist.** No new seam is created.

1. **`tests/test_golden_end_to_end.py`** — the primary net, and the highest seam in the repo:
   one seeded run of prepare → train → refit → rollout → score, with metrics pinned at
   `rel=1e-6` and determinism asserted exactly. It is **parametrised over four model families**,
   and this lands **first, before any structural issue**. A net added afterwards pins whatever
   the refactor produced.

   | arm | asserts |
   |---|---|
   | recurrent (existing) | pinned metrics, shapes, feature axis, determinism, no holdout read |
   | attention | pinned metrics, determinism, no holdout read |
   | Valendin benchmark | pinned metrics, determinism |
   | Pareto/NBD | **shape, finiteness, determinism only — no pinned values** |

   The Pareto arm is deliberately weaker: it never trains and never rolls out, and its
   200-draw single-chain fit would pin noise, generating exactly the re-baselining pressure
   this net forbids.

2. **`tests/test_archive_formats.py`** — its 19 fixture-driven tests are **kept and
   relabelled** as read-path coverage for the suite layout and reader, which is the only
   coverage those have. The 4-5 tests reading the real archive go, as do the two asserting
   warts that are now fixable. The file is not deleted wholesale — that trades one unjustified
   constraint for a coverage hole.

3. **`tests/test_notebooks_current_api.py`** — the existing seam that resolves every notebook
   import and fails on a missing name. It is what makes the rename issues safe, and it binds
   both calls and imports.

**Second deliverable of the net issue, and why it is one session rather than four:** the
reachability tracer imports all four scenarios **from the test**, and its own copy of the
golden pipeline is deleted. This *removes* a duplication rather than adding code.

**Folded into their owning issues at +0 — structure tests, which can only exist against the
new structure:**

- The registry rewrite of `tests/test_model_registration.py`, keeping its per-type class
  assertions, which guard a dispatch-fallthrough risk the registry does not remove.
- Retired-name assertions appended to the existing pattern in `tests/test_imports.py`.
- A ~10-line **subpackage-acyclicity** test riding the prediction-I/O issue.

**The one hard acceptance criterion.** When `to_rollout()` lands, the golden numbers must come
out **unchanged at `rel=1e-6`**. The test forecasts from the checkpoint today, so the training
loop's weights defect is invisible until `to_rollout()` reads the in-memory object. The
unchanged numbers *are* the proof the load-back landed. No re-baselining.

**Prior art.** `test_golden_end_to_end.py` is the model for all of this — fixture, pinned
numbers, a regeneration path behind an environment variable, CPU-only, ~15s. New arms are
fixture parameters against that machinery, not new files.

**Declared uncovered, carried by review** — recorded as accepted risk, not oversight: the
study-suite writer path, `PanelConfig`'s validation raises, the monthly frequency end to end,
the Pareto simulation generator, the segment analysis and plotting paths, the Optuna search
itself, and the covariate-subset search.

**The study-suite smoke test is DECLINED, with reasons**, because it is the item a future
session will keep re-proposing: its main justification (protecting the on-disk format) was
rescinded, its failure mode is loud rather than silent, the refactor changes how the runner
dispatches rather than what it writes, and it is the most expensive candidate for the weakest
claim. It remains reasonable to build later on its own merit.

**There is no CI in this repository.** Nothing runs pytest automatically. Every "the test
passes" in this spec means someone ran it — which sharpens the case for the net rather than
softening it.

## Out of Scope

- **Making the frequency-agnostic panel-config promise real.** Hardcoded frequency assumptions
  stay recorded as evidence; acting on them is a later effort.
- **Running the package on a genuinely new, unfamiliar panel.** The natural successor once
  this lands.
- **PyPI or shipping-grade hardening.** The bar is thesis defence. With no external consumer,
  an unreferenced export is dead rather than public API — which is what licenses the deletions.
- **`notebooks/` as a cleanup target**, and `notebooks/archive/` entirely. Notebooks are a
  **constraint** — every rename updates the four live ones in the same commit — never a target.
- **Archived checkpoints and Optuna storages.** The files are expendable and ~5 GB have already
  been deleted. **The checkpoint *mechanism* is load-bearing and no issue may remove it** — it
  is the only path carrying weights from training to forecasting, and its public surface
  appears in all four notebooks.
- **Archive re-readability.** Rescinded as a floor item. Renaming a results column, changing
  the suite tree, or dropping the legacy read path are all free moves.
- **Re-partitioning the subpackages**, unifying the five on-disk prediction layouts, deduping
  the validation scripts, and the two duplications ticket 06 explicitly left alone.

**The floor no issue may cross — two items, not three:**

1. **The forecasting contract** — categorical head over count classes, cross-entropy on a
   class index, forecast by sampling-and-averaging rollouts.
2. **Benchmark fidelity** — *the numbers, not the surrounding code*. The two validation
   scripts are the executable definition and gate any benchmark-touching issue.

## Further Notes

**Budget: ~15 issues against ticket 06's ~15 tripwire.** There is no headroom. If the carve
lands at 16, the designated deferral is the three-way split of the suite analysis module — the
one decision justified by future legibility rather than by a wrong number or an unreachable
path. It is not the test issue.

**Ordering — two constraints, in this order:**

1. **The parametrised golden test is issue #1**, before everything.
2. **The scripts/root sweep is #2 — first among *structural* issues.** The deleted scripts
   import nine names across three subpackages that later issues reshape and rename; landing
   the deletions last would force every issue in between to keep two broken scripts compiling
   against moving targets.

**Evidence travels with the issue.** Every deletion issue cites the ledger row that justifies
it, so whoever executes it does not re-derive whether a symbol is dead. The ledger is
`.scratch/package-simplification/ledger.csv` / `.md`, 163 rows covering all 10,139 lines.

**Two rules the ledger could not settle, both ruled by Pablo and both binding:**

- **A notebook import is not a caller.** Only a call keeps a symbol alive. Deleting an
  import-only symbol requires stripping the import line from the notebook in the same commit.
- **The thesis carve-out overrides:** anything that produced a figure or a number in the thesis
  is alive regardless of callers.

**Verbatim text exists and must be copied, not re-derived.** Ticket 08 carries the full
finished text of the three new ADRs and the three deferred amendments. Ticket 07 carries four
heading-anchored `CLAUDE.md` edit blocks, **each naming its owning issue**. Ticket 07 requires
that ticket 11 confirm every `CLAUDE.md` edit has an owner **before any code work starts** —
if two land and one stalls, the doc is left half-true, which is the failure this whole exercise
exists to prevent.

**The failure signature to apply to anything an issue is tempted to document rather than fix:**

> Missing structure, failure only after training completes.
