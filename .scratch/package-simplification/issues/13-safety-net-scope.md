# How wide does the safety net need to be before the refactor lands?

Type: grilling
Status: resolved
Blocked by: 06

## Question

Ticket 11 requires that "the golden test from ticket 01 stays green at every step" — it
assumes the ticket-01 net is sufficient to catch a regression. Audit 02 shows it is
narrower than that assumption:

- `tests/test_golden_end_to_end.py` pins exactly ONE weekly happy path
  (`clip_target_upper=4`, two AR features, `add_year_idx` + `add_week_sin_cos`,
  all-`"auto"` embeddings).
- 2068 of the data lane's 2293 lines have no dedicated test. Nothing tests any
  `PanelConfig` validation error, the live monthly path, or `pareto_simulation.py`
  at all — the 544-line generator behind the thesis's synthetic grid panels.
- **Audit 03: the entire Transformer rollout has no test.** `simulate_transformer_path`,
  `run_monte_carlo_forecast_transformer` and `InferenceMultinomialTransformerModel` appear in
  zero test files; `test_golden_end_to_end.py` pins the LSTM path only. 2240 of the model
  lane's 2746 lines have no dedicated test. This is the sharpest case in the map, because
  audit 03 also showed the Transformer + recurrent-stepper crossing fails *silently* rather
  than raising — so any issue reshaping that seam (its options B/C/D) has no net at all
  today. Audit 03 argues this specific test is a **prerequisite** rather than a follow-up.
- **Audit 04: 3389 of the experiment lane's 4880 lines are untested**, with
  `pnbd_grid.py`, `studies/runner.py`, `segment_analysis.py` and `forecast_run.py` (1229 lines)
  having no test at all. `runner.py` is the production entry point for every study suite.
  Note ticket 14 now carves out the archive-format *reader* half of this gap, so this ticket
  decides the rest. Two specific candidates it handed over:
  - **A CPU-scale `run_study_suite` smoke test** — the *writer* is the other half of the on-disk
    floor and is still ungated, so a dropped `_suite_record` key is caught only where an archive
    happens to exist. Ticket 14 reports this is cheap now that its format constants exist: a
    1-trial / 1-epoch / 24-customer suite would pin `_suite_record`, `_model_record`,
    `results.csv` and `metrics.csv` against those same constants.
  - **`tests/test_notebooks_current_api.py` binds calls, not imports**, so a notebook importing a
    name it never calls is checked by nothing. That is a coverage gap *and* the mechanism behind
    the "is an import a caller?" ambiguity ticket 06 must settle.

The redesign is open, so it may reshape precisely the surfaces the net does not cover
(the calendar group, `PanelConfig`'s validation, the schema/embedding resolution). A
behaviour change there surfaces as a slightly different forecast, not a crash — the
failure mode ticket 01 was built to catch.

Decide: is the net widened before the refactor lands, and if so around which specific
surfaces? Or is the golden path plus the two benchmark validation scripts
(`validate_pareto_benchmark.py`, `validate_valendin_lstm.py`) accepted as sufficient,
with the rest carried by review?

Blocked by 06 because "widen the net around X" is only answerable once the target shape
says what X is. Whatever is decided becomes issues in the set ticket 11 cuts, and counts
against its ~15-issue tripwire.

## Answer

Settled with Pablo, 2026-08-12, over three rounds of `/grilling`. **The net widens once,
around one surface, at a cost of +1 issue.** Six decisions below; nothing is left open.

### The ruling

**Widen before the execution issues land — but narrowly.** Not because the refactor is
dangerous in general: the ledger found under 3% of the package deletable and 74 of its rows
are consolidations where the old copy is sitting right there to diff against. The case rests
on one surface only, the one that is simultaneously (i) untested, (ii) silently-failing,
(iii) directly reshaped by a decision in the target shape, and (iv) already written and
already running unasserted. That is the rollout seam.

Widening *after* the refactor was rejected outright: a net added afterwards pins whatever the
refactor produced, including a defect it introduced.

### 1. Budget — the tripwire is hard, and this ticket spends exactly one issue

Ticket 06's tripwire is worded as a check on *the destination* ("the destination was drawn
too wide — cut it"), and a test-only issue is insurance on the same destination rather than a
widening of it. It counts anyway. "We exceeded the budget, but for tests" is exactly the
reasoning that stops a budget working.

So: **+1, running total ~15**, at the tripwire and not over it. No cut is required. If
ticket 11's own carve lands at 16, the deferral is the one ticket 12 already nominated —
the `studies/analysis.py` three-way split — and not this issue.

### 2. Delivery — standalone for the regression baseline, folded for structure

Tickets 09 and 12 both bought their changes by folding them into the issue already touching
the module. Coverage splits along a sharper line than that:

- **A test whose job is "the number did not change" must land BEFORE the change**, and
  therefore stands alone. This is the entire mechanism of ticket 01's net. A test written by
  the issue it is meant to police is baselined on the new behaviour and can only say the new
  behaviour is self-consistent.
- **A test whose job is "the new structure is wired up"** can only exist against the new
  structure, and folds into the issue that creates it at +0.

### 3. The standalone issue — parametrise the golden test over model family

**Extends `tests/test_golden_end_to_end.py`.** Not a new file: the fixture, the
pinned-number machinery, the `PANELCLV_PRINT_GOLDEN=1` regeneration path and the tracer's
import point all already live there.

**Why the rollout seam and not the alternatives.** Two other candidates were weighed and
declined:

- **`run_study_suite` CPU smoke test** — declined, see the "declared uncovered" list below.
- **The data lane** (`PanelConfig`'s 13 `raise` sites, the monthly path,
  `pareto_simulation.py`) — declined. Decision 10's cut list does touch week/period arithmetic
  and the time-flag set, but `prepare_dataset`'s output is *already* pinned by the golden
  test's shape and feature-axis assertions, so the drift-a-number risk here is netted in part.
  The rollout seam is netted not at all.

**Four arms, three of them one shape.** `scripts/trace_golden_reachability.py` already runs
all four families today and asserts nothing about any of them:

| arm | rollout function | asserts |
|---|---|---|
| `lstm` | `run_monte_carlo_forecast` | existing — pinned metrics, shapes, feature axis, determinism, no-holdout-read |
| `transformer` | `run_monte_carlo_forecast_transformer` | pinned metrics + determinism + no-holdout-read |
| `valendin_lstm` | `run_monte_carlo_forecast` | pinned metrics + determinism |
| `pareto_nbd` | none — MCMC fit | **shape, finiteness, determinism-under-seed only. No pinned values.** |

- **The Transformer arm is the point of the issue.** `run_monte_carlo_forecast_transformer`
  is live production — `runner.py:51` `_FORECASTERS` routes every Transformer suite through
  it — and appears in zero test files. Its contract is byte-identical to the LSTM entry
  point (same signature, same return dict, only the stepper differs), so the arm is a fixture
  parameter, not a new pipeline. Cost is roughly the LSTM arm's ~15 s.
- **The Valendin arm buys something different from what it looks like.** It calls the *same*
  recurrent rollout as the LSTM arm, so its marginal value for the seam is near zero. Its
  value is that it is the only always-available end-to-end coverage of `ValendinLSTMModel`:
  `tests/test_valendin_lstm.py` pins architecture and one state-threading step but never a
  full rollout, and `scripts/validate_valendin_lstm.py` needs the gitignored `Datasets/`, so
  it cannot run on a fresh clone.
- **The Pareto arm is deliberately weaker.** `scenario_pareto_nbd` is not a parametrisation
  of the golden pipeline at all — it never trains and never rolls out, it calls
  `compute_pareto_predictions` with a 200-draw / 50-burn-in / single-chain MCMC fit whose own
  docstring says it "records which code runs, not whether it has converged", and ticket 01
  recorded it emitting `divide by zero encountered in log` on this panel. Pinning numbers off
  an unconverged chain pins noise, and would generate exactly the re-baselining pressure
  decision 5 forbids — the next person to hit it could not distinguish a real regression from
  MCMC drift.

**Second deliverable, and the reason the bundle is one session rather than four:** the tracer
imports all four scenarios **from the test**, and its own `_train_and_roll` copy of the
pipeline is deleted. Today the tracer imports only the LSTM scenario from the test —
explicitly "so the two cannot drift" (ticket 01) — and carries its own re-implementation for
the other three. The wide reading of this issue therefore *removes* a duplication rather than
adding code, which is the same consolidate-don't-duplicate move the rest of the map is making.

### 4. Ordering — the net lands first, ahead of ticket 10's sweep

Ticket 10 established the map's one ordering constraint: its scripts/root sweep lands first,
because the two deleted plot scripts import nine names across three subpackages the refactor
reshapes. That constraint is about not forcing later *structural* issues to keep two broken
scripts compiling against moving targets. A test-only commit is not in that class.

**So: this issue is #1, the sweep is #2, and ticket 10's constraint is restated as "first
among structural issues".** Ticket 11 should carry that wording, so the two "lands first"
claims do not read as a contradiction. Landing the net first means every subsequent issue,
the sweep included, is checked by it.

### 5. No re-baselining across `to_rollout()` — the pinned numbers are ticket 07's proof

Ticket 07 deletes the golden test's `inference_model.load_state_dict(torch.load(
fit.checkpoint_path))` construction in favour of `trained.to_rollout()`. **When that issue
lands, the pinned numbers must come out unchanged at `rel=1e-6`. That is its acceptance
criterion.**

The reasoning is worth stating because it is not obvious from either ticket alone. The golden
test today forecasts from the *checkpoint*, and `fit_model` writes `best_state` there — so
ticket 07's `fit_model` fix (load `best_state` back into the model object, which today it
never does) **does not move today's numbers**. But `to_rollout()` shares the training object's
in-memory backbone rather than reloading a file, so after the refactor the golden path reads
that object. Checkpoint-derived and object-derived forecasts agree **only if** the load-back
landed. The numbers are path-independent by construction when the fix is correct — same
weights, same seeds, same data — so a movement in them means it did not, which is precisely
the silent wrong-forecast ticket 07 described as "unobservable today".

Keeping both constructions side by side for one commit was considered and rejected: it buys
the same guarantee at the price of a scaffold that must then be removed.

### 6. Folded into their owning issues at +0

Three structure tests, none of which needs an issue of its own:

- **The registry test.** `tests/test_model_registration.py` exists to catch CLAUDE.md's
  "adding a model touches three places", which decision 4 retires. The registry issue rewrites
  it from three-list membership to one-table checks. It does not simply get deleted — its
  per-type *class* assertions guard the dispatch-fallthrough bug it was built for, and that
  risk survives the registry.
- **Retired names stay retired.** `tests/test_imports.py:91` already has
  `test_retired_metric_helpers_are_gone`. Each kill issue appends its names to that pattern —
  the ~11 rows / ~187 lines of decision 2's revised kill list, plus `rollout_composite`,
  `selection_metric`, `prediction_source` and the `mc_*` aliases.
- **Subpackage acyclicity — a ~10-line AST test over `src/panelclv/*/`, riding decision 6's
  issue.** Ticket 06's closing round found two subpackage cycles (`configs ⇄
  data_preparation`, latent because the module graph is acyclic; `evaluation ⇄ models`, closed
  at module level and surviving only on a deferred import), and decisions 6 and 10 fix both.
  Nothing today would notice either one being re-added. This is the one place where the
  refactor's gain is structural and therefore silently reversible.

  **This does not re-open ticket 08.** Ticket 08 dropped the *torch-free* import test; that
  assertion was a proxy for layering measured through import cost. This one asserts the
  import graph itself.

### Declared uncovered, and carried by review

The honest half of a one-issue budget. These are accepted risks, recorded here rather than
turned into issues:

- **`studies/runner.py` and the entire writer path.** No test calls `run_study_suite`. A
  dropped `_suite_record` / `_model_record` key, or a changed `results.csv` column, surfaces
  on the next real suite run.
- **`PanelConfig`'s 13 `raise` sites**, and the `monthly` frequency end to end.
- **`data_preparation/pareto_simulation.py`** — 544 lines, the generator behind the thesis's
  synthetic grid panels, no test at all.
- **`evaluation/segment_analysis.py` and every plotting path**, including `plot_suite_forecast`
  at 16 notebook calls.
- **`tuning/optuna_tuning.py`'s search itself.** No test drives an Optuna study; the search
  spaces are exercised only through `_FixedTrial` in `test_model_registration.py`.
- **The covariate-subset search** (`removable_features`), which audit 04 found `run_study_suite`
  never reaches at all.

**The `run_study_suite` smoke test is DECLINED, not merely omitted** — written down with its
reason, because it is the item on this list a future session will keep re-proposing:

1. Its main justification was protecting the on-disk format, and Pablo rescinded that floor
   item on 2026-08-11.
2. Its failure mode is loud. A missing key or a renamed column raises or produces a visibly
   wrong CSV; it is not the silent-different-number class this net exists for.
3. The refactor barely touches what it would pin. Decision 4 changes how `runner.py`
   *dispatches*, not what it writes.
4. It is the most expensive candidate — an Optuna trial plus `DEFAULT_REFIT_EPOCHS = 5` — for
   the weakest claim.

It remains a reasonable thing to build later, on its own merit as coverage of the production
entry point. It is not a prerequisite for this refactor.

### Findings this ticket produced

- **There is no CI in this repository.** No `.github/workflows`, and nothing runs pytest or
  the tracer automatically. Every claim in this map about a test "running" means someone ran
  it. This sharpens the Transformer gap rather than softening it: the seam executes only when
  a person manually invokes `scripts/trace_golden_reachability.py`.
- **The Transformer rollout is already written, exercised and unasserted.**
  `trace_golden_reachability.py:179-204` runs the full prepare → train →
  `run_monte_carlo_forecast_transformer` → `compute_forecast_metrics` path today. The gap in
  audit 03 is not "no code exercises it" but "no assertion is attached to the code that does".
- **The tracer holds a second implementation of the golden pipeline.** `_train_and_roll` plus
  three `build()` closures, for the three families it does not import from the test. Not in
  the ledger's 26 cross-lane duplications, because the ledger scanned `src/`.
- **The four tracer scenarios are not one shape.** Three train and roll out; `pareto_nbd`
  does neither. Any future attempt to "parametrise the four families" uniformly will hit this.
- **Ticket 05's characterisation of the golden test's model reload was incomplete in a way
  that matters to ticket 07.** The test reloads `best_state` from the checkpoint, so it is
  already forecasting from best weights — which is why ticket 07's `fit_model` defect is
  invisible today and becomes load-bearing the moment `to_rollout()` replaces the reload.
