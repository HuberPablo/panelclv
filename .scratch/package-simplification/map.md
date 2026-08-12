# Map: package simplification

Label: wayfinder:map

## Destination

A decided kill/keep/refactor ruling for every module and public symbol in
`src/panelclv/`, `tests/` and `scripts/`, plus a target architecture the package is
refactored toward — handed off as an executable issue set under
`.scratch/package-cleanup/issues/`, with `docs/adr/` and `CONTEXT.md` reconciled to
match. The map is done when nothing is left to decide. It writes no production code.

## Notes

**Domain.** Read `CONTEXT.md` for the vocabulary, `docs/adr/` for prior decisions and
`docs/feature_engineering.md` before touching anything read during a rollout. HITL
tickets invoke `/grilling` and `/domain-modeling`; the synthesis ticket also invokes
`/codebase-design`.

**Redesign is open** — any subpackage may be re-architected. The floor, which no
ticket may cross:

1. **The forecasting contract** — categorical head over count classes, cross-entropy
   on a class index, forecast by sampling-and-averaging rollouts.
2. **Benchmark fidelity** — frozen means *the numbers*, not the surrounding code.
   `benchmarks/` plumbing may be reshaped provided `scripts/validate_pareto_benchmark.py`
   and `scripts/validate_valendin_lstm.py` still land in their bands afterwards. Those
   two scripts are the executable definition of this floor and gate any benchmark-touching
   ticket.
3. ~~**On-disk study formats**~~ — **RESCINDED by Pablo, 2026-08-11.** This floor item is gone.
   Archived studies do not have to stay re-readable by the refactored package. Clean code outranks
   backward compatibility with the archive. The floor is now TWO items, not three.

   Consequences, so no ticket re-imposes the constraint by habit:

   - **No ticket owes the archive anything.** Renaming a `results.csv` column, changing the suite
     tree, or dropping `study_metrics`' legacy path are all free moves. The format warts ticket 14
     pinned — `aggregated_*.csv` keyed `customer_id` beside `Id`, `study_metrics` raising on the
     four legacy suites — are now **fixable rather than preservable**.
   - **The data is not at risk, only package-level readability.** The 444 `results.csv`,
     1071 `Prediction_*.csv` and 1377 `config.json` under `Studies/` remain on disk and are plain
     CSV/JSON. If a refactor stops `analysis.py` reading them, they are still readable with a few
     lines of pandas. So this ruling costs convenience, not evidence.
   - **Ticket 13's case for a `run_study_suite` writer test weakens**, because its main
     justification was protecting the archive format. Re-weigh it there as ordinary coverage of
     the production entry point instead.

   Archived `.pth` checkpoints and Optuna storages were already ruled off the floor the same day,
   and 2917 of them (5.27 GB) have since been deleted.

**What counts as dead.** No caller in `src/` or a live entry point (`run_studies.py`,
the two `validate_*.py`, the two `main_plot*.py`, the four live notebooks). Tests and
one-shot scripts are not callers. **Carve-out:** anything that produced a figure or a
number in the thesis is alive regardless of callers — deleting it makes a published
result unreproducible.

**Notebooks are a constraint, not a target.** Any ticket renaming a public name updates
the four live notebooks in the same commit, so `tests/test_notebooks_current_api.py`
never goes red. `notebooks/archive/` is deliberately frozen and out of scope.

**Budget tripwire.** If ticket 06 proposes a target shape whose execution exceeds ~15
issues, the destination was drawn too wide — cut it rather than start a rewrite.

**Settled by Pablo, 2026-08-11 — archived checkpoints are expendable. Read this precisely.**
Two different things share the word "checkpoint", and only one is expendable:

- **Expendable: the archived FILES.** 2572 `.pth` under `Studies/` (4.7 GB) and 345 under
  `checkpoints/` (542 MB). Nothing needs to reload them. The thesis numbers live in the stored
  `Prediction_*.csv` and `results.csv`, which ticket 14's gate pins and which remain on the floor,
  so deleting the weights does not make a published figure unreproducible.
- **Load-bearing: the checkpoint MECHANISM.** It is the only path carrying weights from training
  to forecasting — `training_utils.py:348,462` writes, `experiment_utils.py:43` reads, and
  `refit_best_trial` (every production run) reads one twice: warm-start, then reload of the refit
  it just wrote. `checkpoint_dir`, `checkpoint_path` and `keep_only_best_checkpoint` are live
  public surface in all four notebooks. **No ticket may delete this.** A ticket that reads
  "checkpoints are deletable" as licence to remove `torch.save` / `load_state_dict` breaks the
  train-to-forecast handoff.

What the ruling unlocks: ticket 06 may reshape model constructors freely, because the CLAUDE.md
invariant "an inference model loads its `state_dict` from the trained model, so their constructor
arguments must match" now only has to hold WITHIN a run, not across the archive. The three
`_cached_mask` back-compat pops become plainly killable (they exist for a checkpoint format no
archived file uses), and `scripts/migrations/rename_embedder_checkpoint_keys.py` loses its last
purpose — input to ticket 10.

Accepted cost, stated once: without archived weights you cannot re-run a rollout with more
simulated paths, or forecast a different holdout window, without retraining first.

**Settled by Pablo, 2026-08-11 — a notebook IMPORT is not a caller.** Only a call keeps a
symbol alive. `from panelclv.x import f` with no `f(...)` anywhere in the notebook counts as no
caller at all. This resolves the one rule ambiguity the ledger could not: the three lane audits
had applied three different readings.

Two consequences, both verified:

- **Deleting an import-only symbol requires stripping the import line from the notebook in the
  same commit.** `tests/test_notebooks_current_api.py::test_panelclv_imports_resolve` (`:115`)
  resolves every `from panelclv... import X` in the four live notebooks and fails on a missing
  name, so a forgotten import line turns the suite red rather than passing silently. (This
  corrects ticket 05's claim that the test "binds calls, not imports" — it does both.)
- **Rows the rule moves to `kill`**, each previously alive only through notebook imports:
  `evaluation/plot_utils.weekly_aggregate_predictions`, `evaluation/plot_utils.alignment_check`,
  `evaluation/plot_utils.weekly_actuals` (was `conditional-10`; its only other caller was the
  broken `main_plot.py`), and `studies/analysis.describe_dataset`. Ticket 06 should re-apply the
  rule against `ledger.csv`'s `callers_live` column to catch any row this list misses.
  Note `models/losses.compute_class_weights` is NOT moved — two notebooks genuinely call it; that
  its result is then discarded by the cross-entropy branch is a separate finding.
  The thesis carve-out still overrides: a symbol that produced a published figure stays alive.

**Correction to the entry-point list — BOTH `main_plot*.py` scripts are broken.** Verified
independently, from audits 03 and 04:

- `main_plot_covar.py` calls `evaluation.plot_utils.forecast_from_checkpoint` with six keyword
  arguments it does not accept (`calibration`, `holdout_calendar`, `seq_cols`, `target_col`,
  `model_type`, `batch_size`) and reads a `result["predictions"]` key nothing returns → `TypeError`.
- `main_plot.py` is a two-phase library, and its intended entry path is broken. You call
  `compute_and_save_lstm` / `_transformer` / `_pareto_nbd` to write prediction CSVs, then
  `main(holdout=...)` reads them back and plots. `main()` breaks at `:244-251`: it feeds
  `weekly_actuals`' 1-D `(T_HOLD,)` aggregate to `metrics_table`, which raises when
  `actuals.ndim != 2`. (Its `__main__` block also raises, but that is a *deliberate* stub —
  `_load_dataset` is documented as "replace with your own loader, or import `main(holdout=...)`
  from a notebook" — so the stub is a seam, not the defect.) The two scripts are diverged
  copies of one file and `main_plot_covar.py:171-188` is the *fixed* form of the very call
  `main_plot.py` gets wrong, so each holds the other's fix.

**Consequence for audit 03's stepper ruling:** verified that **nothing anywhere calls
`compute_and_save_transformer`** (or `compute_and_save_lstm`) — not `main()`, not a notebook,
not a test. So the silently-wrong Transformer + recurrent-stepper crossing that audit 03
demonstrated has **no caller at all**, runnable or otherwise. Audit 03's structural argument
stands; its claim that a live entry point "takes that branch today" does not, and its option (B)
loses its bug-fix justification while keeping its make-it-unrepresentable one. Ticket 06 should
weigh it as latent-risk-prevention, not as a fix.

**So the live entry points are: `scripts/run_studies.py`, `scripts/validate_pareto_benchmark.py`,
`scripts/validate_valendin_lstm.py`, and the four live notebooks.** Neither plot script keeps
anything alive — that removes the only runnable caller of `forecast_from_checkpoint`,
`holdout_actuals_NT` and `weekly_actuals`. The thesis carve-out still protects whatever produced
a published figure, but note audit 04's finding that `holdout_actuals_NT`/`weekly_actuals` consume
a `Sequence[pd.DataFrame]` shape nothing in `src/` produces any more, so those figures are already
not reproducible either way. Ticket 10 rules on both scripts' fate.

**Already fixed during charting, so audits need not re-report it:** `studies/analysis.py`
carried a fourth model-type list (`_NEURAL_TYPES`) that had drifted out of sync with
`studies/config.NEURAL_MODEL_TYPES`, silently collapsing the Valendin benchmark's
across-study spread to a single study. Replaced with an import; regression test added
to `tests/test_model_registration.py`.

## Decisions so far

<!-- one line per closed ticket: gist + link -->

- [Golden end-to-end reproducibility test](issues/01-golden-end-to-end-test.md) — `tests/test_golden_end_to_end.py` pins one seeded run of the whole pipeline (determinism asserted exactly, regression at rel=1e-6, CPU-only, ~15s); `scripts/trace_golden_reachability.py` traces four model families and writes `reachability.md`/`.csv` — 79 of 206 symbols reached, to be read as proof of life, never proof of death.
- [Audit: the data lane](issues/02-audit-data-lane.md) — nothing in `configs/` or `data_preparation/` is unreachable; the rot is 15 public symbols with no caller outside their own module (incl. the vestigial `PanelConfig.data_config`/`.schema` adapter), 3 return keys nobody reads (`"N  "` — trailing spaces, `"panel"`, `"holdout_panel"`), 5 concepts implemented twice (3 week-index conventions, 2 PNBD simulators, 2 cohort filters, 4 encodings of the time-flag set, 2 config→dict views), an inert `observed_past` role, an unexercised `daily` frequency, and 2068 of 2293 lines with no dedicated test.
- [Audit: the model lane](issues/03-audit-model-lane.md) — nothing in `models/`, `benchmarks/` or `training/` is a dead module, but the model→stepper pairing is written 3 times and enforced 0 times, so a Transformer + recurrent stepper is *silently* wrong (a TypeError the other way) — `plot_utils.forecast_from_checkpoint` takes that branch today; 3 of 4 `build_criterion` branches (`focal`/`emd`/`weighted_ce`) have never been selected by any run ever and `compute_class_weights`' output is discarded on every live run; `models`→`evaluation` inverts ADR-0002; 3 unread forecast keys + `FitResult.best_val_f1` computed every epoch via sklearn and read nowhere; 18 of 27 hoisted `seq_cols`/`target_col` assignments unread, 21 byte-identical lines shared by the two rollouts, 3 dead `_cached_mask` pops (0/108 checkpoints carry it); and 2240 of 2746 lines with no dedicated test — including the whole Transformer rollout.
- [Audit: the experiment lane](issues/04-audit-experiment-lane.md) — two genuinely dead units (`ForecastRun`, 111 lines, zero callers plus a 5th on-disk prediction format with no writer; `group_metrics_suite_distribution`); **`scripts/main_plot.py` is broken too** (1-D aggregate into `metrics_table` → `ValueError`), so with both plot scripts dead 5 `evaluation/` symbols have no runnable caller and 2 consume a shape `prepare_dataset` no longer produces; `run_study_suite` passes neither `selection_metric` nor `removable_features`, so ADR-0003's rollout selection and the whole covariate search are unreachable from the production path; **`compute_forecast_metrics` is NOT the single authority** — `tuning.weekly_aggregate_rollout_metrics` recomputes all three, rmse 62× apart (per-cell vs customer-summed) and mape by a different estimator, bias alone matching; 5 prediction layouts, 3 Student-t intervals, 2 period-day tables disagreeing on `monthly` (30.0 vs 30.4368) both feeding the Pareto fit, 5 encodings of the group set, 2 contradictory misalignment policies; 3389 of 4880 lines untested with `pnbd_grid`/`runner`/`segment_analysis`/`forecast_run` (1229 lines) untested entirely.
- [Merge the audits into one kill/keep/refactor ledger](issues/05-reachability-ledger.md) — 163 rows in `ledger.csv`/`ledger.md` (32 modules, 131 public symbols, all 10,139 lines): 70 keep, 74 refactor, 13 kill (~282 lines, 2.8% — `forecast_run.py`, the two unused losses, two dead `mc_simulate_*` aliases, `group_metrics_suite_distribution`), 3 kill-candidates, 3 conditional on ticket 10; 26 deduped cross-lane duplications incl. one neither audit found (`DataBuilder` defined twice, in `experiment_utils.py:41` and `optuna_tuning.py:116`) and reconciled counts (5 prediction layouts, 7 model-type registries, 4 week conventions + 3 period tables); 5 audit conflicts and 4 places the dead-code rule was applied inconsistently — of which "is a notebook import a caller?" decides 6 rows and is a policy call, not an evidence question.
- [Give the on-disk-format floor an executable definition](issues/14-archive-format-gate.md) — `tests/test_archive_formats.py` (24 tests, 3.5 s, CPU-only): a literal-text fixture suite always pins the tree, `results.csv`'s leading columns, `Id,week_0..week_{T-1}`, and the `config.json`→`from_dict`→`to_dict` identity; skip-if-absent tests add the real archive (`Studies/` AND `Datasets/` are both gitignored, so CI can never see it) — all nine stored `results.csv` values of `..._TestDimanche` recomputed from its predictions at rel=1e-12. 11 mutations (4 archive, 7 code, in scratch copies) each fail it; degrades to 19 passed / 4 skipped on a fresh clone. Pins both known warts as-is (`customer_id`/`Id` aggregates; `study_metrics` raising on the 4 legacy suites) and records that the archive read path is NOT torch-free — `load_predictions_from_csv` sits in `plot_utils`, which imports torch at module level. The **writer** (`run_study_suite`) remains ungated.

- [Target architecture synthesis](issues/06-target-architecture.md) — **consolidate in place, do not re-partition:** all nine subpackages keep their boundaries (`experiments/` survives), ~10-11 execution issues. Twelve decisions: `rollout_composite` deleted outright and **ADR-0003 retired** (~175 lines + `selection_metric`); the loss cluster **kept whole**, shrinking the kill list from 13 rows/282 lines to **11 rows/~187 lines**; refit-only, so `prediction_source` and `REFIT_ON_FULL_CALIBRATION` both go; **one registry entry per model** (search space, builder, inference builder, forecaster, rollout function) which **retires CLAUDE.md's "three places"**; the model→rollout pairing *declared* through that registry, not sealed (settles ticket 03/C1); the ADR-0002 breach fixed but the five prediction layouts left unified-never; `test_archive_formats.py` cut to its 19 fixture tests and relabelled read-path coverage; timestamps out of 3 output folder names (not all 6 D19 sites); `validate_*.py`'s copies frozen as deliberate insulation (C5 resolved for audit 03, against 02); and a cut list of 3 correctness duplications (D7 week/period, D16 target column, D13 time flags) with D1/D17/D22 folded in and D9/D24 left alone. **Four new findings:** the paper's RMSE is *individual-level* so `compute_forecast_metrics` is correct and D8/C2 resolve against the tuning code (deleting it makes the "single scoring authority" claim true rather than needing a carve-out); `rollout_composite` *was* live in two notebooks; **all 1256 archived `selection_metric` values are `val_loss`**, zero composite, which is what makes its deletion free; and **`configs` ↔ `data_preparation` import each other** at module level (subpackage cycle, acyclic module graph — not in the ledger). **Closed 2026-08-12** — registry lands in a **tenth subpackage** `registry/model_registry.py` (both `models/` and `studies/` blocked by real top-level cycles; a root module refused as clutter), entries holding *lazy* references so `studies/config.py` need not import torch to validate a `model_type`; `evaluation/`'s internal split handed to ticket 12; the `configs` ↔ `data_preparation` cycle folded into decision 10. **Two closing findings:** a *second* subpackage cycle `evaluation ⇄ models`, which decision 6 already closes (so it fixes a cycle, not just an ADR-0002 breach); and the torch-free guarantee measured at ~1.2 s / ~540 MB with torch a *hard* dependency — worth restating as a layering rule plus a one-line test (tickets 08/13), or dropping.

- [Which documented invariants get collapsed](issues/07-collapse-invariants.md) — **all four `CLAUDE.md` warnings close, and the "Invariants worth knowing before you hit them" section disappears entirely.** Registration collapses to **one table with optional fields** so `pareto_nbd` sits in it (`VALID_MODEL_TYPES` = its keys; `NEURAL_MODEL_TYPES` becomes the derived predicate "has a training builder" — the copy that already drifted); its entry is declarative only, `runner`'s two paths stay separate. The state-dict invariant closes by **`trained.to_rollout()`** — the training class hands over its own backbone, shared not copied, so the registry needs *no* rollout-class field and `benchmarks/valendin_lstm.py` declares its pair inside the frozen file (permitted; `validate_valendin_lstm.py` is the gate) — and the path that made it necessary is **deleted**: `_build_inference_model_for`, `build_inference_from_trial`, and the two notebook cells, since decision 3 already removes their only live callers. The target-column and `clip_target_upper` warnings need **zero code changes** — already enforced by the base `Embedder.__init__`, `prepare_dataset` step 0 and `resolve_embedded_cols` — so they are deleted and the head-size fact relocates. `_suggest_param` registers a fixed scalar as a single-choice categorical, killing the late `KeyError`. **Amends ticket 06 decision 4:** the entry's *inference builder* field is gone and no rollout-class field replaces it. **The one addition:** `fit_model` must load `best_state` back into the model — it snapshots at `:325`, saves at `:347` and never loads back, so the object holds the *last* epoch's weights and early stopping guarantees they differ; unobservable today, a silent wrong-forecast the moment `to_rollout()` reads that object. Invariant set closed; ticket 11 gets the failure signature ("missing structure, failure only after training completes"), four heading-anchored `CLAUDE.md` edit blocks each naming its owning issue, and **+1 net issue**.

- [Reconcile the ADRs and CONTEXT.md with the redesign](issues/08-reconcile-adrs-and-vocabulary.md) — **0001 amended** (a stale "both selection metrics" line no audit caught), **0002 amended** (a line recording that "never the other way round" is finally *true*, not just ruled), **0003 retired in place** with a `Retired 2026-08-12` header — not `Superseded by`, since nothing replaces it — plus a "Why it goes" section keeping the three facts worth keeping, **0004 amended** three ways (*frozen means the numbers, not the surrounding code*, with the two `validate_*.py` scripts named as the test; decision 9's insulation ruling, which had no home outside a ticket; and its torch-free consequence deleted), **0005 untouched** — the only ADR an open redesign leaves alone. **Three new ADRs, written in full here:** 0006 *one registry entry per model*, 0007 *a rollout model is obtained from a trained model*, 0008 *a forecast comes from a refit*. **The charter item turned out unnecessary:** `CLAUDE.md`'s "single scoring authority" needs neither carve-out nor rename — `src/` holds exactly two RMSE computations and decision 1 deletes the second, so the claim just becomes true. **Torch-free removed as an idea** (Pablo), with a six-site inventory, a keep-list for the two lazy imports that are about `wandb`/`optuna` and the MCMC fitter, and the finding that three of the deferrals were **already inert** — `panelclv.studies` pulls torch at package import, so `analysis.py`'s deferrals never saved anything and one of them defers `pandas`. **Consequence for ticket 06: the registry may hold direct references**, and ticket 13 loses an import test. **`CONTEXT.md`** gains *Registry*, *Rollout model* (which finally gives ticket 09 a correct name to rename `Inference*` toward) and *Refit*, and states there is deliberately **no term for "experiment"** — so ticket 09 inherits that the `experiments/` subpackage needs renaming. Delivery splits: decision records landed with this commit, structure descriptions attach to their execution issues with full text supplied.

- [Settle module and subpackage naming](issues/09-module-naming.md) — **thirteen decisions; the rule is rename unless the name alone cannot tell the truth, which reshapes exactly three things.** `experiments/` → **`trials/`** (the one candidate `CONTEXT.md` already defines, giving the ladder trials → tuning → studies), split into `trials/loaders.py` + `trials/refit.py`; `make_loaders` → **`split_calibration`** returning a named `CalibrationSplit(train_loader, val_loader, recipe)` — placement ruled stay-put for the first time, and the `__init__` docstring's "no modeling logic" claim fixed, since it is the sole enforcement point of ADR-0001; `data_info` **splits into `search_space` + `training`**, making `validate_data_info`'s two allowlists the interface; **every `mc_*` alias deleted** with the rollout pair named by *mechanism* — `forecast_recurrent` / `forecast_attention`, `simulate_*_path` to match; `Inference*` → **`Rollout*`** (prefix); `mape_aggregate_style` → **`mape_aggregate`**, which closes a drift `CLAUDE.md` already had; five file renames (`training/loop.py`, `panel_dataset.py`, and the Pareto trio unified on `pareto_nbd_*`); D25's collisions settled — `study_name` → `suite_name`, pnbd's "study" → `dataset_dir`, its `(rate, churn)` `group` → `cell`, and **`max_trans` killed** in favour of `num_target_classes` / `clip_target_upper`. **Costs were measured per symbol and the ticket's own header was wrong:** they span two orders of magnitude — `max_trans` 13 notebook occurrences, `Inference*` 4 bare import names with **no call sites**, `mape_aggregate_style` 1, `make_loaders` **zero** — because notebooks import subpackages, not modules, with four exceptions. `evaluation/plot_utils.py`'s name **deferred to ticket 12** (decision 6 is still moving its contents), which also inherits the finding that two notebooks import the **private** `_pareto_from_data` from it. `CONTEXT.md` gains *Recipe* and *Customer group* (not *Cell*), appended as an amendment to ticket 08. **Three items closed as moot:** `evaluation_utils` → `metrics` names a module already retired (five carried-over items, not six), and *Registry* was already added by 08. Delivery is hybrid — each rename folds into the execution issue already touching its module, leaving **one** orphan sweep issue: **+1 net, ~13 total**, inside the tripwire.

- [Where thesis-figure code lives](issues/12-thesis-figure-code-home.md) — **the ticket's premise was wrong and the answer inverts because of it: nothing relocates to `scripts/`.** `make_grid_figures.py` imports only `collect_grid_results` and **re-implements** `dead_customer_mass()` / `shape_correlation()` itself, so the three grid functions it was said to drive are called from `Pareto_Datasets.ipynb` alone — genuinely *called*, so alive. That makes the finding a **duplication to collapse** (script's arithmetic wins, package trio becomes its notebook-facing surface, D24's 51 byte-identical lines factored once), not a misplacement. **The boundary test is replaced**: not "did it produce a thesis figure" — which would evacuate the package of `plot_suite_forecast`, its **most-called function at 16 notebook calls**, into a `scripts/` that is not importable — but **"can it run on a real panel?"**, a property readable from the code. Only the three ground-truth readers fail it, so `pnbd_grid.py` **splits**: `studies/pareto_nbd_grid.py` keeps the stored-results readers, `studies/synthetic_grid.py` takes the trio (amends ticket 09 decision 3). `_CANONICAL_GROUPS` turns out to be **the third of five encodings of one set** (D13's second half, which ticket 06 decision 10 left open), collapsing into `segment_analysis._GROUP_PREDICATES` whose keys *are* the set. **`evaluation/plot_utils.py` ceases to exist** — only four functions still have live callers: `evaluation/predictions.py` (06 decision 6's module, closing the `evaluation`/`models` cycle), `evaluation/plots.py`, and `pareto_forecast`/`_pareto_from_data` moved **out of `evaluation/`** into `benchmarks/pareto_nbd.py`, which answers ticket 09's deferred naming question and removes the last `_utils`. Two D20 private cross-boundary imports are **promoted to public** rather than relocated. `studies/analysis.py` splits three ways (`suite_reader`/`suite_plots`/`suite_metrics`) at **zero notebook cost**, justified not by size but because one 1195-line file hid both `_CANONICAL_GROUPS` and a second Student-t interval — **D22 closes here**, one implementation in `suite_metrics.py`. `forecast_from_checkpoint`/`holdout_actuals_NT` are left **conditional on ticket 10**, deliberately not absorbed. **+2 issues, ~15 total — at ticket 06's tripwire**, with the `analysis.py` split named as the cheapest thing for ticket 11 to defer.

## Not yet specified

- ~~**How much of `evaluation/` survives.**~~ **SETTLED by ticket 12.** `plot_utils.py`
  ceases to exist (three destinations, one of them outside `evaluation/`), and the group
  vocabulary collapses into `segment_analysis._GROUP_PREDICATES`.
- **Execution ordering and issue sizing.** Which cleanups are safe to land first, and
  how they carve into issues, depends on the target shape from ticket 06.

## Out of scope

- **Making the frequency-agnostic `PanelConfig` promise real.** Audits still record
  every hardcoded dataset or frequency assumption as evidence, but acting on it is a
  later effort. Ruled out by Pablo during charting: not needed yet.
- **Running the package on a genuinely new, unfamiliar panel.** The natural successor
  once the cleanup lands; premature while the code is still moving.
- **PyPI / shipping-grade hardening.** The bar is thesis defence — the audience is an
  examiner and future-you, not a third-party installer. This is what licences deletion:
  with no external consumer, an unreferenced export is dead rather than public API.
- **`notebooks/` as a cleanup target**, and `notebooks/archive/` entirely.
- **The three floor items above** (forecasting contract, benchmark arithmetic, on-disk
  formats).
- **Checkpoints and Optuna storages as a fourth floor item.** Ticket 14 proposed it after finding
  that nothing gates checkpoint reload and a constructor-signature change could silently orphan
  2572 archived files. Ruled out by Pablo on 2026-08-11: the files are expendable, so there is
  nothing to protect. Do not re-open as a gate; see the Notes for the mechanism-vs-files
  distinction that survives it.
