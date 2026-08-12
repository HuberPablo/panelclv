# Target architecture synthesis

Type: grilling
Status: resolved
Blocked by: 05

## Question

Given the ledger, what shape should `panelclv` have? This is the map's centre of
gravity: the audits gather evidence, this ticket exercises judgement, and every
remaining ticket falls out of its answer.

To settle:

- The subpackage set and what each owns. Does `experiments/` survive as its own
  subpackage, merge into `studies/`, or absorb it?
- The seams. The embedder is already one (ADR-0005). Does the rollout stepper become
  one (ticket 03)? Does model registration?
- Which of the ledger's refactor rows are actually worth doing, versus rows where the
  right answer is to leave working code alone.

Invoke `/grilling`, `/domain-modeling` and `/codebase-design`. Respect the floor in the
map's Notes — the forecasting contract, benchmark arithmetic and on-disk formats are
not negotiable.

**Tripwire:** if the target shape's execution exceeds ~15 issues, say so and cut the
destination rather than proposing a rewrite.

## Policy calls the ledger surfaced

**Both are now SETTLED.** They are kept here as recorded context, not as open questions.
Do not re-litigate either; the rulings and their consequences are in the map's Notes.

Neither can be resolved by more evidence; both change what the rest of the map may delete.

1. **SETTLED — a notebook import is NOT a caller.** Pablo ruled on 2026-08-11; the rule and the
   four rows it moves to `kill` are recorded in the map's Notes. Re-apply it against
   `ledger.csv`'s `callers_live` column for any row the list misses. Do not re-litigate. Original
   framing, for context: The four live notebooks *import* names they never call —
   audit 04 flagged `alignment_check` and `weekly_aggregate_predictions` as imported by 3-4
   notebooks and called by none. The ledger applied the strict reading (import ≠ caller) in the
   experiment lane, an ad-hoc one in the model lane, and never tested it in the data lane. **It
   decides 6 ledger rows.** Compounding it: `tests/test_notebooks_current_api.py` binds *calls*,
   not imports, so a notebook that imports a name it never calls is checked by nothing while
   reading as a live caller to a grep. Settle the rule, then the ledger's kill list is stable.

2. **SETTLED — checkpoints and Optuna storages are NOT on the floor, and the archived files may
   be deleted.** Pablo ruled on 2026-08-11. This ticket may reshape model constructors freely.
   But the checkpoint *mechanism* stays: see the map's Notes for the files-vs-mechanism split.
   Original framing, for context: The map's third floor item
   says "archived `Studies/` predictions and `results.csv` stay readable". `Studies/` also holds
   `.pth` checkpoints and Optuna DBs whose reload depends on the inference model's constructor
   arguments matching the trained model's — a `CLAUDE.md` invariant that nothing gates, that
   ticket 14's gate deliberately does not cover, and that audit 03 found is enforced nowhere.
   If those GPU-hours are worth protecting, the floor as charted is drawn too narrowly and this
   ticket's freedom to reshape model constructors shrinks accordingly. If they are not, say so
   explicitly so ticket 11 can order deletions without hedging.

   **Scale, verified:** 2572 archived `.pth` under `Studies/` (4.7 GB) plus 345 loose in
   `checkpoints/` (542 MB). Note `keep_only_best_checkpoint=True` already exists to stop these
   accumulating (`Pareto_Datasets.ipynb`, "over 100+ runs, saves a lot of disk"), so deleting the
   *files* is independent of any code decision here.

   **What is genuinely load-bearing vs genuinely unused**, verified while investigating this:
   the checkpoint *core* is the only mechanism carrying weights from training to forecasting —
   `training_utils.py:348,462` writes, `experiment_utils.py:43` reads, and `refit_best_trial`
   (the path every production run takes) reads one twice: warm-start, then reload of the refit
   it just wrote. `checkpoint_dir`, `checkpoint_path` and `keep_only_best_checkpoint` are live
   public surface across all four notebooks, so renaming any of them turns
   `tests/test_notebooks_current_api.py` red. Only three thin slices are unused:
   `forecast_from_checkpoint` (already `conditional-10` in the ledger), the three `_cached_mask`
   pops (0 of 108 checkpoints carry the key), and `prediction_source="checkpoint"` — see below.

   **`prediction_source="checkpoint"` is a 27th cross-lane duplication the ledger does not have,
   and it is a reason to decide rather than sweep.** `run_studies.py:125` and `Study.ipynb` both
   pass `"refit"`, so the `else` at `runner._rebuild_winner:173` is never taken. But the same
   choice is *also* expressed one altitude up, as the notebook boolean
   `REFIT_ON_FULL_CALIBRATION` (in `Data_integration_LSTM_v2`, `Data_integration_TRANSFORMER_v2`
   and `Study.ipynb`), whose `False` branch is documented as "forecast with the tuning checkpoint
   as-is — **the published rfm2lstm GitHub behaviour**". So deleting the suite-level knob would
   leave a documented published baseline reachable only from a notebook boolean.
   **Precision matters here:** neither `validate_*.py` gate exercises that path, so it is
   *adjacent* to floor item 2, **not protected by it** — the floor as charted does not cover it.
   Decide whether the paper's as-is baseline is something the suite must still be able to
   express, and if so collapse the two knobs into one.

Both were Pablo's calls, not the ledger's. Ticket 06 now starts with no open policy questions.

## Scope change landed after the ledger: floor item 3 is rescinded

Pablo rescinded the on-disk-format floor on 2026-08-11 — archived studies need not stay readable,
and clean code outranks archive compatibility. This *widens* what this ticket may propose:

- Suite tree, `results.csv` columns and the prediction-CSV layout are all free to change. The five
  competing prediction layouts the ledger found can collapse to one.
- The two format warts are now bugs to fix, not behaviour to preserve.
- **Decide what happens to `tests/test_archive_formats.py`.** 19 of its 24 tests drive live code
  from a synthetic fixture and are the only coverage `studies/layout.py` and `analysis.py`'s read
  path have; 4-5 depend on the real archive and two assert warts you may now fix. Recommendation:
  keep the fixture-driven tests as read-path coverage, drop the rest. Do not delete the file
  wholesale — that trades one unjustified constraint for a coverage hole.

## Answer

Settled with Pablo, 2026-08-11, over four rounds of `/grilling`. **Twelve decisions are final** — the ten numbered below, plus the two `SETTLED`
policy calls above. The closing round (Q13-Q16) is resolved at the foot of this file;
nothing is left open.

### The shape: consolidate in place, do not re-partition

**All nine subpackages stay where they are, with their current boundaries.** The ledger's
own numbers argue against re-partitioning: under 3% of the package is deletable, and the
rot is 74 `refactor` rows — one idea written out in many places, which no amount of moving
folders fixes. `experiments/` survives as its own subpackage: 311 lines, 5 functions, used
by `studies/runner.py` and three live notebooks, and folding it into `studies/` would cost
three notebook edits to buy nothing.

The budget lands at **~10-11 execution issues**, inside the ticket's ~15 tripwire.

### Decisions

1. **`rollout_composite` is removed entirely**, and **ADR-0003 is retired** — the ADR
   exists solely to document the deleted feature (ticket 08 does the paperwork). Scope:
   ~175 lines (`optuna_tuning.py:648-721` `weekly_aggregate_rollout_metrics`, `:722-824`
   `_validation_rollout_score`), the `ROLLOUT_METRIC` constant, the eleven `rollout_*`
   parameters on `objective`/`run_optuna_study`, and `selection_metric` itself — with the
   composite gone it is a parameter with one legal value. No test touches it; two notebooks
   do (`Data_integration_LSTM_v2`, `Data_integration_TRANSFORMER_v2`), so their tuning cells
   are edited in the same commit. Accepted cost, recorded once: selection on rollout quality
   is gone, so "good next-step, drifts over a long horizon" is unguarded again; restoring it
   would be a re-implementation, not a flag flip.

2. **The loss-variant cluster is KEPT WHOLE** — `FocalLoss`, `SquaredEMDLoss`,
   `weighted_ce`, `focal_gamma`, `compute_class_weights` all stay, including
   `tests/test_losses.py`. This reverses two of the ledger's 13 kills: **the kill list is
   11 rows, ~187 lines, not 13 rows / 282 lines.** C4's "kill the cluster or keep it whole"
   is resolved as keep-whole.

3. **Refit only — the "as-is checkpoint" path is removed.** `prediction_source` goes
   entirely (one legal value left) along with the three notebooks' `REFIT_ON_FULL_CALIBRATION`
   toggles. Rationale is *paper fidelity*, and it is stronger than the ticket assumed:
   Valendin et al. take the lowest-validation-loss model and "perform several 'fine-tuning'
   training epochs using the entire calibration data set" — i.e. the paper **does** refit
   (`DEFAULT_REFIT_EPOCHS = 5` already matches "several"). The notebook comment calling the
   as-is branch "the published rfm2lstm GitHub behaviour" was right about the *code* and
   wrong about the *paper*; the two disagree, and this package follows the paper. **This
   closes the ticket's 27th duplication without preserving the capability.**

4. **One registry entry per model**, holding: parameter search space, builder, inference
   builder, forecaster, **and the rollout function it requires**. The seven scattered
   enumerations (D11) derive from that table's keys. This **retires CLAUDE.md's "adding a
   model touches three places"** rather than restating it — the highest-value item in the
   ledger. Input to ticket 07.

   **Amended by ticket 07 (2026-08-12).** The entry's *inference builder* field is gone —
   `_build_inference_model_for` is deleted outright, along with `build_inference_from_trial`
   and the two notebook cells that call it. No *rollout class* field replaces it either:
   `trained.to_rollout()` puts that pairing on the training class, inside `models/` and
   `benchmarks/`. Remaining fields: **search space, builder, forecaster / rollout function.**
   **Amended by ticket 08 (2026-08-12).** Entries hold **direct** references, not lazy ones.
   The lazy design existed so `studies/config.py` could validate a `model_type` without
   importing torch; the torch-free guarantee is dropped, and it was protecting a property
   `panelclv.studies` does not have — that package already pulls torch at import. There is no
   cycle either way, since `models/` does not import `studies/`.

   Ticket 07 also fixes the entry shape — one table with optional fields, so `pareto_nbd`
   sits in it and `NEURAL_MODEL_TYPES` derives as "this entry has a training builder".
   Decision 5 is untouched: the model-to-rollout-*function* pairing is a different pairing
   from the model-to-rollout-*class* one ticket 07 settles.

5. **The model to rollout pairing is declared through the registry, not enforced by
   sealing.** The forecast entry point reads the required rollout from the registry instead
   of trusting the caller. Both rollout functions stay importable: the residual hole (a
   caller bypassing the registry) has never been hit outside an already-broken script.
   Resolves ticket 03's stepper question and C1 — option (B)'s structural argument is
   accepted, its demolished live-bug argument is not needed, and no `Stepper` abstraction
   is built for two implementations.

6. **The ADR-0002 breach is fixed; the five prediction layouts are NOT unified.**
   `save_predictions_to_csv` moves out of `evaluation/plot_utils.py` into its own
   prediction-I/O module, which removes the lazy mid-function import `models/` uses to dodge
   the circular dependency (D4). Unifying the five on-disk layouts (D6) is dropped — its
   main beneficiary was archive compatibility, which Pablo rescinded — except that the dead
   fifth layout dies with `forecast_run.py` regardless.

7. **`tests/test_archive_formats.py`: keep the 19 fixture-driven tests, drop the rest,
   relabel the file.** It stops being "the archive format is frozen" and becomes read-path
   coverage for `studies/layout.py` and `analysis.py` — the only coverage they have. The
   4-5 real-archive tests and the two asserting now-fixable warts go.

8. **Wall-clock timestamps come out of output folder names.** Three live sites put
   `datetime.now()` into a folder name (`models/monte_carlo_forecasting.py:328`,
   `data_preparation/pareto_simulation.py:287`, `evaluation/plot_utils.py:340`), so an
   output path is not derivable from config + seed — a direct hit on CLAUDE.md priority 2.
   Derive those names from config and seed and keep the timestamp as a metadata field.
   **Not all six D19 sites:** `studies/runner.py:242` writes a `"created"` metadata field
   (correct provenance, keep), `tuning/optuna_tuning.py:1141` already has `append_timestamp=False`,
   and `evaluation/forecast_run.py:79` dies with its module. Fold into the issues already
   touching each module.

9. **`scripts/validate_*.py`'s internal re-implementations are deliberate insulation —
   frozen.** Their own `WEEKS_PER_YEAR = 52`, cohort filter and `dayofyear // 7` week index
   are features, not duplication: a gate that imports the code it gates stops being a gate,
   because a future bug in a shared cohort filter would move the benchmark and its own check
   in lockstep and still pass. **C5 is resolved in favour of audit 03 / the map, against
   audit 02.** No later ticket may dedupe D3, or the D7 copies that live inside these scripts.

10. **The cut list.** Three duplications become issues because they can make a *number*
    wrong, not merely annoy:
    - **Week and period arithmetic (D7)** — 4 week-numbering conventions and 3 period-length
      tables, two disagreeing on `monthly` (30.0 vs 30.4368), *both feeding the Pareto/NBD
      fit*. Inert today, wrong by construction the day a monthly panel runs.
    - **The target column (D16)** — produced once by `prepare_dataset`, re-derived 6 more
      times; a drift scores the wrong column silently.
    - **The time-flag set (D13)** — written 4 times and **already drifted**, which is what
      produced the orphan `dayofyear` column.

    Folded in opportunistically, not given their own issues: the `id_col` two-fallback-string
    problem (D17, ~9 sites, `runner` uses both in one function), the twice-defined `DataBuilder`
    alias (D1), the Student-t interval written three times (D22), and the root `__init__`
    docstring listing 8 of 9 subpackages.

    **Explicitly left alone:** the suite-traversal pattern (D24), the 21 byte-identical lines
    shared by the two rollouts (D9 — only worth factoring if decision 5 had gone the other
    way), and D18/D25, which are ticket 09's material.

### Findings this ticket produced, which the audits and ledger do not have

- **The paper's RMSE is individual-level, and `compute_forecast_metrics` is correct.**
  Valendin et al. §3.3: *"To evaluate the predictive performance at the individual level, we
  report the Root Mean Squared Error (RMSE)"*, and `Original_paper_model/banking_transactions_demo.ipynb:1236`
  spells it out — `sqrt(mean((pred-act)**2))` over the `(N, T_HOLD)` arrays, "individual RMSE",
  with an aggregate RMSE computed separately and labelled *"for completeness"*. All three of the
  authority's metrics match the paper: `bias_percent` matches the accumulated-transactions
  definition, and `mape_aggregate_style` is algebraically identical to the paper's equation 5
  (`M = (100/n)·Σ|Aₜ−Pₜ|/Ā` = `100·Σ|Aₜ−Pₜ|/ΣAₜ`).
  **So D8/C2 resolve against the tuning code, not the authority** — and decision 1 deletes the
  offender outright, which makes CLAUDE.md's "single scoring authority" claim true for the first
  time rather than needing the carve-out C2 proposed.

- **`rollout_composite` was live, correcting the ledger's implication.** Audit 04 established
  `run_study_suite` never passes `selection_metric` — true — but `Data_integration_LSTM_v2`
  (:499, :547) and `Data_integration_TRANSFORMER_v2` (:305) call
  `run_optuna_study(selection_metric="rollout_composite")`, and a stored output shows the study
  `lstm_cross_entropy_rollout_composite_20260601_1651` really ran. The divergence sat on a path
  that had been used.

- **No archived study was ever tuned on it.** All **1256** `"selection_metric"` values recorded
  under `Studies/` read `"val_loss"`; zero composite. This is what makes decision 1 free — nothing
  stored needs re-tuning. **Caveat for ticket 11:** those notebook runs wrote to
  `checkpoints/lstm_optuna/`, not into a study, so a thesis figure sourced from a notebook run
  rather than a suite is outside what the archive can show.

- **`configs` and `data_preparation` import each other.** `configs/panel_config.py:98` imports
  `validate_ar_features` from `data_preparation`; `data_preparation/dynamic_panel_dataset.py:76`
  imports `PanelConfig` back. Both module-level, neither deferred. It does not break because the
  cycle is at *subpackage* granularity only — the module graph is acyclic (`ar_features` never
  imports `configs`) — which is why no audit caught it. `PanelConfig` is meant to be the bottom
  of the stack and imports upward. **New; not in the ledger.**

### Resolved — the closing round, 2026-08-12

Settled with Pablo over one round of `/grilling`. **Ticket 06 is closed; nothing is left
to decide.** Every claim below was verified against an AST scan of all cross-subpackage
imports, not assumed from the ledger.

- **Q13 — where the unified registry lives. ANSWER: a new subpackage, `src/panelclv/registry/`.**

  Both homes the ticket proposed are blocked by real cycles:

  - **`models/` is blocked.** The registry must name `ValendinLSTMModel` /
    `InferenceValendinLSTMModel`, which live in `benchmarks/`, while
    `benchmarks/valendin_lstm.py:60` already imports `models.embedders` at top level.
    Both arrows at once. Note `tuning` dodges exactly this today by deferring its two
    benchmark imports into function bodies (`optuna_tuning.py:510`, `:621`) — the same
    dodge decision 6 exists to remove elsewhere.
  - **`studies/` is blocked.** `tuning` is what needs the registry, and
    `studies/runner.py:41` already imports `tuning`. A registry in `studies/` gives
    `tuning → studies` alongside `studies → tuning`: a top-level cycle, and conceptually
    the thing that runs *one* study depending on the thing that runs *many*.
  - **`configs/` was raised and rejected.** `data_preparation/dynamic_panel_dataset.py:76`
    imports `configs.panel_config`, so `configs/` is the bottom of the stack; a registry
    there makes the bottom name the top.

  **Revision to the shape decision, recorded so the Answer does not contradict itself:**
  this is a **tenth subpackage**. Pablo's call, on the grounds that a loose module at
  `src/panelclv/` — which today holds only `__init__.py` beside nine folders — is
  repository clutter. "Consolidate in place, do not re-partition" still holds for the
  existing nine: none of them moves, and no boundary between them changes.

- **Q16 — the folder's internal layout. ANSWER: `registry/model_registry.py`, re-exported
  from `registry/__init__.py`.** That matches the seven of nine existing subpackages that
  re-export, and leaves room for a second table without renaming the folder. The
  `suggest_*_params` functions move into it — that is what "one entry per model"
  (decision 4) requires — leaving `tuning` importing the registry with no cycle in either
  direction. **`CONTEXT.md` has no term for this concept: input to ticket 08.**

  **Design constraint for the execution issue.** Registry entries must hold **lazy**
  references (dotted paths or factory callables), not imported classes. With direct
  references, `studies/config.py` would import torch merely to validate a `model_type`
  string. The package already has the counter-pattern — `benchmarks/__init__.py` states
  "Torch is imported lazily here".

- **Q14 — whether this ticket rules on `evaluation/`'s internal shape. ANSWER: no;
  ticket 12 decides.** Only decision 6's breach fix stays here. `evaluation/plot_utils.py`
  is 609 lines and 12 functions of which exactly **one** plots: prediction I/O (2), array
  reshaping (4), scoring (2), plotting (1), rebuild-and-forecast (3). Decision 6 removes
  the first group. The remainder cannot be sensibly organised until ticket 12 decides how
  much plotting leaves the package for `scripts/` — a tension its own text already names
  ("plotting living in `studies/` while `evaluation/plot_utils.py` (609 lines) exists for it").

- **Q15 — whether the `configs` ↔ `data_preparation` cycle gets fixed. ANSWER: yes, folded
  into decision 10**, not given its own issue. It is the one upward import at
  `configs/panel_config.py:98`; moving `validate_ar_features` (or the name list it checks)
  settles it.

### Two findings from the closing round

- **There is a SECOND subpackage-level cycle; this ticket had recorded only the first.**
  The AST scan found `evaluation ⇄ models` as well as `configs ⇄ data_preparation`.
  `evaluation/plot_utils.py:35` imports `models` at top level, and
  `models/monte_carlo_forecasting.py:324` imports `evaluation.plot_utils.save_predictions_to_csv`
  from *inside a function*, with a comment stating a top-level import "would create a
  circular import at load time". **Decision 6 already closes it** — moving
  `save_predictions_to_csv` into its own prediction-I/O module removes that deferred
  import. So decision 6 fixes a *cycle*, not merely an ADR-0002 breach; it should be
  described that way in the execution issue.

  The two cycles survive for different reasons, which is the distinction ticket 13 needs:
  `configs ⇄ data_preparation` has both legs at top level but an acyclic **module** graph
  (`ar_features.py` imports nothing from `panelclv`), so it is latent; `evaluation ⇄ models`
  closes at module level and survives only by the deferred import.

- **The torch-free guarantee is worth less than `CLAUDE.md` implies, and nothing protects it.**
  Measured on the project venv: `data_preparation` alone 0.24 s / 65 MB; `torch` alone
  1.26 s / 573 MB. And `torch` is a **hard** dependency in `pyproject.toml`, so the package
  cannot be installed without it — the four `pytest.importorskip("torch")` guards can never
  skip in a correct install. The guarantee therefore buys ~1.2 s and ~540 MB on a
  data-prep-only path; it does **not** buy the ability to run where torch is absent. Its
  real value is as a **layering rule** — data preparation must not depend on the model layer
  — for which "is torch loaded" is a cheap observable proxy. It holds today only because
  `configs/` and `data_preparation/` are the **only two** of the nine subpackages whose
  `__init__.py` does not re-export; one convenience re-export, which seven siblings already
  have, breaks it silently. **Either restate it in `CLAUDE.md` as a layering rule (ticket 08)
  and add the one-line test (ticket 13), or drop it as not worth its weight.**
