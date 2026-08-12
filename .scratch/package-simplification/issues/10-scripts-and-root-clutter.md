# Rule on scripts/ and the tracked root clutter

Type: grilling
Status: resolved
Blocked by: 05

## Question

`scripts/` splits three ways and each group has a different fate:

- **One-shot, already applied** — `migrations/relabel_archived_pareto_mle.py`,
  `migrations/rename_embedder_checkpoint_keys.py`, `verify_transformer_training.py`
  ("one-off check" by its own docstring). Do applied migrations stay as provenance or go?
- **Thesis-reproducibility, protected by the carve-out** — `make_grid_figures.py`,
  `recheck_season_churn.py`, and the two `validate_*.py`, which are the executable
  definition of the benchmark-fidelity floor.
- **Live entry points** — `run_studies.py`, `main_plot.py`, and *nominally*
  `main_plot_covar.py`. Note these are the only callers keeping `forecast_from_checkpoint`
  and `holdout_actuals_NT` alive, so their fate decides those symbols' fate.

  **Corrected by audit 03, and it sharpens this ticket:** `main_plot_covar.py` does not
  execute. It passes six keyword arguments `forecast_from_checkpoint` does not accept
  (`calibration`, `holdout_calendar`, `seq_cols`, `target_col`, `model_type`, `batch_size`)
  and reads a `result["predictions"]` key nothing returns, so it raises `TypeError` on
  invocation. So it keeps nothing alive, and `forecast_from_checkpoint` /
  `holdout_actuals_NT` rest on `main_plot.py` alone. Note the dropped `model_type=` argument
  is the remains of a stepper dispatch that once existed — see audit 03's ruling on the
  rollout-stepper seam, which is what a fix would have to restore. Rule on it as a fourth
  group: **broken, decide whether the covariate-comparison figure is still wanted.** If it
  is dropped, ticket 06 gains the freedom to delete more of `evaluation/`.

  **Audit 04: `main_plot.py` is broken too**, in the complementary way. Line 244 takes
  `weekly_actuals`' 1-D `(T_HOLD,)` aggregate and line 251 hands it to `metrics_table`, which
  raises when `actuals.ndim != 2`. The two scripts are diverged copies of one file
  (`main_plot.py:1-3` calls itself "counterpart to the old `main_plot.py`") and
  `main_plot_covar.py:171-188` is the *fixed* form of the exact call `main_plot.py` gets wrong —
  so each holds the other's fix. Their shared `_load_dataset` stubs both point at
  `scripts/data/data_loader.py`, which does not exist. So **no runnable entry point calls
  `forecast_from_checkpoint`, `holdout_actuals_NT` or `weekly_actuals`**, and the shape those
  last two consume (`Sequence[pd.DataFrame]`) has no producer in `src/` any more. Fixing one and
  dropping the other is likely the shape; whichever survives needs the stepper dispatch audit 03
  found missing.

Then the tracked root clutter, which is small and cheap to sweep: `figures/`,
`inputs_configs/`, `VastAI/`, `archive/`, `.agents/`, `Package_Notebook.code-workspace`,
`instruction`, `PUBLISHING.md`. Keep, move or delete — one line each. (Untracked,
gitignored dirs like `Datasets/`, `Studies/` and `checkpoints/` are not package concerns.)

## Answer

Settled with Pablo over two rounds of `/grilling`, 2026-08-12. **Nine decisions, nothing left
open.** This ticket also resolves the conditional ticket 12 left hanging, and fixes the one
ordering constraint the map does not otherwise state.

### Corrections to this ticket's inventory

- **Two of the eight root-clutter items are moot.** `instruction` and
  `Package_Notebook.code-workspace` are **already gitignored** — `.gitignore` ends with
  `*.code-workspace` and `/instruction` — and neither is tracked. The ticket lists them under
  "tracked root clutter"; they are neither.
- **The list badly understates `.agents/`.** It is **74 tracked files / 608 KB** — more tracked
  files than `src/` (32) — a vendored copy of `mattpocock/skills`, paired with `skills-lock.json`
  and **26 symlinks** in `.claude/skills/` pointing into it. It is the largest block of tracked
  non-package content in the repo, and the ticket gives it one bullet among eight.
- **`figures/` contains no covariate-comparison figure.** Only `fig1_no_death_state` and
  `fig2_seasonal_shape` (PDF + PNG) plus four source CSVs — the outputs of `make_grid_figures.py`
  and `recheck_season_churn.py`. This is decisive for `main_plot_covar.py`.
- **Neither plot script has ever produced a tracked output.** Both write to `Predictions/`,
  which is gitignored.
- **`inputs_configs/` has zero references** anywhere in code, docs or notebooks.

### Decisions

1. **Both plot scripts are deleted — `main_plot.py` (278 lines) and `main_plot_covar.py`
   (212).** The ticket proposed fixing one, since "each holds the other's fix". Against that,
   three measured facts: their job — plot weekly aggregate actuals against each model — is what
   `studies.plot_suite_forecast` does, and that is **called 16 times** across the notebooks; the
   covariate comparison is equally live but done elsewhere, with `removable_features` appearing
   **16 times** in the notebooks and driving the search in `Data_integration_TRANSFORMER_v2`;
   and neither script has a tracked figure to its name. The thesis carve-out protects code that
   produced a published result — neither did. "Fix" was also not a one-line repair: whichever
   survived needed the stepper dispatch audit 03 found missing.

   **This resolves the conditional ticket 12 left open.** `forecast_from_checkpoint` and
   `holdout_actuals_NT` die with these scripts, so ticket 12's `evaluation/` reshape has no
   pending branch — it is the three-file shape, full stop. (`weekly_actuals` was already killed
   by the map's import-is-not-a-caller rule, so those two are the only new deaths.)

2. **The three one-shots are deleted** — `verify_transformer_training.py` (109 lines) and both
   `migrations/` (200 lines). Each has done its job: `relabel_archived_pareto_mle.py` is applied
   and the archive reads `ParetoNBD_MLE`; `rename_embedder_checkpoint_keys.py` lost its last
   purpose when the 2917 archived `.pth` were deleted (map Notes); `verify_transformer_training.py`
   calls itself a "one-off check" and answered its question. Git history is the provenance — an
   applied migration kept as a file reads as one you might still need to run.

   **One fact is relocated rather than lost:** *why* the archive says `ParetoNBD_MLE` moves into
   `archive/README.md`, which exists for exactly this purpose and which
   `make_grid_figures.py` depends on being true.

3. **`trace_golden_reachability.py` (336 lines) is kept, and is more useful now than during the
   map.** It is documented at `docs/running-a-model.md:206` as running the golden test's exact
   function under a tracer "so the reachability evidence and the pinned test describe one code
   path". Ticket 11 is about to move, rename, split and delete across nine subpackages; a tool
   answering "what does a real run actually touch?" is the check that nothing was orphaned.
   **Being documented is what separates it from decision 2's one-shots** — that is the line
   between a tool and a leftover.

4. **`.agents/` + `skills-lock.json` + the 26 `.claude/skills/` symlinks stay vendored.** The
   symlinks break the moment `.agents/` is absent, so gitignoring it leaves a repo that is
   broken-on-clone until a re-vendor step runs, and the lockfile's value is that it pins content
   you actually have. **The counter-argument is recorded rather than dismissed:** it is the
   largest tracked non-package block in a repo whose stated bar is thesis defence, and it is not
   Pablo's code. If the repo is handed to an examiner, this is the one item where "not mine, not
   the thesis" is the strongest case on the list. Revisit then, not now.

5. **The root, one line each.**

   | entry | ruling | why |
   |---|---|---|
   | `figures/` (8 files) | **keep** | the published fig1/fig2 and their 4 source CSVs — thesis evidence, and the only tracked output of any script |
   | `archive/` (2 files) | **keep** | `pareto_nbd.py` + a README stating its role; `benchmarks/__init__.py` points at it, and decision 2 gives it one more fact to hold |
   | `VastAI/` (5 files) | **keep, plus one line in `README.md`** | GPU-rental launch scripts, referenced by no doc but modified in the working tree, so live — it looks like clutter only because nothing says what it is |
   | `inputs_configs/` (1 file) | **delete** | `full_transactions_gender.json`, zero references anywhere; audit 03 already searched it and found nothing |
   | `PUBLISHING.md` | **delete** | copy-paste PyPI steps for a path the map lists as **explicitly out of scope**; keeping it invites someone to follow it |

6. **`.Rhistory` goes into `.gitignore`, and the tracked `.claude/.Rhistory` removal is
   finished.** R rewrites these on exit, so without the ignore rule they will keep reappearing —
   one has already been committed once and a second is queued at the repo root.

7. **One line in `CLAUDE.md` on what earns a slot in `scripts/`.** Decision 2 deletes 409 lines
   of scripts that were correct when written and became clutter by succeeding; nothing prevents
   the next three. Proposed wording:

   > A script in `scripts/` is either a live entry point, a benchmark gate, or a documented
   > tool. A one-off check goes in the commit that needed it and is deleted with it.

   `CLAUDE.md` is read by every agent working here, which a `scripts/README.md` would not be,
   and one line is proportionate.

8. **Budget: this ticket folds to +0, and the tripwire now binds on ticket 13, not on this
   one.** Running total is ~14-15 (ticket 06 ~10-11, +1 from 07, +1 from 09, +2 from 12), which
   is ticket 06's ~15 tripwire exactly. This ticket's work is deletions plus two
   `.gitignore`/`README` lines and touches no `src/` code except decision 1's two symbol
   deletions, which fold into ticket 12's `evaluation/` reshape issue. Charging it an issue
   would charge for the wrong thing: it removes ~1000 lines and adds no structural work.

   **The consequence ticket 13 inherits, stated plainly so it is not discovered late:
   roughly one issue of headroom.** Its answer is therefore very likely "widen the net around
   *one* surface, not four" — and audit 03's Transformer-rollout test is the one with a stated
   prerequisite argument, since the Transformer + recurrent-stepper crossing fails silently
   rather than raising. If ticket 13 needs more, ticket 12 already nominated its `analysis.py`
   three-way split as the cheapest thing to drop.

9. **Ordering: this sweep lands FIRST, before any structural issue.** The deleted scripts import
   `forecast_from_checkpoint`, `ProjectedEmbedder`, `InferenceMultinomialLSTMModel`,
   `InferenceMultinomialTransformerModel`, `metrics_table`, `plot_weekly_aggregated`,
   `save_predictions_to_csv`, `load_predictions_from_csv` and `weekly_actuals` — names spread
   across `models/`, `evaluation/` and `benchmarks/`, three of the subpackages ticket 11 is
   about to reshape and ticket 09 is about to rename. Landing the deletions last would force
   every issue in between to keep two broken scripts compiling against moving targets. It also
   means the `evaluation/` reshape never has to decide what to do with `forecast_from_checkpoint`,
   because it will already be gone.

   **This is the one ordering constraint in the map.** Ticket 11 cuts the issue *set*; nothing
   else fixes an order, so this is recorded as a constraint on it rather than a preference.
