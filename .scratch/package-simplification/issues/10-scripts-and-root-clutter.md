# Rule on scripts/ and the tracked root clutter

Type: grilling
Status: open
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
