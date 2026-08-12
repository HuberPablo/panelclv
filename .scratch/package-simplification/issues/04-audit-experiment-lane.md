# Audit: the experiment lane

Type: task
Status: resolved

## Question

Read `tuning/` (1240 lines), `experiments/` (311), `studies/` (2212) and `evaluation/`
(920) — the largest lane, and where the most rot is expected. Report against the four
shared dimensions (ticket 02).

Known inputs, already established, that this audit should build on rather than
rediscover:

- **`experiments/` vs `studies/` is a layering, not a duplication.** `experiments/` is
  plumbing for *one* run (loaders, the Optuna data closure, rebuilding the winning
  trial, warm-start refit); `studies/` runs *many* studies across *many* models and
  archives them. `studies/runner.py` calls `experiments`. The problem is that the two
  names are near-synonyms and `CONTEXT.md` defines *Trial*, *Study* and *Study suite*
  but never *experiment*. Rule on whether the layering is right; naming goes to 09.
- **`studies/analysis.py` (1192 lines) splits three ways**: genuine read-side library
  (~350 lines), figure scaffolding (~430, including a 204-line `plot_suite_forecast`
  and a hardcoded `_CANONICAL_GROUPS = ("At Risk", "Opportunity")`), and notebook
  convenience printing (~120, `describe_dataset` / `describe_suite_dataset`). Where the
  figure code lives is ticket 12; this audit rules on the rest.
- **Metric vocabulary disagrees inside one subpackage**: `pnbd_grid._METRIC_SOURCE`
  maps `"mape"` to `"mape_aggregate_style"`, which `analysis._STUDY_METRIC_COLS` uses
  directly. `CLAUDE.md` names `compute_forecast_metrics` the single scoring authority —
  check that everything actually delegates to it.

Report is evidence, not decisions. Do not change code.

## Answer

**Scope read in full:** `tuning/optuna_tuning.py` (1240), `experiments/experiment_utils.py`
(311), `studies/analysis.py` (1195), `studies/pnbd_grid.py` (623), `studies/runner.py`
(295), `studies/config.py` (175), `studies/layout.py` (121), `evaluation/plot_utils.py`
(609), `evaluation/segment_analysis.py` (200), `evaluation/forecast_run.py` (111) — 4880
lines plus the four `__init__.py` files (176 more).

**Method.** Enumerated every module-level symbol by AST (146 of them) and grepped each
against three populations kept separate — `src/panelclv/`, the live entry points, and
`tests/` — plus the non-entry scripts as a fourth. Notebooks were parsed as JSON and
scanned by AST, distinguishing *imported* from *called* and reading stored outputs. Every
dict a lane function returns was checked key-by-key for readers. Behavioural claims were
run, not read: the metric-authority comparison, `study_metrics` against an archived
`results.csv`, `StudySuiteConfig.validate()`'s coverage, the import-cost of the lazy-import
scheme, and both `main_plot*` failures.

**Entry-point correction applied.** The briefing's live-entry-point list is smaller than
stated. `scripts/main_plot_covar.py` raises `TypeError` (confirmed independently, §1), and
`scripts/main_plot.py` **also cannot run** — a break the audit found and verified (§1). So
five `evaluation/` symbols the map counted as reachable have no runnable caller at all.

**Headline.** Unlike the data lane, this lane *does* contain dead code: `ForecastRun` (111
lines, 6 public methods, zero callers) and `group_metrics_suite_distribution` (76 lines,
the only `studies/__init__` export nothing imports). Beyond that the rot is four-shaped.
**(a) The read side of `evaluation/` has rotted out from under two broken scripts** — five
public symbols (`forecast_from_checkpoint`, `holdout_actuals_NT`, `weekly_actuals`,
`alignment_check`, `weekly_aggregate_predictions`) have no runnable caller, and two of them
consume a per-customer-DataFrame shape nothing in the package has produced since
`prepare_dataset`. **(b) The production path cannot reach two of the package's own
decisions**: `run_study_suite` passes neither `selection_metric` nor `removable_features`,
so ADR-0003's rollout selection and the entire covariate-subset search are unreachable from
`scripts/run_studies.py` and from every suite cell in the notebooks. **(c) Four on-disk
prediction layouts, three copies of the Student-t interval, two period-length tables that
disagree on a month, five encodings of the two-customer-group set, and a 55-line block
duplicated with four lines changed.** **(d) `compute_forecast_metrics` is *not* the single
scoring authority** — `tuning` recomputes all three numbers itself and two of the three
come out different (§ verdict). Add to that: 3389 of the lane's 4880 lines have no
dedicated test, `pnbd_grid.py` (623 lines, behind thesis grid figures) has none at all, and
the careful lazy-import scheme in `analysis.py` buys nothing (measured).

---

### 1. Unreachable

**Two things are dead by the map's rule.**

| symbol | lines | callers anywhere |
| --- | --- | --- |
| `evaluation/forecast_run.ForecastRun` (+ `.new/.open/.path/.save_config/.save_predictions/.predictions`, `_slug`) | `forecast_run.py` 1-111 | **none.** Grepping `ForecastRun` over `src/`, `scripts/`, `tests/` and all notebooks (live and archived) returns only its own module and `evaluation/__init__.py:33,49`. |
| `studies/analysis.group_metrics_suite_distribution` | `analysis.py:701-776` | **none.** Exported at `studies/__init__.py:35`; no importer in `src/`, no entry point, no test. The only `studies/__init__` export with zero importers. |

`ForecastRun` also defines a **fifth on-disk prediction layout**
(`<root>/<config_name>/<n>/manifest.json` + `<slug>/predictions.csv`) that no writer
produces and no reader consumes. `find . -name manifest.json` over the repo returns
nothing, so no archived study is in that format — it is not protected by the on-disk floor.

`group_metrics_suite_distribution` is the function that would satisfy the "report metrics as
a distribution across studies" convention; it works (I ran it against an archived suite and
it returned a correct `(group, model) × (metric, stat)` frame), it is simply never called.
Note `group_metrics_suite_table` — the point-estimate sibling — *is* called twice from
`Study.ipynb`.

**Two "live entry points" cannot run, and five `evaluation/` symbols rest on them alone.**

- `scripts/main_plot_covar.py:100-111` calls `forecast_from_checkpoint` with six keywords it
  does not accept (`calibration`, `holdout_calendar`, `seq_cols`, `target_col`,
  `model_type`, `batch_size`) against `(checkpoint_path, inference_model_factory, data,
  n_simulations=30, device=None)`, then reads `result["predictions"]` (`:113`) — a key
  nothing returns. Verified: `TypeError: forecast_from_checkpoint() got an unexpected
  keyword argument 'calibration'`.
- **`scripts/main_plot.py` is broken too, differently.** `main()` builds
  `actuals = weekly_actuals(holdout, count_col=count_col)` at `:244` — the **aggregated
  `(T_HOLD,)` vector** — and feeds it to `metrics_table(actuals, predictions)` at `:251`.
  `metrics_table` rejects anything not 2-D (`plot_utils.py:489-495`). Verified:
  `ValueError: actuals must be (N, T_HOLD); got shape (26,). If you have the aggregated
  (T_HOLD,) vector, use holdout_actuals_NT(...) instead of weekly_actuals(...)` — the error
  message names the exact mistake the script makes. `main_plot_covar.py:171-188` is the
  *fixed* version of this same call (it uses `holdout_actuals_NT` and comments on why), so
  the fix landed in one script and not the other, and then that script broke elsewhere.
  Separately, both scripts' `__main__` blocks call a `_load_dataset` stub that raises
  `NotImplementedError` (`main_plot.py:264-272`, `main_plot_covar.py:201-208`), and
  `scripts/data/data_loader.py` — the loader its docstring points at — does not exist.

Consequently, with no runnable caller anywhere:

| symbol | only call sites | status |
| --- | --- | --- |
| `plot_utils.forecast_from_checkpoint` (`:586-609`) | `main_plot.py:119,155` (broken script), `main_plot_covar.py:100` (broken call) | no runnable caller |
| `plot_utils.holdout_actuals_NT` (`:125-170`) | `main_plot_covar.py:173` only | no runnable caller |
| `plot_utils.weekly_actuals` (`:173-191`) | `main_plot.py:244-245`, `main_plot_covar.py:178` | no runnable caller |
| `plot_utils.alignment_check` (`:526-578`) | — | **imported by 3 live notebooks, called by none** |
| `plot_utils.weekly_aggregate_predictions` (`:194-227`) | `plot_weekly_aggregated:417`, `alignment_check:551` (both intra-module) | **imported by 4 live notebooks, called by none** |

`holdout_actuals_NT` and `weekly_actuals` are worse than caller-less: their parameter type is
`Sequence[pd.DataFrame]` — a list of per-customer frames — and **nothing in
`src/panelclv/` produces that shape.** `prepare_dataset` returns `(N,T,F)` ndarrays and a
single `holdout_panel` DataFrame. Even the archived notebooks
(`notebooks/archive/Data_integration_LSTM.ipynb`, `august test.ipynb`) only *import*
`weekly_actuals`; neither calls it. These two functions were orphaned by a data-format
change, not merely by caller absence.

**Notebook imports are keeping names alive that nothing calls.** Parsed by AST across the
four live notebooks, imported-but-never-called: `weekly_actuals` (LSTM, Study),
`alignment_check` (LSTM, Study), `save_predictions_to_csv` (LSTM, Study),
`weekly_aggregate_predictions` (all four), `metrics_table` (TRANSFORMER),
`describe_dataset` (Study, imported 3×, called 0×). `tests/test_notebooks_current_api.py`
binds *calls* against signatures, so an import with no call is checked by nothing and yet
reads as a live caller to a grep.

**Public but with no caller outside its own module** — over-exposure, for ticket 05's
ledger rather than as kill candidates:

| symbol | intra-module caller | external callers |
| --- | --- | --- |
| `optuna_tuning.objective` (`:825`) | `run_optuna_study:1174` | none (not even in `tuning/__init__`) |
| `.validate_removable_features` (`:135`) | `run_optuna_study:1128` | `tuning/__init__` re-export only |
| `.validate_data_info` (`:416`) | `run_optuna_study:1120` | re-export; tests only |
| `.weekly_aggregate_rollout_metrics` (`:648`) | `_validation_rollout_score:815` | re-export only |
| `.suggest_covariate_selection` (`:162`) | `objective:860` | none |
| `.suggest_lstm_params` / `.suggest_valendin_params` / `.suggest_transformer_params` | `_SUGGESTERS` table | none (documented extension point per CLAUDE.md) |
| `experiment_utils.make_refit_loader` (`:137`) | `refit_best_trial:290` | none |
| `studies/config.VALID_MODEL_TYPES` (`:31`) | `validate:153` | tests only (CLAUDE.md registration point) |
| `.VALID_PREDICTION_SOURCES` (`:32`), `.REQUIRED_DATA_KEYS` (`:37`) | `validate` | none |
| `studies/layout.jsonify` (`:81`) | `write_json:120` | tests only |
| `studies/analysis.describe_dataset` (`:1071`) | `describe_suite_dataset:1195` | notebook import, never called |
| `evaluation/segment_analysis.aggregate_bias` (`:39`) | `group_metrics_table:191` | none — its own docstring says "Kept here because this is its only caller" |

Of `tuning/__init__.py`'s six exports, only `run_optuna_study` is imported through the
package path by anything; `experiments` reaches `select_features` /
`select_features_for_trial` via the *module* path (`optuna_tuning`), so the other five
re-exports have no consumer.

**A private name reached across a subpackage boundary from two live notebooks.**
`plot_utils._pareto_from_data` (`:239`) is imported directly by
`notebooks/Data_integration_LSTM_v2.ipynb` cell 26 and `notebooks/Study.ipynb` cell 39, both
with the comment "*`_pareto_from_data` is the internal benchmark fitter `metrics_table`
uses; we call it directly here just to get the benchmark's total*". A live caller reaching
through a leading underscore because the public surface offers no "fit the benchmark and
give me the array" — the same seam-in-the-wrong-place shape ticket 02 found for
`_seasonal_weekly_multiplier`, and the third instance of it in the package (the second is
`experiments` importing `optuna_tuning._build_model_for` / `._build_inference_model_for`,
§ rulings).

**Return-dict keys with no reader** (the ticket-02 prong):

| producer | key | readers |
| --- | --- | --- |
| `layout.model_dirs` (`:54-68`) | `"predictions_dir"`, `"optuna_dir"` | **none** outside `tests/test_studies_layout.py:46`. `runner.py:92-93, 186-187` read only `"model_dir"`; the dict's real job is the `mkdir` side effect. |
| `weekly_aggregate_rollout_metrics` (`:713-719`) | `"rollout_mae"` | **none.** Computed at `:685`, returned, written as a trial user-attr — not in the composite, not read by any code. |
| `pnbd_grid._mean_ci` (`:485-497`) | `"std"`, `"n"` | **none.** `group_summary` spreads them into the frame; `compare_models_table` and `plot_pattern` read only `mean`/`ci_low`/`ci_high`. |
| `plot_utils.pareto_forecast` (`:326-329`) | `"actual"` | **none.** `analysis:538` reads `prediction_mean`; `Study.ipynb` cell 32 reads `predictions_path`. (Deliberate: it mirrors `mc_forecast`'s contract.) |
| `runner._suite_record` (`:229-273`) | 11 of 13 top-level fields; 7 of 7 `data_summary` fields | Only `models[].name` and `panel_config` are read (`analysis:71,254`). `_id_col:100` reads `data_summary["id_col"]`, **a key `_suite_record` never writes.** |
| `runner._model_record` (`:276-295`) | all but `model_type` | Only `analysis._is_deterministic_model:126` reads it. |
| `runner`'s per-model `metrics.csv` (`:157, 225`) | whole file | **no reader** in `src/` or any entry point. `results.csv` is its superset (verified: same rows, unioned columns). Only `scripts/migrations/relabel_archived_pareto_mle.py` touches it. |
| `run_optuna_study`'s `<name>_trials.csv` / `<name>_best.json` (`:1188, 1202`) | whole files | no programmatic reader; `analysis.py`'s docstring states it deliberately never reads them. So `selected_features` / `dropped_features` / `rollout_*` user-attrs have a human as their only consumer. |

**Two decisions the production path cannot reach.**

- **ADR-0003's rollout selection is unreachable from `run_study_suite`.**
  `runner._run_neural_model:112-126` passes `model_type, data_builder, data_info, n_trials,
  device, study_name, append_timestamp, summary_dir, sampler, keep_only_best_checkpoint` —
  no `selection_metric`, and `StudySuiteConfig` (`config.py:120-131`) has no field for one.
  So `scripts/run_studies.py`, `Pareto_Datasets.ipynb` and `Study.ipynb`'s suite cells all
  run `selection_metric="val_loss"`. The only callers of `"rollout_composite"` are three
  `run_optuna_study` cells in `Data_integration_LSTM_v2.ipynb` (#17, #19) and
  `Data_integration_TRANSFORMER_v2.ipynb` (#15). Everything downstream of it —
  `_validation_rollout_score` (95 lines), `weekly_aggregate_rollout_metrics` (71 lines),
  `ROLLOUT_METRIC`, the horizon validation and warm-up warning at `:1074-1102`, and the
  nine `rollout_*` parameters — is reachable only from those cells.
- **The covariate-subset search is unreachable from `run_study_suite` too.** `runner` never
  passes `removable_features`, and `StudySuiteConfig` has no field for it. So
  `suggest_covariate_selection`, `validate_removable_features` and half of
  `select_features`' stated purpose ("Optuna proposes a feature subset") are reachable only
  from the same three notebook cells. `Study.ipynb`'s own `run_optuna_study` calls (#21,
  #23) pass no `removable_features` either.

**Dead branches inside reachable functions.**

- `select_features:240` — `embedded_cols=kept_embedded if kept_embedded else None`. The
  target column is never droppable (`:204-205`) and CLAUDE.md's invariant puts the target in
  `embedded_cols`, so `kept_embedded` is never empty. The `None` branch cannot fire.
- `analysis.plot_suite_forecast:582` passes `data=data` to `plot_weekly_aggregated`, which
  reads `data` only when `pareto_benchmark=True` — which `plot_suite_forecast` never sets
  (it pre-computes the benchmark itself at `:536-539`). A dead argument.
- `runner._rebuild_winner:173` — the `prediction_source="checkpoint"` branch. Every caller
  in the repo passes `"refit"` explicitly (`run_studies.py`, `Pareto_Datasets.ipynb` ×2,
  `Study.ipynb`). `build_inference_from_trial` stays alive through the notebooks and
  `refit_best_trial:309`, not through this branch.
- `_reduce_to_customer_week:49-52` and `weekly_aggregate_predictions:210-221` each dispatch
  on three prediction shapes. `mc_forecast` returns `simulations` as `(S, N, T_HOLD)`
  (`monte_carlo_forecasting.py:404`) and `prediction_mean` as `(N, T_HOLD)`, so `(S,N,T,1)`
  and `(N,T,1)` have **no producer in the package.** The 4-D branch is reached only because
  `Study.ipynb` #37 and `Data_integration_LSTM_v2.ipynb` #24 append a dummy axis by hand,
  commented "*Add a trailing 1-axis so `weekly_aggregate_predictions` takes the (S, N, T, 1)
  branch and draws the 95% MC ribbon*" — the caller compensating for the dispatch.

**`StudySuiteConfig.validate()` does not cover the path it guards.** `REQUIRED_DATA_KEYS`
(`config.py:37`) is `("ids","holdout","target_idx","train_panel","T_HOLD")`; two of those
(`train_panel`, `T_HOLD`) are used only by the Pareto path (`runner:200-204`). Verified: a
dict with exactly those five keys passes `validate()`, then dies at
`make_loaders` → `_require_val_start_idx` with `KeyError: data['val_start_idx'] is
missing`. A neural-only suite is validated against two keys it never reads and not against
`val_start_idx` / `samples` / `targets` / `seq_cols` / `embedded_cols`, which it does — the
exact "fail loudly before any training" promise the docstring makes (`config.py:18-19`).

**Thesis carve-out — what it protects here.** Stored PNG outputs in the live notebooks:
`plot_suite_forecast` produced 11 figures (`Study.ipynb` cells 18, 53, 54, 57-61, 65, 69,
70); `plot_weekly_aggregated` 2 (Study #37, LSTM #24); `pnbd_grid`'s plots 6
(`Pareto_Datasets.ipynb` #9, #18). All are alive by callers anyway. The carve-out does real
work for `holdout_actuals_NT` / `weekly_actuals` / `forecast_from_checkpoint`, whose only
historical consumers are the two broken scripts and `notebooks/archive/` — and note their
input shape no longer exists, so whatever figures they produced are **not currently
reproducible** whether or not the code stays. That is ticket 12's call, not this audit's.

---

### 2. Duplicated

**Five on-disk prediction layouts, one dead.** Same artefact — per-customer holdout means
as a wide CSV — written five different ways:

| writer | layout |
| --- | --- |
| `layout.prediction_path` (`:76-78`) | `<Model>/Predictions/Prediction_{i}.csv` (the archive format; on the floor) |
| `analysis.aggregate_suite_predictions` (`:200-221`) | `<suite root>/aggregated_<Model>.csv`, flat |
| `monte_carlo_forecasting._save_predictions_run` (`:296-341`) | `{tag}_n{n}_seed{seed}_{YYYYMMDD_HHMMSS}/predictions.csv` |
| `plot_utils.pareto_forecast` (`:331-348`) | `{tag}_{YYYYMMDD_HHMMSS}/{file_name}` — the same idea, a *different* subfolder scheme |
| `ForecastRun` (`forecast_run.py`) | `<config_name>/<n>/<slug>/predictions.csv` + `manifest.json` — **no writer, no reader** |

All five funnel through the same `save_predictions_to_csv`; only the naming differs. The
third and fourth exist side by side for the neural and the Pareto forecaster respectively,
whose contracts are otherwise deliberately mirrored.

**Three copies of the Student-t interval on the mean**, algebraically identical, written
three ways:

| site | expression |
| --- | --- |
| `analysis._across_study_band:403-404` | `std(ddof=1)`; `stats.t.ppf(1 - (1-ci)/2, df=n-1) * std/√n` |
| `analysis._study_metrics_from_data:918-932` | same, pandas-shaped, with a per-model `tcrit` Series |
| `pnbd_grid._mean_ci:495-496` | `stats.t.ppf(0.5 + ci/2.0, n-1) * std/√n` |

`1 - (1-ci)/2 == 0.5 + ci/2`. Three implementations of "across-replicate CI", two of them in
one file.

**Two period-length tables that disagree.**
`runner._PERIOD_DAYS = {"daily":1.0,"weekly":7.0,"monthly":30.0}` (`:58`) vs
`plot_utils._PERIOD_IN_DAYS = {"weekly":7.0,"monthly":30.4368,"daily":1.0}` (`:236`). Same
concept, same purpose (the Pareto/NBD RFM time scale), **different value for `monthly`** —
30.0 vs 30.4368. Both feed `compute_pareto_predictions(period_in_days=...)`, so a monthly
panel scored through the suite and through the plot helper would get two different Pareto/NBD
fits. Inert today (`monthly` reaches neither path), latent by construction. A third copy sits
in `main_plot.py:170` as a `period_in_days: float = 7.0` default.

**Five encodings of the two-customer-group set.**
`segment_analysis._GROUP_PREDICATES` (`:63-66`, the definition),
`assign_customer_groups`' default (`:121`), `analysis._CANONICAL_GROUPS` (`:285`),
`group_metrics_suite_table`'s `groups=("At Risk","Opportunity")` default (`:645`), and
`group_metrics_suite_distribution`'s (`:706`). Same shape as ticket 02's four encodings of
the time-flag set. The `"Other"` catch-all is then derived at three separate call sites
(`analysis:360, 696, 745`), each doing `{**group_ids, "Other": _other_ids(...)}`.

**Two customer-id → row-index resolvers.** `analysis._rows_for_ids` (`:299-320`) dedups and
preserves order; `segment_analysis._resolve_rows` (`:82-93`) does not dedup. Both key on
`str(cid)`, both raise on missing ids, with different messages. One concept, two
implementations, one lane.

**Two incompatible policies for prediction/actual misalignment.**
`segment_analysis._load_aligned` (`:96-112`) **reorders** a prediction file's rows to
`data["ids"]`; `analysis._study_metrics_from_data:894-898` and
`plot_suite_forecast:514-519` **refuse**, raising `ValueError` when the id arrays are not
`array_equal`. So the group table silently repairs an ordering the metrics table treats as
fatal — and both are called on the same files by the same notebook.

**Two ways to find the target channel.** `data["target_idx"]` (`analysis:269, 884, 1100`;
`runner:215`) vs `list(data["seq_cols"]).index(data["target_col"])`
(`plot_utils.pareto_forecast:323`; `segment_analysis._counts:75`; and in the notebooks,
`Study.ipynb` #37 / `Data_integration_LSTM_v2.ipynb` #24 recompute it by hand rather than
reading the key).

**Two model-discovery implementations.** `analysis._discover_models` (`:57-88`) orders
models from `config.json` and skips empty `Predictions/` folders, explicitly so the legend is
reproducible rather than filesystem-ordered. `pnbd_grid` instead does `sorted(p for p in
suite.iterdir() if p.is_dir())` three times (`:236, 335, 438`) — filesystem order, no
config, and it would pick up any non-model subdirectory.

**A 55-line block duplicated with four lines changed.** `alive_volume_ratio_grid:319-373`
and `dead_volume_leakage_grid:422-477` are byte-identical for 51 of 55 lines (verified by
`diff`); the divergence is `alive_week = weeks < tau_i` / `p_alive` versus
`dead_week = weeks >= tau_i` / `leaked`, plus the output column name. The `oracle_alive`
block (`:351-352` vs `:454-455`) is identical. Their own docstrings say
`R_A + L_D = total predicted / total oracle` — i.e. the two numbers come from one
decomposition — yet each makes a full independent pass over every dataset, reloading the
panels, the ground truth and every prediction file. `seasonality_grid:218-268` repeats the
same outer scaffold a third time.

**Four re-globs of the same prediction files.** `sorted(preds.glob("Prediction_*.csv"),
key=_prediction_index)` at `analysis:170` and `:392`, and
`sorted(_prediction_index(p) for p in preds.glob(...))` at `:756` and `:892`. Each analysis
entry point re-enumerates and re-parses every CSV: `study_metrics`,
`group_metrics_suite_distribution`, `_across_study_band` and `load_model_predictions` all do
it separately, so a `plot_suite_forecast(confidence_interval=True)` call reads each file
three times.

**The same numbers are produced twice, by two paths.** `runner` computes
`mc_compute_metrics` at train time and writes `results.csv` (`:144, 216`);
`analysis._study_metrics_from_data` **recomputes** them at read time from the stored
predictions. I verified they agree exactly — on
`Studies/cross_entropy_cfg_2yTrain_1yPred_NoCov_TestDimanche`, `study_metrics` reproduced all
nine stored `results.csv` values to ~13 significant digits (rel=1e-12 — *corrected by ticket 14*, which pinned the numbers: LSTM `bias_percent` recomputes to -53.40695337290191 against a stored -53.40695337290179, so this is not bit-exact). Correct, but it means the archive's
authoritative metrics have two producers, and `analysis` reads neither of them (its docstring
states it never touches `results.csv`) while `pnbd_grid` reads only the stored ones.

**Two Pareto-benchmark-on-a-plot mechanisms.** `plot_weekly_aggregated(pareto_benchmark=True,
data=...)` fits it inside the plot (`:396-397`); `plot_suite_forecast(pareto_benchmark=True)`
pre-computes it (`:536-539`) *specifically* so it can honour the customer selection. The
notebooks use the first; nothing uses the second.

**`experiments` and `tuning` document each other.** `optuna_tuning.py`'s module docstring
(`:26-27`) tells the reader that `experiments.make_loaders` / `make_data_builder` produce the
contract; `experiment_utils.py`'s docstring (`:1-11`) describes the same contract from the
other side. One protocol, two prose definitions, in the two modules that sit either side of
it. `tuning/__init__.py:4-7` states it a third time.

**A read function with a write side effect, in two places.** `plot_suite_forecast:542-543`
and `_suite_prediction_paths:629` both call `aggregate_suite_predictions(root)`, which
writes `aggregated_<Model>.csv` **into the archived suite folder**. Verified by running it:
`group_metrics_suite_table(root, data=data)` created two files in the suite root. Documented
(`analysis.py:11`), but it means every read-side analysis mutates an archive the map's floor
protects — and it writes them under a *different* id-column name than the files it read
(next item).

---

### 3. Over-parameterised

**`run_optuna_study` has 21 parameters; 8 are set by nothing.** From an AST scan of every
call in `src/`, the entry points and the notebooks, the exercised set is
`model_type, data_builder, data_info, device, n_trials, study_name, storage, summary_dir,
removable_features, selection_metric, rollout_data, rollout_n_simulations, pruner,
sampler, append_timestamp, keep_only_best_checkpoint`. Never passed by anything:

- `direction` (`:964`) — always `"minimize"`. Both selection metrics are lower-is-better and
  the docstrings say so, so `"maximize"` would silently invert the whole search. A parameter
  whose only non-default value is a correctness bug.
- `rollout_horizon` (`:972`) — never passed. (`grep` finds the word in
  `Data_integration_LSTM_v2.ipynb`, but only inside a **stored ANSI traceback** at JSON line
  385 quoting `~/Desktop/Thesis/Package_Notebook_refactored/Models/optuna_tuning.py:761` — a
  pre-package module path, with a `run_optuna_study` parameter order that no longer matches.)
- `rollout_seed`, `rollout_mape_clip`, `rollout_min_actual_for_mape`, `rollout_weight_rmse`,
  `rollout_weight_mape`, `rollout_weight_bias` (`:974-979`) — never passed. All six are
  plumbed through four layers (`run_optuna_study` → `rollout_cfg["metric_kwargs"]` →
  `_validation_rollout_score(metric_kwargs=)` → `weekly_aggregate_rollout_metrics(**)`) to
  reach five keyword defaults nobody overrides.

**`weekly_aggregate_rollout_metrics`' five tuning knobs** (`:652-656`) are therefore all
dead, and one of its five return keys (`rollout_mae`) is read by nothing (§1).

**`analysis`'s read-side options, exercised versus declared** (AST over `Study.ipynb`, its
only caller):

| function | exercised | never passed |
| --- | --- | --- |
| `plot_suite_forecast` (13 params) | `root, panel_path, study, group, ci, confidence_interval` | `data`, `customer_ids`, `save_path`, `title`, `pareto_benchmark`, `**plot_kwargs` |
| `study_metrics` (7) | `confidence_interval, display` | `standard_deviation`, `ci`, `decimals` |
| `compare_study_metrics` (7) | `decimals, confidence_interval, display` | `standard_deviation`, `ci` |
| `group_metrics_suite_table` (6) | `panel_path` | `study`, `data`, `groups`, `save_path` |
| `group_metrics_suite_distribution` (7) | — (never called) | all |
| `describe_dataset` (2) | — (never called directly) | `name` (set by `describe_suite_dataset`) |

`standard_deviation` is never used despite the docstring at `:809-816` carefully explaining
why it answers a different question from `confidence_interval` — and the convention this
repo follows is to report the study-to-study spread, which is `standard_deviation`, not the
CI on the mean that the notebook actually asks for.

**The `data=` escape hatch is not an escape hatch — it is the only path for most of the
archive.** Four of the five archived electronics suites under `Studies/` have
`panel_config: None` in `config.json` (only `..._TestDimanche` carries one), so
`_actuals_from_panel` raises for them. Verified: `study_metrics(legacy_root, panel_path)` →
`ValueError: ... has no panel_config`. `plot_suite_forecast`,
`group_metrics_suite_table`, `group_metrics_suite_distribution` and `describe_suite_dataset`
all accept `data=` and can therefore still read those suites; **`study_metrics` and
`compare_study_metrics` do not** (`:790-798`, `:985-993`) and are simply unusable on 4 of 5
archived suites. That asymmetry sits directly against the on-disk floor.

**`_id_col`'s fallback disagrees with the archive it reads.** `analysis._id_col:91-100`
resolves `panel_config["id_col"] or data_summary["id_col"] or "customer_id"`. For a legacy
suite the first is absent and the second is a key `_suite_record` never writes, so it returns
`"customer_id"` — while the `Prediction_*.csv` files it just read use `"Id"`. Verified twice:
by running `aggregate_suite_predictions` on a copy of
`cross_entropy_cfg_2y_Train_1yPred_NoCov_V12` (input header `Id,week_0,...`, output header
`customer_id,week_0,...`), and on the real archive —
`Studies/cross_entropy_cfg_2y_Train_1yPred_NoCov_V1_10Studies_100_simulations/` already holds
three `aggregated_*.csv` files headed `customer_id` next to the `Prediction_*.csv` files
headed `Id`, so this has already happened on disk. Harmless only because
`load_predictions_from_csv`'s `id_col_candidates` sniffs both.

**Study-suite knobs nobody sets.** `ModelSpec.pareto_kwargs` (`config.py:70`) — never set
anywhere; every Pareto spec is `ModelSpec(name=..., model_type="pareto_nbd")`
(`run_studies.py`, `Pareto_Datasets.ipynb` ×2, `Study.ipynb`), so the documented MCMC knobs
(`mcmc`, `burnin`, `thin`, `chains`, `seed`, `param_init`) are unreachable through the suite.
`StudySuiteConfig.refit_kwargs` (`:129`) — never set, so `_rebuild_winner`'s
explicit-then-override merge (`:167-171`) has never merged anything.
`prediction_source="checkpoint"` — never chosen (§1).

**Other unexercised options in `evaluation/`.** `save_predictions_to_csv(week_offset=)`
(`:65`) — never set. `load_predictions_from_csv(id_col_candidates=)` — never overridden;
`(holdout_length=)` set only in `main_plot*.py`. `plot_weekly_aggregated(figsize=)` — never
passed; `(save_path=)` only from `main_plot*.py`. `alignment_check(max_lag=)` — the function
is never called. `holdout_actuals_NT` / `weekly_actuals`' whole `count_col` legacy branch
(name-or-integer-index, `:164-167`) — reachable only through the two broken scripts, one of
which still passes the deprecated integer default `count_col: str | int = 3`
(`main_plot.py:224`) that the docstring says was removed "because it produced silently-wrong
arrays".

**Checked and cleared** (genuinely exercised): `keep_only_best_checkpoint` and `overwrite`
(`Pareto_Datasets.ipynb`), `pareto_forecast`'s four dump arguments plus `seed`
(`Study.ipynb` #32), `plot_weekly_aggregated(pareto_benchmark, show_ci, data, title,
train_actuals)`, `metrics_table(pareto_benchmark, data)`, `group_metrics_table(save_path)`,
`layout.model_dirs(make_optuna)` (both values, from `runner`), `_suggest_param`'s whole spec
mini-language (`data_info` in `run_studies.py` and all four notebooks uses sets, tuples and
scalars), `assign_customer_groups(groups=)` — though note `_GROUP_PREDICATES` has exactly
two entries, so `groups=` can only ever *narrow*, never add.

**The lazy-import scheme buys nothing — measured.** `analysis.py` carries **11** deferred
imports (`:150, 211, 239-241, 387-388, 484, 681, 735-736, 879-880, 924, 1021-1023, 1098`),
six of them with a comment explaining the torch cost being avoided ("*keeps discovery
torch-free*", "*keep the module import cheap / torch-free*"). But
`studies/__init__.py:29` does `from .runner import run_study_suite`, and `runner.py` imports
`optuna`, `panelclv.models` and `panelclv.benchmarks` at module scope. Verified:

    import panelclv.studies.analysis   →  torch: True   optuna: True   matplotlib: True
    5.06 s   (vs 0.28 s for panelclv.data_preparation.dynamic_panel_dataset)

Importing the submodule executes the package `__init__` first, so all eleven deferrals are
paid anyway. `import panelclv.evaluation` also loads torch (`plot_utils.py:31` is a
top-level `import torch`).

---

### 4. Hardcoded dataset assumptions

Recorded only; acting on these is out of scope per the map.

- **`"Transactions"` is the fallback target in seven places in `tuning/`**:
  `optuna_tuning.py:497, 515, 528, 593, 766, 870, 1131`, all
  `metadata.get("target_col", "Transactions")`. `PanelConfig` exists to keep this string out
  of model-adjacent code; the tuning layer defaults to it seven times.
- **`runner._run_pareto_model:192-199` hardcodes three column names**: `id_col` defaults to
  `"Id"`, `target_col` to `"Transactions"`, and `"time_col": "period_start"` is a bare
  literal with no override. `prepare_dataset` actually returns `id_col`
  (`dynamic_panel_dataset.py:955`) so the fallbacks never fire — but the *same key* is read
  with **two different fallbacks in one function**: `data.get("id_col", "Id")` at `:193` for
  the Pareto fit and `data.get("id_col", "customer_id")` at `:210` for the CSV. Also
  `:141-142`, `plot_utils:347`, `forecast_run:96` and `monte_carlo_forecasting:340` all
  default the same key to `"customer_id"`.
- **`pnbd_grid` hardcodes `"Id"` three times** — `pred.drop(columns=["Id"])` (`:182`) and
  `pd.read_csv(pred_path).set_index("Id")` (`:339, 442`) — while
  `load_predictions_from_csv` sits one subpackage away with an id-sniffing candidate list.
  A grid study written with any other id column silently `KeyError`s.
- **`week_*` is the on-disk column format.** `save_predictions_to_csv:87` emits
  `week_{i}`; `analysis._prediction_index:110` matches `Prediction_(\d+)\.csv`;
  `aggregate_suite_predictions`' docstring names the format `id_col + week_0..week_{T-1}`.
  Weekly vocabulary baked into the archive format of a frequency-parameterised package.
  (On the floor, so recording only.)
- **`pnbd_grid` is weekly throughout**: `_DETREND_WINDOW = 13` ("*Quarter-year window*",
  `:143`), `_holdout_weeks` / `_holdout_season` in "absolute week numbers", the imported
  `WEEKS_PER_YEAR` and `_seasonal_weekly_multiplier` (`:42-47`), `weeks + 1` as a one-period
  step in the `alive_frac` arithmetic (`:351, 454`), and `week_0..week_{H-1}` assumed
  in-order from `pred.shape[1]` (`:340, 443`).
- **`plot_diff_grid` hardcodes model folder names**: `model_a="LSTM"`,
  `model_b="ParetoNBD"` (`:589-590`). The docstring notes archived suites store the
  benchmark as `"ParetoNBD_MLE"` — and indeed
  `Studies/cross_entropy_cfg_2yTrain_1yPred_NoCov_TestDimanche/ParetoNBD_MLE/` is exactly
  that, so the default is wrong for part of the archive it reads.
- **`_METRIC_SOURCE` and `_STUDY_METRIC_COLS` are on-disk column contracts.**
  `pnbd_grid:51-55` maps `"mape" → "mape_aggregate_style"` reading a stored `results.csv`
  header; `analysis:787` lists the same three names. Verified against archived headers
  (`model,model_type,study,seed,objective,rmse,bias_percent,mape_aggregate_style,param_*`).
  Renaming `mape_aggregate_style` breaks archived reads — a floor constraint on the metric
  vocabulary, worth stating explicitly for ticket 06.
- **Weekly / count vocabulary in plot chrome**: `plot_weekly_aggregated:425-426`
  (`"Week"`, `"Aggregate transactions"`), `analysis:550-552` (`"Weekly aggregated
  transactions — averaged over N studies"`), `pnbd_grid:577, 616` (`"churn_rate"`,
  `"mean transaction rate"`).
- **`weekly_aggregate_rollout_metrics(min_actual_for_mape=5.0)`** (`:656`) — an absolute
  volume floor in transaction counts, so its masking behaviour depends on panel scale; the
  docstring acknowledges scale-sensitivity for RMSE and normalises for it, but not here.
- **CWD-relative defaults.** `run_optuna_study(summary_dir="./optuna_summaries")` (`:967`),
  `data_info["checkpoint_dir"]` default `"./checkpoints"` (`:902, 1147`),
  `refit_best_trial(checkpoint_dir="./checkpoints")` (`:240`), and `pnbd_grid`'s
  `train_base = Path("Studies") / study_dir.name` in three functions (`:96, 220, 321`) —
  which the docstrings state is "resolved relative to the current working directory".
- **Non-reproducible timestamps** (same shape as ticket 02's `_auto_study_name` finding):
  `run_optuna_study:1141` `f"{study_name}_{datetime.now():%Y%m%d_%H%M}"`,
  `runner._suite_record:242` `datetime.now().isoformat()`,
  `pareto_forecast:340` and `forecast_run:79` both `datetime.now()`, all local and
  tz-naive. Study/run folder names are not derivable from config + seed.
- **`describe_dataset` bakes in the analysis thresholds** — `< 5`, `>= 2`, p90/p99, and the
  literal key `"customers_with_1_transaction"` (`:1159-1167`) — and reads `frequency` from
  the loose `data` key (`:1135`) while reading the other six window fields through the
  `panel_config`-preferring `cfg()` helper, so `frequency` is `None` for a legacy dict.

---

### The rulings the ticket asked for

**Ruling 1 — the `experiments/` ↔ `studies/` layering is sound in direction and wrong at
the seam.** The dependency edges are one-way and cycle-free at import time:
`models` ← `training` ← `tuning` ← `experiments` ← `studies`. `tuning` never imports
`experiments` (it only names it in a docstring), and `runner.py:33-44` imports downward
only. The layering is real: `experiment_utils` is 5 functions about *one* run (loader
shaping, the Optuna closure, rebuilding the winner, the warm-start refit) and `studies/` is
about *many*. Nothing here is duplicated between them. Three concrete problems, none of them
"merge the two":

1. **`experiments` imports two private names from `tuning`.**
   `experiment_utils.py:28-33` pulls `_build_inference_model_for` and `_build_model_for`
   alongside the two public `select_features*`. The comment at `:26-27` explains why
   ("*Models are built through the tuning registry rather than constructed here, so this
   module never has to know which architectures exist*") — a legitimate need reaching
   through a leading underscore. So **the model registry, not the layering, is the misplaced
   seam.** It is currently spread over three subpackages and six enumerations:
   `optuna_tuning._SEARCH_DEFAULTS` / `._SUGGESTERS` / `._BUILDERS` /
   `._build_inference_model_for`'s if-cascade, `studies/config.NEURAL_MODEL_TYPES` +
   `.VALID_MODEL_TYPES`, and `studies/runner._FORECASTERS`. CLAUDE.md's "adding a model
   touches three places" is a description of this split.
   Note the internal inconsistency: `_build_model_for` dispatches through a table, and the
   comment at `:539-542` says the table exists precisely so "a type is either wired
   everywhere or nowhere" — yet `_build_inference_model_for` (`:583-633`), added later, is
   an `if`-cascade ending in `sorted(_BUILDERS)` for its error message.
   `tests/test_model_registration.py` pins the six lists into agreement, so this is
   currently *safe*, not *simple* — and one of its assertions is a source-string check
   (`:129-132`).
2. **The naming is worse than "undefined".** `CONTEXT.md`'s **Study** entry lists
   `_Avoid_: run, experiment, sweep` — the vocabulary does not merely omit "experiment", it
   explicitly proscribes the word, while the package has an `experiments/` subpackage whose
   docstring opens "Experiment orchestration glue". Two further collisions inside `studies/`
   itself: `study_name` names the **suite** (`config.py:121`) while `layout.study_dir`
   (`:71`) names one **Optuna study** inside a model, giving
   `Studies/<study_name>/<Model>/Optuna_Studies/study_01/`; and `study_dir` means two
   different things in one subpackage — `layout.study_dir(model_dir, index)` (an Optuna study
   folder) versus `pnbd_grid.collect_grid_results(study_dir=...)` (a *generation* study, i.e.
   a folder of synthetic datasets). Also `group` means a customer segment in
   `analysis.group_metrics_*` / `segment_analysis` and a `(rate, churn)` grid cell in
   `pnbd_grid.group_summary`. Ticket 09's material; recorded here as evidence.
3. **`experiments/__init__.py`'s docstring is wrong about its own dependencies.** It says
   the helpers "*import from `panelclv.models` and `panelclv.tuning`*" (`:7-8`).
   `experiment_utils.py` imports from `panelclv.training` and `panelclv.tuning`; there is no
   `panelclv.models` import.

**Ruling 2 — `studies/analysis.py`'s three-way split, on the two parts this audit rules
on.** Measured boundaries:

| part | lines | count |
| --- | --- | --- |
| suite discovery + prediction I/O + actuals rebuild + selection | `:43-405` (minus the plot band) | ~300 |
| figure scaffolding (`_across_study_band`, `plot_suite_forecast`) | `:376-609` | 233 |
| metric tables (`_suite_prediction_paths`, `group_metrics_suite_*`, `study_metrics`, `_study_metrics_from_data`, `compare_study_metrics`) | `:612-1063` | 451 |
| notebook description (`describe_dataset`, `describe_suite_dataset`) | `:1066-1195` | 129 |

The **read-side library is genuinely reusable and genuinely shared**: `_read_suite_config`,
`_discover_models`, `_prediction_index`, `_is_deterministic_model`, `_actuals_from_panel`,
`_other_ids`, `load_model_predictions` are each called from 2-4 of the four public
groups, so cutting them out is a real seam, not a cosmetic one. Three findings against it:
its `_id_col` fallback contradicts the archive (§3); `_actuals_from_panel` is the single
choke point that makes 4 of 5 archived suites unreadable to two of the five public entry
points (§3); and `_discover_models` is reimplemented three times in `pnbd_grid` (§2).

The **description part is the weakest**: `describe_dataset` (`:1071-1173`) is a 40-key
`pd.Series` of dataset statistics with hardcoded thresholds, no caller other than its own
wrapper, and no test. It is the only part of `analysis.py` that never reads a prediction file
or a suite folder — it takes a `prepare_dataset` dict and returns descriptive statistics, so
by altitude it does not belong in the *study-archive reader* at all.

`_across_study_band` (`:376-405`) is classified above as figure scaffolding because
`plot_suite_forecast` is its only caller, but note it computes a statistic
(the across-studies CI of the aggregate curve) that `study_metrics` computes independently
for the same suites — so ticket 12 cannot move it without deciding whether it is a plot
helper or the third copy of the CI (§2).

---

### Verdict: is `compute_forecast_metrics` the single scoring authority?

**Within `evaluation/` and `studies/`, yes — and I verified it end to end. Package-wide,
no: `tuning/` has its own definitions of all three numbers, and two of the three disagree.**

Delegating correctly:

| site | how |
| --- | --- |
| `plot_utils.metrics_table:513` | direct call; returns exactly its three keys (`:518`) |
| `segment_analysis.group_metrics_table:186` | direct call; `rmse`, `mape_aggregate_style`, `bias_percent` all taken from its dict |
| `runner._run_neural_model:144`, `_run_pareto_model:216` | `mc_compute_metrics` = the same function (`models/__init__.py:51` aliases it) |
| `analysis._study_metrics_from_data:904` | direct call — and **verified numerically**: on `Studies/cross_entropy_cfg_2yTrain_1yPred_NoCov_TestDimanche` it reproduced all nine stored `results.csv` values to full precision |
| `pnbd_grid.collect_grid_results:119` | transitively — it reads the `rmse` / `bias_percent` / `mape_aggregate_style` columns the runner wrote |

The exceptions:

1. **`tuning.weekly_aggregate_rollout_metrics` (`optuna_tuning.py:648-719`) recomputes all
   three from scratch**, and it is what `objective` returns to Optuna under
   `selection_metric="rollout_composite"`. Run on the same `(N, T)` arrays:

   | | authority | rollout | agree? |
   | --- | --- | --- | --- |
   | rmse | 0.1237 (per-cell) | 7.7207 (`rollout_rmse`, on the customer-summed curve) | **no — 62× apart** |
   | mape | 9.5909 (`sum|Δ|/sum actual`) | 9.6204 (`rollout_mape`, mean of per-week %, masked at 5.0, clipped at 300) | **no** |
   | bias % | -9.5909 | -9.5909 (`rollout_bias_percent`) | yes — identical formula, duplicated code |

   So ADR-0003's stated rationale — "*aligning selection with the metric actually
   reported*" — holds for bias only. RMSE is a different statistic under the same name, and
   MAPE is a different estimator. These names then land on disk as trial user-attrs in every
   `<study>_trials.csv`.
2. **Three metric vocabularies for one number.** `mape_aggregate_style` (the authority,
   `results.csv`, `analysis._STUDY_METRIC_COLS`), `mape` (`pnbd_grid._METRIC_SOURCE:54` and
   `segment_analysis.group_metrics_table:190`, both renaming it on the way out), and
   `rollout_mape` (`tuning`, a *different formula* under a similar name). Three names, two
   formulas, one concept.
3. **`segment_analysis.aggregate_bias` (`:39-47`)** — the acknowledged and legitimate
   exception (raw-count bias for a group whose actual total is zero), documented as such at
   `evaluation/__init__.py:11-13`.
4. **`plot_utils.alignment_check:574-575`** computes `total_actual` / `total_pred` — raw
   totals whose difference is `aggregate_bias`. A fourth site computing the bias concept, in
   a function nothing calls.
5. **`pnbd_grid`'s three grid metrics** (`seasonal_corr`, `alive_volume_ratio`,
   `dead_volume_leakage`) legitimately do not delegate — they are new quantities with no
   equivalent in the authority. They do, however, sum predictions into weekly totals
   themselves (`_weekly_prediction:179-182`, and `pred.to_numpy()` at `:346, 449`) rather
   than through `weekly_aggregate_predictions`.

`evaluation/__init__.py:9-13`'s claim ("*everything here delegates to it*") is true as
scoped, but it reads as a package-wide statement and CLAUDE.md makes it one. The claim needs
either the `tuning` carve-out spelled out or `weekly_aggregate_rollout_metrics` renamed off
the shared vocabulary.

---

### Cross-lane observations (recorded so they aren't lost; not this lane's to decide)

- **ADR-0002 is violated in exactly one place, and it is my lane's fault.** ADR-0002's
  consequence reads "*`evaluation/` imports the simulator from `models/`, never the other way
  round.*" But `models/monte_carlo_forecasting.py:322-324` does
  `from panelclv.evaluation.plot_utils import save_predictions_to_csv`, with the comment
  "*Lazy import: `evaluation.plot_utils` already imports from this module, so a top-level
  import here would create a circular import at load time.*" The cause is that
  `save_predictions_to_csv` — pure prediction I/O, called from 6 modules and 5 entry points,
  and the widest-used symbol in `evaluation/` — lives inside a *plotting* module.
- **`evaluation/__init__.py` documents 2 of its 3 modules** — `forecast_run` is absent from
  the docstring while `ForecastRun` is imported at `:33` and exported at `:49`. Exactly
  ticket 02's `data_preparation/__init__` finding, one subpackage over.
- **A stale stored traceback in a live notebook.**
  `Data_integration_LSTM_v2.ipynb` cell 15's output is a `KeyboardInterrupt` traceback from
  `/home/virthian/Desktop/Thesis/Package_Notebook_refactored/Models/optuna_tuning.py` — a
  pre-package layout — and it quotes a `run_optuna_study` parameter list in a different
  order from today's. `tests/test_notebooks_current_api.py`'s blacklist exists for exactly
  this ("*a saved traceback quoting an old signature reads to the reader as current API*")
  but only blacklists six benchmark-refactor names, none of which appear here.
- **`experiment_utils.py` annotates with a name it never imports.** `Path` appears in two
  annotations (`:180`, `:240`) with no `from pathlib import Path`. Harmless at runtime
  (`from __future__ import annotations`), but `typing.get_type_hints` on either function
  raises `NameError` — verified (it hits the TYPE_CHECKING-guarded `optuna` first, which is
  deliberate; `Path` is simply an omission).
- **`main_plot.py` / `main_plot_covar.py` diverged from one file.** `main_plot.py:1-3`'s
  docstring calls itself the "*Counterpart to the old `main_plot.py`*", and the two share
  `_model_dir` / `latest_prediction_path` / `load_all_*_predictions` / `main` structure
  verbatim. The `metrics_table` fix landed in the covar copy only (`:171-188` versus
  `main_plot.py:244-251`), which is why they are now broken in two different ways.
  Ticket 10's call.
- **`Study.ipynb`'s `ROOT` paths all point outside the repo**, at
  `/home/virthian/Desktop/Thesis/Study_historique/{Electronic,Gift,MultiChanel}/...` (9
  distinct suites across cells 18-70). The in-repo `Studies/` trees are older and mostly
  `panel_config`-less. So the archive the on-disk floor protects is largely *not in the
  repo*, and its `panel_config` coverage cannot be audited from here.

### Test coverage of this lane

Three dedicated test files cover 1491 of the lane's 4880 lines:
`tests/test_studies_analysis.py` (388 lines → `analysis.py`, 1195, well: 21 tests over
`load_model_predictions`, `aggregate_suite_predictions`, `plot_suite_forecast`,
`study_metrics` and `compare_study_metrics`), `tests/test_studies_layout.py` (151 →
`layout.py` 121 and `config.py`'s `validate` 175), `tests/test_model_registration.py` (168 →
the `optuna_tuning` registry and `analysis._is_deterministic_model`).

**3389 lines have no dedicated test**: `optuna_tuning.py` (1240), `pnbd_grid.py` (623),
`plot_utils.py` (609), `experiment_utils.py` (311), `runner.py` (295),
`segment_analysis.py` (200), `forecast_run.py` (111). Indirect coverage is thin and
uneven — by import census across `tests/`:

- `optuna_tuning.py`: only the registry helpers (`validate_data_info`,
  `_suggest_params_for`, `_build_model_for`, `_build_inference_model_for`,
  `VALENDIN_SEARCH_DEFAULTS`). Untested: `_suggest_param`'s whole spec mini-language,
  `_merge_specs`, `select_features`, `select_features_for_trial`,
  `suggest_covariate_selection`, `validate_removable_features`,
  `weekly_aggregate_rollout_metrics`, `_validation_rollout_score`, `objective`,
  `run_optuna_study` — including the horizon validation, the warm-up warning and the
  `keep_only_best_checkpoint` deletion loop, which deletes files.
- `experiment_utils.py`: `make_loaders` only, via `test_golden_end_to_end.py:40`.
  `make_refit_loader`, `make_data_builder`, `build_inference_from_trial` and
  `refit_best_trial` are import-checked by `test_imports.py:48` and nothing more.
- `plot_utils.py`: `save_predictions_to_csv` / `load_predictions_from_csv` via
  `test_studies_analysis.py:23`; `metrics_table` and `plot_weekly_aggregated` are
  import-checked only (`test_imports.py:46`).
- **`pnbd_grid.py` (623), `runner.py` (295), `segment_analysis.py` (200) and
  `forecast_run.py` (111) have no coverage at all** — direct or indirect. That is 1229
  lines, including the code behind the thesis's synthetic-grid figures
  (`pnbd_grid.seasonality_grid` / `alive_volume_ratio_grid` / `dead_volume_leakage_grid`,
  whose oracle arithmetic is the duplicated 55-line block of §2) and the whole suite
  orchestrator that costs GPU-hours to re-run.

Two gaps worth naming for ticket 06: nothing tests that an archived suite stays readable
(the on-disk floor has no executable definition, unlike the benchmark floor's two validate
scripts), and nothing tests `runner` end to end, so the format `Studies/` is written in is
pinned by no test at all.
