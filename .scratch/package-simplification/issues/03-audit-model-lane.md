# Audit: the model lane

Type: task
Status: resolved

## Question

Read `models/` (1660 lines), `benchmarks/` (534) and `training/` (473) and report
against the four shared dimensions (see ticket 02 for the list and the map's Notes for
the dead-code rule).

Two things are **settled and not to be re-litigated**: `models/multinomial_lstm.py`
taking an `embedder` argument while `benchmarks/valendin_lstm.py` hardcodes
`ValendinEmbedder` is the design working as intended (ADR-0004/0005) — one is
parameterised and free to change, one is nailed down. Their separate backbones are
deliberate, not duplication. Same for Pareto/NBD.

The sharp question this ticket must rule on: **should the rollout stepping strategy be
a swappable seam, the way the embedder already is?** `monte_carlo_forecasting.py` has
`run_monte_carlo_forecast` (line 442) and `run_monte_carlo_forecast_transformer` (502),
both delegating to `_run_monte_carlo`, differing only in the per-path stepper —
`simulate_one_path` vs `simulate_transformer_path`. The difference is genuine (the LSTM
carries hidden state; the Transformer re-reads its window), but it means a third
architecture needs a third top-level forecast function *plus* a third `_FORECASTERS`
entry. Report the options and their cost; ticket 06 chooses.

Also worth a look: `losses.py` defines `FocalLoss` and `SquaredEMDLoss`, reachable only
through `build_criterion`. Are they configured by anything, ever?

Report is evidence, not decisions. Do not change code.

## Answer

**Scope read in full:** `models/embedders.py` (248), `losses.py` (202),
`monte_carlo_forecasting.py` (619), `multinomial_lstm.py` (211),
`multinomial_transformer.py` (305), `models/__init__.py` (75);
`benchmarks/pareto_benchmark.py` (351), `valendin_lstm.py` (183), `__init__.py` (55);
`training/training_utils.py` (473), `__init__.py` (24) — 2746 lines.

**Method.** For every public symbol, every key of the forecast return dict, and every
field of `FitResult`, grepped for callers across `src/panelclv/`, the live entry points
(`scripts/run_studies.py`, the two `validate_*.py`, the two `main_plot*.py`, the four
live notebooks) and `tests/`, keeping those three populations separate; notebooks parsed
as JSON, stored outputs included. Behavioural claims were confirmed by running code in
`venvs/thesis_rocm` with `PYTHONPATH=src`, not read off docstrings. Nothing was
installed or changed.

**Headline.** The lane has almost no dead *modules* but two live wrong-behaviour seams
and a large ring of unread bookkeeping. The rot is: the model→rollout-stepper pairing is
encoded in **three** independent places and enforced in none, so a Transformer driven by
the recurrent stepper is silently misforecast — and `scripts/main_plot.py` does exactly
that; three of `losses.py`'s four loss branches have never been selected by any run in
the repo's history, while the class-weight vector every live notebook computes is thrown
away by the CE branch that receives it; `_save_predictions_run` makes `models` import
`evaluation`, the one direction ADR-0002 forbids; the same three attributes are hoisted
onto 9 classes and 18 of those 27 assignments have no reader; `simulate_one_path` and
`simulate_transformer_path` share 21 byte-identical lines of preamble; and the entire
Transformer rollout — one of the thesis's two developed models — has zero test coverage.

---

### 1. Unreachable

**No module in this lane is unreachable.** All ten have a caller in `src/` or a live
entry point. What is unreachable is finer-grained, and in three cases it is a *branch*
that a live caller can never take.

**Three of `build_criterion`'s four branches have never been selected — anywhere, ever.**
`build_criterion` (`losses.py:174`) is the only route to `FocalLoss` (`:34`) and
`SquaredEMDLoss` (`:69`), and it dispatches on `loss_type`. Every site that supplies
that string supplies `"cross_entropy"`:

| site | value |
| --- | --- |
| `scripts/run_studies.py:31` | `LOSS_TYPE = "cross_entropy"` |
| `scripts/validate_valendin_lstm.py:211` | `loss_type="cross_entropy"` |
| `notebooks/Data_integration_LSTM_v2.ipynb` cell 12 | `LOSS_TYPE = "cross_entropy"` |
| `notebooks/Data_integration_TRANSFORMER_v2.ipynb` cell 11 | `LOSS_TYPE = "cross_entropy"` |
| `notebooks/Study.ipynb` cell 14 | `LOSS_TYPE = "cross_entropy"` |
| `notebooks/Pareto_Datasets.ipynb` cells 11, 13 | `"loss_type": "cross_entropy"` |
| `notebooks/archive/*` (3 notebooks) | `"cross_entropy"` |
| defaults in `training_utils.py:216`, `:389`, `experiment_utils.py:241`, `optuna_tuning.py:876` | `"cross_entropy"` |

No stored artefact names another: every one of the 6 archived study folders under
`Studies/` is `cross_entropy_*`, and a repo-wide grep for `weighted_ce` / `focal` /
`emd` outside comments returns nothing in `Studies/`, `optuna_summaries/`,
`Fine_tuning_optuna/`, `FOR_ANALYSIS/`, `figures/` or `inputs_configs/`. **The thesis
carve-out therefore does not apply** — no figure or number came from these branches.
Verified they are functional, not broken (so this is unused code, not rotted code):

    cross_entropy  -> 2.3113    focal -> 1.8868
    weighted_ce    -> 2.3113    emd   -> 1.2487

`FocalLoss`, `SquaredEMDLoss` and `focal_gamma` (`losses.py:178`, plus the Optuna
`focal_gamma` search at `optuna_tuning.py:874-886`) are ~95 lines and one tuning branch
serving three never-taken strings.

**`compute_class_weights` is live but its output is discarded on every live run.** Two
live notebooks compute it (`Data_integration_LSTM_v2` cell 12 line 14,
`Data_integration_TRANSFORMER_v2` cell 11 line 6), print it, and pass it into `data_info`
as `"class_weights"` (LSTM cells 15/17/19, Transformer cells 13/15). With
`loss_type="cross_entropy"`, `build_criterion:186` returns a plain `nn.CrossEntropyLoss()`
and never touches the tensor — the notebooks' own comment says so
("harmless for 'cross_entropy'"). `Study.ipynb` cell 21 has the line commented out.
`Pareto_Datasets.ipynb` cell 0 imports it and never calls it. So the function has live
callers and no live *consumer*.

**Three of the forecast dict's eight keys have no reader.** Checked every key of the dict
`_run_monte_carlo` returns (`monte_carlo_forecasting.py:409-434`) against `src/`, entry
points and tests, distinguishing reads of `forecast[...]` from reads of `data[...]`:

| key | readers |
| --- | --- |
| `"prediction_mean"` | 12 sites (src, entry, tests) |
| `"actual"` | 11 sites |
| `"simulations"` | 3 live notebooks |
| `"n_simulations"` | 3 live notebooks (in a staleness `assert` only) |
| `"target_col"` | **none** |
| `"target_idx"` | **none** |
| `"seed"` | **none** |
| `"predictions_path"` | **none** (only `pareto_forecast`'s same-named key is read, `Study.ipynb` cell 32) |

`"target_col"` and `"target_idx"` are duplicates of `data["target_col"]` /
`data["target_idx"]` (`dynamic_panel_dataset.py:943-944`), which is what every reader
actually uses (`studies/runner.py:215`, `analysis.py:269, 884, 1100`). `"seed"` is echoed
back unchanged. `"predictions_path"` is written for traceability and read by nobody — its
side effect (the CSV) is the point.

**`FitResult.best_val_f1` is computed every epoch and read by nobody.**
`validate_one_epoch`'s `compute_f1` (`training_utils.py:132`) is never passed by any
caller, so it is always `True`: every validation epoch concatenates predictions to CPU
(`:178-179`) and calls `sklearn.metrics.f1_score` (`:188-189`) — the lane's only sklearn
dependency. The result lands in `FitResult.best_val_f1` (`:41`), is printed
(`:318`), and is stored as an Optuna user attr (`optuna_tuning.py:916`). Grepping
`best_val_f1` across `src/`, all live entry points, `tests/` and the four notebooks'
sources and stored outputs: **no reader anywhere.** `FitResult.history` (`:44`) is read
only by `scripts/verify_transformer_training.py:98,105` — a one-shot, not a caller by the
map's rule.

**`InferenceMultinomialTransformerModel._seq_len_hint` is assigned and never read**
(`multinomial_transformer.py:294`). `main_plot_covar.py:96` and `optuna_tuning.py:531`
both pass `seq_len=` to inference models where the docstring already says it is
"accepted for API symmetry; unused here" (`:276`).

**Three `state.pop("_cached_mask", None)` guards cannot fire.** At
`experiment_utils.py:223`, `training_utils.py:426` and `optuna_tuning.py:808`, each with
a comment justifying itself. The buffer is registered `persistent=False`
(`multinomial_transformer.py:237`), so it is not in a `state_dict` — verified at runtime
(`'_cached_mask' in tr.state_dict()` → `False`) — and it is not in any stored checkpoint
either: **0 of the 108 transformer `.pth` files under `checkpoints/` carry the key.** Only
`experiment_utils.py:221-222` claims a legacy reason ("older checkpoints can still carry
the key"); the evidence on disk does not support it.

**`Embedder` is exported as public API and cannot be used.** It is in `models.__all__`
(`__init__.py:63`) and is the seam ADR-0005 names, but it declares `output_dim: int` as a
bare annotation (`embedders.py:65`) and defines no `forward`. Verified: it constructs
without error, then

    Embedder(...)(x)                            -> NotImplementedError
    MultinomialLSTMModel(embedder=Embedder(...)) -> AttributeError: no attribute 'output_dim'

so it is an ABC in intent, not in declaration, advertised on the headline surface.

**Public but with no caller outside its own module** — over-exposure, listed for ticket
05's ledger rather than as kill candidates:

| symbol | intra-module caller | external callers |
| --- | --- | --- |
| `monte_carlo_forecasting.simulate_one_path` | `run_monte_carlo_forecast` | none |
| `.simulate_transformer_path` | `run_monte_carlo_forecast_transformer` | none |
| `losses.FocalLoss` | `build_criterion` (never-taken branch) | none |
| `losses.SquaredEMDLoss` | `build_criterion` (never-taken branch) | none |
| `losses.build_criterion` | — | `training_utils` (lane-internal) only |
| `multinomial_transformer.SinePositionalEncoding` | `_MultinomialTransformerBackbone.__init__` | none |
| `_MultinomialTransformerBackbone.generate_causal_mask` (public method) | `.forward`, `MultinomialTransformerModel.__init__` | none |
| `training_utils.train_one_epoch` | `fit_model`, `refit_full_calibration` | none |
| `training_utils.validate_one_epoch` | `fit_model` | none |
| `training_utils.FitResult` | `fit_model`, `refit_full_calibration` | none (fields read piecemeal) |
| `training_utils.refit_full_calibration` | — | `experiments.refit_best_trial` only |
| `models.Embedder` | base class + type annotation | none |
| `models.mc_simulate_one_path` (alias) | — | **none** |
| `models.mc_simulate_transformer_path` (alias) | — | **none** |

The last two matter because `models/__init__.py:46` advertises them as "the short `mc_*`
aliases used throughout the notebooks" — verified false: no notebook, live or archived,
mentions either name. `mc_forecast`, `mc_forecast_transformer` and `mc_compute_metrics`
*are* used throughout the notebooks; those two are not.

**Two live entry points in this lane's blast radius do not work as written.**

- `scripts/main_plot_covar.py:100-111` calls `evaluation.forecast_from_checkpoint` with
  six keyword arguments it does not accept — verified against the live signature:
  `['batch_size', 'calibration', 'holdout_calendar', 'model_type', 'seq_cols', 'target_col']`
  — and then reads `result["predictions"]`, a key nothing returns. The script raises
  `TypeError` on its first forecast call. Note the *lost* `model_type="transformer"`
  argument: it is the fossil of a `forecast_from_checkpoint` that once dispatched the
  rollout on model type.
- `scripts/main_plot.py` runs, but forecasts the Transformer through the **LSTM** rollout
  — see the ruling below. It has produced no output in this checkout (`scripts/Predictions/`
  does not exist), so on the evidence available it has not corrupted a thesis figure; the
  real Transformer numbers in `FOR_ANALYSIS/transformer_..._n200_seed42_*` came from
  `Study.ipynb` cell 30, which calls `mc_forecast_transformer` correctly.

---

### 2. Duplicated

**The model→rollout-stepper pairing is written three times.**

| site | form |
| --- | --- |
| `tuning/optuna_tuning.py:583-631` `_build_inference_model_for` | if/elif returning `(inference_model, forecaster)` |
| `studies/runner.py:51-55` `_FORECASTERS` | `dict[model_type -> forecaster]` |
| the notebooks / `main_plot.py` | the human picks `mc_forecast` vs `mc_forecast_transformer` by hand |

The first two agree today and are structurally free to drift. Worse, the first one's
answer is **computed and thrown away**: `experiments/experiment_utils.py:216` does
`inference_model, _ = _build_inference_model_for(...)`, discarding the forecaster, and
`studies/runner.py:130` then re-derives it from `_FORECASTERS`. Only
`optuna_tuning.py:799` (the `rollout_composite` path) actually uses the returned pair.

**21 byte-identical lines of preamble in the two rollouts.** `diff` of
`monte_carlo_forecasting.py:129-153` against `:227-252` differs **only in comments** —
every line of code (`_get_target_idx`, device resolution, `model.to(device).eval()`, the
two `_as_tensor` casts, the shape unpacks, the `sampled_path` allocation, `ar_features` /
`ar_idx` / `ar_state` / `ar_norm` construction) is the same text twice. The AR
update-and-restandardize block is a second verbatim pair (`:166-176` vs `:272-279`),
identical modulo `x_t`/`x_next` and `previous_sample`/`sample`.

**The same three attributes are hoisted onto nine classes; 18 of the 27 assignments have
no reader.** `seq_cols`, `target_col` and `num_target_classes` are set on the embedder
(`embedders.py:92-95`), then re-set on each of three backbones
(`multinomial_lstm.py:108-110`, `multinomial_transformer.py:113-115`,
`valendin_lstm.py:92-94`) and each of six wrappers (`multinomial_lstm.py:163-165, 203-205`;
`multinomial_transformer.py:224-226, 290-292`; `valendin_lstm.py:135-137, 175-177`).
Grepping `.seq_cols` and `.target_col` on a model object across `src/`, entry points and
tests returns **nothing**: only `model.num_target_classes` is read externally
(`optuna_tuning.py:892`, `experiment_utils.py:294`), and only the backbone's copy sizes
the head. The comment "Hoist commonly accessed fields for convenience"
(`multinomial_lstm.py:162`) documents an access pattern that no longer exists — the
simulator reads `data["seq_cols"]`, not the model's.

**The sampling head is written three times, identically.**
`InferenceMultinomialLSTMModel.forward:207-211`,
`InferenceMultinomialTransformerModel.forward:296-305` and
`InferenceValendinLSTMModel.forward:179-183` all do
`softmax → dist.Categorical(probs).sample() → unsqueeze(-1) → float()`. The separate
*backbones* are settled (ADR-0004); the sampling head is not architecture — ADR-0004
lists "the simulator" among the shared infrastructure — so the benchmark's copy of it
sits on the wrong side of the line the ADR draws. Recorded, not decided.

**Three prediction-CSV writers, two folder-naming schemes.**
`monte_carlo_forecasting._save_predictions_run:290-341` writes
`{tag}_n{n}_seed{seed}_{ts}/`; `evaluation/plot_utils.pareto_forecast:331-346` inlines a
near-copy writing `{tag}_{ts}/`; `studies/runner.py:137-142` and `:205-210` bypass both
and call `save_predictions_to_csv` with a deterministic `layout.prediction_path`. All
three end at the same `save_predictions_to_csv`, and both timestamped schemes appear side
by side in `FOR_ANALYSIS/`. (`evaluation/forecast_run.ForecastRun.save_predictions:84` is
a fourth — ticket 04's to weigh.)

**`models` imports `evaluation`, the one direction ADR-0002 forbids.**
`monte_carlo_forecasting.py:324` does a lazy
`from panelclv.evaluation.plot_utils import save_predictions_to_csv`, and its own comment
(`:322-323`) says the lazy form exists *because* a top-level import would be circular.
ADR-0002: "`evaluation/` imports the simulator from `models/`, never the other way round."
This is the only cross-subpackage edge in the lane besides
`training → models.losses` and `benchmarks → models.embedders`, both of which are intended.

**One concept, three names, for the forecast entry point.**
`run_monte_carlo_forecast` (canonical; used by `optuna_tuning.py:605,628` and
`scripts/validate_valendin_lstm.py:231`), `mc_forecast` (alias; used by `studies/runner.py:52`
and three live notebooks) and `_mc_forecast` (a third rebinding at
`evaluation/plot_utils.py:37`). All three are live, so no one name can be called the real
one from usage alone.

**One concept, two names, for the head size.** `num_target_classes` in `models/` and
`benchmarks/` (24 occurrences) vs `max_trans` in `training/` and `experiments/`
(`training_utils.py:52,84,131,203,379`, `experiment_utils.py:294`,
`optuna_tuning.py:892`, and the live notebooks). The translation happens at the two call
sites that write `max_trans=model.num_target_classes`. A third spelling,
`clip_target_upper`, names the same quantity minus one in `configs/`.

**`target_idx` is derived four ways.** `prepare_dataset` returns it
(`dynamic_panel_dataset.py:944`); `_run_monte_carlo:370` recomputes it via
`_get_target_idx(seq_cols, target_col)` instead of reading it; `plot_utils.py:323` and
`segment_analysis.py:75` each inline `list(data["seq_cols"]).index(data["target_col"])`.
The same holds for the actuals extraction `np.asarray(data["holdout"])[:, :, target_idx]`,
written at `monte_carlo_forecasting.py:407`, `plot_utils.py:324`, `runner.py:215` and
`analysis.py:884`.

**`id_col` has two defaults inside the lane.** `monte_carlo_forecasting.py:340` falls back
to `"customer_id"`; `pareto_benchmark.py:282` to `"Id"`. `studies/runner.py` uses both in
one function — `"Id"` for the estimator (`:194`) and `"customer_id"` for the CSV (`:209`,
`:224`).

---

### 3. Over-parameterised

- **`loss_type`'s three non-CE values, `focal_gamma`, and `class_weights`** — §1. Never
  selected in any run in the repo's history.
- **`log_wandb`** — `training_utils.py:213`. Never `True`: no live entry point sets it,
  and `optuna_tuning.py:905` reads `data_info.get("log_wandb", False)`. The lazy import
  (`:265-271`), the per-epoch `wandb.log` (`:310`) and the artifact upload with its bare
  `except Exception: pass` (`:350-357`) are all unexercised.
- **`validate_targets`** — `training_utils.py:86,133,215,392`. Never set `False`, so
  `_validate_targets` runs `targets.min()`/`.max()` on **every batch of every epoch**
  (`:97-98`, `:157-158`) — two device→host syncs per batch to re-check an invariant
  `prepare_dataset` already established.
- **`compute_f1`** — `training_utils.py:132`. Never passed; always `True`; result unread
  (§1).
- **`ValendinLSTMModel`/`InferenceValendinLSTMModel`'s `memory_units` / `dense_units`**
  (`valendin_lstm.py:82-83, 124-125, 165-166`). Never overridden anywhere, and
  `tests/test_model_registration.py:135-148` exists specifically to assert that Optuna
  must *not* search them ("would silently unfreeze the reference implementation"). Two
  constructor knobs on a model whose defining property is that its sizes are fixed, with
  a test guarding against their use.
- **`param_init`** (`pareto_benchmark.py:292`) — never overridden; the docstring notes it
  "is washed out by burn-in".
- **`mcmc` / `burnin` / `thin` / `chains`** — the only live explicit passes restate the
  defaults (`scripts/validate_pareto_benchmark.py:137`), and `Study.ipynb` cell 17's
  `pareto_kwargs` line is commented out with "these are the defaults". Weakly flagged:
  being explicit is arguably part of the R cross-check, and `studies/config.pareto_kwargs`
  plumbs them.
- **`MultinomialTransformerModel(seq_len=...)` and its `_cached_mask`** — verified the
  cached path yields output identical to recomputing the mask, so the knob buys a
  `torch.triu` per forward on a fixed-length batch, at the cost of a buffer, a
  `_cached_seq_len` companion, a `persistent=False` subtlety and three `state.pop` guards
  (§1). Live (`optuna_tuning.py:531`), but the ratio of machinery to saving is worth
  ticket 06 seeing.
- **`InferenceMultinomialTransformerModel(seq_len=...)`** — accepted, stored as
  `_seq_len_hint`, never read (§1). Passed by two callers.
- **`_run_monte_carlo`'s `model_type` parameter** (`:358`) — only ever the two literals
  `"lstm"` / `"transformer"` hardcoded at `:498` and `:538`, used solely as a folder-name
  fallback. Notably it is *not* the Valendin benchmark's name even when the benchmark is
  what ran.

Checked and cleared: `grad_clip` (genuinely exercised — `validate_valendin_lstm.py:207`
passes `None` for benchmark fidelity); `return_simulations` (default `True` for the
notebooks, `False` from every programmatic caller, and at thesis scale — 829 customers ×
52 periods × 200 paths ≈ 34 MB — the default costs nothing); `verbose`; `dropout`;
`val_score_start` (live, ADR-0001); `save_predictions`/`output_dir`/`file_name`/`run_name`
(live via `Study.ipynb` cells 28/30 — that is where `FOR_ANALYSIS/` came from);
`customer_ids` reordering in `compute_pareto_predictions` (live, `runner.py:203`).

**A reproducibility hole worth recording here**, because it is what the seeding design
costs: `_run_monte_carlo:375-378` seeds by calling **global** `torch.manual_seed(seed)`
rather than threading a `torch.Generator`. Verified: a seeded forecast permanently moves
the process RNG (`torch.rand(1)` before/after differs). So a notebook cell's forecast
changes the sampling of every later cell, and the ordering of cells becomes part of the
result — which is exactly the failure mode priority 2 in `CLAUDE.md` names ("results never
depend on the order notebook cells were run in"). Benign today only because every live
caller passes an explicit `seed`.

---

### 4. Hardcoded dataset assumptions

Recorded only; acting on these is out of scope per the map.

- **`target_col: str = "Transactions"` as a default in six constructors/functions**:
  `embedders.py:71, 137, 219`, `monte_carlo_forecasting.py:93, 194`,
  `valendin_lstm.py:81, 123, 163`, `pareto_benchmark.py:283`. Plus
  `optuna_tuning.py:593, 766, 870, 1131` and `plot_utils.py`-side
  `metadata.get("target_col", "Transactions")` fallbacks. The exact column name
  `PanelConfig` exists to keep out of model code, as a default in the model code.
- **`compute_pareto_predictions` hardcodes the whole panel schema as defaults**:
  `id_col="Id"`, `target_col="Transactions"`, `time_col="period_start"`,
  `period_in_days=7.0` (`pareto_benchmark.py:282-285`). `period_in_days` is the only
  frequency knob in the benchmark and its default is weekly; `studies/runner.py:58`
  supplies it from `_PERIOD_DAYS = {"daily": 1.0, "weekly": 7.0, "monthly": 30.0}`, where
  `"monthly": 30.0` silently mis-scales a real calendar month.
- **`_build_cbs` measures recency and age in days ÷ `period_in_days`**
  (`pareto_benchmark.py:105-106`), so a monthly panel's sufficient statistics inherit the
  30-day approximation, and `cbs["T_cal"].clip(lower=1.0)` (`:110`) floors age at "one
  period" whose length is whatever that division assumed.
- **`ValendinEmbedder` is documented against the paper's banking panel** — "roughly 12
  dimensions ... where the features are 52 weeks and the transaction-count classes"
  (`embedders.py:204-206`). Frozen by ADR-0004, so this is a note not a defect, but the
  52 is a dataset fact living in a docstring that governs the head width.
- **`_save_predictions_run` writes weekly column names**: its docstring promises
  `week_0..week_{H-1}` (`monte_carlo_forecasting.py:306`) and
  `plot_utils.save_predictions_to_csv:87` emits exactly `f"week_{i}"` whatever the
  frequency. This is on-disk format, so it is also floor item 3 — recorded, not to be
  changed.
- **Non-deterministic run-folder names.** `_save_predictions_run:328` uses
  `datetime.now()` — local, no timezone — so the prediction folder name is not
  reproducible from config + seed. Same defect ticket 02 recorded for
  `_auto_study_name`, in a second subpackage.
- **`_MEMORY_UNITS = _DENSE_UNITS = 128`** (`valendin_lstm.py:63-64`) and `nhead=8`,
  `d_model=64`, `dim_feedforward=d_model*4` (`multinomial_transformer.py:100-133`) — the
  first pair is the paper's and deliberate; the second set is our own defaults that
  Optuna always overrides, so nothing depends on them.

---

### The ruling the ticket asked for: should the stepping strategy be a swappable seam?

**The evidence, before the options.**

1. **The pairing is already a seam three times over, and enforced nowhere** — §2. Two
   registries in `src/` plus the human's memory in notebooks and `main_plot.py`.

2. **Getting the pairing wrong fails asymmetrically, and the silent direction is the one
   that matters.** Verified by running both crossings on the same weights:

   | crossing | result |
   | --- | --- |
   | LSTM inference model + `run_monte_carlo_forecast_transformer` | `TypeError: InferenceMultinomialLSTMModel.forward() got an unexpected keyword argument 'only_last'` |
   | **Transformer inference model + `run_monte_carlo_forecast`** | **no error** — returns a plausible `(N, T_HOLD)` forecast |

   The silent direction happens because `InferenceMultinomialTransformerModel.forward`
   accepts `state=None` for "API parity with the LSTM"
   (`multinomial_transformer.py:299`) and ignores it. The recurrent stepper then feeds
   single periods (`monte_carlo_forecasting.py:164`), which resets the positional
   encoding to index 0 and discards all history — the failure
   `simulate_transformer_path`'s docstring explicitly warns about ("this is exactly why a
   single-step feed ... is wrong for this model", `:216-217`). Confirmed the two steppers
   on identical weights and seed give different answers (50 paths: totals 32.24 vs 32.06,
   `allclose` → `False`).

3. **A live entry point already takes the silent branch.**
   `evaluation/plot_utils.forecast_from_checkpoint:585-608` unconditionally calls
   `_mc_forecast` — the recurrent stepper — for whatever model the factory built, and
   `scripts/main_plot.py:128-155` builds an `InferenceMultinomialTransformerModel` and
   hands it to that function. `main_plot_covar.py` still passes a `model_type="transformer"`
   argument the function no longer has, so the dispatch that would have prevented this
   once existed and was dropped. Mitigating: `scripts/Predictions/` does not exist in this
   checkout, and the Transformer numbers in `FOR_ANALYSIS/` came from `Study.ipynb`'s
   correct `mc_forecast_transformer` call — so this is a live latent bug, not (on the
   available evidence) a corrupted thesis figure.

4. **The two steppers are 90% shared text** — §2, 21 byte-identical preamble lines plus a
   duplicated AR block. The genuine difference is small and local: warm-up-then-thread-state
   vs grow-the-context, and `only_last=` vs `state=`.

5. **The cost of a third architecture today is five registries, not the three CLAUDE.md
   names**: `studies/config.NEURAL_MODEL_TYPES`, `optuna_tuning._SUGGESTERS`,
   `optuna_tuning._BUILDERS`, `optuna_tuning._build_inference_model_for`'s if-chain, and
   `studies/runner._FORECASTERS` — plus a top-level `run_monte_carlo_forecast_*` function
   and an `__init__` alias if the current shape is kept.

**The options and their costs.**

- **(A) Keep two top-level functions; add nothing.** Cost: the silent miscrossing stays
  reachable from `plot_utils`, the notebooks and `main_plot.py`; a third architecture adds
  a 7th registry entry; the two `_FORECASTERS`-shaped tables keep the freedom to drift.
  Benefit: zero churn, notebooks untouched, benchmark arithmetic untouched.

- **(B) Keep two functions but make the pairing checkable.** Give each inference model a
  declared stepper (a class attribute, or an `isinstance` assertion inside each stepper)
  so the wrong crossing raises. Cost: one attribute per inference class and one guard per
  stepper (~10 lines); `plot_utils.forecast_from_checkpoint` and `main_plot.py` must then
  either dispatch or be fixed, so it turns a silent wrong number into a red script — which
  is work, and is the point. Benefit: closes the demonstrated bug without touching the
  rollouts or any public name; notebooks keep calling `mc_forecast*` as they do.

- **(C) One `run_monte_carlo_forecast(model, data, ...)` that dispatches on the model.**
  Cost: renames the public surface, so all four live notebooks change in the same commit
  (map's notebook rule) and `test_notebooks_current_api.py` must be kept green;
  `run_monte_carlo_forecast_transformer` needs a deprecation shim or a coordinated edit at
  `optuna_tuning.py:618`, `runner.py:53`, three notebooks. Benefit: one entry point, one
  registry, the miscrossing becomes unrepresentable, `_FORECASTERS` disappears and
  `_build_inference_model_for` stops returning a value its main caller discards.

- **(D) A `Stepper` object mirroring the embedder seam** — the literal reading of the
  ticket's question. Cost: a third abstraction (embedder, backbone, stepper) for exactly
  two implementations that differ in ~15 lines; the shared preamble must be factored out
  first or the seam duplicates it again; it does not by itself prevent the miscrossing
  unless the model also declares which stepper it wants, so it needs (B) anyway.
  Benefit: a genuinely new history mechanism (state-space, retrieval, fixed-window) becomes
  one class rather than one function plus registry entries — but nothing in the thesis's
  scope is that third mechanism today.

Two facts ticket 06 should weigh: **(D) is the only option that pays off if a third
history mechanism arrives, and nothing in the thesis's plan is one**; and **(B) is the
only option that closes the demonstrated `main_plot.py` bug at a cost measured in lines
rather than in notebook edits.** They are not exclusive — (B) is a precondition for (D)
and a subset of (C).

**And `losses.py`: are `FocalLoss` / `SquaredEMDLoss` configured by anything, ever?**
**No.** Not by any live entry point, not by any archived notebook, not by any of the 6
archived study configs under `Studies/`, not by any Optuna summary. Every `loss_type` in
the repo is the string `"cross_entropy"`. Both classes work when called directly
(verified), and both are untested (`tests/test_losses.py` covers only
`compute_class_weights`). The thesis carve-out does not shelter them: nothing they
produced is in `figures/`, `Studies/` or `FOR_ANALYSIS/`. The same verdict extends to
`focal_gamma`, to `build_criterion`'s `weighted_ce` branch, and — by consequence — to
`compute_class_weights`, whose output every live notebook computes, prints and hands to a
loss that discards it.

---

### Cross-lane observations (recorded so they aren't lost; not this lane's to decide)

- **`models/__init__.py:19` is stale**: it describes `panelclv.benchmarks` as "the
  non-neural Pareto/NBD comparator", but `benchmarks/` has held the neural Valendin LSTM
  since ADR-0004. `benchmarks/__init__.py:15-17` correctly says otherwise, so the two
  `__init__` docstrings disagree about what `benchmarks/` contains.
- **`models/__init__.py:46` advertises two aliases no notebook uses** (§1).
- **`refit_full_calibration`'s docstring drifts from its only caller.** It says `n_epochs`
  is "typically the `best_epoch` found by `fit_model`" (`training_utils.py:406`);
  `refit_best_trial` deliberately uses `DEFAULT_REFIT_EPOCHS` instead and explains why
  (`experiment_utils.py:270-273`). It also returns a `FitResult` whose `best_val_loss` /
  `best_val_f1` are NaN and whose `best_epoch` is synthetic — of five fields its caller
  reads one (`checkpoint_path`).
- **A false comment in two live notebooks.** `Study.ipynb` cells 28 and 30 say
  `save_predictions=True` "saves the raw simulations, the mean, and the actuals for each
  customer". `_save_predictions_run:336-341` writes **only** `prediction_mean`.
- **`compute_forecast_metrics` returns `"mape_aggregate_style"`** (`:580`) while
  `CLAUDE.md` names the metric `mape_aggregate`. The key is an on-disk column in every
  archived `metrics.csv` / `results.csv`, so it is floor item 3 — the doc is what is wrong,
  not the code.
- **`_emb_size` is private but imported across modules by a test**
  (`tests/test_embedders.py:17`) — the same misplaced-seam shape ticket 02 found for
  `_seasonal_weekly_multiplier`, one severity lower because the importer is a test.
- **`archive/` holds only `pareto_nbd.py` + `README.md`**, matching ADR-0004's claim about
  the retired MLE variant. Verified; nothing else to reconcile.

### Test coverage of this lane

Dedicated tests cover `embedders.py` (248 lines, by `tests/test_embedders.py`, 290 lines —
both strategies, shapes, the covariate rejection, and a state-dict round-trip),
`benchmarks/valendin_lstm.py` (183, by `tests/test_valendin_lstm.py`, 172 — parameter
counts pinned against the paper's `model.summary()`), and roughly a third of `losses.py`
(`compute_class_weights` only, by `tests/test_losses.py`, 142).

**No dedicated test exists for 2240 of the lane's 2746 lines**, and three gaps are
load-bearing:

- **`monte_carlo_forecasting.py` (619 lines) has no unit test.** It is reached only
  through `tests/test_golden_end_to_end.py`, which runs the **LSTM path only**
  (`run_monte_carlo_forecast`, `simulate_one_path`). `simulate_transformer_path`,
  `run_monte_carlo_forecast_transformer` and `InferenceMultinomialTransformerModel` appear
  in **zero** test files — verified across all ten. The growing-window rollout that
  produces every Transformer number in the thesis is exercised only by
  `scripts/trace_golden_reachability.py`, which is a tracer, not an assertion.
- **`training_utils.py` (473) has no dedicated test.** Nothing tests early stopping,
  the `best_state is None` fallback (`:343-346`), `val_score_start` slicing in isolation,
  the Optuna pruning hook, or `refit_full_calibration` at all — the warm-start step that
  produces the weights every reported forecast uses when `prediction_source="refit"`.
- **`pareto_benchmark.py` (351) has no test in `tests/`.** Its correctness gate is
  `scripts/validate_pareto_benchmark.py`, which needs an R installation with BTYDplus, so
  it cannot run in CI. That is by design (it *is* floor item 2) but it means an ordinary
  `pytest` run says nothing about the Gibbs sampler.

`multinomial_lstm.py` (211) and `multinomial_transformer.py` (305) are covered
incidentally: `tests/test_embedders.py:222-290` builds both and checks forward shapes and
a cross-strategy state-dict guard, and the golden test trains the LSTM end to end.
`losses.py`'s two loss classes and `build_criterion`'s four branches are untested — and,
per §1, unrun.
