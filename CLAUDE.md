# CLAUDE.md

Guidance for working in this repository. This is a thesis project; the goal below
takes precedence over generic "make it fancy" instincts.

## Project overview

This is a modular customer-base forecasting package for predicting per-customer
transaction counts over a holdout window. It contains three model families:
a multinomial **LSTM**, a multinomial **Transformer**, and a **Pareto/NBD** benchmark.

The thesis objective is twofold and **ordered**:

1. **Reproduce the Valendin et al. customer-base LSTM workflow first** — data prep,
   training, autoregressive Monte Carlo simulation, and evaluation — and confirm it
   tracks the benchmark before doing anything else.
2. **Then extend it** into a clean, package-quality framework that adds a Transformer
   variant on the same data/eval/simulation contract.

Priority is a **reliable, reusable, dataset-agnostic package** over model complexity.
A simple model that runs end-to-end on a new dataset with no code edits is worth more
than a sophisticated one that only works on a single hardcoded panel.

## Main development principle

When choices conflict, resolve them in this order:

1. **Correctness** — the model trains and simulates the way Valendin et al. describe.
2. **Reproducibility** — same config + seed → same result; nothing depends on notebook
   cell run-order.
3. **Package quality** — importable modules under `panelclv/models/`, `panelclv/training/`,
   `panelclv/tuning/`, `panelclv/evaluation/`, `panelclv/benchmarks/`, `panelclv/experiments/`,
   `panelclv/data_preparation/`, `panelclv/configs/`; no logic that only lives in a
   notebook.
4. **Clear interfaces** — schema-driven, dataset-agnostic (`seq_cols`, `embedded_cols`,
   `FEATURE_SCHEMA`), not column names baked into model code.
5. **Simplicity** — prefer the smallest design that satisfies the above.
6. **Robustness** — validate inputs at boundaries with clear errors (the modules already
   do this; keep it up).
7. **Interpretability** — favour outputs and metrics a thesis reader can reason about.

**Do not** make complex architecture changes until the basic LSTM → data → training →
evaluation → Monte Carlo simulation workflow runs cleanly end-to-end.

## Code comments

When proposing code, comment it well enough that the user can follow it without prior
context. Explain the *why* and the non-obvious *what* — the intent of a block, what a
tensor shape means, why a step is needed — not trivia that restates the syntax. Favour
clear, instructive comments on data-prep, model, training, and simulation logic so the
code reads as a thesis-quality explanation, not just a working script.

Primary metrics: **RMSE**, **MAPE** (where a positive denominator makes it meaningful —
see `mape_positive` / `cumulative_mape`), and **aggregate bias / tracking quality**
(`aggregate_bias`, `bias_percent`). These live in `panelclv/evaluation/evaluation_utils.py`
and `panelclv/models/monte_carlo_forecasting.py`.

## Critical modeling distinction

The Valendin LSTM is **not** a point regressor. It is a **classifier + autoregressive
Monte Carlo simulator**:

- The model outputs a **softmax over transaction-count classes** `P(y=0..K-1)` at each
  step (`MultinomialLSTMModel`, logits `(B, T, max_trans)`).
- It is trained by **classification loss** — cross-entropy / NLL (optionally weighted CE,
  focal, or squared-EMD), not MSE (`panelclv/training/training_utils.py`,
  `panelclv/models/losses.py`).
- Forecasting = **autoregressive Monte Carlo simulation**
  (`panelclv/models/monte_carlo_forecasting.py`): warm up the LSTM state on the full calibration
  window, then step through the holdout one period at a time, **sampling** a count class
  from the multinomial output and feeding that sample back as the next step's input
  (true holdout targets are never fed in). Average many simulated paths to get the
  expected count per customer per step.

Keep this distinction intact in any extension (including the Transformer): the head is
categorical over counts, the training target is a class index, and evaluation runs through
the sampling-and-averaging simulator — not a single deterministic forward pass.

## Repository layout

**src-layout.** The importable package lives under **`src/panelclv/`** — the repo root
is never on `sys.path` by accident, so you must install the package to import it (this
surfaces packaging bugs before they ship). `notebooks/`, `scripts/` and `tests/` sit
*outside* `src/` and are not part of the wheel. The module paths below are all relative
to `src/` (i.e. `src/panelclv/models/` etc.), but you still import them as `panelclv.models`.

The package is split by *altitude* into subpackages so each holds one concern. Each
subpackage has an `__init__.py` re-exporting its public names — import from the
subpackage root, e.g. `from panelclv.tuning import run_optuna_study`, `from
panelclv.evaluation import plot_weekly_aggregated`. There is **no umbrella re-export**
across subpackages; a name lives in exactly one of them.

- `panelclv/models/` — the **model definition only**: `multinomial_lstm.py`,
  `multinomial_transformer.py`, `losses.py`, and `monte_carlo_forecasting.py` (the AR
  simulator stays here because, per the Valendin design, the simulator *is* the model's
  forecast mechanism, not a post-hoc step). `__init__.py`'s `__all__` is a curated
  headline set for the model family; the rest stay importable by explicit name. Note
  `mc_forecast` is an alias for `run_monte_carlo_forecast` (both names exported).
- `panelclv/training/` — `training_utils.py`: the training loop (`fit_model`,
  `train_one_epoch`, `validate_one_epoch`, `FitResult`).
- `panelclv/tuning/` — `optuna_tuning.py`: Optuna architecture / covariate-subset search
  (`run_optuna_study`, `select_features`, ...). Model-aware but a layer above the model.
- `panelclv/evaluation/` — metrics + diagnostics: `evaluation_utils.py` (`compute_metrics`,
  `rmse`, `mae`, `mape_positive`, `aggregate_bias`) and `plot_utils.py` (weekly aggregation,
  `plot_weekly_aggregated`, `metrics_table`, `alignment_check`, prediction CSV I/O).
- `panelclv/experiments/` — `experiment_utils.py`: thin orchestration glue
  (`make_loaders`, `make_data_builder`, `build_inference_from_trial`) tying
  `prepare_dataset` → Optuna → forecast. Sits on top of `models` + `tuning`.
- `panelclv/benchmarks/` — the non-neural comparators. **Two Pareto/NBD benchmarks**
  (same `(train_panel, holdout_length, ...) → (N, H)` contract, drop-in interchangeable):
  - `pareto_nbd.compute_pareto_predictions` — frequentist **MLE** via `lifetimes`
    (fast; the default benchmark).
  - `pareto_paper.compute_pareto_paper_predictions` — **hierarchical-Bayes MCMC**, a
    pure-NumPy port of R's **BTYDplus** (`pnbd.mcmc.DrawParameters`), the estimator
    Valendin et al. actually use. Validated against the installed R package by
    `scripts/validate_pareto_paper.py` (aggregate within ~0.25%, per-customer corr
    ~0.99). In `plot_weekly_aggregated` / `metrics_table` pass `pareto_nbd_benchmark=True`
    (MLE, "Pareto/NBD") and/or `pareto_paper_benchmark=True` (HB, "Pareto/NBD (HB)").
- `panelclv/data_preparation/` — `dynamic_panel_dataset.py` (`prepare_dataset` →
  model-ready `data` dict) and `dataset_building.py` (raw → panel).
- `panelclv/configs/` — `transformations_spec.py`: INPUT_SPEC validation + JSON
  save/load; `panel_config.py`: the `PanelConfig` dataclass.
- `inputs_configs/` — saved INPUT_SPEC JSONs (e.g. `full_transactions_gender.json`).
- `Datasets/` — source panels (`.Rdata`, `.csv`, `.npz`).
- `notebooks/` — all orchestration / experiment notebooks (outside the package). The
  `Data_integration_*_v2.ipynb` are the current helper-based notebooks (thin glue over
  the `panelclv` subpackages); the un-suffixed ones are kept as reference, and
  `dataset_building.ipynb` (raw → panel) was pulled out of the package. Each notebook
  starts with a small repo-root bootstrap cell (walks up to `pyproject.toml`, `chdir`s
  there so `Datasets/...` paths resolve, and adds `src/` to `sys.path` as a fallback).
- `scripts/` — loose driver scripts (not shipped): `validate_pareto_paper.py`,
  `main_plot.py`, `main_plot_covar.py`.
- `tests/` — pytest smoke tests (`pip install -e ".[dev]"` for the runner).
- `Fine_tuning_optuna/` — Optuna study databases.

## How the pieces fit (canonical workflow)

1. Build/load a customer-period panel (one row per customer per period).
2. `data = prepare_dataset(panel, DATA_CONFIG, FEATURE_SCHEMA, TIME_FEATURES)` →
   `calibration`, `holdout`, `samples`, `targets`, `seq_cols`, `target_col`, ...
3. Wrap `samples`/`targets` in DataLoaders; train with
   `MultinomialLSTMModel(seq_cols, embedded_cols, ...)` + `fit_model(...)`
   (cross-entropy on `(B, T, K)` logits vs `(B, T)` class targets).
4. Load the trained weights into `InferenceMultinomialLSTMModel(..., mode="sample")`.
5. `forecast = run_monte_carlo_forecast(inference_model, data, n_simulations=...)` then
   `compute_forecast_metrics(forecast["actual"], forecast["prediction_mean"])`.

The Transformer mirrors this exact contract (`MultinomialTransformerModel` /
`InferenceMultinomialTransformerModel`), so the data, training loop, and simulator are shared.

## Optuna model selection (`panelclv/tuning/optuna_tuning.py`)

`run_optuna_study(...)` tunes architecture + (optionally) the covariate subset
(`removable_features`). The **selection objective** is configurable:

- `selection_metric="val_loss"` (default) — teacher-forced next-step **validation
  cross-entropy**, the same loss the training loop optimises. Cheap, but blind to
  the autoregressive sampling rollout the final forecast actually uses, so it can
  pick feature sets / architectures that look fine on validation yet **drift badly
  at forecast time** (e.g. keeping an out-of-range trend feature and dropping the
  seasonal/recency signals — over-predicting with no decay).
- `selection_metric="rollout_composite"` — after training each trial, run a
  **leak-free validation Monte Carlo rollout** and select on its weekly-aggregate
  forecast quality. This aligns model selection with the reported metric.

Both selection metrics score the **same temporal validation window** (see
"Validation split" below): the calibration tail after `validation_start`. For
`val_loss`, `fit_model` is given `val_score_start` so the teacher-forced CE is
computed on that suffix only. For `rollout_composite`, the **last `n_val_periods`
weeks of the calibration window** (i.e. everything after `validation_start`) are
carved off as a pseudo-holdout **for all customers** (`calibration[:, :-V]` warms
up, `calibration[:, -V:]` is scored). The real `data["holdout"]` is **never** read
during tuning. Each trial is re-sliced to its own feature subset (`select_features`),
the matching inference model is rebuilt + checkpoint-loaded, and the existing MC
forecaster is reused. The score is `weekly_aggregate_rollout_metrics(...)` — a
scale-normalised composite of RMSE + masked MAPE + |bias| (weights
`rollout_weight_{rmse,mape,bias}`, default `1.0 / 0.5 / 0.3`); lower is better, so the
study stays `direction="minimize"`. All sub-metrics (`rollout_rmse`, `rollout_mape`,
`rollout_bias_percent`, `rollout_score`) and `val_loss` are logged as trial user-attrs.

Caveats: (a) the composite score is a **different scale** than cross-entropy — a
`rollout_composite` run needs its **own fresh study / storage**, never a `val_loss`
study's DB; (b) the pruner still acts on per-epoch CE, so it only prunes clearly
bad-CE trials early; (c) the pseudo-holdout sits *inside* calibration, so it captures
sampling-drift and seasonality but **not** out-of-range known-future extrapolation
(that would require peeking at holdout covariates). Requires `rollout_data=data_full`.
`rollout_horizon` defaults to `data_full["n_val_periods"]` (the validation window, so
the two selection metrics stay comparable); pass an int to override. It is validated
up front — it must satisfy `0 < horizon < T_CAL` (else a `ValueError` fires before any
training, not mid-study), and a horizon past `T_CAL/2` warns that the warm-up prefix
is too short to trust the score.

## Validation split (temporal, not customer-wise)

The train/validation split is a **time window over all customers**, set by
`validation_start` (a required `PanelConfig` date alongside `training_*`/`holdout_*`,
checked `training_start < validation_start <= training_end`). `prepare_dataset` maps
it to a calibration period index `val_start_idx = s` and returns it (plus
`n_val_periods`). Weights train only on periods `[0, s)` (`make_loaders` truncates the
train loader to `samples[:, :s-1]`); the validation window `[s, T_CAL)` is fed in full
for warm-up but scored only on the suffix via `val_score_start = s-1`. There is **no**
customer-wise split — `make_loaders(data, batch_size)` / `make_data_builder(data_full)`
take no `train_idx`/`val_idx`. **This is a deliberate exception** to "reproduce
Valendin et al. first": their reference impl uses a random 10%-of-customers split, which
the thesis judges wrong for model selection (it validates on the same periods it trains
on). `compute_class_weights(data)` likewise weights on the training prefix
(`training_only=True`) so the held-out window never leaks into the loss.

**Final retrain (paper step, optional).** `experiments.refit_best_trial(study,
data_full, model_type, ...)` warm-starts the selected checkpoint and fine-tunes it for
a few big-batch epochs on the **full** calibration window (validation tail included) via
`training.refit_full_calibration`, then returns `(inference_model, data_best)` with the
refit weights. Valendin's *paper* does this; their *GitHub* does not — so it is a
notebook flag (`REFIT_ON_FULL_CALIBRATION`, default on). Warm-start = keep the tuned
weights and keep optimising, not restart from scratch.

## Gotchas

- **INPUT_SPEC directory:** `panelclv/configs/transformations_spec.py` has no default location
  (`DEFAULT_INPUT_SPEC_DIR = None`). `save_input_spec` / `load_input_spec` /
  `list_input_specs` therefore **require** an explicit `directory=` (e.g. the repo's
  `inputs_configs/`) and raise a clear `ValueError` if it is omitted. (This replaced a
  previously hardcoded absolute path that only resolved on one machine.)
- **Target column rules:** `target_col` must be in **both** `seq_cols` and
  `embedded_cols`; its cardinality sets the softmax head size (`max_trans`).
  If you use `clip_target_upper`, it must be strictly less than that cardinality.
- **Shared backbone:** the inference model loads its `state_dict` from the trained
  `MultinomialLSTMModel` — keep their constructor args identical.

## Dependencies

`torch`, `numpy`, `pandas`, `scikit-learn`, `optuna`, `matplotlib`; `lifetimes` for the
MLE Pareto/NBD; `wandb` optional (lazily imported; the `wandb` extra). These are declared
in **`pyproject.toml`** — `pip install -e .` from the repo root makes the `panelclv`
package and all its subpackages (`panelclv.models`, `panelclv.training`, `panelclv.tuning`,
`panelclv.evaluation`, `panelclv.benchmarks`, `panelclv.experiments`,
`panelclv.data_preparation`, `panelclv.configs`) importable with no `sys.path` hacks. The
`[tool.setuptools.packages.find]` `include = ["panelclv*"]` auto-discovers new subpackages,
so adding one needs no packaging edits.

`panelclv/benchmarks/pareto_paper.py` (the HB-MCMC Pareto/NBD) is **pure NumPy/SciPy — no R needed
at run time**. R is only needed to *re-validate* the port: `scripts/validate_pareto_paper.py`
shells out to `Rscript` with the **BTYDplus** package (installed here: R 4.6,
`~/R/x86_64-pc-linux-gnu-library/4.6`). Skip that script and `pareto_paper` runs anywhere.

## Environment / venv

The project venv is **`/home/virthian/Desktop/Thesis/venvs/thesis_rocm/`** (PyTorch on ROCm).
Use its interpreter to run code and tests, e.g.
`/home/virthian/Desktop/Thesis/venvs/thesis_rocm/bin/python <script>`
(running `prepare_dataset` / data-prep needs only numpy + pandas, so it won't load torch).

**Do not modify this venv** — it is user-maintained. Never `pip install`, upgrade, or remove
packages in it. If a dependency is missing, tell the user instead of changing the environment.