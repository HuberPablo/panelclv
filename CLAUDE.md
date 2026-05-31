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
3. **Package quality** — importable modules under `Models/`, `Data_preparation/`,
   `configs/`; no logic that only lives in a notebook.
4. **Clear interfaces** — schema-driven, dataset-agnostic (`seq_cols`, `input_spec`,
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
(`aggregate_bias`, `bias_percent`). These live in `Models/evaluation_utils.py` and
`Models/monte_carlo_forecasting.py`.

## Critical modeling distinction

The Valendin LSTM is **not** a point regressor. It is a **classifier + autoregressive
Monte Carlo simulator**:

- The model outputs a **softmax over transaction-count classes** `P(y=0..K-1)` at each
  step (`MultinomialLSTMModel`, logits `(B, T, max_trans)`).
- It is trained by **classification loss** — cross-entropy / NLL (optionally weighted CE,
  focal, or squared-EMD), not MSE (`Models/training_utils.py`, `Models/losses.py`).
- Forecasting = **autoregressive Monte Carlo simulation**
  (`Models/monte_carlo_forecasting.py`): warm up the LSTM state on the full calibration
  window, then step through the holdout one period at a time, **sampling** a count class
  from the multinomial output and feeding that sample back as the next step's input
  (true holdout targets are never fed in). Average many simulated paths to get the
  expected count per customer per step.

Keep this distinction intact in any extension (including the Transformer): the head is
categorical over counts, the training target is a class index, and evaluation runs through
the sampling-and-averaging simulator — not a single deterministic forward pass.

## Repository layout

- `Models/` — the package. `multinomial_lstm.py`, `multinomial_transformer.py`,
  `pareto_nbd.py`, `pareto_paper.py`, `monte_carlo_forecasting.py`, `training_utils.py`,
  `losses.py`, `evaluation_utils.py`, `optuna_tuning.py`, `plot_utils.py`. `__init__.py`
  is the public API (note `mc_forecast` is an alias for `run_monte_carlo_forecast`).

  **Two Pareto/NBD benchmarks** (same `(train_panel, holdout_length, ...) → (N, H)`
  contract, drop-in interchangeable):
  - `pareto_nbd.compute_pareto_predictions` — frequentist **MLE** via `lifetimes`
    (fast; the default benchmark).
  - `pareto_paper.compute_pareto_paper_predictions` — **hierarchical-Bayes MCMC**, a
    pure-NumPy port of R's **BTYDplus** (`pnbd.mcmc.DrawParameters`), the estimator
    Valendin et al. actually use. Validated against the installed R package by
    `scripts/validate_pareto_paper.py` (aggregate within ~0.25%, per-customer corr
    ~0.99). In `plot_weekly_aggregated` / `metrics_table` pass `pareto_nbd_benchmark=True`
    (MLE, "Pareto/NBD") and/or `pareto_paper_benchmark=True` (HB, "Pareto/NBD (HB)").
- `Data_preparation/` — `dynamic_panel_dataset.py` (`prepare_dataset` → model-ready
  `data` dict) and `Datastet_building.py` (raw → panel).
- `configs/` — `transformations_spec.py`: INPUT_SPEC validation + JSON save/load.
- `inputs_configs/` — saved INPUT_SPEC JSONs (e.g. `full_transactions_gender.json`).
- `Datasets/` — source panels (`.Rdata`, `.csv`, `.npz`).
- `Data_integration_LSTM.ipynb`, `Data_integration_TRANSFORMER.ipynb`,
  `Data_integration.ipynb` — orchestration / experiment notebooks (thin glue over `Models/`).
- `Fine_tuning_optuna/` — Optuna study databases.
- `Original_paper_model/` — reference notebooks for the paper's setup.

## How the pieces fit (canonical workflow)

1. Build/load a customer-period panel (one row per customer per period).
2. `data = prepare_dataset(panel, DATA_CONFIG, FEATURE_SCHEMA, TIME_FEATURES)` →
   `calibration`, `holdout`, `samples`, `targets`, `seq_cols`, `target_col`, ...
3. Wrap `samples`/`targets` in DataLoaders; train with
   `MultinomialLSTMModel(seq_cols, input_spec, ...)` + `fit_model(...)`
   (cross-entropy on `(B, T, K)` logits vs `(B, T)` class targets).
4. Load the trained weights into `InferenceMultinomialLSTMModel(..., mode="sample")`.
5. `forecast = run_monte_carlo_forecast(inference_model, data, n_simulations=...)` then
   `compute_forecast_metrics(forecast["actual"], forecast["prediction_mean"])`.

The Transformer mirrors this exact contract (`MultinomialTransformerModel` /
`InferenceMultinomialTransformerModel`), so the data, training loop, and simulator are shared.

## Optuna model selection (`Models/optuna_tuning.py`)

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

How the rollout selection stays leak-free: for the validation customers
(`val_idx`), the **last `rollout_horizon` weeks of the calibration window** are
carved off as a pseudo-holdout (`calibration[:, :-V]` warms up, `calibration[:, -V:]`
is scored). The real `data["holdout"]` is **never** read during tuning. Each trial
is re-sliced to its own feature subset (`select_features`), the matching inference
model is rebuilt + checkpoint-loaded, and the existing MC forecaster is reused. The
score is `weekly_aggregate_rollout_metrics(...)` — a scale-normalised composite of
RMSE + masked MAPE + |bias| (weights `rollout_weight_{rmse,mape,bias}`, default
`1.0 / 0.5 / 0.3`); lower is better, so the study stays `direction="minimize"`. All
sub-metrics (`rollout_rmse`, `rollout_mape`, `rollout_bias_percent`, `rollout_score`)
and `val_loss` are logged as trial user-attrs.

Caveats: (a) the composite score is a **different scale** than cross-entropy — a
`rollout_composite` run needs its **own fresh study / storage**, never a `val_loss`
study's DB; (b) the pruner still acts on per-epoch CE, so it only prunes clearly
bad-CE trials early; (c) the pseudo-holdout sits *inside* calibration, so it captures
sampling-drift and seasonality but **not** out-of-range known-future extrapolation
(that would require peeking at holdout covariates). Requires `rollout_data=data_full`
and `val_idx=`; default `rollout_horizon=52` (match the real holdout length).
`rollout_horizon` is validated up front — it must satisfy `0 < horizon < T_CAL`
(else a `ValueError` fires before any training, not mid-study), and a horizon past
`T_CAL/2` warns that the warm-up prefix is too short to trust the score.

## Gotchas

- **Hardcoded path:** `configs/transformations_spec.py` sets
  `DEFAULT_INPUT_SPEC_DIR` to a local absolute path
  (`/home/virthian/Desktop/Thesis/Package_Notebook/inputs_configs`). On any other machine,
  pass `directory=` explicitly to `save_input_spec` / `load_input_spec` / `list_input_specs`,
  or fix the constant.
- **Target column rules:** `target_col` must be in **both** `seq_cols` and
  `input_spec["embedded_cols"]`; its cardinality sets the softmax head size (`max_trans`).
  If you use `clip_target_upper`, it must be strictly less than that cardinality.
- **Shared backbone:** the inference model loads its `state_dict` from the trained
  `MultinomialLSTMModel` — keep their constructor args identical.

## Dependencies

`torch`, `numpy`, `pandas`, `scikit-learn`, `optuna`; `lifetimes` for the MLE
Pareto/NBD; `wandb` optional (lazily imported). There is no `requirements.txt`/
`pyproject.toml` yet.

`Models/pareto_paper.py` (the HB-MCMC Pareto/NBD) is **pure NumPy/SciPy — no R needed
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