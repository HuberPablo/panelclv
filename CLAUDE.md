# panelclv

Customer-base forecasting for a thesis: predicting per-customer transaction counts
over a holdout window, comparing a Transformer against two frozen benchmarks.

Read `CONTEXT.md` for the vocabulary and `docs/adr/` for decisions before proposing
changes.

## What the models are

These are **classifiers driving a simulator**, not point regressors. Getting this
wrong breaks everything downstream, so it is worth stating plainly:

- The model emits a softmax over transaction-count classes at each period — logits
  of shape `(B, T, K)`, where a count is a **category**, never a quantity.
- It trains by cross-entropy against a class index.
- A forecast is a **rollout**: warm up on the calibration window, then step through
  the holdout one period at a time, sampling a count and feeding that sample back as
  the next period's input. Average many simulated paths. True holdout values are
  never fed in.

Any new model keeps this shape: categorical head, class-index target, evaluation
through sampling-and-averaging.

## Priorities

When choices conflict, resolve in this order:

1. **Correctness** — benchmarks reproduce their source; models train and roll out as
   described in `docs/adr/`.
2. **Reproducibility** — same config and seed gives the same result, and results
   never depend on the order notebook cells were run in.
3. **Dataset-agnostic interfaces** — a model that runs end-to-end on a new panel with
   no code edits beats a better model welded to one dataset. Columns are named in
   `PanelConfig`, never in model code.
4. **Simplicity** — the smallest design satisfying the above.

## Where things live

Subpackages under `src/panelclv/`, split by altitude. A name lives in exactly one of
them and there is no umbrella re-export — import from the subpackage that owns it.

- `models` — architectures under development, their losses, and the Monte Carlo
  simulator (ADR-0002).
- `benchmarks` — frozen reference implementations we reproduce, not develop
  (ADR-0004). Torch is imported lazily here.
- `data_preparation` — panel to model-ready tensors, and leak-free autoregressive
  features.
- `configs` — `PanelConfig`, which carries every column role, window date and
  embedding declaration.
- `training` — the training loop.
- `tuning` — Optuna architecture and covariate-subset search.
- `evaluation` — metrics, plots, forecast diagnostics, prediction I/O.
- `experiments` — glue tying prepare, tune and forecast together.
- `studies` — running many studies across many models and archiving the results.

Each subpackage's `__init__.py` documents its own contents. Read those rather than
expecting this file to list modules.

## Working in this repo

**Before touching features, autoregressive features, or anything read during a
rollout, read `docs/feature_engineering.md`.** Leakage is silent and expensive.

**Metrics.** `models.monte_carlo_forecasting.compute_forecast_metrics` is the single
scoring authority — `rmse`, `bias_percent`, `mape_aggregate`, on per-customer
per-period arrays. Plots, tables and study results all delegate to it so they agree
to the last decimal.

**Comments.** Comment so a thesis reader follows the code without prior context:
the intent of a block, what a tensor's shape means, why a step exists. Explain the
non-obvious.

**Adding a model** touches three places, and missing the second fails only after
training completes:

1. `studies/config.py` — `VALID_MODEL_TYPES`
2. `studies/runner.py` — `_FORECASTERS`
3. `tuning/optuna_tuning.py` — a `suggest_*_params` branch

**Invariants worth knowing before you hit them:**

- The target column appears in both `seq_cols` and `embedded_cols`; its cardinality
  sets the softmax head size.
- `clip_target_upper` caps counts and therefore sets that head size.
- An inference model loads its `state_dict` from the trained model, so their
  constructor arguments must match.

## Environment

The project venv is `/home/virthian/Desktop/Thesis/venvs/thesis_rocm/` (PyTorch on
ROCm). Run code with its interpreter. It is **user-maintained: leave its packages
exactly as they are.** When a dependency is missing, say so and stop.

Data preparation needs only numpy and pandas, so it runs without loading torch.

## Agent skills

### Issue tracker

Issues and specs live as markdown files under `.scratch/<feature>/` in this repo. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, used verbatim. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
