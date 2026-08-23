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
- The target's own column is one of the model's inputs, and the number of classes it
  can take is the size of that softmax head. `Embedder` refuses to build a model
  where those disagree.
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
- `registry` — the one table declaring every model: its search space, its builder and
  the rollout it forecasts through (ADR-0006).
- `benchmarks` — frozen reference implementations we reproduce, not develop
  (ADR-0004).
- `data_preparation` — panel to model-ready tensors, and leak-free autoregressive
  features.
- `configs` — `PanelConfig`, which carries every column role, window date and
  embedding declaration.
- `training` — the training loop.
- `tuning` — Optuna architecture and covariate-subset search.
- `evaluation` — metrics, plots and forecast diagnostics.
- `predictions` — the wide per-customer CSV every forecast is written in and read
  back from. A leaf: it imports nothing from the package, which is what lets the
  model layer write a forecast without naming anything above it (ADR-0002).
- `trials` — assembling and refitting one trial: the temporal calibration split
  (ADR-0001) and the full-calibration refit every forecast comes from (ADR-0008).
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

**`scripts/`.** A script there is either a live entry point, a benchmark gate, or a
documented tool. A one-off check goes in the commit that needed it and is deleted with
it.

**Comments.** Comment so a thesis reader follows the code without prior context:
the intent of a block, what a tensor's shape means, why a step exists. Explain the
non-obvious.

**Adding a model** touches one place: an entry in the model registry, holding its
search space, its builder and the rollout function it forecasts through. Every
model-type list in the package derives from that table's keys, and whether a type
is neural is read off the entry rather than restated.

## Environment

Run code with the project venv's interpreter, never the system Python. Which venv
that is depends on the machine — a ROCm build on the workstation, CPU-only torch on
a VM or a rented GPU box — so no path is written down here. Take it from an
activated `$VIRTUAL_ENV`; if nothing is activated, ask which interpreter to use
rather than guessing one.

The venv is **user-maintained: leave its packages exactly as they are.** When a
dependency is missing, say so and stop.

## Agent skills

### Issue tracker

Issues and specs live as markdown files under `.scratch/<feature>/` in this repo. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical triage roles, used verbatim. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
