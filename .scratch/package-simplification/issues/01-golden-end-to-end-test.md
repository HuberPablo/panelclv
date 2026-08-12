# Golden end-to-end reproducibility test

Type: task
Status: resolved

## Question

Nothing in the 1810-line test suite runs the pipeline end to end. The redesign moves
the embedder seam, the registration lists and the autoregressive-feature contract —
exactly the places where a behaviour change surfaces as a slightly different forecast
rather than a crash. Build the net before anything moves.

A tiny synthetic panel through `prepare_dataset` → train (2 epochs, fixed seed) →
rollout → `compute_forecast_metrics`, asserting exact numbers. Small enough for CI,
no GPU.

This is also the only thing that would test priority #2 from `CLAUDE.md` — same config
and seed gives the same result — which currently nothing asserts.

Second deliverable, needed by ticket 05: a **reachability trace** of which `src/`
symbols the golden run actually touches. A symbol the full pipeline never reaches is a
deletion candidate with evidence attached, rather than an argument.

Resolve by recording: where the test lives, what it pins, and the trace's location.

## Answer

Both deliverables landed. Full suite: 159 passed.

### The net — `tests/test_golden_end_to_end.py`

One pinned run of panel → `prepare_dataset` → 2 epochs → checkpoint reload → Monte Carlo
rollout → `compute_forecast_metrics`, on a synthetic Poisson/gamma panel of 24 customers
× 78 weeks. CPU-only, ~15s, no GPU and no dataset on disk. Five tests, each failing for a
different reason: pinned cohort/window shapes, pinned feature-axis order, pinned metrics,
determinism, and a leakage check that the rollout never reads the holdout.

The fixture deliberately exercises the fragile paths — two autoregressive features,
derived time features, an embedded target whose cardinality sets the softmax head size,
and a checkpoint round-trip between the training and inference models.

**One deviation from the ticket, deliberate.** The ticket asked for exact numbers. The
run *is* bit-reproducible — verified identical across separate processes — and
`test_pipeline_is_deterministic` asserts exact array equality, which is the property
`CLAUDE.md` priority #2 actually names and which nothing previously tested. But the
*pinned* numbers are asserted at `rel=1e-6` rather than bit-exact, because CPU float
reduction order is not guaranteed identical across BLAS builds and this repo runs on
ROCm locally, on Colab and on VastAI. 1e-6 is far tighter than any real behaviour change.

Golden values, for reference: `rmse=2.0019012702059444`,
`bias_percent=mape_aggregate_style=247.03757225433526`. The model is tiny and
undertrained on purpose — this pins *what the pipeline computes*, not how well it
forecasts.

### The trace — `scripts/trace_golden_reachability.py`

Static AST inventory of every `def` under `src/panelclv` (206 symbols, 110 public),
matched against a `sys.settrace` record of what the interpreter actually enters.
`coverage` is not installed and the venv is user-maintained, so this is stdlib only.

Four scenarios, not one: **lstm** (the golden pipeline, imported from the test so the two
cannot drift), **transformer**, **valendin_lstm** and **pareto_nbd**. Tracing only the
LSTM would have marked three correct implementations unreached and produced actively
misleading evidence.

Output: `.scratch/package-simplification/reachability.md` (per-module summary) and
`reachability.csv` (one row per symbol). **Reached 79 of 206 symbols; 37 of 110 public.**

**How ticket 05 must read this: reached is proof of life, unreached is not proof of
death.** The scenarios are one small panel with no covariates, no Optuna search, no study
suite and no plotting, so `tuning/`, `studies/` and `evaluation/` are unreached for want
of a caller here, not for want of a purpose. Cross it with the audits' static caller
census — this file answers "definitely live", the audits answer "definitely dead".

The strongest signals, where dynamic and static evidence already agree:

- `evaluation/forecast_run.py` — 0 of 9 symbols reached, and the charting census found no
  caller anywhere in `src`, `scripts`, `notebooks`, `tests` or `docs`. An entire
  111-line module with no evidence of life from either direction.
- `models/losses.py` — `FocalLoss.forward` and `SquaredEMDLoss.forward` unreached; only
  the plain cross-entropy branch of `build_criterion` runs. Whether any config ever
  selects them is ticket 03's question.
- `models/monte_carlo_forecasting.py` — 8 of 10 reached, 0 public unreached. The
  simulator is fully exercised by these four scenarios, which makes it safe ground for
  the rollout-seam decision in ticket 03.

### Two incidental findings, for the audits rather than for here

- **`fit_model(max_trans=...)` means the class *count*, not the maximum class index** —
  `_validate_targets` checks `t_max >= max_trans`. The name says the opposite of what it
  does and cost a debug cycle to discover. Input to ticket 09.
- **`compute_pareto_predictions` emits `divide by zero encountered in log` on short
  chains** over this tiny panel. Plausibly just the small-data regime rather than a
  defect, but ticket 03 should look rather than assume.
