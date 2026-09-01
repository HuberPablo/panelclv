# 07 — ADR-0004 and the benchmark's own docstring say opposite things about what it inherits

**Status:** ready-for-agent

Of everything in this audit, this is the one most likely to mislead a thesis reader — it is
about what the word "frozen" protects.

## Doc claim

`docs/adr/0004-frozen-reference-implementations.md:25-31`:

> ## Deviations from Valendin et al.
>
> Their code is in `Original_paper_model/`. These departures describe **our** model in
> `models/`, **not the benchmark** — `benchmarks/valendin_lstm.py` is the published
> architecture, and **the whole point of it being a separate module is that it does not
> inherit our choices**:
>
> - **Validation split** — temporal rather than a random 10% of customers (ADR-0001).
> - **Tuning** — Optuna over architecture and feature subset; they use fixed sizes.

## Code reality

The benchmark inherits both.

`src/panelclv/benchmarks/valendin_lstm.py:30-32` — the module's own docstring says so
directly, and it is the one that matches the code:

> Deliberate departures that stay
> -------------------------------
> Temporal validation split (ADR-0001) and Optuna tuning. Everything else matches.

Traced:

- `valendin_lstm` is a neural registry entry with a search space
  (`src/panelclv/registry/model_registry.py:369-378`), so `is_neural("valendin_lstm")` is
  `True`.
- `src/panelclv/studies/runner.py:59` therefore routes it into `_run_neural_model`, which
  calls `run_optuna_study` (`:129-144`) over `make_data_builder(...)` → `split_calibration`
  — the sole enforcement point of ADR-0001's temporal split
  (`src/panelclv/trials/loaders.py:102-129`) — and then `refit_best_trial` (`:162`).
- Confirmed in stored output:
  `Studies/seasonal_4x4x10__ValendinLSTM/Dataset_5_80__Dataset_9/ValendinLSTM/config.json`
  records `{"name": "ValendinLSTM", "model_type": "valendin_lstm", "n_trials": 20}`, under a
  suite whose `panel_config` sets `"validation_start": "2000-01-01"`. The benchmark was
  tuned, over a temporal split, twenty trials at a time.

## ADR-0004 also contradicts itself

`docs/adr/0004-frozen-reference-implementations.md:19-21`, four lines above the passage
quoted:

> Everything around the architecture — data preparation, embeddings as configured, **the
> training loop, tuning**, the simulator, evaluation — is shared infrastructure applied
> identically to every model. That is what makes a comparison isolate architecture.

That is the correct reading, and the "Deviations" preamble undercuts it.

## Fix

The code's position is the defensible one: freeze the *architecture*, share the *protocol*,
which is precisely what makes the comparison isolate architecture. So the ADR should say it.

Rewrite the preamble at `:25-28` to something like: these departures apply to every model
here, benchmark included — what the benchmark does **not** inherit is our architecture, our
embedding strategy or our covariates. Then keep the three bullets, since all three are still
true departures from the paper; only the "not the benchmark" framing is wrong.

Note that the third bullet (**Covariates**) *is* benchmark-specific and is handled correctly
further down (`:33-38`, "The covariate line is settled for the benchmark: **it takes
none**"), so the rewrite has to keep the bullets' differing scopes distinguishable.

## Related

Issue `10` — ADR-0005 has the mirror-image framing problem about which embedder is "ours".
