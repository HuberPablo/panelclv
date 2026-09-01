# 02 — A Valendin-benchmark forecast is saved into a directory named `lstm_…`

**Status:** needs-triage

A stored result that does not name its own model is the failure ADR-0004 already legislated
against once.

## Doc claim

`docs/adr/0006-one-registry-entry-per-model.md:11-12`:

> A model is now **one entry in one registry table** … **Every model-type list in the
> package derives from that table's keys.**

`docs/adr/0004-frozen-reference-implementations.md:71-73`, on why an estimator's stored
output must name it:

> Study results produced before that decision are stored under `ParetoNBD_MLE`, never
> `ParetoNBD`: the two are different models, and a stored result that does not name its
> estimator cannot be defended later.

## Code reality

The two rollout functions each hardcode a model-type string, which becomes the run
directory's name:

- `src/panelclv/models/monte_carlo_forecasting.py:495` — `model_type="lstm"`
- `src/panelclv/models/monte_carlo_forecasting.py:533` — `model_type="transformer"`
- `src/panelclv/models/monte_carlo_forecasting.py:326` —
  `tag = run_name if run_name else model_type`, then
  `create_run_directory(output_dir, f"{tag}_n{n_simulations}_seed{seed_label}")` (`:328`)

But `forecast_recurrent` is the declared rollout for **two** registered types, not one:

- `src/panelclv/registry/model_registry.py:343` — `lstm` → `rollout=forecast_recurrent`
- `src/panelclv/registry/model_registry.py:377` — `valendin_lstm` → `rollout=forecast_recurrent`

So `rollout_for("valendin_lstm")(model, data, save_predictions=True, output_dir=…)` writes
its predictions into `lstm_n{N}_seed{S}/`. The benchmark's output is labelled as the
developed model it exists to be compared against.

These are also the only two hardcoded model-type values left in `src/` — the registry
otherwise genuinely is the single source (`MODEL_TYPES = tuple(MODEL_REGISTRY)`,
`registry/model_registry.py:386`).

## Reachability

Latent, not live. `studies/runner.py:174` calls `save_predictions_to_csv` directly with a
path from `studies/layout.py`, so the suite never goes through this branch, and no archived
result under `Studies/` is mislabelled by it. It is reachable from a notebook or a script
passing `save_predictions=True`.

## Fix options

**(a) Take the tag from the caller.** `forecast_recurrent` / `forecast_attention` already
accept `run_name`; the defaulting could simply require it when `save_predictions=True`,
rather than falling back to a guessed model type.

**(b) Pass the real type through.** The rollout is looked up by type
(`rollout_for(model_type)`), so the type is known at the call site and could be threaded in
instead of being restated inside the function.

**(c) Derive it from the model.** The rollout model knows its own class; a name read off it
cannot disagree with the weights being rolled out.

## Secondary, same theme

`src/panelclv/studies/config.py:55-57` restates the four registry keys in prose. The
validation itself reads `MODEL_TYPES` (`studies/config.py:164`), so nothing breaks — but it
is a prose copy of the list ADR-0006 exists to keep singular.
