# 06 — `feature_engineering.md` §11 says "No per-feature scaling"; standardisation is unconditional

**Status:** ready-for-agent

This is the exact bug the doc-currency test's own docstring records as having been fixed
once. A second instance survived, in the same file.

## Doc claim

`docs/feature_engineering.md:541-542`, under "## 11. Limitations and open extensions":

> - **No per-feature scaling.** Continuous channels are projected raw; prefer bounded AR
>   features, or pre-scale in the panel, when magnitudes differ by orders of magnitude.

## Code reality

`standardize_covariates` puts every non-embedded, non-target channel on mean 0 / std 1,
fitted on the calibration window:

- `src/panelclv/data_preparation/panel_dataset.py:453` — the function
- `src/panelclv/data_preparation/panel_dataset.py:921-934` — called from `prepare_dataset`
  unconditionally; there is no flag to turn it off
- The returned `covariate_stats` are carried into the rollout and re-applied at every step
  (`models/monte_carlo_forecasting.py:172-176`, `:278-282`)

## It also contradicts its own file

`docs/feature_engineering.md` says the opposite twice elsewhere:

- `:48` — the pipeline diagram: `├─ 10. standardise the numeric channels  (calibration-fitted; §5)`
- `:289-319` — a whole subsection, "**Numeric channels are standardised, fitted on
  calibration**", which describes the transform, its two exclusions and its rationale
  correctly.

So §11's bullet is not merely stale against the code; it contradicts §5 of the same chapter,
and the advice it gives ("pre-scale in the panel") would double-standardise.

## Why the test cannot catch it

`tests/test_docs_are_current.py:16-18` names this bug by name, as the reason the test's own
coverage is admitted to be partial:

> this chapter once asserted the package applies "no per-feature standardisation" while
> `standardize_covariates` had been running for a day. Only a reader catches that one.

The gate checks backticked paths and symbols, and this sentence names neither. `git log -S`
puts the surviving bullet in `5149f69` (2026-08-03); the reconciliation commit `ee8bbad`
(2026-08-13) removed the *other* phrasing and left this one.

## Fix

Delete the bullet from §11 and, if the limitation is worth keeping in some form, replace it
with the real one: standardisation is fitted on the calibration window only, so a covariate
whose distribution shifts in the holdout is transformed with stale statistics — which §5
(`:316-319`) already frames as the deliberate choice it is.

Consider also adding a line to §11 pointing at §5, so the two sections cannot drift apart
again silently.
