# Optuna can select on rollout quality instead of validation cross-entropy

**Retired 2026-08-12.** This decision is reversed, and the feature it describes has been
removed from the package. Nothing supersedes it — see "Why it goes" at the foot of this
file. What follows is the decision as it stood, kept for the record.

Teacher-forced validation cross-entropy is cheap and is what training optimises, but
it is blind to the rollout the real forecast uses: a model can score well next-step
and still drift badly over a long horizon, because errors compound when its own
samples become its inputs. `selection_metric="rollout_composite"` instead scores each
trained trial on a leak-free rollout over the validation window, aligning selection
with the metric actually reported.

## Consequences

The composite score is on a different scale from cross-entropy, so a
`rollout_composite` run needs its own fresh study storage — reusing a `val_loss`
study's database compares incomparable numbers. Pruning still acts on per-epoch
cross-entropy, so it only prunes clearly bad trials early. The pseudo-holdout sits
inside the calibration window, so it captures sampling drift and seasonality but not
extrapolation beyond the observed range of covariates.

## Why it goes

Kept rather than deleted, because the idea is sound and someone will propose it again.

It was never reachable from the production path. `studies/runner._run_neural_model`
passes no `selection_metric`, and `StudySuiteConfig` has no field for one, so rollout
selection could only be reached through three `run_optuna_study` cells in the two
`Data_integration` notebooks — never from `scripts/run_studies.py` or any suite cell. It
was used there: a stored study named `lstm_cross_entropy_rollout_composite_20260601_1651`
really ran. But of the **1256** `selection_metric` values recorded under `Studies/`, every
one reads `val_loss` and none reads `rollout_composite`, so no archived result needs
re-tuning and the removal costs nothing already earned.

The claim above that it aligns "selection with the metric actually reported" also held for
only one metric of three. `tuning.weekly_aggregate_rollout_metrics` recomputed RMSE over
customer-summed totals where `models.monte_carlo_forecasting.compute_forecast_metrics`
uses per-cell values — 62x apart — and MAPE under a different estimator (masked at 5.0,
clipped at 300). Only bias agreed. So the alignment the decision rested on was partly
illusory, and the divergence made `compute_forecast_metrics`' claim to be the single
scoring authority false package-wide.

What is genuinely lost, recorded once so it is not rediscovered as a surprise: selection
on rollout quality is gone, so a model that scores well next-step and drifts over a long
horizon is unguarded against again. Restoring it would be a re-implementation, and it
would need the wiring this decision never had.
