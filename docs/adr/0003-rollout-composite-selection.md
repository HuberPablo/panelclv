# Optuna can select on rollout quality instead of validation cross-entropy

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
