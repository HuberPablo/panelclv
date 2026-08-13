# Adding a model means adding one registry entry

Adding a model used to touch three places — `VALID_MODEL_TYPES`, `_FORECASTERS`, and a
`suggest_*_params` branch — and missing one failed only after training completed. Seven
separate enumerations of the model set had in fact accumulated across `src/`, and an
eighth copy in `studies/analysis.py` drifted out of sync and silently collapsed the
Valendin benchmark's across-study spread to a single study. Counting copies was the
problem; no single copy was.

A model is now **one entry in one registry table**, holding its search space, its builder
and the rollout function it forecasts through. Every model-type list in the package
derives from that table's keys.

The table's fields are optional, because Pareto/NBD is a valid model type with no search
space, no builder and no rollout — its entry exists so the enumerations still derive from
one place. Whether a model is neural is *read off* the entry ("it has a training builder")
rather than restated as a second list, because that restatement is exactly the copy that
drifted.

## Consequences

A model type is registered everywhere or nowhere; there is no state in which it is known
to the tuner and unknown to the forecaster. The neural / non-neural distinction cannot
drift, because there is nothing left to keep in sync.

The registry is its own subpackage. Both `models/` and `studies/` were blocked by real
import cycles, and a repository-root module was rejected as clutter.

Entries hold **direct** references. An earlier design used lazy ones so a `model_type`
could be validated without importing torch; that goal was dropped, so the indirection
bought nothing.

Pareto/NBD's entry is declarative. `studies/runner.py` keeps separate neural and
deterministic paths, which differ in more than the forecaster — no Optuna study, no refit,
one prediction rather than several.
