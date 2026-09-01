# 09 — The refit docstring's "typically the `best_epoch`" describes a path that does not exist

**Status:** ready-for-agent

## Doc claim

`src/panelclv/training/loop.py:424-425`, in `refit_full_calibration`'s docstring:

> Because the validation window is now folded into training there is nothing left to
> early-stop on, so this trains for exactly `n_epochs` (**typically the `best_epoch` found
> by `fit_model`**) and persists the FINAL-epoch weights — not a best-by-val checkpoint.

## Code reality

`best_epoch` is never read back. The refit always uses a small fixed count:

- `src/panelclv/trials/refit.py:33` — `DEFAULT_REFIT_EPOCHS = 5`
- `src/panelclv/trials/refit.py:79-83` — applied whenever `n_epochs is None`, with the
  reasoning spelled out: "default to a small fixed count … rather than the tuning run's
  epoch count, which can be large and would over-train the warm-started weights"
- `src/panelclv/studies/runner.py:156-161` — never passes `n_epochs` unless the user puts it
  in `refit_kwargs`, so the production path is always 5

`best_epoch` *is* recorded as a trial user-attr (`src/panelclv/tuning/optuna_tuning.py:367`)
— it is simply not what the refit runs.

## Which document is right

The ADR is. `docs/adr/0008-forecast-from-a-refit.md:23`:

> The refit trains a **fixed few epochs** with no validation set and therefore no early
> stopping, so the weights it ends holding are the weights it saves.

So the ADR and the code agree; the training-loop docstring is the outlier, and it is the one
a reader lands on when tracing `refit_full_calibration`.

## Fix

`src/panelclv/training/loop.py:424-425` — replace the parenthetical. Something like: trains
for exactly `n_epochs`, which on the production path is `trials.refit.DEFAULT_REFIT_EPOCHS`
(5, the paper's "several" fine-tuning epochs) rather than the tuning run's `best_epoch`.

Keep the rest of the sentence — "persists the FINAL-epoch weights, not a best-by-val
checkpoint" is correct and is what makes ADR-0007's `to_rollout()` exact on this path.
