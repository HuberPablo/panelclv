# 02 — A warm-up floor before patience counts

Status: `ready-for-agent`
Blocked by: nothing. Needed only by the `floor50` arm of ticket 03.

## Why

`docs/training-budget.md` §2: with early stopping off, validation loss keeps improving to
epoch 88–247, and the improvements arrive after plateaus of 29–131 non-improving epochs.
Patience 7 cannot survive those, and a larger constant is unreliable — patience 40 found
the later optimum in 1 of 5 electronics replications.

A floor is the smallest change that makes the behaviour deterministic: no run stops before
`min_epochs`, whatever the plateau.

## The change

`src/panelclv/training/loop.py` — `fit_model` gains `min_epochs: int = 0`, and only the
break is gated:

```python
if epoch >= min_epochs and patience_counter >= patience:
    break
```

Best-epoch tracking, the checkpoint and the restored weights are unchanged: the floor
changes how long the loop keeps looking, not what it selects.

`src/panelclv/tuning/optuna_tuning.py` — add `"min_epochs"` to `TRAINING_CONTROLS` and
forward it through `suggest_param` in the `fit_model` call, exactly as `n_epochs` and
`patience` are.

**The pruner has to move with it.** `run_optuna_study` builds
`MedianPruner(n_warmup_steps=3)`, which prunes a trial from epoch 4 — before a floored
trial has shown anything, and 63% of archived electronics trials are already pruned. When
`min_epochs` is set, build the default pruner with `n_warmup_steps=min_epochs`. No new
argument: `run_optuna_study` already accepts a `pruner`.

Docstrings listing the training controls: `studies/config.py` and
`docs/running-a-model.md`.

## Test

`tests/test_training_loop.py`, beside the existing early-stopping test: with
`min_epochs=10`, `patience=2` and a validation loss that never improves, the loop runs 10
epochs rather than 3.

`refit_full_calibration` is untouched — it has no validation set and no stopping rule.
