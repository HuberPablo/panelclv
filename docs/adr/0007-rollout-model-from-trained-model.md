# A rollout model is obtained from a trained model, never rebuilt beside it

A trained model and the rollout model that forecasts with it are two classes over one
backbone: the training class returns logits for cross-entropy, the rollout class draws a
count from the softmax and threads the recurrent state (see "What the models are" in
`CLAUDE.md`). Their constructor arguments used to be written out separately — three times,
counting the two class bodies and the tuning builders — and a mismatch surfaced only when
the state dict failed to load, which is after the training has finished.

The trained model now hands over its own backbone. `trained.to_rollout()` returns the
paired rollout model, sharing the same weights object. Nothing outside `models/` and
`benchmarks/` names a rollout class, and a mismatch is not expressible.

This deletes the path that rebuilt a rollout model from a stored study's parameters and
loaded a checkpoint into it.

For this to be correct the training loop must leave the **best** weights in the model it
returns, not only on disk. Early stopping keeps training past the best epoch by design, so
a loop that saves a snapshot and never loads it back returns a model that quietly differs
from its own checkpoint.

## Consequences

`to_rollout()` shares the backbone rather than copying it. Sharing is what makes a
mismatch unconstructible, and a copy would double peak memory at the moment a large model
has just finished training.

A stored Optuna study plus a checkpoint, with no live model, can no longer be rebuilt and
loaded — you run the refit's few epochs instead. Consistent with ADR-0008, which makes the
refit the only path to a forecast anyway.

The two-class shape survives in `benchmarks/valendin_lstm.py`, which declares its own
pairing inside the frozen file. Editing a frozen benchmark's *surrounding code* is
permitted (ADR-0004); `scripts/validate_valendin_lstm.py` is the gate that proves the
numbers did not move.

`fit_model` returns a model whose weights match its checkpoint. That property needs its
own test: the golden end-to-end fixture cannot catch its absence, because its two epochs
improve monotonically, so both channels agree there by luck rather than by construction.
