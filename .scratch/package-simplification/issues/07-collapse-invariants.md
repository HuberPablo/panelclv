# Which documented invariants get collapsed, and how

Type: grilling
Status: open
Blocked by: 06

## Question

`CLAUDE.md` warns about footguns that exist because structure is missing. A warning in
a docs file is a workaround; the target architecture is the chance to delete the
footgun instead. Decide which of these get collapsed, and what replaces them:

- **Three-place model registration** — `VALID_MODEL_TYPES`, `_FORECASTERS`, and a
  `suggest_*_params` branch, "missing the second fails only after training completes".
  A fourth copy in `studies/analysis.py` had already drifted and silently corrupted the
  Valendin benchmark's across-study spread; it was fixed during charting, which is the
  argument that the count is the problem rather than any one copy.
- **State-dict must match constructor arguments** — an inference model loads weights
  from the trained model, so their constructor arguments must agree. Nothing enforces
  it; a mismatch is a runtime failure after training. `scripts/migrations/rename_embedder_checkpoint_keys.py`
  exists because this bit once already.
- **The target column appears in both `seq_cols` and `embedded_cols`**, and
  `clip_target_upper` sets the softmax head size.

The test for any new abstraction: it must let a warning be deleted from `CLAUDE.md`.
If it doesn't, it isn't earning its place.

## Comments

**The state-dict invariant has a concrete cause and a cheap structural fix — recorded 2026-08-12.**

Why it exists at all: `MultinomialLSTMModel` and `InferenceMultinomialLSTMModel` are two
*independently constructed* classes over one shared `_MultinomialLSTMBackbone`. The
backbone already returns `(logits, state)`; the wrappers differ only in `forward` — the
training one discards the state and returns logits, the rollout one threads the state,
softmaxes and draws from `Categorical`. Their constructor signatures are identical and
written out twice. That duplication *is* the invariant.

Options, judged by this ticket's own test (does it let a `CLAUDE.md` warning be deleted?):

- **(a) Leave both, add a test gating the constructor match.** Tests the problem rather
  than removing it. The warning stays.
- **(b) `InferenceX.from_trained(model)`** — the rollout class takes the trained model's
  backbone directly, so a mismatch becomes unconstructible. One classmethod. **Deletes the
  warning.**
- **(c) Collapse to one class with a second method** (`forward` for logits, `sample_step`
  for the rollout). Cleanest on paper — but see the constraint.
- **(d) Leave it.** Decision 4's registry already reduces the risk.

**Constraint on (c).** `benchmarks/valendin_lstm.py` carries the same two-class pair and is
a frozen reference implementation (ADR-0004), so the shape survives there whatever happens
in `models/`. Collapsing only `models/` leaves two different shapes for one idea. Note the
split is not a PyTorch requirement — its LSTM takes state as an argument. The pair mirrors
Keras, whose `stateful` flag cannot be changed after a model is built, so Valendin et al.
train a `stateful=False` model and copy its weights into a `stateful=True` twin.

**Decision 4's registry is a partial collapse, not a full one.** `_build_inference_model_for`
builds both from one place, which makes the risk small — but any caller constructing either
class by hand bypasses it. So the `CLAUDE.md` warning cannot be deleted on the registry's
strength alone; (b) is what closes it.

**A second late failure of the same family — found 2026-08-12 by executing the tuning path.**
Not a `CLAUDE.md` invariant, but the identical shape: missing structure, failure only after
all training completes. Pinning an architecture hyperparameter to a scalar in `data_info`
(`"dropout": 0.0`) makes `_suggest_param` return it *without registering an Optuna trial
parameter*, so it never reaches `best_params`. `build_inference_from_trial` rebuilds from
`study.best_trial.params` and indexes required keys directly:

    KeyError: 'dropout'    in _build_inference_model_for

Affects `embedding_dim`, `lstm_hidden_size`, `dense_units`, `dropout` (LSTM) and `d_model`,
`nhead`, `num_encoder_layers`, `dropout` (Transformer). The workaround is a one-element set
(`{0.0}`). No archived suite is affected — they all use lists for these keys. **Decide here**
whether the registry's build entry reads params defensively, or whether the spec
mini-language should register fixed scalars as single-choice categoricals. The second is the
structural fix; the first is another warning.
