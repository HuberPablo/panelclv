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
