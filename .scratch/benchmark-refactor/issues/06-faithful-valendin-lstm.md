# Build a faithful `ValendinLSTMModel` in `benchmarks/`

Status: done
Blocked by: 05

`MultinomialLSTMModel` is not the paper's architecture, so renaming it would give a
benchmark that quietly differs from what it claims to reproduce (ADR-0004). Comparing
`Original_paper_model/banking_transactions_demo.ipynb` against `models/multinomial_lstm.py`
found two departures nobody chose:

- **Embeddings** — theirs are raw `sqrt(n)+1` vectors; ours pass through LayerNorm and
  a projection to a common width.
- **Combination** — theirs concatenates every embedding into a roughly 12-dimensional
  LSTM input; ours sums the context and concatenates the target, giving 256.

Build `benchmarks/valendin_lstm.py` using the Valendin embedder from ticket 05, and
leave `MultinomialLSTMModel` in `models/` as our own variant. The result is a simpler
model than the current one; the training loop, simulator and evaluation are already
shared.

Import it lazily from `benchmarks/__init__.py` so Pareto-only callers do not pay for
torch.

Deliberate departures that stay: temporal validation split, Optuna tuning, and the
covariate path. Everything else should match.

Done when: the architecture matches the notebook layer for layer, and a run on the
banking demo data reproduces the published numbers. Ask Pablo for those numbers if
they are not in the notebook output.

## Comments
Done in `a1c793c` (architecture) and `18a06c0` (reproduction). Both halves of the
done-when are met.

**Architecture.** `benchmarks/valendin_lstm.py` transcribes the notebook: two raw
embeddings concatenated to 12 dims, LSTM(128), Dense(128), Dense(K). The notebook's
`model.summary()` gives an exact parameter count per layer, so the tests pin those rather
than trusting the transcription — 416 / 48 / 72192 / 16512 / 1548, plus structural
assertions that no LayerNorm or Dropout exists anywhere in the module. The LSTM is the one
line that cannot match exactly: Keras carries one bias vector per gate, PyTorch two, so
ours has 4*128 more parameters. Framework convention, not architecture.

It deliberately does **not** reuse `_MultinomialLSTMBackbone`. Sharing it would mean the
frozen reference silently followed every change to the model under development, which is
the drift ADR-0004 exists to prevent. Lazily imported from `benchmarks/__init__.py` via
PEP 562, verified in a fresh interpreter: importing the subpackage does not load torch.

**Reproduction.** `scripts/validate_valendin_lstm.py`, the neural counterpart to
`validate_pareto_benchmark.py`. The notebook's metric cells (34, 45) are unrun, so there
are no computed numbers to match; the targets come from its markdown instead — cell 16
"around 0.44" and cell 35 "less than 1%". Pablo confirmed the notebook's own runs vary and
that "quite similar" is the bar, so it judges against those with a tolerance band.

The data prep hits the notebook's printed figures exactly: 2,239 accounts, 744,015
transactions, max 11 per account-week, 155-step sequences, 12 count classes. Result:
validation loss **0.4760** at epoch 70 vs a published "around 0.44", and aggregate holdout
bias **0.51%** vs a published "less than 1%" — inside the published bound outright, not
just inside the tolerance. Bias by holdout year is +0.35% / -0.22% / +1.41%, so the
156-step rollout is not drifting.

The first run scored 5.15% bias, and that was the script rather than the architecture: the
notebook's `EarlyStopping` uses `restore_best_weights=True`, while `fit_model` saves the
best epoch to its checkpoint and leaves the model on the final one. Checked the package's
own callers afterwards — the Optuna objective and the rollout selection both go through
`result.checkpoint_path`, so nothing in the package has this bug.

The reproduction runs the notebook's random 10%-of-customers split, not our temporal one
(ADR-0001): a reproduction has to run the protocol it reproduces or the validation loss is
not the same quantity. The departure applies to our studies, not to this check.

**Not done — study-suite integration.** The benchmark is reachable only from its
validation script: `studies/config.py` `VALID_MODEL_TYPES`, `studies/runner.py`
`_FORECASTERS` and `tuning/optuna_tuning.py`'s `suggest_*_params` are all untouched, so it
cannot yet be tuned or run in a suite. This ticket's done-when did not ask for it, but the
"Optuna tuning" departure implies it, and it is what would let the benchmark enter the
comparison. It needs a decision first: ADR-0004 freezes the architecture, so a
`suggest_valendin_params` branch can only search *training* hyperparameters (learning
rate, batch size, patience) — searching widths would unfreeze the benchmark.

Note for whoever does it: `optuna_tuning.py:749`
(`_build_lstm if model_type == "lstm" else _build_transformer`) and the same shape at
`:655` are unguarded `else` branches. They are unreachable today because `:402` and `:729`
raise on an unknown `model_type`, but adding a type to only some of the four sites builds
a **Transformer** silently. See ticket 08.

**Unresolved — needs your call.** This ticket lists "the covariate path" among deliberate
departures that stay, while ticket 05 specifies the Valendin embedder has "no covariate
path". I followed ticket 05 and the notebook, whose model has exactly two `Input` layers
(`week`, `transaction`) with nowhere for a covariate to enter, so `ValendinLSTMModel`
raises on a non-embedded column. My reading is that this ticket's line restated ADR-0004's
deviations list, which describes *our* model, not the benchmark. If you meant the
benchmark to carry a covariate path, it stops matching the notebook layer for layer and
the pinned parameter counts change.
