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

**Study-suite integration.** Done. `model_type="valendin_lstm"` is registered in
`studies/config.py` `NEURAL_MODEL_TYPES`, `studies/runner.py` `_FORECASTERS` (the stateful
LSTM simulator, same as ours) and `tuning/optuna_tuning.py`.

The search space is `VALENDIN_SEARCH_DEFAULTS = {learning_rate, weight_decay, batch_size}`
— training hyperparameters only. ADR-0004 freezes the architecture, so `memory_units` and
`dense_units` stay at the published 128/128 and are never sampled; searching a width would
quietly unfreeze the reference implementation. A test asserts the search space is exactly
those three and that the built model still has the published sizes.

Verified end to end on a synthetic panel: a one-trial suite runs Optuna -> refit ->
rollout -> `results.csv`, and its only `param_*` columns are the three training knobs.

That run is also what caught the last dispatch site. `run_optuna_study` carried its own
hardcoded `{"lstm", "transformer"}` guard, so the new type passed every unit test and was
rejected at the entry point. Now registry-driven, and pinned by a test.

**Resolved: the benchmark takes no covariates.** This ticket's "deliberate departures
that stay" line listed the covariate path, which read as conflicting with ticket 05's "no
covariate path". Pablo confirmed the benchmark takes none and `ValendinEmbedder` should
keep raising on a non-embedded column. No code change — that is what shipped. The
departures line describes our model in `models/`, not the benchmark; ADR-0004 now says so
explicitly, so the same question does not come back.
