# Which documented invariants get collapsed, and how

Type: grilling
Status: resolved
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

## Answer

Settled with Pablo, 2026-08-12, over four rounds of `/grilling` (Q1-Q13). **Every
warning in this ticket's Question is ruled on; the frontier is empty.** The Comments
below are the working material the rounds started from.

The headline: `CLAUDE.md`'s **"Invariants worth knowing before you hit them" section
disappears entirely** — it had three bullets and all three are now closed — and
"Adding a model touches three places" becomes one place.

### 1. Three-place model registration → one registry table

Ticket 06's decision 4, with the shape now fixed:

- **One table, with optional fields**, so `pareto_nbd` sits in it beside the three
  neural types rather than being a hand-written addend.
- **`VALID_MODEL_TYPES` becomes the table's keys.**
- **`NEURAL_MODEL_TYPES` stops being a list** and becomes the derived predicate *"this
  entry has a training builder"*. This is the copy that already drifted once — the
  `_NEURAL_TYPES` fourth copy in `studies/analysis.py`, which silently collapsed the
  Valendin benchmark's across-study spread to a single study. A derived predicate makes
  that class of bug unwritable; a second list does not.
- **`pareto_nbd`'s entry is declarative only.** It exists so the enumerations derive
  from one table. `studies/runner.py` keeps `_run_neural_model` and `_run_pareto_model`
  separate: they differ in more than the forecaster (no Optuna study, no refit, one
  prediction rather than *n*), so merging them is a separate refactor with its own risk
  and this ticket's test is already satisfied without it.

**Verified — the seven enumerations (D11), all in `src/`:** `studies/config.py:30`
`NEURAL_MODEL_TYPES`, `:31` `VALID_MODEL_TYPES`, `tuning/optuna_tuning.py:339`
`_SEARCH_DEFAULTS`, `:544` `_SUGGESTERS`, `:550` `_BUILDERS`, `:595`
`_build_inference_model_for`'s if-chain, `studies/runner.py:51` `_FORECASTERS`. Six list
the same three neural types; only `VALID_MODEL_TYPES` carries `pareto_nbd`, which is
absent from `_FORECASTERS`. That asymmetry is what forces the optional-field shape.

### 2. State-dict must match constructor arguments → the pairing is handed over

**`trained.to_rollout()`** returns the paired rollout model, **sharing the same
`_MultinomialLSTMBackbone` object**. Option (b) of the Comments, with the arrow pointing
from the training class outward rather than `InferenceX.from_trained(trained)` pointing
in. Consequences of that direction:

- Nothing outside `models/` ever names a rollout class, so the registry needs **no**
  rollout-class field — the API removes a table row instead of adding one.
- `benchmarks/valendin_lstm.py` declares its own pair **inside the frozen file**, next
  to the classes. Editing it is permitted: the map's floor item 2 reads *"frozen means
  the numbers, not the surrounding code"*, and `scripts/validate_valendin_lstm.py` is
  the executable gate that proves it.
- Pairing the wrong two classes stops being expressible anywhere.
- Sharing rather than deep-copying is deliberate: sharing is what makes a mismatch
  unconstructible, and a deep copy would double peak memory at the moment a large model
  has just finished training.

**The path that made the invariant necessary is deleted**, not wrapped:
`_build_inference_model_for`, `build_inference_from_trial`, and the two notebook cells
that call it. `refit_best_trial` ends with `return trained.to_rollout(), data_best`.

**Accepted cost, stated once:** a stored Optuna study plus a `.pth`, in a fresh process
with no live model, can no longer be rebuilt-and-loaded — you run the refit's few
epochs to forecast. This is decision 3's already-accepted cost applied consistently, and
the map already ruled archived checkpoints expendable.

### 3 & 4. Target column and `clip_target_upper` → already enforced, warnings deleted

**No code changes.** Both bullets describe structure that already exists, in one place,
for every model family — verified by reading it:

- `models/embedders.py`, the base `Embedder.__init__` inherited by both
  `ProjectedEmbedder` and `ValendinEmbedder`, raises on `target_col not in seq_cols` and
  on `target_col` missing from `embedded_cols`, the second with the message *"its
  cardinality drives the output head size"*. There is no path to a model that skips it.
- `dynamic_panel_dataset.prepare_dataset` step 0 raises when a **pinned** cardinality is
  `<= clip_target_upper`, with the required arithmetic in the message.
- `resolve_embedded_cols` sizes `"auto"` as `clip_upper + 1` and range-checks pinned
  values against the observed window.

So these are not footguns; nothing bites you. The *warning* framing is wrong, but the
head-size fact is real orientation for a thesis reader — it **relocates** into "What the
models are", which already explains the softmax head. Both bullets leave the invariants
list, which empties it.

### 5. The scalar-pinning `KeyError` → fix the spec mini-language

`_suggest_param` **registers a fixed scalar as a single-choice categorical**, so it
reaches `best_params` like every other key. Structural, not defensive.

The rejected alternative — the build entry reading params defensively with defaults —
would silently substitute a default where the user pinned a specific value: a wrong
number instead of a loud failure, which is worse than today's behaviour. It is also
larger, repeating a default across eight keys in two model families.

### The one thing this ticket ADDS rather than removes

**`fit_model` must load `best_state` back into the model before returning** —
`training/training_utils.py`, one line after `:347`:

    torch.save(best_state, checkpoint_path)
    model.load_state_dict(best_state)     # in-memory == on-disk, always

**Why this is not optional.** `fit_model` snapshots the best weights with
`copy.deepcopy` at `:325` and keeps training past that epoch — that is what patience is
for. At `:347` the **best** weights go to disk; nothing writes them back into the object,
so `model` holds the **last** epoch's weights. Early stopping makes the two differ *by
construction*: the loop only breaks after `patience` consecutive non-improving epochs, so
the best epoch is never the last one. Default `patience=5` (`training_utils.py:205`, and
`optuna_tuning.py:897` suggests it with the same default).

This has been unobservable because the `.pth` file is the only channel out of training —
the Optuna objective drops the model, `refit_best_trial` warm-starts a fresh one from the
checkpoint, and the golden test loads the checkpoint. **`to_rollout()` opens that closed
channel, and opens it onto the wrong end.** Without this line the ticket would trade a
loud shape mismatch after training for a quiet, plausible, wrong forecast — making the
codebase worse by this ticket's own standard.

It moves no numbers: `best_state` is byte-identical to what `test_golden_end_to_end.py`
loads from disk today, and `fit_model` returns a `FitResult`, not the model, so no caller
reads the final-epoch weights. It also makes `fit_model`'s own contract honest — it
advertises best-by-validation selection and currently half-delivers it.

**It needs its own test**, because nothing currently asserts that best-by-validation
selection is delivered to the returned object. Note the golden fixture cannot catch this:
its two epochs improve monotonically (`val_loss` 1.714376 → 1.692941, `best_epoch = 1` of
1, measured), so its two channels agree by luck — a 2-epoch run with `patience=2` cannot
early-stop.

### Amendment to ticket 06's decision 4

Decision 4 listed five fields: search space, builder, **inference builder**, forecaster,
and the rollout function. Two of those change here:

- **The inference-builder field is gone** — nothing feeds it once
  `_build_inference_model_for` is deleted.
- **No rollout-class field replaces it** — `to_rollout()` puts that pairing on the
  training class.

Remaining fields: **search space, builder, forecaster / rollout function.** Decision 5
(the model-to-rollout-*function* pairing declared through the registry) is untouched —
that is a different pairing from the model-to-rollout-*class* one settled here.

### The invariant set is closed

No further sweep. Audits 02/03/04 already covered all 10,139 lines three ways, and ticket
06's budget sits at ~10-11 issues against a ~15 tripwire. What ticket 11 gets instead is
the **failure signature** these five share, as a test to apply to anything it is tempted
to document rather than fix:

> **Missing structure, failure only after training completes.**

### Handoff to ticket 11 — the `CLAUDE.md` edits

Per Q6, **each execution issue carries its own `CLAUDE.md` edit in the same commit**, so
the doc never describes structure that has not landed. The four edits below belong to at
least three different issues; if two ship and one stalls, `CLAUDE.md` is left half-true —
which is exactly what ticket 08 exists to prevent. So each block names its owner, and
ticket 11 must confirm every row has one **before** any code work starts.

Blocks are anchored by **section heading, not line number** — the lines will drift. A
heading rename means re-reading this table, which is cheap.

**Edit 1 — section "Working in this repo", the paragraph beginning "**Adding a model**".**
Owner: the registry issue.

Remove:

    **Adding a model** touches three places, and missing the second fails only after
    training completes:

    1. `studies/config.py` — `VALID_MODEL_TYPES`
    2. `studies/runner.py` — `_FORECASTERS`
    3. `tuning/optuna_tuning.py` — a `suggest_*_params` branch

Replace with:

    **Adding a model** touches one place: an entry in the model registry, holding its
    search space, its builder and the rollout function it forecasts through. Every
    model-type list in the package derives from that table's keys, and whether a type
    is neural is read off the entry rather than restated.

**Edit 2 — section "Working in this repo", bullet 1 of "Invariants worth knowing before
you hit them".** Owner: whichever issue lands first among the three, since it is a pure
doc move. Remove:

    - The target column appears in both `seq_cols` and `embedded_cols`; its cardinality
      sets the softmax head size.

and add to the "What the models are" bullet list, after the softmax bullet:

    - The target's own column is one of the model's inputs, and the number of classes it
      can take is the size of that softmax head. `Embedder` refuses to build a model
      where those disagree.

**Edit 3 — same section, bullet 2.** Owner: same issue as edit 2. Remove outright:

    - `clip_target_upper` caps counts and therefore sets that head size.

**Edit 4 — same section, bullet 3, and the heading itself.** Owner: the
`to_rollout()` / params-path-deletion issue. Remove:

    **Invariants worth knowing before you hit them:**

    - An inference model loads its `state_dict` from the trained model, so their
      constructor arguments must match.

Leave no empty heading behind: with edits 2-4 applied the section has no bullets left, so
the heading goes with them. Whichever issue applies the last of the three removes it.

### Execution budget

**Roughly +1 net new issue** against ticket 06's ~10-11, inside the ~15 tripwire. The
registry issue was already counted. The new one is `to_rollout()` plus the deletion of
`_build_inference_model_for` / `build_inference_from_trial` and the two notebook cells;
the `fit_model` one-liner and its test, and the `_suggest_param` fix, fold into it.

### Findings this ticket produced

- **Invariants 3 and 4 are already structurally enforced** (see §3 & 4 above), so two of
  the four warnings are deletable with **zero code changes**. Neither the audits nor the
  ledger recorded this.
- **`fit_model` never restores the best weights into the model** (`:325` snapshots,
  `:347` saves, nothing loads back). Latent and unobservable today; a silent
  wrong-numbers bug the moment `to_rollout()` exists. Not in any audit.
- **The constructor arguments are written three times, not twice.** Beyond the two
  `__init__` bodies in `models/multinomial_lstm.py`, `_build_lstm`
  (`optuna_tuning.py:490-503`) and the `lstm` branch of `_build_inference_model_for`
  (`:595-606`) are the same eleven lines with one word changed.
- **Decision 3 already deletes the only live callers of the params-rebuild path**, which
  is what makes §2's deletion free. `build_inference_from_trial` has three live callers:
  `runner._rebuild_winner:173` (the `prediction_source="checkpoint"` branch decision 3
  removes), `Data_integration_LSTM_v2` cell 20 and `Data_integration_TRANSFORMER_v2`
  cell 16. Both notebook cells are *unconditional* and the very next cell **rebinds over
  them** when `REFIT_ON_FULL_CALIBRATION = True` — they exist to provide the
  "forecast with the tuning checkpoint as-is" baseline, precisely what decision 3 rules
  out. Afterwards the only caller left is `refit_best_trial`'s own last line, handing
  itself a checkpoint for a model it is still holding.
- **`refit_full_calibration` has no early stopping** and persists the final-epoch weights
  (`training_utils.py:459`), so on the production path the in-memory model and the
  checkpoint are the same weights — `to_rollout()` there is bit-identical to today's disk
  round-trip, not a behaviour change. This is *why* the trap in Fact D is confined to
  `fit_model`.

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
