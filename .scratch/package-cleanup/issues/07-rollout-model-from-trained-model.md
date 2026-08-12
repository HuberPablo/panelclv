# 07 — A rollout model comes from its trained model

**What to build:** you get a rollout model by asking a trained model for one. Pairing the
wrong two classes stops being expressible anywhere, and the `CLAUDE.md` invariants section
disappears entirely.

**Blocked by:** 05, 06

**Status:** ready-for-agent

Source: `.scratch/package-simplification/issues/07-collapse-invariants.md` (decisions 2, 5),
`08-reconcile-adrs-and-vocabulary.md`, `09-module-naming.md` (decision 4),
`13-safety-net-scope.md` (decision 5)

## The handover

`trained.to_rollout()` returns the paired rollout model, **sharing the same backbone
object**. The arrow points from the training class outward, which has three consequences:

- Nothing outside the model and benchmark subpackages ever names a rollout class, so the
  registry needs **no** rollout-class field — the API removes a table row instead of adding
  one.
- The frozen benchmark declares its own pair **inside the frozen file**, next to the
  classes. This is permitted: frozen means *the numbers, not the surrounding code*, and the
  validation script is the executable gate that proves it.
- **Sharing rather than deep-copying is deliberate.** Sharing is what makes a mismatch
  unconstructible, and a copy would double peak memory at the moment a large model has just
  finished training.

## The path that made the invariant necessary is deleted, not wrapped

The inference-model builder, the rebuild-from-stored-trial function, and the two notebook
cells that call it. Issue 05 already removed their only other live caller, which is what
makes this free. The refit ends by handing back a rollout model directly.

**Accepted cost:** a stored study plus a checkpoint, in a fresh process with no live model,
can no longer be rebuilt-and-loaded — you run the refit's few epochs to forecast.
Consistent with issue 05, which makes the refit the only path to a forecast anyway.

## The one thing this issue ADDS

**The training loop must load the best-by-validation snapshot back into the model before
returning.** It snapshots the best weights, keeps training past that epoch — that is what
patience is for — then saves the snapshot to disk and never writes it back into the object.
So the returned model holds the **last** epoch's weights, and early stopping makes the two
differ **by construction**.

This has been unobservable because the checkpoint file is the only channel out of training.
**`to_rollout()` opens that closed channel, and opens it onto the wrong end.** Without this
line the issue would trade a loud shape mismatch after training for a quiet, plausible,
wrong forecast.

**It needs its own test.** The golden fixture cannot catch its absence: its two epochs
improve monotonically, so both channels agree there by luck rather than by construction, and
a two-epoch run cannot early-stop.

## Folded in

- **The pinned-scalar fix.** Pinning an architecture hyperparameter to a scalar makes the
  suggestion helper return it without registering a trial parameter, so it never reaches the
  best-params dict and the rebuild raises a `KeyError` after all training completes. Fix
  structurally: **register a fixed scalar as a single-choice categorical.** The rejected
  alternative — reading params defensively with defaults — would silently substitute a
  default where the user pinned a value, which is a wrong number instead of a loud failure.
- **`Inference*` → `Rollout*`, prefix position.** `CONTEXT.md` lists *inference* under
  `_Avoid_` and now defines **Rollout model**. Cost is far below the usual assumption:
  **four occurrences across the notebooks, all bare names in import lists, zero call sites** —
  and this issue already deletes the cells that constructed them, so the notebooks are
  touched once.

## Hard acceptance criterion — no re-baselining

**The golden numbers must come out unchanged at rel=1e-6.** The test forecasts from the
checkpoint today, and the checkpoint already holds the best weights, so the training-loop fix
does not move today's numbers. But `to_rollout()` shares the in-memory backbone rather than
reloading a file, so after this issue the golden path reads that object. Checkpoint-derived
and object-derived forecasts agree **only if** the load-back landed. **A movement in the
numbers means it did not.** Do not re-baseline.

Keeping both constructions side by side for one commit was considered and rejected — same
guarantee, at the price of a scaffold that must then be removed.

## Docs

- **ADR-0007** — full text in ticket 08 of the map. Copy it.
- **`CLAUDE.md` Edit 4** — the state-dict bullet goes, **and the "Invariants worth knowing
  before you hit them" heading goes with it.** Issue 06 removed the other two bullets, so
  this issue applies the last of the three and leaves no empty heading behind. Verbatim in
  ticket 07 of the map.

- [ ] `to_rollout()` returns the paired rollout model sharing the backbone
- [ ] Registry has no rollout-class field; the benchmark declares its own pair in the frozen file
- [ ] Inference-model builder, rebuild-from-trial function and the two notebook cells deleted
- [ ] Training loop loads the best snapshot back into the returned model, with its own test
- [ ] A pinned scalar registers as a single-choice categorical and reaches the best-params dict
- [ ] `Inference*` renamed to `Rollout*`; the four notebook import names updated
- [ ] ADR-0007 present, copied verbatim from ticket 08
- [ ] `CLAUDE.md` Edit 4 applied and the invariants heading removed — that section no longer exists
- [ ] **Golden numbers unchanged at rel=1e-6**; notebook API test green
