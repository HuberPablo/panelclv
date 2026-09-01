# 08 — `running-a-model.md` §3 gives the `prepare_dataset` order wrong

**Status:** ready-for-agent

Two docs describe the same eleven steps and disagree about where window slicing sits.

## Doc claim

`docs/running-a-model.md:195-198`:

> Inside, in order: time features → period index → AR columns registered → cohort
> filter → target clipping (calibration only) → AR columns computed → **window slicing** →
> `val_start_idx` → embedded-column resolution → reshape to `(N, T, F)` → standardisation.

## Code reality

Slicing comes **first** of those four. In `prepare_dataset`:

- `src/panelclv/data_preparation/panel_dataset.py:754`, `:767-774` — step 5, build
  `train_panel` and `holdout_panel` from the window dates
- `:802-810` — step 5a, the cohort filter, applied to *both* slices
- `:822-825` — step 5b, clip the target on the training window only
- `:827` onward — step 5c, fill the AR columns on the calibration window

It cannot be otherwise: the cohort filter takes `train_panel` as its argument
(`select_active_cohort(train_panel, …)`, `:808`), and the AR fill reads the clipped
calibration target. Slicing is what creates the object the other three operate on.

## The other doc has it right

`docs/feature_engineering.md:42-48`:

```
├─ 5. slice calibration / holdout windows
├─ 6. cohort filter + target clipping
├─ 7. FILL AR features from calibration only
├─ 8. resolve embedding cardinalities
```

So the two chapters contradict each other, and the correct one is the chapter `CLAUDE.md`
tells you to read before touching features.

## Fix

Correct `docs/running-a-model.md:195-198` to match the code and `feature_engineering.md`:

> … AR columns registered → **window slicing** → cohort filter → target clipping
> (calibration only) → AR columns computed → `val_start_idx` → …

Worth adding the one-clause reason, since it is what makes the order non-arbitrary and hard
to get wrong again: the cohort filter and the AR fill both read `train_panel`, so they
necessarily follow the slice.
