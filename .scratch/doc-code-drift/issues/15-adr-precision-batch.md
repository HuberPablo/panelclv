# 15 — ADR precision batch: four sentences that name the wrong thing

**Status:** ready-for-agent

Four small, independent inaccuracies across ADRs 0002, 0003, 0004 and 0007. None of them
breaks an invariant — in every case the decision itself holds — but each sends a reader to
something that is not there. Grouped because they are one editing pass.

---

## 15a — ADR-0004: "Each carries its own `WEEKS_PER_YEAR`, cohort filter and week index"

`docs/adr/0004-frozen-reference-implementations.md:79-80`:

> **The validation scripts re-implement what they check, deliberately.** **Each** carries its
> own `WEEKS_PER_YEAR`, cohort filter and week index rather than importing the package's.

True of one script, not both:

- `scripts/validate_valendin_lstm.py` has all three — `WEEKS_PER_YEAR = 52` (`:77`), its own
  cohort filter (`:101-103`), its own week index (`:112`, `:123`, `:126-127`).
- `scripts/validate_pareto_benchmark.py` has **none** of the three. It has no
  `WEEKS_PER_YEAR` anywhere; it *generates* its cohort rather than filtering one (`:40-83`);
  it has no week index. Its one insulated constant is a different quantity,
  `PERIOD_DAYS = 7.0` (`:33`).

**Fix:** the principle — a gate that imports the code it gates stops being a gate — is right
and should stay. Attribute the three names to `validate_valendin_lstm.py`, and say what
`validate_pareto_benchmark.py` insulates instead (its own synthetic cohort and `PERIOD_DAYS`),
so "these copies are insulation and are not to be deduplicated" still covers both files.

---

## 15b — ADR-0007: "`fit_model` returns a model"

`docs/adr/0007-rollout-model-from-trained-model.md:37`:

> **`fit_model` returns a model whose weights match its checkpoint.** That property needs its
> own test …

`fit_model` returns a `FitResult` dataclass — `best_val_loss`, `best_val_f1`, `best_epoch`,
`checkpoint_path`, `history` (`src/panelclv/training/loop.py:38-44`, returned at `:380-386`).
No model.

The **property is real**, and load-bearing: `training/loop.py:366` does
`model.load_state_dict(best_state)`, mutating the caller's own object in place (`.to()` at
`:256` is in-place, so it is the same object throughout). `scripts/validate_valendin_lstm.py:233`
and `tests/test_training_loop.py:118` both depend on exactly that.

**Fix:** "`fit_model` leaves the best weights in the model it was given, not only on disk."
That is what the following two sentences of the ADR already argue for, and it is what a
reader needs to look for.

---

## 15c — ADR-0002: "`evaluation/` imports the simulator from `models/`"

`docs/adr/0002-simulator-lives-with-the-model.md:11`:

> `evaluation/` **imports the simulator** from `models/`, never the other way round.

`evaluation/` imports only `compute_forecast_metrics`
(`src/panelclv/evaluation/plots.py:30`, `src/panelclv/evaluation/segment_analysis.py:43`).
Neither `forecast_recurrent` nor `forecast_attention` appears anywhere under `evaluation/`.

The direction claim — the load-bearing half — is true and was checked: `models/` imports
*down* into `predictions` (`models/monte_carlo_forecasting.py:50-54`) and never into
`evaluation`; there is no `evaluation/plot_utils`; and no deferred upward import survives in
a function body.

**Fix:** "`evaluation/` depends on `models/`, never the other way round" — which is both true
and the thing the ADR is actually asserting.

---

## 15d — ADR-0003 cites a stored study that was never written

`docs/adr/0003-rollout-composite-selection.md:31`:

> It was used there: a stored study named `lstm_cross_entropy_rollout_composite_20260601_1651`
> **really ran**.

That name appears in exactly three places in the repo, none of them an artifact: this ADR,
the allowlist at `tests/test_docs_are_current.py:63` (which exists *because* the name does not
resolve), and `.scratch/package-simplification/issues/06-target-architecture.md`.

The name that does appear in a real artifact is
`lstm_cross_entropy_rollout_composite_2026**0531_1638**`, in
`notebooks/archive/Data_integration_LSTM.ipynb`.

This is the ADR's single piece of evidence for "the feature was really used", which is part of
why the decision was recorded rather than deleted — so the name should be the one that exists.

**Fix:** correct the timestamp to `…_20260531_1638` and update the allowlist entry at
`tests/test_docs_are_current.py:63` to match. Check the `.scratch` note too, since it is where
the wrong name most likely originated.
