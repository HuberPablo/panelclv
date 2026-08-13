# 05 — Every forecast comes from a refit, and `experiments/` becomes `trials/`

**What to build:** one way to produce a forecast — refit the winning trial on the full
calibration window — and a subpackage whose name has a referent in the project's own
vocabulary, with the ADR-0001 enforcement point somewhere you can find it.

Two changes in one issue because they rewrite the same module, and neither moves a number.

**Blocked by:** 02

**Status:** done

Source: `.scratch/package-simplification/issues/06-target-architecture.md` (decision 3),
`08-reconcile-adrs-and-vocabulary.md`, `09-module-naming.md` (decisions 2, 8, 11)

## Refit-only

`prediction_source` goes entirely — one legal value is not a choice — along with the three
notebooks' full-calibration-refit toggles. The two knobs expressed the same decision at
different altitudes, which is how they came to disagree.

**Rationale is paper fidelity.** Valendin et al. take the lowest-validation-loss model and
perform several fine-tuning epochs using the entire calibration set; the default refit
epoch count already matches "several". The notebook comment calling the as-is branch "the
published GitHub behaviour" was right about the *code* and wrong about the *paper*, and
this package follows the paper.

**Accepted cost:** the published-GitHub baseline is no longer expressible from this
package. Restoring it would be a re-implementation, not a flag.

**ADR-0008** lands with this issue. Full text in ticket 08 of the map — copy it.

## `trials/`

The subpackage is renamed. `CONTEXT.md` states there is deliberately **no term for
"experiment"**, so the old name had no referent in the ubiquitous language; *trial* is the
one candidate the vocabulary already defines, and it makes the altitude ladder legible —
`trials/` assembles and refits one trial, `tuning/` searches over trials, `studies/` runs
many studies. Cost: four notebook import lines. This is a rename, not a re-partition.

It **splits in two**: a loaders module and a refit module. Burying the calibration split in
a 311-line catch-all is how its off-by-one stayed invisible; the refit is a different
altitude and now a named concept in the glossary.

`make_loaders` → **`split_calibration`**, returning a named `CalibrationSplit` carrying the
two loaders and the **recipe**. "Split" names the decision the function actually makes: it
is the **sole enforcement point of ADR-0001**, and nothing else in the package decides that
training truncates before the validation window while validation keeps the full sequence
and scores from a later index. The third return element is load-bearing — it is what every
model constructor is rebuilt from — so a dataclass makes it first-class rather than a
trailing dict. `make_refit_loader` → `refit_loader`.

**Both stay put.** Data preparation is disqualified (deliberately numpy-only; this is
precisely where numpy becomes tensors) and training inverts the import direction.

**The docstring lie is fixed in the same commit:** the subpackage claims to hold "no
modeling logic". It holds one.

**Cost correction, verified:** **no notebook calls or imports either function.** Two
mention `make_loaders` in code comments only — update them for accuracy, but nothing breaks
if missed. Notebook cost is measured per symbol, never assumed.

- [x] `prediction_source` gone; the three notebook toggles removed
      (with `VALID_PREDICTION_SOURCES`, its `validate()` branch, `runner._rebuild_winner`,
      and the key both `_suite_record` and `_model_record` wrote — archived suites keep
      theirs, so `test_archive_formats.py` still asserts it as a read-path fact)
      **One deletion beyond the checklist, and it is the point of the ticket:** each
      notebook's cell 8 called `build_inference_from_trial` and cell 8b then rebound it.
      Removing only the toggle would have made *cell run order* the toggle — run 8, skip
      8b, and you get the deleted `prediction_source="checkpoint"` forecast, against
      `CLAUDE.md` priority 2. The three cells and the now-unused import go; the function
      stays, since ticket 07 owns its removal.
- [x] ADR-0008 present, copied verbatim from ticket 08
- [x] Subpackage renamed, split into loaders and refit halves, four notebook imports updated
      (`build_inference_from_trial` went to the refit half: `refit_best_trial` is its only
      caller in `src/` now that the checkpoint branch is gone)
- [x] `split_calibration` returns a named `CalibrationSplit`; `refit_loader` renamed
      (`make_data_builder` flattens the split back to the `(train, val, metadata)` tuple
      `run_optuna_study`'s contract asks for — that contract is the tuner's, untouched here)
- [x] Subpackage docstring no longer claims to hold no modeling logic
- [x] `CLAUDE.md`'s "Where things live" updated for the new subpackage name
- [x] Golden test green at rel=1e-6; notebook API test green (full suite: 193 passed)
