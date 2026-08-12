# Give the on-disk-format floor an executable definition

Type: task
Status: resolved

## Question

The map declares three floor items. Two have executable definitions; the third does not.

- **Benchmark fidelity** is gated by `scripts/validate_pareto_benchmark.py` and
  `scripts/validate_valendin_lstm.py` — the map's Notes name them "the executable
  definition of this floor".
- **The forecasting contract** is pinned by `tests/test_golden_end_to_end.py` (ticket 01).
- **On-disk study formats** — "archived `Studies/` predictions and `results.csv` stay
  readable; re-running a suite costs GPU-hours, orphaning one destroys real work" — is
  gated by **nothing**.

Audit 04 established how exposed this is: no test reads an archived suite, `studies/runner.py`
has no test at all, and `studies/layout.py` (the module that *defines* the on-disk layout) is
covered only for its path-construction helpers. So the format that real GPU-hours are stored
in is pinned by no assertion anywhere. Audit 04 also found the format has already drifted in a
real archived suite on disk — `_id_col` writes `aggregated_*.csv` under `customer_id` beside
`Prediction_*.csv` headed `Id` — which is exactly the class of silent breakage this floor exists
to prevent, and it went unnoticed.

Build the gate: a test (or a script, if a test cannot reach the archives) that reads a real
archived suite from `Studies/` and asserts the reader path still recovers it — the predictions,
the `results.csv` columns, and the `config.json` → `PanelConfig` round-trip that
`analysis.py` depends on. Audit 04 verified `study_metrics` reproduces all nine stored
`results.csv` values of an archived suite to full float precision, so the gate has a known-good
baseline to pin.

Scope note: this ticket **builds a safety net**, the same shape as ticket 01, which is why it
is a `task` and not a decision. It does not decide anything about the target architecture.

Deliberately unblocked — the formats on disk are a present fact, so nothing has to be decided
first. Taking it early is worth more than taking it late: until it exists, every
format-touching decision in tickets 06 and 11 is ungated, and ticket 11's requirement that
"the tree is never broken" cannot be checked for the archive floor.

Resolve by recording where the gate lives, which archived suite it pins, and what it does
**not** cover.

## Superseded in part — read this first

**Pablo rescinded floor item 3 on 2026-08-11**, after this ticket resolved. Archived studies no
longer have to stay re-readable, so **this gate no longer defends a floor.** The report below is
still accurate; its *mandate* is gone.

What survives, and it is most of the file: **19 of the 24 tests run from a synthetic fixture and
exercise live code** — `studies/layout.py`'s path helpers, the
`config.json` → `PanelConfig.from_dict` → `to_dict` identity, `save_predictions_to_csv`, and the
`analysis.py` read path through `prepare_dataset`. That code serves *current* studies, not only
archived ones, and this file is its only coverage. Those tests are worth keeping on their own
merit, retargeted as read-path coverage rather than as a floor gate.

What is now dead weight: the 4-5 skip-if-absent tests that read the real `Studies/` archive, and
the two warts deliberately pinned as-is (`customer_id`/`Id`; `study_metrics` raising on the legacy
suites). Under the new ruling those warts should be **fixed**, so tests asserting them will block
the fix. Ticket 06 decides; recommendation is to drop the archive-dependent tests and the wart
assertions, and keep the fixture-driven read-path coverage.

## Answer

**Where the gate lives:** `tests/test_archive_formats.py` (24 tests, ~3.5 s, no GPU).

```
PYTHONPATH=src pytest -q tests/test_archive_formats.py     # 24 passed in 3.47s
```

(`pyproject.toml` already sets `pythonpath = ["src"]`, so bare `pytest -q
tests/test_archive_formats.py` works too — and note that setting *overrides* an
environment `PYTHONPATH`, which matters if you ever want to point the tests at a
patched copy of `src/`.)

### Design choice: both halves, and why

**Both** — a synthetic fixture that always runs, plus skip-if-absent real-archive tests.
Forced by three facts on disk:

- `Studies/` is gitignored (`.gitignore:25`) and the suites are 7.4 MB – 2.6 GB, so no real
  suite can be committed and an unconditional real-archive test fails on every fresh clone.
- `Datasets/` is gitignored too, so the numeric check — which needs the panel CSV to rebuild
  the actuals — cannot run in CI *at all*. Skip-if-absent is not a preference there, it is
  the only option.
- `Predictions/` is *itself* a gitignore pattern, so a committed fixture tree containing
  `<Model>/Predictions/Prediction_1.csv` would be silently untracked. The fixture is
  therefore built into `tmp_path` from constants in the test module, not committed as files.

The fixture is written with `Path.write_text` from **literal text**, never through
`save_predictions_to_csv` / `layout.write_json`. That matters: a round-trip through the
package's own writer and reader still passes when both are changed together, whereas a
literal `Id,week_0,week_1,week_2,week_3` does not. One test (`test_writer_still_emits_the_
archived_prediction_format`) then drives the real writer and compares *it* to the literal,
so both directions are pinned. The literals were verified against the real archive — the
real-archive tests assert the same constants against
`Studies/cross_entropy_cfg_2yTrain_1yPred_NoCov_TestDimanche`, so the fixture is a checked
excerpt rather than a reconstruction.

### What it pins

Which suite: `cross_entropy_cfg_2yTrain_1yPred_NoCov_TestDimanche` (the smallest neural
suite at 7.4 MB, and the **only** archived electronics suite carrying a `panel_config` —
hence the only one `study_metrics(root, panel_path)` can score at all). Plus a shape check
over every finished `cross_entropy*` suite, and the Pareto/NBD grid.

Always (fixture):

- the tree `layout.py` documents — `config.json`, `results.csv`, `<Model>/config.json`,
  `<Model>/metrics.csv`, `<Model>/Predictions/Prediction_{i}.csv`,
  `<Model>/Optuna_Studies/study_{i:02d}/` — addressed through `layout`'s own helpers
- `results.csv`'s eight fixed leading columns in order plus a `param_*` union, and that the
  three metric names are exactly `rmse` / `bias_percent` / `mape_aggregate_style`
  (`analysis._STUDY_METRIC_COLS` **and** `pnbd_grid._METRIC_SOURCE` read them off disk)
- `Prediction_{i}.csv` naming, the `Id,week_0..week_{T_HOLD-1}` header, row count, and the
  numeric `Prediction_10` sort key
- `config.json` → `PanelConfig.from_dict` → `.to_dict()` as an **exact dict identity**
- `_discover_models` order from `config.json`; `_is_deterministic_model` from `model_type`
- reading one study, averaging across studies (exact), the deterministic benchmark ignoring
  the requested index, `aggregated_<Model>.csv` at the suite root
- the **full** read path `config.json` → `PanelConfig` → `prepare_dataset` → id alignment →
  `compute_forecast_metrics`, with the fixture's six metric values pinned and required to
  equal its `results.csv` — i.e. audit 04's runner/reader agreement property, reproduced
  synthetically so it runs where the archive does not
- both generations of the suite record: current (with `panel_config`, `overwrite`,
  `keep_only_best_checkpoint`) and legacy (without), since 4 of 5 archived suites are legacy

With the archive present:

- the same format assertions against the real suite, and header+row-count checks over every
  finished `cross_entropy*` suite (headers only, so the 554 MB and 2.6 GB suites stay cheap)
- the Pareto/NBD grid as *a directory of standard suites* (`<grid>/<combo>__<dataset>/`), and
  `collect_grid_results` still joining `Studies/` to the generation tree under
  `Datasets/Synthetic/` with **no sub-suite silently dropped** (it `continue`s past a missing
  `results.csv`, so the row count is checked against what is on disk)
- **the nine numbers.** `study_metrics` recomputes all nine stored `results.csv` values from
  the stored `Prediction_*.csv` files. Both directions are asserted: the file still holds the
  pinned literal, and the reader still reproduces it.

**Correction to audit 04:** the agreement is to ~13 significant digits, not "full float
precision" — e.g. LSTM `bias_percent` is `-53.40695337290191` recomputed vs
`-53.40695337290179` stored. Float reduction order, not a format issue. Tolerance `rel=1e-12`.

### Proof the assertions bite

Never mutated `Studies/`. Two scratch harnesses under the session scratchpad: a fake repo
root with no `Studies/`/`Datasets/` (→ **19 passed, 4 skipped**, so the fixture half is not
vacuous), and a full repo copy with its own `src/` and a 13 MB copy of the pinned suite.

Archive mutations (in the copy): renaming `week_0`→`t0` in one prediction header → 2 failures;
renaming `mape_aggregate_style`→`mape` in `results.csv` → 3; dropping `clip_target_upper`
from the stored `panel_config` → 2; perturbing one stored prediction value → the nine-value
regression fails alone.

Code mutations (scratch `src/` copy): `save_predictions_to_csv` emitting `w{i}` → 3 failures;
`layout.prediction_path` → `Pred_{i}.csv` → 2; renaming `analysis._STUDY_METRIC_COLS`' MAPE
→ 3; `pnbd_grid._METRIC_SOURCE`'s MAPE → 3; a `collect_grid_results` path typo → 1;
`PanelConfig.to_dict` dropping `clip_target_upper` → 3. Every mutation reverted to green.

### Format warts pinned, not fixed

- **`_id_col`'s `customer_id` fallback.** A suite with no `panel_config` gets
  `aggregated_*.csv` headed `customer_id` next to `Prediction_*.csv` headed `Id` — already
  on disk in `cross_entropy_cfg_2y_Train_1yPred_NoCov_V1_10Studies_100_simulations/`.
  Pinned by `test_legacy_suite_aggregate_uses_the_customer_id_fallback`; "fixing" the
  fallback to `"Id"` in a scratch copy failed 2 tests, as intended. Tickets 06/11 rule.
- **`study_metrics` raises on a legacy suite** rather than degrading, so it is unusable on
  4 of the 5 archived electronics suites. Pinned as a `pytest.raises`.
- **The read path is not torch-free.** `analysis.py` is written to stay numpy/pandas until a
  plot is drawn, but `load_predictions_from_csv` lives in `evaluation/plot_utils.py`, which
  imports torch at module level — so *any* prediction read pulls torch in. The structural
  half of the gate runs without torch; the reading half is marked `needs_torch`. Moving that
  one function out of `plot_utils` would make the whole archive floor torch-free. Worth a
  line in ticket 06's target shape.
- **`week_*` / `Id` are baked into the archive**, in a package that parameterises columns
  through `PanelConfig`. On the floor, so recorded only.
- **`ParetoNBD_MLE/metrics.csv` in the archive carries a `param_penalizer_coef` column the
  current `runner._run_pareto_model` cannot produce** (its row dict has no `param_*` keys),
  and archived Pareto `results.csv` rows have an empty `seed` although the runner now
  records one. Two writer generations, same as the suite record. The fixture models the
  *current* writer; `_assert_suite_shape` only requires the leading columns, so both parse.

### What the gate does NOT cover

- **No writer test.** `studies/runner.py` still has no test. The gate asserts the *format*
  the runner produces, reconstructed from literals — it never calls `run_study_suite`, which
  needs Optuna + GPU-scale training. A change to `_suite_record` / `_model_record` that
  drops a key would be caught only by the real-archive tests, i.e. only where an archive
  exists.
- **Where the archive is absent, the numeric half is gone entirely.** The nine-value check
  and every real-suite assertion skip on a fresh clone or in CI. The fixture's own six
  pinned values still run, but they pin the *fixture's* arithmetic, not the archive's.
- **Prediction *values* of the real archive are unpinned except through the nine metrics.**
  A corruption that leaves those three aggregates intact passes.
- **Only headers and row counts** are read for the non-pinned suites, and only three of the
  grid's 160 sub-suites get a full shape check.
- **Checkpoints, Optuna databases and trial CSVs are not covered.** The fixture creates the
  `Optuna_Studies/study_NN/` folders and stub files so `layout`'s helpers are checked
  against them, but nothing loads a `.pth` or an Optuna storage — so a change that orphans
  archived checkpoints (e.g. `prediction_source="checkpoint"`, or a constructor-signature
  change breaking `state_dict` reload) is *not* gated here.
- **The generation-study format under `Datasets/Synthetic/`** (`<combo>/<dataset>/config.json`
  + panel files, read by `list_pnbd_datasets`) is only touched via `collect_grid_results`.
  It is a second on-disk format, outside this floor item's wording, and unpinned in itself.
- **`plot_suite_forecast` and the group/segment tables are not exercised** — they draw
  figures and duplicate the id-alignment logic with a *different* policy (reorder vs raise).
  Only the metrics path's policy is pinned.
- The pinned suite name, panel path and grid name are hardcoded. Renaming or deleting them
  turns the real-archive tests into skips rather than failures — silent, by construction.
