# 12 — Derive the target column and the time flags once

**What to build:** the target column is produced in one place and read everywhere else, and
the time-flag set is written once. The bottom of the import stack stops naming something
above it.

**Blocked by:** 02

**Status:** done

Source: `.scratch/package-simplification/issues/05-reachability-ledger.md` (D13, D16, D17,
D1), `06-target-architecture.md` (decision 10, Q15)

## The target column

Produced once by the dataset preparation step, then **re-derived six more times**. A drift
between any two of those scores the wrong column — silently, because the shapes all still
match and the numbers all still look like counts.

## The time flags

Written **four times, and already drifted** — which is what produced an orphan day-of-year
column nobody reads. This is the half of the time-flag finding that ticket 06 left open when
it closed the customer-group half (issue 09 owns that one).

## The subpackage cycle

`configs` imports a validator from `data_preparation`, and `data_preparation` imports the
panel config back. Both at module level, neither deferred. It does not break today because
the cycle is at **subpackage granularity only** — the module graph underneath is acyclic,
which is why no audit caught it. But the panel config is meant to be the bottom of the stack
and it imports upward.

Fixed here rather than given its own issue: it is one upward import, and moving the validator
(or the name list it checks) settles it. Issue 08's acyclicity test will hold this closed.

## Folded in opportunistically

- The id-column two-fallback-string problem, ~9 sites, with the suite runner using both
  spellings in one function.
- The twice-defined data-builder alias — defined in two different subpackages, a duplication
  neither lane audit found.
- The root package docstring, which lists 8 of the 9 subpackages.

- [x] Target column produced once; the six re-derivations read it
- [x] Time-flag set written once; the orphan column gone
- [x] `configs` no longer imports `data_preparation`; acyclicity test passes
- [x] Id-column fallback resolved to one spelling
- [x] Data-builder alias defined once
- [x] Root docstring lists every subpackage
- [x] Golden test green at rel=1e-6 — its feature-axis assertions are the net here

## Comments

Landed 2026-08-13. Full suite green (263 passed, up from 234: twenty-nine new tests),
golden end-to-end included and unmoved at rel=1e-6. Both gates re-run and in band:
`validate_pareto_benchmark.py` PASS (aggregate diff -0.47 %, per-customer corr 0.9947),
`validate_valendin_lstm.py` PASS (val loss 0.4760 vs 0.44 ± 0.06, bias 0.51 %) — the same
numbers ticket 11 recorded, to the digit for the LSTM and inside MC noise for the Pareto
fit.

### The target column

`src/panelclv/data_preparation/target_channel.py` — a leaf importing only numpy, holding
`target_index(seq_cols, target_col)` (the package's one derivation) and the two reads of
that channel, `calibration_counts(data)` and `holdout_actuals(data)`. `prepare_dataset`
now makes the derivation through it and records the answer as `target_idx`; everything
else reads the record.

Seven re-derivations went, one more than the ledger counted: `_get_target_idx` in
`monte_carlo_forecasting` (deleted), `_run_monte_carlo`,
`evaluation/segment_analysis._counts`, `benchmarks/pareto_forecast` — which D16 missed —
and the identical line in all three live notebooks, where D16 counted two.
`target_index` is left with exactly two callers, `prepare_dataset` and
`tuning.select_features`, and both are *producers*: each builds a feature axis and
records the index it just decided. Nothing consumes by deriving.

**The rollout steppers take the index, not the column name.** `simulate_recurrent_path` /
`simulate_attention_path` used to take `target_col: str = "Transactions"` and derive the
index themselves, so one forecast could overwrite the rollout input at a derived channel
while scoring the actuals at the recorded one — the exact drift the ticket describes,
inside a single call. They now take `target_idx: int`, handed down from
`_run_monte_carlo`. A defaulted `"Transactions"` goes with it. Both are internals of the
two forecast entry points (`models/__init__.py` says so and does not re-export them), and
nothing outside this module called either.

The four hand-written `holdout[:, :, target_idx]` slices became `holdout_actuals(data)`,
and two `calibration[:, :, ti]` slices `calibration_counts(data)`. Both return float64;
the counts are small integers exactly representable in float32, so no metric moved — the
golden numbers are the check on that.

**One public parameter went with it.** `forecast_recurrent` / `forecast_attention` took
`target_col=None`, an override that let a caller name a different column from the one the
dataset recorded — and then `actual` came out of a channel the dict itself disagreed
with. Nothing in the package, the tests, the gates or the notebooks ever passed it, and
it is the exact mechanism this ticket exists to remove, so it is gone: the entry points
read `data["target_col"]` and `data["target_idx"]`.

**The contract this creates, stated once:** a dict handed to a forecast or a metric must
carry `target_idx`. `prepare_dataset`, `select_features` and `studies.REQUIRED_DATA_KEYS`
all already did. Two hand-built dicts did not and now do — `tests/test_customer_groups`'s
fixture, and `validate_valendin_lstm.py`'s dataset, which builds its own panel by design
(ticket 11's freeze) and so states its own index rather than importing one. A dict without
it fails in `target_channel._channel` with a message naming the key and what sets it,
rather than a bare `KeyError` from a slice.

### The time flags

`configs.panel_config.TIME_FEATURE_FLAGS`: one `{flag: TimeFeatureSpec(frequencies,
columns, auto_time)}` table replacing `_KNOWN_TIME_FLAGS`, `_COMPATIBLE_TIME_FLAGS`,
`_FLAG_TIME_COLUMNS` **and** `add_time_features`' own frequency guards, which are now a
single loop over the table. The legal keys are its keys; the compatibility filter reads
`frequencies`; `PanelConfig.schema` auto-roles `columns` where `auto_time` is set.

`auto_time` is what `add_year_idx`'s absence from the old column map was expressing —
`year_idx` is created but placed explicitly, because its role (usually known_future) is
the caller's choice. Saying that as a field rather than as a missing row is what lets the
table also be the complete list of created columns.

**The orphan is gone.** `add_dayofyear_sin_cos` wrote `dayofyear` alongside `day_sin` /
`day_cos`; no role table mentioned it, so it reached the panel and nothing ever read it.
The write is deleted, and `tests/test_time_features.py` asserts the *difference* between
the panel's columns before and after equals the table's `columns` — an extra write now
fails rather than passing unnoticed. Inert today (no daily panel runs), which is why it
survived. `docs/feature_engineering.md`'s column row is corrected and now says the table
is the source rather than restating it.

One behaviour change: an unknown key in a hand-passed `time_features` dict used to be
ignored by `add_time_features` and is now an error naming the valid keys. `PanelConfig`
already rejected it, so this only reaches a caller bypassing the config.

### The subpackage cycle

`configs/ar_feature_names.py` — the AR-feature *name grammar* (the six fixed names, the
`active_in_last_<K>_periods` pattern, `parse_ar_feature`, `validate_ar_features`), moved
down out of `data_preparation/ar_features.py`, which now imports it and keeps only the
computation. `configs` imports nothing from `panelclv` at all, and
`tests/test_import_graph.py` drops `KNOWN_CYCLES` for `== set()`.

The grammar goes down rather than the validation call moving up because `PanelConfig`
validating every field at construction is a property worth keeping — a typo in
`ar_features` still fails when the config is built, not one line later at
`prepare_dataset`. The new module is standard-library-only, so `ar_features` stays
numpy-plus-stdlib and nothing pulls pandas that did not before.

### Folded in

- **The id-column fallback.** The nine sites turn out to be two different questions
  wearing one key, which is why the ledger saw the runner "use both spellings in one
  function":

  - *What is the id column of the CSV being written?* One spelling,
    `predictions.DEFAULT_ID_COL = "customer_id"` — read by the four
    `data.get("id_col", DEFAULT_ID_COL)` writer sites, by `suite_reader._id_col`'s
    config-less fallback, and by `save_predictions_to_csv`'s own default.
  - *What is the id column of the panel being read?* Not a fallback question at all.
    `runner._run_pareto_model` now reads `data["id_col"]` and `data["target_col"]` with
    no default, the way `benchmarks.pareto_from_data` already did: they are a matched
    pair naming the panel's own columns, and guessing either fits the benchmark on the
    wrong column — silently, which is this ticket's whole subject. Falling back to the
    CSV spelling would have produced the pair `("customer_id", "Transactions")`, which
    describes no panel. `studies.REQUIRED_DATA_KEYS` gains both so a suite fails on the
    up-front check rather than after writing its `config.json`.

  `compute_pareto_predictions`'s `id_col="Id"` / `target_col="Transactions"` signature
  defaults are deliberately left, for the same reason: they are that same matched pair,
  every caller in the package, the tests and the gates passes both explicitly, and
  changing one of the two would make the pair incoherent. Same reasoning ticket 11
  applied to `period_in_days=7.0`. A comment at the signature says so.
- **`DataBuilder`.** Declared once in `tuning.optuna_tuning`, where the contract it
  describes lives; `trials.loaders` imports it instead of restating it `DataLoader`-typed.
  The arrow already pointed that way (`trials` → `tuning`).
- **The root docstring** already named all eleven subpackages — ticket 08 had fixed the
  8-of-9 the map found. But `tests/test_imports.SUBPACKAGES` turned out to be a *fourth*
  copy of the same set and had already dropped `studies`, so nothing was checking that
  subpackage imports cleanly. It is now derived from the tree, and a new test asserts the
  root docstring names every subpackage the tree holds.

### Deliberately not done

- **`data_preparation/pareto_simulation._PANEL_SCHEMA["id_col"] = "Id"`.** Not a fallback:
  the simulator really does write a column called `Id`.
- **`add_time_features` still writes `panel["week_sin"]` and friends by name.** The table
  says which columns each flag creates; the arithmetic still spells them at the point it
  computes them, because driving the assignment off `spec.columns[0]` would trade a name a
  reader can follow for an index they cannot. `test_a_flag_creates_exactly_the_columns_the_table_declares`
  pins the equality instead, which is the property that matters.
- **`pareto_from_data`'s missing-key guard does not list `target_idx`.** That function
  does not read it — `pareto_forecast` does, one level up, and `_channel`'s own error
  covers every consumer at once. Adding it there would constrain the evaluation callers
  that only want the fit.
- **The `"N  "` key in `prepare_dataset`'s return dict** — trailing spaces in a key name,
  read by nothing. An orphan rename, which is ticket 15's material.
