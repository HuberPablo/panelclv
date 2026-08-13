# 14 — Retarget the archive tests as read-path coverage

**What to build:** the test file that pinned the on-disk archive format becomes what it
actually earns its place as — the only coverage the suite layout and the suite read path
have — and stops asserting constraints that no longer exist.

**Blocked by:** 09

**Status:** done

Source: `.scratch/package-simplification/issues/06-target-architecture.md` (decision 7),
`14-archive-format-gate.md`

## Why it changes

The file was built to give the on-disk-format floor an executable definition. **Pablo
rescinded that floor item** — archived studies no longer have to stay re-readable, and clean
code outranks archive compatibility. So the file's *mandate* is gone.

What survives is most of the file: **19 of its 24 tests run from a synthetic fixture and
exercise live code** — the layout path helpers, the config round-trip, prediction saving, and
the read path through dataset preparation. That code serves *current* studies, not only
archived ones, and this file is its only coverage.

## What goes

- The 4-5 skip-if-absent tests that read the real archive. Both the archive and the datasets
  directory are gitignored, so those can never run in CI — and there is no CI.
- The two tests pinning known warts **as-is**: aggregate files keyed by one id spelling
  beside predictions headed with another, and the metrics function raising on the legacy
  suites. Under the rescinded floor those warts are **fixable rather than preservable**, so
  tests asserting them would block the fix.

**Do not delete the file wholesale.** That trades one unjustified constraint for a coverage
hole.

## Relabelling

The file's name and docstring should say read-path coverage, not frozen format. Blocked by
issue 09 because the suite read path it exercises is split three ways there.

- [x] The 19 fixture-driven tests kept and passing (17 after the two warts go)
- [x] Real-archive tests removed
- [x] The two wart-pinning tests removed
- [x] File and docstring name what it now does
- [x] Suite passes on a fresh clone with no archive present

## Comments

Landed 2026-08-13. `tests/test_archive_formats.py` -> `tests/test_suite_read_path.py`,
24 tests -> 17. Full suite green (265 passed).

### What went, and the arithmetic

Five real-archive tests (`test_archived_suite_matches_the_pinned_format`,
`test_every_archived_neural_suite_still_parses`, the two `pnbd_grid` ones and
`test_archived_suite_metrics_reproduce_its_stored_results_csv`), plus `_assert_suite_shape`
which only they called, plus the constants that addressed the archive: `REPO_ROOT`,
`ARCHIVE_ROOT`, `PINNED_SUITE`, `PINNED_PANEL`, `PNBD_GRID`, `PNBD_GENERATION`,
`PINNED_SUITE_RESULTS`, the `needs_archive` / `needs_panel` gates, and the three
`ARCHIVE_*` lists carrying the pre-rename `mape_aggregate_style` spelling.

Then the two wart tests: `test_legacy_suite_aggregate_uses_the_customer_id_fallback` and
`test_legacy_suite_has_no_panel_config_to_rebuild_actuals`. 24 - 5 - 2 = 17.

### A third test asserted the same wart, so it lost that half

`test_id_col_resolution_including_the_legacy_fallback` was counted among the 19 kept, but
its second assertion was `_id_col(legacy_suite) == "customer_id"` — the *same* fallback the
aggregate test pinned, and the one that heads `aggregated_*.csv` differently from the
`Prediction_*.csv` beside it. Leaving it would have left the fix blocked by a test the
ticket thought it had cleared. The test survives as
`test_id_col_comes_from_the_stored_panel_config`, asserting the live fact only: the id
column comes out of the run's own recipe rather than being hard-coded or sniffed.

### `needs_torch` gated nothing and went with them

The file's marker skipped ~half the tests when torch was absent, on the claim that the
structural half ran torch-free. It never did: the module-level
`from panelclv.studies import ...` pulls torch in through the model registry, so without
torch the file fails at *collection* and the marker is unreachable. Ticket 08 had already
corrected the marker's `reason` string without noticing it was dead. Removed, and the
docstring now says plainly that the whole file needs torch.

### The regeneration hook was named after the rescinded floor

`PANELCLV_PRINT_ARCHIVE_GATE=1` prints a paste-ready `FIXTURE_METRICS` block. There is no
gate any more, so it is `PANELCLV_PRINT_FIXTURE_METRICS=1`. Likewise `ARCHIVE_ID_COL` ->
`FIXTURE_ID_COL`, and `test_writer_still_emits_the_archived_prediction_format` ->
`test_writer_emits_the_format_the_reader_reads` — that test is the file's one writer-side
assertion and it is what keeps the literal-text fixture honest, so it stays.

### What is deliberately still there

`prediction_source` in `SUITE_CONFIG_KEYS` / `MODEL_CONFIG_KEYS` and in the fixture, though
ticket 05 removed it from the writer; and the `legacy_suite` fixture with
`test_legacy_suite_config_is_still_discoverable`. Neither is an archive-compatibility
assertion — both say the reader does not *require* keys it does not use, which is why
`_discover_models` opens a suite older than itself. Removing them would be a reader change,
not a test change.
