# 13 — Output paths derivable from config and seed

**What to build:** the same config and the same seed produce the same output paths. Finding
a previous run's output does not require knowing what time it was started.

**Blocked by:** 02

**Status:** done

Source: `.scratch/package-simplification/issues/05-reachability-ledger.md` (D19),
`06-target-architecture.md` (decision 8)

## Scope — three sites, not six

Three live sites put wall-clock time into a **folder name**, so an output path is not
derivable from config and seed. That is a direct hit on the project's second priority:
reproducibility means the same config and seed give the same result, and a path is part of
the result.

Derive those three names from config and seed. **Keep the timestamp as a metadata field** —
provenance is worth having, it just should not be load-bearing for lookup.

**The other three sites of the same pattern are deliberately left alone:** the suite runner
writes a creation-time metadata field, which is correct provenance; the tuner already opts
out of appending a timestamp; and the third dies with the module issue 03 deletes. Do not
sweep them by pattern-match.

- [x] The three folder names derive from config and seed
- [x] Timestamp preserved as metadata at each of those sites
- [x] The three deliberately-excluded sites untouched
- [x] Running the same config and seed twice writes to the same path
- [x] Golden test green

## Comments

Landed 2026-08-13. Full suite green (272 passed), including the golden end-to-end test
and the notebook API test.

### The three names, and what each is now made of

| site | was | is |
|---|---|---|
| `models/monte_carlo_forecasting._save_predictions_run` | `{tag}_n{n_sims}_seed{seed}_{YYYYMMDD_HHMMSS}` | `{tag}_n{n_sims}_seed{seed}` |
| `benchmarks/pareto_benchmark.pareto_forecast` | `{tag}_{YYYYMMDD_HHMMSS}` | `{tag}_seed{seed}` |
| `data_preparation/pareto_simulation._auto_study_name` | `pnbd_study_{R}x{C}x{D}_{YYYYMMDD-HHMMSS}` | `pnbd_study_{R}x{C}x{D}_seed{base_seed}` |

The first only had to drop a suffix — it already named its config and seed. The other two
named neither, so each gained the seed that decides what is inside it.

### The Pareto fit's seed was a literal the folder could not have read

`pareto_forecast` takes the seed inside `**fit_kwargs` and forwards it; when the caller
omits it the chain still runs seeded, on `compute_pareto_predictions`' own `seed: int = 42`
default. Naming the folder `pareto_seed42` from a second literal would have been two
declarations of one number, and a folder claiming a seed the chain never used the moment
one of them moved. The default is now `_DEFAULT_SEED`, declared once and read by both.

### Where the timestamp went

The study generator already wrote `created_at` into `study_config.json`, so site three
lost the clock from its name and kept its provenance untouched. The two prediction dumps
had nowhere to put one — the run folder held only the CSV — so `predictions` gained
`run_directory.py`: `create_run_directory(base_dir, run_name)` makes the folder the caller
named and drops `run_metadata.json` beside the predictions with the wall-clock time.

It lives in `predictions` rather than at either call site because the two writers sit in
different subpackages (`models` and `benchmarks`) and produce **one** on-disk layout
between them; a sidecar written under two spellings would not be one layout. Both already
import the leaf, so no new arrow appears in the import graph and `test_import_graph` stays
green.

`mkdir(exist_ok=True)` is what makes re-running a config land on top of its own run rather
than beside it, and the sidecar is rewritten each time — the recorded time then describes
the files actually sitting there.

### `tests/test_output_paths.py`, 9 tests

The three sites, plus the two excluded ones. Each site is tested at its cheapest reachable
level: the Monte Carlo dump through `_save_predictions_run` directly (a trained model would
add minutes without adding evidence), the Pareto dump through a real `pareto_forecast` call
on a hand-built 12-customer dict with a 40-draw chain (3.5s for the file), the study name as
a pure function plus one end-to-end regenerate-twice-one-folder test.

The last test pins the **two sites this issue deliberately did not touch** — the suite
runner's `created` metadata field and the tuner's `append_timestamp` opt-out. Both are
correct as they stand, and "left alone" is a decision that a later pattern-match over
`datetime.now()` would otherwise silently reverse.

### Archived studies are unaffected

The timestamped folders already on disk (`pnbd_study_4x4x10_20260802-160937`, and the two
`scripts/` figure paths pointing at `..._20260716-154143`) are named by literal path or by
an explicit `study_name=`, never by the default. Nothing re-derives them, so nothing
re-derives them wrongly.
