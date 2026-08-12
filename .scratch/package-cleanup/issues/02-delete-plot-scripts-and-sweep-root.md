# 02 — Delete the broken plot scripts and sweep the repository root

**What to build:** `scripts/` holds only live entry points, benchmark gates and documented
tools. Nothing in it raises on invocation, and nothing in it has already done its job. The
repository root holds only what the thesis needs.

This is **first among structural issues**. The deleted scripts import nine names spread
across three subpackages that later issues reshape and rename; landing the deletions last
would force every issue in between to keep two broken scripts compiling against moving
targets.

**Blocked by:** 01

**Status:** done

Source: `.scratch/package-simplification/issues/10-scripts-and-root-clutter.md`

## Deletions

**Both plot scripts.** Neither executes — one passes six keyword arguments its callee does
not accept, the other feeds a 1-D aggregate to a function that raises unless it is 2-D.
Neither has ever produced a tracked figure; both write to a gitignored directory. Their
job is done by the suite forecast plot, called 16 times across the notebooks, and the
covariate comparison is done in the notebooks directly. Two evaluation symbols die with
them — the checkpoint-rebuild-and-forecast helper and the holdout actuals reshaper — which
is what removes a pending branch from issue 08.

**The three one-shots.** The transformer verification check calls itself a one-off and
answered its question; one migration is applied and visible in the archive's own labels;
the other lost its purpose when the archived checkpoints were deleted. Git history is the
provenance — an applied migration kept as a file reads as one you might still need to run.

**Root.** The unreferenced input-config directory (zero references anywhere in code, docs
or notebooks) and the PyPI publishing guide (documents a path the effort lists as
explicitly out of scope; keeping it invites someone to follow it).

## Kept, deliberately

The reachability tracer stays — it is documented as running the golden test's exact
function under a tracer, and after this set moves, renames and splits across ten
subpackages it is the check that nothing was orphaned. **Being documented is the line
between a tool and a leftover.**

Also kept: the figures directory, the archive directory, the GPU-rental scripts, and the
vendored skills directory.

## Additions

- The `ParetoNBD_MLE` provenance fact relocates into the archive's README, which exists for
  this purpose and which a live figure script depends on being true.
- `.Rhistory` gains a gitignore rule it never had, and the tracked copy's removal is
  finished.
- One line in the root README saying what the GPU-rental scripts are — they look like
  clutter only because nothing says otherwise.
- One line in `CLAUDE.md` on what earns a slot in `scripts/`: a live entry point, a
  benchmark gate, or a documented tool; a one-off check goes in the commit that needed it
  and is deleted with it.

- [x] Both plot scripts and all three one-shots deleted
- [x] The two orphaned evaluation symbols deleted with them
- [x] Input-config directory and publishing guide deleted
- [x] Provenance fact present in the archive README
- [x] `.Rhistory` ignored; no tracked copy remains
- [x] README line for the GPU-rental scripts; `CLAUDE.md` line for `scripts/`
- [x] Golden test green; notebook API test green

## Comments

Landed 2026-08-12. Full suite green (191 passed), including the golden end-to-end test
and the notebook API test.

### `weekly_actuals` was a thin wrapper over a symbol this issue deletes

`holdout_actuals_NT` had one caller left inside `src/`: `weekly_actuals`, which summed its
`(N, T_HOLD)` result down the customer axis. Deleting the reshaper alone would have left
`weekly_actuals` raising `NameError` on call. It is not this issue's to delete — issue 08
kills it, along with `weekly_aggregate_predictions` and `alignment_check` — so its two-line
body absorbed the stacking loop and it keeps working until then. The count is still two
symbols: `plot_utils.py` goes 609 -> 553 lines, 71 deleted against 15 re-added as
`weekly_actuals`' new body.

### Two dead imports went with `forecast_from_checkpoint`

It was the only user of `torch` and of `run_monte_carlo_forecast as _mc_forecast` in
`plot_utils.py`, so both top-level imports are gone. No `evaluation` module imports `torch`
directly any more — it still arrives transitively through
`models.monte_carlo_forecasting`, which supplies `compute_forecast_metrics`, so this is one
fewer edge for issue 08's acyclicity work rather than a load-cost win.
`metrics_table`'s docstring and its `ndim != 2` error
message both named `holdout_actuals_NT` as the way to get a per-customer array; they now
point at `forecast["actual"]` instead.

### Three dangling doc references, not listed in the issue

Deleting the files broke references that the issue's inventory did not carry:

- `docs/running-a-model.md:263` sent pre-ADR-0005 checkpoints to
  `scripts/migrations/rename_embedder_checkpoint_keys.py`. It now states the key mismatch
  and says the migration is recoverable from git history — which is decision 2's own
  "git history is the provenance", written where someone would hit the problem.
- `notebooks/archive/README.md` cited `scripts/main_plot.py` as the current
  `ProjectedEmbedder` call shape and the same migration for reloading old checkpoints.
  Both now point at `docs/running-a-model.md`, which shows that call shape anyway.

### The provenance fact, checked against the migration before relocating

`archive/README.md` gains a *Why the archived results say `ParetoNBD_MLE`* section. It
lists what the relabel actually touched — model folder, `aggregated_<model>.csv` filename,
`name` in both the model-level and suite-level `config.json`, and the `model` column of
`metrics.csv` and `results.csv` — read off the deleted script rather than paraphrased, and
records that a bare `ParetoNBD` folder is therefore safe to read as hierarchical-Bayes.

### The tracer confirms the orphan count, and its report was left alone

`scripts/trace_golden_reachability.py` runs clean over all four arms after the deletions.
Its regenerated report differs from the committed one by exactly the two deleted symbols
(206 → 204 defined, 110 → 108 public), so nothing else was orphaned. The regenerated
`.scratch/package-simplification/reachability.{md,csv}` were reverted: they are the closed
effort's dated record, not this issue's to rewrite, and issues 03-15 will churn them far
harder.

### `.Rhistory`

The rule has no leading slash, so it covers both the queued root copy and the tracked
`.claude/.Rhistory`, whose removal is now staged. The unrelated `.claude/settings.json`
deletion already in the working tree was left as-is.
