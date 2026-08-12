# 02 — Delete the broken plot scripts and sweep the repository root

**What to build:** `scripts/` holds only live entry points, benchmark gates and documented
tools. Nothing in it raises on invocation, and nothing in it has already done its job. The
repository root holds only what the thesis needs.

This is **first among structural issues**. The deleted scripts import nine names spread
across three subpackages that later issues reshape and rename; landing the deletions last
would force every issue in between to keep two broken scripts compiling against moving
targets.

**Blocked by:** 01

**Status:** ready-for-agent

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

- [ ] Both plot scripts and all three one-shots deleted
- [ ] The two orphaned evaluation symbols deleted with them
- [ ] Input-config directory and publishing guide deleted
- [ ] Provenance fact present in the archive README
- [ ] `.Rhistory` ignored; no tracked copy remains
- [ ] README line for the GPU-rental scripts; `CLAUDE.md` line for `scripts/`
- [ ] Golden test green; notebook API test green
