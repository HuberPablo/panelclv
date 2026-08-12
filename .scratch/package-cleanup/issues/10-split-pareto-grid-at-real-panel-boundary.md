# 10 — Split the Pareto grid module at the real-panel boundary

**What to build:** whether a function can run on a real panel is answerable by reading an
import line. The measurements behind the thesis's two grid figures exist once, not twice.

**Blocked by:** 02

**Status:** ready-for-agent

Source: `.scratch/package-simplification/issues/12-thesis-figure-code-home.md` (decisions 1,
2, 3, 6), `09-module-naming.md` (decision 3)

## Correction this issue is built on

**The figure script does not use the three grid functions.** It imports exactly one name
from the grid module and **re-implements** its own death-mass and shape-correlation
measurements. So those three functions are called from one notebook only — genuinely
*called*, so alive under the import-is-not-a-caller rule.

This inverts the obvious reading twice over: relocating them next to the script would put
them beside code that ignores them, and deleting them as script-superseded would remove the
notebook's only path to those tables.

## The boundary test

**"Can it run on a real panel?"** — a property of the code, readable from it. Not "did it
produce a thesis figure", which fails on the evidence: the suite forecast plot produced
thesis figures and is called 16 times from notebooks, and `scripts/` is not importable, so
relocating there makes code unreachable from the notebooks that *are* the thesis's analysis
surface.

The three grid functions read the **generator's ground truth** — the true death week and the
true seasonal multiplier — so by construction they can never run on a real panel. **Nothing
else in either module fails the test, so nothing moves to `scripts/` or a `thesis/`
directory.**

## The split

- The stored-results half keeps the grid collection, group summary, model comparison and the
  two plot functions — all read stored results and run on any grid suite. **This module also
  absorbs its rename** onto the unified Pareto/NBD spelling, so it drops out of issue 15's
  sweep.
- A **synthetic-grid module** takes the three ground-truth-reading functions and their
  helpers.

A boundary between two files is enforceable by reading an import line; a docstring marker
inside one file is not.

## The duplication collapse

Two implementations measure the same two things on the synthetic grid — the death-state
failure and the seasonal-tracking strength. **The script's arithmetic wins**, because its
output is what is in the thesis; the package functions become that arithmetic exposed for the
notebook, and the script becomes a thin caller.

Folded in: the three package functions share **51 of 55 byte-identical lines** with each
other plus a third copy of the same traversal scaffold. Factored once as part of the same
work.

## Promotion

The seasonal multiplier helper becomes **public**. The synthetic-grid module exists precisely
to reconstruct the pattern a stored study was generated with, so reaching across a subpackage
boundary for a private name is the wrong shape.

- [ ] Two modules, split on the real-panel test; the stored-results half carries its new name
- [ ] One implementation of the death-mass and shape-correlation measurements, script's arithmetic
- [ ] The script is a thin caller; the traversal scaffold exists once
- [ ] Seasonal multiplier helper public
- [ ] The notebook's three grid calls still work
- [ ] Golden test green; notebook API test green
