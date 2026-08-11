# Move `archive/` out of `src/`

Status: done
Blocked by: 01

Everything under `src/` reads as shippable package code. `benchmarks/archive/` escapes
the wheel only because it lacks an `__init__.py`, which is accidental rather than
intentional, and an agent exploring `benchmarks/` will read the archived MLE
Pareto/NBD as live code.

Move it to a repo-root `archive/`, following the convention `Original_paper_model/`
already sets: reference material lives at the root, not in the package.

Done when: nothing under `src/` refers to the archived module and the wheel contents
are unchanged.

## Comments
Done in `9a30935`, landed **after** ticket 03 rather than before it: `src/` cannot stop
referencing the archived module while the MLE code paths still exist, so the removal had
to go first. Both tickets' done-conditions hold at the end of the pair.

`src/panelclv/benchmarks/archive/` -> repo-root `archive/`, with a `README.md` recording
what the module is and which stored results it produced. Setuptools discovery under `src/`
finds the same ten `panelclv.*` packages as before the move.
