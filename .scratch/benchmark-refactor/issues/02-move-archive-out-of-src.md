# Move `archive/` out of `src/`

Status: ready-for-agent
Blocked by: 01

Everything under `src/` reads as shippable package code. `benchmarks/archive/` escapes
the wheel only because it lacks an `__init__.py`, which is accidental rather than
intentional, and an agent exploring `benchmarks/` will read the archived MLE
Pareto/NBD as live code.

Move it to a repo-root `archive/`, following the convention `Original_paper_model/`
already sets: reference material lives at the root, not in the package.

Done when: nothing under `src/` refers to the archived module and the wheel contents
are unchanged.
