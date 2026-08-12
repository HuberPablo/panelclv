# 01 — Parametrise the golden end-to-end test over four model families

**What to build:** the golden end-to-end test covers every model family the package
ships, not just the recurrent one. Someone running the suite on a fresh clone gets a
pinned, deterministic regression signal for the attention rollout and the Valendin
benchmark — the two paths that are live production today and appear in zero test files.

This lands **before every other issue in the set**. A net added after a change pins
whatever the change produced, including a defect it introduced.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

Source: `.scratch/package-simplification/issues/13-safety-net-scope.md`

## Arms

Extend the existing test rather than adding a file — the fixture, the pinned-number
machinery, the regeneration path behind an environment variable and the tracer's import
point all already live there.

| arm | asserts |
|---|---|
| recurrent | existing behaviour, unchanged |
| attention | pinned metrics, determinism, no holdout read |
| Valendin benchmark | pinned metrics, determinism |
| Pareto/NBD | **shape, finiteness, determinism only — no pinned values** |

The Pareto arm is deliberately weaker and must stay that way. It never trains and never
rolls out; its short single-chain fit is documented as recording which code runs rather
than whether it converged, and it emits a divide-by-zero warning on this panel. Pinning
numbers off an unconverged chain pins noise, and the next person could not tell a real
regression from sampler drift.

The attention arm is the point of the issue. Its entry point has the same signature and
return shape as the recurrent one — only the stepper differs — so the arm is a fixture
parameter, not a new pipeline.

The Valendin arm is not redundant despite sharing the recurrent rollout: it is the only
always-available end-to-end coverage of that model, because its validation script needs a
gitignored dataset and cannot run on a fresh clone.

## Second deliverable

The reachability tracer imports **all four** scenarios from the test, and its own copy of
the golden pipeline is deleted. Today it imports only the recurrent scenario and carries a
re-implementation for the other three — a duplication the ledger missed because it scanned
the package only. This is what makes the issue one session rather than four.

- [ ] Four arms present; the three new ones pass on a fresh clone, CPU-only
- [ ] Pareto/NBD arm asserts shape, finiteness and determinism, and pins no values
- [ ] The tracer imports every scenario from the test and defines no pipeline of its own
- [ ] Whole suite still runs CPU-only in a time comparable to today's
- [ ] Regeneration path still works for the newly pinned numbers
