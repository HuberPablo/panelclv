# 01 — Parametrise the golden end-to-end test over four model families

**What to build:** the golden end-to-end test covers every model family the package
ships, not just the recurrent one. Someone running the suite on a fresh clone gets a
pinned, deterministic regression signal for the attention rollout and the Valendin
benchmark — the two paths that are live production today and appear in zero test files.

This lands **before every other issue in the set**. A net added after a change pins
whatever the change produced, including a defect it introduced.

**Blocked by:** None — can start immediately.

**Status:** done

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

- [x] Four arms present; the three new ones pass on a fresh clone, CPU-only
- [x] Pareto/NBD arm asserts shape, finiteness and determinism, and pins no values
- [x] The tracer imports every scenario from the test and defines no pipeline of its own
- [x] Whole suite still runs CPU-only in a time comparable to today's
- [x] Regeneration path still works for the newly pinned numbers

## Comments

**Implemented 2026-08-12.**

`tests/test_golden_end_to_end.py` now defines four scenarios and a `SCENARIOS` dict.
The three rollout arms share one `_fit_and_roll` tail and differ only in the config, the
model pair and the forecaster, so each arm is ~15 lines. Newly pinned:

| arm | rmse | bias_percent | mape_aggregate_style |
|---|---|---|---|
| transformer | 1.8498824874546234 | 211.56069364161849 | 211.56069364161849 |
| valendin_lstm | 1.869680860932991 | 216.257225433526 | 216.257225433526 |

Generated through the `PANELCLV_PRINT_GOLDEN=1` path, then re-asserted from a fresh
process and from three separate `-k` runs, so the numbers are neither process- nor
order-dependent. The `lstm` arm's numbers are untouched.

The Pareto arm pins its cohort (23), its `(23, 25)` shape and determinism under seed.
No value.

`scripts/trace_golden_reachability.py` lost `_train_and_roll`, its three `build()`
closures and its four `scenario_*` functions (~110 lines) and now imports `SCENARIOS`.
One trace row moved: `benchmarks.__getattr__` (the lazy-torch hook) now runs at import
time, before the tracer is installed, so it reads unreached — private, and noted in the
script's docstring. `reachability.md`/`.csv` regenerated; no other symbol lost coverage.

Suite: 183 tests in 16.5 s before, 191 in 20.6 s after, CPU-only.

### One defect found while moving the tracer's copy

The tracer's Pareto scenario rebuilt `period_start` itself from the ISO year/week pair,
under a comment claiming it did so "the way prepare_dataset does". It does not:
`add_period_start` anchors a weekly period at `Jan-1 + week*7 days`, so the hand-rolled
version put **53** periods in the calibration window where `prepare_dataset` puts 52, and
fed the benchmark all 24 raw customers rather than the cohort. The arm now takes
`prepare_dataset`'s own `train_panel`, `T_HOLD` and `ids` — which is also how
`studies/runner.py` feeds it in production — so the benchmark sees the same window and
the same customers as the three neural arms, and the cohort assertion is true by
construction rather than by luck.

### Deviations from the letter of the spec

- `test_rollout_never_reads_the_holdout` is parametrised over all three rollout arms, so
  `valendin_lstm` gets it too. The table asks for it on two arms; excluding the third
  costs more code than including it.
- The recurrent scenario is `run_lstm_pipeline`, renamed from `run_golden_pipeline` now
  that it is one arm of four rather than the whole file. `docs/running-a-model.md`
  follows the rename.
