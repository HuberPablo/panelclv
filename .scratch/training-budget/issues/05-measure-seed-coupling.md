# 05 — How far does the replication RNG coupling reach?

Status: `resolved`

## The observation

`forecast_recurrent` calls `torch.manual_seed(42 + replication)` before sampling
(`models/monte_carlo_forecasting.py:374`), and in a suite each replication's forecast runs
immediately before the next replication trains. The global RNG state entering replication
*r+1* is therefore fixed by replication *r*'s forecast seed, so weight initialisation and
`DataLoader` shuffling repeat across any two runs that share a seed list.

It showed up while measuring the training budget: in two separate runs of the same
electronics experiment, replications 1–4 produced identical validation losses to five
decimals (0.08891, 0.09127, 0.08893, 0.08907).

## Why it matters

`CLAUDE.md` priority 3 says training is deliberately unseeded, so a suite reports a
distribution across replications. If replications 2…n are determined by replication 1's
forecast seed, that distribution is narrower than it looks — it varies the sampler, not
the initialisation.

## What to measure (read-only)

Across archived suites that share a seed list, compare `user_attrs_best_epoch` and the
winning validation loss by replication index, and report how often they repeat exactly
between suites. That says whether the archived spreads are genuine draws or repeats.

## What NOT to do yet

Do not change the seeding. Any fix makes new replication spreads non-comparable with every
archived family, so it belongs in its own ticket with a decision about how the archive is
read afterwards.

## Answer (2026-09-20)

**The coupling is real inside one process and does not reach the archive.**

Checked every archived suite (3,020 of them, 5,415 suite/study rows): among suites of the
same family and model, the winning validation loss repeats exactly at **no** replication
index — 0 at replication 1 and 0 averaged over replications 2..n. The only exact repeats
anywhere are Pareto/NBD, which is deterministic by design.

The reason is that an Optuna search consumes a variable number of random draws — trial
counts differ, pruning differs — so by the time a suite reaches its next replication the
RNG states have diverged. The replication spreads reported in
`docs/benchmarks-real-panels.md` and the insights documents are genuine draws, and
`CLAUDE.md` priority 3 holds.

Where it does bite: a controlled rerun whose arms do the same work in the same order. The
`end_to_end.py` A/B hit it exactly — replications 1-4 produced identical validation losses
across both arms and across a re-run. That made the A/B a paired test, which was a benefit
there, but it is a trap for any future experiment with one trial per study, which includes
family T's `paper` and `paper90` arms.

**No fix.** Nothing in the archive is wrong, so changing the seeding would only make new
spreads non-comparable with old ones. The note belongs in the write-up
(`docs/training-budget.md` §7), not in the code.

Measurement: `.scratch/training-budget/seed_coupling.py`.
