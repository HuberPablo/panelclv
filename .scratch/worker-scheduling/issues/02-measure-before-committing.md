# 02 — Rank machines by measured $/suite, not by published specs

Status: ready-for-agent

## Problem

`vast_search.py` ranks by CPU generation and clock. Measured on identical work, the two
highest-clock CPUs available were the two slowest machines in the fleet — 49% and 95%
slower than a 3.0 GHz EPYC. Cost per unit work varied 2.5x across machines whose hourly
price varied 20%. Ranking on specs picks the wrong machines; ranking on $/hr optimises the
wrong quantity.

## What to build

A calibration step between "rented" and "given a shard": train one dataset, record

    $/suite = dph_total * min_per_suite / 60

Reject and destroy a worker above a threshold — at 2.5x worse it is not worth keeping at
any of these hourly prices. Persist the measurement keyed by `(gpu_name, cpu_name)` so
later runs start from evidence instead of repeating the calibration.

The calibration is not free: it costs 4–10 minutes on a box that is already billing. It
pays for itself only because the shard that follows is hours long — say so in the code, so
nobody later "optimises" it away for a short run where it would not.

## Also in scope

- Delete the CPU-tier table and the "pick from the top rows" advice from `vast_search.py`,
  and the CPU-bound reasoning in its docstring. Keep its filtering role: price ceiling,
  reliability, CUDA floor.
- Raise `IMAGE_CUDA` to track the image tag rather than sitting at 12.4 (F2 — a host
  advertising 12.4 cannot necessarily run a cu128 image; CUDA minor-version compatibility
  still requires a driver at least as new as the runtime).

## Open question

Whether the threshold is absolute (`$/suite > 0.010` → destroy) or relative to the best
machine seen this run. Relative adapts to a bad market; absolute is predictable. Decide
with a number from the cached table once it has a few runs in it.

## Comments

**2026-09-02 — measured, and one recommendation here is wrong.**

Built `VastAI/survey_machines.py` and ran it: ten machines rented, timed on a real arm of
the CDNOW ablation (`ar_bounded_16 --shard a`, full trials and simulations). Results in
`VastAI/machine_benchmarks.csv`; `Rules.md` §7 rewritten around them.

What held:

- Ranking by measured `$/study` works and is cheap — ~$0.02 per machine probed, $0.21 for
  the whole survey.
- `IMAGE_CUDA` **should** be raised to track the image tag. Verified: an RTX 3060 whose
  host advertised CUDA 12.8 failed with `Error 804: forward compatibility was attempted on
  non supported HW` against the cu128/CUDA 12.9 image. 12.9 is the right floor.

What did not hold — **"delete the CPU tier table" is the wrong fix**:

- The tier table is the only part of `vast_search.py` that carries signal. CPU family
  correlates r = **+0.94** with seconds-per-study; AMD Zen 2/3 averaged 169 s against 249 s
  for Xeon E5 v3/v4, a **1.48x** gap with the GPU varying freely inside both groups.
- What had to go is the **`cpu_ghz` secondary sort**, which carries none (r = +0.21), and
  the `cpu_cores_effective>=8` floor (r = +0.30, and the slow group spans 4-32 cores).
- This ticket's premise — "the two highest-clock CPUs were the two slowest machines" —
  reproduces, but it is a coincidence of that sample, not a law: every high-clock machine
  in it was a Xeon. Adding a Zen 3 machine (Ryzen 9 5900X, the fastest measured at 154
  s/study) drops the clock correlation from +0.61 to +0.21.

Also settled, and not anticipated by this ticket:

- **The threshold question ("absolute or relative to the best machine seen") is premature.**
  At 4 studies per box the per-machine 95% CIs span ~±25% and overlap freely, so a
  per-offer accept/reject rule would be rejecting on noise. The survey supports ranking
  CPU *families*, not individual offers. A rejection threshold needs more studies per probe
  than a probe can afford — which argues for caching by `(gpu, cpu)` across runs, as this
  ticket already proposed, and thresholding only once a family has several rows.
- **Seeds cost more than machines.** `--shard a` vs `--shard b` on one box: 262 vs 144
  s/study, **1.82x**. Any calibration must fix the shard, and §5's equal split of an arm
  across two shards is not an equal split of work.

Remaining from this ticket: raise `IMAGE_CUDA` to 12.9 (verified above), and add the GPU
architecture filter — torch 2.8+ cu128 ships sm_75+ only, which disqualifies roughly a
third of the sub-$0.06 market. Both are in `survey_machines.py` already; folding them into
`vast_search.py` is the outstanding work.
