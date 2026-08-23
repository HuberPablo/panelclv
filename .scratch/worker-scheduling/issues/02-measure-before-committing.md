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
