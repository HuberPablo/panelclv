# 03 — Write the fleet-sizing rule into Rules.md §7

Status: ready-for-agent
Blocked by: 01, 02

## Problem

Rules.md §7 currently says to pick "the cheapest GPU that works, on the fastest
single-thread CPU", which the measurements contradict. It offers no guidance on how many
machines to rent, so the first run picked 10 because that was the authorised cap, and
sized the split by model count rather than by measured cost.

## What to write

Replace §7's selection heuristic with the two-quantity rule:

- **Total cost** follows `$/suite`. Minimise that; hourly price is not the target.
- **Wall-clock** follows aggregate throughput. Add machines until projected completion
  meets the deadline, then stop. Each machine costs 5–10 minutes of billed provisioning
  and carries a first-launch dud rate this run measured at roughly 50% (5 of 10: two
  keyless F3, one CUDA-804 F2, two queued-start F12).
- Once shards are dynamic (ticket 01) the two are independent: a slow-but-cheap machine
  shortens wall-clock without raising cost per unit work. Under fixed strides the same
  machine *raised* wall-clock, which is why the old rule feared slow machines.

Record the measured table from the spec as the evidence, with the date and the grid it
came from, so a future reader can see when it goes stale.

## Also

Given the ~50% dud rate, state the over-provisioning rule: rent N+2, keep the first N that
pass calibration (ticket 02), destroy the rest. Serial launch-fail-notice-relaunch cycles
are what made the first run take an hour to reach a full fleet.
