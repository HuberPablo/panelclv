# 01 — Replace fixed strides with a pull-based work queue

Status: ready-for-agent
Blocked by: the running grid must finish first

## Problem

`scripts/run_pnbd_grid.py --shard i/N` gives every worker an equal slice regardless of
speed, so wall-clock is `max(shard_time)` rather than `total_work / total_throughput`. On
the first distributed run the fastest worker was 2.5x the slowest, so roughly a third of
the fleet's paid time was spent idle-after-finishing.

## What to decide

Where the queue lives. Three candidates, in increasing order of moving parts:

1. **Orchestrator-served claims.** The supervisor hands each worker its next dataset over
   SSH when the previous one lands. Simple and needs no new service, but a worker whose
   network drops (F10) stalls until the supervisor reaches it again.
2. **Deterministic self-service.** Each worker holds the full manifest and picks the next
   dataset not present in its *local* results tree, offset by a per-worker start position
   so two workers rarely collide. Needs no coordinator at all; collisions waste duplicate
   work rather than corrupting anything, and the supervisor's existing result-seeding
   (it rsyncs collected suites onto workers) shrinks the collision window each cycle.
3. **A real queue** (a small service, or a lock file on shared storage). Correct, and more
   infrastructure than this problem justifies.

Option 2 is the one to beat: it keeps the property that a worker needs nothing from the
orchestrator to make progress, which is what kept this run alive through repeated SSH
failures. Its cost is occasional duplicated datasets.

## Requirements

- No dataset is trained twice in a way that corrupts output. Duplicate *work* is
  acceptable if the loser's result is simply overwritten or discarded.
- A worker killed mid-dataset loses at most that dataset.
- A worker with no network keeps training.
- The output tree is unchanged: one suite per dataset under
  `Studies/<grid>__<Model>/<combo>__<dataset>/`.
- `--shard i/N` may remain as an escape hatch for reproducing a run exactly, but must not
  be the default path.
