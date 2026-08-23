# Spec: schedule grid work by measurement, not by prediction

Status: ready-for-agent

Blocked until the `seasonal_4x4x10` grid currently running finishes — this restructures
sharding, and doing it mid-run would strand workers holding fixed shards.

Source: the first distributed run of the Pareto/NBD grid (2026-08-23), eight rented
workers on `seasonal_4x4x10`. Every number below is measured from that run, not modelled.
Failure modes referenced as F<n> are `VastAI/known_failures.md`.

## Problem statement

Two decisions in the current design are made by **predicting** how fast a machine will
be, and both predictions are wrong.

### 1. Machine selection ranks on the wrong signal

`VastAI/vast_search.py` ranks offers by CPU generation and clock, on the documented
reasoning that the workload is CPU-bound: small panels, few batches per epoch, and a
Monte Carlo rollout that loops simulations sequentially in Python, so throughput should
follow single-thread speed. It advises "pick from the top rows, not the cheapest row".

Six workers running the *same* transformer shard say otherwise:

| GPU | CPU | min/suite | $/hr | $/suite |
| --- | --- | ---: | ---: | ---: |
| RTX 3070 | EPYC 7302P @ 3.0 GHz | 3.9 | 0.0844 | **0.0055** |
| RTX 3070 | EPYC 7302P @ 3.0 GHz | 4.0 | 0.0844 | 0.0056 |
| RTX 3060 | Ryzen 9 7950X @ 5.9 GHz | 5.8 | 0.0757 | 0.0073 |
| RTX 2060 | Ryzen 9 5950X @ 5.5 GHz | 7.6 | 0.0711 | 0.0090 |
| RTX 3070 | EPYC 7282 @ 2.8 GHz | 8.4 | 0.0844 | 0.0118 |
| RTX A2000 | EPYC 7282 @ 2.8 GHz | 9.8 | 0.0852 | **0.0139** |

The two highest-clock CPUs available — 7950X at 5.9 GHz and 5950X at 5.5 GHz — are 49%
and 95% *slower* than a 3.0 GHz EPYC. Following the script's own ranking buys the worst
machines. Throughput tracks the GPU tier (3070 > 3060 > 2060 > A2000) more closely than
anything about the CPU, and `nvidia-smi` reports 95–99% GPU utilisation on every worker.

**This does not establish that the workload is GPU-bound.** `utilization.gpu` only
measures that some kernel was resident, which a stream of tiny launch-bound kernels also
achieves; n = 6; GPU and CPU are confounded; host contention is uncontrolled. The
defensible conclusion is narrower and more useful: **published specs do not predict this
workload's throughput, so no ranking built on them can be trusted.**

Cost consequence: a 2.5x spread in $/suite across machines whose $/hr differs by 20%.
Choosing on $/hr optimises the wrong quantity.

### 2. Fixed strides make wall-clock hostage to the slowest worker

`scripts/run_pnbd_grid.py` takes `--shard i/N` and processes manifest rows `i-1::N`.
Every worker gets the same number of datasets regardless of speed. With the fleet above,
the fastest worker finishes its 23 datasets in ~1.5 h and the slowest needs ~3.8 h; the
grid is not complete until the slowest finishes, and the fast machines are destroyed
having idled the difference. Wall-clock is set by `max(shard_time)` when it could be set
by `total_work / total_throughput`.

Equal shards are only correct when workers are equal. Rented workers never are, and
their speed is not knowable in advance (see 1).

## What to build

### A. A pull-based work queue, replacing fixed strides

A worker asks for the next untrained dataset instead of being handed a fixed list. Fast
machines then do proportionally more, automatically, and heterogeneity stops costing
anything. This also removes the need to predict speed at all, which is what makes it the
primary change rather than a refinement.

The claim is not tied to a specific coordinator; the tickets settle that. The constraint
is that the queue must survive a worker dying mid-dataset, must not hand the same dataset
to two workers, and must not require the orchestrator to be reachable at all times — a
worker whose network drops (F10) has to keep working.

Note the existing resume rule already does most of the work: `run_pnbd_grid.py` skips a
suite whose `results.csv` exists. A queue is the coordination that makes "which dataset
next" a runtime question rather than a launch-time one.

### B. Selection by measured cost, not by specs

Before committing a shard to a freshly rented box, time one dataset on it. That is 4–10
minutes on a machine already billing, and it yields the only figure that ranks machines
correctly:

    $/suite = $/hr * min_per_suite / 60

Reject a box above a threshold and destroy it — a 2.5x-worse machine is not worth keeping
even at the same hourly price. Cache the measurement per `(gpu_name, cpu_name)` so the
fleet accumulates knowledge across runs rather than re-learning every time.

`vast_search.py` keeps its role as the offer filter (price ceiling, reliability, CUDA
floor) but stops claiming to rank by expected throughput. Its CPU-tier table and the
"pick from the top rows" advice should go, along with the CPU-bound reasoning in its
docstring, which this run contradicts. Its `IMAGE_CUDA` floor should track the image tag
rather than sitting at 12.4 (F2).

### C. The trade-off rule, written down

With A and B in place the "stronger machine vs more machines" question has a plain answer,
and Rules.md §7 should state it:

- **Total cost** is governed by `$/suite`. Pick machines that minimise it; hourly price is
  not the quantity to minimise.
- **Wall-clock** is governed by aggregate throughput. Add machines until projected
  completion meets the deadline, then stop — each extra machine also costs 5–10 minutes of
  billed provisioning and carries the ~50% first-launch dud rate this run measured.
- The two are independent once shards are dynamic: adding a slow-but-cheap machine
  shortens wall-clock without raising cost per unit work, which is exactly the case fixed
  strides currently punish.

## Out of scope

- Any change to what the models compute. This is scheduling only; the trained artefacts
  must be identical to what fixed strides would have produced.
- Self-destructing workers and credential-scoped API keys. Related, separately motivated.
- Re-litigating whether the workload is GPU-bound. The design deliberately does not depend
  on the answer — that is the point of measuring.

## Acceptance

- A grid completes with workers of visibly different speeds and no worker idle while
  another still has queued datasets.
- Killing a worker mid-dataset loses at most that dataset; its remaining work is picked up
  by others without manual reassignment.
- The trained tree is byte-identical in structure to a fixed-stride run: one suite per
  dataset under `Studies/<grid>__<Model>/<combo>__<dataset>/`, and
  `collect_grid_results` reads it unchanged.
- Machine selection reports a measured `$/suite` per worker, and the run's summary states
  what was rejected and why.
