# Running a grid on rented machines

This is the runbook: what to do, in order, to train a grid on rented
[vast.ai](https://vast.ai) boxes and get every result back.

Two other files sit behind it and neither repeats what is here:

- **`Rules.md`** — the *contract*. Why the work is split the way it is, why each
  (model, arm) gets its own tree, how machines are chosen, what the money rules are.
  Read it once before your first run, and again before changing how a run is split.
- **`known_failures.md`** — the *catalogue*, F1–F20. Every entry cost real billed hours.
  Read it before renting; you will hit some of them.

## The shape of a run

One **orchestrator** (this machine) holds the only complete copy of anything. It
generates the panels, rents **workers**, pushes data, and pulls results back. Workers are
disposable: a worker that dies is re-rented, never repaired. Nothing is analysed on a
worker, and nothing exists only on one.

```
   choose/            launch/                    supervise/
 pick an offer  ->  rent it, provision it,  ->  keep it honest, pull results,
 ($/study)          push panels, start           retire it when done
                    the shard
```

Work is split into **shards**. A shard is a stride over the grid's whole
(arm × dataset) product, taken arm-major so every worker draws evenly from every arm —
losing a worker costs a slice of all arms rather than deleting whole arms from the
comparison (`Rules.md` §5). Each (model, arm) writes its own tree, so collecting results
is a plain copy with no merge step (`Rules.md` §4).

## The folders

| Folder | Contents |
| --- | --- |
| `choose/` | `vast_search.py` filters offers and drops GPUs the image cannot run on. `survey_machines.py` rents candidates and *times* them, writing `machine_benchmarks.csv`. |
| `launch/` | `vast_launch.sh` rents an offer and drives it to `running`. `vast_onstart.sh` runs **on the worker** to provision it. `start_shard.sh` waits for provisioning, pushes the panels, and starts the trainer detached. |
| `supervise/` | `supervise.py` is the reconciliation loop. `healthcheck.sh` probes one box. `pull_results.sh`, `reap_finished.sh`, `watch_fleet.sh`, `pin_workers.sh` are standalone loops (below). |
| `state/` | Every log and the fleet state file. Git-ignored, and **not** a source of truth — see step 6. |

## Before you rent anything

Three checks, in this order. Each has cost money to skip.

**1. Push. Workers clone from GitHub, not from this machine (F18).**

```bash
git status --short                       # must be clean
git rev-parse HEAD                       # must equal:
git ls-remote origin main | cut -f1
```

A grid change that is committed locally but unpushed simply is not on the worker. The
fleet trains the *previous* declaration and writes results that look perfectly valid.
Do not push to `main` while a fleet is running either — `vast_onstart.sh` replays
`git pull --ff-only` on every restart, so a restarted box would diverge from its peers.

**2. Generate the panels once, on this machine.**

```bash
python scripts/generate_pnbd_grid.py --grid seasonal_4x4x10
```

**3. Check the grid actually runs before paying for it.**

```bash
python scripts/preflight_grid_arms.py --grid seasonal_4x4x10
```

`ValendinLSTM` cannot take engineered time features (F11); a grid that declares both
crashes every worker on its first suite.

## Renting and running

```bash
python VastAI/choose/vast_search.py --max-price 0.10        # ranked offers
./VastAI/launch/vast_launch.sh <OFFER_ID>                   # rent one, drive to running
```

Pick on **CPU generation**, not price and not GPU. Measured over ten machines, CPU family
correlates with throughput at r = +0.94 while clock gives no usable signal and GPU tier is
close to irrelevant (`Rules.md` §7). The quantity to minimise is **$/study**, not $/hr.

`vast_launch.sh` prints both a direct endpoint and vast's proxy. **Use the direct one**
— the proxy does not reliably accept an instance-attached key (F1):

```bash
vastai show instances --raw | python -c "
import json,sys
for i in json.load(sys.stdin):
    p=(i.get('ports') or {}).get('22/tcp') or []
    print(i['id'], (i.get('public_ipaddr') or '').strip(), p[0]['HostPort'] if p else None)"
```

Then either drive the whole fleet:

```bash
python VastAI/supervise/supervise.py --grid seasonal_4x4x10          # add --dry-run first
```

or start one shard by hand:

```bash
./VastAI/launch/start_shard.sh <HOST> <PORT> seasonal_4x4x10 transformer 3/8
```

`supervise.py` declares the desired state from the grid's `workers` dict, observes the
fleet each cycle, and acts on the difference — renting, restarting, pulling, destroying.
**Always `--dry-run` first**: it prints one cycle and rents nothing.

## While it runs

Three loops, each doing one job. Start them detached; they are independent of
`supervise.py` and safe beside it.

```bash
nohup ./VastAI/supervise/pull_results.sh  900 >> VastAI/state/pull_results.log  2>&1 &
nohup ./VastAI/supervise/reap_finished.sh 300 >> VastAI/state/reap_finished.log 2>&1 &
nohup ./VastAI/supervise/watch_fleet.sh   900 >> VastAI/state/watch_fleet.log   2>&1 &
```

- **`pull_results.sh`** rsyncs every worker's trees back each cycle, so a box lost at 90%
  costs one dataset instead of a shard.
- **`reap_finished.sh`** waits for a shard's exit code, pulls, verifies every remote
  `results.csv` is local *by path*, and only then destroys the box.
- **`watch_fleet.sh`** prints fleet size, `$/hr`, stalls and **billed-to-date**. This is
  the number to watch; check it rather than reasoning about the hourly rate.

`pin_workers.sh <sha>` is the F18 reconciler — it hard-resets any box off the target
commit, skipping boxes whose shard has already started.

To inspect one box: `./VastAI/supervise/healthcheck.sh <INSTANCE_ID>`, which reports
`state / ssh / onstart / cuda / pkg / data / shard`.

**Kill loops by PID, never `pkill -f`** — the pattern matches the killing shell's own
command line (F7, and F13 when the bracket trick is not enough).

## Ending a run — the step that gets skipped

An empty fleet does **not** mean a complete grid. Nothing in the fleet tooling knows how
many suites the grid *owes*; each part checks only its own denominator. Ask directly:

```bash
python scripts/reconcile_grid.py --grid seasonal_4x4x10
```

It expands the declaration, counts `results.csv` on disk, and exits non-zero on any
shortfall. A tree reported `SHORT` is the dangerous case — it looks trained and is not
(F19). Confirm nothing is still billing:

```bash
vastai show instances          # must say: No instances found
```

Do **not** answer "is this shard done" from `VastAI/state/<grid>.json`. That file is a
cache written only by `supervise.py`, and the reaper, `pin_workers.sh` and any manual
`vastai destroy` all move boxes behind its back, so it drifts in both directions (F20).
Read the disk.

## Recovering a partial run

`run_pnbd_grid.py` skips any suite that already has a `results.csv` and passes
`overwrite=True` for the rest, so a resume trains only what is missing — half-written
directories included:

```bash
./VastAI/launch/start_shard.sh <HOST> <PORT> <grid> transformer 1/1 <arm-name>
```

**Seed the worker first, or it retrains the whole arm.** The resume test reads the
*worker's* disk, and `start_shard.sh` does not seed (only `supervise.py` does). Send just
the `results.csv` files — kilobytes against a multi-gigabyte tree, and bandwidth is billed
per GB (F14):

```bash
rsync -az --include='*/' --include='results.csv' --exclude='*' \
  -e "ssh -i ~/.ssh/id_ed25519 -p <PORT>" \
  "Studies/<tree>/" "root@<HOST>:/root/panelclv/Studies/<tree>/"
```

## Money

A worker bills from `start` to `destroy`, not until it goes idle. That is the main way
money is lost here.

- **Standing authorization: up to 10 workers at or below $0.10/hr each**, no approval
  needed. The ceiling is *per machine* — ten at $0.09 is fine, one at $0.11 is not.
  Anything outside that envelope needs a human to pick from a shortlist.
- The ceiling bounds `$/hr` only. **Bandwidth is billed per GB** and the image pull is
  the largest transfer; one crash-looping box once re-pulled the image ~20 times for
  $5.37, against $0.36 of GPU across the whole fleet (F14).
- **The offer price is not the rental price.** You are billed
  `offer + disk_gb * storage_cost / 730` — about +23% at 40 GB on a cheap offer, which is
  why the launcher rents 20 GB (F15).
- Provisioning is billed too, ~5–10 min per box, so more workers is not linearly faster.

## When something breaks

Go to `known_failures.md` first — 20 entries, each with the symptom as it actually
appears. The ones that cost the most were all silent: the fleet looked healthy and the
results were wrong or missing.

Only five (F1, F2, F3, F9, F10) are caught by `healthcheck.sh`. F4 and F12 are caught at
launch. F17 and F18 pass every health check and are caught by `choose/vast_search.py` and
`supervise/pin_workers.sh`. F19 and F20 are properties of the *grid*, not a box, and only
`scripts/reconcile_grid.py` sees them.
