# Rules for the distributed grid study

The decisions behind running the Pareto/NBD grid study across rented vast.ai
machines. This file is the contract: an agent driving a run reads it first, and
anything it does that is not derivable from here is a change of plan, not a
detail.

Written as rules rather than a tutorial, because the expensive failures here are
not "the command was wrong" but "the run finished and the results cannot be
combined".

---

## 1. Roles

**Orchestrator** — the local QEMU/KVM Ubuntu VM. It generates the grid, holds the
split specification, rents the workers, pushes data to them, polls them, pulls the
results back and aggregates. It is the only machine that holds a complete copy of
anything. Claude Code runs here.

**Workers** — rented vast.ai instances. Disposable by design: a worker holds one
model's shard of the grid and nothing that is not reproducible from the repo plus
the orchestrator's copy of the data. A worker that dies is re-rented, never
recovered.

Nothing is ever generated on a worker that the orchestrator does not already have,
and no analysis runs on a worker. Workers train and forecast; that is all.

---

## 2. The grid is fixed, and it is generated once

Each grid is declared by a module in **`grids/`** — its axes, its panels, its
seasonality, the `PanelConfig` that reads it, the models that train on it and how
many workers each model gets. One module per grid, selected by name:

```bash
python scripts/generate_pnbd_grid.py --grid seasonal_4x4x10
python scripts/generate_pnbd_grid.py --list          # what is declared
```

The declaration is the study design, which is why it is committed rather than
edited into a notebook. `scripts/generate_pnbd_grid.py` holds no parameters of its
own; it is the entry point that reads one.

**A grid names its own directory.** `GridSpec.name` is passed to the generator as
`dataset_dir_name`, and every path derives from it —
`Datasets/Synthetic/<name>/` and `Studies/<name>__<Model>/`. Do *not* fall back to
the generator's derived name: it keys only on grid shape and seed, so two 4x4x10
grids at seed 42 that differ in seasonality or panel size collide in one folder and
overwrite each other. An explicit name is equally deterministic across machines —
which is the property the workers actually need — and unique per grid.

Current grid — `seasonal_4x4x10`:

| | |
|---|---|
| axes | `mean_transaction_rate` ∈ {0.01, 0.05, 0.10, 0.30} × `churn_rate` ∈ {0.20, 0.40, 0.60, 0.80} |
| shapes / horizon | `r = s = 2.0`, churn rates defined at 52 weeks |
| seasonality | peaks at weeks 12, 25, 30, 47; amplitude 1.5; width 3 (fixed, not an axis) |
| panels | 1000 customers × 156 weeks, 10 replicates per cell |
| size | 16 cells × 10 = **160 datasets**, 340 MB on disk, 50 MB as `.tar.gz` |
| build time | ~24 s |
| panel checksum | `046541915abe14c560cb38b9ecb4b0c7` |

Reproduce the checksum with:

```bash
find Datasets/Synthetic/seasonal_4x4x10 -name '*.csv' ! -name index.csv \
  | sort | xargs cat | md5sum
```

Checksum the **panels**, and exclude `index.csv` — that file embeds absolute
`panel_path`s, so it differs per machine by design. Nothing reads that column:
both `list_pnbd_datasets` and `load_pnbd_dataset` rebuild from disk, which is what
makes a generated tree relocatable between machines.

---

## 3. Data reaches a worker by rsync, not by regeneration

The orchestrator generates the grid once and pushes it:

```bash
rsync -avz --partial -e "ssh -p <PORT>" \
  Datasets/Synthetic/seasonal_4x4x10/ \
  root@<HOST>:/root/panelclv/Datasets/Synthetic/seasonal_4x4x10/
```

`--partial` resumes a dropped transfer rather than restarting it, and a re-run
skips what is already there, so this is safe to repeat after a worker reboots.
Workers may be pushed to in parallel.

At 50 MB compressed this costs a few minutes across a dozen workers, which is why
it beats the alternatives: **no public artifact** to publish, version and later
clean up, and it is what `vast_onstart.sh` already provisions rsync for.

*Rejected:* publishing the grid as a GitHub Release asset. It works and would be
marginally faster per worker, but it puts a 50 MB blob in public for no gain at
this scale. Revisit only if the worker count grows enough that home upstream
becomes the bottleneck.

*Fallback:* a worker can regenerate the grid itself with
`python scripts/generate_pnbd_grid.py --grid <name>` (~30 s, zero bandwidth). This is reproducible
because seeds are handed out deterministically in generation order — but it relies
on NumPy's PCG64 producing identical draws, which holds for a given NumPy version
and is not guaranteed across versions. **If a worker regenerates, verify the panel
checksum against §2 before trusting the results.** rsync is the default precisely
because it needs no such check.

---

## 4. Every (model, arm) pair gets its own tree

A trained grid is one study suite per dataset per arm, with `n_studies_per_model=1`:

```
<train_base>/<combo>__<dataset>/results.csv
```

Two workers running **different models on the same dataset** would both target
`<train_base>/Dataset_10_20__Dataset_1/`, and `create_suite_root` refuses a
pre-existing folder. The same collision happens between two *arms* of one model. So
both axes go into the root:

```
Studies/seasonal_4x4x10__LSTM__no_ar-no_cluster-valendin/<combo>__<dataset>/
Studies/seasonal_4x4x10__Transformer__ar_bounded-kmeans_8-projected/<combo>__<dataset>/
Studies/seasonal_4x4x10__ParetoNBD/<combo>__<dataset>/
```

A grid declaring **no** arms keeps the un-suffixed `Studies/<grid>__<Model>/` path —
`GridSpec.train_base(model, arm=None)`. That is not a courtesy: the archived
`seasonal_4x4x10` suites are stored under it and `docs/insights-study.md` cites it, so
adding an arm axis must not move a grid's own history.

Within one (model, arm) tree every worker writes disjoint `<combo>__<dataset>/`
directories, so **collecting results is a plain copy** — rsync each worker's tree
into the matching local tree and the shards reassemble themselves. There is no
merge step and no merge code anywhere in this design; if you find yourself writing
one, the split has gone wrong.

**A probe is not a result.** `run_pnbd_grid.py --n-trials/--n-simulations` trains at a
budget the grid does not declare, so it refuses to run without `--suite-suffix`, which
moves its output to `<train_base>__<suffix>/`. That tree is deleted, never collected.

---

## 5. How the work is split

The split is declared in the grid module, as **a number of workers per model
type**:

```python
GRID = GridSpec(
    ...,
    workers={"transformer": 8, "lstm": 4, "pareto_nbd": 0},   # -> 12 workers
)
```

It lives beside the models it splits because the right number depends on the trial
budget declared a few lines above it; separating them is how the two drift apart.

- `N > 0` — that model's 160 datasets are divided across `N` rented workers.
- `0` — the model does **not** go to vast; it runs on the orchestrator. This is
  the right setting for `pareto_nbd`: a single deterministic MCMC fit per dataset,
  pure NumPy/SciPy, no Optuna stage and no GPU need. It runs locally while the
  rented workers churn.

Worker `i` of `N` takes work items `i::N` — **strided, not a contiguous block**,
so no worker draws only the sparse low-rate corner and a lost worker costs an even
slice of every cell rather than four whole cells.

**A work item is an (arm, dataset) pair, not a dataset.** A grid that declares arms
(`grids.Arm` — the feature and embedding configurations every model is trained under)
multiplies its work by the arm count: `seasonal_4x4x10`'s 12 arms x 160 panels is 1,920
suites per model. Twenty-four (model, arm) pairs against a ten-worker ceiling (§8) means
a worker cannot own a pair, so it takes a stride of the whole product and trains a little
of every arm.

**The product is ordered arm-major, and that is load-bearing.** Striding `i::N` over an
arm-major list gives worker `i` exactly `160/N` datasets from every arm. Order it
dataset-major instead and the stride walks the arms in steps of `N mod A`: at ten workers
and twelve arms that is a step of ten, whose orbit covers only six arms, so each worker
sees half of them and a lost worker guts those six and leaves the rest untouched. An arm
missing entirely is not a noisier result, it is no result — which is why this matters more
than the cell-coverage argument above.

`--arm <name>` narrows a run to one arm, for a probe or a targeted resume.

Split counts are set per model because the models are not the same size of job: a
100-trial Optuna search per dataset is worth many workers, and a single MCMC fit
is worth none.

**Trial budget: 20 Optuna trials per dataset per neural model**, the figure
`Pareto_Datasets.ipynb` used for the existing 4x4x10 grids — not the 100 a
single-panel suite uses, because this budget is paid 160 times over *per arm*. It buys a
tuned model per dataset, not an exhaustively tuned one.

The budget must be **equal across arms and across models**, or a difference between two
of them is attributable to search effort rather than to the axis under test. That is why
the LSTM and the Transformer both declare 20: the grid module briefly carried 10 for the
LSTM, which would have made the model comparison unequal.

**Shards must be resumable.** A worker skips any `<combo>__<dataset>/` that already
exists, so a box that dies mid-run is restarted rather than rewound, and re-running
a completed shard is a no-op.

---

## 6. Aggregation concatenates tables, never files

Both grid readers take a `train_base` and return a long table keyed by `model`,
skipping datasets with nothing trained yet. So the per-(model, arm) trees are combined
*after* reading, not on disk:

```python
results = pd.concat([
    collect_grid_results(spec.dataset_dir, spec.train_base(m, a.name), arm=a.name)
    for m in ("LSTM", "Transformer")
    for a in spec.arms
] + [
    # Pareto/NBD has no arm axis: it reads the panel, not the engineered features.
    collect_grid_results(spec.dataset_dir, spec.train_base("ParetoNBD"))
])
```

**Pass `arm=`.** Every arm trains the same models on the same panels, so without the
label their rows are identical in every field you could group by — `model` reads "LSTM"
for all twelve. The tree is what identifies the arm, so the label has to come from the
caller; `collect_grid_results` cannot recover it from anything stored inside. Omitting
the argument returns exactly the frame the archived reads return, which is what keeps
`make_grid_figures.py` and the pre-arm suites working untouched.

Everything downstream — `cell_summary`, `compare_models_table`, `plot_pattern`,
`plot_diff_grid`, and `synthetic_grid`'s latent-truth measures — consumes that
frame rather than paths, so nothing else changes. The result is identical to what
a single machine running the whole grid would have produced.

---

## 7. Choosing machines

**The workload is CPU-bound, not GPU-bound.** A synthetic panel is 1000 x 156, so an
epoch is a handful of batches, and `run_monte_carlo_forecast` loops simulations
sequentially in Python — thousands of tiny autoregressive forward passes dominated by
kernel-launch latency. VRAM demand is trivial.

What follows was **measured** on 2026-09-02, not reasoned from specs. Ten machines were
rented and timed on one real arm of the CDNOW AR-encoding ablation
(`--arm ar_bounded_16 --shard a`, full trial and simulation counts), and the results
live in `VastAI/machine_benchmarks.csv`. `VastAI/survey_machines.py` is the instrument;
re-run it when the market has moved or the workload changes.

### The quantity to minimise is $/study, not $/hr

    $/study = billed_dph * seconds_per_study / 3600

`billed_dph`, not the offer's price — see F15. An offer quotes GPU rental only; the
instance is billed `offer + disk_gb * storage_cost / 730`, about $0.011/hr at 40 GB,
which is a 23% markup on a $0.041 offer.

| CPU | gen | GPU | billed $/hr | s/study | $/study |
| --- | --- | --- | ---: | ---: | ---: |
| Ryzen 9 5900X | Zen 3 | RTX 2080 Ti | 0.0911 | 154 | 0.00390 |
| **EPYC 7452** | Zen 2 | RTX 3060 | 0.0655 | 175 | **0.00318** |
| **EPYC 7402** | Zen 2 | RTX 3060 | 0.0644 | 178 | **0.00319** |
| Xeon E5-2620 v4 | Broadwell | RTX 3080 | 0.0456 | 262 | 0.00332 |
| Xeon E5-2667 v3 | Haswell | RTX 3060 Ti | 0.0578 | 223 | 0.00357 |
| Xeon E5-2637 v4 | Broadwell | RTX 3060 | 0.0644 | 243 | 0.00435 |
| Xeon E5-1660 v3 | Haswell | RTX 3060 | 0.0618 | 261 | 0.00448 |
| Xeon E5-2686 v4 | Broadwell | RTX 4060 | 0.0644 | 257 | 0.00460 |

The first row is the fleet these were compared against, running the same command at the
same seed. **An RTX 3060 on an EPYC Rome is the buy: 18% cheaper per replication at
1.13x the wall-clock.**

### What predicts throughput, and what does not

Across the eight machines above, against seconds-per-study:

| field | r | verdict |
| --- | ---: | --- |
| CPU family is Xeon E5 v3/v4 | **+0.94** | this is the whole effect |
| `cpu_ghz` | +0.21 | no usable signal |
| `cpu_cores_effective` | +0.30 | no usable signal |

AMD Zen 2/3 averaged 169 s/study against 249 s for Xeon E5 v3/v4 — **1.48x**, with the
GPU varying freely inside both groups.

**GPU tier is close to irrelevant here.** An RTX 3080 on a Xeon E5-2620 v4 ran 262
s/study; an RTX 3060 on an EPYC 7452 ran 175. Two tiers of GPU lost to the CPU. This
corrects `.scratch/worker-scheduling/spec.md`, which read the first fleet as "throughput
tracks the GPU tier" — in that fleet the fast GPUs happened to sit on EPYCs, and holding
one variable fixed separates them.

**Clock is not the signal, and the old rule's second half was wrong.** This section used
to say "the cheapest GPU that works, on the fastest single-thread CPU available", and
`vast_search.py` sorted on `cpu_ghz`. Generation is what matters; clock only correlates
with it in samples where all the fast-clocked machines are one family.

### Rent 20 GB, not 40

The image layers live on the host, outside the instance's writable overlay — a box rented
with 20 GB reports `23M used, 20G avail` once provisioned. Nothing here needs 40: the
panels are megabytes and a full grid shard writes ~435 MB. This is ~$0.006/hr, about 11%
of a cheap machine's total, and **larger than the gap between the best and worst machine
choice above.**

### What the survey does not establish

- **Per-machine rankings are not resolved.** At 4 studies per box the individual 95% CIs
  span roughly +/-25% and overlap freely. Rank CPU *families*; do not read the table as an
  ordering of individual offers.
- **A third of the cheap market cannot run the image at all.** torch 2.8+ cu128 ships
  sm_75 and up, so every Pascal and Maxwell card is out. Verified on a rented GTX 1070
  with a current driver: `torch.cuda.is_available()` returned True and the first kernel
  launch failed with `no kernel image is available for execution on the device`. The
  health check in `vast_onstart.sh` cannot catch this, so `vast_search.py` filters it.
- **One workload.** Everything above is the LSTM on CDNOW. The transformer on the
  synthetic grid may rank machines differently, though it is weak corroboration that the
  first grid fleet's fastest machine was also an EPYC.
- **The seed costs more than the machine.** On one box, `--shard a` ran 262 s/study and
  `--shard b` 144 s — **1.82x** from `base_seed` alone. Never compare timings across
  shards, and note that this makes an arm's two shards unequal work (section 5 splits
  them as though they were equal).

## 8. Cost and lifecycle

**A worker bills from `start` until `destroy`, not until it goes idle.** This is
the main way money is lost here.

- Every worker gets a **watchdog**: a maximum lifetime after which it is destroyed
  unconditionally, whatever state the run is in. A shard that overruns its budget
  is a shard to re-run, not to keep paying for.
- A worker is destroyed **as soon as its results are pulled and verified**, not at
  the end of the whole study. Models finish at different times; the fast ones stop
  costing immediately.
- `vastai show instances` is checked at the end of every run. Nothing is left
  running.
- Provisioning is billed too — roughly 5–10 minutes per worker for image pull, apt,
  pip install and the data push. **Worker count is chosen from measured
  per-dataset wall-clock, not from the shape of the grid**, because past some point
  more workers buy overhead and failure modes rather than speed.

**Standing authorization: up to 10 workers at or below $0.10/hr may be rented
without asking.** Inside that envelope the agent searches, picks, launches,
provisions, runs, polls, retrieves and destroys on its own judgement, and reports
what it spent. Outside it — a higher price, or more than 10 machines at once — the
agent shortlists offers with their `$/hr` and waits for a human pick.

The ceiling is per machine, not per fleet: ten boxes at $0.09/hr is authorized,
one at $0.11/hr is not.

**The ceiling bounds `$/hr` only.** Bandwidth is a separate meter, billed per GB,
and the image pull is the largest transfer a worker makes — so a box inside the
hourly ceiling can still run up a bill many times its rental. One did: $5.37 of
downloads on an instance that never finished provisioning (F14). `vast_search.py`
now carries its own `--max-bandwidth-cost` ceiling, and worker count has to be
weighed against the per-worker image pull, not just the compute.

Two constraints the ceiling does not express, which still apply:

- **Take the CPU generation, not the cheapest row.** `vast_search.py` labels each
  offer `current` / `recent` / `older` / `ancient`. Only `current` and `recent`
  qualify — the `ancient` Xeon E5 v3/v4 class is where a launch-bound workload
  crawls regardless of price, and there are usually enough recent-generation
  offers under the ceiling to fill a fleet of ten.
- **The watchdog still binds.** An authorized rental is not an unbounded one; every
  worker carries a maximum lifetime (§8 above) and is destroyed when its results
  are pulled.

---

## 9. Reproducibility

The split does not change the results. Study `i` of a model uses `base_seed + i`,
models never interact, and each dataset's suite is independent — so a grid trained
across twelve workers is the same grid trained on one machine, provided every
worker runs the same commit, the same image and the same data.

The one caveat is hardware: workers land on different GPUs, and floating-point
results can differ slightly between them. Same seeds and same statistics, not
guaranteed bit-identical. If a thesis claim needs bitwise reproduction, it must be
re-run on one machine.

---

## 10. Prerequisites on the orchestrator

Before a run can be driven from the QEMU VM, that machine needs:

- `vastai` CLI and an API key at `~/.config/vastai/vast_api_key`
- an SSH keypair at `~/.ssh/id_ed25519` (+ `.pub`) — `vast_launch.sh` attaches the
  public half to every instance before first boot
- `rsync` (both ends need it; `vast_onstart.sh` installs it on the worker)
- the repo cloned, and a Python environment with the package importable

These currently live on the workstation, not on the QEMU VM. **Moving the
orchestrator means moving the vast API key and the SSH key with it**, or
generating a new keypair and registering it with vast.

---

## Open — not yet decided

- **Whether `valendin_lstm` is in scope.** The registry declares four models; this
  study currently names three.
- **The watchdog's actual value and the `$/hr` ceiling.** Both are pending a pilot
  that measures per-dataset wall-clock on rented hardware.
