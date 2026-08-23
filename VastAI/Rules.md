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

## 4. One model per worker, and each model gets its own tree

A trained grid is one study suite per dataset, with `n_studies_per_model=1`:

```
<train_base>/<combo>__<dataset>/results.csv
```

Two workers running **different models on the same dataset** would both target
`<train_base>/Dataset_10_20__Dataset_1/`, and `create_suite_root` refuses a
pre-existing folder. So the model axis gets its own root:

```
Studies/seasonal_4x4x10__LSTM/<combo>__<dataset>/
Studies/seasonal_4x4x10__Transformer/<combo>__<dataset>/
Studies/seasonal_4x4x10__ParetoNBD/<combo>__<dataset>/
```

Within one model's tree every worker writes disjoint `<combo>__<dataset>/`
directories, so **collecting results is a plain copy** — rsync each worker's tree
into the matching local tree and the shards reassemble themselves. There is no
merge step and no merge code anywhere in this design; if you find yourself writing
one, the split has gone wrong.

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

Worker `i` of `N` takes manifest rows `i::N` — **strided, not a contiguous block**,
so no worker draws only the sparse low-rate corner and a lost worker costs an even
slice of every cell rather than four whole cells.

Split counts are set per model because the models are not the same size of job: a
100-trial Optuna search per dataset is worth many workers, and a single MCMC fit
is worth none.

**Trial budget: 20 Optuna trials per dataset per neural model**, the figure
`Pareto_Datasets.ipynb` used for the existing 4x4x10 grids — not the 100 a
single-panel suite uses, because this budget is paid 160 times over. It buys a
tuned model per dataset, not an exhaustively tuned one.

**Shards must be resumable.** A worker skips any `<combo>__<dataset>/` that already
exists, so a box that dies mid-run is restarted rather than rewound, and re-running
a completed shard is a no-op.

---

## 6. Aggregation concatenates tables, never files

Both grid readers take a `train_base` and return a long table keyed by `model`,
skipping datasets with nothing trained yet. So the per-model trees are combined
*after* reading, not on disk:

```python
results = pd.concat([
    collect_grid_results(dataset_dir, f"Studies/{grid}__{m}")
    for m in ("LSTM", "Transformer", "ParetoNBD")
])
```

Everything downstream — `cell_summary`, `compare_models_table`, `plot_pattern`,
`plot_diff_grid`, and `synthetic_grid`'s latent-truth measures — consumes that
frame rather than paths, so nothing else changes. The result is identical to what
a single machine running the whole grid would have produced.

---

## 7. Choosing machines

**The workload is CPU-bound, not GPU-bound.** A synthetic panel is 1000 × 156, so
an epoch is a handful of batches, and `run_monte_carlo_forecast` loops simulations
sequentially in Python — thousands of tiny autoregressive forward passes dominated
by kernel-launch latency. VRAM demand is trivial.

So the selection rule is **the cheapest GPU that works, on the fastest
single-thread CPU available** — the opposite of a normal GPU search.
`vast_search.py` already encodes this: it filters server-side on the numeric CPU
fields and ranks client-side on a CPU-generation tier table, because vast exposes
CPU generation only as a model-name string. Use it; do not hand-pick from
`vastai search offers`.

Rent with `vast_launch.sh <OFFER_ID>`, which exists because `create` alone is not
enough — it does the create, attaches the SSH key *before* first boot, starts the
container, and polls `cur_state` until the box is genuinely up. Offers go stale in
minutes; a reaped instance means the offer was taken between search and create, so
take the next row rather than retrying the same one.

The image is pinned in `vast_launch.sh`
(`vastai/pytorch:2.10.0-cu128-cuda-12.9-mini-py312`). Keeping every worker on one
image keeps floating-point behaviour comparable across shards.

---

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
