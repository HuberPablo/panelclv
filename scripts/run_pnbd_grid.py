"""Train one model over one shard of a grid's datasets — the worker's entry point.

A grid is 160 synthetic datasets; a *shard* is a strided slice of them. One worker
runs one model over one shard:

    python scripts/run_pnbd_grid.py --grid seasonal_4x4x10 --model lstm --shard 1/4

Each dataset gets its own study suite with ``n_studies_per_model=1``, written to
``Studies/<grid>__<Model>/<combo>__<dataset>/`` — the layout the grid readers expect.

Why shards are strided, not contiguous
--------------------------------------
Shard ``i/N`` takes manifest rows ``i-1::N``. A contiguous block would hand one worker
only the sparse low-rate corner of the grid, so a worker that dies would cost four
whole cells rather than an even slice of every cell.

Why each model has its own tree
-------------------------------
Two workers training *different* models on the *same* dataset would both target
``<train_base>/<combo>__<dataset>/``, and ``create_suite_root`` refuses a folder that
already exists. Splitting the tree on model keeps every worker's output disjoint, which
is what lets shards be recombined by plain copy (VastAI/Rules.md §4).

Resuming
--------
A suite whose ``results.csv`` exists is skipped — that file is written last, so its
presence means the suite finished. A suite folder without one is a crashed run and is
redone. So re-running a shard resumes it, and re-running a finished shard is a no-op.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# `python scripts/foo.py` puts scripts/ on sys.path, not the repo root, so the
# top-level `grids` package would not be importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from panelclv.data_preparation import panel_dataset
from panelclv.data_preparation import pareto_nbd_simulation as ps
from panelclv.studies import StudySuiteConfig, run_study_suite

from grids import available_grids, load_grid


def parse_shard(text: str) -> tuple[int, int]:
    """``"3/8"`` -> ``(3, 8)``, validated. Shards are 1-indexed for the human."""
    try:
        index, total = (int(part) for part in text.split("/", 1))
    except ValueError:
        raise SystemExit(f"--shard must look like 3/8, got {text!r}")
    if not 1 <= index <= total:
        raise SystemExit(f"--shard {text}: index must be between 1 and {total}")
    return index, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", required=True, help=f"one of: {', '.join(available_grids())}")
    parser.add_argument("--model", required=True, help="model *type*, e.g. lstm")
    parser.add_argument("--shard", default="1/1", help="this worker's slice, e.g. 3/8")
    args = parser.parse_args()

    spec = load_grid(args.grid)
    index, total = parse_shard(args.shard)

    # The grid declares models by type; the worker is told which one it owns.
    models = [m for m in spec.models if m.model_type == args.model]
    if not models:
        raise SystemExit(
            f"grid {spec.name!r} declares no model of type {args.model!r}; "
            f"it has: {', '.join(m.model_type for m in spec.models)}"
        )
    model = models[0]

    if not spec.dataset_dir.is_dir():
        raise SystemExit(
            f"no data at {spec.dataset_dir}. Push it with rsync, or regenerate with:\n"
            f"  python scripts/generate_pnbd_grid.py --grid {spec.name}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_base = spec.train_base(model.name)
    # run_study_suite requires its base directory to already exist.
    train_base.mkdir(parents=True, exist_ok=True)

    datasets = ps.list_pnbd_datasets(spec.dataset_dir)
    shard = list(datasets.itertuples(index=False))[index - 1 :: total]

    print(f"grid    : {spec.name} ({len(datasets)} datasets)")
    budget = f"{model.n_trials} trials/dataset" if model.is_neural else "no search (single fit)"
    print(f"model   : {model.name} ({model.model_type}), {budget}")
    print(f"shard   : {index}/{total} -> {len(shard)} datasets")
    print(f"device  : {device}")
    print(f"output  : {train_base}", flush=True)

    started = time.time()
    done = skipped = 0
    for position, row in enumerate(shard, start=1):
        suite_name = f"{row.combo}__{row.dataset}"
        if (train_base / suite_name / "results.csv").exists():
            skipped += 1
            continue

        panel, _ground_truth, _cfg = ps.load_pnbd_dataset(
            spec.dataset_dir, row.combo, row.dataset
        )
        data_full = panel_dataset.prepare_dataset(panel, spec.panel)

        suite_config = StudySuiteConfig(
            studies_base_path=train_base,
            suite_name=suite_name,
            data=data_full,
            models=[model],
            n_studies_per_model=1,          # one tuned model per dataset, not replicates
            n_simulations=spec.n_simulations,
            base_seed=spec.base_seed,
            device=device,
            keep_only_best_checkpoint=True,  # 160 suites of checkpoints is a lot of disk
            overwrite=True,                  # only reached for a crashed/absent suite
        )
        run_study_suite(suite_config)
        done += 1

        # Elapsed / projected, so a watchdog value can be set from the first few.
        elapsed = time.time() - started
        rate = elapsed / done
        left = rate * (len(shard) - skipped - done)
        print(
            f"[{position}/{len(shard)}] {suite_name}  "
            f"({rate/60:.1f} min/dataset, ~{left/60:.0f} min left)",
            flush=True,
        )

    total_min = (time.time() - started) / 60
    print(f"\nshard {index}/{total} complete: {done} trained, {skipped} already done, "
          f"{total_min:.1f} min")


if __name__ == "__main__":
    main()
