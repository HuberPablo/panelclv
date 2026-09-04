"""Train one model over one shard of a grid's datasets — the worker's entry point.

A grid is 160 synthetic datasets crossed with its declared *arms* — feature and
embedding configurations (``grids.Arm``). One worker runs one model over a strided slice
of that whole (arm x dataset) product:

    python scripts/run_pnbd_grid.py --grid seasonal_4x4x10 --model lstm --shard 1/4

Each (arm, dataset) gets its own study suite with ``n_studies_per_model=1``, written to
``Studies/<grid>__<Model>__<arm>/<combo>__<dataset>/`` — the layout the grid readers
expect, with one tree per arm. A grid declaring no arms keeps the un-suffixed
``Studies/<grid>__<Model>/`` path its archived suites already use.

Why a shard spans arms rather than owning one
---------------------------------------------
Twelve arms times two models is twenty-four (model, arm) pairs, against a fleet ceiling
of ten workers (VastAI/Rules.md §8), so a worker cannot own a pair. It owns a stride of
the (arm x dataset) product instead, taken over an **arm-major** list so that every
worker draws the same number of datasets from every arm. A worker lost mid-run then
costs an even slice of all of them rather than deleting whole arms from the comparison —
see the ordering note on ``work`` below, which is where that property comes from.
``--arm`` narrows to one when you want a probe or a targeted resume.

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
from dataclasses import replace
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
    parser.add_argument("--arm", help="restrict to one declared arm (default: all of them)")
    # Probe overrides. A shard at the grid's declared budget is hours of work, which is
    # too coarse to answer "is this rented box fast enough to be worth its hour?" — so
    # the same entry point runs a cheap, throwaway version of itself. The suites a probe
    # writes are NOT results: they are trained at a budget the grid does not declare, so
    # a probe writes under --suite-suffix and its tree is deleted, never collected.
    parser.add_argument("--n-trials", type=int,
                        help="override the grid's Optuna budget (probe only)")
    parser.add_argument("--n-simulations", type=int,
                        help="override the grid's Monte Carlo path count (probe only)")
    parser.add_argument("--max-suites", type=int,
                        help="stop after this many suites (probe only)")
    parser.add_argument("--suite-suffix", default="",
                        help="suffix the output tree, so a probe cannot be mistaken for a result")
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

    # A probe is only honest if it is visibly not a result. Overriding the trial budget
    # forces a suffix, so its tree can never be collected into the grid by the plain copy
    # VastAI/Rules.md §4 relies on.
    probing = args.n_trials is not None or args.n_simulations is not None
    if probing and not args.suite_suffix:
        raise SystemExit(
            "--n-trials/--n-simulations change the declared budget, so they need "
            "--suite-suffix to keep the probe's output out of the grid's tree."
        )
    if args.n_trials is not None:
        model = replace(model, n_trials=args.n_trials)
    n_simulations = args.n_simulations or spec.n_simulations

    if not spec.dataset_dir.is_dir():
        raise SystemExit(
            f"no data at {spec.dataset_dir}. Push it with rsync, or regenerate with:\n"
            f"  python scripts/generate_pnbd_grid.py --grid {spec.name}"
        )

    # A grid with no arm axis still has one unit of work per dataset: the `None` arm,
    # whose config is the grid's own and whose tree keeps the un-suffixed path.
    arms = [spec.arm(args.arm)] if args.arm else (list(spec.arms) or [None])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    datasets = ps.list_pnbd_datasets(spec.dataset_dir)
    rows = list(datasets.itertuples(index=False))

    # The unit of work is (arm, dataset), and the stride is taken over the whole
    # product. Arms vary SLOWEST — the list is arm-major, 160 datasets at a time.
    #
    # That ordering is load-bearing, not cosmetic. Striding `i::N` over an arm-major
    # list hands worker `i` exactly `len(rows)/N` datasets out of EVERY arm, because
    # each arm's block is a whole number of strides long (160 datasets, 10 workers ->
    # 16 each). Order the list dataset-major instead and the stride walks the arms in
    # steps of N mod A: at N=10 workers and A=12 arms that is a step of 10, whose orbit
    # is only 6 of the 12 arms, so each worker sees half the arms and a worker lost
    # mid-run guts those six while leaving the other six untouched. A missing arm is
    # not a noisier comparison, it is no comparison — so the even split matters more
    # here than it does across the grid's cells.
    work = [(arm, row) for arm in arms for row in rows]
    shard = work[index - 1 :: total]
    if args.max_suites is not None:
        shard = shard[: args.max_suites]

    print(f"grid    : {spec.name} ({len(rows)} datasets x {len(arms)} arm(s))")
    budget = f"{model.n_trials} trials/dataset" if model.is_neural else "no search (single fit)"
    print(f"model   : {model.name} ({model.model_type}), {budget}")
    print(f"arms    : {', '.join(a.name for a in arms if a) or '(none declared)'}")
    print(f"shard   : {index}/{total} -> {len(shard)} of {len(work)} suites")
    print(f"device  : {device}", flush=True)

    started = time.time()
    done = skipped = 0
    for position, (arm, row) in enumerate(shard, start=1):
        train_base = spec.train_base(model.name, arm.name if arm else None)
        if args.suite_suffix:
            train_base = train_base.with_name(f"{train_base.name}__{args.suite_suffix}")
        # run_study_suite requires its base directory to already exist.
        train_base.mkdir(parents=True, exist_ok=True)

        suite_name = f"{row.combo}__{row.dataset}"
        label = f"{arm.name}/{suite_name}" if arm else suite_name
        if (train_base / suite_name / "results.csv").exists():
            skipped += 1
            continue

        panel, _ground_truth, _cfg = ps.load_pnbd_dataset(
            spec.dataset_dir, row.combo, row.dataset
        )
        # Rebuilt per item rather than cached per arm: `ar_features` and
        # `cluster_features` are PanelConfig fields, so an arm's dataset is a different
        # object, and the stride means consecutive items rarely share one anyway. It
        # costs about a second against a study's minutes.
        data_full = panel_dataset.prepare_dataset(panel, spec.panel_for(arm), verbose=False)

        # The embedder is a search-space key both neural entries declare, so an arm
        # pins it over whatever the model declared rather than needing its own ModelSpec.
        arm_model = model
        if arm is not None and model.is_neural:
            arm_model = replace(
                model, search_space={**model.search_space, "embedder": arm.embedder}
            )

        suite_config = StudySuiteConfig(
            studies_base_path=train_base,
            suite_name=suite_name,
            data=data_full,
            models=[arm_model],
            n_studies_per_model=1,          # one tuned model per dataset, not replicates
            n_simulations=n_simulations,
            base_seed=spec.base_seed,
            device=device,
            keep_only_best_checkpoint=True,  # 1920 suites of checkpoints is a lot of disk
            overwrite=True,                  # only reached for a crashed/absent suite
        )
        run_study_suite(suite_config)
        done += 1

        # Elapsed / projected, so a watchdog value can be set from the first few.
        elapsed = time.time() - started
        rate = elapsed / done
        left = rate * (len(shard) - skipped - done)
        print(
            f"[{position}/{len(shard)}] {label}  "
            f"({rate/60:.1f} min/suite, ~{left/60:.0f} min left)",
            flush=True,
        )

    total_min = (time.time() - started) / 60
    print(f"\nshard {index}/{total} complete: {done} trained, {skipped} already done, "
          f"{total_min:.1f} min")


if __name__ == "__main__":
    main()
