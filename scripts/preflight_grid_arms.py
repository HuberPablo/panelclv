"""Pre-flight: build and train every arm a grid declares, tiny, before renting anything.

A grid crosses its panels with an *arm* axis — AR encoding x behavioural cluster x
embedding strategy (`grids.Arm`). Most of those cells have never been run together, and
the ways they break are structural rather than statistical: a feature that lands in
`seq_cols` as a plain numeric the embedder refuses, a bounded flag whose deepest bin
exceeds the calibration window, a cluster label whose cardinality collapses on a small
cohort, a rollout that reads a channel the warm-up never built.

Every one of those raises in the first seconds of a shard. On a rented worker that costs
an image pull, a provision and a billed hour to discover (VastAI/known_failures.md F11 is
exactly this failure, found the expensive way). Here it costs a couple of minutes on the
orchestrator's CPU.

The arms are READ FROM THE GRID, never restated here. This script's whole job is to
answer "will the thing I am about to rent for run?", so a second copy of the arm table
would be a copy that can disagree with the answer — the same reason the model registry is
one table (ADR-0006).

The test is deliberately not a statistical one. One trial, three epochs, two simulated
paths: the forecast it produces is meaningless and is thrown away. The only question
asked is **does this arm run end to end**, plus one thing a bare exception check would
miss — what the arm actually handed the model. An arm whose feature quietly did nothing
passes a crash test and fails the study, so the channel list is printed per arm and
compared against the grid's own unmodified config.

Usage:
    # both stages, every arm (a few minutes on CPU)
    python scripts/preflight_grid_arms.py --grid seasonal_4x4x10

    # stage 1 only: build the datasets, train nothing (seconds)
    python scripts/preflight_grid_arms.py --stage config

    # narrow to what you are about to rent for
    python scripts/preflight_grid_arms.py --model transformer --arm ar_bounded-kmeans_8-projected

Exit status is non-zero if any cell failed, so this can gate a launch script.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import traceback
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

# --- how small "tiny" is ---------------------------------------------------------
# Enough customers that k-means into 8 groups is not degenerate and a batch is a batch;
# few enough that a trial is seconds. The probe panel keeps the grid's own `n_weeks`,
# because the window dates in its PanelConfig are absolute and must land inside it.
PROBE_CUSTOMERS = 200
PROBE_RATE = 0.10
PROBE_CHURN = 0.40

# One trial at one pinned architecture point, not a search. A pinned scalar still
# registers with Optuna (`registry.suggest_param`), so this exercises the same
# suggest -> build -> train -> rollout path a real trial takes, while removing the only
# thing that would make the probe slow or its outcome depend on which point the sampler
# happened to draw. A crash here is the arm's, not the sampler's.
PROBE_SPACE: dict[str, dict[str, object]] = {
    # `embedding_dim` is ignored under the `valendin` embedder, which has no common
    # width; the LSTM suggester skips it there rather than spending the trial on a
    # number that never reaches the model.
    "lstm": {"embedding_dim": 32, "lstm_hidden_size": 32, "dense_units": 32,
             "dropout": 0.0, "learning_rate": 1e-3, "weight_decay": 1e-5,
             "batch_size": 64},
    # nhead divides d_model, so the trial is never pruned before it builds anything.
    "transformer": {"d_model": 32, "nhead": 2, "num_encoder_layers": 1,
                    "dropout": 0.0, "learning_rate": 1e-3, "weight_decay": 1e-5,
                    "batch_size": 64},
    "valendin_lstm": {"learning_rate": 1e-3, "weight_decay": 1e-5, "batch_size": 64},
}

PROBE_TRAINING = {"n_epochs": 3, "patience": 2, "loss_type": "cross_entropy",
                  "verbose": False}


def probe_panel(work: Path, n_weeks: int, start_year: int):
    """One small Pareto/NBD panel, generated fresh so the probe needs no grid on disk.

    Deliberately not the grid's own panels: the point is to test the *arms*, and a
    200-customer panel exercises every code path a 1000-customer one does. Generating it
    also means this runs on a machine that has never pulled the grid.
    """
    dataset_dir, _manifest = ps.generate_pnbd_study(
        [PROBE_RATE], [PROBE_CHURN],
        n_customers=PROBE_CUSTOMERS, n_weeks=n_weeks, n_datasets=1,
        out_path=work, r=2.0, s=2.0, n_weeks_for_churn_rate=52,
        base_seed=42, start_year=start_year, dataset_dir_name="preflight_probe",
    )
    row = next(ps.list_pnbd_datasets(Path(dataset_dir)).itertuples(index=False))
    panel, _truth, _cfg = ps.load_pnbd_dataset(Path(dataset_dir), row.combo, row.dataset)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", default="seasonal_4x4x10",
                        help=f"whose arms to check; one of: {', '.join(available_grids())}")
    parser.add_argument("--stage", choices=("config", "train"), default="train",
                        help="'config' builds the datasets and stops; 'train' also runs one trial")
    parser.add_argument("--model", action="append",
                        help="restrict to these model types (repeatable)")
    parser.add_argument("--arm", action="append",
                        help="restrict to these arm names (repeatable)")
    args = parser.parse_args()

    grid = load_grid(args.grid)
    arms = [grid.arm(n) for n in args.arm] if args.arm else list(grid.arms)
    if not arms:
        raise SystemExit(f"grid {grid.name!r} declares no arms — nothing to check.")

    # Only the trainable models: pareto_nbd has no builder and no rollout, so there is
    # no arm-shaped way for it to break (registry.is_neural).
    models = [m for m in grid.models if m.is_neural]
    if args.model:
        models = [m for m in models if m.model_type in args.model]

    n_cells = len(arms) * len(models)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"grid    : {grid.name}")
    print(f"probe   : {PROBE_CUSTOMERS} customers x {grid.n_weeks} weeks, "
          f"rate {PROBE_RATE}, churn {PROBE_CHURN}")
    print(f"arms    : {len(arms)} x {len(models)} model(s) = {n_cells} cells")
    print(f"stage   : {args.stage}")
    print(f"device  : {device}\n", flush=True)

    failures: list[tuple[str, str]] = []
    channels: dict[str, str] = {}

    with tempfile.TemporaryDirectory(prefix="preflight-") as tmp:
        work = Path(tmp)
        panel = probe_panel(work, grid.n_weeks, grid.start_year)
        studies_base = work / "Studies"
        studies_base.mkdir()

        # The dataset depends on the PanelConfig only, so arms that differ solely in
        # `embedder` share one — the same sharing the real ablation uses, and the reason
        # the embedder axis is cheap. Keyed by the two fields that move it.
        built: dict[tuple, dict] = {}

        for arm in arms:
            key = (arm.ar_features, arm.cluster_features)
            feature_arm = f"{'+'.join(k for k in key[0][:1] + key[1]) or 'baseline'}"
            if key not in built:
                try:
                    started = time.time()
                    data = panel_dataset.prepare_dataset(
                        panel, grid.panel_for(arm), verbose=False
                    )
                except Exception:
                    print(f"FAIL  {arm.name:44s} prepare_dataset")
                    print(traceback.format_exc(limit=3))
                    failures.append((arm.name, "prepare_dataset"))
                    built[key] = None
                    continue
                built[key] = data
                cols = list(data["seq_cols"])
                embedded = sorted(data["embedded_cols"] or {})
                # What the arm actually handed the model. An arm whose feature silently
                # did nothing has the same channels as another and would otherwise pass
                # every crash check while measuring nothing.
                channels[arm.name.rsplit("-", 1)[0]] = (
                    f"F={len(cols)}  {cols}  embedded={embedded}"
                )
                print(f"ok    {arm.name:44s} prepare_dataset  F={len(cols)} "
                      f"({time.time()-started:.1f}s)", flush=True)

            data = built[key]
            if data is None or args.stage == "config":
                continue

            for model in models:
                cell = f"{model.model_type}/{arm.name}"
                started = time.time()
                try:
                    run_study_suite(StudySuiteConfig(
                        studies_base_path=studies_base,
                        # Unique per cell: create_suite_root refuses an existing folder.
                        suite_name=cell.replace("/", "__"),
                        data=data,
                        models=[replace(
                            model,
                            # The arm's embedder pinned exactly as run_pnbd_grid.py pins
                            # it, over a space narrowed to one architecture point.
                            search_space={**PROBE_SPACE[model.model_type],
                                          "embedder": arm.embedder}
                            if model.model_type != "valendin_lstm"
                            else PROBE_SPACE[model.model_type],
                            training=PROBE_TRAINING,
                            n_trials=1,
                        )],
                        n_studies_per_model=1,
                        n_simulations=2,     # a forecast, not an estimate
                        base_seed=grid.base_seed,
                        device=device,
                        keep_only_best_checkpoint=True,
                    ))
                except Exception:
                    print(f"FAIL  {cell}")
                    print(traceback.format_exc(limit=4))
                    failures.append((cell, "run_study_suite"))
                    continue
                print(f"ok    {cell}  ({time.time()-started:.0f}s)", flush=True)

    # --- the report ---------------------------------------------------------------
    print("\n--- channels per feature configuration ---")
    seen: dict[str, str] = {}
    for arm_name, line in channels.items():
        # Two different feature configurations reading identical channels means one of
        # them declared a feature that never reached the tensor. A silent no-op, not a
        # pass, so it is called out rather than printed alongside.
        clash = seen.get(line)
        seen.setdefault(line, arm_name)
        print(f"{arm_name:34s} {line}" + (f"  <-- IDENTICAL TO {clash}" if clash else ""))
        if clash:
            failures.append((arm_name, f"channels identical to {clash}"))

    print(f"\n{n_cells - len(failures)}/{n_cells} cells ok" if args.stage == "train"
          else f"\n{len(channels)} feature configurations built")
    if failures:
        print(f"{len(failures)} FAILED:")
        for cell, where in failures:
            print(f"  {cell}  ({where})")
        raise SystemExit(1)
    print("no failures — the arms are safe to rent for")


if __name__ == "__main__":
    main()
