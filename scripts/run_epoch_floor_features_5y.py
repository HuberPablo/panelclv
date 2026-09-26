"""Do bounded activity flags or a behaviour cluster reduce ValendinLSTM's bias?

The companion of `scripts/run_epoch_floor_5y.py`, which trains the frozen ValendinLSTM
on the 5y electronics split with the least-biased study's hyperparameters pinned and
the kept weights taken from epoch 20 (`from20`) or 30 (`from30`) onward. This runner
repeats both arms with one input added.

The model
---------
The frozen benchmark reads week and count only and refuses anything else (ADR-0004),
so the inputs go to `models.MultinomialLSTMModel` built in the benchmark's shape: the
`valendin` embedder (raw sqrt(n)+1 embeddings, concatenated), LSTM 128, dense 128,
dropout 0. With dropout 0 its forward pass is the benchmark's layer for layer, and its
parameter shapes on the base inputs are identical, so each cell here differs from the
matching `run_epoch_floor_5y` cell in the added input and nothing else.

The features
------------
`ar_bounded_52`  nested flags active_in_last_{2,4,8,16,32,52}_periods plus
                 has_transacted_before (`run_real_panel_arms.bounded_flags`).
                 Recomputed each holdout week from the sampled path, so no true holdout
                 value is read (`docs/feature_engineering.md`). Concatenated raw.
`kmeans_8`       one static label per customer: k-means with K=8 on (t_x, x, T) at the
                 end of calibration. Embedded (8 classes).

Both keep the embedded calendar week, as the benchmark does.

Budget
------
2 features x 2 arms x 20 replications = 80 suites. Replication r seeds the forecast from
`BASE_SEED + r`, as in the companion run, so cells pair by replication.

Usage:
    python scripts/run_epoch_floor_features_5y.py --preflight
    python scripts/run_epoch_floor_features_5y.py --worker 3/4
    python scripts/run_epoch_floor_features_5y.py --check-complete
    python scripts/run_epoch_floor_features_5y.py --report
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from panelclv.configs.panel_config import PanelConfig
from panelclv.data_preparation.panel_dataset import prepare_dataset
from panelclv.data_preparation.target_channel import holdout_actuals
from panelclv.studies import ModelSpec, StudySuiteConfig, run_study_suite

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_epoch_floor_5y as floor  # noqa: E402
import run_real_panel_benchmarks as benchmarks  # noqa: E402
from run_real_panel_arms import bounded_flags  # noqa: E402

STUDIES_BASE = floor.STUDIES_BASE
EXPERIMENT = floor.EXPERIMENT
PANEL, CAL = floor.PANEL, floor.CAL
N_REPLICATIONS = floor.N_REPLICATIONS

FEATURES: dict[str, dict[str, tuple[str, ...]]] = {
    "ar_bounded_52": {"ar_features": bounded_flags(52)},
    "kmeans_8":      {"cluster_features": ("kmeans_8",)},
}

# The benchmark's architecture, pinned, plus the same pinned training settings.
PINNED = {"embedder": "valendin", "lstm_hidden_size": 128, "dense_units": 128,
          "dropout": 0.0, **floor.PINNED}


def suite_name(feature: str, arm: str, replication: int) -> str:
    """`epoch_floor_cal5y__LSTM__electronics__<feature>-<arm>__r<NN>`."""
    return f"{EXPERIMENT}__LSTM__{PANEL}__{feature}-{arm}__r{replication:02d}"


def work_list() -> list[tuple[str, str, int]]:
    """Every (feature, arm, replication), REPLICATION-major so each worker gets all four cells."""
    return [(f, a, r) for r in range(N_REPLICATIONS) for f in FEATURES for a in floor.ARMS]


def forecast_path(feature: str, arm: str, replication: int) -> Path:
    return (STUDIES_BASE / suite_name(feature, arm, replication) / "LSTM"
            / "Predictions" / "Prediction_1.csv")


def model_spec(arm: str, training_override: dict | None = None) -> ModelSpec:
    training = {**floor.TRAINING, "select_from_epoch": floor.ARMS[arm],
                **(training_override or {})}
    return ModelSpec(name="LSTM", model_type="lstm", n_trials=1,
                     search_space=dict(PINNED), training=training)


def build_data(feature: str) -> dict:
    """The benchmark's 5y panel config (count and embedded week) plus one feature."""
    config = PanelConfig(
        id_col="Id",
        target_col="Transactions",
        frequency="weekly",
        time_cols=("year", "week"),
        time=("week",),
        embedded_cols={"Transactions": "auto", "week": "auto"},
        **FEATURES[feature],
        **benchmarks.WINDOWS_5Y[PANEL],
    )
    data = prepare_dataset(pd.read_csv(benchmarks.panel_path(PANEL, CAL)), config,
                           verbose=False)
    data["panel_name"] = PANEL
    return data


def run_worker(index: int, total: int) -> int:
    """Train this worker's stride. A suite with a forecast is skipped; one without is
    a replication cut short and is overwritten (a suite cannot be resumed)."""
    mine = work_list()[index - 1::total]
    print(f"worker {index}/{total}: {len(mine)} suites", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    STUDIES_BASE.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}
    for n, (feature, arm, rep) in enumerate(mine, start=1):
        name = suite_name(feature, arm, rep)
        if forecast_path(feature, arm, rep).exists():
            print(f"[{n}/{len(mine)}] {name}: done, skipping", flush=True)
            continue
        if feature not in cache:
            cache[feature] = build_data(feature)
        print(f"[{n}/{len(mine)}] {name}: training", flush=True)
        run_study_suite(StudySuiteConfig(
            studies_base_path=str(STUDIES_BASE),
            suite_name=name,
            n_studies_per_model=1,
            n_simulations=floor.N_SIMULATIONS,
            device=device,
            data=cache[feature],
            models=[model_spec(arm)],
            base_seed=floor.BASE_SEED + rep,
            overwrite=(STUDIES_BASE / name).exists(),
            keep_only_best_checkpoint=True,
        ))
    return 0


def preflight() -> int:
    """Build both panels and train every cell tiny in a temp dir. Costs nothing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with tempfile.TemporaryDirectory() as tmp:
        for feature in FEATURES:
            data = build_data(feature)
            print(f"{feature}: N={len(data['ids'])} T_CAL={int(data['T_CAL'])} "
                  f"seq_cols={data['seq_cols']} embedded={data['embedded_cols']}")
            for arm in floor.ARMS:
                tiny = {"n_epochs": 4, "patience": 1, "select_from_epoch": 3}
                run_study_suite(StudySuiteConfig(
                    studies_base_path=tmp, suite_name=f"preflight__{feature}__{arm}",
                    n_studies_per_model=1, n_simulations=2, device=device, data=data,
                    models=[model_spec(arm, tiny)], base_seed=floor.BASE_SEED,
                    keep_only_best_checkpoint=True,
                ))
                print(f"  {arm} ok")
    print("\nEvery cell builds and trains. Safe to launch.")
    return 0


def check_complete() -> int:
    missing = [suite_name(*item) for item in work_list() if not forecast_path(*item).exists()]
    for feature in FEATURES:
        for arm in floor.ARMS:
            have = sum(forecast_path(feature, arm, r).exists() for r in range(N_REPLICATIONS))
            print(f"{feature:14s} {arm:7s} {have:2d}/{N_REPLICATIONS}")
    for name in missing:
        print(f"  MISSING {name}")
    return 1 if missing else 0


def report() -> None:
    """Each cell's distribution beside the no-feature ValendinLSTM cell of the same arm."""
    base = floor.build_data()
    actual = holdout_actuals(base)                               # (N, T_HOLD)
    ref_ids = np.asarray(base["ids"])
    rows = []
    for arm in floor.ARMS:
        for rep in range(N_REPLICATIONS):
            path = floor.forecast_path(arm, rep)
            if path.exists():
                rows.append({"cell": f"none (ValendinLSTM) {arm}",
                             **benchmarks.score(path.parents[1], actual, ref_ids)})
        for feature in FEATURES:
            for rep in range(N_REPLICATIONS):
                path = forecast_path(feature, arm, rep)
                if path.exists():
                    rows.append({"cell": f"{feature} {arm}",
                                 **benchmarks.score(path.parents[1], actual, ref_ids)})
    df = pd.DataFrame(rows)
    print(f"N = {len(ref_ids)}, holdout transactions = {int(actual.sum())}.\n")
    print("| cell | n | metric | mean | sd | median | min | max |")
    print("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for cell, g in df.groupby("cell", sort=False):
        for m in benchmarks.METRICS:
            s = g[m]
            print(f"| {cell} | {len(g)} | {m} | {s.mean():.4f} | {s.std(ddof=1):.4f} | "
                  f"{s.median():.4f} | {s.min():.4f} | {s.max():.4f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker", metavar="I/N", help="train this worker's stride")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--check-complete", action="store_true")
    mode.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.worker:
        index, total = (int(x) for x in args.worker.split("/"))
        sys.exit(run_worker(index, total))
    if args.preflight:
        sys.exit(preflight())
    if args.check_complete:
        sys.exit(check_complete())
    report()


if __name__ == "__main__":
    main()
