"""Does ValendinLSTM forecast with less bias when its weights are trained longer?

Family `real_panel_benchmarks_cal5y` (ValendinLSTM on the paper's electronics split,
20 studies) spread its holdout bias from -7.9% to +37.4% although every winning trial
sat at the same validation loss to three decimals. The hyperparameters did not explain
it: the least and most biased studies were searched to the same learning rate and batch
size. The one weak signal was the epoch the weights came from — a later `best_epoch`
went with a lower bias (Spearman -0.43 over 19 studies) — and those epochs were early,
10-16, because patience 7 stops a run on a flat validation curve.

This experiment takes that signal and tests it directly.

The arms
--------
Both pin the hyperparameters of the least-biased study of that family (r13: bias -2.0%,
learning rate 0.002195, batch 32, weight decay 0) and run one trial, so nothing is
selected by Optuna and the arms differ only in how long the kept weights trained.

`from20`   weights from epoch 20 or later: `select_from_epoch=20`.
`from30`   weights from epoch 30 or later: `select_from_epoch=30`.

Within that window the best-by-validation epoch is still the one kept, and patience 7
counts from there. The full-calibration refit (ADR-0008) and the 500-path forecast are
the family's, unchanged.

Budget
------
2 arms x 20 replications = 40 suites, one per work item. Replication r seeds the
forecast from `BASE_SEED + r`, as the benchmark family does, so replication r of each
arm is paired with replication r there. Training is unseeded (CLAUDE.md priority 3).

Usage:
    # costs nothing: builds the panel and trains both arms tiny
    python scripts/run_epoch_floor_5y.py --preflight

    # one worker's slice (this is what a rented box runs)
    python scripts/run_epoch_floor_5y.py --worker 3/4

    python scripts/run_epoch_floor_5y.py --check-complete
    python scripts/run_epoch_floor_5y.py --report
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from panelclv.data_preparation.target_channel import holdout_actuals
from panelclv.models import compute_forecast_metrics
from panelclv.studies import ModelSpec, StudySuiteConfig, run_study_suite

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_real_panel_benchmarks as benchmarks  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDIES_BASE = REPO_ROOT / "Studies"
EXPERIMENT = "epoch_floor_cal5y"
PANEL, CAL = "electronics", "5y"

N_REPLICATIONS = 20
N_SIMULATIONS = benchmarks.N_SIMULATIONS
BASE_SEED = benchmarks.BASE_SEED

# r13 of `real_panel_benchmarks_cal5y`, read from its study_01_best.json. A scalar in a
# search space is pinned by the registry's spec mini-language.
PINNED = {"learning_rate": 0.002195217471728685, "weight_decay": 0.0, "batch_size": 32}
TRAINING = {"n_epochs": 100, "patience": 7, "verbose": False, "loss_type": "cross_entropy"}
ARMS: dict[str, int] = {"from20": 20, "from30": 30}      # arm -> select_from_epoch


def suite_name(arm: str, replication: int) -> str:
    """`epoch_floor_cal5y__ValendinLSTM__electronics__<arm>__r<NN>`."""
    return f"{EXPERIMENT}__ValendinLSTM__{PANEL}__{arm}__r{replication:02d}"


def work_list() -> list[tuple[str, int]]:
    """Every (arm, replication), REPLICATION-major so each worker gets both arms."""
    return [(a, r) for r in range(N_REPLICATIONS) for a in ARMS]


def forecast_path(arm: str, replication: int) -> Path:
    return (STUDIES_BASE / suite_name(arm, replication) / "ValendinLSTM"
            / "Predictions" / "Prediction_1.csv")


def model_spec(arm: str, training_override: dict | None = None) -> ModelSpec:
    training = {**TRAINING, "select_from_epoch": ARMS[arm], **(training_override or {})}
    return ModelSpec(name="ValendinLSTM", model_type="valendin_lstm", n_trials=1,
                     search_space=dict(PINNED), training=training)


def build_data() -> dict:
    data = benchmarks.build_data(PANEL, CAL)
    reason = benchmarks.refuses(data)
    if reason:
        raise RuntimeError(f"ValendinLSTM refuses {PANEL}: {reason}")
    return data


def run_worker(index: int, total: int) -> int:
    """Train this worker's stride. A suite with a forecast is skipped; one without is
    a replication cut short and is overwritten (a suite cannot be resumed)."""
    mine = work_list()[index - 1::total]
    print(f"worker {index}/{total}: {len(mine)} suites", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    STUDIES_BASE.mkdir(parents=True, exist_ok=True)
    data = build_data()
    for n, (arm, rep) in enumerate(mine, start=1):
        name = suite_name(arm, rep)
        if forecast_path(arm, rep).exists():
            print(f"[{n}/{len(mine)}] {name}: done, skipping", flush=True)
            continue
        print(f"[{n}/{len(mine)}] {name}: training", flush=True)
        run_study_suite(StudySuiteConfig(
            studies_base_path=str(STUDIES_BASE),
            suite_name=name,
            n_studies_per_model=1,
            n_simulations=N_SIMULATIONS,
            device=device,
            data=data,
            models=[model_spec(arm)],
            base_seed=BASE_SEED + rep,
            overwrite=(STUDIES_BASE / name).exists(),
            keep_only_best_checkpoint=True,
        ))
    return 0


def preflight() -> int:
    """Build the panel and train both arms tiny in a temp dir. Costs nothing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = build_data()
    print(f"N={len(data['ids'])} T_CAL={int(data['T_CAL'])} T_HOLD={int(data['T_HOLD'])} "
          f"seq_cols={data['seq_cols']}")
    with tempfile.TemporaryDirectory() as tmp:
        for arm in ARMS:
            # 4 epochs with selection from epoch 3, so the restriction is exercised.
            tiny = {"n_epochs": 4, "patience": 1, "select_from_epoch": 3}
            root = run_study_suite(StudySuiteConfig(
                studies_base_path=tmp, suite_name=f"preflight__{arm}",
                n_studies_per_model=1, n_simulations=2, device=device, data=data,
                models=[model_spec(arm, tiny)], base_seed=BASE_SEED,
                keep_only_best_checkpoint=True,
            ))
            print(f"  {arm} ok -> {root}")
    print("\nBoth arms build and train. Safe to launch.")
    return 0


def check_complete() -> int:
    missing = [suite_name(a, r) for a, r in work_list() if not forecast_path(a, r).exists()]
    for arm in ARMS:
        have = sum(forecast_path(arm, r).exists() for r in range(N_REPLICATIONS))
        print(f"{arm:7s} {have:2d}/{N_REPLICATIONS}")
    for name in missing:
        print(f"  MISSING {name}")
    return 1 if missing else 0


METRICS = benchmarks.METRICS


def report() -> None:
    """Each arm's distribution beside the searched family it is read against."""
    data = build_data()
    actual = holdout_actuals(data)                               # (N, T_HOLD)
    ref_ids = np.asarray(data["ids"])
    rows = []
    for arm in ARMS:
        for rep in range(N_REPLICATIONS):
            if forecast_path(arm, rep).exists():
                rows.append({"arm": arm, "replication": rep,
                             **benchmarks.score(forecast_path(arm, rep).parents[1],
                                                actual, ref_ids)})
    for rep in range(benchmarks.N_REPLICATIONS):
        path = benchmarks.forecast_path(PANEL, rep, CAL)
        if path.exists():
            rows.append({"arm": "searched (family cal5y)", "replication": rep,
                         **benchmarks.score(path.parents[1], actual, ref_ids)})
    df = pd.DataFrame(rows)
    print(f"N = {len(ref_ids)}, holdout transactions = {int(actual.sum())}.\n")
    print("| arm | n | metric | mean | sd | median | min | max |")
    print("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for arm, g in df.groupby("arm", sort=False):
        for m in METRICS:
            s = g[m]
            print(f"| {arm} | {len(g)} | {m} | {s.mean():.4f} | {s.std(ddof=1):.4f} | "
                  f"{s.median():.4f} | {s.min():.4f} | {s.max():.4f} |")
    zero = compute_forecast_metrics(actual, np.zeros_like(actual, dtype=float))
    print(f"\nall-zero: bias {zero['bias_percent']:.1f}, MAPE {zero['mape_aggregate']:.1f}, "
          f"RMSE (customer total) {zero['rmse_customer_total']:.4f}")


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
