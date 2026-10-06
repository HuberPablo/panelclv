"""Does Prechelt's PQ1 stopping rule forecast the holdout better than patience 7?

`docs/hyperparameter-search.md` §5.2 chose PQ1 on the validation window only: replayed on
recorded curves, it kept epochs whose validation forecast beat patience 7's on every
panel. This is the holdout test of that choice, inside full Optuna searches.

**Two arms, identical but for the stopping rule.** ValendinLSTM (count and week, both
embedded), 100 TPE trials, 20 replications, 500 Monte Carlo paths, the ADR-0008 refit and
the default pruner — family N's protocol — under today's code, which includes the
per-cell validation score (ADR-0010):
- `patience7`: the stopping rule every archived study used;
- `pq1`: `fit_model(stop_pq=1.0)`.
Both allow 200 epochs, so neither is capped. The search space is the archive's (weight
decay searched over 1e-6..1e-2 log, batch over {64, 128, 256}), so family N's and W's
archived studies remain a third, older reference.

**Panels.** Every calibration `run_real_panel_benchmarks` declares: `2y` (CDNOW,
electronics, gift, multichannel), `3y` (electronics, gift, multichannel) and `5y` (the
paper's electronics cohort). `--calibration` picks which, comma-separated, so one fleet
can take the 5y studies (which need a 16 GB GPU for the holdout warm-up) and another the
rest.

Usage:
    python scripts/run_stopping_rule.py --preflight --calibration 2y,3y,5y
    python scripts/run_stopping_rule.py --worker 3/60 --calibration 2y,3y
    python scripts/run_stopping_rule.py --check-complete --calibration 2y,3y,5y
    python scripts/run_stopping_rule.py --report --calibration 2y
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
from panelclv.evaluation.effects import effect
from panelclv.models import compute_forecast_metrics
from panelclv.studies import ModelSpec, StudySuiteConfig, load_model_predictions, run_study_suite

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_real_panel_benchmarks import (CALIBRATIONS, build_data,           # noqa: E402
                                       calibration_tag, spearman)

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDIES_BASE = REPO_ROOT / "Studies"

EXPERIMENT = "stopping"
N_REPLICATIONS = 20
N_TRIALS = 100
N_SIMULATIONS = 500
BASE_SEED = 42
SEARCH_SPACE = {"weight_decay": (1e-6, 1e-2, "log"), "batch_size": {64, 128, 256}}
COMMON = {"n_epochs": 200, "verbose": False, "loss_type": "cross_entropy"}
ARMS = {
    "patience7": {**COMMON, "patience": 7},
    "pq1": {**COMMON, "stop_pq": 1.0},
}
CALS = ["2y"]                   # the calibrations this invocation covers


def suite_name(cal: str, panel: str, arm: str, rep: int) -> str:
    return f"{EXPERIMENT}{calibration_tag(cal)}__ValendinLSTM__{panel}__{arm}__r{rep:02d}"


def forecast_path(cal: str, panel: str, arm: str, rep: int) -> Path:
    return (STUDIES_BASE / suite_name(cal, panel, arm, rep) / "ValendinLSTM"
            / "Predictions" / "Prediction_1.csv")


def work_list() -> list[tuple[str, str, str, int]]:
    """Replication-major, so a strided worker draws from every panel and both arms."""
    return [(c, p, a, r) for r in range(N_REPLICATIONS) for a in ARMS
            for c in CALS for p in sorted(CALIBRATIONS[c])]


def spec(arm: str, n_trials: int = N_TRIALS, training: dict | None = None) -> ModelSpec:
    return ModelSpec(name="ValendinLSTM", model_type="valendin_lstm", n_trials=n_trials,
                     search_space=dict(SEARCH_SPACE), training=training or dict(ARMS[arm]))


def run_worker(index: int, total: int) -> int:
    mine = work_list()[index - 1::total]
    print(f"worker {index}/{total}: {len(mine)} studies", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    STUDIES_BASE.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple[str, str], dict] = {}
    for n, (cal, panel, arm, rep) in enumerate(mine, start=1):
        name = suite_name(cal, panel, arm, rep)
        if forecast_path(cal, panel, arm, rep).exists():
            print(f"[{n}/{len(mine)}] {name}: done, skipping", flush=True)
            continue
        if (cal, panel) not in cache:
            cache.clear()               # one panel at a time; the 5y panel is large
            cache[(cal, panel)] = build_data(panel, cal)
        print(f"[{n}/{len(mine)}] {name}: training", flush=True)
        run_study_suite(StudySuiteConfig(
            studies_base_path=str(STUDIES_BASE), suite_name=name,
            n_studies_per_model=1, n_simulations=N_SIMULATIONS, device=device,
            data=cache[(cal, panel)], models=[spec(arm)], base_seed=BASE_SEED + rep,
            overwrite=(STUDIES_BASE / name).exists(), keep_only_best_checkpoint=True))
    return 0


def preflight() -> int:
    """Every panel and both arms, 2 trials of 6 epochs and 2 paths, in a temp tree."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for cal in CALS:
            for panel in sorted(CALIBRATIONS[cal]):
                data = build_data(panel, cal)
                for arm in ARMS:
                    try:
                        run_study_suite(StudySuiteConfig(
                            studies_base_path=tmp, suite_name=f"pre__{cal}__{panel}__{arm}",
                            n_studies_per_model=1, n_simulations=2, device=device, data=data,
                            models=[spec(arm, 2, {**ARMS[arm], "n_epochs": 6})],
                            base_seed=BASE_SEED, keep_only_best_checkpoint=True))
                        print(f"  {cal} {panel:13s} {arm:10s} ok")
                    except Exception as exc:            # noqa: BLE001 — report them all
                        print(f"  {cal} {panel:13s} {arm:10s} FAIL {type(exc).__name__}: {exc}")
                        failures.append(f"{cal}/{panel}/{arm}")
    if failures:
        print(f"\nfailed: {failures}")
        return 1
    print("\nEvery panel trains and forecasts under both rules. Safe to launch.")
    return 0


def check_complete() -> int:
    missing = [w for w in work_list() if not forecast_path(*w).exists()]
    for cal in CALS:
        for panel in sorted(CALIBRATIONS[cal]):
            for arm in ARMS:
                have = sum(forecast_path(cal, panel, arm, r).exists()
                           for r in range(N_REPLICATIONS))
                print(f"{cal} {panel:13s} {arm:10s} {have:2d}/{N_REPLICATIONS}")
    if missing:
        print(f"{len(missing)} missing")
        return 1
    print("complete")
    return 0


def score(cal: str, panel: str, arm: str, data: dict) -> pd.DataFrame:
    """Holdout metrics, Spearman and the winner's kept epoch, one row per replication."""
    import json
    actual, rows = holdout_actuals(data), []
    for rep in range(N_REPLICATIONS):
        if not forecast_path(cal, panel, arm, rep).exists():
            continue
        mdir = STUDIES_BASE / suite_name(cal, panel, arm, rep) / "ValendinLSTM"
        values, _ = load_model_predictions(mdir, study=1)
        best = json.loads((mdir / "Optuna_Studies/study_01/study_01_best.json").read_text())
        rows.append({"rep": rep, **compute_forecast_metrics(actual, values),
                     "spearman": spearman(values.sum(axis=1), actual.sum(axis=1)),
                     "best_epoch": best["best_user_attrs"]["best_epoch"],
                     "batch": best["best_params"]["batch_size"]})
    d = pd.DataFrame(rows)
    if len(d):
        d["abs_bias"] = d.bias_percent.abs()
    return d


def report(cal: str) -> None:
    """Per panel: both arms' holdout scores and Δ (PQ1 − patience 7) with its interval.

    Resampled independently: training is unseeded and the two arms' searches diverge from
    their first trial, so a shared replication seed does not make them pairs.
    """
    out = []
    for panel in sorted(CALIBRATIONS[cal]):
        data = build_data(panel, cal)
        a, b = score(cal, panel, "patience7", data), score(cal, panel, "pq1", data)
        if len(a) < 2 or len(b) < 2:
            print(f"{cal} {panel}: {len(a)} / {len(b)} replications, skipped")
            continue
        print(f"\n== {cal} {panel}: patience 7 n={len(a)}, PQ1 n={len(b)}; median kept epoch "
              f"{a.best_epoch.median():g} -> {b.best_epoch.median():g}")
        for m, name in [("mape_aggregate", "MAPE"), ("abs_bias", "|bias| %"),
                        ("bias_percent", "bias %"), ("spearman", "Spearman"), ("rmse", "RMSE")]:
            e = effect(b[m], a[m], paired=False, metric=m, panel=panel)
            flag = "*" if e.lo > 0 or e.hi < 0 else " "
            print(f"  {name:9s} {a[m].mean():9.4f} -> {b[m].mean():9.4f}   "
                  f"Δ {e.delta:+.4f} [{e.lo:+.4f}, {e.hi:+.4f}] {flag}")
            out.append({"calibration": cal, "panel": panel, "metric": name,
                        "patience7": a[m].mean(), "pq1": b[m].mean(),
                        "delta": e.delta, "lo": e.lo, "hi": e.hi})
    if out:
        path = REPO_ROOT / ".scratch" / "stopping-rule" / f"report_{cal}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(out).to_csv(path, index=False)
        print(f"\n-> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker", metavar="I/N")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--check-complete", action="store_true")
    mode.add_argument("--report", action="store_true")
    parser.add_argument("--calibration", default="2y",
                        help="comma-separated, from " + ", ".join(CALIBRATIONS))
    args = parser.parse_args()
    CALS[:] = args.calibration.split(",")
    if any(c not in CALIBRATIONS for c in CALS):
        parser.error(f"unknown calibration in {CALS}")
    if args.worker:
        i, n = (int(x) for x in args.worker.split("/"))
        sys.exit(run_worker(i, n))
    if args.preflight:
        sys.exit(preflight())
    if args.check_complete:
        sys.exit(check_complete())
    for cal in CALS:
        report(cal)


if __name__ == "__main__":
    main()
