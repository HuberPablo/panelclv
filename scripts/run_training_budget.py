"""Does training the way the paper trains fix the electronics collapse?

`docs/training-budget.md` measures that every archived neural study stopped while its
validation loss was still falling: patience 7 ends a run at a median epoch of 8-16, and
the same model trained without early stopping keeps improving to epoch 88-247. The cause
is not the constant. It is that our recipe is not the reference notebook's -- it trains
at batch 32 with plain Adam for about 90 epochs (~2,300 gradient updates on electronics)
while our winners receive about 32, because patience 7 stops small-batch trials before
their advantage appears and the searched batch set does not contain 32 at all.

This experiment tests that on the one panel where the collapse is documented.

The arms
--------
`archive`   today's settings -- lr / weight decay / batch searched, patience 7,
            n_epochs 100. The control: it reproduces family N.
`paper`     the notebook's recipe, every hyperparameter PINNED and one trial:
            lr 1e-3, weight decay 0 (AdamW with no decay IS Adam), batch 32,
            patience 5, n_epochs 150. No search, so nothing is confounded by
            selection -- which matters because `docs/benchmarks-real-panels.md`
            shows the winning validation loss does not predict the forecast.
`paper90`   the same pinned recipe with `min_epochs=90`. It exists because `paper`
            alone does not reproduce the notebook's TRAINING, only its settings:
            measured here, patience 5 at batch 32 stops electronics at epoch 1 with a
            worse validation loss than `archive`, where the notebook reports ~90
            epochs on its own data. 90 is the notebook's own figure.
`floor50`   `archive`'s search plus `min_epochs=50`: no trial stops before epoch 50,
            whatever the plateau. The form of the fix available to models that have
            no published recipe. Setting it also widens the Optuna pruner's warm-up,
            or floored trials are pruned at epoch 4 before the floor can pay.

Two models, so the arms say something about the developed model and not only the
benchmark: **ValendinLSTM**, whose architecture ADR-0004 freezes, and **LSTM** on
`no_ar-no_cluster-valendin` -- the arm whose per-customer Spearman collapses to 0.036.
They cannot share a dataset: the benchmark refuses any non-embedded channel (F11), so
each model builds its panel through the runner that already owns that configuration.

What this does NOT do
---------------------
It does not touch the published ValendinLSTM rows. These suites are their own family and
sit beside family N; whether any published number is regenerated is
`.scratch/training-budget/issues/06-report-and-decide.md`, after this reports.

Budget
------
160 work items (4 arms x 2 models x 20 replications). Measured on vast.ai, family N ran
this panel at 100 trials and 500 paths in ~102 s per suite, so the searched arms cost
~1.5 h of machine time between them and the pinned arms ~1 h; `floor50` dominates at
~19 h because every one of its 100 trials runs at least 50 epochs. Call it ~22
machine-hours, about $1.60 at the $0.07/hr an RTX 3060 on an EPYC bills at 20 GB
(Rules §7). Money is not the binding constraint here -- wall clock is, and that is what
worker count buys.

Usage:
    # costs nothing: builds both datasets and trains all eight cells tiny
    python scripts/run_training_budget.py --preflight

    # one worker's slice (this is what a rented box runs)
    python scripts/run_training_budget.py --worker 3/8

    # the cheap arms answer the question; floor50 is the expensive follow-up
    python scripts/run_training_budget.py --worker 3/4 --arms archive paper paper90

    # after the run
    python scripts/run_training_budget.py --check-complete
    python scripts/run_training_budget.py --report
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

# The datasets come from the runners that already declare them, so this experiment
# cannot drift onto different windows or a different feature set than the runs it is
# compared against. Both live in scripts/, so add it to the path when this file is
# invoked directly rather than imported.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_real_panel_arms import Arm, build_data as arm_data          # noqa: E402
from run_real_panel_benchmarks import (                              # noqa: E402
    build_data as benchmark_data,
    refuses,
    spearman,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDIES_BASE = REPO_ROOT / "Studies"

EXPERIMENT = "training_budget"
PANEL = "electronics"

# --- budget -----------------------------------------------------------------------
# One suite per (model, arm, replication), so a box lost mid-run costs the replication
# it was on and nothing else -- a multi-study suite cannot be resumed (F26).
N_REPLICATIONS = 20
N_SIMULATIONS = 200
BASE_SEED = 42
# Trials per searched arm. Family N gave the same model 100 on this panel, so the
# searched arms match it: an `archive` row is only a control if its search is the one
# the archive ran.
N_TRIALS = 100

# The LSTM arm under test: no AR channel, no cluster label, the panel's default calendar
# (year index + week sin/cos). `docs/insights-cluster-ablation.md` §5.1 measures its
# per-customer Spearman at 0.036 -- it is the cell this experiment is trying to move.
LSTM_ARM = Arm(name="no_ar-no_cluster-valendin")

# --- the arms ---------------------------------------------------------------------
# Each entry is (search_space, training, n_trials) per model type. A scalar in a search
# space is PINNED by the registry's spec mini-language, which is how the `paper` arm
# reaches batch 32 without touching the registry's searched set {64, 128, 256}.
_SEARCHED = {
    "valendin_lstm": {},                      # the frozen entry's own ranges
    "lstm": {},                               # ditto
}
# The reference notebook's recipe, cell by cell: Adam() at its defaults (lr 1e-3, no
# weight decay), BATCH_SIZE_TRAIN = 32, EarlyStopping(patience=5), MAX_EPOCHS = 150,
# memory_units = dense_units = 128. `scripts/validate_valendin_lstm.py` runs the same
# recipe and reproduces the notebook's published validation loss.
_PAPER = {
    "valendin_lstm": {"learning_rate": 1e-3, "weight_decay": 0.0, "batch_size": 32},
    "lstm": {"embedder": "valendin", "lstm_hidden_size": 128, "dense_units": 128,
             "dropout": 0.0, "learning_rate": 1e-3, "weight_decay": 0.0,
             "batch_size": 32},
}

ARMS: dict[str, dict[str, object]] = {
    "archive": dict(search=_SEARCHED, n_trials=N_TRIALS,
                    training={"n_epochs": 100, "patience": 7}),
    "paper":   dict(search=_PAPER,    n_trials=1,
                    training={"n_epochs": 150, "patience": 5}),
    # The notebook trains ~90 epochs; its patience-5 rule fires only after that. Ours
    # fires at epoch 1 on electronics (measured), so `paper` alone reproduces the
    # settings without reproducing the training. This arm reproduces both.
    "paper90": dict(search=_PAPER,    n_trials=1,
                    training={"n_epochs": 150, "patience": 5, "min_epochs": 90}),
    "floor50": dict(search=_SEARCHED, n_trials=N_TRIALS,
                    training={"n_epochs": 300, "patience": 7, "min_epochs": 50}),
}

MODELS = {"ValendinLSTM": "valendin_lstm", "LSTM": "lstm"}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def build_data(model: str) -> dict:
    """The panel this model reads, from the runner that already declares it.

    Two configurations, not one: ValendinLSTM refuses any non-embedded channel (F11),
    so it takes the benchmark's count-and-embedded-week panel, while the LSTM takes the
    arms runner's `no_ar-no_cluster-valendin` panel with engineered calendar columns.
    Sharing one would either crash the benchmark or change the LSTM's inputs, and then
    the arms below would not be comparable with the runs they are read against.
    """
    if model == "ValendinLSTM":
        data = benchmark_data(PANEL, "2y")
        reason = refuses(data)
        if reason:
            raise RuntimeError(f"ValendinLSTM refuses {PANEL}: {reason}")
        return data
    return arm_data(PANEL, LSTM_ARM)


# ---------------------------------------------------------------------------
# Work items
# ---------------------------------------------------------------------------


def suite_name(model: str, arm: str, replication: int) -> str:
    """`training_budget__<Model>__electronics__<arm>__r<NN>` -- one per work item."""
    return f"{EXPERIMENT}__{model}__{PANEL}__{arm}__r{replication:02d}"


def work_list(arms: list[str]) -> list[tuple[str, str, int]]:
    """Every (model, arm, replication), ARM-MAJOR.

    Arm-major so a strided worker draws a slice of all six cells: a lost box then thins
    every arm evenly instead of emptying one, and a partial run still compares.
    """
    return [(m, a, r) for a in arms for m in MODELS for r in range(N_REPLICATIONS)]


def forecast_path(model: str, arm: str, replication: int) -> Path:
    return (STUDIES_BASE / suite_name(model, arm, replication) / model
            / "Predictions" / "Prediction_1.csv")


def model_spec(model: str, arm: str, training_override: dict | None = None) -> ModelSpec:
    """This (model, arm) cell's declaration, with the arm's search space and controls."""
    cfg = ARMS[arm]
    model_type = MODELS[model]
    training = {**cfg["training"], "verbose": False, "loss_type": "cross_entropy",
                **(training_override or {})}
    return ModelSpec(name=model, model_type=model_type,
                     n_trials=cfg["n_trials"], search_space=dict(cfg["search"][model_type]),
                     training=training)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_worker(index: int, total: int, arms: list[str]) -> int:
    """Train this worker's stride of the work list. Finished items are skipped.

    An item counts as finished when its forecast exists. A folder without one is a
    replication cut short, and a suite cannot be resumed, so it is overwritten.
    """
    mine = work_list(arms)[index - 1::total]
    print(f"worker {index}/{total}: {len(mine)} suites", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    STUDIES_BASE.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}

    for n, (model, arm, rep) in enumerate(mine, start=1):
        name = suite_name(model, arm, rep)
        if forecast_path(model, arm, rep).exists():
            print(f"[{n}/{len(mine)}] {name}: done, skipping", flush=True)
            continue
        if model not in cache:
            cache[model] = build_data(model)
        print(f"[{n}/{len(mine)}] {name}: training", flush=True)
        run_study_suite(StudySuiteConfig(
            studies_base_path=str(STUDIES_BASE),
            suite_name=name,
            n_studies_per_model=1,
            n_simulations=N_SIMULATIONS,
            device=device,
            data=cache[model],
            models=[model_spec(model, arm)],
            base_seed=BASE_SEED + rep,
            overwrite=(STUDIES_BASE / name).exists(),
            keep_only_best_checkpoint=True,
        ))
    return 0


def preflight(arms: list[str]) -> int:
    """Build both datasets and train every cell tiny in a temp dir. Costs nothing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for model in MODELS:
            data = build_data(model)
            print(f"{model:13s} N={len(data['ids']):5d} T_CAL={int(data['T_CAL']):3d} "
                  f"T_HOLD={int(data['T_HOLD']):3d} seq_cols={data['seq_cols']}")
            for arm in arms:
                # 3 epochs and a floor of 2, so a floored cell still exercises the
                # floor rather than silently running as if it were absent.
                tiny = {"n_epochs": 3, "patience": 2}
                if "min_epochs" in ARMS[arm]["training"]:
                    tiny["min_epochs"] = 2
                try:
                    run_study_suite(StudySuiteConfig(
                        studies_base_path=tmp, suite_name=f"preflight__{model}__{arm}",
                        n_studies_per_model=1, n_simulations=2, device=device, data=data,
                        models=[model_spec(model, arm, tiny)],
                        base_seed=BASE_SEED, keep_only_best_checkpoint=True,
                    ))
                    print(f"  {arm:8s} ok")
                except Exception as exc:            # noqa: BLE001 -- report them all
                    print(f"  {arm:8s} FAIL {type(exc).__name__}: {exc}")
                    failures.append(f"{model}/{arm}")
    if failures:
        print(f"\n{len(failures)} cell(s) failed: {failures}. Fix before renting anything.")
        return 1
    print("\nEvery cell builds and trains. Safe to launch.")
    return 0


# ---------------------------------------------------------------------------
# Completeness and scoring
# ---------------------------------------------------------------------------


def check_complete(arms: list[str]) -> int:
    """What the declaration owes against what is on disk. Non-zero on any shortfall."""
    missing: list[str] = []
    for arm in arms:
        for model in MODELS:
            have = sum(forecast_path(model, arm, r).exists() for r in range(N_REPLICATIONS))
            print(f"{arm:8s} {model:13s} {have:2d}/{N_REPLICATIONS}")
            missing += [suite_name(model, arm, r) for r in range(N_REPLICATIONS)
                        if not forecast_path(model, arm, r).exists()]
    for name in missing:
        print(f"  MISSING {name}")
    if missing:
        return 1
    print("complete")
    return 0


METRICS = ("bias_percent", "mape_aggregate", "rmse", "spearman")

# The rows this experiment is read against: family N / family H electronics, as
# `docs/benchmarks-real-panels.md` and `docs/insights-cluster-ablation.md` report them.
REFERENCE = {
    "ValendinLSTM (family N, patience 7)": dict(bias_percent=46.03, mape_aggregate=70.80,
                                                rmse=0.3770, spearman=0.032),
    "LSTM no_ar-no_cluster (family H)":    dict(spearman=0.036),
    "Pareto/NBD":                          dict(bias_percent=-63.02, mape_aggregate=65.65,
                                                rmse=0.3758, spearman=0.297),
}


def score(model_dir: Path, actual: np.ndarray, ref_ids: np.ndarray) -> dict[str, float]:
    """Metrics for one stored forecast, via the single scoring authority."""
    from panelclv.studies import load_model_predictions

    values, ids = load_model_predictions(model_dir, study=1)
    if ids is not None and not np.array_equal(np.asarray(ids), ref_ids):
        raise ValueError(f"{model_dir}: prediction ids do not match the rebuilt cohort")
    return {**compute_forecast_metrics(actual, values),
            "spearman": spearman(values.sum(axis=1), actual.sum(axis=1)),
            "pred_sd": float(values.sum(axis=1).std())}


def report(arms: list[str]) -> None:
    """One table per model: each arm's distribution over its replications.

    Read MAPE and Spearman, not bias: `docs/benchmarks-real-panels.md` measures a refit
    noise floor of 8.9 points of sd on electronics, which is wider than the bias
    differences this experiment is likely to produce.
    """
    for model in MODELS:
        data = build_data(model)
        actual = holdout_actuals(data)
        ref_ids = np.asarray(data["ids"])

        print(f"\n### {model} -- {PANEL}\n")
        print("| arm | n | bias % | sd | MAPE | sd | RMSE | Spearman | sd | pred sd |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for arm in arms:
            rows = [score(forecast_path(model, arm, r).parents[1], actual, ref_ids)
                    for r in range(N_REPLICATIONS) if forecast_path(model, arm, r).exists()]
            if not rows:
                print(f"| {arm} | 0 | — | — | — | — | — | — | — | — |")
                continue
            df = pd.DataFrame(rows)
            print(f"| {arm} | {len(df)} | {df.bias_percent.mean():.2f} | "
                  f"{df.bias_percent.std(ddof=1):.2f} | {df.mape_aggregate.mean():.2f} | "
                  f"{df.mape_aggregate.std(ddof=1):.2f} | {df.rmse.mean():.4f} | "
                  f"{df.spearman.mean():.3f} | {df.spearman.std(ddof=1):.3f} | "
                  f"{df.pred_sd.mean():.2f} |")

    print("\n### Reference rows (archived, for comparison)\n")
    print("| row | bias % | MAPE | RMSE | Spearman |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for name, m in REFERENCE.items():
        cells = " | ".join(f"{m[k]:.4f}" if k in m else "—"
                           for k in ("bias_percent", "mape_aggregate", "rmse", "spearman"))
        print(f"| {name} | {cells} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS),
                        help="which arms to run or read (default: all four)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker", metavar="I/N", help="train this worker's stride")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--check-complete", action="store_true")
    mode.add_argument("--report", action="store_true")
    args = parser.parse_args()

    # Declaration order, never the order they were typed: a worker's stride is an
    # index into the work list, so two workers passing the same arms in a different
    # order would train different items under the same `--worker i/N`.
    arms = [a for a in ARMS if a in set(args.arms)]
    if args.worker:
        index, total = (int(x) for x in args.worker.split("/"))
        sys.exit(run_worker(index, total, arms))
    if args.preflight:
        sys.exit(preflight(arms))
    if args.check_complete:
        sys.exit(check_complete(arms))
    report(arms)


if __name__ == "__main__":
    main()
