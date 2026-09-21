"""Do the two levers add? Training budget x cluster label, crossed, on all four panels.

Everything measured so far moved one lever at a time, and on electronics each moved the
per-customer Spearman from the same floor:

    training  (patience 7 -> floored)   0.027 -> 0.177   family T, no cluster label
    inputs    (no cluster -> kmeans_8)  0.039 -> 0.27    cluster ablation, unfloored
    Pareto/NBD                                   0.297

**The cell where both are on has never been run.** If they compose, the model passes the
statistical benchmark on the one metric where it still loses. If they do not -- if both
roads end near 0.27 -- then a ceiling exists that neither training nor this input crosses,
which is the case `docs/absorbing-death-state.md` makes for a per-customer survival
variable. Either answer is worth the run; the cost of not knowing is that every existing
comparison confounds the two.

The design
----------
2 x 2, crossed with two models and four panels, 20 replications a cell.

    training   `archive`  patience 7, n_epochs 100, the registry's search, 100 trials
               `floored`  the paper's recipe pinned, min_epochs 90, ONE trial
    inputs     `no_cluster` / `kmeans_8`

**Why the floored arm searches nothing.** On family T the pinned single-trial recipe and
the 100-trial floored search are statistically indistinguishable on the frozen benchmark
(MAPE p = 0.32, Spearman p = 0.56, |bias| p = 0.62), and `docs/training-budget.md` §14
shows the search's criterion is wrong-signed against the holdout anyway. One trial costs
30 s where the search costs 22 minutes, and that is what makes four panels affordable.

Two configurations, because the benchmark refuses any non-embedded channel (F11)
------------------------------------------------------------------------------
`ValendinLSTM` reads the count, the calendar week and (when the arm carries it) the
cluster label, all embedded. `LSTM` reads the count embedded, `week_sin`/`week_cos`, and
the same label.

**The LSTM here carries no year index, on any panel.** Family T's electronics arm did,
inherited from the archived configs; `run_real_panel_arms.py` already argues against it
for CDNOW, where a year index is constant in calibration and out of its fitted range for
the back half of the holdout. That objection holds on every panel in this family, so the
calendar is sin/cos everywhere -- bounded, periodic, with no range to leave. It makes this
family internally consistent and means an electronics cell here is NOT the same
configuration as family T's; read the controls in this family, not across.

One thing to say wherever the result is quoted
----------------------------------------------
`kmeans_8` is k-means over the Pareto/NBD sufficient statistics `(t_x, x, T)`. A model
carrying it has been handed the summary the statistical benchmark fits, so "it beats
Pareto/NBD" means something narrower than it sounds.

Usage:
    python scripts/run_factorial.py --preflight
    python scripts/run_factorial.py --worker 3/8
    python scripts/run_factorial.py --check-complete
    python scripts/run_factorial.py --report
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
from panelclv.data_preparation import panel_dataset
from panelclv.data_preparation.target_channel import holdout_actuals
from panelclv.models import compute_forecast_metrics
from panelclv.studies import ModelSpec, StudySuiteConfig, load_model_predictions, run_study_suite

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_real_panel_benchmarks import CLEAN, WINDOWS, refuses, spearman   # noqa: E402
from run_training_budget import _PAPER                                    # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDIES_BASE = REPO_ROOT / "Studies"

EXPERIMENT = "factorial"
N_REPLICATIONS = 20
N_SIMULATIONS = 200
BASE_SEED = 42

MODELS = {"ValendinLSTM": "valendin_lstm", "LSTM": "lstm"}
CLUSTERS = {"no_cluster": (), "kmeans_8": ("kmeans_8",)}

TRAINING_ARMS: dict[str, dict] = {
    # The status quo, and this family's control: the search and the stopping rule every
    # archived neural result was produced under.
    "archive": dict(search=lambda family: {}, n_trials=100,
                    training={"n_epochs": 100, "patience": 7}),
    # The paper's recipe, pinned, trained for the paper's own ~90 epochs.
    "floored": dict(search=lambda family: dict(_PAPER[family]), n_trials=1,
                    training={"n_epochs": 150, "patience": 5, "min_epochs": 90}),
}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def panel_config(panel: str, model: str, cluster: str) -> PanelConfig:
    """The panel as this model must read it, with or without the cluster label."""
    shared = dict(id_col="Id", target_col="Transactions", frequency="weekly",
                  time_cols=("year", "week"), cluster_features=CLUSTERS[cluster],
                  **WINDOWS[panel])
    if model == "ValendinLSTM":
        # Every channel embedded, or the frozen benchmark refuses the dataset (F11).
        return PanelConfig(time=("week",),
                           embedded_cols={"Transactions": "auto", "week": "auto"},
                           **shared)
    return PanelConfig(time_features={"add_week_sin_cos": True},
                       embedded_cols={"Transactions": "auto"}, **shared)


def build_data(panel: str, model: str, cluster: str) -> dict:
    path = CLEAN / f"{panel}_customer_week_panel.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Datasets/ is gitignored, so a rented worker needs the "
            f"panel pushed to it (VastAI/Rules.md §3).")
    data = panel_dataset.prepare_dataset(pd.read_csv(path),
                                         panel_config(panel, model, cluster), verbose=False)
    data["panel_name"] = panel
    if model == "ValendinLSTM" and (reason := refuses(data)):
        raise RuntimeError(f"ValendinLSTM refuses {panel}/{cluster}: {reason}")
    return data


# ---------------------------------------------------------------------------
# Work items
# ---------------------------------------------------------------------------


def suite_name(panel: str, model: str, training: str, cluster: str, rep: int) -> str:
    return f"{EXPERIMENT}__{model}__{panel}__{training}-{cluster}__r{rep:02d}"


def cells() -> list[tuple[str, str, str, str]]:
    """Every (panel, model, training, cluster) — the 32 cells of the design."""
    return [(p, m, t, c) for p in WINDOWS for m in MODELS
            for t in TRAINING_ARMS for c in CLUSTERS]


def work_list() -> list[tuple[str, str, str, str, int]]:
    """Every cell x replication, CELL-major, so a strided worker draws from all 32."""
    return [(*cell, r) for cell in cells() for r in range(N_REPLICATIONS)]


def forecast_path(panel: str, model: str, training: str, cluster: str, rep: int) -> Path:
    return (STUDIES_BASE / suite_name(panel, model, training, cluster, rep) / model
            / "Predictions" / "Prediction_1.csv")


def model_spec(model: str, training: str, override: dict | None = None) -> ModelSpec:
    arm = TRAINING_ARMS[training]
    family = MODELS[model]
    return ModelSpec(
        name=model, model_type=family, n_trials=arm["n_trials"],
        search_space=arm["search"](family),
        training={**arm["training"], "verbose": False, "loss_type": "cross_entropy",
                  **(override or {})})


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_worker(index: int, total: int) -> int:
    mine = work_list()[index - 1::total]
    print(f"worker {index}/{total}: {len(mine)} suites", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    STUDIES_BASE.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple[str, str, str], dict] = {}

    for n, (panel, model, training, cluster, rep) in enumerate(mine, start=1):
        name = suite_name(panel, model, training, cluster, rep)
        if forecast_path(panel, model, training, cluster, rep).exists():
            print(f"[{n}/{len(mine)}] {name}: done, skipping", flush=True)
            continue
        key = (panel, model, cluster)
        if key not in cache:
            cache.clear()               # one dataset at a time; four panels do not fit
            cache[key] = build_data(*key)
        print(f"[{n}/{len(mine)}] {name}: training", flush=True)
        run_study_suite(StudySuiteConfig(
            studies_base_path=str(STUDIES_BASE), suite_name=name,
            n_studies_per_model=1, n_simulations=N_SIMULATIONS, device=device,
            data=cache[key], models=[model_spec(model, training)],
            base_seed=BASE_SEED + rep,
            overwrite=(STUDIES_BASE / name).exists(), keep_only_best_checkpoint=True))
    return 0


def preflight() -> int:
    """Build all 16 datasets and train every cell tiny. Costs nothing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for panel, model, training, cluster in cells():
            if training != "archive":       # one training arm is enough to prove wiring
                continue
            try:
                data = build_data(panel, model, cluster)
                for arm in TRAINING_ARMS:
                    tiny = {"n_epochs": 3, "patience": 2}
                    if "min_epochs" in TRAINING_ARMS[arm]["training"]:
                        tiny["min_epochs"] = 2
                    spec = model_spec(model, arm, tiny)
                    spec.n_trials = 1
                    run_study_suite(StudySuiteConfig(
                        studies_base_path=tmp,
                        suite_name=f"pre__{panel}__{model}__{arm}__{cluster}",
                        n_studies_per_model=1, n_simulations=2, device=device, data=data,
                        models=[spec], base_seed=BASE_SEED, keep_only_best_checkpoint=True))
                print(f"  {panel:13s} {model:13s} {cluster:11s} ok  "
                      f"seq_cols={data['seq_cols']}")
            except Exception as exc:                    # noqa: BLE001 — report them all
                print(f"  {panel:13s} {model:13s} {cluster:11s} FAIL "
                      f"{type(exc).__name__}: {exc}")
                failures.append(f"{panel}/{model}/{cluster}")
    if failures:
        print(f"\n{len(failures)} cell(s) failed: {failures}")
        return 1
    print("\nEvery cell builds and trains. Safe to launch.")
    return 0


def check_complete() -> int:
    missing = [suite_name(*w) for w in work_list() if not forecast_path(*w).exists()]
    for panel, model, training, cluster in cells():
        have = sum(forecast_path(panel, model, training, cluster, r).exists()
                   for r in range(N_REPLICATIONS))
        flag = "" if have == N_REPLICATIONS else "   <-- short"
        print(f"{panel:13s} {model:13s} {training:8s} {cluster:11s} "
              f"{have:2d}/{N_REPLICATIONS}{flag}")
    print(f"\n{len(missing)} missing of {len(work_list())}")
    return 1 if missing else 0


def report() -> None:
    """The 2x2 per panel and model, and what the interaction says."""
    rows = []
    for panel, model, training, cluster in cells():
        data = build_data(panel, model, cluster)
        actual = holdout_actuals(data)
        ref_ids = np.asarray(data["ids"])
        for rep in range(N_REPLICATIONS):
            p = forecast_path(panel, model, training, cluster, rep)
            if not p.exists():
                continue
            values, ids = load_model_predictions(p.parents[1], study=1)
            if ids is not None and not np.array_equal(np.asarray(ids), ref_ids):
                raise ValueError(f"{p}: prediction ids do not match the rebuilt cohort")
            rows.append(dict(panel=panel, model=model, training=training,
                             cluster=cluster, rep=rep,
                             **compute_forecast_metrics(actual, values),
                             spearman=spearman(values.sum(axis=1), actual.sum(axis=1))))
    d = pd.DataFrame(rows)
    if d.empty:
        print("no forecasts yet")
        return
    d.to_csv(REPO_ROOT / ".scratch" / "training-budget" / "results" / "factorial.csv",
             index=False)

    for metric, arrow in (("spearman", "higher is better"),
                          ("mape_aggregate", "lower is better")):
        print(f"\n## {metric}  ({arrow})\n")
        print("| panel | model | archive / no_cluster | archive / kmeans_8 "
              "| floored / no_cluster | **floored / kmeans_8** | interaction |")
        print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for (panel, model), g in d.groupby(["panel", "model"]):
            cell = {(t, c): g[(g.training == t) & (g.cluster == c)][metric].mean()
                    for t in TRAINING_ARMS for c in CLUSTERS}
            # Additive prediction vs what the crossed cell actually did: positive means
            # the two levers reinforce, negative means they overlap.
            add = (cell[("archive", "kmeans_8")] + cell[("floored", "no_cluster")]
                   - cell[("archive", "no_cluster")])
            inter = cell[("floored", "kmeans_8")] - add
            print(f"| {panel} | {model} | {cell[('archive','no_cluster')]:.3f} | "
                  f"{cell[('archive','kmeans_8')]:.3f} | "
                  f"{cell[('floored','no_cluster')]:.3f} | "
                  f"**{cell[('floored','kmeans_8')]:.3f}** | {inter:+.3f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker", metavar="I/N")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--check-complete", action="store_true")
    mode.add_argument("--report", action="store_true")
    args = parser.parse_args()

    if args.worker:
        i, n = (int(x) for x in args.worker.split("/"))
        sys.exit(run_worker(i, n))
    if args.preflight:
        sys.exit(preflight())
    if args.check_complete:
        sys.exit(check_complete())
    report()


if __name__ == "__main__":
    main()
