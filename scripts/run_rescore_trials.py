#!/usr/bin/env python
"""Score every retained trial of an archived study through the production forecast path.

Optuna ranks a study's trials by teacher-forced validation cross-entropy on the
validation window; the number the thesis reports is a Monte Carlo rollout over the
holdout year. Whether the first ranking is the second one is an open question, and the
archive cannot answer it: a suite forecasts its winning trial and nothing else.

It can be answered without re-searching anything, because some archived studies still
hold the checkpoints of their non-winning trials (the `keep_only_best_checkpoint`
cleanup did not always finish). This tool takes such a study, and for every trial whose
checkpoint survived, runs exactly what the runner runs for the winner — the
full-calibration refit (ADR-0008) and the registry's rollout (ADR-0006) — then scores it
with `compute_forecast_metrics`. The output is one row per trial: its validation loss
beside its holdout error.

The study is read entirely from its own archive: the panel and windows come from the
suite's `config.json`, the model type and forecast seed from the model's, the
hyperparameters and feature subset from `study_01_trials.csv`. Nothing about a
particular panel or model is written here.

Training is unseeded (CLAUDE.md priority 3), so the refit of one checkpoint is not
reproducible run to run. `--repeat` refits the same trial several times, which measures
that noise floor — without it a difference between two trials cannot be read.

    python scripts/run_rescore_trials.py --out-dir Rescored \
        Studies/real_panel_benchmarks__ValendinLSTM__multichannel__r14

Several suites at once, sharded across rented workers exactly as every other runner here
is (`VastAI/launch/add_workers.sh`), one output CSV per suite, a finished suite skipped.
The `run_` prefix is load-bearing: it is how `VastAI/supervise/reap_finished.sh` sees
that a box is still working and does not retire it mid-slice.

    python scripts/run_rescore_trials.py --out-dir Rescored --worker 3/6 \
        "Studies/real_panel_benchmarks__ValendinLSTM__*"
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from panelclv.configs.panel_config import PanelConfig
from panelclv.data_preparation import panel_dataset
from panelclv.models import compute_forecast_metrics
from panelclv.registry import rollout_for
from panelclv.trials import refit_best_trial

# Spearman over customer holdout totals is not part of `compute_forecast_metrics`; the
# benchmark report owns it, and the arm report already reaches for it the same way.
from run_real_panel_benchmarks import spearman

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN = REPO_ROOT / "Datasets" / "Dataset_clean"

# Integer hyperparameters come back from the trials CSV as floats; the builders index
# and size tensors with them, so they have to be handed over as ints.
INT_PARAMS = ("lstm_hidden_size", "dense_units", "batch_size", "n_epochs", "patience",
              "d_model", "nhead", "num_encoder_layers", "embedding_dim")


def archived_study(suite: Path) -> tuple[dict, dict, Path, str]:
    """The suite's own record of what it ran: panel config, model config, study dir."""
    suite_cfg = json.loads((suite / "config.json").read_text())
    model_name = suite_cfg["models"][0]["name"]
    model_cfg = json.loads((suite / model_name / "config.json").read_text())
    study_dir = suite / model_name / "Optuna_Studies" / "study_01"
    return suite_cfg, model_cfg, study_dir, model_name


def build_data(suite_cfg: dict, panel_csv: Path) -> dict:
    """Rebuild the study's tensors from the panel config it stored verbatim."""
    config = PanelConfig(**suite_cfg["panel_config"])
    return panel_dataset.prepare_dataset(pd.read_csv(panel_csv), config, verbose=False)


def trial_shim(row: pd.Series, study_dir: Path, family: str) -> SimpleNamespace:
    """A stand-in for the `FrozenTrial` the refit reads: params, user attrs, number.

    `refit_best_trial` touches only `study.best_trial`, so pointing it at one trial is a
    matter of handing it that trial instead of the winner. The archived checkpoint paths
    are the rented worker's (`/root/panelclv/...`), so the path is rebuilt from this
    study's own directory rather than trusted verbatim.
    """
    params = {c[len("params_"):]: row[c] for c in row.index if c.startswith("params_")}
    params = {k: (int(v) if k in INT_PARAMS else v) for k, v in params.items()}
    dropped = row.get("user_attrs_dropped_features", "")
    return SimpleNamespace(
        number=int(row["number"]),
        params=params,
        value=float(row["value"]),
        user_attrs={
            "dropped_features": "" if pd.isna(dropped) else str(dropped),
            "checkpoint_path": str(study_dir / "checkpoints" / study_dir.name
                                   / f"{family}_trial_{int(row['number'])}.pth"),
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("suites", nargs="+", help="archived suite directories; a wildcard is "
                                             "expanded here, so it can be quoted")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="one CSV per suite, named after it; a complete one is skipped")
    p.add_argument("--worker", default=None, metavar="I/N",
                   help="take this worker's stride of the suite list (1-based)")
    p.add_argument("--panel-csv", type=Path, default=None,
                   help="panel to rebuild from; default: Datasets/Dataset_clean/<panel>_"
                        "customer_week_panel.csv, named after the suite")
    p.add_argument("--repeat", type=int, default=1,
                   help="refits per trial; >1 measures the unseeded refit's own spread")
    p.add_argument("--limit", type=int, default=None, help="score only the N best trials")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    # A wildcard reaches here unexpanded when the caller quoted it (the fleet launcher
    # passes the runner as one string), so expand it the same way either way.
    suites = sorted({Path(m) for pattern in a.suites for m in (glob.glob(pattern) or [pattern])})
    if a.worker:
        index, total = (int(x) for x in a.worker.split("/"))
        suites = suites[index - 1::total]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(suites)} suite(s) to score", flush=True)

    for suite in suites:
        out = a.out_dir / f"{suite.name}.csv"
        rows = rescore(suite, a, out)
        print(f"{suite.name}: {len(rows)} refit(s) scored -> {out}", flush=True)


def rescore(suite: Path, a: argparse.Namespace, out: Path) -> list[dict]:
    """Refit and forecast every retained trial of one suite, writing as it goes."""
    suite_cfg, model_cfg, study_dir, _ = archived_study(suite)
    family = model_cfg["model_type"]
    panel = suite.name.split("__")[2]
    panel_csv = a.panel_csv or CLEAN / f"{panel}_customer_week_panel.csv"

    kept = {int(f.stem.rsplit("_", 1)[1])
            for f in (study_dir / "checkpoints" / study_dir.name).glob(f"{family}_trial_*.pth")}
    trials = pd.read_csv(study_dir / "study_01_trials.csv")
    trials = trials[(trials.state == "COMPLETE") & trials.number.isin(kept)].sort_values("value")
    if a.limit:
        trials = trials.head(a.limit)
    if trials.empty:
        # A suite whose cleanup did finish kept only the winner's checkpoint, and a
        # single trial answers nothing; it is listed, reported and passed over.
        print(f"{suite.name}: no retained trial checkpoints, skipping", flush=True)
        return []

    # Resume: a suite whose CSV already holds every refit it owes is not redone, so a
    # lost worker's slice can be re-run without repeating what it finished.
    expected = len(trials) * a.repeat
    if out.exists() and len(pd.read_csv(out)) >= expected:
        print(f"{suite.name}: already complete ({expected} refits), skipping", flush=True)
        return []

    data = build_data(suite_cfg, panel_csv)
    forecaster = rollout_for(family)
    # The runner forecasts study i from `base_seed + i`; the model config stores the
    # resulting seed, so the rollout here draws the same simulations the archive did.
    seed = int(model_cfg["seeds"][0])
    n_simulations = int(model_cfg["n_simulations"])

    rows: list[dict] = []
    for _, row in trials.iterrows():
        for repeat in range(a.repeat):
            started = time.time()
            trial = trial_shim(row, study_dir, family)
            rollout_model, data_best = refit_best_trial(
                SimpleNamespace(best_trial=trial), data, family,
                device=a.device, checkpoint_dir=str(out.parent / "refit_checkpoints"),
                verbose=False)
            forecast = forecaster(rollout_model, data_best, n_simulations=n_simulations,
                                  seed=seed, device=a.device, return_simulations=False)
            rows.append({
                "suite": suite.name, "trial": trial.number, "refit": repeat,
                "val_loss": trial.value,
                "best_epoch": row.get("user_attrs_best_epoch"),
                **compute_forecast_metrics(forecast["actual"], forecast["prediction_mean"]),
                # Whether the run ranks customers at all, the metric the arm tables lead
                # with; it moves far less under the unseeded refit than bias does.
                "spearman": spearman(forecast["prediction_mean"].sum(axis=1),
                                     forecast["actual"].sum(axis=1)),
                "seconds": round(time.time() - started, 1),
            })
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    return rows


if __name__ == "__main__":
    main()
