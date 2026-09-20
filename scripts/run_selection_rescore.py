"""Does a rollout over the validation window pick better trials than cross-entropy does?

`docs/training-budget.md` §13.2 measures that one-step teacher-forced cross-entropy on a
temporal validation window barely moves after the first epoch, while the forecast keeps
improving for another 200. `docs/benchmarks-real-panels.md` measures the same thing across
trials: a study's winning validation loss does not predict its holdout bias. Both point at
the selection criterion rather than the split, and §13.4 names the cheapest decisive test.

This is that test. For every trial of a study it computes two things:

  the CANDIDATE criteria — a Monte Carlo rollout over the VALIDATION window, scored with
      `compute_forecast_metrics`, which is what selection could use instead of val CE;
  the TARGET             — the ADR-0008 refit and a rollout over the HOLDOUT, which is
      what the thesis reports.

Then the question is a rank correlation, per study: does the candidate order trials the
way the holdout does, better than val CE orders them? One row per trial; the analysis is
`.scratch/training-budget/selection_analysis.py`.

**The validation rollout is leak-free and is built the way the real forecast is.** The
calibration tensor is cut at `val_start_idx`: periods before it are the warm-up the
simulator conditions on, periods after it are a pseudo-holdout the model must reach by
sampling its own counts forward. No validation period is ever an input.

**Why this cannot read the archive.** Every family-T suite ran with
`keep_only_best_checkpoint=True`, so 99 of each study's 100 trials have no weights left.
This runner therefore trains its own studies with the cleanup off, rescores them on the
spot, and deletes the checkpoints — the boxes return CSVs, not gigabytes.

Usage:
    python scripts/run_selection_rescore.py --preflight          # 3 trials, costs nothing
    python scripts/run_selection_rescore.py --worker 3/8         # a rented box's slice
    python scripts/run_selection_rescore.py --check-complete
    python scripts/run_selection_rescore.py --report
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from panelclv.models import compute_forecast_metrics
from panelclv.registry import build_model, rollout_for
from panelclv.studies import ModelSpec, StudySuiteConfig, run_study_suite
from panelclv.trials import refit_best_trial

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_real_panel_benchmarks import spearman                       # noqa: E402
from run_rescore_trials import trial_shim                            # noqa: E402
from run_training_budget import (                                    # noqa: E402
    ARMS, BASE_SEED, MODELS, N_SIMULATIONS, PANEL, build_data, model_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDIES_BASE = REPO_ROOT / "Studies"

EXPERIMENT = "selection_rescore"
# The per-trial table is written INSIDE the suite it describes, because that is the one
# tree `VastAI/supervise/pull_results.sh` and `reap_finished.sh` move and verify. A file
# anywhere else on a rented box is a file that never comes home.
OUT_NAME = "selection_rescore.csv"
# The unit of analysis is the STUDY — one within-study rank correlation each — so power
# comes from studies, not from trials. 40 per model against family T's 20.
N_STUDIES = 40
ARM = "archive"                 # the status quo: the arm whose selection is in question
# Fewer paths than a reported forecast (200): this ranks trials against each other, and
# both rollouts carry the same sampling noise, so the ranking is what has to be stable.
N_PATHS = 100


def suite_name(model: str, replication: int) -> str:
    return f"{EXPERIMENT}__{model}__{PANEL}__r{replication:02d}"


def out_path(model: str, replication: int) -> Path:
    return STUDIES_BASE / suite_name(model, replication) / OUT_NAME


def work_list() -> list[tuple[str, int]]:
    """Every (model, replication), MODEL-major so a strided worker draws from both."""
    return [(m, r) for m in MODELS for r in range(N_STUDIES)]


# ---------------------------------------------------------------------------
# The validation window as a pseudo-holdout
# ---------------------------------------------------------------------------


def validation_view(data: dict) -> dict:
    """`data` re-cut so the simulator forecasts the VALIDATION window.

    The calibration tensor is split at `val_start_idx` (= s, the first validation
    period): periods `< s` become the calibration the rollout warms up on, periods `>= s`
    become the holdout it must predict. Everything else — `seq_cols`, `target_idx`,
    `embedded_cols`, the ids — is passed through untouched, so the forecaster reads this
    dict exactly as it reads a real one.

    This is the leak-free pseudo-holdout ADR-0003 described before it was retired: the
    model sees no validation period as an input, it reaches them by sampling.
    """
    s = int(data["val_start_idx"])
    calib = np.asarray(data["calibration"])                 # (N, T_CAL, F)
    view = dict(data)
    view["calibration"] = calib[:, :s, :]
    view["holdout"] = calib[:, s:, :]
    view["T_CAL"], view["T_HOLD"] = s, calib.shape[1] - s
    return view


def score_rollout(model, view: dict, family: str, seed: int, device: str) -> dict:
    """Roll this model out over `view`'s holdout and score it the one authorised way."""
    forecast = rollout_for(family)(model, view, n_simulations=N_PATHS, seed=seed,
                                   device=device, return_simulations=False)
    pred, actual = forecast["prediction_mean"], forecast["actual"]
    return {**compute_forecast_metrics(actual, pred),
            "spearman": spearman(pred.sum(axis=1), actual.sum(axis=1)),
            "pred_sd": float(pred.sum(axis=1).std())}


# ---------------------------------------------------------------------------
# One work item
# ---------------------------------------------------------------------------


def run_item(model: str, replication: int, device: str,
             spec: "ModelSpec | None" = None, limit: int | None = None) -> Path:
    """Train one study keeping every checkpoint, score all its trials, drop the weights.

    `spec` defaults to the `archive` arm's declaration — the one under investigation.
    `--preflight` passes a three-trial version of the same thing, which is why the
    parameter exists rather than the arm being read twice.
    """
    spec = spec or model_spec(model, ARM)
    name = suite_name(model, replication)
    out = out_path(model, replication)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = build_data(model)
    suite_dir = STUDIES_BASE / name

    if not (suite_dir / model / "Optuna_Studies" / "study_01").exists():
        print(f"{name}: training", flush=True)
        run_study_suite(StudySuiteConfig(
            studies_base_path=str(STUDIES_BASE), suite_name=name,
            n_studies_per_model=1, n_simulations=N_SIMULATIONS, device=device, data=data,
            models=[spec], base_seed=BASE_SEED + replication,
            overwrite=suite_dir.exists(),
            # The whole point: keep the losers' weights long enough to score them.
            keep_only_best_checkpoint=False,
        ))

    study_dir = suite_dir / model / "Optuna_Studies" / "study_01"
    family = MODELS[model]
    kept = {int(f.stem.rsplit("_", 1)[1])
            for f in (study_dir / "checkpoints" / study_dir.name).glob(f"{family}_trial_*.pth")}
    trials = pd.read_csv(study_dir / "study_01_trials.csv")
    trials = trials[(trials.state == "COMPLETE") & trials.number.isin(kept)].sort_values("value")
    if limit:
        trials = trials.head(limit)

    view = validation_view(data)
    seed = BASE_SEED + replication
    rows: list[dict] = []
    for n, (_, row) in enumerate(trials.iterrows(), start=1):
        started = time.time()
        trial = trial_shim(row, study_dir, family)

        # CANDIDATE: the trial's own weights, rolled out over the validation window.
        val_model = build_model(family, trial.params, {
            "seq_cols": data["seq_cols"], "embedded_cols": data["embedded_cols"],
            "target_col": data["target_col"], "seq_len": view["calibration"].shape[1],
        })
        val_model.load_state_dict(torch.load(trial.user_attrs["checkpoint_path"],
                                             map_location="cpu"))
        val = score_rollout(val_model.to_rollout(), view, family, seed, device)

        # TARGET: the production path — full-calibration refit, then the holdout.
        rollout_model, data_best = refit_best_trial(
            SimpleNamespace(best_trial=trial), data, family, device=device,
            checkpoint_dir=str(suite_dir / "refit_checkpoints"), verbose=False)
        hold = score_rollout(rollout_model, data_best, family, seed, device)

        rows.append({
            "suite": name, "model": model, "replication": replication,
            "trial": trial.number, "val_loss": trial.value,
            "best_epoch": row.get("user_attrs_best_epoch"),
            **{f"val_{k}": v for k, v in val.items()},
            **{f"hold_{k}": v for k, v in hold.items()},
            "seconds": round(time.time() - started, 1),
        })
        pd.DataFrame(rows).to_csv(out, index=False)
        if n % 10 == 0 or n == len(trials):
            print(f"  {name}: {n}/{len(trials)} trials", flush=True)

    # The weights have done their job. A study's checkpoints are ~40 MB and nothing
    # downstream reads them, so they go rather than filling a 20 GB box.
    shutil.rmtree(study_dir / "checkpoints", ignore_errors=True)
    shutil.rmtree(suite_dir / "refit_checkpoints", ignore_errors=True)
    return out


def run_worker(index: int, total: int) -> int:
    mine = work_list()[index - 1::total]
    print(f"worker {index}/{total}: {len(mine)} studies", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for n, (model, rep) in enumerate(mine, start=1):
        if out_path(model, rep).exists():
            print(f"[{n}/{len(mine)}] {suite_name(model, rep)}: done, skipping", flush=True)
            continue
        print(f"[{n}/{len(mine)}] {suite_name(model, rep)}", flush=True)
        run_item(model, rep, device)
    return 0


def preflight() -> int:
    """Both models, three trials each, in a temp tree. Proves the wiring, costs nothing."""
    import tempfile
    global STUDIES_BASE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_studies = STUDIES_BASE
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        STUDIES_BASE = Path(tmp) / "Studies"
        STUDIES_BASE.mkdir(parents=True)
        for model in MODELS:
            try:
                tiny = model_spec(model, ARM, {"n_epochs": 3, "patience": 2})
                tiny.n_trials = 3
                out = run_item(model, 0, device, spec=tiny, limit=3)
                d = pd.read_csv(out)
                print(f"  {model}: {len(d)} trials scored, "
                      f"val MAPE {d.val_mape_aggregate.mean():.1f}, "
                      f"holdout MAPE {d.hold_mape_aggregate.mean():.1f}")
            except Exception as exc:                # noqa: BLE001 — report them all
                print(f"  {model}: FAIL {type(exc).__name__}: {exc}")
                failures.append(model)
    STUDIES_BASE = real_studies
    if failures:
        print(f"\nfailed: {failures}. Fix before renting anything.")
        return 1
    print("\nBoth models train, roll out and score. Safe to launch.")
    return 0


def check_complete() -> int:
    missing = [suite_name(m, r) for m, r in work_list() if not out_path(m, r).exists()]
    for m in MODELS:
        have = sum(out_path(m, r).exists() for r in range(N_STUDIES))
        print(f"{m:13s} {have:2d}/{N_STUDIES}")
    for name in missing:
        print(f"  MISSING {name}")
    if missing:
        return 1
    print("complete")
    return 0


def report() -> None:
    """Per study, does each candidate order trials the way the holdout does?

    Spearman between a candidate criterion and the holdout outcome, computed WITHIN a
    study (the trials of one study are the only things ever compared against each other),
    then summarised across studies. `val_loss` is the status quo; the rest are what §13.3
    proposes to replace it with.
    """
    files = sorted(STUDIES_BASE.glob(f"{EXPERIMENT}__*/{OUT_NAME}"))
    if not files:
        print(f"no {OUT_NAME} under {STUDIES_BASE}/{EXPERIMENT}__*")
        return
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"{len(d)} trials from {d.suite.nunique()} studies\n")

    # Lower is better for a loss; negate the ones where higher is better, so every
    # candidate is read the same way: a positive correlation means it ranks correctly.
    candidates = {
        "val_loss (status quo)": lambda x: x.val_loss,
        "val rollout MAPE": lambda x: x.val_mape_aggregate,
        "val rollout |bias|": lambda x: x.val_bias_percent.abs(),
        "val rollout Spearman": lambda x: -x.val_spearman,
        "composite (MAPE+|bias|+rank)": lambda x: (
            x.val_mape_aggregate.rank() + x.val_bias_percent.abs().rank()
            + (-x.val_spearman).rank()),
    }
    targets = {"holdout MAPE": lambda x: x.hold_mape_aggregate,
               "holdout |bias|": lambda x: x.hold_bias_percent.abs(),
               "holdout Spearman": lambda x: -x.hold_spearman}

    for tname, tf in targets.items():
        print(f"### ranking against {tname}\n")
        print("| criterion | mean rho | sd | median | studies where it beats val_loss |")
        print("| --- | ---: | ---: | ---: | ---: |")
        per_study = {c: [] for c in candidates}
        for _, g in d.groupby("suite"):
            if len(g) < 10:
                continue
            for cname, cf in candidates.items():
                per_study[cname].append(cf(g).corr(tf(g), method="spearman"))
        base = np.array(per_study["val_loss (status quo)"], dtype=float)
        for cname, vals in per_study.items():
            v = np.array(vals, dtype=float)
            beats = "—" if cname.startswith("val_loss") else f"{int((v > base).sum())}/{len(v)}"
            print(f"| {cname} | {np.nanmean(v):.3f} | {np.nanstd(v, ddof=1):.3f} | "
                  f"{np.nanmedian(v):.3f} | {beats} |")
        print()


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
