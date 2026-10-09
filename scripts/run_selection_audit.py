"""Does the search pick the best trial, and does one setting forecast alike when retrained?

Two questions of the model-selection protocol, answered from the same searches:

1. **Does the search pick well?** Every trial of a study is a candidate the search could
   have kept. The winner and a random sample of the other completed trials are each
   forecast on the holdout, so their places can be compared: the winner's rank among them,
   and the rank correlation of validation loss with holdout error, per study.
2. **Does the winner forecast alike when retrained?** The winner's settings are trained
   again from scratch `N_RETRAINS` times, unchanged, and each retrain is forecast. The
   spread across them is what one setting is worth on its own.

**Every forecast is made twice, with and without the refit** (ADR-0008, ADR-0011): from
the trial's checkpoint as it stands, and from a refit of it on the full calibration
window. Both use the study's own forecast seed, so the two are paired on the same Monte
Carlo draws and differ in the refit alone. Without the refit, question 1 is asked of what
the search actually selected; with it, of what the archive reports.

**The searches are the stopping-rule run's `patience7` arm** (`run_stopping_rule.py`): 100
TPE trials, patience 7, at most 200 epochs, the default pruner, the corrected validation
loss (ADR-0010) and the archive's search space. ValendinLSTM's winners can therefore be set
beside that run's 20 replications per panel. What differs is only that every checkpoint is
kept until it has been scored.

**Two models.**
- `ValendinLSTM`: count and week, both embedded; frozen architecture, so its search
  covers learning rate, weight decay and batch size only (ADR-0004).
- `LSTM_AR52`: count (embedded), week sin/cos and the bounded activity flags
  `active_in_last_{2,...,52}_periods` + `has_transacted_before`. It searches the same
  training settings plus its own widths and dropout (registry ranges), so it is the model
  where the search chooses an architecture. CDNOW calibrates on 39 weeks, so a 52-week
  flag cannot vary there (`run_real_panel_ar.check_arm_depth`); CDNOW stops at 32.

**Panels**: every calibration of `run_real_panel_benchmarks` — `2y` (CDNOW, electronics,
gift, multichannel), `3y` (electronics, gift, multichannel), `5y` (the paper's electronics
cohort, which needs a 16 GB GPU for the holdout warm-up).

**Training is unseeded, and this runner keeps it so.** `forecast_recurrent` calls
`torch.manual_seed` before sampling (`.scratch/training-budget/issues/05`), so every
refit or retrain that followed a forecast would start from the same RNG state: the same
initial weights and batch order every time, and a spread of zero by construction. The
global RNG is therefore reseeded from OS entropy (`torch.seed()`) before each refit and
each retrain.

**Output**: `selection_audit.csv` inside each suite, one row per scored model — `kind` is
`winner`, `trial` or `retrain` — with its settings, validation loss, kept epoch and both
forecasts' scores. It is written under a temporary name and renamed when the study is
complete, so its presence marks a finished work item. The suite's own `Prediction_1.csv`
is the production forecast (winner, refit). The checkpoints are deleted afterwards.

Usage:
    python scripts/run_selection_audit.py --preflight --calibration 2y,3y,5y
    python scripts/run_selection_audit.py --worker 3/40 --calibration 2y,3y
    python scripts/run_selection_audit.py --check-complete --calibration 2y,3y,5y
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import optuna
import pandas as pd
import torch

from panelclv.configs.panel_config import PanelConfig
from panelclv.data_preparation import panel_dataset
from panelclv.models import compute_forecast_metrics
from panelclv.registry import MODEL_REGISTRY, rollout_for
from panelclv.studies import ModelSpec, StudySuiteConfig, run_study_suite
from panelclv.trials import load_best_trial, make_data_builder, refit_best_trial
from panelclv.tuning import run_optuna_study

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_real_panel_benchmarks as benchmarks                              # noqa: E402
from run_real_panel_ar import check_arm_depth                               # noqa: E402
from run_real_panel_arms import bounded_flags                               # noqa: E402
from run_rescore_trials import INT_PARAMS, trial_shim                       # noqa: E402
from run_stopping_rule import ARMS, SEARCH_SPACE                            # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDIES_BASE = REPO_ROOT / "Studies"
CALIBRATIONS = benchmarks.CALIBRATIONS

EXPERIMENT = "selection_audit"
OUT_NAME = "selection_audit.csv"
TODO_NAME = "selection_audit_todo.txt"   # optional, see run_worker
N_REPLICATIONS = 20
N_TRIALS = 100
N_SIMULATIONS = 500
BASE_SEED = 42
# Completed non-winning trials scored per study, drawn at random. Random rather than the
# best by validation loss: choosing them on the criterion under test would restrict its
# range and deflate its correlation with the holdout (`run_selection_rescore.py`).
# 15 and 4 rather than 20 and 5 fit the run in its budget (8 October, $35 of credit):
# each scored model costs two 500-path rollouts, ~90 s on a 2y panel and ~300 s on 5y.
# Power comes from the 20 replications, which are kept; these only sharpen each study.
TRIALS_PER_STUDY = 15
# From-scratch retrains of the winner's settings per study.
N_RETRAINS = 4
TRAINING = dict(ARMS["patience7"])

# Deepest activity flag per panel: 52 weeks, except where the calibration is too short
# for a 52-week silence to occur while fitting.
AR_DEPTH = {"cdnow": 32}
MODELS = {"ValendinLSTM": "valendin_lstm", "LSTM_AR52": "lstm"}
CALS = ["2y"]                   # the calibrations this invocation covers


def panel_config(model: str, panel: str, cal: str) -> PanelConfig:
    windows = CALIBRATIONS[cal][panel]
    if model == "ValendinLSTM":
        return benchmarks.panel_config(panel, cal)
    return PanelConfig(
        id_col="Id", target_col="Transactions", frequency="weekly",
        time_cols=("year", "week"), time_features={"add_week_sin_cos": True},
        ar_features=bounded_flags(AR_DEPTH.get(panel, 52)),
        embedded_cols={"Transactions": "auto"}, **windows,
    )


def build_data(model: str, panel: str, cal: str) -> dict:
    data = panel_dataset.prepare_dataset(pd.read_csv(benchmarks.panel_path(panel, cal)),
                                         panel_config(model, panel, cal), verbose=False)
    data["panel_name"] = panel
    check_arm_depth(data)
    return data


def spec(model: str, n_trials: int = N_TRIALS, training: dict | None = None) -> ModelSpec:
    # The archive's training space for both models; the LSTM's widths and dropout keep
    # their registry ranges, so the architecture is the one thing only it searches.
    return ModelSpec(name=model, model_type=MODELS[model], n_trials=n_trials,
                     search_space=dict(SEARCH_SPACE), training=training or dict(TRAINING))


def suite_name(cal: str, panel: str, model: str, rep: int) -> str:
    return f"{EXPERIMENT}{benchmarks.calibration_tag(cal)}__{model}__{panel}__r{rep:02d}"


def out_path(cal: str, panel: str, model: str, rep: int) -> Path:
    return STUDIES_BASE / suite_name(cal, panel, model, rep) / OUT_NAME


def work_list() -> list[tuple[str, str, str, int]]:
    """Replication-major, so a strided worker draws from every panel and both models."""
    return [(c, p, m, r) for r in range(N_REPLICATIONS) for m in MODELS
            for c in CALS for p in sorted(CALIBRATIONS[c])]


# ---------------------------------------------------------------------------
# Scoring one model both ways
# ---------------------------------------------------------------------------


def forecast_scores(rollout_model, data_best: dict, family: str, seed: int, device: str,
                    n_paths: int) -> dict:
    """Roll out over the holdout and score through the one scoring authority."""
    forecast = rollout_for(family)(rollout_model, data_best, n_simulations=n_paths,
                                   seed=seed, device=device, return_simulations=False)
    pred, actual = forecast["prediction_mean"], forecast["actual"]
    return {**compute_forecast_metrics(actual, pred),
            "spearman": benchmarks.spearman(pred.sum(axis=1), actual.sum(axis=1))}


def score_both_ways(trial, data: dict, family: str, seed: int, device: str,
                    refit_dir: Path, n_paths: int) -> dict:
    """The trial's checkpoint forecast as it stands, then after a refit; same seed."""
    study = SimpleNamespace(best_trial=trial)       # both routes read only `best_trial`
    noref = forecast_scores(*load_best_trial(study, data, family), family, seed, device,
                            n_paths)
    torch.seed()                                    # unseeded refit; see module docstring
    refit = forecast_scores(*refit_best_trial(study, data, family, device=device,
                                              checkpoint_dir=str(refit_dir), verbose=False),
                            family, seed, device, n_paths)
    return {**{f"noref_{k}": v for k, v in noref.items()},
            **{f"refit_{k}": v for k, v in refit.items()}}


def retrain(winner, data: dict, model: str, seed: int, device: str, workdir: Path,
            training: dict):
    """Train the winner's settings again from scratch: one pinned trial, unseeded."""
    torch.seed()
    # Every searched setting pinned to the winner's value. The trials table also lists
    # the training controls (`n_epochs`, `patience`, ...), which come from `training`
    # instead; a parameter the winner never sampled (NaN) keeps its default, as it did.
    searched = MODEL_REGISTRY[MODELS[model]].search_space
    pinned = {k: (int(v) if k in INT_PARAMS else v) for k, v in winner.params.items()
              if k in searched and not pd.isna(v)}
    study = run_optuna_study(
        model_type=MODELS[model], data_builder=make_data_builder(data),
        search_space=pinned,
        training={**training, "seed": seed, "checkpoint_dir": str(workdir / "checkpoints")},
        n_trials=1, device=device, study_name="retrain", append_timestamp=False,
        summary_dir=workdir, sampler=optuna.samplers.TPESampler(seed=seed))
    return study.best_trial


# ---------------------------------------------------------------------------
# One work item
# ---------------------------------------------------------------------------


def run_item(cal: str, panel: str, model: str, rep: int, data: dict, device: str,
             model_spec: ModelSpec | None = None, n_scored: int = TRIALS_PER_STUDY,
             n_retrains: int = N_RETRAINS, n_paths: int = N_SIMULATIONS) -> Path:
    """Search keeping every checkpoint, score winner + sample + retrains, drop weights."""
    model_spec = model_spec or spec(model)
    name = suite_name(cal, panel, model, rep)
    suite_dir = STUDIES_BASE / name
    family = MODELS[model]
    # Study i of a suite is seeded base_seed + i; these suites hold one study.
    seed = BASE_SEED + rep + 1

    print(f"{name}: searching", flush=True)
    run_study_suite(StudySuiteConfig(
        studies_base_path=str(STUDIES_BASE), suite_name=name, n_studies_per_model=1,
        n_simulations=n_paths, device=device, data=data, models=[model_spec],
        base_seed=BASE_SEED + rep, overwrite=suite_dir.exists(),
        # The whole point: keep the losers' weights long enough to score them.
        keep_only_best_checkpoint=False))

    study_dir = suite_dir / model / "Optuna_Studies" / "study_01"
    trials = pd.read_csv(study_dir / "study_01_trials.csv")
    complete = trials[trials.state == "COMPLETE"]
    winner_row = complete.loc[complete.value.idxmin()]
    others = complete[complete.number != winner_row.number]
    # Seeded by the replication, so a lost item re-run scores the same trials.
    others = others.sample(n=min(n_scored, len(others)), random_state=BASE_SEED + rep)

    meta = {"suite": name, "calibration": cal, "panel": panel, "model": model,
            "replication": rep, "seed": seed, "n_complete": len(complete),
            "n_pruned": int((trials.state == "PRUNED").sum())}
    partial = suite_dir / f"{OUT_NAME}.partial"
    rows: list[dict] = []

    def record(kind: str, trial, best_epoch) -> None:
        started = time.time()
        scores = score_both_ways(trial, data, family, seed, device,
                                 suite_dir / "refit_checkpoints", n_paths)
        rows.append({**meta, "kind": kind, "trial": trial.number, "val_loss": trial.value,
                     "best_epoch": best_epoch,
                     **{f"param_{k}": v for k, v in trial.params.items()},
                     **scores, "seconds": round(time.time() - started, 1)})
        pd.DataFrame(rows).to_csv(partial, index=False)

    for kind, rows_ in (("winner", [winner_row]), ("trial", [r for _, r in others.iterrows()])):
        for row in rows_:
            record(kind, trial_shim(row, study_dir, family), row.get("user_attrs_best_epoch"))
    print(f"  {name}: winner + {len(others)} trials scored", flush=True)

    winner = trial_shim(winner_row, study_dir, family)
    for k in range(n_retrains):
        workdir = suite_dir / "retrains" / f"retrain_{k:02d}"
        trial = retrain(winner, data, model, seed, device, workdir, model_spec.training)
        record("retrain", trial, trial.user_attrs.get("best_epoch"))
    print(f"  {name}: {n_retrains} retrains scored", flush=True)

    partial.rename(suite_dir / OUT_NAME)
    # The weights have done their job; a box's disk is not an archive.
    for d in (study_dir / "checkpoints", suite_dir / "refit_checkpoints",
              study_dir / "refit_checkpoints", suite_dir / "retrains"):
        shutil.rmtree(d, ignore_errors=True)
    return suite_dir / OUT_NAME


def run_worker(index: int, total: int) -> int:
    work = work_list()
    # A rerun that fills gaps ships a frozen list of the missing suites in its seed, and
    # the workers stride over that list rather than the whole one: the gaps a stopped
    # fleet leaves are clustered, so striding the whole list would leave some boxes idle
    # and hand others a dozen studies. Frozen, so every box (replacements included)
    # agrees on who runs what. It sits at the repo root, not under Studies/: the
    # orchestrator pulls every box's Studies/ tree back, and a list kept there was
    # overwritten by whichever box was pulled last.
    todo = REPO_ROOT / TODO_NAME
    if todo.exists():
        wanted = set(todo.read_text().split())
        work = [w for w in work if suite_name(*w) in wanted]
    mine = work[index - 1::total]
    print(f"worker {index}/{total}: {len(mine)} studies", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    STUDIES_BASE.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple, dict] = {}
    for n, (cal, panel, model, rep) in enumerate(mine, start=1):
        if out_path(cal, panel, model, rep).exists():
            print(f"[{n}/{len(mine)}] {suite_name(cal, panel, model, rep)}: done, skipping",
                  flush=True)
            continue
        if (model, cal, panel) not in cache:
            cache.clear()               # one panel at a time; the 5y panel is large
            cache[(model, cal, panel)] = build_data(model, panel, cal)
        print(f"[{n}/{len(mine)}]", end=" ", flush=True)
        run_item(cal, panel, model, rep, cache[(model, cal, panel)], device)
    return 0


def preflight() -> int:
    """Every panel and model end to end: 3 trials of 4 epochs, 2 scored, 1 retrain."""
    global STUDIES_BASE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real, failures = STUDIES_BASE, []
    with tempfile.TemporaryDirectory() as tmp:
        STUDIES_BASE = Path(tmp)
        for cal in CALS:
            for panel in sorted(CALIBRATIONS[cal]):
                for model in MODELS:
                    try:
                        data = build_data(model, panel, cal)
                        tiny = spec(model, 3, {**TRAINING, "n_epochs": 4})
                        out = run_item(cal, panel, model, 0, data, device, model_spec=tiny,
                                       n_scored=2, n_retrains=1, n_paths=2)
                        d = pd.read_csv(out)
                        assert list(d.kind) == ["winner", "trial", "trial", "retrain"], list(d.kind)
                        assert d.filter(like="_mape_aggregate").notna().all().all()
                        print(f"  {cal} {panel:13s} {model:13s} ok  "
                              f"seq_cols={data['seq_cols']}")
                    except Exception as exc:            # noqa: BLE001 — report them all
                        print(f"  {cal} {panel:13s} {model:13s} FAIL "
                              f"{type(exc).__name__}: {exc}")
                        failures.append(f"{cal}/{panel}/{model}")
    STUDIES_BASE = real
    if failures:
        print(f"\nfailed: {failures}. Fix before renting anything.")
        return 1
    print("\nEvery panel and model searches, scores both ways and retrains. Safe to launch.")
    return 0


def check_complete() -> int:
    missing = [w for w in work_list() if not out_path(*w).exists()]
    for cal in CALS:
        for panel in sorted(CALIBRATIONS[cal]):
            for model in MODELS:
                have = sum(out_path(cal, panel, model, r).exists()
                           for r in range(N_REPLICATIONS))
                print(f"{cal} {panel:13s} {model:13s} {have:2d}/{N_REPLICATIONS}")
    if missing:
        print(f"{len(missing)} missing")
        return 1
    print("complete")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker", metavar="I/N")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--check-complete", action="store_true")
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
    sys.exit(check_complete())


if __name__ == "__main__":
    main()
