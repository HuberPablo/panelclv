"""The two frozen benchmarks on every real panel in `Datasets/Dataset_clean/`.

ValendinLSTM and Pareto/NBD, each on CDNOW, electronics, gift and multichannel, under one
declaration so the four panels are read against the same budget and the same calendar.
This is the reference row a developed model is compared with on each panel.

**Inputs are the published model's, and nothing else.** ValendinLSTM reads the
transaction count and the calendar week, both as embedded categories -- the two inputs
of Valendin et al.'s model. Every other column a panel carries (electronics' Gender,
Income and high.season; multichannel's HIGH_SEASON, CHANNEL and MAIL_COV_1) is discarded
by not naming it in `PanelConfig`. Week is embedded rather than sin/cos because the
frozen benchmark refuses any non-embedded channel (ADR-0004, VastAI/known_failures.md
F11); `refuses()` checks that against the built dataset before anything is scheduled.

**Windows: two calibration years, one holdout year**, cut on the package's week grid
(ADR-0009: Valendin's `dayofyear // 7`, capped at 51), so every date below is the first
or last day of a week bucket. The second calibration year is the validation window
(ADR-0001). CDNOW is the exception: its panel spans 77 weeks, so it keeps the published
39 / 39 split with the last 8 calibration weeks as validation. Gift's panel opens on
2001 week 8, so its years are counted from there rather than from Jan 1.

**Budget.** 20 ValendinLSTM replications per panel, each a 100-trial Optuna search, the
ADR-0008 refit and a 500-path Monte Carlo forecast. Pareto/NBD is one deterministic MCMC
fit per panel: repeating it would reproduce the same numbers, so it runs once, on the
orchestrator (VastAI/Rules.md §5).

**One suite per replication.** A work item is one (panel, replication) pair holding one
study, so a box lost mid-run costs the replication it was on and nothing else -- a
multi-study suite cannot be resumed (F26). Replication r seeds the Optuna sampler and the
forecast from `BASE_SEED + r + 1`; training itself is unseeded (CLAUDE.md priority 3), so
the 20 are genuine replications and the report gives their distribution.

**The work list is panel-major**, so striding `i::N` hands every worker one replication
of every panel: at 20 workers, worker i trains replication i on all four. A lost worker
then thins every panel by one replication instead of emptying one panel.

**`--calibration 3y`** moves electronics, gift and multichannel to a three-year
calibration -- two years to fit the weights, the third as the validation window, then the
year after as holdout -- so the frozen benchmarks can be read beside the developed model's
`_cal3y` runs, which forecast that later year. CDNOW's 77-week panel has no room for it
and is left out. The model, its inputs, the week grid and the budget are unchanged: the
windows are the only thing that differs, and the suites carry a `_cal3y` tag so the `2y`
runs read back untouched. `WINDOWS_3Y` lives here and `run_real_panel_ar.py` reads it from
here, so the benchmark and the developed model cannot drift onto different windows.

**`--calibration 5y`** is electronics on the split of Valendin et al. (2022), Table 3:
260 calibration weeks (four years to fit, the fifth to validate) and the final year as
holdout. It reads the trip-level panel `scripts/build_full_panels.py` writes to
`Dataset_full_clean/`, whose cohort is the paper's, not the 829-household line-item panel
the other calibrations read, so its numbers can be set beside the paper's Table 4.

Usage:
    # gate the launch — builds every panel, trains each tiny, costs nothing
    python scripts/run_real_panel_benchmarks.py --preflight

    # the Pareto/NBD row, locally, once per panel
    python scripts/run_real_panel_benchmarks.py --pareto

    # one worker's slice (this is what a rented box runs)
    python scripts/run_real_panel_benchmarks.py --worker 3/20

    # after the run
    python scripts/run_real_panel_benchmarks.py --check-complete
    python scripts/run_real_panel_benchmarks.py --report

    # the same, on the three-year calibration
    python scripts/run_real_panel_benchmarks.py --calibration 3y --pareto
    python scripts/run_real_panel_benchmarks.py --calibration 3y --worker 3/20

    # Valendin et al.'s own electronics split, on the trip-level panel
    python scripts/run_real_panel_benchmarks.py --calibration 5y --pareto
"""

from __future__ import annotations

import argparse
import json
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
from panelclv.studies import (
    ModelSpec,
    StudySuiteConfig,
    load_model_predictions,
    run_study_suite,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDIES_BASE = REPO_ROOT / "Studies"
CLEAN = REPO_ROOT / "Datasets" / "Dataset_clean"
FULL_CLEAN = REPO_ROOT / "Datasets" / "Dataset_full_clean"

EXPERIMENT = "real_panel_benchmarks"

# --- budget ----------------------------------------------------------------------
# A suite's name does not carry these, so `check_complete` reads them back off every
# suite's config.json and refuses a tree that mixes generations (F23).
N_REPLICATIONS = 20
N_TRIALS = 100
N_SIMULATIONS = 500
BASE_SEED = 42

# The frozen architecture searches only these three (registry entry, ADR-0004), so the
# spec is empty: naming a width here would quietly unfreeze the benchmark.
TRAINING = {"n_epochs": 100, "patience": 7, "verbose": False, "loss_type": "cross_entropy"}

# --- panels ----------------------------------------------------------------------
# Window dates are week-bucket edges under `period_calendar.week_start`: a start date is
# the first day of its bucket and an end date the day before the next one begins.
# `clip_target_upper` is kept where the archived configs set it; gift and multichannel
# never exceed 4 transactions in a week, so their head is sized from the data.
WINDOWS: dict[str, dict[str, object]] = {
    "cdnow": dict(
        training_start="1997-01-01",
        validation_start="1997-08-05",   # 1997 week 31 — last 8 calibration weeks
        training_end="1997-09-29",       # end of 1997 week 38
        holdout_start="1997-09-30",      # 1997 week 39
        holdout_end="1998-06-30",        # end of 1998 week 25 — 39 holdout weeks
        clip_target_upper=4,
    ),
    "electronics": dict(
        training_start="1999-01-01",
        validation_start="2000-01-01",
        training_end="2000-12-31",
        holdout_start="2001-01-01",
        holdout_end="2001-12-31",
        clip_target_upper=6,
    ),
    "gift": dict(
        training_start="2001-02-25",     # 2001 week 8, the panel's first week
        validation_start="2002-02-25",   # 2002 week 8
        training_end="2003-02-24",       # end of 2003 week 7 — 104 calibration weeks
        holdout_start="2003-02-25",      # 2003 week 8
        holdout_end="2004-02-24",        # end of 2004 week 7 — 52 holdout weeks
    ),
    "multichannel": dict(
        training_start="2005-01-01",
        validation_start="2006-01-01",
        training_end="2006-12-31",
        holdout_start="2007-01-01",
        holdout_end="2007-12-31",
    ),
}

# Three calibration years: fit on two, validate on the third, forecast the year after.
# Every date is a week-bucket edge under `period_calendar.week_start` (ADR-0009), and the
# holdout is the year *after* the one the `2y` windows forecast. CDNOW is absent: its
# 77-week panel cannot hold three calibration years.
WINDOWS_3Y: dict[str, dict[str, object]] = {
    "electronics": dict(
        training_start="1999-01-01",
        validation_start="2001-01-01",
        training_end="2001-12-31",
        holdout_start="2002-01-01",
        holdout_end="2002-12-31",        # the panel's last year
        clip_target_upper=6,
    ),
    "gift": dict(
        training_start="2001-02-25",     # 2001 week 8, the panel's first week
        validation_start="2003-02-25",   # 2003 week 8
        training_end="2004-02-24",       # end of 2004 week 7 — 156 calibration weeks
        holdout_start="2004-02-25",      # 2004 week 8
        holdout_end="2005-02-24",        # end of 2005 week 7
    ),
    "multichannel": dict(
        training_start="2005-01-01",
        validation_start="2007-01-01",
        training_end="2007-12-31",
        holdout_start="2008-01-01",
        holdout_end="2008-12-31",
    ),
}

# Valendin et al.'s electronics split: 260 calibration weeks, the last 52 of them the
# validation window, then the final year of data. The dates are those of the panel's own
# config sidecar; the head is sized from the data (weekly counts reach 5).
WINDOWS_5Y: dict[str, dict[str, object]] = {
    "electronics": dict(
        training_start="1998-12-02",     # 1998 week 48, the panel's first week
        validation_start="2002-12-02",   # 2002 week 48
        training_end="2003-12-01",       # end of 2003 week 47 — 260 calibration weeks
        holdout_start="2003-12-02",      # 2003 week 48
        holdout_end="2004-11-30",        # the last day of data
    ),
}

# `--calibration` -> the windows it reads and the panels it covers. `run_real_panel_ar.py`
# reads this same table, so a benchmark and a developed model on the same calibration are
# scored on the same weeks by construction.
CALIBRATIONS: dict[str, dict[str, dict[str, object]]] = {
    "2y": WINDOWS, "3y": WINDOWS_3Y, "5y": WINDOWS_5Y,
}


def calibration_tag(cal: str) -> str:
    """The suffix a calibration puts on an experiment name.

    `2y` carries none, so every suite written before there was a choice reads back under
    the name it was stored with.
    """
    return "" if cal == "2y" else f"_cal{cal}"


def panel_config(panel: str, cal: str) -> PanelConfig:
    """Count and embedded week only; every other column of the panel is left unread."""
    return PanelConfig(
        id_col="Id",
        target_col="Transactions",
        frequency="weekly",
        time_cols=("year", "week"),
        time=("week",),
        embedded_cols={"Transactions": "auto", "week": "auto"},
        **CALIBRATIONS[cal][panel],
    )


def panel_path(panel: str, cal: str) -> Path:
    """The panel CSV calibration `cal` reads.

    `5y` reads the trip-level panel on the paper's cohort; every other calibration reads
    the panels the archived suites were trained on. `run_real_panel_ar.py` asks here too,
    so a developed model cannot put the 5y windows on the wrong panel.
    """
    if cal == "5y":
        return FULL_CLEAN / f"{panel}_5y_customer_week_panel.csv"
    return CLEAN / f"{panel}_customer_week_panel.csv"


def build_data(panel: str, cal: str) -> dict:
    """`prepare_dataset` for one panel, on the windows of calibration `cal`."""
    path = panel_path(panel, cal)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Datasets/ is not in git, so a rented worker needs the "
            f"panel pushed to it (VastAI/Rules.md §3)."
        )
    data = panel_dataset.prepare_dataset(pd.read_csv(path), panel_config(panel, cal),
                                         verbose=False)
    data["panel_name"] = panel
    return data


def refuses(data: dict) -> str | None:
    """Why ValendinLSTM cannot read this dataset, or None if it can.

    The same question `benchmarks/valendin_lstm.py` asks, asked of the same two lists,
    so a refusal is found here rather than on a rented box (F11).
    """
    embedded = set(data.get("embedded_cols") or {})
    covariates = [c for c in data["seq_cols"] if c not in embedded]
    if covariates:
        return f"non-embedded seq_cols {covariates} (ADR-0004)"
    return None


# ---------------------------------------------------------------------------
# Work items
# ---------------------------------------------------------------------------


def suite_name(panel: str, replication: int, cal: str) -> str:
    """`real_panel_benchmarks[_cal3y]__ValendinLSTM__<panel>__r<NN>` — one per work item."""
    return f"{EXPERIMENT}{calibration_tag(cal)}__ValendinLSTM__{panel}__r{replication:02d}"


def pareto_suite_name(panel: str, cal: str) -> str:
    return f"{EXPERIMENT}{calibration_tag(cal)}__ParetoNBD__{panel}"


def work_list(cal: str) -> list[tuple[str, int]]:
    """Every (panel, replication), PANEL-major — see the module docstring for why."""
    return [(p, r) for p in CALIBRATIONS[cal] for r in range(N_REPLICATIONS)]


def forecast_path(panel: str, replication: int, cal: str) -> Path:
    return (STUDIES_BASE / suite_name(panel, replication, cal) / "ValendinLSTM"
            / "Predictions" / "Prediction_1.csv")


def model_spec(n_trials: int, training: dict) -> ModelSpec:
    return ModelSpec(name="ValendinLSTM", model_type="valendin_lstm",
                     n_trials=n_trials, search_space={}, training=dict(training))


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_worker(index: int, total: int, cal: str) -> int:
    """Train this worker's stride of the work list. Finished items are skipped.

    An item counts as finished when its forecast exists. A folder without one is a
    replication cut short, and a suite cannot be resumed, so it is overwritten.
    """
    mine = work_list(cal)[index - 1::total]
    print(f"worker {index}/{total}: {len(mine)} suites", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    STUDIES_BASE.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}

    for n, (panel, rep) in enumerate(mine, start=1):
        name = suite_name(panel, rep, cal)
        if forecast_path(panel, rep, cal).exists():
            print(f"[{n}/{len(mine)}] {name}: done, skipping", flush=True)
            continue
        if panel not in cache:
            cache[panel] = build_data(panel, cal)
            reason = refuses(cache[panel])
            if reason:
                raise RuntimeError(f"ValendinLSTM refuses {panel}: {reason}")
        print(f"[{n}/{len(mine)}] {name}: training", flush=True)
        run_study_suite(StudySuiteConfig(
            studies_base_path=str(STUDIES_BASE),
            suite_name=name,
            n_studies_per_model=1,
            n_simulations=N_SIMULATIONS,
            device=device,
            data=cache[panel],
            models=[model_spec(N_TRIALS, TRAINING)],
            base_seed=BASE_SEED + rep,
            overwrite=(STUDIES_BASE / name).exists(),
            keep_only_best_checkpoint=True,
        ))
    return 0


def run_pareto(cal: str) -> int:
    """One Pareto/NBD fit per panel, skipping panels already fitted."""
    STUDIES_BASE.mkdir(parents=True, exist_ok=True)
    for panel in CALIBRATIONS[cal]:
        name = pareto_suite_name(panel, cal)
        if (STUDIES_BASE / name / "ParetoNBD" / "Predictions").is_dir():
            print(f"{name}: done, skipping")
            continue
        root = run_study_suite(StudySuiteConfig(
            studies_base_path=str(STUDIES_BASE),
            suite_name=name,
            n_studies_per_model=1,
            n_simulations=N_SIMULATIONS,
            device="cpu",
            data=build_data(panel, cal),
            models=[ModelSpec(name="ParetoNBD", model_type="pareto_nbd")],
            base_seed=BASE_SEED,
            overwrite=(STUDIES_BASE / name).exists(),
        ))
        print(f"Pareto/NBD written to {root}")
    return 0


def preflight(cal: str) -> int:
    """Build every panel, check eligibility, and train each one tiny in a temp dir."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for panel in CALIBRATIONS[cal]:
            data = build_data(panel, cal)
            print(f"{panel:12s} N={len(data['ids']):5d} T_CAL={int(data['T_CAL']):3d} "
                  f"T_HOLD={int(data['T_HOLD']):3d} seq_cols={data['seq_cols']}")
            reason = refuses(data)
            if reason:
                print(f"  REFUSED: {reason}")
                failures.append(panel)
                continue
            try:
                run_study_suite(StudySuiteConfig(
                    studies_base_path=tmp, suite_name=f"preflight__{cal}__{panel}",
                    n_studies_per_model=1, n_simulations=2, device=device, data=data,
                    models=[model_spec(1, {**TRAINING, "n_epochs": 3, "patience": 2})],
                    base_seed=BASE_SEED, keep_only_best_checkpoint=True,
                ))
                print("  ok")
            except Exception as exc:                    # noqa: BLE001 — report them all
                print(f"  FAIL {type(exc).__name__}: {exc}")
                failures.append(panel)
    if failures:
        print(f"\n{len(failures)} panel(s) failed: {failures}. Fix before renting anything.")
        return 1
    print("\nAll panels build and train. Safe to launch.")
    return 0


# ---------------------------------------------------------------------------
# Completeness and scoring
# ---------------------------------------------------------------------------


def check_complete(cal: str) -> int:
    """What the declaration owes against what is on disk. Non-zero on any shortfall.

    Also reads every suite's recorded budget: a suite name does not carry the trial or
    path count, so a tree holding two generations looks complete and is not (F23).
    """
    missing: list[str] = []
    wrong_budget: list[str] = []
    for panel in CALIBRATIONS[cal]:
        have = 0
        for rep in range(N_REPLICATIONS):
            if not forecast_path(panel, rep, cal).exists():
                missing.append(suite_name(panel, rep, cal))
                continue
            have += 1
            cfg = json.loads(
                (STUDIES_BASE / suite_name(panel, rep, cal) / "config.json").read_text())
            budget = (cfg["n_simulations"], cfg["models"][0]["n_trials"])
            if budget != (N_SIMULATIONS, N_TRIALS):
                wrong_budget.append(f"{suite_name(panel, rep)} {budget}")
        pareto = (STUDIES_BASE / pareto_suite_name(panel, cal) / "ParetoNBD"
                  / "Predictions").is_dir()
        print(f"{panel:12s} ValendinLSTM {have:2d}/{N_REPLICATIONS}   "
              f"ParetoNBD {'1/1' if pareto else '0/1'}")
        if not pareto:
            missing.append(pareto_suite_name(panel, cal))
    for name in missing:
        print(f"  MISSING      {name}")
    for line in wrong_budget:
        print(f"  WRONG BUDGET {line}  (expected ({N_SIMULATIONS}, {N_TRIALS}))")
    if missing or wrong_budget:
        return 1
    print("complete")
    return 0


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, as Pearson on average-tied ranks (no scipy import)."""
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def score(model_dir: Path, actual: np.ndarray, ref_ids: np.ndarray) -> dict[str, float]:
    """Metrics for one stored forecast, via the single scoring authority."""
    values, ids = load_model_predictions(model_dir, study=1)
    if ids is not None and not np.array_equal(np.asarray(ids), ref_ids):
        raise ValueError(f"{model_dir}: prediction ids do not match the rebuilt cohort")
    return {**compute_forecast_metrics(actual, values),
            "spearman": spearman(values.sum(axis=1), actual.sum(axis=1))}


# `rmse_customer_total` is the "individual-level RMSE" of Valendin et al., Table 4.
METRICS = ("bias_percent", "mape_aggregate", "rmse", "rmse_customer_total", "spearman")


def report(cal: str) -> None:
    """Print one markdown table per panel: ValendinLSTM's distribution, Pareto/NBD, zero."""
    for panel in CALIBRATIONS[cal]:
        data = build_data(panel, cal)
        actual = holdout_actuals(data)                          # (N, T_HOLD)
        ref_ids = np.asarray(data["ids"])

        rows = [
            {"replication": rep,
             **score(forecast_path(panel, rep, cal).parents[1], actual, ref_ids)}
            for rep in range(N_REPLICATIONS) if forecast_path(panel, rep, cal).exists()
        ]
        pareto_dir = STUDIES_BASE / pareto_suite_name(panel, cal) / "ParetoNBD"
        pareto = score(pareto_dir, actual, ref_ids) if (pareto_dir / "Predictions").is_dir() else None
        # The all-zero forecast: the panels are mostly zeros, so RMSE is only readable
        # beside it, and it is what bias and Spearman exclude (-100% bias, no ranking).
        zero = compute_forecast_metrics(actual, np.zeros_like(actual, dtype=float))

        print(f"\n### {panel}\n")
        print(f"N = {len(ref_ids)}, T_CAL = {int(data['T_CAL'])}, T_HOLD = {int(data['T_HOLD'])}, "
              f"holdout transactions = {int(actual.sum())}, "
              f"zero cells = {float((actual == 0).mean()):.1%}. "
              f"ValendinLSTM replications: {len(rows)}/{N_REPLICATIONS}.\n")
        print("| metric | Valendin mean | sd | median | IQR | min | max | Pareto/NBD | all-zero |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        df = pd.DataFrame(rows)
        for m in METRICS:
            s = df[m] if len(df) else pd.Series(dtype=float)
            p = f"{pareto[m]:.4f}" if pareto else "—"
            z = f"{zero[m]:.4f}" if m in zero else "—"
            print(f"| {m} | {s.mean():.4f} | {s.std(ddof=1):.4f} | {s.median():.4f} | "
                  f"{s.quantile(.75) - s.quantile(.25):.4f} | {s.min():.4f} | {s.max():.4f} | "
                  f"{p} | {z} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", choices=sorted(CALIBRATIONS), default="2y",
                        help="2y: the published windows; 3y: fit two years, validate one; "
                             "5y: the paper's electronics split")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker", metavar="I/N", help="train this worker's stride")
    mode.add_argument("--pareto", action="store_true", help="fit Pareto/NBD on every panel")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--check-complete", action="store_true")
    mode.add_argument("--report", action="store_true")
    args = parser.parse_args()

    cal = args.calibration
    if args.worker:
        index, total = (int(x) for x in args.worker.split("/"))
        sys.exit(run_worker(index, total, cal))
    if args.pareto:
        sys.exit(run_pareto(cal))
    if args.preflight:
        sys.exit(preflight(cal))
    if args.check_complete:
        sys.exit(check_complete(cal))
    report(cal)


if __name__ == "__main__":
    main()
