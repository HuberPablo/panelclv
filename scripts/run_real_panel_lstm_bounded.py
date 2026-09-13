"""The LSTM with bounded activity flags on every real panel, at the benchmark's windows.

The developed model's configuration that won the AR-encoding ablation on electronics,
run on CDNOW, electronics, gift and multichannel so it can be read against the frozen
benchmarks (`scripts/run_real_panel_benchmarks.py`) panel by panel.

**Inputs.** The transaction count (embedded), `week_sin` / `week_cos`, and the
`ar_bounded_32` set: nested flags `active_in_last_{2,4,8,16,32}_periods` plus
`has_transacted_before`. Every other panel column is discarded. The flags are a bounded
encoding of recency — all zero past the deepest bin, so no holdout value leaves the range
the weights were fitted on (docs/feature_engineering.md §4) — and they are what lifts the
count-only models out of the forecast collapse on long sparse panels
(docs/benchmarks-real-panels.md).

**Depth 32 on every panel, CDNOW included, by decision.** CDNOW calibrates on 39 weeks,
so a 32-week silence is rare while fitting; the AR-encoding ablation ran CDNOW at 16 for
that reason. `check_arm_depth` still refuses a depth at or above T_CAL, where the deepest
flag would be an exact copy of `has_transacted_before`.

**Windows, cohort and scoring are the benchmark's**, imported from its runner rather than
restated, so the two runs cannot drift apart and `--report` can put ValendinLSTM and
Pareto/NBD beside this model on the same customers.

**Budget.** 100 replications per panel, each a 100-trial Optuna search, the ADR-0008 refit
and a 500-path Monte Carlo forecast. One suite per replication, panel-major, so at 20
workers worker i trains replications `i-1, i+19, ...` — five of every panel.

Usage:
    python scripts/run_real_panel_lstm_bounded.py --preflight
    python scripts/run_real_panel_lstm_bounded.py --worker 3/20      # on a rented box
    python scripts/run_real_panel_lstm_bounded.py --check-complete
    python scripts/run_real_panel_lstm_bounded.py --report
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
from panelclv.studies import ModelSpec, StudySuiteConfig, run_study_suite

import run_real_panel_benchmarks as benchmarks

STUDIES_BASE = benchmarks.STUDIES_BASE
PANELS = benchmarks.PANELS

EXPERIMENT = "real_panel_lstm_bounded32"

# --- budget ----------------------------------------------------------------------
# Suite names do not carry these; `check_complete` reads them back off config.json (F23).
N_REPLICATIONS = 100
N_TRIALS = 100
N_SIMULATIONS = 500
BASE_SEED = 42

DEEPEST_FLAG = 32
AR_FEATURES = tuple(
    f"active_in_last_{k}_periods" for k in (2, 4, 8, 16, 32) if k <= DEEPEST_FLAG
) + ("has_transacted_before",)

# The LSTM space of the AR-encoding ablation and the real-panel arms. `embedding_dim` is
# absent: the default `valendin` embedder has no common width to search.
SEARCH_SPACE: dict[str, object] = {
    "lstm_hidden_size": {32, 64, 128},
    "dense_units":      {32, 64, 128},
    "dropout":          {0.0, 0.2},
    "learning_rate":    (1e-4, 1e-2, "log"),
    "weight_decay":     (1e-6, 1e-2, "log"),
    "batch_size":       {64, 128, 256},
}
TRAINING = {"n_epochs": 100, "patience": 7, "verbose": False, "loss_type": "cross_entropy"}


def panel_config(panel: str) -> PanelConfig:
    """Count, sin/cos week and the bounded flags, on the benchmark's windows."""
    return PanelConfig(
        id_col="Id",
        target_col="Transactions",
        frequency="weekly",
        time_cols=("year", "week"),
        time_features={"add_week_sin_cos": True},
        ar_features=AR_FEATURES,
        embedded_cols={"Transactions": "auto"},
        **benchmarks.WINDOWS[panel],
    )


def check_arm_depth(data: dict) -> None:
    """Refuse an `active_in_last_K` flag that cannot vary on this panel.

    Silence cannot exceed `T_CAL - 1` while fitting, so a flag with K >= T_CAL is 1 for
    every customer who has ever transacted — a copy of `has_transacted_before` — and only
    becomes a distinct signal in the holdout, where nothing constrained it.
    """
    t_cal = int(data["T_CAL"])
    bad = [c for c in data["seq_cols"]
           if c.startswith("active_in_last_") and int(c.split("_")[3]) >= t_cal]
    if bad:
        raise ValueError(f"{data.get('panel_name')}: {bad} cannot vary with T_CAL={t_cal}")


def build_data(panel: str) -> dict:
    path = benchmarks.CLEAN / f"{panel}_customer_week_panel.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — push the panels to this box (Rules.md §3)")
    data = panel_dataset.prepare_dataset(pd.read_csv(path), panel_config(panel), verbose=False)
    data["panel_name"] = panel
    check_arm_depth(data)
    return data


def suite_name(panel: str, replication: int) -> str:
    return f"{EXPERIMENT}__LSTM__{panel}__r{replication:03d}"


def work_list() -> list[tuple[str, int]]:
    """Every (panel, replication), PANEL-major, so a worker's stride spans all panels."""
    return [(p, r) for p in PANELS for r in range(N_REPLICATIONS)]


def forecast_path(panel: str, replication: int) -> Path:
    return STUDIES_BASE / suite_name(panel, replication) / "LSTM" / "Predictions" / "Prediction_1.csv"


def model_spec(n_trials: int, training: dict) -> ModelSpec:
    return ModelSpec(name="LSTM", model_type="lstm", n_trials=n_trials,
                     search_space=dict(SEARCH_SPACE), training=dict(training))


def run_worker(index: int, total: int) -> int:
    """Train this worker's stride; a finished replication is skipped, a cut one redone."""
    mine = work_list()[index - 1::total]
    print(f"worker {index}/{total}: {len(mine)} suites", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    STUDIES_BASE.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict] = {}
    for n, (panel, rep) in enumerate(mine, start=1):
        name = suite_name(panel, rep)
        if forecast_path(panel, rep).exists():
            print(f"[{n}/{len(mine)}] {name}: done, skipping", flush=True)
            continue
        if panel not in cache:
            cache[panel] = build_data(panel)
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


def preflight() -> int:
    """Build every panel and train each one tiny in a temporary directory."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for panel in PANELS:
            try:
                data = build_data(panel)
                print(f"{panel:12s} N={len(data['ids']):5d} T_CAL={int(data['T_CAL']):3d} "
                      f"T_HOLD={int(data['T_HOLD']):3d} seq_cols={data['seq_cols']}")
                run_study_suite(StudySuiteConfig(
                    studies_base_path=tmp, suite_name=f"preflight__{panel}",
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


def check_complete() -> int:
    """What the declaration owes against what is on disk, budget included (F23)."""
    missing, wrong = [], []
    for panel in PANELS:
        have = 0
        for rep in range(N_REPLICATIONS):
            if not forecast_path(panel, rep).exists():
                missing.append(suite_name(panel, rep))
                continue
            have += 1
            cfg = json.loads((STUDIES_BASE / suite_name(panel, rep) / "config.json").read_text())
            budget = (cfg["n_simulations"], cfg["models"][0]["n_trials"])
            if budget != (N_SIMULATIONS, N_TRIALS):
                wrong.append(f"{suite_name(panel, rep)} {budget}")
        print(f"{panel:12s} LSTM {have:3d}/{N_REPLICATIONS}")
    for name in missing[:20]:
        print(f"  MISSING      {name}")
    if len(missing) > 20:
        print(f"  ... and {len(missing) - 20} more")
    for line in wrong:
        print(f"  WRONG BUDGET {line}  (expected ({N_SIMULATIONS}, {N_TRIALS}))")
    if missing or wrong:
        return 1
    print("complete")
    return 0


METRICS = ("bias_percent", "mape_aggregate", "rmse", "spearman")


def report() -> None:
    """One markdown table per panel: this LSTM, the two benchmarks, and the zero forecast."""
    for panel in PANELS:
        data = build_data(panel)
        actual = holdout_actuals(data)
        ref_ids = np.asarray(data["ids"])

        def distribution(model_dirs: list[Path]) -> pd.DataFrame:
            return pd.DataFrame([benchmarks.score(d, actual, ref_ids) for d in model_dirs])

        lstm = distribution([forecast_path(panel, r).parents[1] for r in range(N_REPLICATIONS)
                             if forecast_path(panel, r).exists()])
        valendin = distribution([benchmarks.forecast_path(panel, r).parents[1]
                                 for r in range(benchmarks.N_REPLICATIONS)
                                 if benchmarks.forecast_path(panel, r).exists()])
        pareto_dir = benchmarks.STUDIES_BASE / benchmarks.pareto_suite_name(panel) / "ParetoNBD"
        pareto = distribution([pareto_dir]) if (pareto_dir / "Predictions").is_dir() else pd.DataFrame()
        zero = compute_forecast_metrics(actual, np.zeros_like(actual, dtype=float))

        def cell(df: pd.DataFrame, m: str, digits: int) -> str:
            if df.empty:
                return "—"
            s = df[m]
            return f"{s.mean():.{digits}f}" if len(s) == 1 else f"{s.mean():.{digits}f} ± {s.std(ddof=1):.{digits}f}"

        print(f"\n### {panel}\n")
        print(f"LSTM + ar_bounded_32 replications: {len(lstm)}/{N_REPLICATIONS}; "
              f"ValendinLSTM: {len(valendin)}; holdout transactions {int(actual.sum())}.\n")
        print("| model | n | bias % | MAPE | RMSE | Spearman |")
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        for label, df in (("LSTM + ar_bounded_32, sin/cos", lstm),
                          ("ValendinLSTM (benchmark)", valendin),
                          ("Pareto/NBD (benchmark)", pareto)):
            print(f"| {label} | {len(df)} | {cell(df, 'bias_percent', 1)} | "
                  f"{cell(df, 'mape_aggregate', 1)} | {cell(df, 'rmse', 4)} | {cell(df, 'spearman', 3)} |")
        print(f"| all-zero forecast | — | {zero['bias_percent']:.1f} | {zero['mape_aggregate']:.1f} | "
              f"{zero['rmse']:.4f} | — |")


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
