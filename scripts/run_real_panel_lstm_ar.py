"""The LSTM with an AR encoding on every real panel, at the benchmark's windows.

The developed model run on CDNOW, electronics, gift and multichannel under each of the
AR encodings the AR-encoding ablation compared, so every encoding can be read against the
frozen benchmarks (`scripts/run_real_panel_benchmarks.py`) and against each other, panel
by panel, on the same customers.

**Inputs.** The transaction count (embedded), `week_sin` / `week_cos`, and one encoding of
the customer's history from `ENCODINGS`. Every other panel column is discarded.

    bounded32  nested flags active_in_last_{2,4,8,16,32}_periods + has_transacted_before:
               all zero past the deepest bin, so nothing leaves the fitted range, at the
               cost of every distinction past 32 weeks
    log        log(1 + recency), cumulative_transactions, log(1 + tenure): the
               coordinate in which Pareto/NBD's log-survival is linear; keeps the
               resolution, still drifts past the calibration ceiling
    ratio      recency_over_tenure, transaction_rate, saturating_tenure_<C>,
               has_transacted_before: the bounded Pareto/NBD triple; the first two
               cannot leave their calibration range (`.scratch/ar-encoding-support/`)
    bounded32ratio  both sets together: the flags, which protected the level on the
               panels where it is hard, and the ratio triple, which reached Pareto/NBD's
               ranking on three of four (docs/benchmarks-real-panels.md)

Definitions are those of `scripts/run_ar_encoding_ablation.py`, so a result here reads
straight against that ablation. The saturation constant C is a quarter of the calibration
window, as there: 26 on the 104-week panels, 10 on CDNOW's 39.

**Depth 32 on every panel, CDNOW included, by decision.** CDNOW calibrates on 39 weeks,
so a 32-week silence is rare while fitting. `check_arm_depth` still refuses a flag at or
above T_CAL, where it would be an exact copy of `has_transacted_before`.

**Windows, cohort and scoring are the benchmark's**, imported from its runner rather than
restated, so the runs cannot drift apart.

**Budget.** 100 replications per (encoding, panel), each a 100-trial Optuna search, the
ADR-0008 refit and a 500-path Monte Carlo forecast. One suite per replication. The work
list is encoding-major, then panel, then replication, so striding it `i::N` gives every
worker the same number of replications of every (encoding, panel) cell.

Usage:
    python scripts/run_real_panel_lstm_ar.py --encodings log,ratio --preflight
    python scripts/run_real_panel_lstm_ar.py --encodings log,ratio --worker 3/20
    python scripts/run_real_panel_lstm_ar.py --encodings log,ratio --check-complete
    python scripts/run_real_panel_lstm_ar.py --encodings bounded32,log,ratio --report
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

# --- budget ----------------------------------------------------------------------
# Suite names do not carry these; `check_complete` reads them back off config.json (F23).
N_REPLICATIONS = 100
N_TRIALS = 100
N_SIMULATIONS = 500
BASE_SEED = 42

# Half-saturation constant of `saturating_tenure_<C>_periods`: about a quarter of the
# calibration window, as in the AR-encoding ablation.
SATURATION = {"cdnow": 10, "electronics": 26, "gift": 26, "multichannel": 26}


def ar_features(encoding: str, panel: str) -> tuple[str, ...]:
    """The AR channels one encoding carries on one panel."""
    if encoding == "bounded32":
        return tuple(f"active_in_last_{k}_periods" for k in (2, 4, 8, 16, 32)) + (
            "has_transacted_before",)
    if encoding == "log":
        return ("log_period_since_last_transaction", "cumulative_transactions",
                "log_period_since_first_transaction")
    if encoding == "ratio":
        return ("recency_over_tenure", "transaction_rate",
                f"saturating_tenure_{SATURATION[panel]}_periods", "has_transacted_before")
    if encoding == "bounded32ratio":
        # The flags hold the level where it is hard; the ratio triple holds the ranking.
        # `has_transacted_before` belongs to both sets and is carried once.
        return ar_features("bounded32", panel) + tuple(
            c for c in ar_features("ratio", panel) if c != "has_transacted_before")
    raise ValueError(f"unknown encoding {encoding!r}; choose from {ENCODINGS}")


ENCODINGS = ("bounded32", "log", "ratio", "bounded32ratio")

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


def panel_config(encoding: str, panel: str) -> PanelConfig:
    """Count, sin/cos week and one AR encoding, on the benchmark's windows."""
    return PanelConfig(
        id_col="Id",
        target_col="Transactions",
        frequency="weekly",
        time_cols=("year", "week"),
        time_features={"add_week_sin_cos": True},
        ar_features=ar_features(encoding, panel),
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


def build_data(encoding: str, panel: str) -> dict:
    path = benchmarks.CLEAN / f"{panel}_customer_week_panel.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — push the panels to this box (Rules.md §3)")
    data = panel_dataset.prepare_dataset(
        pd.read_csv(path), panel_config(encoding, panel), verbose=False)
    data["panel_name"] = panel
    check_arm_depth(data)
    return data


def suite_name(encoding: str, panel: str, replication: int) -> str:
    """`real_panel_lstm_<encoding>__LSTM__<panel>__r<NNN>` — disjoint per work item.

    `bounded32` resolves to `real_panel_lstm_bounded32`, the name its first run was
    stored under, so that run is read by this script unchanged.
    """
    return f"real_panel_lstm_{encoding}__LSTM__{panel}__r{replication:03d}"


def work_list(encodings: tuple[str, ...]) -> list[tuple[str, str, int]]:
    """Every (encoding, panel, replication), encoding-major then panel-major."""
    return [(e, p, r) for e in encodings for p in PANELS for r in range(N_REPLICATIONS)]


def forecast_path(encoding: str, panel: str, replication: int) -> Path:
    return (STUDIES_BASE / suite_name(encoding, panel, replication)
            / "LSTM" / "Predictions" / "Prediction_1.csv")


def model_spec(n_trials: int, training: dict) -> ModelSpec:
    return ModelSpec(name="LSTM", model_type="lstm", n_trials=n_trials,
                     search_space=dict(SEARCH_SPACE), training=dict(training))


def run_worker(encodings: tuple[str, ...], index: int, total: int) -> int:
    """Train this worker's stride; a finished replication is skipped, a cut one redone."""
    mine = work_list(encodings)[index - 1::total]
    print(f"worker {index}/{total}: {len(mine)} suites", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    STUDIES_BASE.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple[str, str], dict] = {}
    for n, (enc, panel, rep) in enumerate(mine, start=1):
        name = suite_name(enc, panel, rep)
        if forecast_path(enc, panel, rep).exists():
            print(f"[{n}/{len(mine)}] {name}: done, skipping", flush=True)
            continue
        if (enc, panel) not in cache:
            cache[(enc, panel)] = build_data(enc, panel)
        print(f"[{n}/{len(mine)}] {name}: training", flush=True)
        run_study_suite(StudySuiteConfig(
            studies_base_path=str(STUDIES_BASE),
            suite_name=name,
            n_studies_per_model=1,
            n_simulations=N_SIMULATIONS,
            device=device,
            data=cache[(enc, panel)],
            models=[model_spec(N_TRIALS, TRAINING)],
            base_seed=BASE_SEED + rep,
            overwrite=(STUDIES_BASE / name).exists(),
            keep_only_best_checkpoint=True,
        ))
    return 0


def preflight(encodings: tuple[str, ...]) -> int:
    """Build every (encoding, panel) and train each one tiny in a temporary directory."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for enc in encodings:
            for panel in PANELS:
                try:
                    data = build_data(enc, panel)
                    print(f"{enc:9s} {panel:12s} N={len(data['ids']):5d} "
                          f"T_CAL={int(data['T_CAL']):3d} seq_cols={data['seq_cols']}")
                    run_study_suite(StudySuiteConfig(
                        studies_base_path=tmp, suite_name=f"preflight__{enc}__{panel}",
                        n_studies_per_model=1, n_simulations=2, device=device, data=data,
                        models=[model_spec(1, {**TRAINING, "n_epochs": 3, "patience": 2})],
                        base_seed=BASE_SEED, keep_only_best_checkpoint=True,
                    ))
                    print("  ok")
                except Exception as exc:                # noqa: BLE001 — report them all
                    print(f"  FAIL {type(exc).__name__}: {exc}")
                    failures.append(f"{enc}/{panel}")
    if failures:
        print(f"\n{len(failures)} cell(s) failed: {failures}. Fix before renting anything.")
        return 1
    print("\nAll cells build and train. Safe to launch.")
    return 0


def check_complete(encodings: tuple[str, ...]) -> int:
    """What the declaration owes against what is on disk, budget included (F23)."""
    missing, wrong = [], []
    for enc in encodings:
        for panel in PANELS:
            have = 0
            for rep in range(N_REPLICATIONS):
                if not forecast_path(enc, panel, rep).exists():
                    missing.append(suite_name(enc, panel, rep))
                    continue
                have += 1
                cfg = json.loads(
                    (STUDIES_BASE / suite_name(enc, panel, rep) / "config.json").read_text())
                budget = (cfg["n_simulations"], cfg["models"][0]["n_trials"])
                if budget != (N_SIMULATIONS, N_TRIALS):
                    wrong.append(f"{suite_name(enc, panel, rep)} {budget}")
            print(f"{enc:9s} {panel:12s} LSTM {have:3d}/{N_REPLICATIONS}")
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


def report(encodings: tuple[str, ...]) -> None:
    """One markdown table per panel: every encoding, the two benchmarks, the zero forecast."""
    for panel in PANELS:
        # The cohort and holdout do not depend on the encoding, so any one rebuilds them.
        data = build_data(encodings[0], panel)
        actual = holdout_actuals(data)
        ref_ids = np.asarray(data["ids"])

        def distribution(model_dirs: list[Path]) -> pd.DataFrame:
            return pd.DataFrame([benchmarks.score(d, actual, ref_ids) for d in model_dirs])

        rows = [(f"LSTM + ar_{enc}, sin/cos",
                 distribution([forecast_path(enc, panel, r).parents[1]
                               for r in range(N_REPLICATIONS)
                               if forecast_path(enc, panel, r).exists()]))
                for enc in encodings]
        rows.append(("ValendinLSTM (benchmark)",
                     distribution([benchmarks.forecast_path(panel, r).parents[1]
                                   for r in range(benchmarks.N_REPLICATIONS)
                                   if benchmarks.forecast_path(panel, r).exists()])))
        pareto_dir = benchmarks.STUDIES_BASE / benchmarks.pareto_suite_name(panel) / "ParetoNBD"
        rows.append(("Pareto/NBD (benchmark)",
                     distribution([pareto_dir]) if (pareto_dir / "Predictions").is_dir()
                     else pd.DataFrame()))
        zero = compute_forecast_metrics(actual, np.zeros_like(actual, dtype=float))

        def cell(df: pd.DataFrame, m: str, digits: int) -> str:
            if df.empty:
                return "—"
            s = df[m]
            if len(s) == 1:
                return f"{s.mean():.{digits}f}"
            return f"{s.mean():.{digits}f} ± {s.std(ddof=1):.{digits}f}"

        print(f"\n### {panel}\n")
        print(f"Holdout transactions {int(actual.sum())}.\n")
        print("| model | n | bias % | MAPE | RMSE | Spearman |")
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        for label, df in rows:
            print(f"| {label} | {len(df)} | {cell(df, 'bias_percent', 1)} | "
                  f"{cell(df, 'mape_aggregate', 1)} | {cell(df, 'rmse', 4)} | "
                  f"{cell(df, 'spearman', 3)} |")
        print(f"| all-zero forecast | — | {zero['bias_percent']:.1f} | "
              f"{zero['mape_aggregate']:.1f} | {zero['rmse']:.4f} | — |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encodings", required=True,
                        help=f"comma-separated, from {', '.join(ENCODINGS)}")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--worker", metavar="I/N", help="train this worker's stride")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--check-complete", action="store_true")
    mode.add_argument("--report", action="store_true")
    args = parser.parse_args()

    encodings = tuple(e.strip() for e in args.encodings.split(",") if e.strip())
    unknown = [e for e in encodings if e not in ENCODINGS]
    if unknown or not encodings:
        parser.error(f"unknown encoding(s) {unknown}; choose from {ENCODINGS}")

    if args.worker:
        index, total = (int(x) for x in args.worker.split("/"))
        sys.exit(run_worker(encodings, index, total))
    if args.preflight:
        sys.exit(preflight(encodings))
    if args.check_complete:
        sys.exit(check_complete(encodings))
    report(encodings)


if __name__ == "__main__":
    main()
