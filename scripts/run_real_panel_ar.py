"""The LSTM or Transformer with an AR encoding on every real panel, at the benchmark's windows.

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

**`--calibration 3y`** moves electronics, gift and multichannel to a three-year calibration:
two years to fit the weights, the third as the validation window, then the year after as
holdout (CDNOW's 77-week panel has no room for it and is left out). The encodings are
unchanged — C stays 26 — so the calibration window is the only thing that differs from
the `2y` runs, and the suites carry a `_cal3y` tag. The windows themselves are
`run_real_panel_benchmarks.WINDOWS_3Y`, read from there rather than restated, because the
frozen benchmarks run on them too (`--calibration 3y` there); a benchmark row appears in
this report once that run exists for the calibration being reported.

**Budget.** 100 replications per (encoding, panel), each a 100-trial Optuna search, the
ADR-0008 refit and a 500-path Monte Carlo forecast. One suite per replication. The work
list is encoding-major, then panel, then replication, so striding it `i::N` gives every
worker the same number of replications of every (encoding, panel) cell.

**`--model transformer`** runs the Transformer instead, under the same encodings,
windows, budget and scoring, with the Transformer space of the real-panel arms. Encoding
`none` carries no AR channel at all: count and sin/cos week only. LSTM suite names are
unchanged, so every finished LSTM run is read as before.

Usage:
    python scripts/run_real_panel_ar.py --encodings log,ratio --preflight
    python scripts/run_real_panel_ar.py --encodings log,ratio --worker 3/20
    python scripts/run_real_panel_ar.py --encodings log,ratio --check-complete
    python scripts/run_real_panel_ar.py --encodings bounded32,log,ratio --report
    python scripts/run_real_panel_ar.py --calibration 3y --encodings log --worker 1/20
    python scripts/run_real_panel_ar.py --model transformer --calibration 3y \
        --encodings none,bounded32,ratio --worker 1/20
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

# --- budget ----------------------------------------------------------------------
# Suite names do not carry these; `check_complete` reads them back off config.json (F23).
# Replications per (encoding, panel), per model. The Transformer's rollout re-reads its
# whole growing context at every step for every path, so a study costs many times the
# LSTM's; it runs 20.
REPLICATIONS = {"lstm": 100, "transformer": 20}
N_TRIALS = 100
N_SIMULATIONS = 500
BASE_SEED = 42

# Half-saturation constant of `saturating_tenure_<C>_periods`: about a quarter of the
# calibration window, as in the AR-encoding ablation.
SATURATION = {"cdnow": 10, "electronics": 26, "gift": 26, "multichannel": 26}

# Both calibrations' windows live in the benchmark runner and are read from there, so a
# benchmark and a developed model on the same calibration cannot drift onto different
# weeks. `--calibration` picks the table; its keys are the panels that calibration covers.
CALIBRATIONS = benchmarks.CALIBRATIONS


def ar_features(encoding: str, panel: str) -> tuple[str, ...]:
    """The AR channels one encoding carries on one panel."""
    if encoding == "none":
        return ()
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


ENCODINGS = ("none", "bounded32", "log", "ratio", "bounded32ratio")

# The spaces of the AR-encoding ablation and the real-panel arms. `embedding_dim` is
# absent: the default `valendin` embedder has no common width to search.
SEARCH_SPACES: dict[str, dict[str, object]] = {
    "lstm": {
        "lstm_hidden_size": {32, 64, 128},
        "dense_units":      {32, 64, 128},
        "dropout":          {0.0, 0.2},
        "learning_rate":    (1e-4, 1e-2, "log"),
        "weight_decay":     (1e-6, 1e-2, "log"),
        "batch_size":       {64, 128, 256},
    },
    "transformer": {
        "d_model":            {32, 64, 128},
        "nhead":              {2, 4, 8},
        "num_encoder_layers": (1, 3, "int"),
        "dropout":            {0.0, 0.1, 0.2, 0.3},
        "learning_rate":      (1e-4, 3e-3, "log"),
        "weight_decay":       (1e-6, 1e-2, "log"),
        "batch_size":         {64, 128, 256},
    },
}
MODEL_NAMES = {"lstm": "LSTM", "transformer": "Transformer"}
TRAINING = {"n_epochs": 100, "patience": 7, "verbose": False, "loss_type": "cross_entropy"}


def panel_config(encoding: str, panel: str, cal: str) -> PanelConfig:
    """Count, sin/cos week and one AR encoding, on the windows of calibration `cal`."""
    return PanelConfig(
        id_col="Id",
        target_col="Transactions",
        frequency="weekly",
        time_cols=("year", "week"),
        time_features={"add_week_sin_cos": True},
        ar_features=ar_features(encoding, panel),
        embedded_cols={"Transactions": "auto"},
        **CALIBRATIONS[cal][panel],
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


def build_data(encoding: str, panel: str, cal: str) -> dict:
    path = benchmarks.CLEAN / f"{panel}_customer_week_panel.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — push the panels to this box (Rules.md §3)")
    data = panel_dataset.prepare_dataset(
        pd.read_csv(path), panel_config(encoding, panel, cal), verbose=False)
    data["panel_name"] = panel
    check_arm_depth(data)
    return data


def suite_name(model: str, encoding: str, panel: str, replication: int, cal: str) -> str:
    """`real_panel_<model>_<encoding>[_cal3y]__<Model>__<panel>__r<NNN>` — one per item.

    The `2y` runs carry no tag, so the LSTM's `bounded32` resolves to
    `real_panel_lstm_bounded32`, the name its first run was stored under, and the finished
    runs read unchanged.
    """
    tag = benchmarks.calibration_tag(cal)
    return (f"real_panel_{model}_{encoding}{tag}__{MODEL_NAMES[model]}__{panel}"
            f"__r{replication:03d}")


def work_list(model: str, encodings: tuple[str, ...], cal: str) -> list[tuple[str, str, int]]:
    """Every (encoding, panel, replication), encoding-major then panel-major."""
    return [(e, p, r) for e in encodings for p in CALIBRATIONS[cal]
            for r in range(REPLICATIONS[model])]


def forecast_path(model: str, encoding: str, panel: str, replication: int, cal: str) -> Path:
    return (STUDIES_BASE / suite_name(model, encoding, panel, replication, cal)
            / MODEL_NAMES[model] / "Predictions" / "Prediction_1.csv")


def model_spec(model: str, n_trials: int, training: dict) -> ModelSpec:
    return ModelSpec(name=MODEL_NAMES[model], model_type=model, n_trials=n_trials,
                     search_space=dict(SEARCH_SPACES[model]), training=dict(training))


def run_worker(model: str, encodings: tuple[str, ...], cal: str, index: int, total: int) -> int:
    """Train this worker's stride; a finished replication is skipped, a cut one redone."""
    mine = work_list(model, encodings, cal)[index - 1::total]
    print(f"worker {index}/{total}: {len(mine)} suites", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    STUDIES_BASE.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple[str, str], dict] = {}
    for n, (enc, panel, rep) in enumerate(mine, start=1):
        name = suite_name(model, enc, panel, rep, cal)
        if forecast_path(model, enc, panel, rep, cal).exists():
            print(f"[{n}/{len(mine)}] {name}: done, skipping", flush=True)
            continue
        if (enc, panel) not in cache:
            cache[(enc, panel)] = build_data(enc, panel, cal)
        print(f"[{n}/{len(mine)}] {name}: training", flush=True)
        run_study_suite(StudySuiteConfig(
            studies_base_path=str(STUDIES_BASE),
            suite_name=name,
            n_studies_per_model=1,
            n_simulations=N_SIMULATIONS,
            device=device,
            data=cache[(enc, panel)],
            models=[model_spec(model, N_TRIALS, TRAINING)],
            base_seed=BASE_SEED + rep,
            overwrite=(STUDIES_BASE / name).exists(),
            keep_only_best_checkpoint=True,
        ))
    return 0


def preflight(model: str, encodings: tuple[str, ...], cal: str) -> int:
    """Build every (encoding, panel) and train each one tiny in a temporary directory."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for enc in encodings:
            for panel in CALIBRATIONS[cal]:
                try:
                    data = build_data(enc, panel, cal)
                    print(f"{enc:9s} {panel:12s} N={len(data['ids']):5d} "
                          f"T_CAL={int(data['T_CAL']):3d} T_HOLD={int(data['T_HOLD']):3d} "
                          f"val_periods={data.get('n_val_periods')} seq_cols={data['seq_cols']}")
                    run_study_suite(StudySuiteConfig(
                        studies_base_path=tmp, suite_name=f"preflight__{enc}__{panel}",
                        n_studies_per_model=1, n_simulations=2, device=device, data=data,
                        models=[model_spec(model, 1, {**TRAINING, "n_epochs": 3, "patience": 2})],
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


def check_complete(model: str, encodings: tuple[str, ...], cal: str) -> int:
    """What the declaration owes against what is on disk, budget included (F23)."""
    missing, wrong = [], []
    for enc in encodings:
        for panel in CALIBRATIONS[cal]:
            have = 0
            for rep in range(REPLICATIONS[model]):
                if not forecast_path(model, enc, panel, rep, cal).exists():
                    missing.append(suite_name(model, enc, panel, rep, cal))
                    continue
                have += 1
                cfg = json.loads(
                    (STUDIES_BASE / suite_name(model, enc, panel, rep, cal) / "config.json").read_text())
                budget = (cfg["n_simulations"], cfg["models"][0]["n_trials"])
                if budget != (N_SIMULATIONS, N_TRIALS):
                    wrong.append(f"{suite_name(model, enc, panel, rep, cal)} {budget}")
            print(f"{enc:9s} {panel:12s} {MODEL_NAMES[model]} {have:3d}/{REPLICATIONS[model]}")
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


def report(model: str, encodings: tuple[str, ...], cal: str) -> None:
    """One markdown table per panel: every encoding, the benchmarks, the zero forecast.

    A benchmark row is scored on this run's actuals and is shown only where that
    benchmark has been run on this calibration's own windows; on any other it forecasts a
    different holdout year.
    """
    for panel in CALIBRATIONS[cal]:
        # The cohort and holdout do not depend on the encoding, so any one rebuilds them.
        data = build_data(encodings[0], panel, cal)
        actual = holdout_actuals(data)
        ref_ids = np.asarray(data["ids"])

        def distribution(model_dirs: list[Path]) -> pd.DataFrame:
            return pd.DataFrame([benchmarks.score(d, actual, ref_ids) for d in model_dirs])

        rows = [(f"{MODEL_NAMES[model]} + ar_{enc}, sin/cos",
                 distribution([forecast_path(model, enc, panel, r, cal).parents[1]
                               for r in range(REPLICATIONS[model])
                               if forecast_path(model, enc, panel, r, cal).exists()]))
                for enc in encodings]
        # A benchmark row is shown only where that benchmark was run on *this*
        # calibration's windows: on any other it forecasts a different holdout year, and
        # setting it beside these rows would compare two different questions.
        valendin = [benchmarks.forecast_path(panel, r, cal).parents[1]
                    for r in range(benchmarks.N_REPLICATIONS)
                    if benchmarks.forecast_path(panel, r, cal).exists()]
        if valendin:
            rows.append(("ValendinLSTM (benchmark)", distribution(valendin)))
        pareto_dir = (benchmarks.STUDIES_BASE / benchmarks.pareto_suite_name(panel, cal)
                      / "ParetoNBD")
        if (pareto_dir / "Predictions").is_dir():
            rows.append(("Pareto/NBD (benchmark)", distribution([pareto_dir])))
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
    parser.add_argument("--model", choices=sorted(MODEL_NAMES), default="lstm")
    parser.add_argument("--calibration", choices=sorted(CALIBRATIONS), default="2y",
                        help="2y: the benchmark's windows; 3y: fit two years, validate one")
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

    cal = args.calibration
    if args.worker:
        index, total = (int(x) for x in args.worker.split("/"))
        sys.exit(run_worker(args.model, encodings, cal, index, total))
    if args.preflight:
        sys.exit(preflight(args.model, encodings, cal))
    if args.check_complete:
        sys.exit(check_complete(args.model, encodings, cal))
    report(args.model, encodings, cal)


if __name__ == "__main__":
    main()
