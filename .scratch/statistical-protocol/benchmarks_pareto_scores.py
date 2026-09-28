"""Score every seeded Pareto/NBD fit on the real panels, once, into one CSV.

Pareto/NBD is now a replicated condition (`docs/statistical-protocol.md` §1): 20 MCMC fits
per panel and calibration, replication r seeded `BASE_SEED + r`, suites
`real_panel_benchmarks{,_cal3y,_cal5y}__ParetoNBD__<panel>__rNN`. Every doc that sets a
neural condition against Pareto/NBD compares it with these 20 values under the
independent bootstrap, so they are scored here with the benchmark report's own `score`
(the single scoring authority plus per-customer Spearman) and written to

    .scratch/statistical-protocol/results/pareto_seeded_scores.csv
        columns: cal, panel, rep, bias_percent, mape_aggregate, rmse,
                 rmse_customer_total, spearman, cv

`cv` is the forecast CV (std / mean of per-customer predicted holdout totals), the
collapse diagnostic of the protocol's §4. A calibration whose fits are not all on disk is
skipped with a message rather than scored partially, so a half-finished background fit
cannot leak an n < 20 into a doc.

    PYTHONPATH=src:scripts python .scratch/statistical-protocol/benchmarks_pareto_scores.py [2y 3y 5y]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from panelclv.data_preparation.target_channel import holdout_actuals
from panelclv.studies import load_model_predictions
from run_real_panel_benchmarks import (CALIBRATIONS, N_REPLICATIONS, build_data,
                                       pareto_path, score)

OUT = Path(__file__).resolve().parent / "results" / "pareto_seeded_scores.csv"


def cv_of(model_dir: Path) -> float:
    """Forecast CV of one stored forecast: spread of per-customer totals over their mean."""
    values, _ = load_model_predictions(model_dir, study=1)
    totals = values.sum(axis=1)
    return float(totals.std() / totals.mean())


def score_calibration(cal: str) -> pd.DataFrame | None:
    rows = []
    for panel in CALIBRATIONS[cal]:
        done = [r for r in range(N_REPLICATIONS)
                if (pareto_path(panel, r, cal) / "Predictions").is_dir()]
        if len(done) < N_REPLICATIONS:
            print(f"{cal}/{panel}: {len(done)}/{N_REPLICATIONS} fits on disk -- skipped")
            return None
        data = build_data(panel, cal)
        actual, ids = holdout_actuals(data), np.asarray(data["ids"])
        for r in done:
            path = pareto_path(panel, r, cal)
            rows.append(dict(cal=cal, panel=panel, rep=r, **score(path, actual, ids),
                             cv=cv_of(path)))
    return pd.DataFrame(rows)


def load(cal: str, panel: str) -> pd.DataFrame:
    """The 20 seeded fits of one (calibration, panel), as scored by this script."""
    d = pd.read_csv(OUT)
    g = d[(d.cal == cal) & (d.panel == panel)]
    if len(g) != N_REPLICATIONS:
        raise RuntimeError(f"{cal}/{panel}: {len(g)} scored fits in {OUT}; rerun this script")
    return g


def main() -> None:
    cals = sys.argv[1:] or list(CALIBRATIONS)
    old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["cal"])
    for cal in cals:
        new = score_calibration(cal)
        if new is None:
            continue
        old = pd.concat([old[old.cal != cal], new], ignore_index=True)
        print(f"\n{cal}: mean (sd) over {N_REPLICATIONS} seeded fits")
        print(new.groupby("panel")[["bias_percent", "mape_aggregate", "rmse",
                                    "rmse_customer_total", "spearman", "cv"]]
              .agg(["mean", "std", "min", "max"]).round(4).T.to_string())
    OUT.parent.mkdir(exist_ok=True)
    old.to_csv(OUT, index=False)


if __name__ == "__main__":
    main()
