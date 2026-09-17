"""Isolate the two `_build_cbs` choices flagged in pareto-nbd-cdnow-replication.md §9(b).

Three arms, identical cohort / windows / seeds, differing only in the sufficient
statistics handed to the sampler:

  baseline  — the production path, exactly as every archived Pareto/NBD row was fit.
  shift     — T_cal advanced by one period, so customer age is measured to the END of
              the last calibration period rather than its start. Under `baseline`,
              forecast column 0 covers the last calibration week; under `shift` it
              covers holdout week 0, which is what it is scored against.
  no_collapse — x counts transactions instead of active periods (the collapse the
              docstring documents). Diverges on panels where a period can hold
              several transactions; kept here to show that it does.

Paired on seed: the CDNOW posterior's death process is weakly identified, so only the
within-seed difference is readable.
"""
import sys, json, time
import numpy as np, pandas as pd

sys.path.insert(0, "scripts")
from run_real_panel_benchmarks import build_data
from panelclv.benchmarks import pareto_nbd as pn
from panelclv.models.monte_carlo_forecasting import compute_forecast_metrics
from panelclv.data_preparation.target_channel import holdout_actuals
from scipy.stats import spearmanr

PANELS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["cdnow", "electronics"]
SEEDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [42, 43, 44]

_orig_build_cbs = pn._build_cbs


def patched(mode):
    def build(train_panel, *, id_col, target_col, time_col, period_in_days):
        cbs = _orig_build_cbs(train_panel, id_col=id_col, target_col=target_col,
                              time_col=time_col, period_in_days=period_in_days)
        if mode == "shift":
            # age to the end of the last calibration period, not its start
            cbs["T_cal"] = cbs["T_cal"] + 1.0
        elif mode == "no_collapse":
            panel = train_panel.copy()
            totals = panel.groupby(id_col)[target_col].sum()
            cbs["x"] = (totals.reindex(cbs.index) - 1).clip(lower=0).astype(float)
        return cbs
    return build


rows = []
for panel in PANELS:
    data = build_data(panel, "2y")
    actual = holdout_actuals(data)                      # (N, T_HOLD)
    actual_tot = actual.sum(axis=1)
    for mode in ("baseline", "shift", "no_collapse"):
        pn._build_cbs = _orig_build_cbs if mode == "baseline" else patched(mode)
        for seed in SEEDS:
            t0 = time.time()
            pred = pn.pareto_from_data(data, seed=seed)  # (N, T_HOLD)
            m = compute_forecast_metrics(actual, pred)
            rho = spearmanr(actual_tot, pred.sum(axis=1)).statistic
            row = dict(panel=panel, mode=mode, seed=seed,
                       predicted=float(pred.sum()), actual=float(actual.sum()),
                       bias=m["bias_percent"], mape=m["mape_aggregate"],
                       rmse=m["rmse"], spearman=float(rho), secs=round(time.time() - t0, 1))
            rows.append(row)
            print(json.dumps(row), flush=True)
    pn._build_cbs = _orig_build_cbs

df = pd.DataFrame(rows)
df.to_csv(".scratch/pnbd-cdnow-replication/window_shift_results.csv", index=False)
print()
print(df.groupby(["panel", "mode"])[["bias", "mape", "rmse", "spearman"]]
        .agg(["mean", "std"]).round(3).to_string())
