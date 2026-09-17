"""What the Pareto/NBD's weekly-clock conventions are worth, measured.

Background: `docs/pareto-nbd-cdnow-replication.md` §9(b) flagged two choices in
`benchmarks/pareto_nbd._build_cbs` that it could not separate. This script separates
them. The conclusion is in
`.scratch/pnbd-cdnow-replication/issues/01-pareto-weekly-discretisation-convention.md`;
the short version is that the second is not a defect and neither is worth much.

The model is continuous-time, so it needs each customer's observation length `T_cal` as a
real number. `_build_cbs` builds it from differences of week LABELS, which is exact under
the convention that a week's transactions happen at its end. Under that convention every
window lines up and there is nothing to correct. What a daily-resolution event log would
give instead — which is what BTYDplus reads — is a first purchase somewhere inside its
week, so an observation window longer by half a week in expectation.

Four arms, identical cohort / windows / seeds, differing only in the sufficient statistics
handed to the sampler. Paired on seed, because CDNOW's death parameters sit on a flat
likelihood ridge and only the within-seed difference is readable:

  baseline     the production path, exactly as every archived Pareto/NBD row was fit.
  T+0.5        the daily-resolution equivalent: first purchase placed mid-week on average.
               This is the only arm that is a candidate correction.
  T+1          the start-of-week convention. NOT a correction — it is the wrong convention
               applied deliberately, kept as a sensitivity bound on a whole week of clock.
  no_collapse  `x` counts transactions instead of active periods, the other §9(b) choice.
               Diverges on panels where a period can hold several transactions; included
               to show that it does, and to size the collapse on CDNOW where it does not.
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
MODES = ("baseline", "T+0.5", "T+1", "no_collapse")
OUT = ".scratch/pnbd-cdnow-replication/window_shift_results.csv"

_orig_build_cbs = pn._build_cbs


def patched(mode):
    """Return a `_build_cbs` that applies one arm's change to the production output."""
    def build(train_panel, *, id_col, target_col, time_col, period_in_days):
        cbs = _orig_build_cbs(train_panel, id_col=id_col, target_col=target_col,
                              time_col=time_col, period_in_days=period_in_days)
        if mode == "T+0.5":
            cbs["T_cal"] = cbs["T_cal"] + 0.5
        elif mode == "T+1":
            cbs["T_cal"] = cbs["T_cal"] + 1.0
        elif mode == "no_collapse":
            totals = train_panel.groupby(id_col)[target_col].sum()
            cbs["x"] = (totals.reindex(cbs.index) - 1).clip(lower=0).astype(float)
        return cbs
    return build


rows = []
for panel in PANELS:
    data = build_data(panel, "2y")
    actual = holdout_actuals(data)                       # (N, T_HOLD)
    actual_tot = actual.sum(axis=1)
    for mode in MODES:
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
df.to_csv(OUT, index=False)
print()
print(df.groupby(["panel", "mode"])[["bias", "mape", "rmse", "spearman"]]
        .agg(["mean", "std"]).round(3).to_string())
