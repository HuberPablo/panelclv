"""Is the LSTM better than Pareto/NBD on electronic_5y? Backs the "LSTM against
Pareto/NBD" claims in `docs/feature_engineering.md`.

The cell tables' intervals resample only the 20 studies and hold Pareto/NBD's single
fit fixed, which ignores that both models are scored on one sample of customers. Here
each bootstrap draw resamples the 3,755 customers (the same draw for both models, so
the comparison is paired) and the 20 studies together:

    delta = mean over resampled studies of LSTM metric - Pareto/NBD metric,

both on the resampled customers. Supported when the 95% percentile interval excludes
zero. `wins_*` counts how many of the 20 studies' stored scores beat Pareto/NBD's.

    PYTHONPATH=src python .scratch/feature-engineering-5y/lstm_vs_pareto.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / ".scratch/feature-engineering-5y"))
sys.path.insert(0, str(ROOT / ".scratch/training-budget"))
import effects_5y as E
from panelclv.data_preparation.target_channel import holdout_actuals
from panelclv.studies.suite_reader import load_model_predictions
from panelclv.models.monte_carlo_forecasting import compute_forecast_metrics

base = E.floor.build_data(); act = holdout_actuals(base).astype(float); ids = np.asarray(base["ids"])
N = act.shape[0]
def load(md):
    v, i = load_model_predictions(md, study=1); assert np.array_equal(np.asarray(i), ids); return v.astype(float)
pn = load(E.S / "real_panel_benchmarks_cal5y__ParetoNBD__electronics" / "ParetoNBD")
print("PNBD check", compute_forecast_metrics(act, pn))

def metrics_w(pred_tot, pred_wk, w, act_tot, act_wk_w):
    # w: customer multiplicity weights (N,). pred_tot (N,), pred_wk (N,T)
    rmse = np.sqrt(np.sum(w * (pred_tot - act_tot) ** 2) / w.sum())
    pw = w @ pred_wk; aw = act_wk_w
    bias = 100 * (pw.sum() - aw.sum()) / aw.sum()
    mape = 100 * np.abs(aw - pw).sum() / aw.sum()
    return rmse, abs(bias), mape

def spear(idx, p_tot, a_tot):
    a = rankdata(a_tot[idx]); p = rankdata(p_tot[idx]); return np.corrcoef(a, p)[0, 1]

rng = np.random.default_rng(0)
B = 1000
a_tot = act.sum(1); pn_tot = pn.sum(1)
cells = [("none", "searched")] + [(f, a) for a in E.ARMS for f in E.FEATURES]
rows = []
per = pd.read_csv(ROOT / ".scratch/feature-engineering-5y/results/per_forecast.csv")
pref = per[per.model == "ParetoNBD"].iloc[0]
for f, arm in cells:
    preds = np.stack([load(E.suite("LSTM", f, arm, r)[0] / E.suite("LSTM", f, arm, r)[1]) for r in range(20)])
    tots = preds.sum(2)
    g = per[(per.model == "LSTM") & (per.feature == f) & (per.arm == arm)]
    wins = {"rmse": (g.rmse_customer_total < pref.rmse_customer_total).sum(),
            "absbias": (g.abs_bias < pref.abs_bias).sum(),
            "mape": (g.mape_aggregate < pref.mape_aggregate).sum(),
            "spearman": (g.spearman > pref.spearman).sum()}
    D = np.empty((B, 4))
    for b in range(B):
        idx = rng.integers(0, N, N); w = np.bincount(idx, minlength=N).astype(float)
        J = rng.integers(0, 20, 20)
        aw = w @ act
        p = metrics_w(pn_tot, pn, w, a_tot, aw); ps = spear(idx, pn_tot, a_tot)
        m = np.mean([metrics_w(tots[j], preds[j], w, a_tot, aw) for j in J], axis=0)
        ms = np.mean([spear(idx, tots[j], a_tot) for j in J])
        D[b] = [m[0] - p[0], m[1] - p[1], m[2] - p[2], ms - ps]
    lo, hi = np.percentile(D, [2.5, 97.5], axis=0)
    names = ["dRMSE", "d|bias|", "dMAPE", "dSpearman"]
    row = {"input": f, "rule": arm}
    for k, n in enumerate(names):
        row[n] = f"{D[:, k].mean():+.3f} [{lo[k]:+.3f}, {hi[k]:+.3f}]"
    row.update({f"wins_{k}": int(v) for k, v in wins.items()})
    rows.append(row); print(row, flush=True)
pd.DataFrame(rows).to_csv(Path(__file__).resolve().parent / "results" / "lstm_vs_pareto.csv", index=False)
