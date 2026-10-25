"""Heavy buyers: are customers with >5 / >10 / >20 calibration purchases forecast better
than the rest, and does any model or architecture forecast them better?

One row per (window, model, forecast, threshold, side) with the package metrics
(`compute_forecast_metrics`) plus Spearman, AUC and scale-free error ratios, so a heavy
group can be compared with the light one. Read-only on the repo.

    python .scratch/heavy-buyers/collect.py .scratch/heavy-buyers/results/heavy.pkl
"""
import sys, glob, warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_real_panel_benchmarks as b
from panelclv.data_preparation.target_channel import calibration_counts, holdout_actuals
from panelclv.models.monte_carlo_forecasting import compute_forecast_metrics

S = ROOT / "Studies"
THRESHOLDS = (5, 10, 20)

# (panel, cal) -> {model label: suite glob (relative to Studies/, ending in the model dir)}
SETS = {("cdnow", "2y"): {
    "ValendinLSTM": "real_panel_benchmarks__ValendinLSTM__cdnow__r*/ValendinLSTM",
    "LSTM+bounded32": "real_panel_lstm_bounded32__LSTM__cdnow__r*/LSTM",
    "LSTM+log": "real_panel_lstm_log__LSTM__cdnow__r*/LSTM",
    "LSTM+ratio": "real_panel_lstm_ratio__LSTM__cdnow__r*/LSTM",
    "ParetoNBD": "real_panel_benchmarks__ParetoNBD__cdnow__r*/ParetoNBD"}}
for p in ("electronics", "gift", "multichannel"):
    SETS[(p, "2y")] = {
        "ValendinLSTM": f"real_panel_benchmarks__ValendinLSTM__{p}__r*/ValendinLSTM",
        "LSTM+bounded32": f"real_panel_lstm_bounded32__LSTM__{p}__r*/LSTM",
        "LSTM+log": f"real_panel_lstm_log__LSTM__{p}__r*/LSTM",
        "LSTM+ratio": f"real_panel_lstm_ratio__LSTM__{p}__r*/LSTM",
        "ParetoNBD": f"real_panel_benchmarks__ParetoNBD__{p}__r*/ParetoNBD"}
    SETS[(p, "3y")] = {
        "LSTM+bounded32": f"real_panel_lstm_bounded32_cal3y__LSTM__{p}__r*/LSTM",
        "LSTM+log": f"real_panel_lstm_log_cal3y__LSTM__{p}__r*/LSTM",
        "LSTM+ratio": f"real_panel_lstm_ratio_cal3y__LSTM__{p}__r*/LSTM",
        "ParetoNBD": f"real_panel_benchmarks_cal3y__ParetoNBD__{p}__r*/ParetoNBD"}
SETS[("electronics", "2y")]["Transformer"] = \
    "real_panel_arms__Transformer__electronics__ar_bounded-no_cluster-valendin__a/Transformer"
SETS[("electronics", "5y")] = {
    "ValendinLSTM": "real_panel_benchmarks_cal5y__ValendinLSTM__electronics__r*/ValendinLSTM",
    "LSTM+bounded52": "epoch_floor_cal5y__LSTM__electronics__ar_bounded_52-nofloor__r*/LSTM",
    "Transformer": "attention_cal5y__Transformer__electronics__ar_bounded_52-nofloor__r*/Transformer",
    "Transformer-noAR": "attention_cal5y__Transformer__electronics__none-nofloor__r*/Transformer",
    "ParetoNBD": "real_panel_benchmarks_cal5y__ParetoNBD__electronics__r*/ParetoNBD"}


def auc(score, pos):
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def group_row(P, h, m):
    """Metrics of one forecast on the customers in mask m."""
    a, p = h[m].sum(1), P[m].sum(1)
    met = compute_forecast_metrics(h[m], P[m])
    mean_act = a.mean()
    return dict(
        n=int(m.sum()), act=float(a.sum()), pred=float(p.sum()), ret=float((a > 0).mean()),
        **met,
        # scale-free versions, so a heavy group's error can be set beside a light group's
        nrmse_total=float(met["rmse_customer_total"] / mean_act) if mean_act > 0 else np.nan,
        rel_mae=float(np.abs(p - a).mean() / mean_act) if mean_act > 0 else np.nan,
        rho=float(spearmanr(p, a).correlation) if m.sum() > 5 and a.std() > 0 and p.std() > 0 else np.nan,
        auc=auc(p, a > 0) if m.sum() > 5 else np.nan,
        cv=float(p.std() / p.mean()) if p.mean() > 0 else np.nan,
    )


def load_window(panel, cal):
    data = b.build_data(panel, cal)
    cc, h = calibration_counts(data), holdout_actuals(data)
    return [str(i) for i in data["ids"]], cc.sum(1), h


def score(job):
    panel, cal, model, pat = job
    ids, f, h = load_window(panel, cal)
    files = sorted(glob.glob(str(S / pat / "Predictions" / "Prediction_*.csv")))
    if model == "ParetoNBD":          # __r00 duplicates the bare original fit
        files = [x for x in files if "__r" in x]
    rows, seen = [], set()
    for fp in files:
        fr = pd.read_csv(fp)
        pos = {str(i): k for k, i in enumerate(fr.iloc[:, 0])}
        if len(pos) != len(ids):
            continue
        P = fr.iloc[:, 1:].to_numpy(float)[[pos[i] for i in ids]]
        if P.shape != h.shape:
            continue
        key = hash(P.tobytes())
        if key in seen:
            continue
        seen.add(key)
        rep = len(seen) - 1
        for k in THRESHOLDS:
            for side, m in (("heavy", f > k), ("light", f <= k)):
                if m.sum() < 2:
                    continue
                rows.append(dict(panel=panel, cal=cal, model=model, rep=rep, k=k, side=side,
                                 **group_row(P, h, m)))
        rows.append(dict(panel=panel, cal=cal, model=model, rep=rep, k=0, side="all",
                         **group_row(P, h, np.ones(len(f), bool))))
    print(panel, cal, model, "forecasts:", len(seen), flush=True)
    return rows


if __name__ == "__main__":
    jobs = [(p, c, m, pat) for (p, c), ms in SETS.items() for m, pat in ms.items()]
    with Pool(16) as pool:
        out = pool.map(score, jobs)
    df = pd.DataFrame([r for rows in out for r in rows])
    df.to_pickle(sys.argv[1])
    print("wrote", sys.argv[1], len(df))
