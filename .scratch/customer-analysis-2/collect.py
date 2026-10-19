"""Per-customer, per-replication group metrics for ValendinLSTM, LSTM+bounded32 and Pareto/NBD.

Writes one long pickle: one row per (panel, cal, model, replication, grouping, group).
Read-only on the repo: loads panels via scripts/run_real_panel_benchmarks.build_data and
the stored Prediction_*.csv files.
"""
import sys, glob, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_real_panel_benchmarks as b
from panelclv.data_preparation.target_channel import calibration_counts, holdout_actuals

S = ROOT / "Studies"
OUT = Path(sys.argv[1])

# (panel, cal) -> {model label: glob of prediction files}
SETS = {
    ("cdnow", "2y"): {"ValendinLSTM": "real_panel_benchmarks__ValendinLSTM__cdnow__r*/ValendinLSTM",
                      "LSTM+bounded32": "real_panel_lstm_bounded32__LSTM__cdnow__r*/LSTM",
                      "ParetoNBD": "real_panel_benchmarks__ParetoNBD__cdnow__r*/ParetoNBD"},
}
for p in ("electronics", "gift", "multichannel"):
    SETS[(p, "2y")] = {"ValendinLSTM": f"real_panel_benchmarks__ValendinLSTM__{p}__r*/ValendinLSTM",
                       "LSTM+bounded32": f"real_panel_lstm_bounded32__LSTM__{p}__r*/LSTM",
                       "ParetoNBD": f"real_panel_benchmarks__ParetoNBD__{p}__r*/ParetoNBD"}
    SETS[(p, "3y")] = {"LSTM+bounded32": f"real_panel_lstm_bounded32_cal3y__LSTM__{p}__r*/LSTM",
                       "ParetoNBD": f"real_panel_benchmarks_cal3y__ParetoNBD__{p}__r*/ParetoNBD"}
SETS[("electronics", "5y")] = {
    "ValendinLSTM": "real_panel_benchmarks_cal5y__ValendinLSTM__electronics__r*/ValendinLSTM",
    "LSTM+bounded52": "epoch_floor_cal5y__LSTM__electronics__ar_bounded_52-nofloor__r*/LSTM",
    "ParetoNBD": "real_panel_benchmarks_cal5y__ParetoNBD__electronics__r*/ParetoNBD"}


def auc(score, pos):
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0: return np.nan
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cut(x, edges, labels):
    return pd.cut(x, edges, labels=labels, right=True).astype(str)


def groupings(cc, h):
    """Customer groupings. All but the 'outcome' ones use calibration data only."""
    T = cc.shape[1]
    f = cc.sum(1)
    active = cc > 0
    weeks_since = np.argmax(active[:, ::-1], 1)              # 0 = bought in final cal week
    rec = T - 1 - weeks_since                                # index of last active week
    first = np.argmax(active, 1)
    tenure = T - first                                       # weeks from first purchase to cal end
    nweeks_active = active.sum(1)
    # mean inter-purchase gap for repeaters, in weeks
    gaps = np.where(nweeks_active > 1, (rec - first) / np.maximum(nweeks_active - 1, 1), np.nan)
    # "overdue" ratio: weeks since last vs own mean gap
    overdue = weeks_since / gaps
    H = h.sum(1)
    mean_f = f.mean()
    g = {}
    g["freq"] = cut(f, [0, 1, 2, 4, 9, np.inf], ["1", "2", "3-4", "5-9", "10+"])
    g["recency"] = cut(weeks_since, [-1, 4, 13, 26, 52, np.inf], ["0-4w", "5-13w", "14-26w", "27-52w", ">52w"])
    g["tenure"] = cut(tenure, [0, 13, 26, 52, 104, np.inf], ["<=13w", "14-26w", "27-52w", "53-104w", ">104w"])
    fq = np.where(f <= 1, "F1", np.where(f <= 4, "F2-4", "F5+"))
    rq = np.where(weeks_since <= 13, "R<=13w", np.where(weeks_since <= 52, "R14-52w", "R>52w"))
    g["RxF"] = np.char.add(np.char.add(fq.astype(str), " "), rq.astype(str))
    g["regularity"] = np.where(nweeks_active <= 1, "one-week only",
                       np.where(overdue <= 1, "within own gap",
                       np.where(overdue <= 3, "overdue 1-3x gap", "overdue >3x gap")))
    g["holdout"] = cut(H, [-1, 0, 1, 3, np.inf], ["0", "1", "2-3", "4+"])
    g["valendin"] = np.where((H == 0) & (f >= mean_f), "At Risk", np.where(H > f, "Opportunity", "Other"))
    g["all"] = np.array(["all"] * len(f))
    return g


rows = []
for (panel, cal), models in SETS.items():
    data = b.build_data(panel, cal)
    cc = calibration_counts(data)
    h = holdout_actuals(data)
    ids = [str(i) for i in data["ids"]]
    H = h.sum(1)
    G = groupings(cc, h)
    Th = h.shape[1]
    q = Th // 4
    for model, pat in models.items():
        files = sorted(glob.glob(str(S / pat / "Predictions" / "Prediction_*.csv")))
        if model == "ParetoNBD":
            # __r00 duplicates the bare original fit; keep the 20 seeded __rNN fits
            files = [f for f in files if "__r" in f]
        seen = set()
        for rep, fp in enumerate(files):
            fr = pd.read_csv(fp)
            pos = {str(i): k for k, i in enumerate(fr.iloc[:, 0])}
            if len(pos) != len(ids):
                print("SKIP size", fp); continue
            P = fr.iloc[:, 1:].to_numpy(float)[[pos[i] for i in ids]]
            if P.shape != h.shape:
                print("SKIP shape", fp, P.shape, h.shape); continue
            key = hash(P.tobytes())
            if key in seen:
                continue
            seen.add(key)
            Pt = P.sum(1)
            for gname, lab in G.items():
                for gl in np.unique(lab):
                    m = lab == gl
                    a, p = H[m], Pt[m]
                    rows.append(dict(
                        panel=panel, cal=cal, model=model, rep=rep, grouping=gname, group=gl,
                        n=int(m.sum()), act=float(a.sum()), pred=float(p.sum()),
                        abs_err=float(np.abs(p - a).sum()),
                        sq_err=float(((P[m] - h[m]) ** 2).sum()),
                        rho=float(spearmanr(p, a).correlation) if m.sum() > 5 and a.std() > 0 and p.std() > 0 else np.nan,
                        cv=float(p.std() / p.mean()) if p.mean() > 0 else np.nan,
                        pred_q1=float(P[m, :q].sum()), act_q1=float(h[m, :q].sum()),
                        pred_q4=float(P[m, -q:].sum()), act_q4=float(h[m, -q:].sum()),
                        sse_tot=float(((p - a) ** 2).sum()),
                        auc=auc(p, a > 0),
                    ))
            print(panel, cal, model, rep, flush=True) if rep % 25 == 0 else None
        print(panel, cal, model, "forecasts:", len(seen), flush=True)

pd.DataFrame(rows).to_pickle(OUT)
print("wrote", OUT, len(rows))
