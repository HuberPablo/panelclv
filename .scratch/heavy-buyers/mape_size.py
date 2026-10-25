"""Is the heavy group's higher aggregate MAPE a group-size artefact? Score the same forecast
on random light-group subsets matched in size to the heavy group."""
import sys, glob
sys.path[:0] = ["src", "scripts", ".scratch/heavy-buyers"]
import numpy as np, pandas as pd
from collect import load_window, S, SETS
from panelclv.models.monte_carlo_forecasting import compute_forecast_metrics as cfm
rng = np.random.default_rng(0)
for (panel, cal, model, k) in [("electronics","3y","LSTM+bounded32",20), ("electronics","3y","ParetoNBD",20),
                               ("electronics","5y","LSTM+bounded52",20), ("electronics","5y","ParetoNBD",20),
                               ("gift","3y","LSTM+bounded32",10), ("gift","3y","ParetoNBD",10)]:
    ids, f, h = load_window(panel, cal)
    fp = sorted(glob.glob(str(S / SETS[(panel, cal)][model] / "Predictions" / "Prediction_*.csv")))
    fp = [x for x in fp if "__r" in x][0] if model == "ParetoNBD" else fp[0]
    fr = pd.read_csv(fp); pos = {str(i): j for j, i in enumerate(fr.iloc[:, 0])}
    P = fr.iloc[:, 1:].to_numpy(float)[[pos[i] for i in ids]]
    heavy, light = f > k, f <= k
    n = heavy.sum()
    m_heavy = cfm(h[heavy], P[heavy])["mape_aggregate"]
    m_light = cfm(h[light], P[light])["mape_aggregate"]
    sub = [cfm(h[s], P[s])["mape_aggregate"] for s in (rng.choice(np.flatnonzero(light), n, replace=False) for _ in range(200))]
    # light subsets matched on holdout volume instead: top-n light customers by holdout total give similar volume
    print(f"{panel} {cal} {model} >{k}: heavy n={n} MAPE {m_heavy:.1f} | light (all) {m_light:.1f} | light random n={n}: median {np.median(sub):.1f} [{np.percentile(sub,5):.1f}, {np.percentile(sub,95):.1f}]")
