"""Holdout score of each criterion's pick minus the expected score of a random trial
(the study's trial mean), paired over studies, 95% bootstrap CI."""
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd
from scipy.stats import bootstrap
d = pd.concat([pd.read_csv(f) for f in sorted((REPO / "Studies").glob("selection_rescore__*/selection_rescore.csv"))])
d["panel"] = d.suite.str.split("__").str[2]; d["abs_bias"] = d.hold_bias_percent.abs()
rng = np.random.default_rng(0)
def ci(v):
    r = bootstrap((np.asarray(v),), np.mean, n_resamples=10000, random_state=rng, method="percentile").confidence_interval
    return f"{np.mean(v):+6.2f} ({r.low:+6.2f}, {r.high:+6.2f})"
C = {"val CE": ("val_loss", 1), "roll MAPE": ("val_mape_aggregate", 1), "roll |bias|": (None, 1),
     "roll Spearman": ("val_spearman", -1), "best epoch": ("best_epoch", -1)}
for panel, dp in d.groupby("panel"):
    studies = [g for _, g in dp.groupby("suite") if len(g) >= 10]
    print(f"\n## {panel}: pick minus random-trial expectation")
    print(f"   random trial means: MAPE {np.mean([g.hold_mape_aggregate.mean() for g in studies]):.1f}, |bias| {np.mean([g.abs_bias.mean() for g in studies]):.1f}, Spearman {np.mean([g.hold_spearman.mean() for g in studies]):.3f}")
    for cn, (col, sign) in C.items():
        rows = []
        for g in studies:
            key = g.val_bias_percent.abs() if col is None else sign * g[col]
            p = g.loc[key.idxmin()]
            rows.append((p.hold_mape_aggregate - g.hold_mape_aggregate.mean(), p.abs_bias - g.abs_bias.mean(), p.hold_spearman - g.hold_spearman.mean()))
        r = np.array(rows)
        print(f"   {cn:13s} MAPE {ci(r[:,0])}  |bias| {ci(r[:,1])}  Spearman {ci(r[:,2])}")
