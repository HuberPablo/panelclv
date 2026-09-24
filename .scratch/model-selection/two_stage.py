"""Offline test of two-stage selection on the selection-rescore trials: shortlist trials
within m% of the study's best val CE, then pick the shortlist's best by a rollout metric.
Reports the holdout metrics of the pick, paired against the plain val-CE pick."""
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd
from scipy.stats import bootstrap
d = pd.concat([pd.read_csv(f) for f in sorted((REPO / "Studies").glob("selection_rescore__*/selection_rescore.csv"))])
d["panel"] = d.suite.str.split("__").str[2]
rng = np.random.default_rng(0)
def ci(v):
    v = np.asarray(v, float)
    if np.allclose(v, 0): return f"{0:+.2f} (all ties)"
    r = bootstrap((v,), np.mean, n_resamples=10000, random_state=rng, method="percentile").confidence_interval
    return f"{v.mean():+6.2f} ({r.low:+6.2f}, {r.high:+6.2f})"
crit = {"roll MAPE": "val_mape_aggregate", "roll Spearman": "val_spearman"}
for panel, dp in d.groupby("panel"):
    studies = [g for _, g in dp.groupby("suite") if len(g) >= 10]
    base = [g.loc[g.val_loss.idxmin()] for g in studies]
    print(f"\n## {panel} ({len(studies)} studies). delta = two-stage pick minus CE pick; MAPE/|bias| lower better, Spearman higher better")
    for m in [0.005, 0.01, 0.02, 0.05]:
        sizes = [(g.val_loss <= g.val_loss.min()*(1+m)).sum() for g in studies]
        for cn, col in crit.items():
            picks = []
            for g in studies:
                s = g[g.val_loss <= g.val_loss.min()*(1+m)]
                picks.append(s.loc[s[col].idxmax() if col == "val_spearman" else s[col].idxmin()])
            dm = [p.hold_mape_aggregate - b.hold_mape_aggregate for p, b in zip(picks, base)]
            db = [abs(p.hold_bias_percent) - abs(b.hold_bias_percent) for p, b in zip(picks, base)]
            ds = [p.hold_spearman - b.hold_spearman for p, b in zip(picks, base)]
            print(f"  m={m:5.1%} shortlist median {np.median(sizes):4.1f}  {cn:13s} dMAPE {ci(dm)}  d|bias| {ci(db)}  dSpear {ci(ds)}")
