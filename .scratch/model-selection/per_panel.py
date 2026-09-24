"""Per-panel re-check of the selection rescore: correlations with bootstrap CIs, paired
deltas against val CE, and what each criterion's pick actually scores on the holdout."""
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd
from scipy.stats import bootstrap


d = pd.concat([pd.read_csv(f) for f in sorted((REPO/"Studies").glob("selection_rescore__*/selection_rescore.csv"))])
d["panel"] = d.suite.str.split("__").str[2]
rng = np.random.default_rng(0)
def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    r = bootstrap((v,), np.mean, confidence_level=0.95, n_resamples=10000, random_state=rng, method="percentile")
    return f"{v.mean():+.3f} ({r.confidence_interval.low:+.3f}, {r.confidence_interval.high:+.3f})"
C = {"valCE": lambda x: x.val_loss, "roll_MAPE": lambda x: x.val_mape_aggregate,
     "roll_Spear": lambda x: -x.val_spearman, "roll_absbias": lambda x: x.val_bias_percent.abs(),
     "best_epoch": lambda x: -x.best_epoch}
T = {"MAPE": lambda x: x.hold_mape_aggregate, "|bias|": lambda x: x.hold_bias_percent.abs(),
     "Spearman": lambda x: -x.hold_spearman}
for panel, dp in d.groupby("panel"):
    studies = [g for _, g in dp.groupby("suite") if len(g) >= 10]
    print(f"\n## {panel}: {len(studies)} studies, {sum(len(g) for g in studies)} trials, models {sorted(dp.model.unique())}")
    for tn, tf in T.items():
        base = np.array([C["valCE"](g).corr(tf(g), method="spearman") for g in studies])
        for cn, cf in C.items():
            v = np.array([cf(g).corr(tf(g), method="spearman") for g in studies])
            s = f"  {tn:9s} {cn:12s} rho {ci(v)}"
            if cn != "valCE": s += f"   delta {ci(v - base)}"
            print(s)
    # What the pick scores: argmin of each criterion, vs oracle and median trial
    print("  pick -> holdout MAPE / |bias| / Spearman (means over studies)")
    for cn, cf in C.items():
        p = [g.loc[cf(g).idxmin()] for g in studies]
        print(f"   {cn:12s} {np.mean([r.hold_mape_aggregate for r in p]):6.1f} {np.mean([abs(r.hold_bias_percent) for r in p]):6.1f} {np.mean([r.hold_spearman for r in p]):.3f}")
    print(f"   {'median':12s} {np.mean([g.hold_mape_aggregate.median() for g in studies]):6.1f} {np.mean([g.hold_bias_percent.abs().median() for g in studies]):6.1f} {np.mean([g.hold_spearman.median() for g in studies]):.3f}")
    print(f"   {'oracle':12s} {np.mean([g.hold_mape_aggregate.min() for g in studies]):6.1f} {np.mean([g.hold_bias_percent.abs().min() for g in studies]):6.1f} {np.mean([g.hold_spearman.max() for g in studies]):.3f}")
    # within-study spread of holdout metrics (how different the trials are)
    print(f"  within-study IQR: MAPE {np.mean([g.hold_mape_aggregate.quantile(.75)-g.hold_mape_aggregate.quantile(.25) for g in studies]):.1f}, bias {np.mean([g.hold_bias_percent.quantile(.75)-g.hold_bias_percent.quantile(.25) for g in studies]):.1f}, Spearman {np.mean([g.hold_spearman.quantile(.75)-g.hold_spearman.quantile(.25) for g in studies]):.3f}")
    # quartiles by val CE
    q = pd.concat([g.assign(q=pd.qcut(g.val_loss.rank(method='first'), 4, labels=["best","2nd","3rd","worst"])) for g in studies])
    print(q.groupby("q", observed=True)[["hold_bias_percent","hold_mape_aggregate","hold_spearman"]].mean().round(2).to_string())
