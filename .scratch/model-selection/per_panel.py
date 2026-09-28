"""Per-panel description of the selection rescore: what each criterion's pick scores on
the holdout (means over studies, no intervals), how far apart a study's trials are, and the
holdout outcome by quartile of a trial's own val CE. Descriptive only — `docs/model-selection.md`
§4 and §5. The claims (each criterion's rank correlation with the holdout, and its
difference from val CE, per panel and model through `effect`) come from
`.scratch/training-budget/selection_analysis.py`."""
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd


d = pd.concat([pd.read_csv(f) for f in sorted((REPO/"Studies").glob("selection_rescore__*/selection_rescore.csv"))])
d["panel"] = d.suite.str.split("__").str[2]
C = {"valCE": lambda x: x.val_loss, "roll_MAPE": lambda x: x.val_mape_aggregate,
     "roll_Spear": lambda x: -x.val_spearman, "roll_absbias": lambda x: x.val_bias_percent.abs(),
     "best_epoch": lambda x: -x.best_epoch}
for panel, dp in d.groupby("panel"):
    studies = [g for _, g in dp.groupby("suite") if len(g) >= 10]
    print(f"\n## {panel}: {len(studies)} studies, {sum(len(g) for g in studies)} trials, models {sorted(dp.model.unique())}")
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
