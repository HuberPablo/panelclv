"""Offline test of two-stage selection on the selection-rescore trials: shortlist trials
within m% of the study's best val CE, then pick the shortlist's best by a rollout metric.
`docs/model-selection.md` S3.

Delta is the two-stage pick's holdout score minus the plain val-CE pick's in the same
study, so the pair is resampled together: `effect(two_stage, ce_pick, paired=True)`
(`docs/statistical-protocol.md`). Each panel and each model is analysed separately."""
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
import numpy as np, pandas as pd
from panelclv.evaluation.effects import effect
d = pd.concat([pd.read_csv(f) for f in sorted((REPO / "Studies").glob("selection_rescore__*/selection_rescore.csv"))])
d["panel"] = d.suite.str.split("__").str[2]; d["abs_bias"] = d.hold_bias_percent.abs()
def ci(e):  # Spearman lives on a 0-1 scale, so it gets three decimals
    f = "{:+.3f}" if "Spear" in e.metric else "{:+.2f}"
    return f"{f.format(e.delta)} ({f.format(e.lo)}, {f.format(e.hi)}){'*' if e.supported else ' '}"
crit = {"roll MAPE": "val_mape_aggregate", "roll Spearman": "val_spearman"}
T = {"dMAPE": "hold_mape_aggregate", "d|bias|": "abs_bias", "dSpear": "hold_spearman"}
for (panel, model), dp in d.groupby(["panel", "model"]):
    studies = [g for _, g in dp.groupby("suite") if len(g) >= 10]
    base = [g.loc[g.val_loss.idxmin()] for g in studies]
    print(f"\n## {panel} / {model} (n = {len(studies)} studies). delta = two-stage pick minus CE pick; "
          "MAPE/|bias| lower better, Spearman higher better (* = CI excludes 0)")
    for m in [0.005, 0.01, 0.02, 0.05]:
        sizes = [(g.val_loss <= g.val_loss.min()*(1+m)).sum() for g in studies]
        for cn, col in crit.items():
            picks = []
            for g in studies:
                s = g[g.val_loss <= g.val_loss.min()*(1+m)]
                picks.append(s.loc[s[col].idxmax() if col == "val_spearman" else s[col].idxmin()])
            out = [f"{t} {ci(effect(np.array([p[c] for p in picks]), np.array([b[c] for b in base]), paired=True, metric=t, panel=panel))}"
                   for t, c in T.items()]
            print(f"  m={m:5.1%} shortlist median {np.median(sizes):4.1f}  {cn:13s} " + "  ".join(out))
