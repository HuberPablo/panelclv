"""Holdout score of each criterion's pick minus the expected score of a random trial (the
study's trial mean). `docs/model-selection.md` §8, S1, S4.

Both numbers come from the same study's trials, so the study is the unit and the pair
(pick_i, random_i) is resampled together: `effect(pick, random, paired=True)`
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
# criterion -> (column, sign); the pick is the argmin of sign * column. None = |val bias|.
C = {"val CE": ("val_loss", 1), "roll MAPE": ("val_mape_aggregate", 1), "roll |bias|": (None, 1),
     "roll Spearman": ("val_spearman", -1), "best epoch": ("best_epoch", -1)}
T = {"MAPE": "hold_mape_aggregate", "|bias|": "abs_bias", "Spearman": "hold_spearman"}
for (panel, model), dp in d.groupby(["panel", "model"]):
    studies = [g for _, g in dp.groupby("suite") if len(g) >= 10]
    rand = {t: np.array([g[c].mean() for g in studies]) for t, c in T.items()}
    print(f"\n## {panel} / {model}, n = {len(studies)} studies: pick minus random-trial expectation (* = CI excludes 0)")
    print("   random trial means: " + ", ".join(f"{t} {v.mean():.3f}" for t, v in rand.items()))
    for cn, (col, sign) in C.items():
        picks = [g.loc[(g.val_bias_percent.abs() if col is None else sign * g[col]).idxmin()] for g in studies]
        out = [f"{t} {ci(effect(np.array([p[c] for p in picks]), rand[t], paired=True, metric=t, panel=panel))}"
               for t, c in T.items()]
        print(f"   {cn:13s} " + "  ".join(out))
