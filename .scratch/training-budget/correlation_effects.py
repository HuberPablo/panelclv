"""The mechanism correlations behind `docs/training-budget.md` §14.2 and §15.3.

The criterion-against-holdout tables (§14.1, §14.3, §15.3) come from
`selection_analysis.py`. This script covers what explains them: within a study, how
validation CE relates to the validation-window MAPE and to the spread of the holdout
forecast, and the holdout outcome by quartile of a trial's own validation CE.

Each correlation is one statistic per study (the study is the replication, its trials are
not), tested against 0 with `panelclv.evaluation.effects.effect(rhos, zeros,
paired=True)` — the protocol's "one statistic per replication" case. Each panel and model
is analysed separately; nothing is pooled across them into one bootstrap. The quartile
table is descriptive (no interval) and pools the two models of a panel.

    PYTHONPATH=src .../python .scratch/training-budget/correlation_effects.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from panelclv.evaluation.effects import effect

REPO = Path(__file__).resolve().parents[2]

files = sorted((REPO / "Studies").glob("selection_rescore__*/selection_rescore.csv"))
d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
d["panel"] = d.suite.str.split("__").str[2]

PAIRS = {
    "val CE vs validation MAPE (in-window)": ("val_loss", "val_mape_aggregate"),
    "spread of holdout forecast vs holdout MAPE": ("hold_pred_sd", "hold_mape_aggregate"),
    "val CE vs spread of holdout forecast": ("val_loss", "hold_pred_sd"),
}

print("## §14.2 — mechanism correlations, one rank correlation per study\n")
print("| panel | model | n studies | relationship | mean rho | 95% CI | supported |")
print("| --- | --- | ---: | --- | ---: | :---: | :---: |")
for (panel, model), dp in d.groupby(["panel", "model"]):
    studies = [g for _, g in dp.groupby("suite") if len(g) >= 10]
    for label, (x, y) in PAIRS.items():
        rho = np.array([g[x].corr(g[y], method="spearman") for g in studies], float)
        e = effect(rho, np.zeros_like(rho), paired=True, metric=label, panel=panel)
        print(f"| {panel} | {model} | {e.n_b} | {label} | {e.delta:+.3f} | "
              f"{e.lo:+.3f} to {e.hi:+.3f} | {'yes' if e.supported else 'no'} |")

print("\n## §14.2 / §15.3 — holdout outcome by quartile of a trial's own val CE "
      "(descriptive, both models)\n")
for panel, dp in d.groupby("panel"):
    studies = [g for _, g in dp.groupby("suite") if len(g) >= 10]
    q = pd.concat([g.assign(q=pd.qcut(g.val_loss.rank(method="first"), 4,
                                      labels=["best CE", "2nd", "3rd", "worst CE"]))
                   for g in studies])
    cols = ["hold_bias_percent", "hold_mape_aggregate", "hold_spearman", "hold_pred_sd"]
    print(f"### {panel} ({len(studies)} studies)\n")
    print(q.groupby("q", observed=True)[cols].mean().round(3).to_string(), "\n")
