"""Which criterion orders a study's trials the way the holdout does? (§14.1, §14.3, §15.3)

Reads the per-trial tables `scripts/run_selection_rescore.py` writes and computes, for
every study, one rank correlation between each candidate criterion and each holdout
target. The study (one complete Optuna search) is the replication; its trials are not.

Every criterion is stated so that LOWER IS BETTER, and every target likewise, so a
POSITIVE correlation means the criterion ranks trials correctly.

Two claims per (criterion, target), both through `panelclv.evaluation.effects.effect`
(`docs/statistical-protocol.md`):

  * does the criterion rank trials correctly at all?  One statistic per study, tested
    against 0: ``effect(rhos, zeros, paired=True)``;
  * does it rank them better than validation CE?  Both criteria are scored on the same
    studies' trials, so the pair (rho_candidate_i, rho_CE_i) shares study i:
    ``effect(rho_candidate, rho_CE, paired=True)``.

Each panel and each model is analysed separately: LSTM and ValendinLSTM are distinct
conditions, and panels are never pooled into one bootstrap.

    PYTHONPATH=src .../python .scratch/training-budget/selection_analysis.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from panelclv.evaluation.effects import effect

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

files = sorted((REPO / "Studies").glob("selection_rescore__*/selection_rescore.csv"))
d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
d["panel"] = d.suite.str.split("__").str[2]
print(f"{len(d):,} trials from {d.suite.nunique()} studies\n")

CANDIDATES = {
    "val CE (status quo)":     lambda x: x.val_loss,
    "val rollout MAPE":        lambda x: x.val_mape_aggregate,
    "val rollout |bias|":      lambda x: x.val_bias_percent.abs(),
    "val rollout Spearman":    lambda x: -x.val_spearman,
    "composite (3 ranks)":     lambda x: (x.val_mape_aggregate.rank()
                                          + x.val_bias_percent.abs().rank()
                                          + (-x.val_spearman).rank()),
    # Not a validation score at all: how long the trial trained (§1, §5).
    "best epoch (longer=better)": lambda x: -x.best_epoch,
}
TARGETS = {
    "holdout MAPE":     lambda x: x.hold_mape_aggregate,
    "holdout |bias|":   lambda x: x.hold_bias_percent.abs(),
    "holdout Spearman": lambda x: -x.hold_spearman,
}
BASE = "val CE (status quo)"


def ci(e) -> str:
    return f"{e.delta:+.3f} | {e.lo:+.3f} to {e.hi:+.3f} | {'yes' if e.supported else 'no'}"


rows = []
for (panel, model), dp in d.groupby(["panel", "model"]):
    # One rank correlation per study; trials are only ever compared within a study.
    studies = [g for _, g in dp.groupby("suite") if len(g) >= 10]
    n = len(studies)
    print(f"## {panel} / {model}   (n = {n} studies, {sum(map(len, studies))} trials)\n")
    print("| target | criterion | mean rho | 95% CI | supported | Δ vs val CE | 95% CI of Δ "
          "| supported |")
    print("| --- | --- | ---: | :---: | :---: | ---: | :---: | :---: |")
    for tname, tf in TARGETS.items():
        rho = {c: np.array([cf(g).corr(tf(g), method="spearman") for g in studies], float)
               for c, cf in CANDIDATES.items()}
        for cname, v in rho.items():
            own = effect(v, np.zeros_like(v), paired=True, metric=tname, panel=panel)
            vs = None if cname == BASE else effect(v, rho[BASE], paired=True,
                                                   metric=tname, panel=panel)
            print(f"| {tname} | {cname} | {ci(own)} | "
                  + ("— | — | — |" if vs is None else f"{ci(vs)} |"))
            rows.append(dict(panel=panel, model=model, target=tname, criterion=cname,
                             n_studies=own.n_b, mean_rho=own.delta, lo=own.lo, hi=own.hi,
                             supported=own.supported,
                             delta_vs_ce=np.nan if vs is None else vs.delta,
                             delta_lo=np.nan if vs is None else vs.lo,
                             delta_hi=np.nan if vs is None else vs.hi,
                             delta_supported=None if vs is None else vs.supported))
    print()

pd.DataFrame(rows).to_csv(RESULTS / "selection_criteria.csv", index=False)

# §14.3's composite claim: is averaging three ranks worse than the single rollout
# criterion that matches the target? Same studies, so paired: composite minus matching.
MATCH = {"holdout MAPE": "val rollout MAPE", "holdout |bias|": "val rollout |bias|",
         "holdout Spearman": "val rollout Spearman"}
print("## composite minus the matching single criterion (paired over studies)\n")
print("| panel | model | n | target | Δ rho | 95% CI | supported |")
print("| --- | --- | ---: | --- | ---: | :---: | :---: |")
for (panel, model), dp in d.groupby(["panel", "model"]):
    studies = [g for _, g in dp.groupby("suite") if len(g) >= 10]
    for tname, tf in TARGETS.items():
        r = {c: np.array([CANDIDATES[c](g).corr(tf(g), method="spearman") for g in studies])
             for c in ("composite (3 ranks)", MATCH[tname])}
        e = effect(r["composite (3 ranks)"], r[MATCH[tname]], paired=True, metric=tname,
                   panel=panel)
        print(f"| {panel} | {model} | {e.n_b} | {tname} | {ci(e)} |")
