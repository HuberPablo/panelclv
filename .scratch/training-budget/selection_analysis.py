"""Which criterion orders a study's trials the way the holdout does?

Reads the per-trial tables `scripts/run_selection_rescore.py` writes, computes one rank
correlation per study for every candidate criterion, and tests each candidate against the
status quo (validation cross-entropy) with a paired Wilcoxon over studies.

Every criterion is stated so that LOWER IS BETTER, and every target likewise, so a
POSITIVE correlation means the criterion ranks trials correctly.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

files = sorted((REPO / "Studies").glob("selection_rescore__*/selection_rescore.csv"))
d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print(f"{len(d):,} trials from {d.suite.nunique()} studies, "
      f"{d.model.nunique()} models\n")

CANDIDATES = {
    "val CE (status quo)":     lambda x: x.val_loss,
    "val rollout MAPE":        lambda x: x.val_mape_aggregate,
    "val rollout |bias|":      lambda x: x.val_bias_percent.abs(),
    "val rollout Spearman":    lambda x: -x.val_spearman,
    "composite (3 ranks)":     lambda x: (x.val_mape_aggregate.rank()
                                          + x.val_bias_percent.abs().rank()
                                          + (-x.val_spearman).rank()),
    # Not a validation score at all: how long the trial trained. `docs/training-budget.md`
    # §1 and the archive both suggest it carries signal the loss does not.
    "best epoch (longer=better)": lambda x: -x.best_epoch,
}
TARGETS = {
    "holdout MAPE":     lambda x: x.hold_mape_aggregate,
    "holdout |bias|":   lambda x: x.hold_bias_percent.abs(),
    "holdout Spearman": lambda x: -x.hold_spearman,
}

rows = []
for tname, tf in TARGETS.items():
    per_study = {c: [] for c in CANDIDATES}
    for suite, g in d.groupby("suite"):
        if len(g) < 10:
            continue
        for cname, cf in CANDIDATES.items():
            per_study[cname].append(cf(g).corr(tf(g), method="spearman"))
    base = np.array(per_study["val CE (status quo)"], float)
    print(f"### {tname}   (n = {len(base)} studies)\n")
    print("| criterion | mean rho | 95% CI | beats val CE | paired Wilcoxon p |")
    print("| --- | ---: | :---: | ---: | ---: |")
    for cname, vals in per_study.items():
        v = np.array(vals, float)
        ok = ~np.isnan(v) & ~np.isnan(base)
        se = np.nanstd(v, ddof=1) / np.sqrt(ok.sum())
        ci = f"{np.nanmean(v) - 1.96 * se:+.3f} to {np.nanmean(v) + 1.96 * se:+.3f}"
        if cname == "val CE (status quo)":
            print(f"| **{cname}** | **{np.nanmean(v):+.3f}** | {ci} | — | — |")
        else:
            p = wilcoxon(v[ok], base[ok]).pvalue
            print(f"| {cname} | {np.nanmean(v):+.3f} | {ci} | "
                  f"{int((v[ok] > base[ok]).sum())}/{ok.sum()} | {p:.1e} |")
        rows.append(dict(target=tname, criterion=cname, mean_rho=np.nanmean(v),
                         n_studies=int(ok.sum())))
    print()

pd.DataFrame(rows).to_csv(RESULTS / "selection_criteria.csv", index=False)

# Is the status-quo criterion worse than useless, or just useless? A one-sample test of
# each mean correlation against zero answers it.
print("### Is validation cross-entropy worse than a coin toss?\n")
print("| target | mean rho | p (vs 0) | reading |")
print("| --- | ---: | ---: | --- |")
for tname, tf in TARGETS.items():
    v = np.array([tf(g).corr(CANDIDATES["val CE (status quo)"](g), method="spearman")
                  for _, g in d.groupby("suite") if len(g) >= 10], float)
    v = v[~np.isnan(v)]
    p = wilcoxon(v).pvalue
    verdict = ("actively misleading" if np.mean(v) < 0 and p < 0.05
               else "no signal" if p >= 0.05 else "ranks correctly")
    print(f"| {tname} | {np.mean(v):+.3f} | {p:.1e} | {verdict} |")
