"""The correlation-shaped claims (§14, §15.3), under the same standard as the rest.

A correlation is not a difference of condition means, so the standard adapts rather than
transfers: the statistic is the mean WITHIN-STUDY rank correlation, the unit of
resampling is the study, and the claim is supported when the 95% bootstrap interval
excludes zero. Where two criteria are compared, the effect is the paired difference of
their within-study correlations and the same rule applies.

Regenerates the tables in `docs/training-budget.md` §14.1, §14.2, §14.3 and §15.3.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load(panel: str) -> pd.DataFrame:
    fs = sorted((REPO / "Studies").glob(
        f"selection_rescore__*__{panel}__*/selection_rescore.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def ci(values: np.ndarray) -> tuple[float, float, float]:
    """Mean of a per-study statistic with its 95% bootstrap interval."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return float(np.mean(v)) if len(v) else np.nan, np.nan, np.nan
    r = bootstrap((v,), np.mean, n_resamples=10000, random_state=0)
    return float(v.mean()), float(r.confidence_interval.low), float(r.confidence_interval.high)


def per_study(d: pd.DataFrame, xf, yf) -> np.ndarray:
    """One rank correlation per study — trials are only ever compared within a study."""
    return np.array([xf(g).corr(yf(g), method="spearman")
                     for _, g in d.groupby("suite") if len(g) >= 10], float)


TARGETS = {"holdout MAPE": lambda x: x.hold_mape_aggregate,
           "holdout |bias|": lambda x: x.hold_bias_percent.abs(),
           "holdout Spearman": lambda x: -x.hold_spearman}
CANDIDATES = {
    "val CE (status quo)": lambda x: x.val_loss,
    "val rollout MAPE": lambda x: x.val_mape_aggregate,
    "val rollout |bias|": lambda x: x.val_bias_percent.abs(),
    "val rollout Spearman": lambda x: -x.val_spearman,
    "composite (3 ranks)": lambda x: (x.val_mape_aggregate.rank()
                                      + x.val_bias_percent.abs().rank()
                                      + (-x.val_spearman).rank()),
}

panels = {p: load(p) for p in ("electronics", "cdnow")}

print("## §14.1 / §15.3 — validation cross-entropy against the holdout\n")
print("| panel | studies | target | mean rho | 95% CI | supported |")
print("| --- | ---: | --- | ---: | :---: | :---: |")
for panel, d in panels.items():
    n = sum(1 for _, g in d.groupby("suite") if len(g) >= 10)
    for tname, tf in TARGETS.items():
        m, lo, hi = ci(per_study(d, CANDIDATES["val CE (status quo)"], tf))
        sup = "yes" if (lo > 0 or hi < 0) else "no"
        print(f"| {panel} | {n} | {tname} | {m:+.3f} | {lo:+.3f} to {hi:+.3f} | {sup} |")

print("\n## §14.3 — each candidate's own correlation, electronics\n")
print("| target | criterion | mean rho | 95% CI | supported |")
print("| --- | --- | ---: | :---: | :---: |")
d = panels["electronics"]
for tname, tf in TARGETS.items():
    for cname, cf in CANDIDATES.items():
        m, lo, hi = ci(per_study(d, cf, tf))
        sup = "yes" if (lo > 0 or hi < 0) else "no"
        print(f"| {tname} | {cname} | {m:+.3f} | {lo:+.3f} to {hi:+.3f} | {sup} |")

print("\n## §14.2 — the mechanism correlations, electronics\n")
PAIRS = {
    "val CE vs validation MAPE (in-window)": (lambda x: x.val_loss,
                                              lambda x: x.val_mape_aggregate),
    "spread of holdout forecast vs holdout MAPE": (lambda x: x.hold_pred_sd,
                                                   lambda x: x.hold_mape_aggregate),
    "val CE vs spread of holdout forecast": (lambda x: x.val_loss,
                                             lambda x: x.hold_pred_sd),
}
print("| relationship | mean rho | 95% CI | supported |")
print("| --- | ---: | :---: | :---: |")
for label, (xf, yf) in PAIRS.items():
    m, lo, hi = ci(per_study(d, xf, yf))
    sup = "yes" if (lo > 0 or hi < 0) else "no"
    print(f"| {label} | {m:+.3f} | {lo:+.3f} to {hi:+.3f} | {sup} |")
