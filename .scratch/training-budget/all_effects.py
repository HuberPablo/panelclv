"""Every comparison the documents make, recomputed under the standard.

`docs/training-budget.md` "How claims are made": one metric per claim, effects reported as
Δ of condition means with a 95% bootstrap CI, supported when the interval excludes zero,
the measured refit floor printed beside it for magnitude, and nothing pooled across panels.

This regenerates the numbers for every table in:

  * training-budget.md §6 (what explains the collapse), §15.1, §15.2, §15.3
  * benchmarks-real-panels.md (the four-panel caveat table)
  * insights-cluster-ablation.md §5.1.1
  * absorbing-death-state.md §3.3 (the superseding note)

Run from the repo root with the project venv:
    PYTHONPATH=src .../python .scratch/training-budget/all_effects.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from effects import REFIT_FLOOR, effect                                  # noqa: E402

F = pd.read_csv(HERE / "results" / "factorial.csv")
PANELS = ["cdnow", "electronics", "gift", "multichannel"]
PNBD = {"cdnow": 0.450, "electronics": 0.297, "gift": 0.383, "multichannel": 0.189}


def cell(panel, model, training, cluster, metric):
    g = F[(F.panel == panel) & (F.model == model)
          & (F.training == training) & (F.cluster == cluster)]
    return g.bias_percent.abs() if metric == "abs_bias" else g[metric]


def contrasts(metric):
    """The four contrasts the 2x2 exists to make, per panel and model."""
    rows = []
    for panel in PANELS:
        for model in ("ValendinLSTM", "LSTM"):
            c = lambda t, k: cell(panel, model, t, k, metric)          # noqa: E731
            for name, b, a in (
                ("label alone", ("archive", "kmeans_8"), ("archive", "no_cluster")),
                ("floor alone", ("floored", "no_cluster"), ("archive", "no_cluster")),
                ("floor | label", ("floored", "kmeans_8"), ("archive", "kmeans_8")),
                ("label | floor", ("floored", "kmeans_8"), ("floored", "no_cluster")),
            ):
                e = effect(c(*b), c(*a), metric if metric != "abs_bias" else "bias_percent",
                           panel)
                rows.append(dict(panel=panel, model=model, contrast=name,
                                 mean_a=e.mean_a, mean_b=e.mean_b, delta=e.delta,
                                 lo=e.lo, hi=e.hi, supported=e.supported, floor=e.floor))
    return pd.DataFrame(rows)


def show(df, title, fmt="{:+.3f}"):
    print(f"\n### {title}\n")
    print("| panel | model | contrast | from | to | Δ | 95% CI | supported | floor |")
    print("| --- | --- | --- | ---: | ---: | ---: | :---: | :---: | ---: |")
    for _, r in df.iterrows():
        print(f"| {r.panel} | {r.model} | {r.contrast} | {r.mean_a:.4g} | {r.mean_b:.4g} | "
              f"{fmt.format(r.delta)} | {fmt.format(r.lo)} to {fmt.format(r.hi)} | "
              f"{'yes' if r.supported else 'no'} | {r.floor:.4g} |")


sp = contrasts("spearman")
show(sp, "Discrimination — Spearman (family U)")
sp.to_csv(HERE / "results" / "effects_spearman.csv", index=False)

mp = contrasts("mape_aggregate")
show(mp, "Level — MAPE (family U)", fmt="{:+.1f}")
mp.to_csv(HERE / "results" / "effects_mape.csv", index=False)

# --- the best cell against the statistical benchmark ------------------------------
print("\n### Best cell against Pareto/NBD — Spearman\n")
print("| panel | model | best cell | 95% CI | Pareto/NBD | reading |")
print("| --- | --- | ---: | :---: | ---: | --- |")
for panel in PANELS:
    for model in ("ValendinLSTM", "LSTM"):
        v = cell(panel, model, "floored", "kmeans_8", "spearman")
        r = bootstrap((np.asarray(v),), np.mean, n_resamples=10000, random_state=0)
        lo, hi = r.confidence_interval
        p = PNBD[panel]
        reading = ("above" if lo > p else "below" if hi < p else "interval contains it")
        print(f"| {panel} | {model} | {v.mean():.3f} | {lo:.3f} to {hi:.3f} | {p:.3f} "
              f"| {reading} |")

# --- §6 / §15.3: correlations, with intervals over studies ------------------------
print("\n### Archive correlations (§6) — bootstrap over studies\n")
sc = pd.read_csv(HERE / "results" / "spearman_electronics.csv")
hp = pd.read_csv(HERE / "results" / "hparams_electronics.csv")
j = sc.merge(hp[["suite", "study", "model", "updates", "best_epoch"]],
             on=["suite", "study", "model"], how="inner", suffixes=("", "_h"))
# The PRE-EXPERIMENT archive only. Family T, U and V suites now sit in `Studies/` too,
# and including them would compute an archive correlation over the experiments the
# document draws its conclusions from.
MINE = ("training_budget__", "factorial__", "selection_rescore__")
nc = j[(j.model == "ValendinLSTM") & ~j.suite.str.startswith(MINE)]
nc = nc[~nc.suite.str.contains("kmeans")]
print("| relationship | n studies | rho | 95% CI |")
print("| --- | ---: | ---: | :---: |")
for col in ("updates", "best_epoch"):
    x, y = nc[col].to_numpy(float), nc.spearman.to_numpy(float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    stat = lambda a, b: pd.Series(a).corr(pd.Series(b), method="spearman")  # noqa: E731
    r = bootstrap((x[ok], y[ok]), stat, n_resamples=5000, paired=True,
                  random_state=0, vectorized=False)
    print(f"| {col} vs holdout Spearman (no label) | {ok.sum()} | {stat(x[ok], y[ok]):+.3f} "
          f"| {r.confidence_interval.low:+.3f} to {r.confidence_interval.high:+.3f} |")
