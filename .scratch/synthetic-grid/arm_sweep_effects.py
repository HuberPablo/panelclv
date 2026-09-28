"""The tests behind `docs/insights-arm-sweep.md` §4, §5 and §7, under the statistical protocol.

Reads results/per_study.csv (one row per panel and tree, recomputed from the stored
forecasts) and prints the markdown the doc quotes. Every comparison follows
`docs/statistical-protocol.md` through `panelclv.evaluation.effects.effect`:

* one replication is one generated panel, and a test runs inside one (rate x churn) cell of
  10 panels, never over panels of different cells;
* two trees fitted on the same panels are paired by panel;
* Δ = mean(B) - mean(A) with its 95% percentile-bootstrap interval, supported iff the
  interval excludes 0.

Across the 16 cells the result is described: the pooled mean over all 160 panels, and in
how many cells the change is supported in each direction.

    PYTHONPATH=src .../python .scratch/synthetic-grid/arm_sweep_effects.py
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from panelclv.evaluation.effects import effect

OUT = Path(".scratch/synthetic-grid/results")
d = pd.read_csv(OUT / "per_study.csv")
# The two Transformer ar_unbounded forecasts that disagree with their stored results.
d = d[(d.mape - d.stored_mape).abs() / d.stored_mape < 0.05]
d = d[d.cohort == "n1000"].copy()
d["abs_bias"] = d.bias.abs()
KEY = ["rate", "churn", "dataset"]
CELLS = list(product(sorted(d.rate.unique()), sorted(d.churn.unique())))


def tree(model, arm):
    return d[(d.model == model) & (d.arm == arm)].set_index(KEY)


def per_cell(a, b, metric):
    """Paired effect of B minus A in every cell, on the panels both trees have."""
    j = a[metric].to_frame("a").join(b[metric].rename("b"), how="inner")
    return {(r, c): effect(g.b, g.a, paired=True, metric=metric, panel=f"{r}/{c}")
            for (r, c), g in j.groupby(level=["rate", "churn"])}, len(j)


def m(x, f="{:+.1f}"):
    return f.format(x).replace("-", "−")


# --- §4: what the arm axis moved, on |bias| ---------------------------------------------
CONTRASTS = [
    ("`ar_bounded` vs `no_ar`", "no_ar-no_cluster", "ar_bounded-no_cluster"),
    ("`ar_unbounded` vs `no_ar`", "no_ar-no_cluster", "ar_unbounded-no_cluster"),
    ("`kmeans_8` vs none, under `no_ar`", "no_ar-no_cluster", "no_ar-kmeans_8"),
    ("`kmeans_8` vs none, under `ar_bounded`", "ar_bounded-no_cluster", "ar_bounded-kmeans_8"),
    ("`kmeans_8` vs none, under `ar_unbounded`", "ar_unbounded-no_cluster", "ar_unbounded-kmeans_8"),
]
print("## §4 — |bias %|, per cell, paired\n")
print("| contrast | model | panels | mean \\|bias\\| before → after (pooled) | cells: \\|bias\\| falls "
      "| cells: rises | cells: no clear difference | supported Δ, range |")
print("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
for label, arm_a, arm_b in CONTRASTS:
    for model in ("LSTM", "Transformer"):
        a, b = tree(model, arm_a), tree(model, arm_b)
        eff, n = per_cell(a, b, "abs_bias")
        sup = [e for e in eff.values() if e.supported]
        falls = sum(e.delta < 0 for e in sup)
        rises = sum(e.delta > 0 for e in sup)
        rng = "—" if not sup else f"{m(min(e.delta for e in sup))} to {m(max(e.delta for e in sup))}"
        j = a.abs_bias.to_frame("a").join(b.abs_bias.rename("b"), how="inner")
        print(f"| {label} | {model} | {n} | {j.a.mean():.1f} → {j.b.mean():.1f} | {falls} | {rises} "
              f"| {16 - len(sup)} | {rng} |")

# --- §5: which architecture is closer to the truth, per cell ----------------------------
print("\n## §5 — |LSTM bias| − |Transformer bias|, `ar_bounded-no_cluster`, per cell, paired\n")
eff, n = per_cell(tree("Transformer", "ar_bounded-no_cluster"), tree("LSTM", "ar_bounded-no_cluster"),
                  "abs_bias")
print(f"({n} panels.) Positive: the Transformer is closer to the truth. Bold: supported.\n")
print("| rate \\ churn | 0.2 | 0.4 | 0.6 | 0.8 |\n| --- | ---: | ---: | ---: | ---: |")
for rate in sorted(d.rate.unique()):
    row = []
    for churn in sorted(d.churn.unique()):
        e = eff[(rate, churn)]
        s = f"{m(e.delta)} [{m(e.lo, '{:+.0f}')}, {m(e.hi, '{:+.0f}')}]"
        row.append(f"**{s}**" if e.supported else s)
    print(f"| {rate:.2f} | " + " | ".join(row) + " |")
print(f"\nLSTM closer in {sum(e.supported and e.delta < 0 for e in eff.values())} cells, "
      f"Transformer closer in {sum(e.supported and e.delta > 0 for e in eff.values())}, "
      f"no clear difference in {sum(not e.supported for e in eff.values())}.")

# --- §7: dead leakage L_D, LSTM ar_bounded vs Pareto/NBD, per cell ----------------------
print("\n## §7 — dead leakage L_D, Pareto/NBD → LSTM `ar_bounded`, per cell, paired\n")
pn, lstm = tree("ParetoNBD", "-"), tree("LSTM", "ar_bounded-no_cluster")
eff, _ = per_cell(pn, lstm, "l_d")
j = pn.l_d.to_frame("a").join(lstm.l_d.rename("b"), how="inner")
wins = (j.b < j.a).groupby(level=["rate", "churn"]).sum()
print("| cell | Pareto/NBD | LSTM | LSTM better on | Δ [95% CI] | supported |")
print("| --- | ---: | ---: | ---: | :---: | :---: |")
for rate, churn in CELLS:
    e = eff[(rate, churn)]
    print(f"| rate {rate:.2f}, churn {churn:.1f} | {e.mean_a:.3f} | {e.mean_b:.3f} | "
          f"{int(wins[(rate, churn)])}/{e.n_a} | {m(e.delta, '{:+.3f}')} [{m(e.lo, '{:+.3f}')}, "
          f"{m(e.hi, '{:+.3f}')}] | {'yes' if e.supported else 'no'} |")
print(f"\nLSTM leaks less on {int((j.b < j.a).sum())} of {len(j)} panels.")
