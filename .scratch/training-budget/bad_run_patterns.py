"""What the bad runs share: the numbers behind insight-training-efficiency §8.2.

Reads `results/worst_best_runs.csv` (one row per study, from `worst_best_runs.py`) and
prints, panel by panel:

1. how much of each metric's spread lies between configurations (cells) rather than
   between runs of the same configuration;
2. which cells the bad runs come from — a panel's worst 10% by MAPE, by |bias| and by
   Spearman;
3. within the searched arms (`archive`, `floor50`), how each searched setting correlates
   with MAPE across the 20 runs of a cell, against what chance gives at n = 20.

Descriptive only: the bad runs are picked by their outcome, so none of this is a
protocol claim.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
d = pd.read_csv(HERE / "results" / "worst_best_runs.csv")
d["absbias"] = d.bias_percent.abs()
d["label"] = np.where(d.family == "U", "/" + d.cluster, "")
d["cell"] = d.family + " · " + d.model + " · " + d.arm + d.label
METRICS = {"MAPE": "mape_aggregate", "|bias|": "absbias", "Spearman": "spearman"}
pd.set_option("display.width", 250)

print("== 1. Share of each metric's variance that lies between cells")
for panel, g in d.groupby("panel"):
    shares = {}
    for name, m in METRICS.items():
        within = g.groupby("cell")[m].var(ddof=0).mean()
        shares[name] = round(1 - within / g[m].var(ddof=0), 2)
    print(f"  {panel:13s}", shares)

print("\n== 2. Cells holding each panel's worst 10%")
for panel, g in d.groupby("panel"):
    k = round(len(g) * 0.1)
    bad = {
        "MAPE": g.mape_aggregate.rank(ascending=False, method="first") <= k,
        "|bias|": g.absbias.rank(ascending=False, method="first") <= k,
        "Spearman": g.spearman.rank(method="first") <= k,
    }
    print(f"\n-- {panel}: {len(g)} runs, worst {k}; MAPE & |bias| share "
          f"{(bad['MAPE'] & bad['|bias|']).sum()}, MAPE & Spearman share "
          f"{(bad['MAPE'] & bad['Spearman']).sum()}")
    counts = pd.DataFrame({n: g[b].cell.value_counts() for n, b in bad.items()})
    print(counts.fillna(0).astype(int).sort_values("MAPE", ascending=False))
    for n, b in bad.items():
        x = g[b]
        print(f"  worst {n:8s}: batch {x.batch_size.value_counts().to_dict()}, "
              f"best epoch {x.best_epoch.median()}, updates {x.updates.median()}, "
              f"CV {x.forecast_cv.median():.2f}, bias {x.bias_percent.median():+.1f}, "
              f"over-forecast {(x.bias_percent > 0).mean():.0%}")

print("\n== 3. Within-cell rank correlation with MAPE, searched arms only")
SETTINGS = ["learning_rate", "batch_size", "weight_decay", "dropout", "hidden", "dense",
            "best_epoch", "val_ce", "forecast_cv"]
rows = []
for cell, g in d[d.arm.isin(["archive", "floor50"])].groupby(["panel", "cell"]):
    rows.append({s: g.mape_aggregate.corr(g[s], method="spearman")
                 for s in SETTINGS if g[s].nunique() > 1})
r = pd.DataFrame(rows)
print(pd.DataFrame({"cells": r.notna().sum(), "median r": r.median().round(2),
                    "r > +0.3": (r > 0.3).sum(), "r < -0.3": (r < -0.3).sum()}))
rng = np.random.default_rng(0)
null = [pd.Series(rng.normal(size=20)).corr(pd.Series(rng.normal(size=20)), method="spearman")
        for _ in range(4000)]
print(f"chance: P(r > 0.3) at n = 20 is {np.mean(np.array(null) > 0.3):.3f} per tail")
batch_mode = d[d.arm.isin(["archive", "floor50"])].groupby(["panel", "cell"]).batch_size.agg(
    lambda s: s.value_counts().iloc[0])
print(f"runs on the cell's most common batch: median {batch_mode.median()} of 20, "
      f"min {batch_mode.min()}")
