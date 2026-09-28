"""Family U: do the training budget and the cluster label add, or overlap? (§15)

Reads the per-replication scores `scripts/run_factorial.py --report` writes
(`results/factorial.csv`, 2 models x 4 panels x 2 trainings x 2 inputs x 20 replications)
and asks the two questions `all_effects.py`'s four contrasts do not:

1. does the crossed cell (floored / kmeans_8) beat the better of the two single levers?
2. where does the floor change the level (MAPE), with and without the label?

Every comparison is `docs/statistical-protocol.md`'s: Δ = mean(B) − mean(A) with a 95%
percentile-bootstrap interval, via the package's `effect` (through this folder's shim,
which prints the panel's refit noise beside it). Every cell is 20 independent searches
or pinned trainings on unseeded weights, so cells share nothing: paired=False. Each panel
and model is its own test; nothing is pooled.

    PYTHONPATH=src python .scratch/training-budget/factorial_analysis.py
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from effects import effect, table                                        # noqa: E402

d = pd.read_csv(HERE / "results" / "factorial.csv")


def cell(g, training, cluster, metric):
    return g[(g.training == training) & (g.cluster == cluster)][metric]


# 1. The crossed cell against whichever single lever has the higher mean Spearman.
rows = []
for (panel, model), g in d.groupby(["panel", "model"]):
    singles = {k: cell(g, *k, "spearman")
               for k in (("archive", "kmeans_8"), ("floored", "no_cluster"))}
    best = max(singles, key=lambda k: singles[k].mean())
    e = effect(cell(g, "floored", "kmeans_8", "spearman"), singles[best], "spearman", panel)
    e.panel = f"{panel} / {model} (A = {best[0]}/{best[1]})"
    rows.append(e)
print("\n### Crossed cell against the better single lever — Spearman\n")
print(table(rows, label="panel / model"))

# 2. The floor's effect on MAPE, without and with the label.
for cluster in ("no_cluster", "kmeans_8"):
    rows = []
    for (panel, model), g in d.groupby(["panel", "model"]):
        e = effect(cell(g, "floored", cluster, "mape_aggregate"),
                   cell(g, "archive", cluster, "mape_aggregate"), "mape_aggregate", panel)
        e.panel = f"{panel} / {model}"
        rows.append(e)
    print(f"\n### Floor on level (A = archive, B = floored), inputs {cluster} — MAPE\n")
    print(table(rows, label="panel / model"))
