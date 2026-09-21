"""Family U: do the training budget and the cluster label add, or overlap?

Reads the per-replication scores `scripts/run_factorial.py --report` writes and tests the
four comparisons the 2x2 exists to make, per panel and model, 20 vs 20 Mann-Whitney.
"""
from pathlib import Path

import pandas as pd
from scipy.stats import mannwhitneyu

RESULTS = Path(__file__).resolve().parent / "results"
d = pd.read_csv(RESULTS / "factorial.csv")
d["abs_bias"] = d.bias_percent.abs()


def cell(g, training, cluster, metric):
    return g[(g.training == training) & (g.cluster == cluster)][metric]


def test(g, metric, a, b):
    """mean(a), mean(b), p — a and b are (training, cluster) pairs."""
    x, y = cell(g, *a, metric), cell(g, *b, metric)
    return x.mean(), y.mean(), mannwhitneyu(x, y).pvalue


for metric, arrow in (("spearman", "higher better"), ("mape_aggregate", "lower better")):
    print(f"\n## {metric} ({arrow})\n")
    print("| panel | model | floor alone | p | label alone | p | floor ON TOP of label | p |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for (panel, model), g in d.groupby(["panel", "model"]):
        base, _, _ = test(g, metric, ("archive", "no_cluster"), ("archive", "no_cluster"))
        _, floor, p_floor = test(g, metric, ("archive", "no_cluster"),
                                 ("floored", "no_cluster"))
        _, label, p_label = test(g, metric, ("archive", "no_cluster"),
                                 ("archive", "kmeans_8"))
        _, both, p_both = test(g, metric, ("archive", "kmeans_8"), ("floored", "kmeans_8"))
        print(f"| {panel} | {model} | {base:.3f}→{floor:.3f} | {p_floor:.1e} | "
              f"{base:.3f}→{label:.3f} | {p_label:.1e} | "
              f"{label:.3f}→{both:.3f} | {p_both:.1e} |")

print("\n## Does the crossed cell beat the better single lever? (spearman)\n")
print("| panel | model | best single | crossed | p |")
print("| --- | --- | ---: | ---: | ---: |")
for (panel, model), g in d.groupby(["panel", "model"]):
    singles = {("archive", "kmeans_8"): cell(g, "archive", "kmeans_8", "spearman"),
               ("floored", "no_cluster"): cell(g, "floored", "no_cluster", "spearman")}
    best_key = max(singles, key=lambda k: singles[k].mean())
    crossed = cell(g, "floored", "kmeans_8", "spearman")
    p = mannwhitneyu(crossed, singles[best_key]).pvalue
    print(f"| {panel} | {model} | {singles[best_key].mean():.3f} ({best_key[0]}/{best_key[1]}) "
          f"| {crossed.mean():.3f} | {p:.2f} |")

print("\n## Where the floor HURTS on level (mape_aggregate)\n")
print("| panel | model | cluster | archive | floored | p |")
print("| --- | --- | --- | ---: | ---: | ---: |")
for (panel, model), g in d.groupby(["panel", "model"]):
    for cl in ("no_cluster", "kmeans_8"):
        a, b, p = test(g, "mape_aggregate", ("archive", cl), ("floored", cl))
        if b > a and p < 0.05:
            print(f"| {panel} | {model} | {cl} | {a:.1f} | **{b:.1f}** | {p:.1e} |")
