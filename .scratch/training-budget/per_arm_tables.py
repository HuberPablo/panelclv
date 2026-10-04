"""The §8.1 tables of insight-training-efficiency, one configuration at a time.

A configuration (a "cell") is panel × arm × model × input set; its 20 runs differ only in
search and training randomness. Results are kept apart by arm and by input set, because
the inputs change which hyperparameters the search picks.

Prints Markdown to stdout:
- per panel, the hyperparameters the search chose in each searched cell;
- per panel and arm, for every cell: the median run, the best and worst run by MAPE and
  the worst by Spearman, each with its hyperparameters and four metrics, plus how many of
  the cell's runs fall in the panel's worst 10% by MAPE, |bias| and Spearman.
Then the within-cell correlations of Finding 2, split by arm and input set.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
d = pd.read_csv(HERE / "results" / "worst_best_runs.csv")
d["absbias"] = d.bias_percent.abs()
d["label"] = np.where(d.family == "U", d.cluster, "no_cluster")

# The columns each cell's model reads (every winner's `selected_features`; identical
# across a cell's 20 runs, since no covariate subset was searched).
def inputs(r):
    base = {"LSTM": "count, week sin/cos", "ValendinLSTM": "count, week"}[r.model]
    if r.family == "T'" and r.model == "LSTM":
        base = "count"
    return base + (", label" if r.label == "kmeans_8" else "")

d["inputs"] = d.apply(inputs, axis=1)
d["fam"] = d.family.str.replace("'", "′")
PANELS = ["electronics", "multichannel", "cdnow", "gift"]
ARMS = ["archive", "floor50", "paper", "paper90", "floored"]

# A panel's worst 10% on each metric, ranked across all its runs.
for m, asc, flag in [("mape_aggregate", False, "badM"), ("absbias", False, "badB"),
                     ("spearman", True, "badS")]:
    k = d.groupby("panel").panel.transform(lambda s: round(len(s) * 0.1))
    d[flag] = d.groupby("panel")[m].rank(ascending=asc, method="first") <= k


def lr(x): return f"{x:.1e}".replace("e-0", "e-")
def wd(x): return "0" if x == 0 else f"{x:.0e}".replace("e-0", "e-")


def arch(r, median=False):
    if r.model == "ValendinLSTM":
        return "128*", "128*", "0*"
    if median:
        return r.hidden, r.dense, r.dropout
    return str(int(r.hidden)), str(int(r.dense)), f"{r.dropout:.2f}"


def run_row(kind, r):
    h, de, p = arch(r)
    return (f"| {kind} | {r.batch_size} | {lr(r.learning_rate)} | {wd(r.weight_decay)} "
            f"| {h} | {de} | {p} | {r.best_epoch} | {int(r.updates):,} | {r.val_ce:.4f} "
            f"| {r.bias_percent:+.1f} | {r.rmse:.4f} | {r.mape_aggregate:.1f} "
            f"| {r.spearman:+.3f} | {r.forecast_cv:.2f} |")


def median_row(g):
    r = g.iloc[0]
    if r.model == "ValendinLSTM":
        h, de, p = "128*", "128*", "0*"
    else:
        h = f"{int(g.hidden.mode()[0])}"
        de = f"{int(g.dense.mode()[0])}"
        p = f"{g.dropout.median():.2f}"
    b = g.batch_size.mode()[0]
    nb = (g.batch_size == b).sum()
    return (f"| **median of 20** | {b}{'' if nb == 20 else f' ({nb})'} "
            f"| {lr(g.learning_rate.median())} | {wd(g.weight_decay.median())} "
            f"| {h} | {de} | {p} | {g.best_epoch.median():g} | {int(g.updates.median()):,} "
            f"| {g.val_ce.median():.4f} | {g.bias_percent.median():+.1f} "
            f"| {g.rmse.median():.4f} | {g.mape_aggregate.median():.1f} "
            f"| {g.spearman.median():+.3f} | {g.forecast_cv.median():.2f} |")


HEAD = ("| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE "
        "| bias % | RMSE | MAPE | Spearman | CV |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
        "| ---: | ---: | ---: |")

out = []
for panel in PANELS:
    p = d[d.panel == panel]
    k = round(len(p) * 0.1)
    out.append(f"#### {panel} ({len(p)} runs; worst 10% = {k})\n")
    for arm in ARMS:
        a = p[p.arm == arm]
        if a.empty:
            continue
        floor = int(a.min_epochs.iloc[0])
        out.append(f"**`{arm}`** (epoch floor {floor})\n")
        for (fam, model, inp), g in a.groupby(["fam", "model", "inputs"], sort=False):
            out.append(
                f"*{fam} · {model} · inputs: {inp}* — in the panel's worst 10%: "
                f"{g.badM.sum()} by MAPE, {g.badB.sum()} by |bias|, "
                f"{g.badS.sum()} by Spearman\n")
            rows = [median_row(g),
                    run_row("best MAPE", g.loc[g.mape_aggregate.idxmin()]),
                    run_row("worst MAPE", g.loc[g.mape_aggregate.idxmax()]),
                    run_row("worst Spearman", g.loc[g.spearman.idxmin()])]
            out.append(HEAD + "\n" + "\n".join(rows) + "\n")
print("\n".join(out))

print("\n== What the search chose (searched arms)\n")
print("| panel | model | arm | inputs | batch (runs of 20) | lr, median [IQR] | dropout | best epoch | updates | MAPE | bias % | Spearman |")
print("| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
s = d[d.arm.isin(["archive", "floor50"])]
for (panel, model, fam, arm, inp), g in s.groupby(["panel", "model", "fam", "arm", "inputs"]):
    b = ", ".join(f"{k}: {v}" for k, v in g.batch_size.value_counts().items())
    drop = f"{g.dropout.median():.2f}" if g.dropout.notna().any() else "0*"
    print(f"| {panel} | {fam} · {model} | `{arm}` | {inp} | {b} "
          f"| {lr(g.learning_rate.median())} [{lr(g.learning_rate.quantile(.25))}, "
          f"{lr(g.learning_rate.quantile(.75))}] | {drop} | {g.best_epoch.median():g} "
          f"| {int(g.updates.median()):,} | {g.mape_aggregate.median():.1f} "
          f"| {g.bias_percent.median():+.1f} | {g.spearman.median():+.3f} |")

print("\n== Within-cell correlation with MAPE, by arm and input set\n")
SETTINGS = {"learning_rate": "learning rate", "batch_size": "batch size",
            "weight_decay": "weight decay", "dropout": "dropout", "hidden": "hidden width",
            "dense": "dense width", "best_epoch": "best epoch", "val_ce": "validation loss",
            "forecast_cv": "forecast CV (an outcome)"}
s = s.assign(group=np.where(s.arm == "floor50", "`floor50`, no label",
                            np.where(s.label == "kmeans_8", "`archive`, label",
                                     "`archive`, no label")))
groups = ["`archive`, no label", "`archive`, label", "`floor50`, no label"]
res = {}
for grp in groups:
    sg = s[s.group == grp]
    r = pd.DataFrame([{m: g.mape_aggregate.corr(g[m], method="spearman")
                       for m in SETTINGS if g[m].nunique() > 1}
                      for _, g in sg.groupby(["panel", "fam", "model"])]).reindex(columns=list(SETTINGS))
    res[grp] = (len(sg.groupby(["panel", "fam", "model"])), r)
print("| setting | " + " | ".join(f"{g} ({res[g][0]} cells)" for g in groups) + " |")
print("| --- |" + " --- |" * len(groups))
for m, name in SETTINGS.items():
    cells = []
    for g in groups:
        r = res[g][1][m]
        cells.append("—" if r.notna().sum() == 0 else
                     f"{r.median():+.2f} ({(r > 0.3).sum()} / {(r < -0.3).sum()} of {r.notna().sum()})")
    print(f"| {name} | " + " | ".join(cells) + " |")

print("\n== MAPE spread inside a cell: searched vs fixed-hyperparameter arms (median over cells)")
c = d.assign(fixed=d.arm.isin(["paper", "paper90", "floored"])).groupby(
    ["panel", "fam", "model", "arm", "label"]).agg(
    fixed=("fixed", "first"), sd=("mape_aggregate", "std"))
print(c.groupby(["panel", "fixed"]).sd.median().round(1))
