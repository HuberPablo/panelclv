"""Mean and 95% t-interval over panels from results/per_study.csv: per tree and rate
(churn pooled, n = 40) and per tree and rate x churn cell (n = 10). Writes the two CSVs
and results/tables.md, the markdown the insights doc's Results section is pasted from."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(".scratch/synthetic-grid/results")
d = pd.read_csv(OUT / "per_study.csv")
# The two Transformer ar_unbounded forecasts that disagree with their stored results.
d = d[(d.mape - d.stored_mape).abs() / d.stored_mape < 0.05]
METRICS = ["rmse", "bias", "mape", "spearman", "ce"]
ORDER = [("ParetoNBD", "-")] + [(m, f"{a}-{c}") for m in ("LSTM", "Transformer")
         for a in ("ar_bounded", "no_ar", "ar_unbounded") for c in ("no_cluster", "kmeans_8")]


def summarise(keys):
    rows = []
    for key, g in d.groupby(keys):
        row = dict(zip(keys, key)) | {"n": len(g)}
        for m in METRICS:
            v = g[m].dropna()
            if len(v) < 2:
                continue
            half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            row |= {m: v.mean(), f"{m}_lo": v.mean() - half, f"{m}_hi": v.mean() + half}
        rows.append(row)
    return pd.DataFrame(rows)


by_rate = summarise(["model", "arm", "rate"])
by_cell = summarise(["model", "arm", "rate", "churn"])
by_rate.to_csv(OUT / "by_rate.csv", index=False)
by_cell.to_csv(OUT / "by_cell.csv", index=False)

# Share of panels where the tree's MAPE beats Pareto/NBD's on the same panel.
pn = d[d.model == "ParetoNBD"].set_index(["rate", "churn", "dataset"]).mape
d["wins"] = d.mape.values < pn.reindex(pd.MultiIndex.from_frame(d[["rate", "churn", "dataset"]])).values

FMT = {"rmse": "{:.2f}", "bias": "{:+.0f}", "mape": "{:.0f}", "spearman": "{:.2f}", "ce": "{:.3f}"}


def cell(r, m):
    if m not in r or pd.isna(r[m]):
        return "—"
    def f(x):
        t = FMT[m].format(x)
        return "0" if t in ("-0", "+0") else t
    return f"{f(r[m])} [{f(r[m + '_lo'])}, {f(r[m + '_hi'])}]".replace("-", "−")


def panels(n):
    return f"{n.max()} panels" if n.min() == n.max() else f"{n.min()}–{n.max()} panels"


def name(model, arm):
    if model == "ParetoNBD":
        return "**Pareto/NBD** | —"
    a, c = arm.split("-")
    return f"{model} | `{a}`" + (" + `kmeans_8`" if c == "kmeans_8" else "")


def table(frame, wins=None):
    head = "| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE |" + (" Beats P/NBD |" if wins is not None else "")
    sep = "| --- | --- | --- | --- | --- | --- | --- |" + (" ---: |" if wins is not None else "")
    lines = [head, sep]
    for model, arm in ORDER:
        r = frame[(frame.model == model) & (frame.arm == arm)]
        if r.empty:
            continue
        r = r.iloc[0]
        line = f"| {name(model, arm)} | " + " | ".join(cell(r, m) for m in METRICS) + " |"
        if wins is not None:
            line += " — |" if model == "ParetoNBD" else f" {wins[(model, arm)]:.0%} |"
        lines.append(line)
    return "\n".join(lines)


md = []
for rate in sorted(d.rate.unique()):
    w = d[d.rate == rate].groupby(["model", "arm"]).wins.mean()
    n = by_rate[by_rate.rate == rate].n
    md.append(f"#### Rate {rate:.2f}, churn pooled ({panels(n)})\n\n"
              + table(by_rate[by_rate.rate == rate], w))
for rate in sorted(d.rate.unique()):
    for churn in sorted(d.churn.unique()):
        f = by_cell[(by_cell.rate == rate) & (by_cell.churn == churn)]
        w = d[(d.rate == rate) & (d.churn == churn)].groupby(["model", "arm"]).wins.mean()
        md.append(f"##### Rate {rate:.2f}, churn {churn:.0%} ({panels(f.n)})\n\n"
                  + table(f, w))
(OUT / "tables.md").write_text("\n\n".join(md) + "\n")
