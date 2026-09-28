"""Mean and 95% t-interval over panels from results/per_study.csv: per tree and rate
(churn pooled, n = 40) and per tree and rate x churn cell (n = 10). Writes the two CSVs
and results/tables.md, the markdown the insights doc's Results section is pasted from. In
those tables each metric's best tree is bold and every tree tied with it carries a †."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(".scratch/synthetic-grid/results")
d = pd.read_csv(OUT / "per_study.csv")
# The two Transformer ar_unbounded forecasts that disagree with their stored results.
d = d[(d.mape - d.stored_mape).abs() / d.stored_mape < 0.05]
# The Results tables cover the 13 trees of the 1,000-customer grid only.
d = d[(d.cohort == "n1000") & ~d.arm.isin(["true_season", "no_ar-no_cluster-small_search"])]
METRICS = ["rmse", "bias", "mape", "spearman", "ce"]
ARMS = ["no_ar-no_cluster", "ar_unbounded-no_cluster", "ar_bounded-no_cluster",
        "ar_bounded-kmeans_8", "ar_unbounded-kmeans_8", "no_ar-kmeans_8"]
ORDER = [("ParetoNBD", "-")] + [(m, a) for m in ("LSTM", "Transformer") for a in ARMS]


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

# Whether the tree beats Pareto/NBD on the same panel: lower MAPE, smaller |bias|, higher
# Spearman. Averaged over a table's panels these are the three "Beats P/NBD" shares.
pn = d[d.model == "ParetoNBD"].set_index(["rate", "churn", "dataset"])
pn = pn.reindex(pd.MultiIndex.from_frame(d[["rate", "churn", "dataset"]]))
d["win_mape"] = d.mape.values < pn.mape.values
d["win_bias"] = d.bias.abs().values < pn.bias.abs().values
d["win_spearman"] = d.spearman.values > pn.spearman.values
WINS = ["win_mape", "win_bias", "win_spearman"]

# Per metric, the best tree in a table and the trees statistically tied with it. Best is the
# lowest mean over the table's panels of RMSE, MAPE or per-panel |bias|, or the highest mean
# Spearman. A tree is tied when a paired Wilcoxon against the best, on the same panels, gives
# p >= 0.05 (uncorrected, as in the ranking tables' "p vs rank 1").
MARKED = {"rmse": 1, "bias": 1, "mape": 1, "spearman": -1}


def marks(sub):
    """{(model, arm, metric): "best" | "tie"} over the panels in `sub`."""
    out = {}
    for m, sign in MARKED.items():
        panel = sub.assign(v=sign * (sub[m].abs() if m == "bias" else sub[m])).pivot_table(
            index=["rate", "churn", "dataset"], columns=["model", "arm"], values="v")
        best = panel.mean().idxmin()
        out[(*best, m)] = "best"
        for tree in panel.columns.drop(best):
            diff = (panel[tree] - panel[best]).dropna()
            if stats.wilcoxon(diff).pvalue >= 0.05:
                out[(*tree, m)] = "tie"
    return out


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


def marked(text, mark):
    return {"best": f"**{text}**", "tie": f"{text} †"}.get(mark, text)


def table(frame, wins=None, mk=None):
    head = "| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE |" + (
        " Beats P/NBD: MAPE | Beats P/NBD: \\|bias\\| | Beats P/NBD: Spearman |" if wins is not None else "")
    sep = "| --- | --- | --- | --- | --- | --- | --- |" + (" ---: | ---: | ---: |" if wins is not None else "")
    lines = [head, sep]
    for model, arm in ORDER:
        r = frame[(frame.model == model) & (frame.arm == arm)]
        if r.empty:
            continue
        r = r.iloc[0]
        line = f"| {name(model, arm)} | " + " | ".join(marked(cell(r, m), (mk or {}).get((model, arm, m))) for m in METRICS) + " |"
        if wins is not None:
            line += " — | — | — |" if model == "ParetoNBD" else "".join(
                f" {wins.loc[(model, arm), w]:.0%} |" for w in WINS)
        lines.append(line)
    return "\n".join(lines)


md = []
for rate in sorted(d.rate.unique()):
    w = d[d.rate == rate].groupby(["model", "arm"])[WINS].mean()
    n = by_rate[by_rate.rate == rate].n
    md.append(f"#### Rate {rate:.2f}, churn pooled ({panels(n)})\n\n"
              + table(by_rate[by_rate.rate == rate], w, marks(d[d.rate == rate])))
for rate in sorted(d.rate.unique()):
    for churn in sorted(d.churn.unique()):
        f = by_cell[(by_cell.rate == rate) & (by_cell.churn == churn)]
        w = d[(d.rate == rate) & (d.churn == churn)].groupby(["model", "arm"])[WINS].mean()
        md.append(f"##### Rate {rate:.2f}, churn {churn:.0%} ({panels(f.n)})\n\n"
                  + table(f, w, marks(d[(d.rate == rate) & (d.churn == churn)])))
(OUT / "tables.md").write_text("\n\n".join(md) + "\n")


# The three best trees per rate x churn cell, by MAPE (lowest) and by Spearman (highest).
def top3(metric, ascending):
    lines = [f"| Rate | Churn | Rank | Model | Arm | {'**MAPE**' if metric == 'mape' else 'MAPE'} | "
             f"{'**Spearman**' if metric == 'spearman' else 'Spearman'} | RMSE | Bias % | Val. CE | p vs rank 1 | Beats P/NBD: MAPE / \\|bias\\| / Spearman |",
             "| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |"]
    for (rate, churn), f in by_cell.groupby(["rate", "churn"]):
        best = f.sort_values(metric, ascending=ascending).head(3)
        # Paired Wilcoxon of each runner-up against the cell's leader, on the same panels.
        panel = d[(d.rate == rate) & (d.churn == churn)].pivot_table(
            index="dataset", columns=["model", "arm"], values=metric)
        leader = panel[(best.iloc[0].model, best.iloc[0].arm)]
        wins = d[(d.rate == rate) & (d.churn == churn)].groupby(["model", "arm"])[WINS].mean()
        for rank, (_, r) in enumerate(best.iterrows(), 1):
            model, arm = name(r.model, r.arm).split(" | ")
            vals = [cell(r, m) if m == metric else
                    ("—" if pd.isna(r.get(m)) else FMT[m].format(r[m]).replace("-", "−"))
                    for m in ("mape", "spearman", "rmse", "bias", "ce")]
            lead = f"{rate:.2f} | {churn:.0%}" if rank == 1 else " | "
            p = "—" if rank == 1 else f"{stats.wilcoxon((panel[(r.model, r.arm)] - leader).dropna()).pvalue:.3f}"
            beats = "—" if r.model == "ParetoNBD" else " / ".join(
                f"{wins.loc[(r.model, r.arm), w]:.0%}" for w in WINS)
            lines.append(f"| {lead} | {rank} | {model} | {arm} | " + " | ".join(vals) + f" | {p} | {beats} |")
    return "\n".join(lines)


(OUT / "top3_mape.md").write_text(top3("mape", True) + "\n")
(OUT / "top3_spearman.md").write_text(top3("spearman", False) + "\n")
