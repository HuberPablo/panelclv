"""Mean and 95% t-interval over panels from results/per_study.csv: per tree and rate
(churn pooled, n = 40) and per tree and rate x churn cell (n = 10). Writes the two CSVs
and results/tables.md, the markdown the insights doc's Results section is pasted from. In
those tables each column's best tree is bold and every tree tied with it is underlined."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(".scratch/synthetic-grid/results")
d = pd.read_csv(OUT / "per_study.csv")
# The two Transformer ar_unbounded forecasts that disagree with their stored results.
d = d[(d.mape - d.stored_mape).abs() / d.stored_mape < 0.05]
full = d[~d.arm.isin(["true_season", "no_ar-no_cluster-small_search"])]
# The Results tables cover the 13 trees of the 1,000-customer grid only; the cohort-size
# tables at the end also read the 3,000-customer grid from `full`.
d = full[full.cohort == "n1000"].copy()
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

# Per column, the best tree in a table and the trees statistically tied with it, all on the
# table's own panels. Best is the lowest mean RMSE, per-panel |bias|, MAPE or Val. CE, the
# highest mean Spearman, or the highest "Beats P/NBD" share. Tied means a paired test against
# the best gives p >= 0.05, uncorrected, as in the ranking tables' "p vs rank 1": Wilcoxon
# signed-rank for the five metrics, exact McNemar for the win shares, which are per-panel
# yes/no outcomes. A win column in which no tree wins a single panel is left unmarked.
MARKED = {"rmse": 1, "bias": 1, "mape": 1, "spearman": -1, "ce": 1}


def marks(sub, columns=(*MARKED, *WINS), tree=("model", "arm")):
    """{(*tree, column): "best" | "tie"} over the panels in `sub`, a tree being one value of
    the `tree` columns. Trees from different cohorts ran on different panels, so they are
    compared unpaired, by Mann-Whitney."""
    out = {}
    for m in columns:
        # Signed so that lower is better; a win becomes -1, a loss 0.
        v = (MARKED.get(m, -1) * (sub[m].abs() if m == "bias" else sub[m])).astype(float)
        # Rows are panels, columns trees. Pareto/NBD has no CE (NaN, dropped by the pivot)
        # and does not compete in the win columns.
        trees = sub.assign(v=v)
        if m in WINS:
            trees = trees[trees.model != "ParetoNBD"]
        panel = trees.pivot_table(index=["rate", "churn", "dataset"], columns=list(tree),
                                  values="v")
        best = panel.mean().idxmin()
        if m in WINS and panel[best].mean() == 0:
            continue
        out[(*best, m)] = "best"
        for other in panel.columns.drop(best):
            pair = panel[[best, other]].dropna()
            if "cohort" in tree and best[tree.index("cohort")] != other[tree.index("cohort")]:
                p = stats.mannwhitneyu(panel[other].dropna(), panel[best].dropna()).pvalue
            elif m in WINS:
                # Discordant panels: the best wins where the tree loses, and the reverse.
                b = int(((pair[best] < 0) & (pair[other] == 0)).sum())
                c = int(((pair[best] == 0) & (pair[other] < 0)).sum())
                p = stats.binomtest(b, b + c).pvalue if b + c else 1.0
            else:
                p = stats.wilcoxon(pair[other] - pair[best]).pvalue
            if p >= 0.05:
                out[(*other, m)] = "tie"
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
    return {"best": f"**{text}**", "tie": f"<ins>{text}</ins>"}.get(mark, text)


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
                f" {marked(f'{wins.loc[(model, arm), w]:.0%}', (mk or {}).get((model, arm, w)))} |" for w in WINS)
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


# The "Impact of cohort size" and "Impact of AR features" tables: per row a tree, per column
# a rate (churns pooled) or a churn (rates pooled), each cell RMSE / bias % / MAPE as means
# over the column's 40 panels, each of the three numbers marked within its column.
def triple(frame, rows, label, by):
    lines = []
    for value in sorted(frame[by].unique()):
        sub = frame[frame[by] == value]
        mk = marks(sub, ("rmse", "bias", "mape"), ("model", "arm", "cohort"))
        for key in rows:
            r = sub[(sub.model == key[0]) & (sub.arm == key[1]) & (sub.cohort == key[2])]
            nums = []
            for m in ("rmse", "bias", "mape"):
                t = FMT[m].format(r[m].mean())
                t = ("0" if t in ("-0", "+0") else t).replace("-", "−")
                nums.append(marked(t, mk.get((*key, m))))
            lines.append((key, " / ".join(nums)))
    cells = {}
    for key, text in lines:
        cells.setdefault(key, []).append(text)
    return "\n".join(f"| {label(k)} | " + " | ".join(v) + " |" for k, v in cells.items())


COHORT_ROWS = [(m, a, c) for m, a in [("ParetoNBD", "-"), ("LSTM", "no_ar-no_cluster"),
               ("LSTM", "ar_bounded-no_cluster")] for c in ("n1000", "n3000")]


def cohort_label(k):
    model = "Pareto/NBD" if k[0] == "ParetoNBD" else f"{k[0]} `{k[1].split('-')[0]}`"
    return f"{model} | {'1,000' if k[2] == 'n1000' else '3,000'}"


AR_ROWS = [("ParetoNBD", "-", "n1000")] + [(m, f"{a}-no_cluster", "n1000") for m in ("LSTM", "Transformer")
                                           for a in ("no_ar", "ar_unbounded", "ar_bounded")]
AR_NAME = {"no_ar": "none", "ar_unbounded": "unbounded", "ar_bounded": "bounded"}


def ar_label(k):
    return "Pareto/NBD | —" if k[0] == "ParetoNBD" else f"{k[0]} | {AR_NAME[k[1].split('-')[0]]}"


cohort = full[full.set_index(["model", "arm", "cohort"]).index.isin(COHORT_ROWS)]
ar = full[full.set_index(["model", "arm", "cohort"]).index.isin(AR_ROWS)]
(OUT / "impact_tables.md").write_text("\n\n".join([
    "#### Cohort size, by rate\n\n" + triple(cohort, COHORT_ROWS, cohort_label, "rate"),
    "#### Cohort size, by churn\n\n" + triple(cohort, COHORT_ROWS, cohort_label, "churn"),
    "#### AR features, by rate\n\n" + triple(ar, AR_ROWS, ar_label, "rate"),
    "#### AR features, by churn\n\n" + triple(ar, AR_ROWS, ar_label, "churn"),
]) + "\n")
