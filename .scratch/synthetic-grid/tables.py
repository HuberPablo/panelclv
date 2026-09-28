"""The Results tables of `docs/insights-synthetic-grid.md`, from results/per_study.csv.

Writes results/by_cell.csv, results/by_rate.csv and the markdown the doc's Results,
top-3 and "Impact of ..." sections are pasted from (tables.md, top3_*.md,
impact_tables.md). Everything follows `docs/statistical-protocol.md`:

* A (rate x churn) cell holds 10 generated panels, and a panel is one replication. Every
  interval and every "tied with the best" mark is computed inside one cell, by
  `panelclv.evaluation.effects.effect`, and never over panels of different cells.
* A per-cell metric entry is the mean over the cell's panels with its 95% percentile
  bootstrap interval (the one-statistic form of `effect`: the panel values against 0).
* Per-cell marks: each column's best tree is **bold**. A tree is <ins>underlined</ins>
  when the paired interval of (tree - best) on the cell's panels contains 0, i.e. it is
  not clearly worse than the best at n = 10.
* Tables that pool cells (one rate with its four churn levels, the cohort-size and AR
  tables) are descriptive: means only, no interval, no marks.
* "Beats P/NBD" columns count the panels where the tree beats Pareto/NBD on that same
  panel. They are plain description and carry no marks.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from panelclv.evaluation.effects import effect

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
KEY = ["rate", "churn", "dataset"]


def mean_ci(v):
    """Mean of one cell's panel values with its 95% percentile-bootstrap interval."""
    v = np.asarray(v, float)
    e = effect(v, np.zeros_like(v), paired=True, metric="", panel="")
    return e.delta, e.lo, e.hi


def summarise(keys, with_ci):
    """One row per group: n and, per metric, the mean (and its interval per cell)."""
    rows = []
    for key, g in d.groupby(keys):
        row = dict(zip(keys, key)) | {"n": len(g)}
        for m in METRICS:
            v = g[m].dropna()
            if len(v) < 2:
                continue
            if with_ci:
                mean, lo, hi = mean_ci(v)
                row |= {m: mean, f"{m}_lo": lo, f"{m}_hi": hi}
            else:
                row[m] = v.mean()
        rows.append(row)
    return pd.DataFrame(rows)


by_rate = summarise(["model", "arm", "rate"], with_ci=False)
by_cell = summarise(["model", "arm", "rate", "churn"], with_ci=True)
by_rate.to_csv(OUT / "by_rate.csv", index=False)
by_cell.to_csv(OUT / "by_cell.csv", index=False)

# Whether the tree beats Pareto/NBD on the same panel: lower MAPE, smaller |bias|, higher
# Spearman. Counted over a table's panels these are the three "Beats P/NBD" columns.
pn = d[d.model == "ParetoNBD"].set_index(KEY)
pn = pn.reindex(pd.MultiIndex.from_frame(d[KEY]))
d["win_mape"] = d.mape.values < pn.mape.values
d["win_bias"] = d.bias.abs().values < pn.bias.abs().values
d["win_spearman"] = d.spearman.values > pn.spearman.values
WINS = ["win_mape", "win_bias", "win_spearman"]

# Direction of "better" per metric: +1 lower is better, -1 higher is better. Bias is
# judged on |bias| per panel, so +50% on one panel and -50% on another is not unbiased.
BETTER = {"rmse": 1, "bias": 1, "mape": 1, "spearman": -1, "ce": 1}


def cell_marks(sub):
    """{(model, arm, metric): "best" | "tie"} over the panels of ONE cell.

    Best is the lowest mean RMSE, |bias|, MAPE or Val. CE, or the highest mean Spearman.
    Tied means the paired 95% interval of (tree - best), on the cell's own panels,
    contains 0. Pareto/NBD has no CE (NaN, dropped by the pivot).
    """
    out = {}
    for m, sign in BETTER.items():
        v = sub[m].abs() if m == "bias" else sub[m]
        panel = sub.assign(v=v).pivot_table(index=KEY, columns=["model", "arm"], values="v")
        best = (sign * panel.mean()).idxmin()
        out[(*best, m)] = "best"
        for other in panel.columns.drop(best):
            pair = panel[[best, other]].dropna()
            if not effect(pair[other], pair[best], paired=True, metric=m, panel="").supported:
                out[(*other, m)] = "tie"
    return out


FMT = {"rmse": "{:.2f}", "bias": "{:+.0f}", "mape": "{:.0f}", "spearman": "{:.2f}", "ce": "{:.3f}"}


def num(m, x):
    t = FMT[m].format(x)
    if float(t) == 0:
        t = t.lstrip("+-")          # no signed zero: "−0.00" -> "0.00", "+0" -> "0"
    return t.replace("-", "−")


def signed(m, x):
    """A difference, always with its sign."""
    t = ("{:+.0f}" if m in ("mape", "bias") else "{:+.2f}" if m in ("rmse", "spearman") else "{:+.3f}").format(x)
    return ("0" if t.strip("+-0.") == "" else t).replace("-", "−")


def entry(r, m, ci):
    if m not in r or pd.isna(r[m]):
        return "—"
    return f"{num(m, r[m])} [{num(m, r[m + '_lo'])}, {num(m, r[m + '_hi'])}]" if ci else num(m, r[m])


def panels(n):
    return f"{n.max()} panels" if n.min() == n.max() else f"{n.min()}–{n.max()} panels"


def name(model, arm):
    if model == "ParetoNBD":
        return "**Pareto/NBD** | —"
    a, c = arm.split("-")
    return f"{model} | `{a}`" + (" + `kmeans_8`" if c == "kmeans_8" else "")


def marked(text, mark):
    return {"best": f"**{text}**", "tie": f"<ins>{text}</ins>"}.get(mark, text)


def table(frame, wins, mk=None, ci=False):
    """One markdown table; `wins` holds the win counts, `mk` the per-cell marks."""
    head = ("| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | "
            "Beats P/NBD: \\|bias\\| | Beats P/NBD: Spearman |")
    sep = "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |"
    lines = [head, sep]
    for model, arm in ORDER:
        r = frame[(frame.model == model) & (frame.arm == arm)]
        if r.empty:
            continue
        r = r.iloc[0]
        line = f"| {name(model, arm)} | " + " | ".join(
            marked(entry(r, m, ci), (mk or {}).get((model, arm, m))) for m in METRICS) + " |"
        if model == "ParetoNBD":
            line += " — | — | — |"
        else:
            w = wins.loc[(model, arm)]
            line += "".join(f" {int(w[c])} of {int(w['n'])} |" for c in WINS)
        lines.append(line)
    return "\n".join(lines)


def win_counts(sub):
    g = sub.groupby(["model", "arm"])
    return g[WINS].sum().join(g.size().rename("n"))


md = []
for rate in sorted(d.rate.unique()):
    sub = d[d.rate == rate]
    n = by_rate[by_rate.rate == rate].n
    md.append(f"#### Rate {rate:.2f}, churn pooled ({panels(n)})\n\n"
              + table(by_rate[by_rate.rate == rate], win_counts(sub)))
for rate in sorted(d.rate.unique()):
    for churn in sorted(d.churn.unique()):
        sub = d[(d.rate == rate) & (d.churn == churn)]
        f = by_cell[(by_cell.rate == rate) & (by_cell.churn == churn)]
        md.append(f"##### Rate {rate:.2f}, churn {churn:.0%} ({panels(f.n)})\n\n"
                  + table(f, win_counts(sub), cell_marks(sub), ci=True))
(OUT / "tables.md").write_text("\n\n".join(md) + "\n")


# The three best trees per rate x churn cell, by MAPE (lowest) and by Spearman (highest).
# Each runner-up carries the paired Δ against the cell's leader on the ranked metric.
def top3(metric, ascending):
    lines = [f"| Rate | Churn | Rank | Model | Arm | {'**MAPE**' if metric == 'mape' else 'MAPE'} | "
             f"{'**Spearman**' if metric == 'spearman' else 'Spearman'} | RMSE | Bias % | Val. CE | "
             f"Δ vs rank 1 [95% CI] | Clearly behind rank 1 | Beats P/NBD: MAPE / \\|bias\\| / Spearman |",
             "| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | :---: | ---: |"]
    for (rate, churn), f in by_cell.groupby(["rate", "churn"]):
        best = f.sort_values(metric, ascending=ascending).head(3)
        sub = d[(d.rate == rate) & (d.churn == churn)]
        panel = sub.pivot_table(index="dataset", columns=["model", "arm"], values=metric)
        leader = (best.iloc[0].model, best.iloc[0].arm)
        wins = win_counts(sub)
        for rank, (_, r) in enumerate(best.iterrows(), 1):
            model, arm = name(r.model, r.arm).split(" | ")
            vals = [entry(r, m, ci=(m == metric)) for m in ("mape", "spearman", "rmse", "bias", "ce")]
            lead = f"{rate:.2f} | {churn:.0%}" if rank == 1 else " | "
            if rank == 1:
                dv, sup = "—", "—"
            else:
                pair = panel[[leader, (r.model, r.arm)]].dropna()
                e = effect(pair[(r.model, r.arm)], pair[leader], paired=True, metric=metric, panel="")
                dv = f"{signed(metric, e.delta)} [{signed(metric, e.lo)}, {signed(metric, e.hi)}]"
                sup = "yes" if e.supported else "no"
            beats = "—" if r.model == "ParetoNBD" else " / ".join(
                f"{int(wins.loc[(r.model, r.arm), w])}" for w in WINS) + f" of {int(wins.loc[(r.model, r.arm), 'n'])}"
            lines.append(f"| {lead} | {rank} | {model} | {arm} | " + " | ".join(vals)
                         + f" | {dv} | {sup} | {beats} |")
    return "\n".join(lines)


(OUT / "top3_mape.md").write_text(top3("mape", True) + "\n")
(OUT / "top3_spearman.md").write_text(top3("spearman", False) + "\n")


# The "Impact of cohort size" and "Impact of AR features" tables: per row a tree, per column
# a rate (churns pooled) or a churn (rates pooled), each entry RMSE / bias % / MAPE as means
# over the column's 40 panels. These pool four cells, so they are description only; the
# per-cell tests are claims 3, 4 and 12 in the Claims section.
def triple(frame, rows, label, by):
    cells = {}
    for value in sorted(frame[by].unique()):
        sub = frame[frame[by] == value]
        for key in rows:
            r = sub[(sub.model == key[0]) & (sub.arm == key[1]) & (sub.cohort == key[2])]
            cells.setdefault(key, []).append(" / ".join(num(m, r[m].mean()) for m in ("rmse", "bias", "mape")))
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
