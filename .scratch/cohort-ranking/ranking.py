"""Does tripling the cohort change the ranking between Pareto/NBD and the LSTM?

Per rate x churn cell, for each LSTM arm, the gap g = LSTM - Pareto/NBD per panel:
* at each cohort, paired by panel (does the LSTM beat or lose to P/NBD here?);
* its change, effect(g_3000, g_1000, paired=False): the two grids are independent.
Also who leads each cell (lowest MAPE / |bias|, highest Spearman) among the three trees
that ran at both sizes.
"""
from itertools import product
from pathlib import Path
import pandas as pd
from panelclv.evaluation.effects import effect

d = pd.read_csv(".scratch/synthetic-grid/results/per_study.csv")
d = d[(d.mape - d.stored_mape).abs() / d.stored_mape < 0.05].copy()
d["abs_bias"] = d.bias.abs()
KEY = ["rate", "churn", "dataset"]
TREES = {"P/NBD": ("ParetoNBD", "-"), "no_ar": ("LSTM", "no_ar-no_cluster"),
         "ar_bounded": ("LSTM", "ar_bounded-no_cluster")}
METRICS = {"mape": ("MAPE", "{:+.0f}", min), "spearman": ("Spearman", "{:+.2f}", max),
           "abs_bias": ("|bias|", "{:+.0f}", min)}
RATES, CHURNS = sorted(d.rate.unique()), sorted(d.churn.unique())

def frame(cohort, tree, rate, churn):
    model, arm = TREES[tree]
    return d[(d.cohort == cohort) & (d.model == model) & (d.arm == arm)
             & (d.rate == rate) & (d.churn == churn)].set_index(KEY)

def gap(cohort, tree, rate, churn, m):
    j = frame(cohort, tree, rate, churn)[m].to_frame("b").join(
        frame(cohort, "P/NBD", rate, churn)[m].rename("a"), how="inner")
    return j.b - j.a

def ci(e, f):
    s = f"{f.format(e.delta)} [{f.format(e.lo)}, {f.format(e.hi)}]".replace("-", "−")
    return f"**{s}**" if e.supported else s

out = []
for m, (name, f, best) in METRICS.items():
    out.append(f"\n## {name}\n\n### Leader per cell (1,000 → 3,000)\n")
    out.append("| Rate | " + " | ".join(f"Churn {int(c*100)}%" for c in CHURNS) + " |")
    out.append("| --- |" + " --- |" * len(CHURNS))
    for r in RATES:
        row = []
        for c in CHURNS:
            lead = []
            for co in ("n1000", "n3000"):
                means = {t: frame(co, t, r, c)[m].mean() for t in TREES}
                lead.append(best(means, key=means.get))
            row.append(lead[0] + " → " + lead[1] if lead[0] != lead[1] else lead[0])
        out.append(f"| {r} | " + " | ".join(row) + " |")
    for arm in ("no_ar", "ar_bounded"):
        out.append(f"\n### LSTM {arm} − P/NBD: gap at 1,000 · gap at 3,000 · change\n")
        out.append("| Rate | Churn | n | Gap 1,000 [95% CI] | Gap 3,000 [95% CI] | Δ gap [95% CI] |")
        out.append("| --- | --- | --- | --- | --- | --- |")
        for r, c in product(RATES, CHURNS):
            g1, g3 = gap("n1000", arm, r, c, m), gap("n3000", arm, r, c, m)
            z1, z3 = g1 * 0, g3 * 0
            e1 = effect(g1, z1, paired=True, metric=m, panel="")
            e3 = effect(g3, z3, paired=True, metric=m, panel="")
            dd = effect(g3, g1, paired=False, metric=m, panel="")
            out.append(f"| {r} | {int(c*100)}% | {len(g1)} / {len(g3)} | {ci(e1, f)} | {ci(e3, f)} | {ci(dd, f)} |")

Path(".scratch/cohort-ranking/ranking.md").write_text("\n".join(out) + "\n")
print("\n".join(out))


# ---------------------------------------------------------------------------
# Doc fragment: per metric, a matrix "gap 1,000 → gap 3,000 (Δ gap)", each number bold
# when its own 95% interval excludes 0; full intervals in a collapsible block.
# ---------------------------------------------------------------------------
def num(e, f):
    s = f.format(e.delta)
    for z in ("+0.00", "-0.00", "+0", "-0"):
        s = "0" if s == z else s
    s = s.replace("-", "−")
    return f"**{s}**" if e.supported else s

def rng(e, f):
    s = f"{f.format(e.delta)} [{f.format(e.lo)}, {f.format(e.hi)}]"
    for z in ("+0.00", "-0.00", "+0", "-0"):
        for end in (" ", ",", "]"):
            s = s.replace(z + end, "0" + end)
    s = s.replace("-", "−")
    return f"**{s}**" if e.supported else s

frag = []
for m, (name, f, best) in METRICS.items():
    frag.append(f"\n#### {name}: LSTM − Pareto/NBD, 1,000 → 3,000 (Δ gap)\n")
    frag.append("| LSTM arm | Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% |")
    frag.append("| --- | --- | --- | --- | --- | --- |")
    full = []
    for arm in ("no_ar", "ar_bounded"):
        for r in RATES:
            row = []
            for c in CHURNS:
                g1, g3 = gap("n1000", arm, r, c, m), gap("n3000", arm, r, c, m)
                e1 = effect(g1, g1 * 0, paired=True, metric=m, panel="")
                e3 = effect(g3, g3 * 0, paired=True, metric=m, panel="")
                dd = effect(g3, g1, paired=False, metric=m, panel="")
                row.append(f"{num(e1, f)} → {num(e3, f)} ({num(dd, f)})")
                full.append(f"| `{arm}` | {r:.2f} | {int(c*100)}% | {len(g1)} / {len(g3)} | "
                            f"{rng(e1, f)} | {rng(e3, f)} | {rng(dd, f)} |")
            frag.append(f"| `{arm}` | {r:.2f} | " + " | ".join(row) + " |")
    frag.append(f"\n<details><summary>{name}: intervals per rate × churn cell</summary>\n")
    frag.append("| LSTM arm | Rate | Churn | n | Gap 1,000 [95% CI] | Gap 3,000 [95% CI] | Δ gap [95% CI] |")
    frag.append("| --- | --- | --- | --- | --- | --- | --- |")
    frag += full
    frag.append("\n</details>")
Path(".scratch/cohort-ranking/fragment.md").write_text("\n".join(frag) + "\n")
