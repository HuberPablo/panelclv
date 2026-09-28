"""The tests behind the Claims section of `docs/insights-synthetic-grid.md`.

Reads results/per_study.csv and writes one markdown file per claim to results/claims/.
Every test follows `docs/statistical-protocol.md` and goes through
`panelclv.evaluation.effects.effect`; nothing here re-implements a bootstrap.

* A replication is one generated panel. A (rate x churn) cell holds 10 of them, and the
  test is run inside each cell only: Δ = mean(B) - mean(A) over the cell's panels, with
  its 95% percentile-bootstrap interval (10,000 resamples), supported iff 0 is outside.
* Two trees fitted on the same panels are paired by panel (`paired=True`). The 1,000- and
  3,000-customer grids are NOT paired (claim 12, `paired=False`): panel j of the two grids
  shares its seed, so the first 1,000 customers share their purchase-rate draw λ, but the
  dropout rates μ are drawn after all N λ's and therefore come from a different part of
  the random stream, and so do the counts. The customer populations are different draws.
* Per rate (churn pooled) and over the whole grid the results are DESCRIPTIVE: the pooled
  mean difference, and how many of the rate's cells support the claim and in which
  direction. No interval is computed over panels of different cells.
* A claim about one number per replication (R_A < 1, L_D > 0) is the paired case against
  the reference value.
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from panelclv.evaluation.effects import effect

OUT = Path(".scratch/synthetic-grid/results")
CLAIMS = OUT / "claims"
CLAIMS.mkdir(exist_ok=True)

d = pd.read_csv(OUT / "per_study.csv")
# The two Transformer ar_unbounded forecasts that disagree with their stored results.
d = d[(d.mape - d.stored_mape).abs() / d.stored_mape < 0.05].copy()
d["abs_bias"] = d.bias.abs()
KEY = ["rate", "churn", "dataset"]
RATES = sorted(d.rate.unique())
CHURNS = sorted(d.churn.unique())
LABEL = {"mape": "MAPE", "abs_bias": "\\|bias\\|", "rmse": "RMSE", "spearman": "Spearman",
         "l_d": "L_D", "r_a": "R_A", "shape": "Shape corr."}
FMT = {"mape": "{:+.0f}", "abs_bias": "{:+.0f}", "rmse": "{:+.2f}", "spearman": "{:+.2f}",
       "l_d": "{:+.2f}", "r_a": "{:+.2f}", "shape": "{:+.2f}"}
LEVEL = {"mape": "{:.0f}", "abs_bias": "{:.0f}", "rmse": "{:.2f}", "spearman": "{:.2f}",
         "l_d": "{:.2f}", "r_a": "{:.2f}", "shape": "{:.2f}"}
MAIN = ["mape", "abs_bias", "rmse", "spearman"]


def minus(s):
    """Typographic minus, and no signed zero ("+0", "−0.00" -> "0")."""
    for z in ("+0.00", "-0.00", "+0", "-0"):
        for end in (" ", ",", "]", "*"):
            s = s.replace(z + end, "0" + end)
        s = "0" if s == z else s
    return s.replace("-", "−")


def fmt(m, x):
    return minus(FMT[m].format(x))


# ---------------------------------------------------------------------------
# Comparisons: A -> B per cell, described per rate and over the grid
# ---------------------------------------------------------------------------

def select(cohort, model, arm):
    return d[(d.cohort == cohort) & (d.model == model) & (d.arm == arm)]


def cell(frame, rate, churn):
    return frame[(frame.rate == rate) & (frame.churn == churn)]


def cell_effect(a, b, m, paired):
    """Δ = mean(B) - mean(A) on metric m in one cell, paired by panel or independent."""
    if paired:
        j = a.set_index(KEY)[m].to_frame("a").join(b.set_index(KEY)[m].rename("b"), how="inner")
        return effect(j.b, j.a, paired=True, metric=m, panel="")
    return effect(b[m].dropna(), a[m].dropna(), paired=False, metric=m, panel="")


def compare(contexts, metrics=MAIN, paired=True):
    """One row per context x cell with the effects, and the descriptive pooled rows.

    A context is (label, frame A, frame B); the effect is B minus A.
    """
    cells, pooled = [], []
    for label, a, b in contexts:
        for rate, churn in product(RATES, CHURNS):
            row = {"label": label, "rate": rate, "churn": churn}
            for m in metrics:
                e = cell_effect(cell(a, rate, churn), cell(b, rate, churn), m, paired)
                row["n"] = f"{e.n_a}" if paired else f"{e.n_a} / {e.n_b}"
                row |= {m + "_a": e.mean_a, m + "_b": e.mean_b, m: e.delta,
                        m + "_lo": e.lo, m + "_hi": e.hi, m + "_sup": e.supported}
            cells.append(row)
        # Pooled rows: plain means over the pooled panels, no interval.
        for rate in RATES + [None]:
            sa = a if rate is None else a[a.rate == rate]
            sb = b if rate is None else b[b.rate == rate]
            row = {"label": label, "rate": rate, "n": f"{sa.dataset.size} / {sb.dataset.size}"}
            for m in metrics:
                row |= {m + "_a": sa[m].mean(), m + "_b": sb[m].mean(), m: sb[m].mean() - sa[m].mean()}
            pooled.append(row)
    return pd.DataFrame(cells), pd.DataFrame(pooled)


def verdicts(c, label, m, rate=None):
    """"k/4 (+)" style count of the cells supporting a difference, with its signs."""
    s = c[(c.label == label) & (True if rate is None else c.rate == rate)]
    s = s[s[m + "_sup"]]
    total = len(c[(c.label == label) & (True if rate is None else c.rate == rate)])
    pos, neg = int((s[m] > 0).sum()), int((s[m] < 0).sum())
    signs = "" if not len(s) else " (+)" if not neg else " (−)" if not pos else f" ({pos}+, {neg}−)"
    return f"{len(s)}/{total}{signs}"


def render_cells(c, metrics, levels):
    multi = c.label.nunique() > 1
    head = (["Comparison"] if multi else []) + ["Rate", "Churn", "n"]
    head += [f"{LABEL[m]} A → B" for m in levels] + [f"Δ {LABEL[m]} [95% CI]" for m in metrics]
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for _, r in c.iterrows():
        cells = ([r.label] if multi else []) + [f"{r.rate:.2f}", f"{r.churn:.0%}", str(r.n)]
        cells += [minus(f"{LEVEL[m].format(r[m + '_a'])} → {LEVEL[m].format(r[m + '_b'])}") for m in levels]
        for m in metrics:
            s = f"{fmt(m, r[m])} [{fmt(m, r[m + '_lo'])}, {fmt(m, r[m + '_hi'])}]"
            cells.append(f"**{s}**" if r[m + "_sup"] else s)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_pooled(c, p, metrics, levels):
    multi = p.label.nunique() > 1
    head = (["Comparison"] if multi else []) + ["Rate", "n (A / B)"]
    head += [f"{LABEL[m]} A → B" for m in levels]
    head += [f"Δ {LABEL[m]} (pooled mean) · cells supported" for m in metrics]
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for _, r in p.iterrows():
        cells = ([r.label] if multi else []) + ["all" if pd.isna(r.rate) else f"{r.rate:.2f}", r.n]
        cells += [minus(f"{LEVEL[m].format(r[m + '_a'])} → {LEVEL[m].format(r[m + '_b'])}") for m in levels]
        rate = None if pd.isna(r.rate) else r.rate
        cells += [f"{fmt(m, r[m])} · {verdicts(c, r.label, m, rate)}" for m in metrics]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def matrix(c, p, label, metrics):
    """Rate x churn grid of one context: per cell "Δa / Δb / Δc", bold = supported.

    The "churn pooled" column and the "all" row are italic: pooled mean differences,
    descriptive only.
    """
    cc, pp = c[c.label == label], p[p.label == label]

    def one(rate, churn):
        r = cc[(cc.rate == rate) & (cc.churn == churn)].iloc[0]
        return " / ".join(f"**{fmt(m, r[m])}**" if r[m + "_sup"] else fmt(m, r[m]) for m in metrics)

    def pooled(rate):
        r = pp[pp.rate.isna()].iloc[0] if rate is None else pp[pp.rate == rate].iloc[0]
        return "*" + " / ".join(fmt(m, r[m]) for m in metrics) + "*"

    head = ["Rate", *(f"Churn {x:.0%}" for x in CHURNS), "Churn pooled (descriptive)"]
    lines = [f"**{label}**: " + " / ".join(f"Δ {LABEL[m]}" for m in metrics), "",
             "| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for rate in RATES:
        lines.append("| " + " | ".join([f"{rate:.2f}"] + [one(rate, x) for x in CHURNS] + [pooled(rate)]) + " |")
    lines.append("| " + " | ".join(["all"] + [""] * len(CHURNS) + [pooled(None)]) + " |")
    return "\n".join(lines)


def write(name, contexts, metrics=MAIN, levels=("mape", "abs_bias"), grid=("mape", "abs_bias", "spearman"),
          paired=True):
    c, p = compare(contexts, metrics, paired)
    mats = "\n\n".join(matrix(c, p, label, list(grid)) for label in c.label.unique())
    (CLAIMS / f"{name}.md").write_text(
        mats + "\n\n<details><summary>By rate, churn pooled (descriptive)</summary>\n\n"
        + render_pooled(c, p, metrics, levels) + "\n\n</details>\n\n"
        "<details><summary>Intervals per rate × churn cell</summary>\n\n"
        + render_cells(c, metrics, levels) + "\n\n</details>\n")
    return c


N = "n1000"
ARMS = [f"{a}-{c}" for a in ("no_ar", "ar_bounded", "ar_unbounded") for c in ("no_cluster", "kmeans_8")]


def short(arm):
    a, c = arm.split("-")[:2]
    return f"`{a}`" + (" + `kmeans_8`" if c == "kmeans_8" else "")


# 1. Pareto/NBD vs the neural models (B - A = neural - Pareto/NBD; positive = neural worse
#    on MAPE, |bias|, RMSE; negative = neural worse on Spearman).
pnbd = select(N, "ParetoNBD", "-")
c1 = [(f"{m} {short(arm)}, 1,000 customers", pnbd, select(N, m, arm))
      for m in ("LSTM", "Transformer") for arm in ("ar_bounded-no_cluster", "no_ar-no_cluster")]
c1 += [(f"LSTM {short(arm)}, 3,000 customers", select("n3000", "ParetoNBD", "-"), select("n3000", "LSTM", arm))
       for arm in ("ar_bounded-no_cluster", "no_ar-no_cluster")]
res = {"1": write("1_pnbd_vs_neural", c1)}

# 3. Bounded flags: no_ar -> ar_bounded, both models at 1,000 and the LSTM at 3,000.
c3 = [(f"{m}, {n[1]},{n[2:]} customers", select(n, m, "no_ar-no_cluster"), select(n, m, "ar_bounded-no_cluster"))
      for m, n in (("LSTM", "n1000"), ("Transformer", "n1000"), ("LSTM", "n3000"))]
res["3"] = write("3_bounded", c3)

# 4. Unbounded counters: no_ar -> ar_unbounded.
c4 = [(m, select(N, m, "no_ar-no_cluster"), select(N, m, "ar_unbounded-no_cluster"))
      for m in ("LSTM", "Transformer")]
res["4"] = write("4_unbounded", c4)

# 5. Cluster label: arm -> arm + kmeans_8, for each AR encoding.
c5 = [(f"{m} {short(a + '-no_cluster')}", select(N, m, a + "-no_cluster"), select(N, m, a + "-kmeans_8"))
      for m in ("LSTM", "Transformer") for a in ("no_ar", "ar_bounded", "ar_unbounded")]
res["5"] = write("5_kmeans", c5)

# 6. Architecture: LSTM -> Transformer, same arm.
c6 = [(short(arm), select(N, "LSTM", arm), select(N, "Transformer", arm)) for arm in ARMS]
res["6"] = write("6_architecture", c6)

# 7. Seasonality: Pareto/NBD as fitted -> with the true season; then Pareto/NBD with the
#    season -> LSTM ar_bounded.
c7 = [("Pareto/NBD → with true season", pnbd, select(N, "ParetoNBD", "true_season")),
      ("Pareto/NBD with season → LSTM `ar_bounded`", select(N, "ParetoNBD", "true_season"),
       select(N, "LSTM", "ar_bounded-no_cluster"))]
res["7"] = write("7_seasonality", c7)

# 7b. Shape correlation of predicted vs actual weekly totals, per cell, for three trees
#     (means only: no comparison is claimed from this table).
shape_trees = [("Pareto/NBD", pnbd), ("LSTM `ar_bounded`", select(N, "LSTM", "ar_bounded-no_cluster")),
               ("Transformer `ar_bounded`", select(N, "Transformer", "ar_bounded-no_cluster"))]
head = ["Rate", *(f"Churn {x:.0%}" for x in CHURNS), "Churn pooled"]
lines = ["Shape correlation, mean over panels: " + " / ".join(l for l, _ in shape_trees), "",
         "| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
for rate in RATES:
    row = [f"{rate:.2f}"]
    for churn in CHURNS + [None]:
        row.append(" / ".join(minus(f"{f[(f.rate == rate) & ((f.churn == churn) if churn else True)]['shape'].mean():+.2f}")
                              for _, f in shape_trees))
    lines.append("| " + " | ".join(row) + " |")
(CLAIMS / "7b_shape.md").write_text("\n".join(lines) + "\n")

# 9. Dead leakage: Pareto/NBD -> neural ar_bounded, on L_D and R_A.
c9 = [(m, pnbd, select(N, m, "ar_bounded-no_cluster")) for m in ("LSTM", "Transformer")]
res["9"] = write("9_dead_leakage", c9, metrics=["l_d", "r_a"], levels=("l_d", "r_a"), grid=("l_d", "r_a"))

# 10. Search budget: archived 10 / 20 trials -> 100 trials, no_ar, same panels.
c10 = [(f"{m}, {k} → 100 trials", select(N, m, "no_ar-no_cluster-small_search"), select(N, m, "no_ar-no_cluster"))
       for m, k in (("LSTM", 10), ("Transformer", 20))]
res["10"] = write("10_search_budget", c10)

# 12. Cohort size: 1,000 -> 3,000 customers. Different customer populations (module
#     docstring), so the two sides are independent replications.
c12 = [(f"{m} {short(arm)}" if m != "ParetoNBD" else "Pareto/NBD", select("n1000", m, arm), select("n3000", m, arm))
       for m, arm in (("LSTM", "no_ar-no_cluster"), ("LSTM", "ar_bounded-no_cluster"), ("ParetoNBD", "-"))]
res["12"] = write("12_cohort_size", c12, paired=False)


# 2. Error vs churn: per tree and rate, the mean at each churn level. A trend ACROSS cells
#    is described, not tested (protocol §7), so there is no correlation or p-value here.
def churn_trend(trees, metrics=("abs_bias", "mape")):
    head = ["Tree", "Customers", "Rate"] + [f"{LABEL[m]} at churn 20 / 40 / 60 / 80%" for m in metrics] + \
           [f"{LABEL[m]} rises at every step" for m in metrics]
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for (tree, n), frame in trees:
        for rate in RATES:
            means = {m: frame[frame.rate == rate].groupby("churn")[m].mean() for m in metrics}
            lines.append("| " + " | ".join(
                [tree, n, f"{rate:.2f}"]
                + [" / ".join(LEVEL[m].format(v) for v in means[m]) for m in metrics]
                + ["yes" if means[m].is_monotonic_increasing else "no" for m in metrics]) + " |")
    return "\n".join(lines)


trees2 = [(("Pareto/NBD", "1,000"), pnbd)] + [
    ((f"{m} {short(a)}", "1,000"), select(N, m, a))
    for m in ("LSTM", "Transformer") for a in ("no_ar-no_cluster", "ar_bounded-no_cluster")] + [
    (("Pareto/NBD", "3,000"), select("n3000", "ParetoNBD", "-"))] + [
    ((f"LSTM {short(a)}", "3,000"), select("n3000", "LSTM", a)) for a in ("no_ar-no_cluster", "ar_bounded-no_cluster")]
(CLAIMS / "2_churn_trend.md").write_text(churn_trend(trees2) + "\n")


# 8. Pareto/NBD's volume split, per cell: the alive ratio R_A against 1 and the dead
#    leakage L_D against 0, each the one-statistic-per-panel case of `effect`. |bias| and
#    RMSE are shown as means beside them.
def volume_split(frame):
    head = ["Rate", "Churn", "n", "\\|bias\\| (mean)", "R_A: mean, R_A − 1 [95% CI]",
            "L_D: mean [95% CI]", "RMSE (mean)"]
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for rate, churn in product(RATES, CHURNS):
        f = cell(frame, rate, churn)
        ra = effect(f.r_a, np.ones(len(f)), paired=True, metric="r_a", panel="")
        ld = effect(f.l_d, np.zeros(len(f)), paired=True, metric="l_d", panel="")
        s_ra = minus(f"{ra.mean_b:.2f}, {ra.delta:+.2f} [{ra.lo:+.2f}, {ra.hi:+.2f}]")
        s_ld = minus(f"{ld.mean_b:.2f} [{ld.lo:.2f}, {ld.hi:.2f}]")
        lines.append("| " + " | ".join([
            f"{rate:.2f}", f"{churn:.0%}", str(len(f)), f"{f.abs_bias.mean():.0f}",
            f"**{s_ra}**" if ra.supported else s_ra, f"**{s_ld}**" if ld.supported else s_ld,
            f"{f.rmse.mean():.2f}"]) + " |")
    for rate in RATES + [None]:
        f = frame if rate is None else frame[frame.rate == rate]
        lines.append("| " + " | ".join([
            "all" if rate is None else f"{rate:.2f}", "pooled (descriptive)", str(len(f)),
            f"{f.abs_bias.mean():.0f}", f"{f.r_a.mean():.2f}", f"{f.l_d.mean():.2f}", f"{f.rmse.mean():.2f}"]) + " |")
    return "\n".join(lines)


(CLAIMS / "8_pnbd_volume_split.md").write_text(volume_split(pnbd) + "\n")


# 11. Hyperparameters: Spearman of each chosen hyperparameter with |bias| and MAPE.
#     Pooled over the grid it is descriptive only: it mixes cells, and a correlation across
#     cells is the rate/churn regime choosing both variables. Within cells, each of the 16
#     cells gives one rank correlation over its 10 panels; those 16 values are tested
#     against 0 by the one-statistic rule, with the CELL as the unit.
HP = {"LSTM": ["batch_size", "learning_rate", "lstm_hidden_size", "dense_units", "dropout"],
      "Transformer": ["batch_size", "learning_rate", "d_model", "nhead", "num_encoder_layers", "dropout"]}
head = ["Tree", "Hyperparameter", "ρ with \\|bias\\|, pooled (descriptive)",
        "ρ with \\|bias\\|, mean within-cell [95% CI], cells", "ρ with MAPE, pooled (descriptive)",
        "ρ with MAPE, mean within-cell [95% CI], cells"]
lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
for m, arm in product(("LSTM", "Transformer"), ("no_ar-no_cluster", "ar_bounded-no_cluster")):
    f = select(N, m, arm)
    for hp in HP[m]:
        col = "param_" + hp
        if col not in f or f[col].nunique() < 2:
            continue
        row = [f"{m} {short(arm)}", f"`{hp}`"]
        for out in ("abs_bias", "mape"):
            row.append(minus(f"{stats.spearmanr(f[col], f[out]).statistic:+.2f}"))
            # A cell where the search chose one value throughout has no correlation (NaN).
            rho = np.array([stats.spearmanr(g[col], g[out]).statistic if g[col].nunique() > 1 else np.nan
                            for _, g in f.groupby(["rate", "churn"])])
            # Fewer than 5 cells with a varying choice is too few units to resample.
            if np.isfinite(rho).sum() < 5:
                row.append(f"not tested: varies in {np.isfinite(rho).sum()} of 16 cells")
                continue
            e = effect(rho, np.zeros_like(rho), paired=True, metric=out, panel="")
            s = minus(f"{e.delta:+.2f} [{e.lo:+.2f}, {e.hi:+.2f}], {e.n_b}")
            row.append(f"**{s}**" if e.supported else s)
        lines.append("| " + " | ".join(row) + " |")
(CLAIMS / "11_hyperparameters.md").write_text("\n".join(lines) + "\n")


# 13. RMSE as a ranking metric: per rate, the spread of each RMSE over the 13 trees
#     (descriptive: how the 13 tree means spread, no test).
rows = []
g13 = d[(d.cohort == N) & ~d.arm.isin(["true_season", "no_ar-no_cluster-small_search"])]
for rate in RATES:
    means = g13[g13.rate == rate].groupby(["model", "arm"])[["rmse_week", "rmse", "mape"]].mean()
    rows.append("| {:.2f} | {:.3f}–{:.3f} | {} of 13 | {:.2f}–{:.2f} | {} |".format(
        rate, means.rmse_week.min(), means.rmse_week.max(),
        int((means.rmse_week.round(2) == means.rmse_week.round(2).mode()[0]).sum()),
        means.rmse.min(), means.rmse.max(),
        minus(f"{stats.spearmanr(means.rmse, means.mape).statistic:+.2f}")))
(CLAIMS / "13_rmse.md").write_text(
    "| Rate | Per-week RMSE, range over trees | Trees at the modal per-week RMSE (2 dp) "
    "| Customer-total RMSE, range | ρ(customer-total RMSE, MAPE) over trees |\n"
    "| --- | --- | --- | --- | --- |\n" + "\n".join(rows) + "\n")


# Results "Reading": does a lower validation CE go with a better holdout, across the 12
# neural arms fitted on the same panel? One rank correlation per panel (12 arms), tested
# per cell against 0 by the one-statistic rule (10 panels a cell).
neural = d[(d.cohort == N) & d.model.isin(["LSTM", "Transformer"])
           & ~d.arm.isin(["no_ar-no_cluster-small_search"])]
per_panel = neural.groupby(KEY).apply(
    lambda g: pd.Series({t: stats.spearmanr(g.ce, g[t]).statistic for t in ("mape", "rmse", "spearman")}),
    include_groups=False).reset_index()
head = ["Rate", "Churn", "n", "ρ(CE, MAPE) [95% CI]", "ρ(CE, RMSE) [95% CI]", "ρ(CE, Spearman) [95% CI]"]
lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
for rate, churn in product(RATES, CHURNS):
    f = cell(per_panel, rate, churn)
    row = [f"{rate:.2f}", f"{churn:.0%}", str(len(f))]
    for t in ("mape", "rmse", "spearman"):
        e = effect(f[t], np.zeros(len(f)), paired=True, metric=t, panel="")
        s = minus(f"{e.delta:+.2f} [{e.lo:+.2f}, {e.hi:+.2f}]")
        row.append(f"**{s}**" if e.supported else s)
    lines.append("| " + " | ".join(row) + " |")
(CLAIMS / "0_ce_vs_holdout.md").write_text("\n".join(lines) + "\n")


# Tally of supported cells per claim context, printed for the doc's prose.
for claim, c in res.items():
    for label in c.label.unique():
        print(f"[{claim}] {label}: " + ", ".join(
            f"{LABEL[m]} {verdicts(c, label, m)}" for m in c.columns if m + '_sup' in c.columns))
print("wrote", sorted(p.name for p in CLAIMS.iterdir()))
