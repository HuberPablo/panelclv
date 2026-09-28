"""Statistical tests behind the insights doc's Claims section, from results/per_study.csv.

Every comparison is reported per rate x churn cell (10 panels), per rate (40 panels) and
over the whole grid (160 panels). Paired comparisons (same panels) use the Hodges-Lehmann
estimate of the paired difference with its exact 95% interval, and the Wilcoxon
signed-rank test; unpaired ones (1,000 vs 3,000 customers) the two-sample Hodges-Lehmann
shift with its exact interval, and the Mann-Whitney U test. Significance is judged at a 5%
false discovery rate (Benjamini-Hochberg) within each table: with 10 panels the smallest
exact Wilcoxon p is 0.002, so a family-wise correction over a 64-test table could never
flag a single cell. Writes one markdown file per claim to results/claims/.
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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


# ---------------------------------------------------------------------------
# Exact Hodges-Lehmann intervals
# ---------------------------------------------------------------------------

def _signed_rank_null(n):
    """Counts of the Wilcoxon T+ statistic under the null, for n non-zero differences."""
    counts = np.zeros(n * (n + 1) // 2 + 1)
    counts[0] = 1
    for k in range(1, n + 1):
        counts[k:] = counts[k:] + counts[:-k].copy()
    return counts / counts.sum()


def _rank_sum_null(m, n):
    """Counts of the Mann-Whitney U statistic under the null, for samples of m and n."""
    # f[i][j][u]: arrangements of i x's and j y's with U = u, built one element at a time.
    f = np.zeros((m + 1, n + 1, m * n + 1))
    f[0, :, 0] = 1
    f[:, 0, 0] = 1
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            f[i, j, j:] += f[i - 1, j, :m * n + 1 - j]
            f[i, j] += f[i, j - 1]
    return f[m, n] / f[m, n].sum()


def _lower_index(null):
    """Largest c with P(stat <= c - 1) <= 2.5%: the interval runs from the c-th to the
    (M - c + 1)-th ordered average, 1-indexed."""
    return int(np.searchsorted(np.cumsum(null), 0.025, side="right"))


def paired_hl(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    walsh = np.sort((x[:, None] + x[None, :])[np.triu_indices(len(x))] / 2)
    c = max(_lower_index(_signed_rank_null(len(x))), 1)
    p = stats.wilcoxon(x).pvalue if np.any(x != 0) else 1.0
    return np.median(walsh), walsh[c - 1], walsh[-c], p


def shift_hl(y, x):
    """Shift of y relative to x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    diffs = np.sort((y[:, None] - x[None, :]).ravel())
    c = max(_lower_index(_rank_sum_null(len(y), len(x))), 1)
    return np.median(diffs), diffs[c - 1], diffs[-c], stats.mannwhitneyu(y, x).pvalue


def bh(p):
    """Benjamini-Hochberg adjusted p-values (NaN stays NaN)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    out[ok] = stats.false_discovery_control(p[ok])
    return out


def fmt_p(p):
    return "<10⁻⁴" if p < 1e-4 else f"{p:.4f}" if p < 0.001 else f"{p:.3f}"


def minus(s):
    for z in ("+0.00", "-0.00", "+0", "-0"):
        s = s.replace(z + " ", "0 ").replace(z + ",", "0,").replace(z + "]", "0]")
        s = s if s != z else "0"
    return s.replace("-", "−")


# ---------------------------------------------------------------------------
# Comparison tables
# ---------------------------------------------------------------------------

def select(cohort, model, arm):
    return d[(d.cohort == cohort) & (d.model == model) & (d.arm == arm)]


def scopes():
    """(rate, churn) scopes: every cell, every rate pooled over churn, the whole grid."""
    yield from ((r, c) for r, c in product(RATES, CHURNS))
    yield from ((r, None) for r in RATES)
    yield (None, None)


def in_scope(frame, rate, churn):
    m = np.ones(len(frame), bool)
    if rate is not None:
        m &= frame.rate.to_numpy() == rate
    if churn is not None:
        m &= frame.churn.to_numpy() == churn
    return frame[m]


def compare(contexts, metrics=MAIN, paired=True, levels=("mape", "abs_bias")):
    """One row per context x scope. A context is (label, frame A, frame B); the effect
    is B minus A. Returns (cell table, rate-and-grid table) as markdown."""
    rows = []
    for label, a, b in contexts:
        for rate, churn in scopes():
            sa, sb = in_scope(a, rate, churn), in_scope(b, rate, churn)
            row = {"label": label, "rate": rate, "churn": churn}
            for m in metrics:
                if paired:
                    j = sa.set_index(KEY)[m].to_frame("a").join(sb.set_index(KEY)[m].rename("b"), how="inner")
                    j = j.dropna()
                    row["n"] = len(j)
                    est, lo, hi, p = paired_hl(j.b - j.a)
                    row[m + "_a"], row[m + "_b"] = j.a.mean(), j.b.mean()
                else:
                    row["n"] = f"{sa[m].notna().sum()} / {sb[m].notna().sum()}"
                    est, lo, hi, p = shift_hl(sb[m], sa[m])
                    row[m + "_a"], row[m + "_b"] = sa[m].mean(), sb[m].mean()
                row |= {m: est, m + "_lo": lo, m + "_hi": hi, m + "_p": p}
            rows.append(row)
    t = pd.DataFrame(rows)
    t_cell = t[t.churn.notna()].copy()
    t_rate = t[t.churn.isna()].copy()
    for frame in (t_cell, t_rate):
        adj = bh(frame[[m + "_p" for m in metrics]].to_numpy().ravel()).reshape(len(frame), -1)
        for k, m in enumerate(metrics):
            frame[m + "_padj"] = adj[:, k]
    return render(t_cell, metrics, levels), render(t_rate, metrics, levels), pd.concat([t_cell, t_rate])


def render(t, metrics, levels):
    multi = t.label.nunique() > 1
    head = (["Comparison"] if multi else []) + ["Rate", "Churn", "n"]
    head += [f"{LABEL[m]} A → B" for m in levels]
    head += [f"Δ {LABEL[m]} [95% CI], p" for m in metrics]
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for _, r in t.iterrows():
        cells = ([r.label] if multi else [])
        cells += ["all" if pd.isna(r.rate) else f"{r.rate:.2f}",
                  "pooled" if pd.isna(r.churn) else f"{r.churn:.0%}", str(r.n)]
        cells += [minus(f"{LEVEL[m].format(r[m + '_a'])} → {LEVEL[m].format(r[m + '_b'])}") for m in levels]
        for m in metrics:
            f = FMT[m]
            s = minus(f"{f.format(r[m])} [{f.format(r[m + '_lo'])}, {f.format(r[m + '_hi'])}]")
            s += f", p {fmt_p(r[m + '_p'])}"
            cells.append(f"**{s}**" if r[m + "_padj"] < 0.05 else s)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def matrix(t, label, metrics):
    """Rate x churn grid of one context's effects, one "Δa / Δb / Δc" cell per scope."""
    t = t[t.label == label]

    def cell(rate, churn):
        r = t[(t.rate.isna() if rate is None else t.rate == rate)
              & (t.churn.isna() if churn is None else t.churn == churn)].iloc[0]
        parts = []
        for m in metrics:
            v = minus(FMT[m].format(r[m]))
            parts.append(f"**{v}**" if r[m + "_padj"] < 0.05 else v)
        return " / ".join(parts)

    head = ["Rate", *(f"Churn {c:.0%}" for c in CHURNS), "Churn pooled"]
    lines = [f"**{label}**: " + " / ".join(f"Δ {LABEL[m]}" for m in metrics), "",
             "| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for rate in RATES:
        lines.append("| " + " | ".join([f"{rate:.2f}"] + [cell(rate, c) for c in CHURNS + [None]]) + " |")
    lines.append("| " + " | ".join(["all"] + [""] * len(CHURNS) + [cell(None, None)]) + " |")
    return "\n".join(lines)


def write(name, result, metrics=("mape", "abs_bias", "spearman")):
    cell, rate, t = result
    mats = "\n\n".join(matrix(t, label, list(metrics)) for label in t.label.unique())
    (CLAIMS / f"{name}.md").write_text(
        mats + "\n\n<details><summary>Intervals and p-values by rate, churn pooled</summary>\n\n"
        + rate + "\n\n</details>\n\n<details><summary>Intervals and p-values per rate × churn cell"
        "</summary>\n\n" + cell + "\n\n</details>\n")


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
write("1_pnbd_vs_neural", compare(c1))

# 3. Bounded flags: no_ar -> ar_bounded, both models at 1,000 and the LSTM at 3,000.
c3 = [(f"{m}, {n[1]},{n[2:]} customers", select(n, m, "no_ar-no_cluster"), select(n, m, "ar_bounded-no_cluster"))
      for m, n in (("LSTM", "n1000"), ("Transformer", "n1000"), ("LSTM", "n3000"))]
write("3_bounded", compare(c3))

# 4. Unbounded counters: no_ar -> ar_unbounded.
c4 = [(m, select(N, m, "no_ar-no_cluster"), select(N, m, "ar_unbounded-no_cluster"))
      for m in ("LSTM", "Transformer")]
write("4_unbounded", compare(c4))

# 5. Cluster label: arm -> arm + kmeans_8, for each AR encoding.
c5 = [(f"{m} {short(a + '-no_cluster')}", select(N, m, a + "-no_cluster"), select(N, m, a + "-kmeans_8"))
      for m in ("LSTM", "Transformer") for a in ("no_ar", "ar_bounded", "ar_unbounded")]
write("5_kmeans", compare(c5))

# 6. Architecture: LSTM -> Transformer, same arm.
c6 = [(short(arm), select(N, "LSTM", arm), select(N, "Transformer", arm)) for arm in ARMS]
write("6_architecture", compare(c6))

# 7. Seasonality: Pareto/NBD as fitted -> with the true season; then Pareto/NBD with the
#    season -> LSTM ar_bounded.
c7 = [("Pareto/NBD → with true season", pnbd, select(N, "ParetoNBD", "true_season")),
      ("Pareto/NBD with season → LSTM `ar_bounded`", select(N, "ParetoNBD", "true_season"),
       select(N, "LSTM", "ar_bounded-no_cluster"))]
write("7_seasonality", compare(c7))

# 7b. Shape correlation of predicted vs actual weekly totals, per cell, for three trees.
shape_trees = [("Pareto/NBD", pnbd), ("LSTM `ar_bounded`", select(N, "LSTM", "ar_bounded-no_cluster")),
               ("Transformer `ar_bounded`", select(N, "Transformer", "ar_bounded-no_cluster"))]
head = ["Rate", *(f"Churn {c:.0%}" for c in CHURNS), "Churn pooled"]
lines = ["Shape correlation, mean over panels: " + " / ".join(l for l, _ in shape_trees), "",
         "| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
for rate in RATES:
    cells = [f"{rate:.2f}"]
    for churn in CHURNS + [None]:
        cells.append(" / ".join(minus(f"{in_scope(f, rate, churn)['shape'].mean():+.2f}") for _, f in shape_trees))
    lines.append("| " + " | ".join(cells) + " |")
(CLAIMS / "7b_shape.md").write_text("\n".join(lines) + "\n")

# 9. Dead leakage: Pareto/NBD -> neural ar_bounded, on L_D and R_A.
c9 = [(m, pnbd, select(N, m, "ar_bounded-no_cluster")) for m in ("LSTM", "Transformer")]
write("9_dead_leakage", compare(c9, metrics=["l_d", "r_a"], levels=("l_d", "r_a")), metrics=("l_d", "r_a"))

# 10. Search budget: archived 10 / 20 trials -> 100 trials, no_ar.
c10 = [(f"{m}, {k} → 100 trials", select(N, m, "no_ar-no_cluster-small_search"), select(N, m, "no_ar-no_cluster"))
       for m, k in (("LSTM", 10), ("Transformer", 20))]
write("10_search_budget", compare(c10))

# 12. Cohort size: 1,000 -> 3,000 customers (different panels, so unpaired).
c12 = [(f"{m} {short(arm)}" if m != "ParetoNBD" else "Pareto/NBD", select("n1000", m, arm), select("n3000", m, arm))
       for m, arm in (("LSTM", "no_ar-no_cluster"), ("LSTM", "ar_bounded-no_cluster"), ("ParetoNBD", "-"))]
write("12_cohort_size", compare(c12, paired=False))


# 2. Error vs churn: per tree and rate, the mean at each churn level and the Spearman
#    correlation of the metric with churn over the rate's 40 panels.
def churn_trend(trees, metrics=("abs_bias", "mape")):
    rows = []
    for label, frame in trees:
        for rate in RATES:
            f = frame[frame.rate == rate]
            row = {"label": label, "rate": rate}
            for m in metrics:
                rho, p = stats.spearmanr(f.churn, f[m], nan_policy="omit")
                row |= {m + "_means": " / ".join(LEVEL[m].format(v) for v in f.groupby("churn")[m].mean()),
                        m + "_rho": rho, m + "_p": p}
            rows.append(row)
    t = pd.DataFrame(rows)
    adj = bh(t[[m + "_p" for m in metrics]].to_numpy().ravel()).reshape(len(t), -1)
    head = ["Tree", "Customers", "Rate"] + sum(([f"{LABEL[m]} at churn 20 / 40 / 60 / 80%",
                                                 f"ρ({LABEL[m]}, churn), p"] for m in metrics), [])
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for i, r in t.iterrows():
        tree, n = r.label
        cells = [tree, n, f"{r.rate:.2f}"]
        for k, m in enumerate(metrics):
            s = minus(f"{r[m + '_rho']:+.2f}, p {fmt_p(r[m + '_p'])}")
            cells += [minus(r[m + "_means"]), f"**{s}**" if adj[i, k] < 0.05 else s]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


trees2 = [(("Pareto/NBD", "1,000"), pnbd)] + [
    ((f"{m} {short(a)}", "1,000"), select(N, m, a))
    for m in ("LSTM", "Transformer") for a in ("no_ar-no_cluster", "ar_bounded-no_cluster")] + [
    (("Pareto/NBD", "3,000"), select("n3000", "ParetoNBD", "-"))] + [
    ((f"LSTM {short(a)}", "3,000"), select("n3000", "LSTM", a)) for a in ("no_ar-no_cluster", "ar_bounded-no_cluster")]
(CLAIMS / "2_churn_trend.md").write_text(churn_trend(trees2) + "\n")


# 8. Pareto/NBD's volume split: alive ratio R_A (vs 1) and dead leakage L_D (vs 0), with
#    |bias|, per cell. One-sample Wilcoxon of R_A - 1 and of L_D.
def level_table(frame, metrics, null):
    rows = []
    for rate, churn in scopes():
        f = in_scope(frame, rate, churn)
        row = {"rate": rate, "churn": churn, "n": int(len(f))}
        for m in metrics:
            v = f[m].dropna().to_numpy() - null.get(m, 0.0)
            est, lo, hi, p = paired_hl(v) if m in null else (np.nan, np.nan, np.nan, np.nan)
            half = stats.t.ppf(0.975, len(v) - 1) * f[m].std(ddof=1) / np.sqrt(len(v))
            row |= {m: f[m].mean(), m + "_lo": f[m].mean() - half, m + "_hi": f[m].mean() + half, m + "_p": p}
        rows.append(row)
    t = pd.DataFrame(rows)
    tested = [m for m in metrics if m in null]
    adj = bh(t[[m + "_p" for m in tested]].to_numpy().ravel()).reshape(len(t), -1)
    head = ["Rate", "Churn", "n"] + [f"{LABEL[m]} [95% CI]" + (f", p vs {null[m]:g}" if m in null else "")
                                     for m in metrics]
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for i, r in t.iterrows():
        cells = ["all" if pd.isna(r.rate) else f"{r.rate:.2f}",
                 "pooled" if pd.isna(r.churn) else f"{r.churn:.0%}", str(int(r.n))]
        for m in metrics:
            f = LEVEL[m]
            s = minus(f"{f.format(r[m])} [{f.format(r[m + '_lo'])}, {f.format(r[m + '_hi'])}]")
            if m in null:
                s += f", p {fmt_p(r[m + '_p'])}"
                s = f"**{s}**" if adj[i, tested.index(m)] < 0.05 else s
            cells.append(s)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


(CLAIMS / "8_pnbd_volume_split.md").write_text(
    level_table(pnbd, ["abs_bias", "r_a", "l_d", "rmse"], {"r_a": 1.0, "l_d": 0.0}) + "\n")


# 11. Hyperparameters: Spearman of each chosen hyperparameter with |bias| and MAPE,
#     pooled over the grid and within cells (both ranked within their rate x churn cell).
HP = {"LSTM": ["batch_size", "learning_rate", "lstm_hidden_size", "dense_units", "dropout"],
      "Transformer": ["batch_size", "learning_rate", "d_model", "nhead", "num_encoder_layers", "dropout"]}
rows = []
for m, arm in product(("LSTM", "Transformer"), ("no_ar-no_cluster", "ar_bounded-no_cluster")):
    f = select(N, m, arm)
    for hp in HP[m]:
        col = "param_" + hp
        if col not in f or f[col].nunique() < 2:
            continue
        row = {"tree": f"{m} {short(arm)}", "hp": hp}
        for out in ("abs_bias", "mape"):
            row[out + "_pool"], row[out + "_pool_p"] = stats.spearmanr(f[col], f[out])
            ranked = f.groupby(["rate", "churn"])[[col, out]].rank()
            row[out + "_within"], row[out + "_within_p"] = stats.spearmanr(ranked[col], ranked[out])
        rows.append(row)
t = pd.DataFrame(rows)
pcols = [c for c in t if c.endswith("_p")]
adj = bh(t[pcols].to_numpy().ravel()).reshape(len(t), -1)
head = ["Tree", "Hyperparameter", "ρ with \\|bias\\|, pooled", "ρ with \\|bias\\|, within cells",
        "ρ with MAPE, pooled", "ρ with MAPE, within cells"]
lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
for i, r in t.iterrows():
    cells = [r.tree, f"`{r.hp}`"]
    for k, key in enumerate(["abs_bias_pool", "abs_bias_within", "mape_pool", "mape_within"]):
        s = minus(f"{r[key]:+.2f}, p {fmt_p(r[key + '_p'])}")
        cells.append(f"**{s}**" if adj[i, pcols.index(key + "_p")] < 0.05 else s)
    lines.append("| " + " | ".join(cells) + " |")
(CLAIMS / "11_hyperparameters.md").write_text("\n".join(lines) + "\n")


# 13. RMSE as a ranking metric: per rate, the spread of each RMSE over the 13 trees.
rows = []
g13 = d[(d.cohort == N) & ~d.arm.isin(["true_season", "no_ar-no_cluster-small_search"])]
for rate in RATES:
    means = g13[g13.rate == rate].groupby(["model", "arm"])[["rmse_week", "rmse", "mape"]].mean()
    rows.append("| {:.2f} | {:.3f}–{:.3f} | {} of 13 | {:.2f}–{:.2f} | {:+.2f} |".format(
        rate, means.rmse_week.min(), means.rmse_week.max(),
        int((means.rmse_week.round(2) == means.rmse_week.round(2).mode()[0]).sum()),
        means.rmse.min(), means.rmse.max(),
        stats.spearmanr(means.rmse, means.mape).statistic).replace("-", "−").replace("−", "-", 0))
(CLAIMS / "13_rmse.md").write_text(
    "| Rate | Per-week RMSE, range over trees | Trees at the modal per-week RMSE (2 dp) "
    "| Customer-total RMSE, range | ρ(customer-total RMSE, MAPE) over trees |\n"
    "| --- | --- | --- | --- | --- |\n" + "\n".join(rows) + "\n")
print("wrote", sorted(p.name for p in CLAIMS.iterdir()))
