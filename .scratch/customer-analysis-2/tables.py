"""Every table of docs/Customer_Analysis_2.md, from results/groups.pkl (written by collect.py).

Run from the repo root with the project venv:
    python .scratch/customer-analysis-2/collect.py .scratch/customer-analysis-2/results/groups.pkl
    python .scratch/customer-analysis-2/tables.py .scratch/customer-analysis-2/results/groups.pkl
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path[:0] = ["src", "scripts"]
from panelclv.evaluation.effects import effect
import run_real_panel_benchmarks as b
from panelclv.data_preparation.target_channel import calibration_counts, holdout_actuals

d = pd.read_pickle(sys.argv[1])
d["model"] = d.model.replace({"LSTM+bounded52": "LSTM+bounded32"})   # 5y substitute, flagged in the doc
d["ppc"] = d.pred / d.n
d["apc"] = d.act / d.n
WIN = [("cdnow", "2y"), ("electronics", "2y"), ("electronics", "3y"), ("electronics", "5y"),
       ("gift", "2y"), ("gift", "3y"), ("multichannel", "2y"), ("multichannel", "3y")]
LABEL = {("electronics", "5y"): "electronic_5y"}
MODELS = ["ValendinLSTM", "LSTM+bounded32", "ParetoNBD"]


def lab(p, c): return LABEL.get((p, c), f"{p} {c}")


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return np.nan, np.nan
    m = np.random.default_rng(0).choice(x, (4000, len(x))).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def sel(p, c, grouping, group, model):
    return d[(d.panel == p) & (d.cal == c) & (d.grouping == grouping) & (d.group == group) & (d.model == model)]


def group_table(title, picks):
    """picks: (p, c) -> (grouping, group) defining the group on that window."""
    print(f"\n#### {title}\n")
    print("| panel | n customers (% of cohort) | actual tx / customer | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |")
    print("|---|---:|---:|---|---|---|")
    for p, c in WIN:
        gr, gl = picks(p, c)
        base = sel(p, c, gr, gl, "ParetoNBD")
        if base.empty: continue
        N = sel(p, c, "all", "all", "ParetoNBD").n.iloc[0]
        cells = []
        for m in MODELS:
            x = sel(p, c, gr, gl, m)
            if x.empty: cells.append("not run"); continue
            r = x.pred / x.act
            lo, hi = ci(r)
            cells.append(f"{x.ppc.mean():.2f} → **{r.mean():.2f}×** [{lo:.2f}, {hi:.2f}]")
        print(f"| {lab(p, c)} | {base.n.iloc[0]} ({100 * base.n.iloc[0] / N:.0f}%) | {base.apc.iloc[0]:.2f} | " + " | ".join(cells) + " |")


def decay_table(title, picks):
    print(f"\n#### {title}\n")
    print("| panel | actual | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |")
    print("|---|---:|---:|---:|---:|")
    for p, c in WIN:
        gr, gl = picks(p, c)
        base = sel(p, c, gr, gl, "ParetoNBD")
        if base.empty: continue
        act = base.act_q4.iloc[0] / base.act_q1.iloc[0] if base.act_q1.iloc[0] > 0 else np.nan
        cells = []
        for m in MODELS:
            x = sel(p, c, gr, gl, m)
            cells.append("not run" if x.empty else f"{(x.pred_q4 / x.pred_q1).mean():.2f}")
        print(f"| {lab(p, c)} | {act:.2f} | " + " | ".join(cells) + " |")


one_timer = lambda p, c: ("freq", "1")
recent = lambda p, c: ("recency", "0-4w")
hot = lambda p, c: ("RxF", "F5+ R<=13w")
cold = lambda p, c: ("RxF", "F1 R14-52w") if p == "cdnow" else ("RxF", "F1 R>52w")

print("## T1 one-timers")
group_table("One-timers: exactly one purchase in calibration", one_timer)
print("\n## T2 recent / recent-heavy")
group_table("Recent buyers: last purchase in the final 5 calibration weeks (0–4 weeks before holdout)", recent)
group_table("Recent heavy buyers: ≥5 purchases in calibration AND last purchase ≤13 weeks before holdout", hot)
decay_table("Recent heavy buyers: forecast in the last holdout quarter ÷ first holdout quarter", hot)
decay_table("One-timers: forecast in the last holdout quarter ÷ first holdout quarter", one_timer)

# --- T3 separation, components shown
print("\n## T3 separation\n")
print("| panel | 'cold' group (n) | 'hot' group (n) | actual: hot / cold per customer | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |")
print("|---|---|---|---|---|---|---|")
for p, c in WIN:
    gh, lh = hot(p, c); gc, lc = cold(p, c)
    H = sel(p, c, gh, lh, "ParetoNBD"); C = sel(p, c, gc, lc, "ParetoNBD")
    cells = []
    for m in MODELS:
        xh = sel(p, c, gh, lh, m).set_index("rep"); xc = sel(p, c, gc, lc, m).set_index("rep")
        if xh.empty: cells.append("not run"); continue
        s = xh.ppc / xc.ppc
        cells.append(f"{xh.ppc.mean():.2f} / {xc.ppc.mean():.2f} = **{s.mean():.1f}×**")
    print(f"| {lab(p, c)} | {lc} ({C.n.iloc[0]}) | {lh} ({H.n.iloc[0]}) | "
          f"{H.apc.iloc[0]:.2f} / {C.apc.iloc[0]:.2f} = **{H.apc.iloc[0] / C.apc.iloc[0]:.1f}×** | " + " | ".join(cells) + " |")

# --- T4 recency gradient among repeat buyers
print("\n## T4 recency gradient (predicted tx / customer; actual first)\n")
for fq in ["F2-4", "F5+"]:
    print(f"\n#### {fq} purchases in calibration\n")
    print("| panel | recency band | n | actual | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for p, c in WIN:
        for rb in ["R<=13w", "R14-52w", "R>52w"]:
            gl = f"{fq} {rb}"
            base = sel(p, c, "RxF", gl, "ParetoNBD")
            if base.empty: continue
            cells = []
            for m in MODELS:
                x = sel(p, c, "RxF", gl, m)
                cells.append("not run" if x.empty else f"{x.ppc.mean():.2f}")
            print(f"| {lab(p, c)} | {rb} | {base.n.iloc[0]} | {base.apc.iloc[0]:.2f} | " + " | ".join(cells) + " |")

# --- T4b AUC/rho for repeat buyers 2-4 (cdnow inversion)
print("\n## T4b within-group discrimination, F2-4 repeat buyers (AUC holdout buyer vs non-buyer; Spearman)\n")
print("| panel | ValendinLSTM AUC / ρ | LSTM + bounded32 AUC / ρ | Pareto/NBD AUC / ρ |")
print("|---|---|---|---|")
for p, c in WIN:
    cells = []
    for m in MODELS:
        x = d[(d.panel == p) & (d.cal == c) & (d.grouping == "freq") & (d.group.isin(["2", "3-4"])) & (d.model == m)]
        if x.empty: cells.append("not run"); continue
        # weight the two bands by customers
        w = x.groupby("group").n.first()
        a = sum(x[x.group == g].auc.mean() * w[g] for g in w.index) / w.sum()
        r = sum(x[x.group == g].rho.mean() * w[g] for g in w.index) / w.sum()
        cells.append(f"{a:.2f} / {r:.2f}")
    print(f"| {lab(p, c)} | " + " | ".join(cells) + " |")

# --- T5 panel characteristics vs differentiation
print("\n## T5 panel characteristics and differentiation\n")
print("| panel | customers | calibration weeks | zero cells, calibration | tx / customer, calibration | one-timers | ≥5 tx | silent >52 w at cut-off | actual separation | separation captured: V / B / P | Spearman: V / B / P | forecast CV: V / B / P |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
for p, c in WIN:
    data = b.build_data(p, c)
    cc = calibration_counts(data); h = holdout_actuals(data)
    f = cc.sum(1); ws = np.argmax((cc > 0)[:, ::-1], 1)
    gh, lh = hot(p, c); gc, lc = cold(p, c)
    act_sep = sel(p, c, gh, lh, "ParetoNBD").apc.iloc[0] / sel(p, c, gc, lc, "ParetoNBD").apc.iloc[0]
    cap, rho, cv = [], [], []
    for m in MODELS:
        xh = sel(p, c, gh, lh, m).set_index("rep"); xc = sel(p, c, gc, lc, m).set_index("rep")
        a = sel(p, c, "all", "all", m)
        if xh.empty: cap.append("–"); rho.append("–"); cv.append("–"); continue
        cap.append(f"{100 * (xh.ppc / xc.ppc).mean() / act_sep:.0f}%")
        rho.append(f"{a.rho.mean():.2f}"); cv.append(f"{a.cv.mean():.2f}")
    print(f"| {lab(p, c)} | {len(f)} | {cc.shape[1]} | {100 * (cc == 0).mean():.1f}% | {f.mean():.2f} | "
          f"{100 * (f == 1).mean():.0f}% | {100 * (f >= 5).mean():.0f}% | {100 * (ws > 52).mean():.0f}% | {act_sep:.1f}× | "
          + " / ".join(cap) + " | " + " / ".join(rho) + " | " + " / ".join(cv) + " |")

# --- T6 headline effects vs Pareto/NBD
print("\n## T6 effects vs Pareto/NBD (independent bootstrap)\n")
d["rmse_tot"] = np.sqrt(d.sse_tot / d.n)
def fmt(e):
    s = "**" if (e.lo > 0 or e.hi < 0) else ""
    return f"{s}{e.delta:+.2f} [{e.lo:+.2f}, {e.hi:+.2f}]{s}"
print("| panel | model (n vs 20) | log2 separation | log2 pred/act, one-timers | log2 pred/act, recent buyers 0–4 w | RMSE of totals, recent buyers | RMSE of totals, one-timers |")
print("|---|---|---|---|---|---|---|")
for p, c in WIN:
    def rep(m, gr, gl, col): return sel(p, c, gr, gl, m).set_index("rep")[col]
    def sep(m):
        gh, lh = hot(p, c); gc, lc = cold(p, c)
        return np.log2(rep(m, gh, lh, "ppc") / rep(m, gc, lc, "ppc")).values
    def lr(m, gr, gl):
        x = sel(p, c, gr, gl, m); return np.log2(x.pred / x.act).values
    for m in MODELS[:2]:
        if sel(p, c, "all", "all", m).empty: continue
        es = [effect(sep(m), sep("ParetoNBD"), paired=False, metric="", panel=p),
              effect(lr(m, "freq", "1"), lr("ParetoNBD", "freq", "1"), paired=False, metric="", panel=p),
              effect(lr(m, "recency", "0-4w"), lr("ParetoNBD", "recency", "0-4w"), paired=False, metric="", panel=p),
              effect(rep(m, "recency", "0-4w", "rmse_tot").values, rep("ParetoNBD", "recency", "0-4w", "rmse_tot").values, paired=False, metric="", panel=p),
              effect(rep(m, "freq", "1", "rmse_tot").values, rep("ParetoNBD", "freq", "1", "rmse_tot").values, paired=False, metric="", panel=p)]
        n = len(sel(p, c, "all", "all", m))
        print(f"| {lab(p, c)} | {m} ({n}) | " + " | ".join(fmt(e) for e in es) + " |")

# --- T7 bias decomposition
print("\n## T7 contribution to aggregate bias (percentage points of the panel's actual total)\n")
tot = d[d.grouping == "all"].set_index(["panel", "cal", "model", "rep"]).act
print("| panel | model | total bias | one-timers | F2-4 | F5+ |")
print("|---|---|---:|---:|---:|---:|")
for p, c in WIN:
    for m in MODELS:
        a = sel(p, c, "all", "all", m)
        if a.empty: continue
        T = a.act.iloc[0]
        def contrib(gr, gls):
            x = d[(d.panel == p) & (d.cal == c) & (d.model == m) & (d.grouping == gr) & (d.group.isin(gls))]
            return 100 * (x.pred - x.act).groupby(x.rep).sum().mean() / T
        print(f"| {lab(p, c)} | {m} | {100 * ((a.pred - a.act) / T).mean():+.0f} | {contrib('freq', ['1']):+.0f} | "
              f"{contrib('freq', ['2', '3-4']):+.0f} | {contrib('freq', ['5-9', '10+']):+.0f} |")

# --- T8 calibration length, same panel: log2(separation captured) 3y - 2y for LSTM + bounded32
print("\n## T8 captured separation, 3y vs 2y (LSTM + bounded32, independent bootstrap)\n")
print("| panel | captured 2y | captured 3y | Δ log2 captured (3y − 2y) [95% CI] |")
print("|---|---:|---:|---|")
for p in ["electronics", "gift", "multichannel"]:
    vals = {}
    for c in ["2y", "3y"]:
        gh, lh = hot(p, c); gc, lc = cold(p, c)
        act = sel(p, c, gh, lh, "ParetoNBD").apc.iloc[0] / sel(p, c, gc, lc, "ParetoNBD").apc.iloc[0]
        xh = sel(p, c, gh, lh, "LSTM+bounded32").set_index("rep"); xc = sel(p, c, gc, lc, "LSTM+bounded32").set_index("rep")
        vals[c] = np.log2((xh.ppc / xc.ppc) / act).values
    e = effect(vals["3y"], vals["2y"], paired=False, metric="", panel=p)
    print(f"| {p} | {100 * np.mean(2 ** vals['2y']):.0f}% | {100 * np.mean(2 ** vals['3y']):.0f}% | {fmt(e)} |")
