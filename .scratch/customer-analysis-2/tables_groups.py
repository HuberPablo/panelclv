"""Tables of docs/Customer_Analysis_2.md §11 (the customer groups), from results/groups.pkl.

Run from the repo root with the project venv, after collect.py:
    python .scratch/customer-analysis-2/tables_groups.py .scratch/customer-analysis-2/results/groups.pkl

Model columns read the stored forecasts through groups.pkl. The "actuals only" tables (return
rates, within-group AUC of recency, the momentum within-cell test, and the two dropped
measures) are recomputed from the panels; their bootstraps are seeded.
"""
import sys, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
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
MODELS = ["ValendinLSTM", "LSTM+bounded32", "ParetoNBD"]
RNG = np.random.default_rng(0)


def lab(p, c): return "electronic_5y" if (p, c) == ("electronics", "5y") else f"{p} {c}"


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return np.nan, np.nan
    m = np.random.default_rng(0).choice(x, (4000, len(x))).mean(1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def sel(p, c, grouping, group, model):
    return d[(d.panel == p) & (d.cal == c) & (d.grouping == grouping) & (d.group == group) & (d.model == model)]


def flag(n): return f"{n}" if n >= 30 else f"{n}†"


def auc(score, pos):
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 < 3 or n0 < 3 or np.ptp(score) == 0: return np.nan
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def group_table(grouping, order, title):
    """n, return rate, actual tx/customer, each model's predicted -> pred/act [95% CI over runs]."""
    print(f"\n#### {title}\n")
    print("| window | group | n | return rate | actual tx / customer | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |")
    print("|---|---|---:|---:|---:|---|---|---|")
    for p, c in WIN:
        for gl in order:
            base = sel(p, c, grouping, gl, "ParetoNBD")
            if base.empty: continue
            n = base.n.iloc[0]
            cells = []
            for m in MODELS:
                x = sel(p, c, grouping, gl, m)
                if x.empty: cells.append("not run"); continue
                if base.act.iloc[0] == 0:
                    cells.append(f"{x.ppc.mean():.2f}"); continue
                r = x.pred / x.act; lo, hi = ci(r)
                cells.append(f"{x.ppc.mean():.2f} → **{r.mean():.2f}×** [{lo:.2f}, {hi:.2f}]")
            print(f"| {lab(p, c)} | {gl} | {flag(n)} | {100 * base.ret.iloc[0] / n:.0f}% | {base.apc.iloc[0]:.2f} | " + " | ".join(cells) + " |")


# ---------------------------------------------------------------- panel-level (actuals only)
PANEL = {}
for p, c in WIN:
    data = b.build_data(p, c)
    cc = calibration_counts(data); h = holdout_actuals(data)
    T = cc.shape[1]; act = cc > 0; F = cc.sum(1); H = h.sum(1); nw = act.sum(1)
    R = np.argmax(act[:, ::-1], 1); first = np.argmax(act, 1); ten = T - first
    mom = np.full(len(F), np.nan)
    for i in np.flatnonzero(nw >= 2):
        w = np.flatnonzero(act[i]); mom[i] = (w >= first[i] + ten[i] / 2).mean()
    ladder = np.select([F <= 1, F == 2, F <= 4], ["1", "2", "3-4"], "5+")
    band = np.select([R <= 13, R <= 52], ["R<=13", "R14-52"], "R>52")
    PANEL[(p, c)] = dict(cc=cc, act=act, T=T, F=F, H=H, nw=nw, R=R, first=first, mom=mom,
                         cell=np.char.add(ladder.astype(str), band.astype(str)), ladder=ladder)


def within_cell(P, flag_, elig, y, boot=500):
    """Weighted mean over frequency x recency cells of mean(y | flag) - mean(y | not flag).

    Only cells with >= 10 customers on each side count; weights are cell sizes. The interval
    resamples customers (seeded)."""
    cell = P["cell"]
    def est(f_, e_, y_, cnt=None):
        diffs, wts = [], []
        for s in np.unique(cell[e_]):
            m = e_ & (cell == s); a = m & f_; z = m & ~f_
            if a.sum() < 10 or z.sum() < 10: continue
            if cnt is None:
                diffs.append(y_[a].mean() - y_[z].mean()); wts.append(m.sum())
            else:
                ca, cz = cnt[a], cnt[z]
                if ca.sum() == 0 or cz.sum() == 0: continue
                diffs.append((y_[a] * ca).sum() / ca.sum() - (y_[z] * cz).sum() / cz.sum()); wts.append(cnt[m].sum())
        return (np.average(diffs, weights=wts), int(sum(wts) if cnt is None else 0)) if diffs else (np.nan, 0)
    e, n = est(flag_, elig, y)
    if np.isnan(e): return e, np.nan, np.nan, 0
    bs = [est(flag_, elig, y, np.bincount(RNG.choice(len(y), len(y)), minlength=len(y)))[0] for _ in range(boot)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return e, lo, hi, n


# ================================================================== 11.1 frequency ladder
print("## 11.1 Frequency ladder")
print("\n#### Actuals only: return rate and within-group AUC of recency (fewer weeks silent ranks higher)\n")
print("| window | 1 purchase | 2 | 3–4 | 5+ |")
print("|---|---|---|---|---|")
for p, c in WIN:
    P = PANEL[(p, c)]; cells = []
    for gl in ["1", "2", "3-4", "5+"]:
        m = P["ladder"] == gl
        cells.append(f"{100 * (P['H'][m] > 0).mean():.0f}% · AUC {auc(-P['R'][m], P['H'][m] > 0):.2f} ({flag(m.sum())})")
    print(f"| {lab(p, c)} | " + " | ".join(cells) + " |")
group_table("ladder", ["1", "2", "3-4", "5+"], "Frequency ladder: forecasts")

print("\n#### Within-group ranking: AUC of the forecast (holdout buyer vs non-buyer), V / B / P\n")
print("| window | 1 | 2 | 3–4 | 5+ |")
print("|---|---|---|---|---|")
for p, c in WIN:
    cells = []
    for gl in ["1", "2", "3-4", "5+"]:
        cells.append(" / ".join(f"{sel(p, c, 'ladder', gl, m).auc.mean():.2f}" if not sel(p, c, 'ladder', gl, m).empty else "–" for m in MODELS))
    print(f"| {lab(p, c)} | " + " | ".join(cells) + " |")

# ================================================================== 11.2 recency within frequency
print("\n## 11.2 Recency within frequency")
print("\n#### Predicted tx / customer by recency band, repeaters (actual first)\n")
print("| window | frequency | recency | n | actual | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |")
print("|---|---|---|---:|---:|---:|---:|---:|")
for p, c in WIN:
    for fq in ["2", "3-4", "5+"]:
        for rb in ["R<=13", "R14-52", "R>52"]:
            gl = f"{fq} {rb}"
            base = sel(p, c, "ladder_x_recency", gl, "ParetoNBD")
            if base.empty: continue
            cells = [("not run" if sel(p, c, "ladder_x_recency", gl, m).empty else f"{sel(p, c, 'ladder_x_recency', gl, m).ppc.mean():.2f}") for m in MODELS]
            print(f"| {lab(p, c)} | {fq} | {rb} | {flag(base.n.iloc[0])} | {base.apc.iloc[0]:.2f} | " + " | ".join(cells) + " |")

# ================================================================== 11.3 overdue
print("\n## 11.3 Overdue (a reading of recency)")
group_table("overdue", ["on schedule", "overdue 1-3x", "overdue >3x"], "Overdue groups, customers with >= 2 purchase weeks")
print("\n#### Does overdue add information beyond frequency x recency? Δ return rate (overdue >3x − on schedule) within cells\n")
print("| window | Δ return, pts [95% CI] | customers in comparable cells |")
print("|---|---|---:|")
for p, c in WIN:
    P = PANEL[(p, c)]
    act, T, first, R, nw = P["act"], P["T"], P["first"], P["R"], P["nw"]
    last = T - 1 - R
    gap = np.where(nw > 1, (last - first) / np.maximum(nw - 1, 1), np.nan)
    od = R / gap
    elig = np.isfinite(od) & ((od > 3) | (od <= 1))
    e, lo, hi, n = within_cell(P, od > 3, elig, (P["H"] > 0).astype(float))
    print(f"| {lab(p, c)} | " + ("no cell has ≥ 10 on each side" if n == 0 else f"{100 * e:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]") + f" | {n} |")

# ================================================================== 11.4 momentum
print("\n## 11.4 Momentum")
group_table("momentum", ["rising", "declining"], "Momentum groups, customers with >= 2 purchase weeks")
print("\n#### Within frequency x recency cells: holdout tx / customer, rising − declining\n")
print("| window | rising / declining | actual [95% CI over customers] | ValendinLSTM [95% CI over runs] | LSTM + bounded32 | Pareto/NBD |")
print("|---|---|---|---|---|---|")
MC = {}
for p, c in WIN:
    P = PANEL[(p, c)]
    elig = np.isfinite(P["mom"]); up = elig & (P["mom"] >= 0.5)
    e, lo, hi, n = within_cell(P, up, elig, P["H"].astype(float))
    cells = []
    for m in MODELS:
        x = d[(d.panel == p) & (d.cal == c) & (d.grouping == "momentum_cell") & (d.model == m) & (d.group != "n/a")]
        if x.empty: cells.append("not run"); continue
        x = x.assign(cellkey=x.group.str.rsplit("|", n=1).str[0], side=x.group.str.rsplit("|", n=1).str[1])
        per = []
        for rep, xr in x.groupby("rep"):
            diffs, wts = [], []
            for ck, xc in xr.groupby("cellkey"):
                if set(xc.side) != {"rising", "declining"}: continue
                ri = xc[xc.side == "rising"].iloc[0]; de = xc[xc.side == "declining"].iloc[0]
                if ri.n < 10 or de.n < 10: continue
                diffs.append(ri.pred / ri.n - de.pred / de.n); wts.append(ri.n + de.n)
            per.append(np.average(diffs, weights=wts))
        MC[(p, c, m)] = np.array(per)
        l2, h2 = ci(per)
        cells.append(f"{np.mean(per):+.2f} [{l2:+.2f}, {h2:+.2f}]")
    print(f"| {lab(p, c)} | {up.sum()} / {(elig & ~up).sum()} | {e:+.2f} [{lo:+.2f}, {hi:+.2f}] | " + " | ".join(cells) + " |")

print("\n#### Momentum threshold sensitivity (actuals only): within-cell Δ return rate, rising − declining, pts [95% CI]\n")
print("| window | cut-off 0.4 | 0.5 | 0.6 |")
print("|---|---|---|---|")
for p, c in WIN:
    P = PANEL[(p, c)]; elig = np.isfinite(P["mom"]); cells = []
    for cut_ in (0.4, 0.5, 0.6):
        e, lo, hi, n = within_cell(P, elig & (P["mom"] >= cut_), elig, (P["H"] > 0).astype(float))
        cells.append(f"{100 * e:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]")
    print(f"| {lab(p, c)} | " + " | ".join(cells) + " |")

# ================================================================== 11.5 Valendin groups
print("\n## 11.5 Valendin et al. groups")
group_table("valendin", ["Opportunity"], "Opportunity: more holdout than calibration transactions")
group_table("at_risk_by_freq", ["At Risk, F 1", "At Risk, F 2", "At Risk, F 3-4", "At Risk, F 5+"],
            "At Risk (≥ average calibration frequency, no holdout purchase), by frequency: predicted tx / customer (actual is 0)")
print("\n#### At Risk: predicted tx / customer, model − Pareto/NBD (lower is better) [95% CI]\n")
print("| window | ValendinLSTM | LSTM + bounded32 |")
print("|---|---|---|")
for p, c in WIN:
    cells = []
    pn = sel(p, c, "valendin", "At Risk", "ParetoNBD").ppc.values
    for m in MODELS[:2]:
        x = sel(p, c, "valendin", "At Risk", m)
        if x.empty: cells.append("not run"); continue
        e = effect(x.ppc.values, pn, paired=False, metric="", panel=p)
        s = "**" if (e.lo > 0 or e.hi < 0) else ""
        cells.append(f"{s}{e.delta:+.2f} [{e.lo:+.2f}, {e.hi:+.2f}]{s}")
    print(f"| {lab(p, c)} | " + " | ".join(cells) + " |")

# ================================================================== 11.6 dropped measures
print("\n## 11.6 Considered and dropped (actuals only)")
_q = {}
def hp(w, N):
    n = len(w); x = np.diff(np.r_[-1, w, N]) / (N + 1); x = x[x > 0]
    return 1 + np.sum(x * np.log(x)) / np.log(n + 1)
def hp_thresh(n, N):
    if (n, N) not in _q:
        _q[(n, N)] = np.quantile([hp(np.sort(RNG.choice(N, n, replace=False)), N) for _ in range(2000)], 0.95)
    return _q[(n, N)]
def burst(g):
    m, s = g.mean(), g.std(); return (s - m) / (s + m) if (s + m) > 0 else np.nan
print("\n| window | ≥ 3 purchase weeks | burstiness B: 1st-half vs 2nd-half Spearman (customers) | clumpy C share of repeaters | within-cell Δ return, clumpy − not, pts [95% CI] |")
print("|---|---:|---|---:|---|")
for p, c in WIN:
    P = PANEL[(p, c)]; act, T = P["act"], P["T"]
    C = np.full(len(P["F"]), np.nan); pairs = []
    for i in range(len(C)):
        w = np.flatnonzero(act[i])
        if len(w) >= 2: C[i] = float(hp(w, T) > hp_thresh(len(w), T))
        w1, w2 = w[w < T // 2], w[w >= T // 2]
        if len(w1) >= 3 and len(w2) >= 3: pairs.append((burst(np.diff(w1)), burst(np.diff(w2))))
    pairs = np.array(pairs)
    stab = f"{spearmanr(pairs[:, 0], pairs[:, 1]).correlation:.2f} ({len(pairs)})" if len(pairs) > 10 else f"– ({len(pairs)})"
    elig = np.isfinite(C)
    e, lo, hi, n = within_cell(P, elig & (C == 1), elig, (P["H"] > 0).astype(float))
    print(f"| {lab(p, c)} | {100 * (P['nw'] >= 3).mean():.0f}% | {stab} | {100 * np.nanmean(C):.0f}% | {100 * e:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] |")
