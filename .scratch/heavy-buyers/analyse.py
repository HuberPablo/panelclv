"""Tables for the heavy-buyer question, from results/heavy.pkl.

A. Within one model, is the heavy group (>k calibration purchases) forecast better than
   the light group? Paired by forecast: the same forecast is scored on both groups.
B. Within the heavy group, which model does best? Each model against Pareto/NBD,
   independent bootstrap (different searches / refits).
Claims follow docs/statistical-protocol.md via evaluation.effects.effect.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from panelclv.evaluation.effects import effect

df = pd.read_pickle(ROOT / ".scratch/heavy-buyers/results/heavy.pkl")
df["ratio"] = df.pred / df.act
df["abs_bias"] = df.bias_percent.abs()
df["window"] = df.panel + " " + df.cal
MIN_N = 20
ORDER = ["ValendinLSTM", "LSTM+bounded32", "LSTM+bounded52", "LSTM+log", "LSTM+ratio",
         "Transformer", "Transformer-noAR", "ParetoNBD"]
WIN = ["cdnow 2y", "electronics 2y", "gift 2y", "multichannel 2y",
       "electronics 3y", "gift 3y", "multichannel 3y", "electronics 5y"]

def fmt(e, unit=""):
    star = "**" if (e.lo > 0 or e.hi < 0) else ""
    return f"{star}{e.delta:+.2f}{unit}{star} [{e.lo:+.2f}, {e.hi:+.2f}]"

out = []
P = out.append

# ---- group sizes and descriptive accuracy --------------------------------------------
P("## 1. Who the heavy buyers are\n")
P("| window | k | heavy n | light n | heavy return rate | light return rate | heavy actual tx/cust | light actual tx/cust |")
P("|---|---:|---:|---:|---:|---:|---:|---:|")
for w in WIN:
    for k in (5, 10, 20):
        d = df[(df.window == w) & (df.k == k) & (df.model == "ParetoNBD") & (df.rep == 0)]
        hv, lt = d[d.side == "heavy"], d[d.side == "light"]
        if hv.empty:
            P(f"| {w} | >{k} | 0 or 1 | — | | | | |"); continue
        hv, lt = hv.iloc[0], lt.iloc[0]
        P(f"| {w} | >{k} | {hv.n}{'†' if hv.n < MIN_N else ''} | {lt.n} | {hv.ret:.0%} | {lt.ret:.0%} | {hv.act/hv.n:.2f} | {lt.act/lt.n:.2f} |")
P("\n† fewer than 20 customers: shown, but no claim is made on it.\n")

# ---- A: heavy vs light within a model -----------------------------------------------
P("## 2. Heavy group vs the rest, within one model (paired by forecast)\n")
P("Δ = heavy − light, mean over forecasts, 95% percentile-bootstrap CI; bold = CI excludes 0.")
P("Negative Δ MAPE / rel. MAE = the heavy group is forecast more accurately. "
  "Positive Δ Spearman = customers inside the heavy group are ranked better.\n")
for k in (5, 10, 20):
    P(f"### >{k} calibration purchases\n")
    P("| window | model | n fc | heavy n | pred/act heavy | pred/act light | Δ agg. MAPE (pts) | Δ rel. MAE | Δ Spearman | Δ AUC |")
    P("|---|---|---:|---:|---:|---:|---|---|---|---|")
    for w in WIN:
        for m in ORDER:
            d = df[(df.window == w) & (df.k == k) & (df.model == m)]
            if d.empty: continue
            hv = d[d.side == "heavy"].set_index("rep").sort_index()
            lt = d[d.side == "light"].set_index("rep").sort_index()
            if hv.empty or hv.n.iloc[0] < MIN_N:
                P(f"| {w} | {m} | {len(lt)} | {0 if hv.empty else hv.n.iloc[0]}† | | {lt.ratio.mean():.2f} | | | | |"); continue
            cells = []
            for met in ("mape_aggregate", "rel_mae", "rho", "auc"):
                e = effect(hv[met].values, lt[met].values, paired=True, metric=met, panel=w)
                cells.append(fmt(e))
            P(f"| {w} | {m} | {len(hv)} | {hv.n.iloc[0]} | {hv.ratio.mean():.2f} | {lt.ratio.mean():.2f} | " + " | ".join(cells) + " |")
    P("")

# ---- B: models within the heavy group -----------------------------------------------
P("## 3. Which model forecasts the heavy group best (each model vs Pareto/NBD, independent)\n")
P("Means over forecasts inside the heavy group. Δ = model − Pareto/NBD with 95% CI; bold = supported. "
  "Bias is the group's signed % error (pred − act)/act.\n")
for k in (5, 10, 20):
    P(f"### >{k} calibration purchases\n")
    P("| window | model | n fc | bias % | agg. MAPE | Spearman | AUC | rel. MAE | Δ MAPE vs P/NBD | Δ Spearman vs P/NBD | Δ rel. MAE vs P/NBD |")
    P("|---|---|---:|---:|---:|---:|---:|---:|---|---|---|")
    for w in WIN:
        base = df[(df.window == w) & (df.k == k) & (df.model == "ParetoNBD") & (df.side == "heavy")]
        if base.empty or base.n.iloc[0] < MIN_N:
            P(f"| {w} | — | | | | | | | heavy n = {0 if base.empty else base.n.iloc[0]}† | | |"); continue
        for m in ORDER:
            d = df[(df.window == w) & (df.k == k) & (df.model == m) & (df.side == "heavy")]
            if d.empty: continue
            row = f"| {w} | {m} | {len(d)} | {d.bias_percent.mean():+.0f} | {d.mape_aggregate.mean():.1f} | {d.rho.mean():.3f} | {d.auc.mean():.3f} | {d.rel_mae.mean():.2f} |"
            if m == "ParetoNBD":
                row += " baseline | baseline | baseline |"
            else:
                cells = [fmt(effect(d[met].values, base[met].values, paired=False, metric=met, panel=w))
                         for met in ("mape_aggregate", "rho", "rel_mae")]
                row += " " + " | ".join(cells) + " |"
            P(row)
    P("")

# ---- C: does any model's advantage grow with the threshold? ---------------------------
P("## 4. Model vs Pareto/NBD on the light side, for contrast (Δ Spearman, Δ agg. MAPE)\n")
P("| window | model | side | Δ Spearman vs P/NBD >5 | >10 | >20 | Δ agg. MAPE vs P/NBD >5 | >10 | >20 |")
P("|---|---|---|---|---|---|---|---|---|")
for w in WIN:
    for m in ORDER:
        if m == "ParetoNBD": continue
        for side in ("heavy", "light"):
            cells_r, cells_m = [], []
            any_ = False
            for k in (5, 10, 20):
                d = df[(df.window == w) & (df.k == k) & (df.model == m) & (df.side == side)]
                base = df[(df.window == w) & (df.k == k) & (df.model == "ParetoNBD") & (df.side == side)]
                if d.empty or base.empty or d.n.iloc[0] < MIN_N:
                    cells_r.append("†"); cells_m.append("†"); continue
                any_ = True
                cells_r.append(fmt(effect(d.rho.values, base.rho.values, paired=False, metric="rho", panel=w)))
                cells_m.append(fmt(effect(d.mape_aggregate.values, base.mape_aggregate.values, paired=False, metric="mape", panel=w)))
            if any_:
                P(f"| {w} | {m} | {side} | " + " | ".join(cells_r + cells_m) + " |")

text = "\n".join(out)
Path(ROOT / ".scratch/heavy-buyers/results/tables.md").write_text(text)
print(text)
