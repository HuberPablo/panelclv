"""Appendix of docs/Customer_Analysis_2.md: customers and transactions per customer group.

One table per window. A row is one §11 customer group; columns give its customers, its actual
holdout transactions, and each model's predicted holdout transactions (mean over runs) with
predicted ÷ actual.

    python .scratch/customer-analysis-2/tables_appendix.py .scratch/customer-analysis-2/results/groups.pkl
"""
import sys
import pandas as pd

d = pd.read_pickle(sys.argv[1])
d["model"] = d.model.replace({"LSTM+bounded52": "LSTM+bounded32"})
WIN = [("cdnow", "2y"), ("electronics", "2y"), ("gift", "2y"), ("multichannel", "2y"),
       ("electronics", "3y"), ("gift", "3y"), ("multichannel", "3y"), ("electronics", "5y")]
TITLE = {("cdnow", "2y"): "CDNOW — 39 weeks calibration, 39 weeks holdout",
         ("electronics", "5y"): "electronic_5y — 5 years calibration, 1 year holdout"}
MODELS = ["ParetoNBD", "ValendinLSTM", "LSTM+bounded32", "Transformer"]
ROWS = [("all", ["all"], "All customers"),
        ("ladder", ["1", "2", "3-4", "5+"], "Frequency (calibration purchases)"),
        ("ladder_x_recency", ["2 R<=13", "2 R14-52", "2 R>52", "3-4 R<=13", "3-4 R14-52", "3-4 R>52",
                              "5+ R<=13", "5+ R14-52", "5+ R>52"], "Recency within frequency (weeks silent)"),
        ("overdue", ["on schedule", "overdue 1-3x", "overdue >3x"], "Overdue (silence ÷ own usual gap)"),
        ("momentum", ["rising", "declining", "one purchase week"], "Momentum"),
        ("valendin", ["At Risk", "Opportunity", "Other"], "Valendin et al. (2022) groups")]
PRETTY = {"R<=13": "silent ≤ 13 w", "R14-52": "silent 14–52 w", "R>52": "silent > 52 w"}


def pretty(gname, gl, panel):
    if gname == "ladder": return f"{gl} purchase" + ("" if gl == "1" else "s")
    if gname == "ladder_x_recency":
        f, r = gl.split(" ")
        # CDNOW calibrates on 39 weeks, so its middle band stops at 38 weeks silent
        band = "silent 14–38 w" if (panel == "cdnow" and r == "R14-52") else PRETTY[r]
        return f"{f} purchases, {band}"
    return gl


for p, c in WIN:
    w = d[(d.panel == p) & (d.cal == c)]
    present = [m for m in MODELS if (w.model == m).any()]
    title = TITLE.get((p, c), f"{p} — {'2' if c == '2y' else '3'} years calibration, 1 year holdout")
    print(f"\n#### {title}\n")
    print("| customer group | customers | actual tx | " + " | ".join(
        {"ParetoNBD": "Pareto/NBD", "ValendinLSTM": "ValendinLSTM", "LSTM+bounded32": "LSTM + bounded32",
         "Transformer": "Transformer"}[m] for m in present) + " |")
    print("|---|---:|---:|" + "---:|" * len(present))
    for gname, groups, head in ROWS:
        print(f"| *{head}* | | |" + " |" * len(present))
        for gl in groups:
            x = w[(w.grouping == gname) & (w.group == gl)]
            if x.empty: continue
            n, act = int(x.n.iloc[0]), x.act.iloc[0]
            cells = []
            for m in present:
                pred = x[x.model == m].pred.mean()
                cells.append(f"{pred:.0f} ({pred / act:.2f}×)" if act > 0 else f"{pred:.0f}")
            mark = "†" if n < 30 else ""
            print(f"| {pretty(gname, gl, p)} | {n}{mark} | {act:.0f} | " + " | ".join(cells) + " |")
