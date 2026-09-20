"""Did the studies that trained longer rank customers any better?

Electronics only (the panel whose ranking collapsed). For every archived study of the
frozen benchmark and of the developed LSTM/Transformer arms, recompute the per-customer
Spearman from the stored forecast and join it to the winning trial's best_epoch.
"""
import glob, os, re, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from pathlib import Path

# The repo root, derived so these run unchanged on a rented box, and an
# output folder beside this script.
REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO / "scripts"))
from run_real_panel_benchmarks import build_data                    # noqa: E402
from panelclv.data_preparation.target_channel import holdout_actuals  # noqa: E402
from panelclv.studies import load_model_predictions                 # noqa: E402

ROOT = REPO / "Studies"
data = build_data("electronics", "2y")
actual_tot = holdout_actuals(data).sum(axis=1)                      # (N,) per-customer totals
print("cohort:", actual_tot.shape, "holdout transactions:", actual_tot.sum())

rows = []
for sdir in sorted(ROOT.glob("*electronics*")):
    if "cal3y" in sdir.name:
        continue
    for mdir in sdir.glob("*"):
        if not (mdir / "Predictions").exists():
            continue
        for tfile in mdir.glob("Optuna_Studies/*/*_trials.csv"):
            sid = int(re.search(r"study_(\d+)", tfile.name).group(1))
            tr = pd.read_csv(tfile)
            tr = tr[tr.state == "COMPLETE"]
            if tr.empty or "user_attrs_best_epoch" not in tr:
                continue
            win = tr.loc[tr.value.idxmin()]
            try:
                pred, _ = load_model_predictions(mdir, study=sid)
            except Exception:
                continue
            if pred.shape[0] != actual_tot.shape[0]:
                continue
            rho = spearmanr(pred.sum(axis=1), actual_tot).statistic
            rows.append(dict(suite=sdir.name, model=mdir.name, study=sid,
                             best_epoch=float(win.user_attrs_best_epoch),
                             val_loss=float(win.value), spearman=rho,
                             pred_sd=float(pred.sum(axis=1).std())))

d = pd.DataFrame(rows)
d["arm"] = d.suite.str.replace(r"__[a-z]$", "", regex=True)
d.to_csv(RESULTS / "spearman_electronics.csv", index=False)
print("studies scored:", len(d))

print("\nSpearman(best_epoch, per-customer rank quality), by model family:")
for m, g in d.groupby("model"):
    if len(g) < 15: continue
    print(f"  {m:14s} n={len(g):5d}  rho(best_epoch, spearman) = "
          f"{g.best_epoch.corr(g.spearman, method='spearman'):+.3f}   "
          f"spearman mean={g.spearman.mean():.3f} sd={g.spearman.std():.3f}  "
          f"best_epoch median={g.best_epoch.median():.0f}")

print("\nRank quality by how long the winner trained (quartiles of best_epoch), per model:")
for m, g in d.groupby("model"):
    if len(g) < 40: continue
    q = pd.qcut(g.best_epoch, 4, duplicates="drop")
    print(f"  {m}:")
    print(g.groupby(q, observed=True).agg(n=("spearman","size"),
                                          spearman=("spearman","mean"),
                                          pred_sd=("pred_sd","mean")).round(3).to_string())
