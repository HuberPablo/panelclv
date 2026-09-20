"""Aggregate best_epoch / early-stopping behaviour across the archived real-panel studies."""
import glob, os, re
import pandas as pd
from pathlib import Path

# The repo root, derived so these run unchanged on a rented box, and an
# output folder beside this script.
REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

ROOT = str(REPO / "Studies")
PANELS = ("cdnow", "electronics", "gift", "multichannel")

rows = []
for path in glob.glob(os.path.join(ROOT, "*", "*", "Optuna_Studies", "*", "*_trials.csv")):
    parts = path.split(os.sep)
    suite, model = parts[-5], parts[-4]
    panel = next((p for p in PANELS if p in suite), None)
    if panel is None:
        continue
    family = suite.split("__")[0]
    try:
        df = pd.read_csv(path)
    except Exception:
        continue
    if "user_attrs_best_epoch" not in df.columns:
        continue
    df["suite"], df["model"], df["panel"], df["family"] = suite, model, panel, family
    rows.append(df)

t = pd.concat(rows, ignore_index=True)
t.to_pickle(RESULTS / "trials.pkl")
print("trials:", len(t), " suites:", t.suite.nunique())
print("\nstates:\n", t.state.value_counts())
c = t[t.state == "COMPLETE"].copy()
c["best_epoch"] = c.user_attrs_best_epoch.astype(float)
c["patience"] = c.params_patience.astype(float)
c["n_epochs"] = c.params_n_epochs.astype(float)
# epochs actually run: early stop fires patience epochs after the best one, capped at n_epochs
c["epochs_run"] = (c.best_epoch + c.patience + 1).clip(upper=c.n_epochs)
c["hit_cap"] = c.epochs_run >= c.n_epochs

print("\npatience / n_epochs settings used:\n",
      c.groupby(["patience", "n_epochs"]).size())

print("\nBest epoch (0-indexed) of every completed trial, by panel and model family:")
g = c.groupby(["panel", "model"]).best_epoch.describe(percentiles=[.5, .9])
print(g.round(2).to_string())

print("\nShare of completed trials whose best epoch was 0 (nothing learned after epoch 1),")
print("and share that ran out the 100-epoch budget instead of early-stopping:")
print(c.groupby(["panel", "model"]).agg(
    n=("best_epoch", "size"),
    frac_best0=("best_epoch", lambda s: (s == 0).mean()),
    frac_best_le2=("best_epoch", lambda s: (s <= 2).mean()),
    frac_hit_cap=("hit_cap", "mean"),
    mean_epochs_run=("epochs_run", "mean"),
).round(3).to_string())
