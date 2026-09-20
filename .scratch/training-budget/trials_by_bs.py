"""Why did the search keep picking the batch size that collapses?

Every COMPLETED trial (not just the winner) of the electronics ValendinLSTM benchmark,
grouped by the batch size it sampled.
"""
import glob, re
import numpy as np, pandas as pd
from pathlib import Path

# The repo root, derived so these run unchanged on a rented box, and an
# output folder beside this script.
REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

ROOT = REPO / "Studies"

frames = []
for tf in ROOT.glob("real_panel_benchmarks__ValendinLSTM__electronics__r*/ValendinLSTM/Optuna_Studies/*/*_trials.csv"):
    df = pd.read_csv(tf)
    df["suite"] = tf.parents[3].name
    frames.append(df)
t = pd.concat(frames, ignore_index=True)
c = t[t.state == "COMPLETE"].copy()
c["best_epoch"] = c.user_attrs_best_epoch.astype(float)
c["updates"] = np.ceil(829 / c.params_batch_size) * (c.best_epoch + 1)
print(f"{len(t)} trials, {len(c)} completed, {len(t)-len(c)} pruned\n")

print("Completed trials by sampled batch size:")
print(c.groupby("params_batch_size").agg(
    n=("value", "size"), val_loss_mean=("value", "mean"), val_loss_min=("value", "min"),
    val_loss_p10=("value", lambda s: s.quantile(.10)),
    best_epoch=("best_epoch", "median"), updates=("updates", "median"),
    frac_under_0085=("value", lambda s: (s < 0.085).mean())).round(4).to_string())

print("\nHow often does each batch size WIN its study (min val loss over the 100 trials)?")
win = c.loc[c.groupby("suite").value.idxmin()]
print(win.params_batch_size.value_counts().sort_index().to_string())

print("\nAmong trials that sampled batch 64, the spread of validation loss:")
b64 = c[c.params_batch_size == 64].value
print("  n =", len(b64), " min", round(b64.min(), 4), " p25", round(b64.quantile(.25), 4),
      " median", round(b64.median(), 4), " p75", round(b64.quantile(.75), 4))
b256 = c[c.params_batch_size == 256].value
print("Among trials that sampled batch 256:")
print("  n =", len(b256), " min", round(b256.min(), 4), " p25", round(b256.quantile(.25), 4),
      " median", round(b256.median(), 4), " p75", round(b256.quantile(.75), 4))
