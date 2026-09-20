"""What actually separates a study that ranks customers from one that does not?

Electronics (the panel whose ranking collapsed). For every archived study: the winning
trial's hyperparameters, how much training it actually got (epochs and gradient updates,
not patience), the arm it belongs to, and the per-customer Spearman of its forecast.
"""
import re
import numpy as np, pandas as pd
from pathlib import Path

# The repo root, derived so these run unchanged on a rented box, and an
# output folder beside this script.
REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

S = RESULTS
ROOT = REPO / "Studies"
N_CUST = 829                                   # electronics cohort = sequences per epoch

d = pd.read_csv(S / "spearman_electronics.csv")
recs = []
for (suite, model, study), g in d.groupby(["suite", "model", "study"]):
    tf = ROOT / suite / model / "Optuna_Studies" / f"study_{study:02d}" / f"study_{study:02d}_trials.csv"
    if not tf.exists():
        continue
    tr = pd.read_csv(tf)
    tr = tr[tr.state == "COMPLETE"]
    if tr.empty:
        continue
    w = tr.loc[tr.value.idxmin()]
    best_epoch = float(w.user_attrs_best_epoch)
    patience, n_epochs = float(w.params_patience), float(w.params_n_epochs)
    bs = float(w.params_batch_size)
    # The loop breaks `patience` non-improving epochs after the best one, so this is how
    # many epochs the winning trial actually ran (capped by the budget).
    epochs_run = min(best_epoch + 1 + patience, n_epochs)
    rec = dict(suite=suite, model=model, study=study,
               spearman=float(g.spearman.iloc[0]), pred_sd=float(g.pred_sd.iloc[0]),
               val_loss=float(w.value), best_epoch=best_epoch, epochs_run=epochs_run,
               lr=float(w.params_learning_rate), batch_size=bs,
               weight_decay=float(w.params_weight_decay),
               # gradient updates to the SELECTED weights: batches/epoch x epochs
               updates=np.ceil(N_CUST / bs) * (best_epoch + 1))
    for col, key in (("params_lstm_hidden_size", "hidden"), ("params_dense_units", "dense"),
                     ("params_dropout", "dropout"), ("params_embedder", "embedder"),
                     ("params_d_model", "d_model"), ("params_n_heads", "n_heads"),
                     ("params_num_layers", "layers")):
        if col in tr.columns:
            rec[key] = w[col]
    recs.append(rec)

p = pd.DataFrame(recs)
p["arm"] = p.suite.str.replace(r"^[a-z_0-9]+__(electronics__)?", "", regex=True).str.replace(r"__[a-z]$", "", regex=True)
p.to_csv(S / "hparams_electronics.csv", index=False)

NUM = ["best_epoch", "epochs_run", "updates", "lr", "batch_size", "weight_decay",
       "val_loss", "hidden", "dense", "dropout", "d_model", "layers"]

for m in ("ValendinLSTM", "LSTM", "Transformer"):
    g = p[p.model == m]
    if len(g) < 15:
        continue
    print(f"\n=== {m}  (n={len(g)} studies, mean Spearman {g.spearman.mean():.3f}) ===")
    print("  Spearman rank-correlation with the forecast's per-customer Spearman:")
    for c in NUM:
        if c in g and g[c].notna().sum() > 10 and pd.api.types.is_numeric_dtype(g[c]) and g[c].nunique() > 2:
            print(f"    {c:14s} {g[c].corr(g.spearman, method='spearman'):+.3f}")
    if "embedder" in g and g.embedder.notna().any():
        print("  by embedder:")
        print(g.groupby("embedder").agg(n=("spearman","size"), spearman=("spearman","mean"),
                                        best_epoch=("best_epoch","mean")).round(3).to_string())

print("\n=== LSTM studies grouped by ARM (feature set), sorted by ranking quality ===")
g = p[p.model == "LSTM"]
tab = g.groupby("arm").agg(n=("spearman", "size"), spearman=("spearman", "mean"),
                           sp_sd=("spearman", "std"), best_epoch=("best_epoch", "median"),
                           updates=("updates", "median"), pred_sd=("pred_sd", "mean"))
print(tab.sort_values("spearman", ascending=False).round(3).to_string())

print("\n=== Training volume WITHIN one arm: does it still matter? ===")
for arm, gg in g.groupby("arm"):
    if len(gg) < 80:
        continue
    q = pd.qcut(gg.updates, 4, duplicates="drop")
    t = gg.groupby(q, observed=True).agg(n=("spearman","size"), spearman=("spearman","mean")).round(3)
    print(f"\n  {arm}  (rho(updates, spearman) = {gg.updates.corr(gg.spearman, method='spearman'):+.3f})")
    print(t.to_string())
