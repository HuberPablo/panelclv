"""Score every family T / T' / U study and join it with its winning trial.

One row per study (= one Optuna search, its refit and its 200-path forecast):
the four holdout metrics, recomputed from the stored forecast through the single
scoring authority, plus the winner's hyperparameters, best epoch, validation CE
and the gradient updates it received. Feeds the worst/best tables in
docs/insight-training-efficiency.md §9.
"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_training_budget as tb
import run_factorial as fa
from panelclv.models.monte_carlo_forecasting import compute_forecast_metrics
from panelclv.data_preparation.target_channel import holdout_actuals
from panelclv.studies import load_model_predictions

def parse(name):
    parts = name.split("__")
    fam, model, panel, arm, rep = parts
    if fam == "training_budget":
        return dict(family="T" if panel == "electronics" else "T'", model=model,
                    panel=panel, arm=arm, cluster="no_cluster", rep=int(rep[1:]))
    training, cluster = arm.split("-", 1)
    return dict(family="U", model=model, panel=panel, arm=training, cluster=cluster,
                rep=int(rep[1:]))

cache = {}
def data_for(row):
    key = (row["family"] in ("T", "T'"), row["panel"], row["model"], row["cluster"])
    if key not in cache:
        if key[0]:
            tb.PANEL = row["panel"]
            d = tb.build_data(row["model"])
        else:
            d = fa.build_data(row["panel"], row["model"], row["cluster"])
        cache[key] = (holdout_actuals(d), np.asarray(d["ids"]))
    return cache[key]

rows = []
for suite in sorted(p for p in (ROOT / "Studies").iterdir()
                    if p.name.startswith(("training_budget__", "factorial__"))):
    row = parse(suite.name)
    mdir = suite / row["model"]
    actual, ids_ref = data_for(row)
    values, ids = load_model_predictions(mdir, study=1)
    assert ids is None or np.array_equal(np.asarray(ids), ids_ref), suite
    pred_tot, act_tot = values.sum(axis=1), actual.sum(axis=1)
    best = json.loads((mdir / "Optuna_Studies/study_01/study_01_best.json").read_text())
    p, ua = best["best_params"], best["best_user_attrs"]
    trials = pd.read_csv(mdir / "Optuna_Studies/study_01/study_01_trials.csv")
    n_cust = len(ids_ref)
    row.update(compute_forecast_metrics(actual, values))
    row.update(
        spearman=tb.spearman(pred_tot, act_tot),
        forecast_cv=float(pred_tot.std() / pred_tot.mean()),
        val_ce=best["best_objective_value"], best_epoch=int(ua["best_epoch"]),
        batch_size=p.get("batch_size"), learning_rate=p.get("learning_rate"),
        weight_decay=p.get("weight_decay"), dropout=p.get("dropout"),
        hidden=p.get("lstm_hidden_size"), dense=p.get("dense_units"),
        min_epochs=p.get("min_epochs", 0), n_customers=n_cust,
        n_complete=int((trials.state == "COMPLETE").sum()),
        suite=suite.name)
    row["updates"] = math.ceil(n_cust / row["batch_size"]) * (row["best_epoch"] + 1)
    rows.append(row)

out = ROOT / ".scratch/training-budget/results/worst_best_runs.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(len(rows), "studies ->", out)
