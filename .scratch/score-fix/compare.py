"""Archived vs per-cell-score rerun of electronics · archive · no cluster label.

Scores both sets of forecasts through the single scoring authority, joins each with its
winner (batch, best epoch, validation loss) and reports, per model:
- which batch the search picked and how long the winners trained;
- the four holdout metrics plus forecast CV, as mean ± sd;
- Δ (rerun − archive) with its 95% percentile-bootstrap interval, resampled
  independently: training is unseeded and the two searches diverge after their first
  trial, so a shared seed does not make the replications pairs.

    python .scratch/score-fix/compare.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / ".scratch" / "training-budget"))
import run_factorial as fa                                            # noqa: E402
from panelclv.data_preparation.target_channel import holdout_actuals   # noqa: E402
from panelclv.evaluation.effects import effect                         # noqa: E402
from panelclv.models.monte_carlo_forecasting import compute_forecast_metrics  # noqa: E402
from panelclv.studies import load_model_predictions                    # noqa: E402

PANEL, CLUSTER = "electronics", "no_cluster"
rows = []
for model in fa.MODELS:
    data = fa.build_data(PANEL, model, CLUSTER)
    actual, ids_ref = holdout_actuals(data), np.asarray(data["ids"])
    for exp in ("factorial", "scorefix"):
        for rep in range(fa.N_REPLICATIONS):
            mdir = fa.STUDIES_BASE / f"{exp}__{model}__{PANEL}__archive-{CLUSTER}__r{rep:02d}" / model
            if not (mdir / "Predictions" / "Prediction_1.csv").exists():
                continue
            values, ids = load_model_predictions(mdir, study=1)
            assert ids is None or np.array_equal(np.asarray(ids), ids_ref)
            best = json.loads((mdir / "Optuna_Studies/study_01/study_01_best.json").read_text())
            pred, act = values.sum(axis=1), actual.sum(axis=1)
            rows.append(dict(
                model=model, exp=exp, rep=rep, **compute_forecast_metrics(actual, values),
                spearman=pd.Series(pred).corr(pd.Series(act), method="spearman"),
                cv=float(pred.std() / pred.mean()),
                batch=best["best_params"]["batch_size"],
                best_epoch=best["best_user_attrs"]["best_epoch"],
                val_ce=best["best_objective_value"]))
d = pd.DataFrame(rows)
d.to_csv(ROOT / ".scratch/score-fix/compare.csv", index=False)

METRICS = {"bias_percent": "bias %", "mape_aggregate": "MAPE", "spearman": "Spearman",
           "rmse": "RMSE", "cv": "forecast CV", "best_epoch": "best epoch"}
for model, g in d.groupby("model"):
    print(f"\n=== {model}: archive n={sum(g.exp == 'factorial')}, rerun n={sum(g.exp == 'scorefix')}")
    for exp, e in g.groupby("exp"):
        print(f"  {exp:9s} batch picked: {e.batch.value_counts().sort_index().to_dict()}")
    for m, name in METRICS.items():
        a, b = g[g.exp == "factorial"][m], g[g.exp == "scorefix"][m]
        if len(b) < 2:
            continue
        ef = effect(b, a, paired=False, metric=m, panel=PANEL)
        print(f"  {name:12s} archive {a.mean():8.4f} ± {a.std():.4f}   rerun {b.mean():8.4f} ± "
              f"{b.std():.4f}   Δ {ef.delta:+.4f} [{ef.lo:+.4f}, {ef.hi:+.4f}]")
