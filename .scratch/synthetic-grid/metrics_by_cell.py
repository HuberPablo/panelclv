"""Per-study RMSE (customer totals) / bias / MAPE / Spearman / validation CE on the
1,000-customer seasonal grid, for every tree, joined to its (rate, churn) cell.

Writes results/per_study.csv (one row per tree x panel) and results/by_cell.csv
(mean and 95% t-interval over the 10 replicate panels of each rate x churn cell).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from panelclv.data_preparation.pareto_nbd_simulation import list_pnbd_datasets, load_pnbd_dataset
from panelclv.models.monte_carlo_forecasting import compute_forecast_metrics
from panelclv.predictions import load_predictions_from_csv
from panelclv.studies.synthetic_grid import _holdout_weeks, _holdout_rows

GRID = Path("Datasets/Synthetic/seasonal_4x4x10")
OUT = Path(".scratch/synthetic-grid/results")
TREES = sorted(p for p in Path("Studies").glob("seasonal_4x4x10__*")
               if p.name.count("__") == 2 or p.name.endswith("ParetoNBD"))

rows = []
for ds in list_pnbd_datasets(GRID).itertuples(index=False):
    panel, _, cfg = load_pnbd_dataset(GRID, ds.combo, ds.dataset)
    sch = cfg["schema"]
    for tree in TREES:
        suite = tree / f"{ds.combo}__{ds.dataset}"
        if not suite.is_dir():
            continue
        model_dir = next(p for p in suite.iterdir() if p.is_dir())
        pred, cust = load_predictions_from_csv(model_dir / "Predictions" / "Prediction_1.csv")
        weeks = _holdout_weeks(cfg, pred.shape[1])
        clip = json.load(open(suite / "config.json"))["panel_config"]["clip_target_upper"]
        # Actual holdout counts, clipped as the models' targets are, in the forecast's customer order.
        hold = _holdout_rows(type("D", (), {"panel": panel, "time_cols": tuple(sch["time_cols"]),
                                            "config": cfg})(), weeks)
        act = (hold.pivot_table(index=sch["id_col"], columns="week_index",
                                values=sch["target_col"], aggfunc="sum")
                   .reindex(index=cust, columns=weeks).fillna(0).to_numpy())
        act = np.minimum(act, clip)
        m = compute_forecast_metrics(act, pred)
        stored = pd.read_csv(suite / "results.csv").iloc[0]
        arm = tree.name.split("__")[2] if tree.name.count("__") == 2 else "-"
        rows.append(dict(
            model=tree.name.split("__")[1], arm=arm.replace("-valendin", ""),
            rate=ds.mean_transaction_rate, churn=ds.churn_rate, dataset=ds.dataset,
            rmse=m["rmse_customer_total"], bias=m["bias_percent"], mape=m["mape_aggregate"],
            spearman=stats.spearmanr(pred.sum(1), act.sum(1)).statistic,
            ce=stored["objective"],
            stored_mape=stored["mape_aggregate"],
        ))

df = pd.DataFrame(rows)
df["mismatch"] = (df.mape - df.stored_mape).abs() > 1e-3
OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "per_study.csv", index=False)
print("studies:", len(df), "mismatched:", int(df.mismatch.sum()))
print(df[df.mismatch][["model", "arm", "rate", "churn", "dataset", "mape", "stored_mape"]])
