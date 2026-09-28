"""Per-study metrics on the seasonal Pareto/NBD grids, one row per tree x panel.

Covers the 1,000-customer grid (13 trees, plus the archived 10/20-trial `no_ar` runs),
the 3,000-customer grid (LSTM `no_ar`, LSTM `ar_bounded`, Pareto/NBD) and a
counterfactual Pareto/NBD rescored with the true seasonal multiplier. Metrics: RMSE on
customer totals and per customer-week, bias, MAPE, per-customer Spearman, validation CE,
alive-volume ratio R_A, dead-volume leakage L_D and shape correlation.

Writes results/per_study.csv.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from panelclv.data_preparation.pareto_nbd_simulation import list_pnbd_datasets, load_pnbd_dataset
from panelclv.models.monte_carlo_forecasting import compute_forecast_metrics
from panelclv.predictions import load_predictions_from_csv
from panelclv.studies.synthetic_grid import (
    _Dataset, _Forecast, _alive_volume_ratio, _dead_volume_leakage, _holdout_rows,
    _holdout_season, _holdout_weeks, _shape_correlation,
)

OUT = Path(".scratch/synthetic-grid/results")
STUDIES = Path("Studies")

# (cohort label, generation dir, {tree dir: (model, arm)}).
GRIDS = [
    ("n1000", Path("Datasets/Synthetic/seasonal_4x4x10"), {
        **{p.name: (p.name.split("__")[1], p.name.split("__")[2].replace("-valendin", ""))
           for p in STUDIES.glob("seasonal_4x4x10__*__*")},
        "seasonal_4x4x10__ParetoNBD": ("ParetoNBD", "-"),
        # The archived small-search runs: `no_ar`, 10 (LSTM) / 20 (Transformer) trials.
        "seasonal_4x4x10__LSTM": ("LSTM", "no_ar-no_cluster-small_search"),
        "seasonal_4x4x10__Transformer": ("Transformer", "no_ar-no_cluster-small_search"),
    }),
    ("n3000", Path("Datasets/Synthetic/seasonal_4x4x10_n3000"), {
        "seasonal_4x4x10_n3000__LSTM__no_ar-no_cluster-valendin": ("LSTM", "no_ar-no_cluster"),
        "seasonal_4x4x10_n3000__LSTM__ar_bounded-no_cluster-valendin": ("LSTM", "ar_bounded-no_cluster"),
        # Pareto/NBD reads no AR columns; the two n3000 Pareto/NBD trees are the same fit.
        "seasonal_4x4x10_n3000__ParetoNBD__no_ar-no_cluster-valendin": ("ParetoNBD", "-"),
    }),
]


def score(ds, forecast, act, suite, model, arm, cohort):
    m = compute_forecast_metrics(act, forecast.predicted)
    stored = pd.read_csv(suite / "results.csv").iloc[0]
    params = {k: v for k, v in stored.items() if k.startswith("param_")}
    return dict(
        cohort=cohort, model=model, arm=arm,
        rate=ds.mean_transaction_rate, churn=ds.churn_rate, dataset=ds.dataset,
        rmse=m["rmse_customer_total"], rmse_week=m["rmse"], bias=m["bias_percent"],
        mape=m["mape_aggregate"],
        spearman=stats.spearmanr(forecast.predicted.sum(1), act.sum(1)).statistic,
        ce=stored["objective"], stored_mape=stored["mape_aggregate"],
        r_a=_alive_volume_ratio(ds, forecast), l_d=_dead_volume_leakage(ds, forecast),
        shape=_shape_correlation(ds, forecast), **params,
    )


rows = []
for cohort, grid, trees in GRIDS:
    for row in list_pnbd_datasets(grid).itertuples(index=False):
        panel, truth, cfg = load_pnbd_dataset(grid, row.combo, row.dataset)
        base = _Dataset(row.mean_transaction_rate, row.churn_rate, row.combo, row.dataset,
                        Path(), panel, truth, cfg)
        for tree, (model, arm) in trees.items():
            suite = STUDIES / tree / f"{row.combo}__{row.dataset}"
            if not suite.is_dir():
                continue
            model_dir = next(p for p in suite.iterdir() if p.is_dir())
            pred, cust = load_predictions_from_csv(model_dir / "Predictions" / "Prediction_1.csv")
            weeks = _holdout_weeks(cfg, pred.shape[1])
            season = _holdout_season(base, weeks)
            forecast = _Forecast(model, pred, cust, weeks, season)
            # Actual holdout counts, clipped as the models' targets are, in forecast order.
            clip = json.load(open(suite / "config.json"))["panel_config"]["clip_target_upper"]
            hold = _holdout_rows(base, weeks)
            act = (hold.pivot_table(index=base.id_col, columns="week_index",
                                    values=base.target_col, aggfunc="sum")
                       .reindex(index=cust, columns=weeks).fillna(0).to_numpy())
            act = np.minimum(act, clip)
            rows.append(score(base, forecast, act, suite, model, arm, cohort))
            if model == "ParetoNBD":
                # Counterfactual: the same fit times the true season, rescaled to mean 1.
                seasonal = _Forecast(model, pred * season / season.mean(), cust, weeks, season)
                r = score(base, seasonal, act, suite, model, "true_season", cohort)
                rows.append(r | {"stored_mape": r["mape"]})

df = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT / "per_study.csv", index=False)
bad = (df.mape - df.stored_mape).abs() / df.stored_mape >= 0.05
print(df.groupby(["cohort", "model", "arm"]).size())
print("forecasts not matching their stored results:\n", df[bad][["cohort", "model", "arm", "rate", "churn", "dataset"]])
