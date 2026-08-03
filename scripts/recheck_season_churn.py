"""Two re-checks of the synthetic-grid findings, using the generator's ground truth.

A. SEASONALITY. The earlier check correlated raw weekly totals with the truth, which
   cannot separate *seasonal shape* from *trend*: a model that merely tracks the slow
   decline in volume as customers die scores well without representing seasonality at
   all. Here both series are detrended (13-week centred rolling mean removed) before
   correlating, and the residual is additionally correlated against the generator's
   KNOWN seasonal multiplier — a signal no amount of trend-fitting can fake.

B. CHURN DETECTION, BOTH DIRECTIONS. Ground truth gives each customer a death week
   `tau`, so "alive at the start of the holdout" is known exactly. Treating each
   model's predicted holdout volume as an alive-score:
     - ROC AUC          — does it rank alive customers above dead ones? (both errors)
     - dead-customer     — volume it assigns to customers already dead (false alarm)
     - alive-customer    — volume assigned to survivors, vs the oracle (miss)
   AUC is the headline because it penalises BOTH failing to spot churners AND wrongly
   writing off customers who are still active.

Run from the repo root:
    PYTHONPATH=src python scripts/recheck_season_churn.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from panelclv.data_preparation import pareto_simulation as ps

GEN = Path("Datasets/Synthetic/pnbd_study_4x4x10_20260716-154143")
TRAIN = Path("Studies/pnbd_study_4x4x10_20260716-154143")
OUT = Path("figures")
MODELS = ["LSTM", "Transformer", "ParetoNBD"]
REPLICATES = [f"Dataset_{i}" for i in range(1, 11)]
HOLDOUT_YEAR, T_CAL, DETREND_WIN = 2001, 104, 13


def seasonal_curve(cfg, weeks):
    """The generator's within-year seasonal multiplier, rebuilt from the dataset config.

    Gaussian bumps at `seasonal_peaks` (week-of-year), wrapped circularly so a peak
    near week 52 still lifts week 1. This is the signal the models are supposed to
    learn; correlating against it is a far stronger test than correlating against a
    realised series that also contains trend and Poisson noise.
    """
    peaks = cfg["seasonal_peaks"]
    amp, width = cfg["seasonal_amplitude"], cfg["seasonal_width"]
    woy = ((weeks - 1) % 52) + 1
    mult = np.ones_like(woy, dtype=float)
    for pk in peaks:
        d = np.abs(woy - pk)
        d = np.minimum(d, 52 - d)                      # circular distance
        mult += (amp - 1.0) * np.exp(-0.5 * (d / width) ** 2)
    return mult


def detrend(series, window=DETREND_WIN):
    """Remove the slow component so only within-year seasonal wiggle remains."""
    s = pd.Series(series, dtype=float)
    return (s - s.rolling(window, center=True, min_periods=1).mean()).values


def analyse():
    season_rows, churn_rows = [], []
    combos = sorted(p.name for p in GEN.iterdir() if p.name.startswith("Dataset_"))

    for combo in combos:
        rate, churn = (int(v) for v in combo.split("_")[1:3])
        for ds in REPLICATES:
            panel, gt, cfg = ps.load_pnbd_dataset(GEN, combo, ds)
            hold = panel[panel["year"] == HOLDOUT_YEAR]

            weeks = np.sort(hold["week"].unique())
            actual_weekly = hold.groupby("week")["Transactions"].sum().sort_index().values
            # Holdout weeks are the calibration length onward, in absolute week index.
            season = seasonal_curve(cfg, T_CAL + np.arange(1, len(weeks) + 1))
            season_resid = detrend(season)
            actual_resid = detrend(actual_weekly)

            actual_per_cust = hold.groupby("Id")["Transactions"].sum()
            # tau is the death week on the same absolute clock; alive into the holdout
            # means surviving past the calibration window.
            alive = gt.set_index("Id")["tau"].reindex(actual_per_cust.index) > T_CAL
            lam = gt.set_index("Id")["lambda"].reindex(actual_per_cust.index)
            # Oracle expected holdout volume: individual rate x weeks actually alive
            # inside the holdout x mean seasonal lift.
            weeks_alive = (gt.set_index("Id")["tau"].reindex(actual_per_cust.index)
                           .clip(T_CAL, T_CAL + len(weeks)) - T_CAL)
            oracle = lam * weeks_alive * season.mean()

            for m in MODELS:
                pred = pd.read_csv(TRAIN / f"{combo}__{ds}" / m / "Predictions" / "Prediction_1.csv")
                pred = pred.set_index("Id")
                weekly = pred.sum(axis=0).values[: len(actual_weekly)]
                per_cust = pred.sum(axis=1).reindex(actual_per_cust.index)

                pr = detrend(weekly)
                ok = pr.std() > 1e-12
                season_rows.append({
                    "rate": rate, "churn": churn, "model": m,
                    # raw = the earlier (trend-contaminated) metric, kept for comparison
                    "corr_raw": np.corrcoef(actual_weekly, weekly)[0, 1]
                                if weekly.std() > 0 else np.nan,
                    "corr_detrended": np.corrcoef(actual_resid, pr)[0, 1] if ok else np.nan,
                    "corr_vs_true_season": np.corrcoef(season_resid, pr)[0, 1] if ok else np.nan,
                })

                mask = per_cust.notna() & alive.notna()
                a, p = alive[mask].astype(bool), per_cust[mask]
                churn_rows.append({
                    "rate": rate, "churn": churn, "model": m,
                    "auc": roc_auc_score(a, p) if a.nunique() > 1 else np.nan,
                    # Direction 1 — volume wrongly given to customers already dead.
                    "share_to_dead": p[~a].sum() / p.sum() if p.sum() > 0 else np.nan,
                    # Direction 2 — volume given to survivors, relative to the oracle.
                    "alive_ratio": (p[a].sum() / oracle[mask][a].sum()
                                    if oracle[mask][a].sum() > 0 else np.nan),
                    "true_alive_share": a.mean(),
                })

    return pd.DataFrame(season_rows), pd.DataFrame(churn_rows)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    season, churn = analyse()
    season.to_csv(OUT / "seasonality_recheck.csv", index=False)
    churn.to_csv(OUT / "churn_detection.csv", index=False)

    pd.set_option("display.width", 200)
    print("=== A. SEASONALITY: raw vs detrended vs against the true seasonal curve ===")
    print(season.groupby("model")[["corr_raw", "corr_detrended", "corr_vs_true_season"]]
          .mean().round(3).to_string())
    print("\n-- detrended correlation, by churn --")
    print(season.pivot_table(index="churn", columns="model",
                             values="corr_detrended", aggfunc="mean").round(3).to_string())
    print("\n-- detrended correlation, by transaction rate --")
    print(season.pivot_table(index="rate", columns="model",
                             values="corr_detrended", aggfunc="mean").round(3).to_string())

    print("\n\n=== B. CHURN DETECTION (alive-vs-dead ranking by predicted volume) ===")
    print("\n-- ROC AUC by churn (0.5 = no discrimination) --")
    print(churn.pivot_table(index="churn", columns="model",
                            values="auc", aggfunc="mean").round(3).to_string())
    print("\n-- ROC AUC by transaction rate --")
    print(churn.pivot_table(index="rate", columns="model",
                            values="auc", aggfunc="mean").round(3).to_string())
    print("\n-- share of predicted volume given to ALREADY-DEAD customers --")
    print(churn.pivot_table(index="churn", columns="model",
                            values="share_to_dead", aggfunc="mean").round(3).to_string())
    print("\n-- predicted / oracle volume for customers who ARE still alive --")
    print(churn.pivot_table(index="churn", columns="model",
                            values="alive_ratio", aggfunc="mean").round(3).to_string())
    print("\n-- true alive share (context) --")
    print(churn.groupby("churn")["true_alive_share"].mean().round(3).to_string())
