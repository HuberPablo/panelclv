"""Pareto/NBD baseline (Schmittlein, Morrison & Colombo 1987) for the
customer transaction-count forecasting setup used in this package.

The model is the classic count-CLV benchmark (also the one used in
Valendin et al.). For each customer it predicts the expected number of
future purchases per period over the holdout window.


Pipeline
--------
    train_panel ──► (frequency, recency, T) per customer
                ──► ParetoNBDFitter.fit
                ──► cumulative E[X(t)] for t = 1..T_HOLD
                ──► differentiate to per-period expectations


Inputs
------
train_panel  : DataFrame
    The training-window slice produced by `prepare_dataset` (it must contain
    `id_col`, `target_col`, and a `period_start` Timestamp column).
holdout_length : int
    Number of forecast periods.
id_col, target_col, time_col
    Names of the relevant columns. Defaults match `dynamic_panel_dataset`.
period_in_days : float = 7.0
    How many days one period covers (weekly = 7, daily = 1, monthly ≈ 30).
penalizer_coef : float
    L2 penalty on the four Pareto/NBD parameters. The default 0.01 matches
    `lifetimes`'s recommendation for sparse data.
customer_ids : list | None
    If given, the returned predictions are reordered to match this list.


Output
------
predictions : ndarray (N, holdout_length)
    Expected number of purchases per customer per period.
ids         : list
    Customer ids in the order of `predictions`.


Notes
-----
- Cohort customers all have ≥1 purchase by `training_end`, so `frequency`,
  `recency`, `T` are well-defined for everyone.
- `frequency = total purchases in calibration − 1` (Pareto/NBD treats the
  first purchase as the cohort-entry event, not a repeat).
- `T` is offset by `+1 period` so customers whose first purchase falls on
  the last calibration period still have `T > 0` (numerical stability).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def compute_pareto_predictions(
    train_panel: pd.DataFrame,
    holdout_length: int,
    *,
    id_col: str = "Id",
    target_col: str = "Transactions",
    time_col: str = "period_start",
    period_in_days: float = 7.0,
    penalizer_coef: float = 0.01,
    customer_ids: Sequence | None = None,
) -> tuple[np.ndarray, list]:
    """Fit Pareto/NBD on `train_panel` and forecast `holdout_length` periods."""
    # Lazy import so the module is at least importable without lifetimes.
    from lifetimes import ParetoNBDFitter

    panel = train_panel.copy()
    panel[time_col] = pd.to_datetime(panel[time_col])

    # ---- 1. RFM summary per customer -------------------------------------
    panel_start = panel[time_col].min()
    cal_end     = panel[time_col].max()

    purchases = panel[panel[target_col] > 0]
    first_t = purchases.groupby(id_col)[time_col].min().rename("first_t")
    last_t  = purchases.groupby(id_col)[time_col].max().rename("last_t")
    total   = panel.groupby(id_col)[target_col].sum().rename("total")

    rfm = pd.DataFrame(index=total.index)
    rfm["total"]   = total
    rfm["first_t"] = first_t.reindex(rfm.index).fillna(panel_start)
    rfm["last_t"]  = last_t.reindex(rfm.index).fillna(panel_start)

    rfm["frequency"] = (rfm["total"] - 1).clip(lower=0)
    rfm["recency"]   = (rfm["last_t"] - rfm["first_t"]).dt.days / period_in_days
    rfm["T"]         = (cal_end - rfm["first_t"]).dt.days / period_in_days + 1.0

    # ---- 2. Fit Pareto/NBD -----------------------------------------------
    fitter = ParetoNBDFitter(penalizer_coef=penalizer_coef)
    fitter.fit(rfm["frequency"].values, rfm["recency"].values, rfm["T"].values)

    # ---- 3. Cumulative E[X(t)] -> per-period differences ----------------
    cum = np.zeros((len(rfm), holdout_length), dtype=np.float64)
    for t in range(1, holdout_length + 1):
        cum[:, t - 1] = fitter.conditional_expected_number_of_purchases_up_to_time(
            t, rfm["frequency"].values, rfm["recency"].values, rfm["T"].values,
        )
    predictions = np.diff(cum, axis=1, prepend=0.0)

    ids = rfm.index.tolist()

    # ---- 4. Optional reorder ---------------------------------------------
    if customer_ids is not None:
        order_map = {cid: i for i, cid in enumerate(ids)}
        missing = [cid for cid in customer_ids if cid not in order_map]
        if missing:
            raise ValueError(
                f"customer_ids contains {len(missing)} ids absent from train_panel "
                f"(first few: {missing[:5]})"
            )
        order = [order_map[cid] for cid in customer_ids]
        predictions = predictions[order]
        ids = list(customer_ids)

    return predictions, ids
