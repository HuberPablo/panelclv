"""Per-customer-group, per-model metric analysis.

Two steps, connected by customer ids:

    1. `assign_customer_groups(data, groups=...)`  -> {group_name: ids}
    2. `group_metrics_table(data, model_predictions, group_ids, ...)`
           -> RMSE / MAPE / bias per (group, model), scored on the saved
              predictions; optionally written to CSV.

Groups are derived from each customer's calibration vs holdout activity:

    "At Risk"     : inactive in holdout AND at least the cohort-average
                    calibration frequency (Valendin et al.'s churned customers).
    "Opportunity" : more transactions in holdout than in calibration.

Inputs
------
- `data` : a `prepare_dataset` output (the `data_best` used for forecasting);
           per-customer actuals and the calibration/holdout counts come from it.
- `model_predictions` : {model_name: csv_path} — the saved per-customer
           prediction CSVs (from `save_predictions_to_csv` /
           `mc_forecast(save_predictions=True)` / `pareto_forecast(...)`). Rows
           are realigned to `data["ids"]` by customer id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from panelclv.models.monte_carlo_forecasting import compute_forecast_metrics
from .plot_utils import load_predictions_from_csv


# ---------------------------------------------------------------------------
# Group predicates (calib count `c`, holdout count `h`, cohort context `ctx`)
# ---------------------------------------------------------------------------

def _at_risk(c: np.ndarray, h: np.ndarray, ctx: dict[str, float]) -> np.ndarray:
    # Inactive in holdout AND at least the cohort-average calibration frequency.
    return (h == 0) & (c >= ctx["calib_mean"])


def _opportunity(c: np.ndarray, h: np.ndarray, ctx: dict[str, float]) -> np.ndarray:
    return h > c


_GROUP_PREDICATES = {
    "At Risk": _at_risk,
    "Opportunity": _opportunity,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _counts(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-customer calibration count, holdout count, holdout actuals (N order)."""
    target_idx = list(data["seq_cols"]).index(data["target_col"])
    calib = np.asarray(data["calibration"])[:, :, target_idx].sum(axis=1)   # (N,)
    actual = np.asarray(data["holdout"])[:, :, target_idx]                   # (N, T_HOLD)
    hold = actual.sum(axis=1)                                                # (N,)
    return calib.astype(float), hold.astype(float), actual.astype(float)


def _resolve_rows(data: dict[str, Any], ids: Sequence) -> np.ndarray:
    """Map a list of customer ids to row indices into the (N, ...) arrays."""
    id_to_row = {str(cid): i for i, cid in enumerate(data["ids"])}
    rows, missing = [], []
    for cid in ids:
        r = id_to_row.get(str(cid))
        (rows if r is not None else missing).append(r if r is not None else cid)
    if missing:
        raise ValueError(
            f"{len(missing)} ids not found in data['ids'] (first few: {missing[:5]})"
        )
    return np.asarray(rows, dtype=int)


def _load_aligned(path: str | Path, data: dict[str, Any]) -> np.ndarray:
    """Load a saved prediction CSV, reordered to match `data['ids']`."""
    values, ids = load_predictions_from_csv(path)
    values = np.asarray(values, dtype=float)
    if ids is None:
        return values  # no id column -> assume already in data order

    ref = [str(cid) for cid in data["ids"]]
    pos = {str(cid): i for i, cid in enumerate(ids)}
    missing = [cid for cid in ref if cid not in pos]
    if missing:
        raise ValueError(
            f"{path}: prediction file is missing {len(missing)} customer ids "
            f"present in data (first few: {missing[:5]})"
        )
    order = [pos[cid] for cid in ref]
    return values[order]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assign_customer_groups(
    data: dict[str, Any],
    groups: Sequence[str] = ("At Risk", "Opportunity"),
) -> dict[str, np.ndarray]:
    """Return {group_name: array of customer ids} for the requested groups."""
    calib, hold, _ = _counts(data)
    ctx = {"calib_mean": float(calib.mean())}
    ids = np.asarray(data["ids"])
    out: dict[str, np.ndarray] = {}
    for name in groups:
        if name not in _GROUP_PREDICATES:
            raise ValueError(
                f"unknown group {name!r}; available: {sorted(_GROUP_PREDICATES)}"
            )
        mask = np.asarray(_GROUP_PREDICATES[name](calib, hold, ctx), dtype=bool)
        out[name] = ids[mask]
    return out


def group_metrics_table(
    data: dict[str, Any],
    model_predictions: Mapping[str, str | Path],
    group_ids: Mapping[str, Sequence],
    *,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """RMSE / MAPE / bias per (group, model) on the saved predictions.

    `group_ids` is the {group_name: ids} mapping from `assign_customer_groups`.
    Returns a MultiIndex (group, model) DataFrame with columns n_customers,
    rmse, mape, bias_percent; if `save_path` is given it is also written to CSV.
    """
    _, _, actual = _counts(data)

    # Load + align every model once, validating shape against the actuals.
    preds: dict[str, np.ndarray] = {}
    for name, path in model_predictions.items():
        arr = _load_aligned(path, data)
        if arr.shape != actual.shape:
            raise ValueError(
                f"model {name!r}: predictions shape {arr.shape} != actual shape "
                f"{actual.shape} (same cohort/holdout length required)"
            )
        preds[name] = arr

    rows: list[dict[str, Any]] = []
    for gname, ids in group_ids.items():
        row_idx = _resolve_rows(data, ids)
        a_g = actual[row_idx]
        for mname, arr in preds.items():
            m = compute_forecast_metrics(a_g, arr[row_idx])
            rows.append({
                "group": gname, "model": mname, "n_customers": len(row_idx),
                "rmse": m["rmse"], "mape": m["mape_aggregate_style"],
                "bias_percent": m["bias_percent"],
            })

    table = pd.DataFrame(rows).set_index(["group", "model"]).sort_index()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(save_path)
    return table
