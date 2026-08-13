"""Per-customer-group, per-model metric analysis.

Two steps, connected by customer ids:

    1. `assign_customer_groups(data, groups=...)`  -> {group_name: ids}
    2. `group_metrics_table(data, model_predictions, group_ids, ...)`
           -> RMSE (individual) / MAPE, bias, bias_percent (aggregate) per
              (group, model), scored on the saved predictions; optionally
              written to CSV.

Groups are derived from each customer's calibration vs holdout activity:

    "At Risk"     : inactive in holdout AND at least the cohort-average
                    calibration frequency (Valendin et al.'s churned customers).
    "Opportunity" : more transactions in holdout than in calibration.

`CUSTOMER_GROUPS` is those predicate names — the group set, defined once by what
defines the groups. `assign_customer_groups(..., with_other=True)` adds the derived
`"Other"` catch-all (matched by no predicate) so a table covers the whole cohort.

Inputs
------
- `data` : a `prepare_dataset` output (the `data_best` used for forecasting);
           per-customer actuals and the calibration/holdout counts come from it.
- `model_predictions` : {model_name: csv_path} — the saved per-customer
           prediction CSVs (from `save_predictions_to_csv` /
           `forecast_recurrent(save_predictions=True)` / `pareto_forecast(...)`). Rows
           are realigned to `data["ids"]` by customer id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from panelclv.models.monte_carlo_forecasting import compute_forecast_metrics
from panelclv.predictions import load_predictions_from_csv


def aggregate_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sum(y_pred) - sum(y_true), in raw counts.

    Positive = over-forecast, negative = under-forecast. This is the one number the
    group table wants that `compute_forecast_metrics` does not return: it reports
    bias as a percentage of actual, which is uninformative for a small group whose
    actual total is near zero. Kept here because this is its only caller.
    """
    return float(np.sum(y_pred) - np.sum(y_true))


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

# The customer-group set, written once: a group exists exactly when a predicate defines
# it, so these keys *are* the set and cannot drift from the definitions. Everything that
# needs "the groups" — defaults, plots, the suite metrics table — reads this rather than
# restating the names.
CUSTOMER_GROUPS = tuple(_GROUP_PREDICATES)


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
    groups: Sequence[str] = CUSTOMER_GROUPS,
    with_other: bool = False,
) -> dict[str, np.ndarray]:
    """Return {group_name: array of customer ids} for the requested groups.

    `groups` defaults to every defined group and can only ever narrow that set.
    `with_other=True` appends the derived `"Other"` catch-all — every customer matched
    by none of the requested predicates — so the mapping covers the whole cohort and
    the group counts sum to N whenever the groups are disjoint (At Risk / Opportunity
    are). This is the only place the catch-all is computed.
    """
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
    if with_other:
        # Compared as `str` (matching how the metric helpers key ids) and returned in
        # the cohort's own order.
        assigned = {str(cid) for group in out.values() for cid in group}
        out["Other"] = np.asarray([cid for cid in ids if str(cid) not in assigned])
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
    rmse, mape, bias, bias_percent; if `save_path` is given it is also written to CSV.

    Each metric is either **individual** (scored on the per-customer, per-week cells,
    so every customer contributes directly) or **aggregate** (customers are summed into
    one weekly curve first, then scored — the tracking-quality view):

    - `rmse`         : INDIVIDUAL. sqrt(mean over all (customer, week) cells of the
                       squared error). Sensitive to per-customer over/under-prediction.
    - `mape`         : AGGREGATE. MAPE of the summed weekly totals; NaN when the group's
                       total actual is 0 (e.g. At Risk), since there is no positive
                       denominator.
    - `bias`         : AGGREGATE. Raw sum(pred) - sum(actual) — the signed count of
                       over/under-forecast transactions. Well-defined for every group
                       (no division), so it is the usable signed metric for zero-actual
                       groups like At Risk, where it equals the total predicted "phantom"
                       activity for customers who were in fact inactive.
    - `bias_percent` : AGGREGATE. The same bias as a % of the total actual; NaN when that
                       total is 0.
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
                "rmse": m["rmse"],                              # individual (per cell)
                "mape": m["mape_aggregate"],                    # aggregate (weekly curve)
                "bias": aggregate_bias(a_g, arr[row_idx]),      # aggregate (raw count)
                "bias_percent": m["bias_percent"],              # aggregate (% of actual)
            })

    table = pd.DataFrame(rows).set_index(["group", "model"]).sort_index()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(save_path)
    return table
