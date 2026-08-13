"""The wide per-customer prediction CSV: one row per customer, one column per period.

Every model in the package dumps its holdout forecast in this one layout —

    id_col, week_0, week_1, ..., week_{T-1}

— and everything that scores or plots a stored forecast reads it back through
here. Written by the Monte Carlo simulator (``models``), the Pareto/NBD benchmark
(``benchmarks``) and the study runner (``studies``); read by the per-group tables
(``evaluation``) and the suite analysis (``studies``).

Why this is its own subpackage: the writers sit in the model layer and the readers
sit above it, so wherever the format lived among them, somebody would have had to
import upward. It lived in ``evaluation/plot_utils.py`` and the model layer reached
back for it through a deferred import that hid the resulting cycle rather than
removing it (ADR-0002). Here it is a leaf — it imports nothing from ``panelclv``,
so every arrow into it points down.

The columns say ``week_`` whatever the panel's frequency. That name is on disk in
every archived study, so it is a floor, not a description.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# The customer-key column name, written once. A `prepare_dataset` dict names its own
# id column and every writer passes that through; this is what they fall back to when
# there is none — a hand-built dict, or a suite tree with no config beside it. It used
# to be two competing spellings ("customer_id" and "Id") at nine sites, with the study
# runner reaching for both inside a single function, which is how one archived suite
# ended up holding `aggregated_*.csv` keyed on `customer_id` next to `Prediction_*.csv`
# keyed on `Id`.
DEFAULT_ID_COL = "customer_id"


def reduce_to_customer_period(predictions: np.ndarray) -> np.ndarray:
    """Collapse a forecast to a 2-D `(n_customers, T)` array of means.

    The three shapes a forecast arrives in, all reduced to the one the CSV holds
    and the metrics score:

        (S, N, T, 1)  Monte Carlo simulations -> mean over the S paths
        (N, T, 1)     a deterministic prediction with a trailing channel
        (N, T)        already a per-customer mean (e.g. Pareto/NBD)
    """
    arr = np.asarray(predictions, dtype=np.float64)
    if arr.ndim == 4:                # (S, N, T, 1)  -> mean over S, drop channel
        return arr.squeeze(-1).mean(axis=0)
    if arr.ndim == 3 and arr.shape[-1] == 1:   # (N, T, 1)
        return arr.squeeze(-1)
    if arr.ndim == 2:                # (N, T)
        return arr
    raise ValueError(
        f"Expected predictions of shape (S, N, T, 1), (N, T, 1), or (N, T); "
        f"got {predictions.shape}"
    )


def save_predictions_to_csv(
    predictions: np.ndarray,
    path: str | Path,
    customer_ids: Sequence | None = None,
    week_offset: int = 0,
    id_col: str = DEFAULT_ID_COL,
) -> Path:
    """Save predictions to a wide CSV: `id_col` + `week_0..week_{T-1}`.

    For Monte Carlo arrays of shape (S, N, T, 1), the saved values are the mean
    across simulations. Deterministic predictions (N, T) or (N, T, 1) are saved
    as-is. The parent folder is created if it doesn't exist.
    """
    arr = reduce_to_customer_period(predictions)
    n_customers, n_weeks = arr.shape

    if customer_ids is None:
        customer_ids = np.arange(n_customers)
    else:
        customer_ids = np.asarray(customer_ids)
        if customer_ids.shape[0] != n_customers:
            raise ValueError(
                f"customer_ids has {customer_ids.shape[0]} rows but predictions "
                f"have {n_customers} customers"
            )

    columns = [f"week_{i + week_offset}" for i in range(n_weeks)]
    df = pd.DataFrame(arr, columns=columns)
    df.insert(0, id_col, customer_ids)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_predictions_from_csv(
    path: str | Path,
    id_col_candidates: Sequence[str] = ("customer_id", "id", "Id", "ID"),
    holdout_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load wide-CSV predictions back as a (n_customers, T) array.

    Returns `(values, ids)`. `ids` is `None` when no id column is found.
    If `holdout_length` is given, trailing/extra week columns are truncated.
    """
    df = pd.read_csv(path)
    ids = None
    for col in id_col_candidates:
        if col in df.columns:
            ids = df[col].to_numpy()
            df = df.drop(columns=[col])
            break
    arr = df.to_numpy(dtype=np.float64)
    if holdout_length is not None:
        arr = arr[:, :holdout_length]
    return arr, ids
