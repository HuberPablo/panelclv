"""Behavioural clusters: one categorical label per customer, from calibration alone.

`PanelConfig.cluster_features` declares them; this module computes them.
`prepare_dataset` writes the result into a panel column that both window slices carry,
and the label is embedded like any other categorical covariate — so nothing in
`models/` knows a cluster exists.

What a customer is clustered on
-------------------------------
The **Pareto/NBD sufficient statistics**, read at the last calibration period:

    period_since_last_transaction   recency, t_x
    cumulative_transactions         frequency, x — the count of ACTIVE PERIODS
    period_since_first_transaction  observation age, T

That triple is not an arbitrary choice of summary. It is exactly what the Pareto/NBD
likelihood conditions on (Schmittlein et al.; Fader & Hardie), and it is exactly what
`grids/seasonal_4x4x10_ar.py` hands the neural models as three continuous AR channels.
Clustering on the same triple makes the two comparable: the *same information* reaches
the model twice, once as three real-valued channels and once compressed into a single
categorical label. A difference between them is a fact about how the model reads its
inputs, not about which inputs it was given.

The three come from `compute_ar_feature_columns`, the same primitive that builds the AR
channels, so the two can never drift apart on what "recency" means.

Why it is leak-free
-------------------
The features are causal functions of the calibration target only, and the label is
frozen: a customer keeps it for every holdout period. `simulate_recurrent_path`
overwrites only the target channel and the AR channels, so a static column rides
through the rollout untouched, with no per-step recomputation and nothing to get wrong
(`docs/feature_engineering.md`).

One deviation is worth stating plainly. The label is fitted on the **full calibration
window**, which includes the temporal validation window (ADR-0001) that early stopping
later scores on. So the label has "seen" the validation periods. This is deliberate:
every other calibration-derived quantity in the package uses the same window —
`resolve_embedded_cols` sizes static cardinalities off it, Pareto/NBD is fitted on it,
and the ADR-0008 refit trains on it — and a label computed on a different window than
all of them would be a subtler inconsistency than the bias it avoids. The bias is also
small and bounded: an unsupervised 3-feature partition reveals nothing about the
target beyond what the model already reads in those same periods.

Determinism
-----------
`KMeans` is run with a fixed `random_state` and `n_init=10`. Cluster labels are
therefore a deterministic property of the panel, like the AR features — NOT a
replication source. This is what keeps `base_seed + i` meaning exactly what
`studies.config` says it means: the Optuna sampler and the Monte Carlo forecast, and
nothing else. Were the clustering seeded per study, it would inject a hidden extra
variance component into a suite that reports across-study SD, and no reader could
separate it from training noise.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from panelclv.configs.ar_feature_names import CUM_TXN, RECENCY, TENURE
from panelclv.configs.cluster_feature_names import parse_cluster_feature
from panelclv.data_preparation.ar_features import compute_ar_feature_columns


# The behaviour vector, in order. Named from `ar_feature_names` rather than spelled as
# strings so a rename upstream reaches here.
CLUSTER_BASIS: tuple[str, ...] = (RECENCY, CUM_TXN, TENURE)

# Fixed so labels are reproducible; see "Determinism" above.
_RANDOM_STATE = 0
_N_INIT = 10


def calibration_behaviour(target_2d: np.ndarray) -> np.ndarray:
    """The (t_x, x, T) triple per customer at the end of calibration.

    Parameters
    ----------
    target_2d : (N, T_CAL) array
        The CLIPPED calibration target — the same array `ARFeatureState` re-seeds
        from at forecast time, so the statistics here and the ones the rollout
        carries come from identical inputs.

    Returns
    -------
    (N, 3) float64 array, columns ordered as `CLUSTER_BASIS`.
    """
    target_2d = np.asarray(target_2d)
    if target_2d.ndim != 2:
        raise ValueError(
            f"target_2d must be (N, T_CAL), got shape {target_2d.shape}"
        )
    cols = compute_ar_feature_columns(target_2d, CLUSTER_BASIS)
    # Each column is (N, T_CAL) over the whole window; the last period is the state a
    # customer arrives at the forecast origin with, which is what defines them.
    return np.stack([cols[name][:, -1] for name in CLUSTER_BASIS], axis=1).astype(
        np.float64
    )


def _standardize(x: np.ndarray) -> np.ndarray:
    """Zero mean / unit variance per column, safe on a constant column.

    k-means minimises Euclidean distance, so an unscaled column with the widest range
    would dominate the partition for no reason other than its units. A column with no
    spread (e.g. every customer has the same tenure on a fixed-length panel) divides by
    1 instead of 0 and contributes nothing, which is the right answer rather than NaN.
    """
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return (x - mean) / std


def compute_cluster_labels(target_2d: np.ndarray, name: str) -> np.ndarray:
    """Cluster customers on calibration behaviour → one integer label each.

    Parameters
    ----------
    target_2d : (N, T_CAL) array
        The clipped calibration target, customers in the row order the caller will
        write the labels back in.
    name : str
        A `PanelConfig.cluster_features` name, e.g. ``"kmeans_8"``.

    Returns
    -------
    (N,) int64 array of labels in ``[0, K)``.
    """
    kind, k = parse_cluster_feature(name)

    features = calibration_behaviour(target_2d)
    n_customers = features.shape[0]
    if n_customers < k:
        raise ValueError(
            f"cluster feature {name!r} asks for {k} clusters but the cohort has only "
            f"{n_customers} customers; k-means needs at least one point per cluster"
        )

    if kind != "kmeans":  # pragma: no cover — the grammar admits nothing else yet
        raise ValueError(f"unsupported cluster kind {kind!r} from name {name!r}")

    model = KMeans(n_clusters=k, n_init=_N_INIT, random_state=_RANDOM_STATE)
    return model.fit_predict(_standardize(features)).astype(np.int64)
