"""Behavioural clusters are static, leak-free, and reach the model as a category.

A cluster label is the first covariate in the package that is *derived from the
target* yet **not** recomputed during the rollout. That combination is what makes it
cheap — no simulator machinery — and it is also what makes it worth pinning down,
because both halves are silent when they break:

* if the label were computed over the whole panel instead of the calibration slice,
  it would carry holdout information into every training step and nothing would raise;
* if the label were left out of `embedded_cols`, `standardize_covariates` would turn
  an arbitrary group index into a z-score, imposing an ordering the labels do not
  have — again with no shape error.

So the tests below assert the properties rather than the implementation: the label is
constant per customer across both windows, it does not move when the holdout is
rewritten, it survives a rollout unchanged, and it arrives at the embedder as a
category of the declared cardinality.

Run:  pytest -q tests/test_cluster_features.py
"""

import numpy as np
import pandas as pd
import pytest
import torch

from panelclv.configs.cluster_feature_names import parse_cluster_feature
from panelclv.configs.panel_config import PanelConfig
from panelclv.data_preparation import panel_dataset
from panelclv.data_preparation.cluster_features import compute_cluster_labels
from panelclv.models.embedders import ProjectedEmbedder, ValendinEmbedder
from panelclv.models.monte_carlo_forecasting import simulate_recurrent_path

PANEL_SEED = 20260901
N_CUSTOMERS = 120
N_CAL_PERIODS = 104
N_HOLD_PERIODS = 52
K = 4
CLUSTER_FEATURE = f"kmeans_{K}"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _panel(holdout_scale: int = 1) -> pd.DataFrame:
    """One row per customer per week, Poisson counts with a spread of activity.

    `holdout_scale` multiplies the HOLDOUT counts only. It exists for the leakage
    test: rewriting the future must not move a label that claims to be computed from
    the past.
    """
    rng = np.random.default_rng(PANEL_SEED)
    rates = rng.gamma(shape=1.2, scale=0.05, size=N_CUSTOMERS)
    weeks = pd.date_range("1999-01-04", periods=N_CAL_PERIODS + N_HOLD_PERIODS, freq="7D")
    iso = [(d.isocalendar()[0], d.isocalendar()[1]) for d in weeks]

    frames = []
    for i, rate in enumerate(rates):
        counts = rng.poisson(rate, size=len(weeks))
        # Guarantee a calibration transaction so the cohort filter keeps the customer.
        if counts[:N_CAL_PERIODS].sum() == 0:
            counts[rng.integers(0, N_CAL_PERIODS)] = 1
        counts[N_CAL_PERIODS:] = counts[N_CAL_PERIODS:] * holdout_scale
        frames.append(
            pd.DataFrame({
                "Id": f"C{i:03d}",
                "year": [y for y, _ in iso],
                "week": [w for _, w in iso],
                "Transactions": counts,
            })
        )
    return pd.concat(frames, ignore_index=True)


def _config(cluster_features=(CLUSTER_FEATURE,)) -> PanelConfig:
    return PanelConfig(
        id_col="Id",
        target_col="Transactions",
        frequency="weekly",
        training_start="1999-01-04",
        training_end="2000-12-25",
        validation_start="2000-01-03",
        holdout_start="2001-01-01",
        holdout_end="2001-12-31",
        time_cols=("year", "week"),
        clip_target_upper=6,
        require_calibration_activity=True,
        time_features={"add_week_sin_cos": True},
        cluster_features=tuple(cluster_features),
        embedded_cols={"Transactions": "auto"},
    )


@pytest.fixture(scope="module")
def prepared() -> dict:
    return panel_dataset.prepare_dataset(_panel(), _config(), verbose=False)


def _cluster_channel(data: dict) -> np.ndarray:
    return np.asarray(data["seq_cols"]).tolist().index(CLUSTER_FEATURE)


# --------------------------------------------------------------------------- #
# The name grammar
# --------------------------------------------------------------------------- #


def test_grammar_reads_the_cluster_count():
    assert parse_cluster_feature("kmeans_8") == ("kmeans", 8)


@pytest.mark.parametrize("name", ["kmeans_1", "kmeans_0", "kmeans", "kmeans_x", "dbscan_4"])
def test_grammar_rejects_unusable_names(name):
    with pytest.raises(ValueError):
        parse_cluster_feature(name)


def test_config_validates_at_construction():
    """A bad name fails when the config is written, not when the panel is prepared."""
    with pytest.raises(ValueError):
        _config(cluster_features=("kmeans_1",))


# --------------------------------------------------------------------------- #
# The column reaches the tensor as a category
# --------------------------------------------------------------------------- #


def test_column_is_last_on_the_feature_axis(prepared):
    """Cluster features come after every role and after the AR block."""
    assert prepared["seq_cols"][-1] == CLUSTER_FEATURE


def test_embedded_automatically_with_the_declared_cardinality(prepared):
    """The caller never writes it into `embedded_cols`; K is pinned, not inferred."""
    assert prepared["embedded_cols"][CLUSTER_FEATURE] == K


def test_not_standardized(prepared):
    """An embedded column is excluded from `standardize_covariates` — so the channel
    still holds integer labels in [0, K), not z-scores."""
    assert CLUSTER_FEATURE not in (prepared["covariate_stats"] or {})
    channel = prepared["calibration"][:, :, _cluster_channel(prepared)]
    assert set(np.unique(channel)).issubset(set(float(i) for i in range(K)))


def test_every_cluster_is_used(prepared):
    """K clusters means K populated clusters — otherwise the pinned cardinality would
    reserve embedding rows that no customer ever indexes."""
    channel = prepared["calibration"][:, 0, _cluster_channel(prepared)]
    assert len(np.unique(channel)) == K


# --------------------------------------------------------------------------- #
# It is static
# --------------------------------------------------------------------------- #


def test_constant_across_calibration_and_holdout(prepared):
    """One label per customer, held for every period of BOTH windows.

    This is the whole reason the rollout needs no cluster machinery: there is nothing
    to advance, so there is nothing to get wrong.
    """
    idx = _cluster_channel(prepared)
    both = np.concatenate(
        [prepared["calibration"][:, :, idx], prepared["holdout"][:, :, idx]], axis=1
    )
    assert (both == both[:, :1]).all()


# --------------------------------------------------------------------------- #
# It is leak-free
# --------------------------------------------------------------------------- #


def test_labels_ignore_the_holdout(prepared):
    """Rewrite the future; the labels must not move.

    The panels differ only in their holdout counts (tripled), so a label that shifts
    is a label computed over a window it has no right to see. This is the test that
    would have caught clustering the full per-customer series instead of the
    calibration slice.
    """
    rewritten = panel_dataset.prepare_dataset(
        _panel(holdout_scale=3), _config(), verbose=False
    )
    assert rewritten["ids"] == prepared["ids"]
    idx = _cluster_channel(prepared)
    np.testing.assert_array_equal(
        prepared["calibration"][:, 0, idx], rewritten["calibration"][:, 0, idx]
    )


def test_labels_are_deterministic(prepared):
    """Same panel, same labels — so cluster assignment is a property of the data and
    never a hidden source of across-study variance (`studies.config`: `base_seed + i`
    drives the Optuna sampler and the forecast, nothing else)."""
    again = panel_dataset.prepare_dataset(_panel(), _config(), verbose=False)
    idx = _cluster_channel(prepared)
    np.testing.assert_array_equal(
        prepared["calibration"][:, 0, idx], again["calibration"][:, 0, idx]
    )


def test_too_few_customers_for_k_raises():
    """k-means cannot fill more clusters than there are customers, and says so."""
    target = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=np.float32)
    with pytest.raises(ValueError, match="at least one point per cluster"):
        compute_cluster_labels(target, "kmeans_8")


# --------------------------------------------------------------------------- #
# It survives the rollout
# --------------------------------------------------------------------------- #


class _ChannelRecorder(torch.nn.Module):
    """A stand-in rollout model: records what it is fed, samples nothing.

    `simulate_recurrent_path` expects `(sampled, state)` with `sampled` holding class
    indices of shape (N, T, 1) — this is the rollout-model contract (ADR-0007), not
    the trained model's logits.
    """

    def __init__(self, channel: int):
        super().__init__()
        self.channel = channel
        self.seen: list[torch.Tensor] = []

    def forward(self, x, state=None):
        self.seen.append(x[:, :, self.channel].clone())
        return torch.ones(x.shape[0], x.shape[1], 1), state


def test_rollout_feeds_the_label_unchanged(prepared):
    """Every holdout step must carry the customer's calibration label.

    `simulate_recurrent_path` overwrites only the target channel and the AR channels;
    this asserts that a static cluster column really does ride through all 52 steps
    untouched, which is the claim the design rests on.
    """
    idx = _cluster_channel(prepared)
    model = _ChannelRecorder(idx)
    simulate_recurrent_path(
        model,
        prepared["calibration"],
        prepared["holdout"],
        seq_cols=prepared["seq_cols"],
        target_idx=int(prepared["target_idx"]),
        device="cpu",
        ar_features=prepared.get("ar_features", []),
        covariate_stats=prepared.get("covariate_stats"),
    )
    expected = torch.as_tensor(prepared["calibration"][:, 0, idx])
    # seen[0] is the warm-up over the whole calibration window; the rest are the
    # single-period holdout steps.
    for step in model.seen:
        assert (step == expected[:, None]).all()


# --------------------------------------------------------------------------- #
# It reaches the embedder as a category
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strategy", [ProjectedEmbedder, ValendinEmbedder])
def test_embedders_build_a_lookup_for_the_label(prepared, strategy):
    """Neither embedder names a column, so both learn a cluster embedding for free —
    which is why no new `Embedder` subclass was needed (ADR-0005)."""
    embedder = strategy(
        seq_cols=prepared["seq_cols"],
        embedded_cols=prepared["embedded_cols"],
        target_col=prepared["target_col"],
    )
    out = embedder(torch.as_tensor(prepared["calibration"][:4]))
    assert out.shape[:2] == (4, prepared["T_CAL"])
    assert out.shape[-1] == embedder.output_dim
