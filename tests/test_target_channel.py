"""The target column is derived in one place, and everything else reads it.

`prepare_dataset` decides where the count column sits on the feature axis and records
that as `target_idx`. Six other sites used to work it out again from `seq_cols` and
`target_col`, and the extraction `holdout[:, :, target_idx]` was spelled out at four of
them. Every one of those re-derivations still produces an `(N, T)` float array of
plausible-looking counts when it goes wrong, so a drift between any two of them scores
the wrong channel in silence — which is what makes this worth pinning.

These tests hold the two halves: the single derivation agrees with what the producer
recorded, and the accessors read the recorded value rather than working it out again.
"""

import numpy as np
import pandas as pd
import pytest

from panelclv.configs.panel_config import PanelConfig
from panelclv.data_preparation import dynamic_panel_dataset
from panelclv.data_preparation.target_channel import (
    calibration_counts,
    holdout_actuals,
    target_index,
)


def _data_dict() -> dict:
    """A hand-built stand-in for `prepare_dataset`'s output, with distinct channels.

    Channel values are made unmistakable — channel `f` holds `100 * f + t` — so a test
    reading the wrong one cannot pass by coincidence.
    """
    n, t_cal, t_hold, f = 3, 5, 4, 3
    calibration = np.zeros((n, t_cal, f), dtype=np.float32)
    holdout = np.zeros((n, t_hold, f), dtype=np.float32)
    for channel in range(f):
        calibration[:, :, channel] = 100 * channel + np.arange(t_cal)
        holdout[:, :, channel] = 100 * channel + np.arange(t_hold)
    return {
        "seq_cols": ["week_sin", "Transactions", "year_idx"],
        "target_col": "Transactions",
        "target_idx": 1,
        "calibration": calibration,
        "holdout": holdout,
    }


# ---------------------------------------------------------------------------
# The one derivation
# ---------------------------------------------------------------------------


def test_target_index_is_the_position_on_the_feature_axis():
    assert target_index(["week_sin", "Transactions", "year_idx"], "Transactions") == 1


def test_target_index_names_the_missing_column():
    """An absent target is a config error, and the message has to say which column.

    The failure it guards against is a renamed target silently indexing channel 0.
    """
    with pytest.raises(ValueError, match="Transactions"):
        target_index(["week_sin", "year_idx"], "Transactions")


def test_prepare_dataset_records_what_the_derivation_computes():
    """The producer's `target_idx` and the single derivation agree on a real dataset.

    This is the "produced once" half: if `prepare_dataset` ever reordered `seq_cols`
    without updating the recorded index, every consumer reading that key would score a
    covariate channel, and this is the only place that would notice.
    """
    data = dynamic_panel_dataset.prepare_dataset(_panel(), _config(), verbose=False)
    assert data["target_idx"] == target_index(data["seq_cols"], data["target_col"])


# ---------------------------------------------------------------------------
# The one extraction
# ---------------------------------------------------------------------------


def test_holdout_actuals_is_the_recorded_target_channel():
    data = _data_dict()
    expected = np.asarray(data["holdout"])[:, :, 1]
    np.testing.assert_array_equal(holdout_actuals(data), expected)


def test_calibration_counts_is_the_recorded_target_channel():
    data = _data_dict()
    expected = np.asarray(data["calibration"])[:, :, 1]
    np.testing.assert_array_equal(calibration_counts(data), expected)


def test_the_accessors_read_the_recorded_index_rather_than_re_deriving_it():
    """`target_idx` is the authority, not `seq_cols.index(target_col)`.

    A tuned trial hands its consumers a *reduced* feature layout, where the target sits
    at a different position from the full dataset's; the dict it carries records that
    position. Pointing the recorded index at a different channel here is the cheapest
    way to state which of the two the accessors are contractually reading.
    """
    data = _data_dict()
    data["target_idx"] = 2

    np.testing.assert_array_equal(
        holdout_actuals(data), np.asarray(data["holdout"])[:, :, 2]
    )
    np.testing.assert_array_equal(
        calibration_counts(data), np.asarray(data["calibration"])[:, :, 2]
    )


# ---------------------------------------------------------------------------
# A tiny weekly panel — enough for `prepare_dataset` to run end to end
# ---------------------------------------------------------------------------


def _panel() -> pd.DataFrame:
    """8 customers × two years of weekly counts, deterministic."""
    rng = np.random.default_rng(0)
    rows = []
    for cid in range(8):
        for year in (2019, 2020):
            for week in range(52):
                rows.append(
                    {
                        "Id": f"c{cid}",
                        "year": year,
                        "week": week,
                        "Transactions": int(rng.integers(0, 3)),
                    }
                )
    return pd.DataFrame(rows)


def _config() -> PanelConfig:
    return PanelConfig(
        id_col="Id",
        target_col="Transactions",
        frequency="weekly",
        training_start="2019-01-01",
        training_end="2019-12-31",
        validation_start="2019-10-01",
        holdout_start="2020-01-01",
        holdout_end="2020-12-31",
        time_cols=("year", "week"),
        time_features={"add_week_sin_cos": True},
        embedded_cols={"Transactions": "auto"},
    )
