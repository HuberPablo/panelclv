"""The time-flag set is written once, and every column a flag creates carries a role.

"The set of engineered calendar flags" used to be spelled out four times — the known-key
tuple, the per-frequency compatibility map, the flag-to-column map, and
`add_time_features`' own if/elif chain — and the four had already drifted:
`add_dayofyear_sin_cos` wrote a raw `dayofyear` column that no role table mentioned, so it
reached the model-ready tensor as a channel nothing ever read.

These tests pin the property that drift broke. The table in `configs.panel_config` is the
single statement of the set; `add_time_features` is checked *against* it rather than
restating it, so a flag added to one and not the other fails here.
"""

import pandas as pd
import pytest

from panelclv.configs.panel_config import TIME_FEATURE_FLAGS, PanelConfig
from panelclv.data_preparation.dynamic_panel_dataset import add_time_features

# One row per period of 2019, in each of the three frequencies the package supports.
# Small enough to read, long enough that every flag has something to compute on.
_DATES = pd.date_range("2019-01-01", "2019-12-31", freq="D")

# How each frequency states its time index, in the two forms that need it: the column
# declaration `PanelConfig` takes, and the bare panel `add_time_features` reads. Written
# once so the two cannot disagree about which column a monthly panel is indexed by.
_TIME_INDEX = {
    "daily": {"date_col": "Date"},
    "weekly": {"time_cols": ("year", "week")},
    "monthly": {"time_cols": ("year", "month")},
}


def _panel(frequency: str) -> pd.DataFrame:
    """A bare panel for `frequency`: one row per period of 2019, no features yet."""
    if frequency == "daily":
        return pd.DataFrame({"Date": _DATES})
    period = _TIME_INDEX[frequency]["time_cols"][1]
    span = range(52) if frequency == "weekly" else range(1, 13)
    return pd.DataFrame({"year": 2019, period: list(span)})


def _config(frequency: str, **overrides) -> PanelConfig:
    """A minimal valid `PanelConfig` on `frequency`, with the windows filled in.

    The windows are not what any of these tests is about — they are the required fields
    a config cannot be built without — so they are supplied once here.
    """
    return PanelConfig(
        id_col="Id",
        target_col="Transactions",
        frequency=frequency,
        training_start="2019-01-01",
        training_end="2019-06-30",
        validation_start="2019-05-01",
        holdout_start="2019-07-01",
        holdout_end="2019-12-31",
        **_TIME_INDEX[frequency],
        **overrides,
    )


def _flag_frequency_pairs() -> list[tuple[str, str]]:
    """Every (flag, frequency) the table declares producible — the full matrix."""
    return [
        (flag, freq)
        for flag, spec in TIME_FEATURE_FLAGS.items()
        for freq in sorted(spec.frequencies)
    ]


@pytest.mark.parametrize("flag,frequency", _flag_frequency_pairs())
def test_a_flag_creates_exactly_the_columns_the_table_declares(flag, frequency):
    """The columns `add_time_features` writes are the ones the table names — no more.

    This is the assertion the orphan `dayofyear` column failed: it was written by the
    daily branch and absent from the table, so nothing downstream could give it a role.
    Comparing the *difference* in columns (rather than checking the declared ones are
    present) is what makes an extra write a failure rather than a silent addition.
    """
    panel = _panel(frequency)
    before = set(panel.columns)

    out = _add(panel, flag, frequency)

    assert set(out.columns) - before == set(TIME_FEATURE_FLAGS[flag].columns)


def _add(panel: pd.DataFrame, flag: str, frequency: str) -> pd.DataFrame:
    """Run `add_time_features` for one flag on `panel`."""
    return add_time_features(
        panel,
        time_features={flag: True},
        frequency=frequency,
        base_year=2019,
        **_TIME_INDEX[frequency],
    )


@pytest.mark.parametrize("flag,frequency", _flag_frequency_pairs())
def test_the_config_accepts_every_flag_the_table_declares_for_a_frequency(flag, frequency):
    """A flag the table calls producible survives `PanelConfig`'s compatibility filter.

    `PanelConfig` drops frequency-incompatible flags with a warning rather than raising,
    so a compatibility map that disagreed with `add_time_features` would silently strip a
    feature the pipeline could have built. Both now read the same table; this holds them
    to it.
    """
    assert _config(frequency, time_features={flag: True}).time_features == {flag: True}


@pytest.mark.parametrize("flag,spec", sorted(TIME_FEATURE_FLAGS.items()))
def test_an_incompatible_frequency_is_rejected_by_both_sides(flag, spec):
    """A flag its frequency cannot produce warns at config time and raises at build time.

    The two halves used to encode the compatibility separately, so one could accept what
    the other refused. Every frequency outside the flag's declared set is checked.
    """
    for frequency in ("weekly", "monthly", "daily"):
        if frequency in spec.frequencies:
            continue

        with pytest.warns(UserWarning, match="not compatible"):
            cfg = _config(frequency, time_features={flag: True})
        assert cfg.time_features == {}

        with pytest.raises(ValueError, match=flag):
            _add(_panel(frequency), flag, frequency)


def test_every_created_column_reaches_the_feature_schema():
    """No flag can create a column that ends up with no role.

    `year_idx` is the one deliberate exception the table records: it is a trend feature
    whose role (usually `known_future`) is the caller's choice, so it is placed
    explicitly rather than auto-assigned. Every other created column must appear under
    some role in `PanelConfig.schema`, which is what stops another orphan appearing.
    """
    # Daily is the only frequency that can produce all four flags at once.
    cfg = _config(
        "daily",
        known_future=("year_idx",),
        time_features={flag: True for flag in TIME_FEATURE_FLAGS},
    )
    roled = {col for cols in cfg.schema.values() for col in cols}

    created = {col for spec in TIME_FEATURE_FLAGS.values() for col in spec.columns}
    assert created <= roled
