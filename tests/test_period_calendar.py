"""The week convention and the period-length table are one answer each.

`period_calendar` exists because they used to be several. Four week-numbering
conventions and three days-per-period tables coexisted, and two of the tables
disagreed on `monthly` — 30.0 against 30.4368 — with both feeding the Pareto/NBD fit.
Nothing raised; a fit on the wrong time scale converges perfectly well.

These tests pin the convention itself rather than any one caller, because the callers
are what drifted. The round-trip tests are the load-bearing ones: they assert that the
date-to-week direction (`week_of_year`, used to build a daily panel's seasonal feature)
and the week-to-date direction (`week_start`, used to cut the train/holdout split) are
inverses. When they were written separately they were not.
"""

import numpy as np
import pandas as pd
import pytest

from panelclv.data_preparation.period_calendar import (
    WEEKS_PER_YEAR,
    complete_week_grid,
    days_per_period,
    flat_week_index,
    week_of_year,
    week_start,
    year_and_week,
)


# ---------------------------------------------------------------------------
# The week convention: date <-> (year, week)
# ---------------------------------------------------------------------------


def test_week_of_year_is_the_valendin_grid():
    """`dayofyear // 7`: week 0 is Jan 1..6, week 1 opens on Jan 7 (ADR-0009).

    Pinned on the day where it differs from `(dayofyear - 1) // 7`, the rule this package
    used before: Jan 7 is week 1 here and was week 0 there.
    """
    dates = pd.Series(pd.to_datetime(["2019-01-01", "2019-01-06", "2019-01-07"]))
    np.testing.assert_array_equal(week_of_year(dates), [0, 0, 1])


@pytest.mark.parametrize(
    "year, first_day_of_week_51", [(2019, "2019-12-23"), (2020, "2020-12-22")]
)
def test_week_51_opens_on_day_357_and_runs_to_new_years_eve(year, first_day_of_week_51):
    """The year-end bucket is nine days in a common year, ten in a leap year."""
    days = pd.Series(pd.date_range(first_day_of_week_51, f"{year}-12-31", freq="D"))
    assert (week_of_year(days) == WEEKS_PER_YEAR - 1).all()
    day_before = pd.Series([pd.Timestamp(first_day_of_week_51) - pd.Timedelta(days=1)])
    assert week_of_year(day_before).item() == WEEKS_PER_YEAR - 2


@pytest.mark.parametrize("year", [2019, 2020])       # a common and a leap year
def test_week_of_year_never_reaches_the_divisor(year):
    """No date maps to week 52 — the trailing day or two fold back into week 51.

    A 52nd week would alias onto week 0 under the `WEEKS_PER_YEAR` sine
    (`sin(2*pi*52/52) == sin(0)`), i.e. New Year's Eve encoded as New Year's Day, and
    would overflow a `week` embedding sized at 52. This is what
    `isocalendar().week - 1` used to allow.
    """
    every_day = pd.Series(pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"))
    weeks = week_of_year(every_day)
    assert weeks.min() == 0
    assert weeks.max() == WEEKS_PER_YEAR - 1


def test_week_start_and_week_of_year_are_inverses():
    """Every week's own start date reads back as that week."""
    weeks = pd.Series(range(WEEKS_PER_YEAR))
    years = pd.Series([2019] * WEEKS_PER_YEAR)
    np.testing.assert_array_equal(week_of_year(week_start(years, weeks)), weeks)


def test_week_start_anchors_week_zero_on_january_first_and_later_weeks_a_day_early():
    """The (year, week) -> date direction, spelled out on known cases.

    CDNOW's calibration cut is the one that matters: 1997 week 39 opens on Sep 30.
    """
    starts = week_start(pd.Series([2019, 2019, 1997]), pd.Series([0, 3, 39]))
    assert list(starts) == [
        pd.Timestamp("2019-01-01"), pd.Timestamp("2019-01-21"), pd.Timestamp("1997-09-30"),
    ]


def test_week_start_inverts_every_day_of_a_leap_and_a_common_year():
    """Each day reads back to a week whose start is on or before it, and the next
    week's start is after it — the property window slicing on `period_start` needs."""
    days = pd.Series(pd.date_range("2019-01-01", "2020-12-31", freq="D"))
    weeks = week_of_year(days)
    years = days.dt.year
    assert (week_start(years, weeks) <= days).all()
    not_last = weeks < WEEKS_PER_YEAR - 1
    assert (week_start(years[not_last], weeks[not_last] + 1) > days[not_last]).all()


# ---------------------------------------------------------------------------
# complete_week_grid: which weeks a panel builder keeps
# ---------------------------------------------------------------------------


def test_complete_week_grid_keeps_the_short_first_and_long_last_week_of_a_full_year():
    grid = complete_week_grid(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))
    assert len(grid) == WEEKS_PER_YEAR
    assert grid.iloc[0].tolist() == [2020, 0]
    assert grid.iloc[-1].tolist() == [2020, WEEKS_PER_YEAR - 1]


def test_complete_week_grid_drops_a_week_the_data_only_partly_covers():
    """CDNOW's window: 1997-01-01..1998-06-30 ends exactly on 1998 week 25 (Jun 24..30),
    and stopping a day earlier loses that week."""
    grid = complete_week_grid(pd.Timestamp("1997-01-01"), pd.Timestamp("1998-06-30"))
    assert grid.iloc[-1].tolist() == [1998, 25]
    assert len(grid) == WEEKS_PER_YEAR + 26
    shorter = complete_week_grid(pd.Timestamp("1997-01-01"), pd.Timestamp("1998-06-29"))
    assert shorter.iloc[-1].tolist() == [1998, 24]


def test_complete_week_grid_drops_a_year_end_week_missing_new_years_eve():
    grid = complete_week_grid(pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-30"))
    assert grid.iloc[-1].tolist() == [2019, WEEKS_PER_YEAR - 2]


# ---------------------------------------------------------------------------
# The week convention: flat counter <-> (year, week)
# ---------------------------------------------------------------------------


def test_year_and_week_rolls_the_year_over_at_the_convention():
    """Week 51 is still the start year; week 52 is week 0 of the next one."""
    year, week = year_and_week(np.array([0, 51, 52, 53]), start_year=1999)
    np.testing.assert_array_equal(year, [1999, 1999, 2000, 2000])
    np.testing.assert_array_equal(week, [0, 51, 0, 1])


def test_flat_week_index_inverts_year_and_week():
    """A generated panel's (year, week) columns fold back to the counter that made them."""
    flat = np.arange(3 * WEEKS_PER_YEAR)
    year, week = year_and_week(flat, start_year=1999)
    np.testing.assert_array_equal(flat_week_index(year, week, start_year=1999), flat)


# ---------------------------------------------------------------------------
# The period-length table
# ---------------------------------------------------------------------------


def test_monthly_is_the_mean_gregorian_month_not_a_flat_thirty():
    """The resolved disagreement: 365.2425 / 12, not 30.0.

    Pinned rather than left implicit because a flat 30.0 is the value this package
    carried in its other table, and it is wrong quietly — it shortens the period by
    1.4 %, which the Pareto/NBD fit absorbs into its dropout rate.
    """
    assert days_per_period("monthly") == 365.2425 / 12
    assert days_per_period("weekly") == 7.0
    assert days_per_period("daily") == 1.0


def test_an_unknown_frequency_raises_rather_than_falling_back_to_weekly():
    """A silent weekly fallback is how a mis-scaled fit would reach the results table."""
    with pytest.raises(ValueError, match="cannot map frequency"):
        days_per_period("quarterly")
