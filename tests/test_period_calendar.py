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
    days_per_period,
    flat_week_index,
    week_of_year,
    week_start,
    year_and_week,
)


# ---------------------------------------------------------------------------
# The week convention: date <-> (year, week)
# ---------------------------------------------------------------------------


def test_week_of_year_starts_at_zero_on_january_first():
    """Day-of-year 1..7 is week 0, 8..14 is week 1 — 0-based, aligned to Jan 1."""
    dates = pd.Series(pd.to_datetime(["2019-01-01", "2019-01-07", "2019-01-08"]))
    np.testing.assert_array_equal(week_of_year(dates), [0, 0, 1])


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


def test_week_start_anchors_on_january_first_plus_seven_days_per_week():
    """The (year, week) -> date direction, spelled out on one known case."""
    starts = week_start(pd.Series([2019, 2019]), pd.Series([0, 3]))
    assert list(starts) == [pd.Timestamp("2019-01-01"), pd.Timestamp("2019-01-22")]


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
