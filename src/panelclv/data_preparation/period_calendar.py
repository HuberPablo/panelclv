"""Calendar time <-> period index, written once.

A panel is indexed by *period*, not by date, so the package converts between the two
constantly: a date has to become a week number, a ``(year, week)`` pair has to become
the Timestamp that anchors a train/holdout split, a flat week counter has to become the
``(year, week)`` columns a panel is written in, and elapsed calendar days have to become
elapsed periods. Each of those conversions used to be written wherever it was needed,
which is how four week-numbering conventions and three period-length tables came to
coexist. The failure mode is silent — a Pareto/NBD fit on mis-scaled sufficient
statistics still converges, it just fits something else — so all of it lives here and
every caller reads it from here.

The weekly convention, stated once
----------------------------------
A year is ``WEEKS_PER_YEAR = 52`` weeks of seven days, numbered **0..51**, and week
``w`` of year ``Y`` starts on ``Y-01-01 + 7w`` days. That covers days 1..364 of the
calendar year; the one or two days left over (Dec 31, plus Dec 30 in a leap year) fold
back into week 51 rather than opening a 53rd week.

This is deliberately *not* ISO 8601. ISO weeks are 1-based, do not start on Jan 1, and
give some years 53 of them — and a 53rd week aliases straight onto week 0 under a
52-week sine, which is a seasonal feature quietly pointing at the wrong season.

Period length in days
---------------------
Only the Pareto/NBD benchmark needs this: its sufficient statistics are recency and
observation age measured in *periods*, obtained by dividing elapsed days by the length
of one period. Two tables used to answer that question and they disagreed on
``monthly`` (30.0 against 30.4368), both feeding the same fit. See ``_DAYS_PER_PERIOD``
for which value won and why.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# One year of a weekly panel. Also the divisor of the week-of-year sine/cosine, so a
# week index must never reach 52: sin(2*pi*52/52) == sin(0), i.e. the new year would be
# encoded as identical to the old one.
WEEKS_PER_YEAR = 52

# Calendar days in one period of each supported PanelConfig.frequency.
#
# `monthly` is the mean Gregorian calendar month, written as the quotient it is so the
# value carries its own derivation (a bare 30.4368 is a rounding of it, and the reader
# has to reverse-engineer the year length). The competing table used a flat 30.0, which
# is not a month by any definition and shortens the period by 1.4 % — on a three-year
# calibration window that inflates every customer's observation age by about half a
# month, and the Pareto/NBD fit absorbs the difference into its dropout rate without
# complaining. Weekly and daily are exact and were never in dispute.
_DAYS_PER_PERIOD: dict[str, float] = {
    "daily": 1.0,
    "weekly": 7.0,
    "monthly": 365.2425 / 12,       # 30.436875
}


def days_per_period(frequency: str) -> float:
    """Calendar days spanned by one period of `frequency`.

    Raises on an unrecognised frequency rather than assuming weekly: a panel whose
    frequency does not map to a period length is a panel whose Pareto/NBD statistics
    would be computed on the wrong time scale, and that is not something to guess at.
    """
    try:
        return _DAYS_PER_PERIOD[frequency]
    except KeyError:
        raise ValueError(
            f"cannot map frequency {frequency!r} to a period length; "
            f"known frequencies: {sorted(_DAYS_PER_PERIOD)}."
        ) from None


def week_of_year(dates: pd.Series) -> pd.Series:
    """0-based week-of-year (0..51) of each date in `dates`.

    The date-to-week direction of the convention: day-of-year 1..7 is week 0, 8..14 is
    week 1, and the year's trailing day or two fold into week 51.
    """
    doy = pd.to_datetime(dates).dt.dayofyear.astype(np.int64)
    return ((doy - 1) // 7).clip(upper=WEEKS_PER_YEAR - 1)


def week_start(years: pd.Series, weeks: pd.Series) -> pd.Series:
    """Timestamp on which 0-based week `weeks` of calendar year `years` begins.

    The inverse direction of `week_of_year`, and the anchor a train/holdout split is
    cut on: Jan 1 of the year, plus seven days per elapsed week.
    """
    return (
        pd.to_datetime(years.astype(str) + "-01-01")
        + pd.to_timedelta(weeks.astype(np.int64) * 7, unit="D")
    )


def year_and_week(week_index, start_year: int):
    """Split a flat 0-based week counter into `(calendar year, week-of-year)`.

    A generated panel counts weeks from 0 at `start_year`; the panel it writes is in
    `(year, week)` columns, so the counter has to be unpacked. Accepts a numpy array or
    a pandas Series and returns the same kind.
    """
    return start_year + week_index // WEEKS_PER_YEAR, week_index % WEEKS_PER_YEAR


def flat_week_index(years, weeks, start_year: int):
    """Fold `(calendar year, week-of-year)` back into a flat 0-based week counter.

    The inverse of `year_and_week`, used to line a stored panel's rows up with the
    forecast columns, which are on the flat index.
    """
    return (years - start_year) * WEEKS_PER_YEAR + weeks
