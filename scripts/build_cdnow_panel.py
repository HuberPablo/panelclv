"""Build the CDNOW weekly customer-period panel from the raw master file.

CDNOW is the standard public customer-base dataset (Fader & Hardie): 23,570
customers who made their first purchase in the first quarter of 1997, tracked to
1998-06-30. The raw `cdnow_master.txt` is whitespace-delimited with four unnamed
fields — customer id, date `YYYYMMDD`, number of CDs bought, dollar value — and one
row per purchase occasion.

Output is the layout every other panel in `Datasets/Dataset_clean/` uses, so
`prepare_dataset` reads it with no special case::

    Id, year, week, Transactions

`Transactions` is the per-customer per-week purchase-occasion count (rows in the
master file — the dollar value and CD count are dropped, since the models forecast
counts). CDNOW carries no covariates, so the panel has none: none are fabricated.

**Week numbering** comes from `data_preparation.period_calendar`, the package's one
calendar-time <-> period-index convention, rather than being restated here. That
makes `week_start` the exact inverse of the bucketing, which is what the window
dates in the study config are sliced on.

**The grid is trimmed to complete weeks.** The raw data stops on 1998-06-30, mid-week;
a panel that kept that partial week would report a genuine low count for it and the
last holdout period would read as a drop in demand that is really a drop in
observation. The last kept week is therefore the last one whose seven days all fall
within the data window.

Usage:
    python scripts/build_cdnow_panel.py                     # default paths
    python scripts/build_cdnow_panel.py --raw path/to/cdnow_master.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from panelclv.data_preparation.period_calendar import (
    WEEKS_PER_YEAR,
    week_of_year,
    week_start,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = REPO_ROOT / "Datasets" / "cdnow_master.txt"
DEFAULT_OUT = REPO_ROOT / "Datasets" / "Dataset_clean" / "cdnow_customer_week_panel.csv"

# The four fields of the master file, in order. Only the first two are used.
RAW_COLUMNS = ("Id", "Date", "cds", "dollars")

# The observation window of the dataset itself, not a modelling choice: CDNOW's first
# purchases fall in 1997 Q1 and the file ends on 1998-06-30.
DATA_START = pd.Timestamp("1997-01-01")
DATA_END = pd.Timestamp("1998-06-30")


def read_master(path: Path) -> pd.DataFrame:
    """Read `cdnow_master.txt` into a (Id, Date) transaction frame."""
    if not path.exists():
        raise FileNotFoundError(
            f"CDNOW master file not found at {path}.\n"
            "Download `cdnow_master.txt` (Fader & Hardie's public CDNOW dataset) and "
            "place it there, or pass --raw with its location."
        )
    tx = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=list(RAW_COLUMNS),
        dtype={"Id": "int64"},
    )
    tx["Date"] = pd.to_datetime(tx["Date"].astype(str), format="%Y%m%d")
    return tx[["Id", "Date"]]


def complete_week_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Every (year, week) cell whose seven days lie inside [start, end].

    Returned in calendar order, so the panel's period axis is already sorted the way
    `prepare_dataset` reshapes it.
    """
    years = range(start.year, end.year + 1)
    grid = pd.DataFrame(
        [(y, w) for y in years for w in range(WEEKS_PER_YEAR)],
        columns=["year", "week"],
    )
    starts = week_start(grid["year"], grid["week"])
    # A week is complete when both its first and its last day are observed. The last
    # week of a calendar year runs a day or two long under this convention, so the
    # test is written on the following week's start rather than on start + 6 days.
    next_starts = starts + pd.Timedelta(days=7)
    keep = (starts >= start) & (next_starts <= end + pd.Timedelta(days=1))
    return grid.loc[keep].reset_index(drop=True)


def build_weekly_panel(tx: pd.DataFrame) -> pd.DataFrame:
    """Dense customer x week panel of purchase-occasion counts."""
    tx = tx[(tx["Date"] >= DATA_START) & (tx["Date"] <= DATA_END)].copy()
    tx["year"] = tx["Date"].dt.year
    tx["week"] = week_of_year(tx["Date"])

    counts = (
        tx.groupby(["Id", "year", "week"], as_index=False)
        .size()
        .rename(columns={"size": "Transactions"})
    )

    # The dense grid: every customer observed in every complete week, zeros included.
    # A customer-period model needs the zeros — they are the periods without a
    # purchase, which is most of them.
    customers = pd.DataFrame({"Id": sorted(tx["Id"].unique())})
    periods = complete_week_grid(DATA_START, DATA_END)
    panel = customers.merge(periods, how="cross")

    panel = panel.merge(counts, on=["Id", "year", "week"], how="left")
    panel["Transactions"] = panel["Transactions"].fillna(0).astype("int64")
    return panel.sort_values(["Id", "year", "week"]).reset_index(drop=True)


def describe(panel: pd.DataFrame) -> None:
    """Print what was built, including the count tail `clip_target_upper` has to cover."""
    n_cust = panel["Id"].nunique()
    n_periods = len(panel) // n_cust
    print(f"customers      : {n_cust}")
    print(f"weeks/customer : {n_periods}")
    print(f"rows           : {len(panel)}")
    periods = panel[["year", "week"]].drop_duplicates().sort_values(["year", "week"])
    first, last = periods.iloc[0], periods.iloc[-1]
    print(
        f"window         : {first['year']} w{first['week']} .. "
        f"{last['year']} w{last['week']}"
    )
    print("\nweekly transaction-count distribution (share of customer-weeks):")
    share = panel["Transactions"].value_counts(normalize=True).sort_index()
    for count, frac in share.items():
        print(f"  {count:>3} : {frac:12.8f}")
    print(
        "\nSet PanelConfig.clip_target_upper at or above the largest count carrying "
        "meaningful mass — it is the softmax head size."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    panel = build_weekly_panel(read_master(args.raw))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out, index=False)
    print(f"wrote {args.out}\n")
    describe(panel)


if __name__ == "__main__":
    main()
