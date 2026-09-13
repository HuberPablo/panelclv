"""Build a weekly customer-period panel from one of the CLVTools `.Rdata` datasets.

The gift and multichannel retailers ship as R workspaces (`Datasets/*.Rdata`) holding a
`mydata` transaction table — one row per transaction line, with `Id` and `Date` — plus a
`covariates.dynamic` calendar this builder does not need. R is the only reliable reader
of those files (the multichannel `Id` is a `bit64::integer64`, which a Python reader
turns into raw double bits), so a short `Rscript` call writes the columns the panel
needs to a temporary CSV and the rest happens here.

Output is the layout every panel in `Datasets/Dataset_clean/` uses::

    Id, year, week, Transactions

**What one transaction is** is a per-dataset fact, declared in `DATASETS`:

- gift: one row of `mydata` (no customer has two rows on one date);
- multichannel: one distinct `ORDER_NO` — `mydata` holds order *lines*, 5,012 of them
  for 2,939 orders, and counting lines would inflate every basket.

**Week numbering** is `period_calendar.week_of_year`, Valendin's `dayofyear // 7`
(ADR-0009), and **the grid keeps only complete weeks** inside the dataset's observation
window (`period_calendar.complete_week_grid`), for the same reason as the CDNOW builder:
a partial week reads as a drop in demand that is really a drop in observation.

Usage:
    python scripts/build_rdata_panel.py gift
    python scripts/build_rdata_panel.py multichannel
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from panelclv.data_preparation.period_calendar import complete_week_grid, week_of_year

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN = REPO_ROOT / "Datasets" / "Dataset_clean"


@dataclass(frozen=True)
class RdataDataset:
    rdata: Path
    # Column whose distinct values are counted per customer-week; None counts rows.
    order_col: str | None
    # The dataset's observation window. The start is the first day of the covariate
    # calendar CLVTools ships with the file; the end is the last day of data.
    data_start: pd.Timestamp
    data_end: pd.Timestamp


DATASETS = {
    # Covariate calendar opens 2001-02-25, which is exactly 2001 week 8 on this grid;
    # data run to the end of 2007.
    "gift": RdataDataset(
        rdata=REPO_ROOT / "Datasets" / "Gift Retailer data (weekly).Rdata",
        order_col=None,
        data_start=pd.Timestamp("2001-02-25"),
        data_end=pd.Timestamp("2007-12-31"),
    ),
    # First transaction 2005-01-01, last 2012-09-15. That last day opens 2012 week 37,
    # which the data do not finish, so the panel ends on week 36.
    "multichannel": RdataDataset(
        rdata=REPO_ROOT / "Datasets" / "Multichannel Retailer data.Rdata",
        order_col="ORDER_NO",
        data_start=pd.Timestamp("2005-01-01"),
        data_end=pd.Timestamp("2012-09-15"),
    ),
}

# Loads the workspace and writes `mydata`'s needed columns. Ids and order numbers go
# out as character so integer64 values keep their decimal digits.
_R_EXTRACT = r"""
suppressMessages(library(bit64))
args <- commandArgs(trailingOnly = TRUE)
env <- new.env()
load(args[1], envir = env)
tx <- as.data.frame(get("mydata", envir = env))
cols <- c("Id", "Date", if (nzchar(args[3])) args[3])
for (c in setdiff(cols, "Date")) tx[[c]] <- as.character(tx[[c]])
tx$Date <- format(tx$Date, "%Y-%m-%d")
write.csv(tx[, cols], args[2], row.names = FALSE)
"""


def read_transactions(ds: RdataDataset) -> pd.DataFrame:
    """`mydata` as an `(Id, Date[, order_col])` frame, read through Rscript."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "tx.csv"
        result = subprocess.run(
            ["Rscript", "-e", _R_EXTRACT, str(ds.rdata), str(out), ds.order_col or ""],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Rscript failed on {ds.rdata.name}:\n{result.stderr}")
        return pd.read_csv(out, dtype=str, parse_dates=["Date"])


def build_weekly_panel(tx: pd.DataFrame, ds: RdataDataset) -> pd.DataFrame:
    """Dense customer x complete-week panel of transaction counts."""
    tx = tx[(tx["Date"] >= ds.data_start) & (tx["Date"] <= ds.data_end)].copy()
    tx["year"] = tx["Date"].dt.year
    tx["week"] = week_of_year(tx["Date"])

    keys = ["Id", "year", "week"]
    grouped = tx.groupby(keys)
    counts = (grouped[ds.order_col].nunique() if ds.order_col else grouped.size())
    counts = counts.rename("Transactions").reset_index()

    # Every customer in every complete week, zeros included. Ids stay the decimal
    # strings R wrote, so an integer64 id is never rounded through a float.
    customers = pd.DataFrame({"Id": sorted(tx["Id"].unique())})
    panel = customers.merge(complete_week_grid(ds.data_start, ds.data_end), how="cross")
    panel = panel.merge(counts, on=keys, how="left")
    panel["Transactions"] = panel["Transactions"].fillna(0).astype("int64")
    return panel.sort_values(keys).reset_index(drop=True)


def describe(panel: pd.DataFrame) -> None:
    """Print what was built, including the count tail `clip_target_upper` has to cover."""
    n_cust = panel["Id"].nunique()
    periods = panel[["year", "week"]].drop_duplicates().sort_values(["year", "week"])
    first, last = periods.iloc[0], periods.iloc[-1]
    print(f"customers      : {n_cust}")
    print(f"weeks/customer : {len(periods)}")
    print(f"transactions   : {panel['Transactions'].sum()}")
    print(f"window         : {first['year']} w{first['week']} .. {last['year']} w{last['week']}")
    print("\nweekly transaction-count distribution (share of customer-weeks):")
    share = panel["Transactions"].value_counts(normalize=True).sort_index()
    for count, frac in share.items():
        print(f"  {count:>3} : {frac:12.8f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    args = parser.parse_args()

    ds = DATASETS[args.dataset]
    panel = build_weekly_panel(read_transactions(ds), ds)
    out = CLEAN / f"{args.dataset}_customer_week_panel.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False)
    print(f"wrote {out}\n")
    describe(panel)


if __name__ == "__main__":
    main()
