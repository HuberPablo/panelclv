"""Group the raw bank transactions (`Datasets/trans.csv`) into customer-period panels.

`trans.csv` is the Czech PKDD'99 bank dataset at **transaction granularity** — one row
per transaction (`trans_id` is unique). PanelConfig / `prepare_dataset` instead expect a
**panel**: one row per customer per time period carrying the per-period transaction count
(the model target). This module performs that grouping, producing files in the exact same
layout as the other clean panels (e.g. `electronicV2_customer_week_panel.csv`):

    Id, year, [week|month], Transactions

It mirrors the conventions of `Dataset_builiding_notebook/dataset_building.ipynb`
(`week = dayofyear // 7` clipped to 51 → 52 buckets/year; a full rectangular
`Id × year × period` grid; periods with no transaction → 0) but is adapted to this source:

* the dates are integers in `YYMMDD` form (e.g. `930101` → 1993-01-01), not datetimes;
* there is **no covariate calendar**, so the customer cohort and the year span are taken
  from the transactions themselves rather than from a covariates file;
* a "transaction" is one row (`trans_id`). This counts *every* account transaction
  (credits and debits alike). To forecast only a subset (say credits, `type == "PRIJEM"`),
  filter `tx` before calling `build_panel`.

Standard library + pandas only. Run directly to (re)build all three frequencies:

    /home/virthian/Desktop/Thesis/venvs/thesis_rocm/bin/python Data_preparation/build_bank_panel.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --- Paths: set these explicitly. ---
SOURCE_CSV = Path("Datasets/trans.csv")              # raw transaction-level file
OUT_DIR = Path("Datasets/Dataset_clean")             # where the panels are written

# Output basename; files become e.g. bank_customer_week_panel.csv (matches the
# "<name>_customer_<freq>_panel.csv" convention used by every other dataset).
#
# This builder intentionally produces the FULL dense panel (every account that ever
# transacts, across the full year span). Cohort selection — keeping only customers
# active during calibration, per Valendin et al. — is handled downstream by
# prepare_dataset (PanelConfig.require_calibration_activity), which knows the
# training_end and applies the same filter to every dataset and to both models.
DATASET_NAME = "bank"

# Source column names in trans.csv.
ID_COL = "account_id"     # the customer id
DATE_COL = "date"         # integer YYMMDD

# Per frequency: the sub-year column name and the full range of its buckets.
# year-level has no sub-column (one bucket per year). Identical to the notebook.
_PERIOD = {
    "week":  ("week",  range(0, 52)),   # weeks 0..51 -> 52 buckets/year
    "month": ("month", range(1, 13)),   # months 1..12
    "year":  (None,    None),           # year only, no sub-column
}

FREQS = ("week", "month", "year")


def load_transactions(csv_path: Path = SOURCE_CSV) -> pd.DataFrame:
    """Read trans.csv, keeping only id + date, and parse the YYMMDD integer dates.

    Returns a frame with columns `Id` (the customer id) and `Date` (datetime), one row
    per transaction — the minimal shape `build_panel` needs.
    """
    # Only the two columns we group on are needed; reading just these keeps it light
    # and sidesteps the mixed-type `bank` column warning.
    tx = pd.read_csv(csv_path, usecols=[ID_COL, DATE_COL])

    # Dates are integers like 930101. Zero-pad to 6 chars (guards any 5-digit value)
    # and parse as %y%m%d: pandas maps 2-digit years 69-99 -> 1900s, so 93 -> 1993.
    tx["Date"] = pd.to_datetime(tx[DATE_COL].astype(str).str.zfill(6), format="%y%m%d")
    tx = tx.rename(columns={ID_COL: "Id"})
    tx["Id"] = tx["Id"].astype(str)   # treat the id as a categorical label, not a number
    return tx[["Id", "Date"]]


def _add_period_cols(df: pd.DataFrame, date_col: str, freq: str) -> None:
    """Add `year` and (for week/month) the sub-year period column from a datetime column."""
    df["year"] = df[date_col].dt.year
    if freq == "week":
        # dayofyear // 7 clipped to 51 -> exactly 52 buckets/year (electronics convention)
        df["week"] = (df[date_col].dt.dayofyear // 7).clip(upper=51)
    elif freq == "month":
        df["month"] = df[date_col].dt.month


def period_key_cols(freq: str) -> list[str]:
    """Grouping/join keys for a frequency, e.g. ['Id','year','week'] or ['Id','year']."""
    sub_col, _ = _PERIOD[freq]
    return ["Id", "year"] + ([sub_col] if sub_col else [])


def build_panel(tx: pd.DataFrame, freq: str = "week") -> pd.DataFrame:
    """Build a dense per-customer panel of transaction counts at the given frequency.

    Parameters
    ----------
    tx : transactions frame with columns `Id`, `Date` (from `load_transactions`).
    freq : "week", "month", or "year".

    Returns columns: Id, year[, week|month], Transactions — a complete rectangular grid
    (every customer × every year in the span × every sub-period), with periods that had
    no transaction filled with 0.
    """
    sub_col, sub_range = _PERIOD[freq]
    keys = period_key_cols(freq)

    tx = tx.copy()
    _add_period_cols(tx, "Date", freq)

    # One count per (customer, period). One row in trans.csv == one transaction.
    counts = tx.groupby(keys).size().reset_index(name="Transactions")

    # Full rectangular grid. With no covariate calendar, the cohort (customer set) and
    # the year span come from the transactions themselves: every account that ever
    # transacts, across the full observed year range.
    ids = tx["Id"].unique()
    year_lo = int(tx["year"].min())
    year_hi = int(tx["year"].max())
    levels: list = [ids, range(year_lo, year_hi + 1)]
    if sub_col:
        levels.append(list(sub_range))
    grid = pd.MultiIndex.from_product(levels, names=keys).to_frame(index=False)

    # Left-join counts onto the grid; unobserved periods become 0 transactions.
    panel = grid.merge(counts, on=keys, how="left")
    panel["Transactions"] = panel["Transactions"].fillna(0).astype(int)

    return panel.sort_values(keys).reset_index(drop=True)


def verify_panel(panel: pd.DataFrame, freq: str) -> None:
    """Sanity-check a built panel (same checks as the dataset-building notebook)."""
    sub_col, sub_range = _PERIOD[freq]
    buckets = len(sub_range) if sub_range else 1
    n_cust = panel["Id"].nunique()
    n_years = panel["year"].nunique()
    expected = n_cust * n_years * buckets

    assert panel.isna().sum().sum() == 0, f"{freq}: unexpected NaNs"
    assert (panel["Transactions"] >= 0).all(), f"{freq}: negative Transactions"
    assert panel["Transactions"].dtype.kind in "iu", f"{freq}: Transactions not integer"
    if sub_col:
        lo, hi = min(sub_range), max(sub_range)
        assert panel[sub_col].between(lo, hi).all(), f"{freq}: {sub_col} out of range"
    assert len(panel) == expected, f"{freq}: {len(panel)} rows != expected {expected}"

    print(f"{DATASET_NAME:<6} {freq:<5} OK | rows={len(panel):>8} | customers={n_cust} | "
          f"years={panel['year'].min()}-{panel['year'].max()} | "
          f"tx max={panel['Transactions'].max():>3} | cols={list(panel.columns)}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tx = load_transactions()
    print(f"loaded {len(tx):,} transactions for {tx['Id'].nunique()} customers from {SOURCE_CSV.name}\n")
    for freq in FREQS:
        panel = build_panel(tx, freq)
        out = OUT_DIR / f"{DATASET_NAME}_customer_{freq}_panel.csv"
        panel.to_csv(out, index=False)
        verify_panel(panel, freq)
        print(f"  wrote {out}\n")


if __name__ == "__main__":
    main()
