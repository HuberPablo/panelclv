"""Build weekly customer panels from the raw sources in `Datasets/Datasets_full/`.

`Datasets_full` holds the complete raw files behind the real panels (electronics, gift,
multichannel), plus four datasets the package had not used before (books, apparel,
video-on-demand, an online game). `docs/datasets.md` records, for each dataset, the
source, the rules below, the codes and the resulting counts.

For each dataset and each calibration it can hold, this writes two files to
`Datasets/Dataset_full_clean/`:

- `<name>_<2y|3y>_customer_week_panel.csv`, with columns
  `Id, year, week, Transactions, <static covariates>`;
- `<name>_<2y|3y>_customer_week_panel.config.json`, a `PanelConfig.to_dict()` holding
  the window dates, the static role and the embedding declarations. A study loads it
  with `PanelConfig.from_dict` rather than restating the dates.

**One transaction is one customer-day with at least one purchase row.** Line items and
same-day orders collapse to one, which is the trip level the source papers model.
`Transactions` for a week is the number of distinct purchase days in it. Each reader
decides what counts as a purchase row: returns and zero-price lines do not, and the
online game's system accounts are not customers.

**The cohort is an acquisition cohort.** It holds the customers whose first purchase in
the file falls in `[cohort_start, cohort_end]`. Anyone who bought before the window is
already a customer at the panel start, so they are excluded (left-censored).
`prepare_dataset` then finds nobody left to drop with its calibration-activity filter.

**Windows are whole years of Valendin week buckets** (ADR-0009). They start at the
panel's first week, and one year later means the same week index one calendar year on.

- `2y` trains on year 1, validates on year 2 and forecasts year 3.
- `3y` trains on years 1-2, validates on year 3 and forecasts year 4.

The panel runs exactly from the start of training to the end of the holdout. A
calibration the data cannot cover raises an error rather than being trimmed.

**Static covariates only.** These are demographics and facts fixed at the first
purchase: its channel, its category and its spend (`log1p`). Weekly spend, channel
mix, returns and contact or message counts are behaviour. A rollout cannot know them
for weeks it simulates (`docs/feature_engineering.md`), so they are left out.
Categorical codes start at 0, and code 0 means "missing" wherever a value can be
missing.

Usage:
    python scripts/build_full_panels.py all
    python scripts/build_full_panels.py electronics
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from panelclv.configs.panel_config import PanelConfig
from panelclv.data_preparation.period_calendar import (
    complete_week_grid,
    week_of_year,
    week_start,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "Datasets" / "Datasets_full"
OUT = REPO_ROOT / "Datasets" / "Dataset_full_clean"

# Calibration name -> number of calibration years (the last one is validation).
CAL_YEARS = {"2y": 2, "3y": 3}


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the generic builder needs to know about one dataset."""

    # First day of the panel. It must be the first day of a week bucket.
    panel_start: pd.Timestamp
    # The acquisition window: a customer belongs to the cohort iff their first
    # purchase in the file falls in [cohort_start, cohort_end].
    cohort_start: pd.Timestamp
    cohort_end: pd.Timestamp
    # Last day the raw data observe; no window may reach past it.
    data_end: pd.Timestamp
    # The calibrations this dataset is built for.
    calibrations: tuple[str, ...]
    # Static covariates: categorical ones with their fixed cardinality (codes
    # 0..card-1, embedded), and numeric ones (standardised by `prepare_dataset`).
    categorical: dict[str, int] = field(default_factory=dict)
    numeric: tuple[str, ...] = ()

    def __post_init__(self):
        start = pd.Series([self.panel_start])
        if week_start(start.dt.year, week_of_year(start)).iat[0] != self.panel_start:
            raise ValueError(f"panel_start {self.panel_start.date()} is not a week edge")
        if not self.panel_start <= self.cohort_start <= self.cohort_end:
            raise ValueError("need panel_start <= cohort_start <= cohort_end")


# ---------------------------------------------------------------------------
# Generic core: transactions -> cohort -> windows -> dense panel
# ---------------------------------------------------------------------------


def select_cohort(tx: pd.DataFrame, spec: DatasetSpec) -> pd.Index:
    """Ids whose first purchase falls inside the cohort window, sorted."""
    first = tx.groupby("Id")["Date"].min()
    inside = (first >= spec.cohort_start) & (first <= spec.cohort_end)
    return first.index[inside].sort_values()


def windows(spec: DatasetSpec, cal: str) -> dict[str, str]:
    """The five `PanelConfig` window dates for a calibration, as ISO strings.

    Each edge is the panel's first week index in a later calendar year, so every
    year-long window holds exactly 52 buckets.
    """
    years = CAL_YEARS[cal]
    start = pd.Series([spec.panel_start])
    y0, w0 = spec.panel_start.year, week_of_year(start).iat[0]

    def edge(k: int) -> pd.Timestamp:
        return week_start(pd.Series([y0 + k]), pd.Series([w0])).iat[0]

    day = pd.Timedelta(days=1)
    holdout_end = edge(years + 1) - day
    if holdout_end > spec.data_end:
        raise ValueError(
            f"a {cal} calibration needs data to {holdout_end.date()}, past the data "
            f"end {spec.data_end.date()}"
        )
    iso = lambda ts: str(ts.date())  # noqa: E731
    return {
        "training_start": iso(edge(0)),
        "validation_start": iso(edge(years - 1)),
        "training_end": iso(edge(years) - day),
        "holdout_start": iso(edge(years)),
        "holdout_end": iso(holdout_end),
    }


def build_panel(tx: pd.DataFrame, statics: pd.DataFrame, spec: DatasetSpec,
                cal: str) -> pd.DataFrame:
    """Dense cohort x complete-week panel of purchase-day counts plus static covariates.

    `tx` holds one row per purchase row, as `(Id, Date)`. `statics` holds one row per
    customer.
    """
    w = windows(spec, cal)
    start, end = pd.Timestamp(w["training_start"]), pd.Timestamp(w["holdout_end"])
    cohort = select_cohort(tx, spec)

    # One row per customer-day, then the number of distinct days per week.
    days = tx.loc[tx["Id"].isin(cohort), ["Id", "Date"]].copy()
    days["Date"] = days["Date"].dt.normalize()
    days = days.drop_duplicates()
    days = days[(days["Date"] >= start) & (days["Date"] <= end)]
    days["year"] = days["Date"].dt.year
    days["week"] = week_of_year(days["Date"])
    keys = ["Id", "year", "week"]
    counts = days.groupby(keys).size().rename("Transactions").reset_index()

    grid = complete_week_grid(start, end)
    panel = pd.DataFrame({"Id": cohort}).merge(grid, how="cross")
    panel = panel.merge(counts, on=keys, how="left")
    panel["Transactions"] = panel["Transactions"].fillna(0).astype("int64")

    # Static covariates: exactly one row per cohort member, repeated on every week.
    cols = [*spec.categorical, *spec.numeric]
    s = statics.set_index("Id")[cols]
    if s.index.duplicated().any():
        raise ValueError("statics must hold one row per customer")
    missing = cohort.difference(s.index)
    if len(missing):
        raise ValueError(f"{len(missing)} cohort members have no static row")
    panel = panel.merge(s, left_on="Id", right_index=True, how="left")
    for col in spec.categorical:
        panel[col] = panel[col].astype("int64")
    return panel.sort_values(keys).reset_index(drop=True)


def panel_config(spec: DatasetSpec, cal: str) -> PanelConfig:
    """The config a study reads the panel with.

    It declares the embedded count and week, the categorical statics embedded at
    their fixed cardinality, and the numeric statics.
    """
    return PanelConfig(
        id_col="Id",
        target_col="Transactions",
        frequency="weekly",
        time_cols=("year", "week"),
        time=("week",),
        static=(*spec.categorical, *spec.numeric),
        embedded_cols={"Transactions": "auto", "week": "auto", **spec.categorical},
        **windows(spec, cal),
    )


def output_paths(name: str, cal: str, out_dir: Path) -> tuple[Path, Path]:
    stem = f"{name}_{cal}_customer_week_panel"
    return out_dir / f"{stem}.csv", out_dir / f"{stem}.config.json"


def write_panel(name: str, spec: DatasetSpec, cal: str, out_dir: Path,
                tx: pd.DataFrame, statics: pd.DataFrame) -> tuple[Path, Path]:
    """Build one panel and write it with its config sidecar; returns both paths."""
    panel = build_panel(tx, statics, spec, cal)
    csv_path, json_path = output_paths(name, cal, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(panel_config(spec, cal).to_dict(), indent=2) + "\n")
    return csv_path, json_path


# ---------------------------------------------------------------------------
# Shared reader helpers
# ---------------------------------------------------------------------------


def _first_day(purchases: pd.DataFrame) -> pd.DataFrame:
    """The purchase rows on each customer's first purchase day, in input order."""
    first = purchases.groupby("Id")["Date"].transform("min")
    return purchases[purchases["Date"] == first]


def _first_spend(purchases: pd.DataFrame, amount: str) -> pd.Series:
    """`log1p` of the spend on each customer's first purchase day. Spend is heavily
    right-skewed, and the log keeps a few large baskets from dominating the scale."""
    return np.log1p(_first_day(purchases).groupby("Id")[amount].sum()).rename("first_spend")


def _code(values: pd.Series, mapping: dict) -> pd.Series:
    """Map raw categories to codes. Anything unmapped, including missing, becomes 0."""
    return values.map(mapping).fillna(0).astype("int64")


# ---------------------------------------------------------------------------
# Electronics: ISMS Durables Dataset 1 (Ni, Neslin & Sun 2012)
# ---------------------------------------------------------------------------

# Product purchase, service-contract purchase, discounted product purchase. Types 2 and
# 4 are returns and type 6 is miscellaneous.
ELECTRONICS_PURCHASE_TYPES = (1, 3, 5)
# Age of the household head (in two-year steps) grouped into decade bands. 0 = missing.
AGE_BANDS = [0, 35, 45, 55, 65, np.inf]


def load_electronics() -> pd.DataFrame:
    return pd.read_csv(
        RAW / "Eletronics Retailer - 6 years" / "durdata1_final.csv",
        usecols=["HOUSEHOLD_ID", "TRANSACTION_DATE", "TRANSACTION_TYPE", "EXTENDED_PRICE",
                 "INCOME", "GENDER_H_HEAD", "AGE_H_HEAD",
                 "CHILDERN_PRESENCE"],
    )


def electronics_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Purchase line items with a positive price, plus household demographics.

    The positive-price purchase rule is the one that reproduces the paper's cohort of
    3,782 households with first purchase from 1998-12-01 to 1999-11-30.

    The online flag is not used: one household in that cohort made its first purchase
    online, so as a covariate it would be constant.
    """
    d = raw.rename(columns={"HOUSEHOLD_ID": "Id"})
    d["Date"] = pd.to_datetime(d["TRANSACTION_DATE"], format="%d%b%Y:%H:%M:%S")
    p = d[d["TRANSACTION_TYPE"].isin(ELECTRONICS_PURCHASE_TYPES) & (d["EXTENDED_PRICE"] > 0)]

    # Demographics are constant within a household, so its first row stands for it.
    h = p.groupby("Id").first()
    age_band = pd.cut(h["AGE_H_HEAD"], AGE_BANDS, right=False, labels=False)
    statics = pd.DataFrame({
        "income": h["INCOME"].fillna(0).astype("int64"),          # 1..9, 0 = missing
        "gender": _code(h["GENDER_H_HEAD"], {"M": 1, "F": 2}),    # 0 = unknown ('U')
        "age_band": (age_band + 1).fillna(0).astype("int64"),    # 1..5, 0 = missing
        "children": _code(h["CHILDERN_PRESENCE"], {"N": 1, "Y": 2}),  # 0 = missing
        "first_spend": _first_spend(p, "EXTENDED_PRICE"),
    }).reset_index()
    return p[["Id", "Date"]], statics


# ---------------------------------------------------------------------------
# Gift: DMEF MultiChannel Gift Company dataset
# ---------------------------------------------------------------------------

# `AcqDate` (YYYYMM) is when the company first added the customer to its database. The
# transactions begin in 2001, so a customer acquired before the cohort month had
# history the file does not show.
GIFT_ACQUIRED_FROM = 200103
GIFT_METHODS = {"ST": 0, "I": 1, "P": 2, "M": 3}  # store, internet, phone, mail


def load_gift() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = RAW / "Gift retailer" / "original_data"
    orders = pd.read_csv(base / "DMEFExtractOrdersV01.CSV", dtype=str)
    lines = pd.read_csv(base / "DMEFExtractLinesV01.CSV", dtype=str,
                        usecols=["Cust_ID", "OrderNum", "LineDollars"])
    lines["LineDollars"] = pd.to_numeric(lines["LineDollars"])
    summary = pd.read_csv(base / "DMEFExtractSummaryV01.CSV", dtype=str,
                          usecols=["Cust_ID", "AcqDate", "AgeCode", "IncCode"])
    return orders, lines, summary


def gift_transactions(orders: pd.DataFrame, lines: pd.DataFrame,
                      summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Orders with positive spend, from customers acquired no earlier than the cohort.

    Spend is summed from the line file. Age and income are the vendor overlay codes
    (DMEF Demo Codes Reference), where a blank means the customer was not matched.
    """
    spend = lines.groupby(["Cust_ID", "OrderNum"])["LineDollars"].sum().rename("spend")
    o = orders.merge(spend, left_on=["Cust_ID", "OrderNum"], right_index=True)
    o = o.rename(columns={"Cust_ID": "Id"})
    o["Date"] = pd.to_datetime(o["OrderDate"], format="%Y%m%d")

    s = summary.rename(columns={"Cust_ID": "Id"}).set_index("Id")
    acquired = pd.to_numeric(s["AcqDate"], errors="coerce")
    new = acquired.index[acquired >= GIFT_ACQUIRED_FROM]
    p = o[(o["spend"] > 0) & o["Id"].isin(new)]
    p = p.sort_values(["Id", "Date", "OrderNum"])

    ids = p["Id"].unique()
    code = lambda col: pd.to_numeric(s.loc[ids, col].str.strip(), errors="coerce")  # noqa: E731
    statics = pd.DataFrame({
        "age_code": code("AgeCode").fillna(0).astype("int64"),     # 1..7, 0 = unmatched
        "income_code": code("IncCode").fillna(0).astype("int64"),  # 1..9, 0 = unmatched
        "first_method": _code(_first_day(p).groupby("Id")["OrderMethod"].first(), GIFT_METHODS),
        "first_spend": _first_spend(p, "spend"),
    })
    statics.index.name = "Id"
    return p[["Id", "Date"]], statics.reset_index()


# ---------------------------------------------------------------------------
# Multichannel: DMEF specialty multichannel catalog retailer
# ---------------------------------------------------------------------------

MULTICHANNEL_CHANNELS = {"ML": 0, "PH": 1, "WE": 2}  # mail, phone, web
MULTICHANNEL_DIVISIONS = {1: 0, 5: 1}  # the two division ids present in the file


def load_multichannel() -> pd.DataFrame:
    return pd.read_csv(
        RAW / "Specialty Multichannel Retailer - 11.5 years" / "special.csv",
        sep=";", dtype={"CUSTNO": str, "ORDER_NO": str},
        usecols=["CUSTNO", "ORDER_NO", "ORDER_LINE", "ORDER_DATE", "EXT_PRICE",
                 "CHANNEL", "DIVISION_ID"],
    )


def multichannel_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Order lines with a positive price.

    A cancelled order still counts, because the customer placed it; the cancellation
    is the retailer's (mostly a stock-out).
    """
    d = raw.rename(columns={"CUSTNO": "Id"})
    d["Date"] = pd.to_datetime(d["ORDER_DATE"])
    p = d[d["EXT_PRICE"] > 0].sort_values(["Id", "Date", "ORDER_NO", "ORDER_LINE"])
    first = _first_day(p).groupby("Id").first()
    statics = pd.DataFrame({
        "first_channel": _code(first["CHANNEL"], MULTICHANNEL_CHANNELS),
        "first_division": _code(first["DIVISION_ID"], MULTICHANNEL_DIVISIONS),
        "first_spend": _first_spend(p, "EXT_PRICE"),
    }).reset_index()
    return p[["Id", "Date"]], statics


# ---------------------------------------------------------------------------
# Books: German book retailer (Kaggle)
# ---------------------------------------------------------------------------

# The 30 category codes in `orders.csv`, in ascending order. There is no documentation
# naming them, and 99 appears to be a catch-all.
BOOK_CATEGORIES = (1, 3, 5, 6, 7, 8, 9, 10, 12, 14, 17, 19, 20, 21, 22, 23, 26, 27, 30,
                   31, 35, 36, 37, 38, 39, 40, 41, 44, 50, 99)


def load_books() -> pd.DataFrame:
    return pd.read_csv(RAW / "German Book retailer" / "Kaggle Files" / "orders.csv")


def books_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Order lines with a positive price. Zero-price lines are free items, and a day
    made only of them is not a sale.

    The first-order category is the category of the most expensive line on the first
    purchase day.
    """
    d = raw.rename(columns={"id": "Id"})
    d["Date"] = pd.to_datetime(d["orddate"], format="%d%b%Y")
    p = d[d["price"] > 0]
    top = _first_day(p).sort_values("price", ascending=False).groupby("Id").first()
    statics = pd.DataFrame({
        "first_category": _code(top["category"], {c: i for i, c in enumerate(BOOK_CATEGORIES)}),
        "first_spend": _first_spend(p, "price"),
    }).reset_index()
    return p[["Id", "Date"]], statics


# ---------------------------------------------------------------------------
# Apparel (Zitzlsperger), video-on-demand, online game: spend-only sources
# ---------------------------------------------------------------------------


def load_apparel() -> pd.DataFrame:
    return pd.read_csv(RAW / "Apparel (Zitzlsperger) - 10 years" / "Apparel.txt",
                       header=None, names=["Id", "date", "amount"])


def apparel_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows with positive spend. The file already holds one row per customer-day."""
    d = raw.assign(Date=pd.to_datetime(raw["date"], format="%d.%m.%Y"))
    p = d[d["amount"] > 0]
    return p[["Id", "Date"]], _first_spend(p, "amount").reset_index()


def load_vod() -> pd.DataFrame:
    return pd.read_csv(RAW / "Video-on-Demand - 5.5 years" / "vod.csv",
                       usecols=["Id", "Date", "Price"], parse_dates=["Date"])


def vod_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every rental, free ones included. On a streaming service a zero-price rental is
    consumption like any other."""
    p = raw[raw["Price"] >= 0]
    return p[["Id", "Date"]], _first_spend(p, "Price").reset_index()


# Receivers that are platform accounts rather than players: 16453 activates campaigns
# and accounts for 30,971 receipts; 509053 receives 4,001. Both are dropped in the
# source's `load.R`.
GAME_SYSTEM_ACCOUNTS = (16453, 509053)


def load_game() -> pd.DataFrame:
    return pd.read_csv(RAW / "Online Game - 6 years" / "transactions.csv", sep=";",
                       header=None, names=["datetime", "sender", "receiver", "amount"])


def game_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Virtual-currency receipts. The receiver is the customer, as in `load.R`."""
    d = raw[~raw["receiver"].isin(GAME_SYSTEM_ACCOUNTS)
            & (raw["sender"] != raw["receiver"]) & (raw["amount"] > 0)]
    d = d.assign(Id=d["receiver"], Date=pd.to_datetime(d["datetime"]).dt.normalize())
    return d[["Id", "Date"]], _first_spend(d, "amount").reset_index()


# ---------------------------------------------------------------------------
# The dataset table
# ---------------------------------------------------------------------------

READERS = {
    "electronics": lambda: electronics_transactions(load_electronics()),
    "gift": lambda: gift_transactions(*load_gift()),
    "multichannel": lambda: multichannel_transactions(load_multichannel()),
    "books": lambda: books_transactions(load_books()),
    "apparel": lambda: apparel_transactions(load_apparel()),
    "vod": lambda: vod_transactions(load_vod()),
    "game": lambda: game_transactions(load_game()),
}

_ts = pd.Timestamp
SPECS: dict[str, DatasetSpec] = {
    # The paper's cohort is first purchase 1998-12-01..1999-11-30. The data open on
    # Dec 1, midway through 1998 week 47, so the panel starts at week 48 (Dec 2).
    # Households whose first purchase was on Dec 1 fall outside it.
    "electronics": DatasetSpec(
        panel_start=_ts("1998-12-02"), cohort_start=_ts("1998-12-02"),
        cohort_end=_ts("1999-11-30"), data_end=_ts("2004-11-30"),
        calibrations=("2y", "3y"),
        categorical={"income": 10, "gender": 3, "age_band": 6, "children": 3},
        numeric=("first_spend",),
    ),
    # The old panel's cohort (first order in Mar-May 2001) and first week (2001 week 8),
    # with the acquisition-date check applied by the reader.
    "gift": DatasetSpec(
        panel_start=_ts("2001-02-25"), cohort_start=_ts("2001-03-01"),
        cohort_end=_ts("2001-05-31"), data_end=_ts("2007-12-30"),
        calibrations=("2y", "3y"),
        categorical={"age_code": 8, "income_code": 10, "first_method": 4},
        numeric=("first_spend",),
    ),
    "multichannel": DatasetSpec(
        panel_start=_ts("2005-01-01"), cohort_start=_ts("2005-01-01"),
        cohort_end=_ts("2005-03-31"), data_end=_ts("2012-09-17"),
        calibrations=("2y", "3y"),
        categorical={"first_channel": 3, "first_division": 2},
        numeric=("first_spend",),
    ),
    "books": DatasetSpec(
        panel_start=_ts("2008-01-01"), cohort_start=_ts("2008-01-01"),
        cohort_end=_ts("2008-03-31"), data_end=_ts("2014-11-24"),
        calibrations=("2y", "3y"),
        categorical={"first_category": len(BOOK_CATEGORIES)},
        numeric=("first_spend",),
    ),
    # The cohort in the source's `clvtools.R` is everyone who first bought before 1997.
    # The data open on 1996-08-07, inside week 31, so the panel starts at week 32
    # (Aug 11).
    "apparel": DatasetSpec(
        panel_start=_ts("1996-08-11"), cohort_start=_ts("1996-08-11"),
        cohort_end=_ts("1996-12-31"), data_end=_ts("2006-11-23"),
        calibrations=("2y", "3y"), numeric=("first_spend",),
    ),
    # `vod.R` sets aside the cohorts before 2011 as behaving differently. The data end
    # in January 2014, so only a 2y calibration fits.
    "vod": DatasetSpec(
        panel_start=_ts("2011-01-01"), cohort_start=_ts("2011-01-01"),
        cohort_end=_ts("2011-03-31"), data_end=_ts("2014-01-31"),
        calibrations=("2y",), numeric=("first_spend",),
    ),
    "game": DatasetSpec(
        panel_start=_ts("2008-01-01"), cohort_start=_ts("2008-01-01"),
        cohort_end=_ts("2008-03-31"), data_end=_ts("2012-10-04"),
        calibrations=("2y", "3y"), numeric=("first_spend",),
    ),
}


def read(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A dataset's `(Id, Date)` purchase rows and its one-row-per-customer statics."""
    return READERS[name]()


def describe(name: str, cal: str, panel: pd.DataFrame, spec: DatasetSpec) -> None:
    """Print what was built, including the weekly count distribution the head must cover."""
    w = windows(spec, cal)
    share = panel["Transactions"].value_counts().sort_index()
    print(f"{name} {cal}: {panel['Id'].nunique()} customers, "
          f"{panel.groupby('Id').size().iat[0]} weeks, "
          f"{panel['Transactions'].sum()} transactions, "
          f"{w['training_start']}..{w['holdout_end']}")
    print("  weekly counts: " + ", ".join(f"{k}:{v}" for k, v in share.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=[*SPECS, "all"])
    args = parser.parse_args()

    for name in SPECS if args.dataset == "all" else [args.dataset]:
        spec = SPECS[name]
        tx, statics = read(name)
        for cal in spec.calibrations:
            csv_path, _ = write_panel(name, spec, cal, OUT, tx, statics)
            describe(name, cal, pd.read_csv(csv_path), spec)


if __name__ == "__main__":
    main()
