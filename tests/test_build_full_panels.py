"""The full-dataset panel builder turns raw transactions into panels `prepare_dataset` reads.

`scripts/build_full_panels.py` reads the raw sources in `Datasets/Datasets_full/`. For
each (dataset, calibration) pair it writes a dense customer-week panel and a
`PanelConfig` sidecar next to it. The rules pinned here are the ones that fail silently
when they break:

- **A transaction is a customer-day.** Line items and same-day orders collapse to one.
  A weekly count that counts lines inflates every basket, which is what happened to the
  old electronics panel.
- **Non-purchase rows are not counted.** Returns, service-contract returns, zero-price
  lines and the online game's system accounts are all excluded.
- **The cohort is an acquisition cohort.** A customer counts only if their *first ever*
  purchase is in the cohort window. A customer who bought earlier is left-censored and
  is excluded even when they also buy inside the window.
- **The grid is dense and uses the package calendar.** It holds complete weeks only,
  numbered by Valendin's rule (ADR-0009), and runs exactly from the panel start to the
  end of the holdout.
- **The windows are whole years of week buckets.** A calibration the data cannot hold
  raises instead of being trimmed.

Everything here runs on synthetic raw frames, so no dataset is needed.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panelclv.configs.panel_config import PanelConfig
from panelclv.data_preparation.panel_dataset import prepare_dataset
from panelclv.data_preparation.period_calendar import week_of_year, week_start

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_full_panels.py"


def _load():
    """Import the script by path, because `scripts/` is outside the wheel.

    The module is registered in `sys.modules` before it runs. `@dataclass` looks up
    its class's module there, so a module that defines a dataclass cannot be run
    unregistered.
    """
    spec = importlib.util.spec_from_file_location("build_full_panels", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bfp = _load()


def _tx(rows):
    """`(Id, Date)` purchase frame from `(id, "YYYY-MM-DD")` pairs."""
    return pd.DataFrame(rows, columns=["Id", "Date"]).assign(
        Date=lambda d: pd.to_datetime(d["Date"])
    )


# A small dataset declaration on the 2003 calendar. 2003 week 0 begins on 2003-01-01,
# the cohort buys in January, and the data run to the end of 2007, so both
# calibrations fit.
SPEC = bfp.DatasetSpec(
    panel_start=pd.Timestamp("2003-01-01"),
    cohort_start=pd.Timestamp("2003-01-01"),
    cohort_end=pd.Timestamp("2003-01-31"),
    data_end=pd.Timestamp("2007-12-31"),
    calibrations=("2y", "3y"),
    categorical={"segment": 3},
    numeric=("first_spend",),
)


def _statics(ids):
    """One static row per customer: a categorical code and a numeric value."""
    ids = list(ids)
    return pd.DataFrame({
        "Id": ids,
        "segment": [i % 3 for i in range(len(ids))],
        "first_spend": np.log1p(np.arange(len(ids), dtype=float) + 10.0),
    })


# ---------------------------------------------------------------------------
# 1. A transaction is a customer-day
# ---------------------------------------------------------------------------


def test_same_day_rows_collapse_and_distinct_days_add():
    """Three lines on one day count as 1. A second day in the same week makes it 2."""
    tx = _tx([
        ("A", "2003-01-08"), ("A", "2003-01-08"), ("A", "2003-01-08"),
        ("A", "2003-01-10"),
        ("B", "2003-01-02"),
    ])
    panel = bfp.build_panel(tx, _statics(["A", "B"]), SPEC, "2y")
    a = panel[panel["Id"] == "A"].set_index(["year", "week"])["Transactions"]
    assert a.loc[(2003, week_of_year(pd.Series([pd.Timestamp("2003-01-08")])).iat[0])] == 2
    assert a.sum() == 2
    assert panel[panel["Id"] == "B"]["Transactions"].sum() == 1


# ---------------------------------------------------------------------------
# 2. Non-purchase rows are not counted
# ---------------------------------------------------------------------------


def _electronics_raw(rows):
    """Minimal raw electronics frame with the columns the transform reads."""
    cols = ["HOUSEHOLD_ID", "TRANSACTION_DATE", "TRANSACTION_TYPE", "EXTENDED_PRICE",
            "INCOME", "GENDER_H_HEAD", "AGE_H_HEAD",
            "CHILDERN_PRESENCE"]
    return pd.DataFrame(rows, columns=cols)


def test_electronics_counts_purchases_only():
    """Types 2, 4 and 6 and rows with price <= 0 are not purchase occasions."""
    raw = _electronics_raw([
        (1, "02DEC1998:00:00:00", 1, 100.0, 6, "M", 44, "Y"),   # purchase
        (1, "03DEC1998:00:00:00", 2, -100.0, 6, "M", 44, "Y"),  # product return
        (1, "04DEC1998:00:00:00", 4, -20.0, 6, "M", 44, "Y"),   # contract return
        (1, "05DEC1998:00:00:00", 6, 5.0, 6, "M", 44, "Y"),     # miscellaneous
        (1, "06DEC1998:00:00:00", 1, 0.0, 6, "M", 44, "Y"),     # zero-price line
        (1, "07DEC1998:00:00:00", 3, 30.0, 6, "M", 44, "Y"),    # service contract
        (1, "08DEC1998:00:00:00", 5, 50.0, 6, "M", 44, "Y"),    # discounted purchase
        (2, "09DEC1998:00:00:00", 2, -10.0, None, "U", None, None),  # only a return
    ])
    tx, statics = bfp.electronics_transactions(raw)
    assert sorted(tx["Date"].dt.day) == [2, 7, 8]
    assert set(tx["Id"]) == {1}
    assert set(statics["Id"]) == {1}


def test_game_drops_system_accounts_self_transfers_and_nonpositive():
    """The game's two system accounts, self-transfers and amounts <= 0 are removed."""
    raw = pd.DataFrame(
        [
            ("2008-01-02 10:00:00", 5, 7, 10),     # a real receipt for customer 7
            ("2008-01-02 11:00:00", 5, 16453, 10),  # system account
            ("2008-01-02 12:00:00", 5, 509053, 10),  # system account
            ("2008-01-03 12:00:00", 7, 7, 10),      # self-transfer
            ("2008-01-04 12:00:00", 5, 7, 0),       # nothing received
        ],
        columns=["datetime", "sender", "receiver", "amount"],
    )
    tx, statics = bfp.game_transactions(raw)
    assert list(tx["Id"]) == [7]
    assert list(tx["Date"]) == [pd.Timestamp("2008-01-02")]
    assert list(statics["Id"]) == [7]


# ---------------------------------------------------------------------------
# 3. The cohort is an acquisition cohort
# ---------------------------------------------------------------------------


def test_cohort_keeps_only_first_purchases_inside_the_window():
    tx = _tx([
        ("early", "2002-12-20"), ("early", "2003-01-10"),  # left-censored
        ("in", "2003-01-15"), ("in", "2004-06-01"),
        ("edge", "2003-01-31"),
        ("late", "2003-02-01"),
    ])
    assert sorted(bfp.select_cohort(tx, SPEC)) == ["edge", "in"]


def test_gift_excludes_accounts_acquired_before_the_cohort():
    """A gift customer whose account predates March 2001 is not new, even if their
    first retained order falls in the cohort window."""
    orders = pd.DataFrame({
        "Cust_ID": ["1", "2"],
        "OrderNum": ["10", "20"],
        "OrderDate": ["20010315", "20010320"],
        "OrderMethod": ["ST", "I"],
    })
    lines = pd.DataFrame({
        "Cust_ID": ["1", "2"], "OrderNum": ["10", "20"], "LineDollars": [25.0, 40.0],
    })
    summary = pd.DataFrame({
        "Cust_ID": ["1", "2"], "AcqDate": ["200103", "199905"],
        "AgeCode": ["4", " "], "IncCode": [" ", "9"],
    })
    tx, statics = bfp.gift_transactions(orders, lines, summary)
    assert set(tx["Id"]) == {"1"}
    assert set(statics["Id"]) == {"1"}


# ---------------------------------------------------------------------------
# 4. The grid is dense and uses the package calendar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cal", ["2y", "3y"])
def test_panel_is_a_dense_complete_week_rectangle(cal):
    tx = _tx([("A", "2003-01-08"), ("B", "2003-01-20"), ("B", "2004-03-03")])
    panel = bfp.build_panel(tx, _statics(["A", "B"]), SPEC, cal)
    w = bfp.windows(SPEC, cal)

    assert list(panel.columns[:4]) == ["Id", "year", "week", "Transactions"]
    assert not panel.duplicated(["Id", "year", "week"]).any()
    assert not panel.isna().any().any()
    assert panel["Transactions"].dtype == np.int64
    sizes = panel.groupby("Id").size()
    assert sizes.nunique() == 1

    starts = week_start(panel["year"], panel["week"])
    assert starts.min() == pd.Timestamp(w["training_start"])
    # The last week ends exactly on the holdout end: the following bucket begins the
    # next day.
    last = panel.sort_values(["year", "week"]).iloc[-1]
    assert week_of_year(pd.Series([pd.Timestamp(w["holdout_end"])])).iat[0] == last["week"]
    assert pd.Timestamp(w["holdout_end"]).year == last["year"]
    # Every row is a week the calendar produces, and the weeks are consecutive.
    n_weeks = {"2y": 3 * 52, "3y": 4 * 52}[cal]
    assert sizes.iat[0] == n_weeks


def test_panel_excludes_customers_outside_the_cohort():
    tx = _tx([("in", "2003-01-15"), ("late", "2003-03-01")])
    panel = bfp.build_panel(tx, _statics(["in", "late"]), SPEC, "2y")
    assert set(panel["Id"]) == {"in"}


# ---------------------------------------------------------------------------
# 5. The windows are whole years of week buckets
# ---------------------------------------------------------------------------


def _n_weeks(start, end):
    """Complete weeks between two window dates on the package calendar."""
    from panelclv.data_preparation.period_calendar import complete_week_grid
    return len(complete_week_grid(pd.Timestamp(start), pd.Timestamp(end)))


@pytest.mark.parametrize("cal, train_weeks", [("2y", 52), ("3y", 104)])
def test_windows_are_year_long_week_blocks(cal, train_weeks):
    w = bfp.windows(SPEC, cal)
    day = pd.Timedelta(days=1)
    val_end = pd.Timestamp(w["training_end"])
    assert _n_weeks(w["training_start"], pd.Timestamp(w["validation_start"]) - day) == train_weeks
    assert _n_weeks(w["validation_start"], val_end) == 52
    assert _n_weeks(w["holdout_start"], w["holdout_end"]) == 52
    assert pd.Timestamp(w["holdout_start"]) == val_end + day
    # PanelConfig accepts them.
    PanelConfig(id_col="Id", target_col="Transactions", frequency="weekly",
                time_cols=("year", "week"), **w)


def test_windows_mid_year_start_keep_the_same_week_index():
    """A start mid-year (Valendin week 48) steps one calendar year per window."""
    spec = bfp.DatasetSpec(
        panel_start=week_start(pd.Series([1998]), pd.Series([48])).iat[0],
        cohort_start=pd.Timestamp("1998-12-02"),
        cohort_end=pd.Timestamp("1999-11-30"),
        data_end=pd.Timestamp("2004-11-30"),
        calibrations=("2y", "3y"),
    )
    w = bfp.windows(spec, "3y")
    for key, year in [("validation_start", 2000), ("holdout_start", 2001)]:
        ts = pd.Timestamp(w[key])
        assert (ts.year, week_of_year(pd.Series([ts])).iat[0]) == (year, 48)


def test_calibration_longer_than_the_data_raises():
    spec = bfp.DatasetSpec(
        panel_start=pd.Timestamp("2011-01-01"),
        cohort_start=pd.Timestamp("2011-01-01"),
        cohort_end=pd.Timestamp("2011-03-31"),
        data_end=pd.Timestamp("2014-01-31"),
        calibrations=("2y",),
    )
    bfp.windows(spec, "2y")  # fits
    with pytest.raises(ValueError, match="data end"):
        bfp.windows(spec, "3y")


# ---------------------------------------------------------------------------
# 6. Static covariates
# ---------------------------------------------------------------------------


def test_statics_are_constant_per_customer_and_codes_start_at_zero():
    tx = _tx([("A", "2003-01-08"), ("A", "2003-05-08"), ("B", "2003-01-20")])
    panel = bfp.build_panel(tx, _statics(["A", "B"]), SPEC, "2y")
    assert (panel.groupby("Id")[["segment", "first_spend"]].nunique() == 1).all().all()
    assert panel["segment"].dtype == np.int64
    assert panel["segment"].min() >= 0


def test_electronics_statics_code_missing_values_separately():
    """Missing income, gender 'U' and missing children presence each get their own
    code. That code is 0, so the known values keep their order above it."""
    raw = _electronics_raw([
        (1, "02DEC1998:00:00:00", 1, 100.0, 6, "M", 44, "Y"),
        (2, "02DEC1998:00:00:00", 1, 100.0, None, "U", None, None),
        (3, "02DEC1998:00:00:00", 1, 100.0, 1, "F", 70, "N"),
    ])
    _, statics = bfp.electronics_transactions(raw)
    s = statics.set_index("Id")
    assert s.loc[2, "income"] == 0 and s.loc[1, "income"] == 6 and s.loc[3, "income"] == 1
    assert s.loc[2, "gender"] == 0 and s.loc[1, "gender"] != s.loc[3, "gender"]
    assert s.loc[2, "children"] == 0 and s.loc[1, "children"] != s.loc[3, "children"]
    assert s.loc[2, "age_band"] == 0 and s.loc[3, "age_band"] > s.loc[1, "age_band"] > 0
    assert s.loc[1, "first_spend"] == pytest.approx(np.log1p(100.0))
    for col, card in bfp.SPECS["electronics"].categorical.items():
        assert statics[col].between(0, card - 1).all()


# ---------------------------------------------------------------------------
# 7. End to end: sidecar config -> prepare_dataset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cal, t_cal", [("2y", 104), ("3y", 156)])
def test_written_panel_and_config_feed_prepare_dataset(tmp_path, cal, t_cal):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(12):
        cid = f"C{i:02d}"
        rows.append((cid, f"2003-01-{1 + i:02d}"))  # every first purchase in January
        for d in rng.integers(0, 4 * 365, size=6):
            rows.append((cid, str((pd.Timestamp("2003-02-01") + pd.Timedelta(days=int(d))).date())))
    tx = _tx(rows)
    ids = sorted(tx["Id"].unique())

    csv_path, json_path = bfp.write_panel("toy", SPEC, cal, tmp_path, tx, _statics(ids))
    panel = pd.read_csv(csv_path)
    config = PanelConfig.from_dict(json.loads(json_path.read_text()))

    assert "segment" in config.static and "first_spend" in config.static
    out = prepare_dataset(panel, config, verbose=False)
    # Every cohort member bought in the first calibration month, so none is dropped.
    assert (out["N"], out["T_CAL"], out["T_HOLD"]) == (len(ids), t_cal, 52)
    assert out["embedded_cols"]["segment"] == 3
