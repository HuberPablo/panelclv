"""Descriptive statistics of the full panels, in the form of Valendin et al.'s Table 3.

`docs/datasets.md` ("Descriptive statistics") is where this table lives; this script is
where it comes from. It reuses `build_full_panels.py`'s readers, cohort rule and
windows, so the numbers describe exactly the panels a study reads. It writes nothing
to the panels.

**Rows.** One per dataset and calibration the raw data can cover (`2y` to `5y`; a
calibration that needs data past the dataset's end is skipped), plus Valendin's own
multichannel split (52 calibration weeks, 338 holdout weeks) as a reproduction check.
The electronics `5y` row is the other one: it is the paper's split.

**Columns.** Valendin et al. (2022), Table 3, all computed on the calibration window
(training + validation) unless they name the holdout:

- `cal_mean`, `non_repeaters_%`: transactions per customer, and the share with exactly
  one (the acquisition purchase).
- `hold_mean`, `inactive_%`: holdout transactions per customer, and the share with none.
- `seasonality`: Valendin's eq. 4. The mean absolute deviation of the weekly aggregate
  of *repeat* transactions (each customer's first excluded) from its median, over that
  median. Below 0.5 is low, 0.5-1 mild, above 1 strong. It cannot tell a trend from a
  season: a declining base scores as seasonal.
- `rWM`: Wheat & Morrison (1990) timing regularity, exactly as BTYDplus
  `estimateRegularity(method = "wheat")` computes it. Customers with at least 3
  purchase days; for each, one of their last two inter-purchase times (in days), drawn
  at random, over the sum of both; `r = (1 - 4 var) / (8 var)`. About 1 is random
  (Poisson) timing, above 1 regular, below 1 irregular.
- `clumpy_%`: share of the cohort classified clumpy by Zhang, Bradlow & Small (2015).
  A customer's `H_p = 1 + sum(x log x) / log(n + 1)` over the n + 1 gaps between the
  window start, their n purchase days and the window end, normalised to sum to 1. They
  are clumpy when `H_p` exceeds the 95% quantile of 1,000 draws of n distinct uniform
  days in the same window. Customers with fewer than 2 purchase days count as not
  clumpy. Valendin does not give his implementation, and this one reproduces his
  electronics value but not his multichannel one (`docs/datasets.md`).

Columns added for this project's models:

- `hold_zero_cells_%`: share of holdout customer-weeks with no transaction.
- `max_week_count`: largest weekly count over calibration and holdout, which sets the
  size of the softmax head.
- `hold/last_cal_year`: holdout transactions over those of the last calibration year.
  Below 1, the base is shrinking.

**Randomness.** `rWM` and the clumpiness critical values draw random numbers. Both come
from a generator seeded with `SEED` afresh for each dataset, so a rerun, or a run on a
subset, reproduces the table exactly.

Run from the repo root (the full run takes a few minutes; apparel and VoD dominate):

    PYTHONPATH=src python scripts/describe_full_panels.py            # every dataset
    PYTHONPATH=src python scripts/describe_full_panels.py books game # a subset

It prints the table as markdown; `--out <file>` also writes it as CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_full_panels as bfp  # noqa: E402

SEED = 0
CALIBRATIONS = ("2y", "3y", "4y", "5y")
# Valendin et al.'s multichannel split: calibrate on 2005, forecast the next 338 weeks.
VALENDIN_MULTICHANNEL = ("2005-01-01", "2005-12-31", "2006-01-01", 338)


def wheat_morrison_r(days: pd.DataFrame, rng: np.random.Generator) -> float:
    """BTYDplus `estimateRegularity(method = "wheat")` on `(Id, Date)` purchase days."""
    M = []
    for _, dates in days.groupby("Id")["Date"]:
        if len(dates) < 3:
            continue
        # The customer's last two inter-purchase times, in days.
        itt = np.diff(np.sort(dates.values).astype("datetime64[D]").astype(float))[-2:]
        M.append(itt[rng.integers(2)] / itt.sum())
    v = np.var(M, ddof=1)
    return (1 - 4 * v) / (8 * v)


def _entropy_clumpiness(pos: np.ndarray, n_days: int) -> float:
    """`H_p` for purchases on day positions 1..n_days of the window."""
    gaps = np.diff(np.concatenate(([0], np.sort(pos), [n_days + 1]))) / (n_days + 1)
    gaps = gaps[gaps > 0]
    return 1 + (gaps * np.log(gaps)).sum() / np.log(len(pos) + 1)


def clumpy_share(days: pd.DataFrame, n_customers: int, start: pd.Timestamp,
                 end: pd.Timestamp, rng: np.random.Generator) -> float:
    """Share of `n_customers` whose purchase days in [start, end] are clumpy."""
    n_days = (end - start).days + 1
    critical: dict[int, float] = {}  # n purchase days -> 95% quantile under the null
    clumpy = 0
    for _, dates in days.groupby("Id")["Date"]:
        n = len(dates)
        if n < 2:
            continue
        if n not in critical:
            null = [_entropy_clumpiness(rng.choice(n_days, n, replace=False) + 1, n_days)
                    for _ in range(1000)]
            critical[n] = float(np.quantile(null, 0.95))
        pos = (dates - start).dt.days.values + 1
        clumpy += _entropy_clumpiness(pos, n_days) > critical[n]
    return clumpy / n_customers


def _week_keys(dates: pd.Series) -> list[pd.Series]:
    return [dates.dt.year.rename("year"), bfp.week_of_year(dates).rename("week")]


def describe(days: pd.DataFrame, cohort: pd.Index, cal_start: pd.Timestamp,
             cal_end: pd.Timestamp, hold_start: pd.Timestamp, hold_end: pd.Timestamp,
             rng: np.random.Generator) -> dict:
    """The row for one cohort and one calibration / holdout split.

    `days` holds one row per cohort member's purchase day, as `(Id, Date)`.
    """
    n = len(cohort)
    cal = days[(days["Date"] >= cal_start) & (days["Date"] <= cal_end)]
    hold = days[(days["Date"] >= hold_start) & (days["Date"] <= hold_end)]
    cal_n = cal.groupby("Id").size().reindex(cohort, fill_value=0)
    hold_n = hold.groupby("Id").size().reindex(cohort, fill_value=0)

    # Seasonality: weekly aggregate of repeat purchases over every calibration week.
    cal_weeks = bfp.complete_week_grid(cal_start, cal_end).set_index(["year", "week"]).index
    first = cal.groupby("Id")["Date"].transform("min")
    repeat = cal.loc[cal["Date"] > first, "Date"]
    weekly = repeat.groupby(_week_keys(repeat)).size().reindex(cal_weeks, fill_value=0)
    median = weekly.median()
    seasonality = (weekly - median).abs().mean() / median if median > 0 else np.nan

    hold_weeks = len(bfp.complete_week_grid(hold_start, hold_end))
    active_cells = hold.groupby(["Id", *_week_keys(hold["Date"])]).ngroups
    both = pd.concat([cal, hold])
    max_week = both.groupby(["Id", *_week_keys(both["Date"])]).size().max()
    last_cal_year = (cal["Date"] > cal_end - pd.Timedelta(days=364)).sum()

    return {
        "customers": n,
        "cal_weeks": len(cal_weeks),
        "cal_mean": cal_n.mean(),
        "non_repeaters_%": 100 * (cal_n == 1).mean(),
        "hold_weeks": hold_weeks,
        "hold_mean": hold_n.mean(),
        "inactive_%": 100 * (hold_n == 0).mean(),
        "seasonality": seasonality,
        "rWM": wheat_morrison_r(cal, rng),
        "clumpy_%": 100 * clumpy_share(cal, n, cal_start, cal_end, rng),
        "hold_zero_cells_%": 100 * (1 - active_cells / (n * hold_weeks)),
        "max_week_count": int(max_week),
        "hold/last_cal_year": len(hold) / last_cal_year * 52 / hold_weeks,
        "cal_transactions": len(cal),
        "hold_transactions": len(hold),
    }


def cohort_days(name: str) -> tuple[pd.Index, pd.DataFrame]:
    """A dataset's cohort, and one `(Id, Date)` row per member's purchase day."""
    spec = bfp.SPECS[name]
    tx, _ = bfp.read(name)
    cohort = bfp.select_cohort(tx, spec)
    days = tx.loc[tx["Id"].isin(cohort), ["Id", "Date"]].copy()
    days["Date"] = days["Date"].dt.normalize()
    return cohort, days.drop_duplicates()


def to_markdown(table: pd.DataFrame) -> str:
    cells = table.astype(str)
    lines = ["| " + " | ".join(table.columns) + " |",
             "|" + "|".join("---" for _ in table.columns) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in cells.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", help=f"any of {', '.join(bfp.SPECS)}")
    parser.add_argument("--out", type=Path, help="also write the table here as CSV")
    args = parser.parse_args()
    unknown = set(args.datasets) - set(bfp.SPECS)
    if unknown:
        parser.error(f"unknown datasets: {', '.join(sorted(unknown))}")

    rows = []
    for name in args.datasets or bfp.SPECS:
        # Seeded per dataset, so a subset run reproduces the full run's rows.
        rng = np.random.default_rng(SEED)
        spec = bfp.SPECS[name]
        cohort, days = cohort_days(name)
        splits = {}
        for cal in CALIBRATIONS:
            try:
                w = {k: pd.Timestamp(v) for k, v in bfp.windows(spec, cal).items()}
            except ValueError:  # the data end before this calibration's holdout does
                continue
            splits[cal] = (w["training_start"], w["training_end"],
                           w["holdout_start"], w["holdout_end"])
        if name == "multichannel":
            cs, ce, hs, weeks = VALENDIN_MULTICHANNEL
            hs = pd.Timestamp(hs)
            splits["valendin_52w_338w"] = (pd.Timestamp(cs), pd.Timestamp(ce), hs,
                                           hs + pd.Timedelta(weeks=weeks, days=-1))
        for cal, split in splits.items():
            rows.append({"dataset": name, "calibration": cal,
                         **describe(days, cohort, *split, rng)})
            print(f"{name} {cal} done", file=sys.stderr, flush=True)

    table = pd.DataFrame(rows)
    if args.out:
        table.to_csv(args.out, index=False)
    print(to_markdown(table.round(2)))


if __name__ == "__main__":
    main()
