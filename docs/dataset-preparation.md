# Dataset preparation: Valendin, Markus and this project

How five pipelines turn transactions into weekly per-customer counts. Written for the
gift retailer, because it is the one dataset all of them touch.

The panels built from the complete raw sources in `Datasets/Datasets_full/` (seven
datasets, acquisition cohorts, 2y and 3y calibrations) are documented in
`docs/datasets.md`.

| Pipeline | Where it lives |
|---|---|
| **Valendin et al.** | `Original_paper_model/banking_transactions_demo.ipynb` — the reference notebook, run on Czech bank data |
| **Markus** | `Markus/Markus models/TransformerMA-main/r_code/gift/Gift-preparation.r`, then `python_code/scripts/data/data_loader.py` and `data_preprocessor.py` |
| **v1** | `~/ML Projects/thesis/Gift/Gift.ipynb`, `Gift2.ipynb`, `Gift 3.ipynb` — the Valendin notebook adapted to gift |
| **v2** | `notebooks/archive/dataset_building.ipynb`, then `prepare_dataset` (the `Study_historique/Gift` runs) |
| **v3 (current)** | `scripts/build_rdata_panel.py`, then `prepare_dataset` |

## Comparison

| Step | Valendin | Markus | v1 | v2 | v3 (current) |
|---|---|---|---|---|---|
| **Raw source** | Bank CSV (`account_id, date`) | `gift.csv` (`ID, date, amount`) | `Gift Retailer datapy.csv`, exported from the Rdata | `Gift Retailer data (weekly).Rdata`, read through Rscript | Same Rdata |
| **Customer set** | First transaction ≤ `training_end` | Full base, then monthly cohorts 1–70 (first purchase Jan 2001 – Oct 2006) and first purchase ≤ 2006-12-25 | 2,062-customer CLVTools cohort, plus first transaction ≤ `training_end` | 2,062 (the ids in `covariates.dynamic`) | 2,062, plus first transaction ≤ `training_end` inside `prepare_dataset` |
| **One transaction is** | One row | One customer-day (same-day purchases merged) | One row | One row | One row (multichannel: one distinct `ORDER_NO`) |
| **Week of year** | `dayofyear // 7`, capped at 51 | `dayofyear // 7`, capped at 51 | `dayofyear // 7`, capped at 51 | `dayofyear // 7`, capped at 51 | `dayofyear // 7`, capped at 51 (ADR-0009) |
| **Time grid** | Every day `training_start` → `holdout_end`, grouped into (year, week) | Every day 2001-01-01 → 2007-12-30, grouped into (year, week) | As Valendin; data cut after 2004-12-31 or 2005-12-31 | Full calendar years 2001–2008, 52 weeks each | Complete weeks only: 2001 w8 (Feb 25) → 2007 w51 |
| **Calibration / holdout** | 1993–95 / 1996–98 | 2001-01-01 → 2006-12-25 / 2006-12-26 → 2007-12-30 | Several, e.g. 2001–03 / 2004–06 and 2001–02 / 2003–04 | 2 years / 1 year | Provisional: 2001-02-25 → 2003-12-31 / 2004–2007 |
| **Count clipping** | None (clip at 6 commented out) | None | None, or clip at 8 on the whole panel (`Gift 3`) | `clip_target_upper`, training target only | `clip_target_upper`, training target only |
| **Model inputs** | Week + count, both embedded | Year index, month, week, count, cohort, sin/cos of week and month | Week + count | Per `PanelConfig` | Per `PanelConfig` (the Valendin arm: count + embedded week) |
| **Validation split** | 10% of customers, unseeded shuffle | 10% of customers, seed 1337 | 10% of customers, unseeded | 10% of customers (`random_state=42`), later a 2-month time window | Time window from `validation_start`, all customers (ADR-0001) |

## What differs, and what it does to a comparison

- **Customer base.** Markus is the only pipeline not on the 2,062-customer cohort. His
  gift plot peaks near 5,000 transactions in a week; the whole 2,062-customer file
  holds 8,795 over seven years. His gift numbers cannot be compared with ours directly.
- **Same-day merging.** Markus merges purchases on one day. On the CLVTools Rdata it
  changes nothing: no customer has two rows on one date.
- **Week rule.** All five now agree. Before ADR-0009 the package, CDNOW, and the first
  v3 gift and multichannel panels used `(dayofyear − 1) // 7`, one day off everywhere
  except Jan 1.
- **Validation.** Only v3 (and the late v2 runs) validate on a time window. That is the
  deliberate departure from Valendin recorded in ADR-0001.

## Caveats

- `gift.csv` and Markus's processed `20250417_1159_gift_processed_co_1_70.csv` are not on
  disk. His column is read from his code and his saved plots, not re-run.
- The exact v2 gift window dates are known only as "2 years / 1 year", from the run
  folder names.
