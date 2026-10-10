# Datasets

How the raw sources in `Datasets/Datasets_full/` become the weekly panels in
`Datasets/Dataset_full_clean/`. The builder is `scripts/build_full_panels.py`. The rules
it must keep are pinned by `tests/test_build_full_panels.py`, which runs on synthetic
rows, and `tests/test_full_panels_real.py`, which runs on the real files. All counts
below were produced by running the builder.

```
python scripts/build_full_panels.py all        # or one name: electronics, gift, ...
```

`Datasets_full/zip/# Heuristics Project.zip` is a byte-for-byte archive of the other
seven folders, so it is not read. There is no CDNOW data in `Datasets_full`; the CDNOW
panel still comes from `scripts/build_cdnow_panel.py`. The old panels in
`Dataset_clean/` and their builders are left unchanged, so past results remain
reproducible.

## Output

Each dataset gets one panel per calibration it can hold:

| File | Content |
|---|---|
| `<name>_<2y\|3y\|5y>_customer_week_panel.csv` | `Id, year, week, Transactions`, then the static covariates |
| `<name>_<2y\|3y\|5y>_customer_week_panel.config.json` | `PanelConfig.to_dict()`: window dates, `static` role, embedding declarations |

The CSV is dense: every customer has every week, with zeros filled in, in the layout
`prepare_dataset` reads. The sidecar means nothing needs restating:

```python
config = PanelConfig.from_dict(json.loads(Path(".../gift_2y_customer_week_panel.config.json").read_text()))
data = prepare_dataset(pd.read_csv(".../gift_2y_customer_week_panel.csv"), config)
```

The sidecar declares:

- the count and the week as embedded (`"auto"`);
- each categorical static at its fixed cardinality;
- `first_spend` as a numeric static, which `prepare_dataset` standardises on
  calibration.

It sets no `clip_target_upper` and no calendar or AR features. Those are modelling
choices a study adds.

## Rules shared by every dataset

**One transaction = one customer-day with at least one purchase row.** Line items and
same-day orders collapse to one, which is the trip level the CLV papers model.
`Transactions` for a week is the number of distinct purchase days in it: at most 7, or
more in week 51, which absorbs the last 9–10 days of the year. Counting line items instead is what inflated the old
electronics panel (2.24 lines per trip).

**What a purchase row is** is decided per dataset. Returns, return contracts and
zero-price lines are not purchases (see each dataset below).

**Acquisition cohort.** A customer belongs to the panel iff their *first purchase in
the file* falls in `[cohort_start, cohort_end]`. Anyone who bought earlier was already
a customer at the panel start, so they are excluded (left-censored), even if they also
buy inside the window. Every cohort member is therefore active in calibration, and
`prepare_dataset`'s calibration-activity filter drops nobody. This is the cohort
design of the source papers, and it keeps every customer's history starting at their
acquisition.

**Calendar.** Weeks are Valendin's `dayofyear // 7`, capped at 51 (ADR-0009, via
`period_calendar`), and only complete weeks are kept. The panel starts on the first
complete week of the cohort window.

**Windows.** They are whole years of week buckets, each starting at the panel's first
week index one calendar year later, so every year holds exactly 52 weeks. Validation
is always the last calibration year (ADR-0001).

| Calibration | Training | Validation | Holdout | Panel length |
|---|---|---|---|---|
| `2y` | year 1 | year 2 | year 3 | 156 weeks |
| `3y` | years 1–2 | year 3 | year 4 | 208 weeks |
| `4y` | years 1–3 | year 4 | year 5 | 260 weeks |
| `5y` | years 1–4 | year 5 | year 6 | 312 weeks |

`5y` exists for electronics only. It is the split Valendin et al. use for that dataset
(260 calibration weeks, one-year holdout). `4y` is defined but no dataset lists it yet,
so no `4y` panel is on disk; "Descriptive statistics" below says which datasets can
hold it.

The panel ends at the end of the holdout. A calibration the data cannot cover raises
an error instead of being trimmed, which is why VoD has no `3y` panel.

**Covariates are static only.** Two kinds are kept:

- demographics;
- facts fixed at the first purchase: its channel, its category, and its spend as
  `first_spend = log1p(spend on the first purchase day)`. The log keeps a handful of
  large first baskets from setting the scale.

Categorical codes start at 0, and 0 means *missing / unmatched / unknown* wherever a
value can be missing.

Time-varying covariates in the raw files are **not** used:

- weekly spend and basket size;
- channel mix and returns;
- catalog/e-mail contacts;
- game messages, logins, visits and friendships.

Each of these is either the customer's own behaviour or not available over the
windows. A rollout simulates the holdout one week at a time from its own sampled
counts, so any covariate the model reads there must be known in advance or computable
from that simulated history (`docs/feature-engineering.md`). Behaviour covariates are
`observed_past`, which `prepare_dataset` drops.

## Summary

| Dataset | Transaction | Cohort (first purchase) | Customers | Calibrations | Holdout transactions 2y / 3y | Customers active in holdout 2y / 3y |
|---|---|---|---|---|---|---|
| electronics | household-day, purchase types, price > 0 | 1998-12-02 .. 1999-11-30 | 3,755 | 2y, 3y, 5y | 2,625 / 2,598 | 34.3% / 32.5% |
| gift | customer-day with an order | 2001-03-01 .. 2001-05-31, acquired ≥ 2001-03 | 918 | 2y, 3y | 416 / 391 | 22.4% / 22.2% |
| multichannel | customer-day with a priced order line | 2005-01-01 .. 2005-03-31 | 1,379 | 2y, 3y | 220 / 160 | 11.2% / 9.3% |
| books | customer-day with a priced line | 2008-01-01 .. 2008-03-31 | 1,218 | 2y, 3y | 2,264 / 2,152 | 70.0% / 65.3% |
| apparel | customer-day with spend > 0 | 1996-08-11 .. 1996-12-31 | 14,358 | 2y, 3y | 15,679 / 13,803 | 48.5% / 43.3% |
| vod | customer-day with a rental | 2011-01-01 .. 2011-03-31 | 4,843 | 2y | 33,626 / – | 56.5% / – |
| game | receiver-day | 2008-01-01 .. 2008-03-31 | 6,173 | 2y, 3y | 850 / 258 | 4.5% / 1.7% |

## Descriptive statistics

The columns of Valendin et al.'s Table 3, for every calibration the raw data can cover,
plus four columns this project's models need. They come from
`scripts/describe_full_panels.py`, which reuses the builder's readers, cohorts and
windows, so they describe exactly the panels a study reads:

```
PYTHONPATH=src python scripts/describe_full_panels.py
```

Its docstring defines every column exactly. In short:

- **Cal. mean, Non-repeaters.** Purchase days per customer over the whole calibration
  (training + validation), and the share with exactly one: the acquisition purchase.
- **Hold. mean, Inactive.** Holdout purchase days per customer, and the share with none.
- **Seasonality.** Valendin's eq. 4: the mean absolute deviation of weekly aggregate
  repeat purchases from their median, over that median. Below 0.5 is low, 0.5–1 mild,
  above 1 strong.
- **rWM.** Wheat & Morrison timing regularity, as BTYDplus `estimateRegularity` computes
  it. About 1 is random timing, above 1 regular, below 1 irregular.
- **Clumpy.** The share of the cohort that is clumpy by Zhang, Bradlow & Small's
  entropy measure, against a 95% critical value simulated per purchase count.
- **Hold. zero cells.** The share of holdout customer-weeks with no purchase.
- **Max/week.** The largest weekly count, which sets the size of the softmax head.
- **Hold. ÷ last cal. year.** Holdout purchases over those of the last calibration year.
  Below 1 means the base is shrinking.

All of these are computed on purchase days, the transaction unit of every panel.
rWM and the clumpiness critical values are random draws from a seeded generator, so a
rerun reproduces them exactly.

| Dataset | Cal. | Cohort | Cal. weeks | Cal. mean | Non-repeaters | Hold. mean | Inactive | Seasonality | rWM | Clumpy | Hold. zero cells | Max/week | Hold. ÷ last cal. year |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| electronics | `2y` | 3,755 | 104 | 2.51 | 44% | 0.70 | 66% | 0.25 | 0.53 | 12% | 98.8% | 5 | 0.89 |
| electronics | `3y` | 3,755 | 156 | 3.21 | 33% | 0.69 | 67% | 0.31 | 0.51 | 17% | 98.8% | 5 | 1.00 |
| electronics | `4y` | 3,755 | 208 | 3.91 | 26% | 0.64 | 71% | 0.32 | 0.50 | 22% | 98.9% | 5 | 0.93 |
| electronics | `5y` | 3,755 | 260 | 4.55 | 23% | 0.57 | 73% | 0.33 | 0.49 | 26% | 99.0% | 5 | 0.90 |
| gift | `2y` | 918 | 104 | 1.52 | 70% | 0.45 | 78% | 0.64 | 0.82 | 6% | 99.2% | 3 | 1.75 |
| gift | `3y` | 918 | 156 | 1.98 | 60% | 0.43 | 78% | 0.59 | 0.84 | 9% | 99.2% | 3 | 0.94 |
| gift | `4y` | 918 | 208 | 2.40 | 53% | 0.40 | 81% | 0.51 | 0.82 | 13% | 99.3% | 3 | 0.93 |
| gift | `5y` | 918 | 260 | 2.80 | 49% | 0.34 | 83% | 0.50 | 0.97 | 14% | 99.3% | 3 | 0.88 |
| multichannel | `2y` | 1,379 | 104 | 1.42 | 74% | 0.16 | 89% | 0.48 | 0.82 | 6% | 99.7% | 2 | 0.86 |
| multichannel | `3y` | 1,379 | 156 | 1.58 | 69% | 0.12 | 91% | 0.61 | 0.81 | 8% | 99.8% | 2 | 0.73 |
| multichannel | `4y` | 1,379 | 208 | 1.69 | 66% | 0.11 | 92% | 0.58 | 0.79 | 9% | 99.8% | 2 | 0.97 |
| multichannel | `5y` | 1,379 | 260 | 1.81 | 64% | 0.10 | 94% | 0.56 | 0.88 | 11% | 99.8% | 2 | 0.86 |
| multichannel | Valendin 52 / 338 w | 1,379 | 52 | 1.23 | 83% | 0.79 | 69% | 0.39 | 0.80 | 3% | 99.8% | 2 | 0.10 |
| books | `2y` | 1,218 | 104 | 4.83 | 10% | 1.86 | 30% | 0.36 | 1.23 | 7% | 96.5% | 2 | 0.93 |
| books | `3y` | 1,218 | 156 | 6.69 | 6% | 1.77 | 35% | 0.36 | 1.17 | 10% | 96.6% | 2 | 0.95 |
| books | `4y` | 1,218 | 208 | 8.46 | 4% | 1.29 | 44% | 0.37 | 1.09 | 11% | 97.5% | 2 | 0.73 |
| books | `5y` | 1,218 | 260 | 9.75 | 4% | 1.24 | 49% | 0.43 | 1.06 | 15% | 97.6% | 2 | 0.96 |
| apparel | `2y` | 14,358 | 104 | 3.17 | 31% | 1.09 | 52% | 0.55 | 0.86 | 9% | 98.0% | 4 | 1.00 |
| apparel | `3y` | 14,358 | 156 | 4.27 | 24% | 0.96 | 57% | 0.69 | 0.84 | 13% | 98.2% | 4 | 0.88 |
| apparel | `4y` | 14,358 | 208 | 5.23 | 21% | 1.02 | 55% | 0.69 | 0.87 | 15% | 98.1% | 4 | 1.07 |
| apparel | `5y` | 14,358 | 260 | 6.25 | 19% | 0.92 | 59% | 0.67 | 0.88 | 18% | 98.3% | 4 | 0.90 |
| vod | `2y` | 4,843 | 104 | 18.83 | 12% | 6.94 | 44% | 0.17 | 0.86 | 42% | 90.1% | 8 | 0.84 |
| game | `2y` | 6,173 | 104 | 4.89 | 46% | 0.14 | 96% | 1.84 | 0.60 | 44% | 99.8% | 8 | 0.30 |
| game | `3y` | 6,173 | 156 | 5.03 | 46% | 0.04 | 98% | 2.64 | 0.57 | 48% | 99.9% | 8 | 0.31 |

The holdout totals behind *Hold. mean* match the summary table above for every `2y` and
`3y` panel, and the `5y` electronics total (2,153) matches the electronics section.

**Which calibrations the data can hold.** A row is listed only when the raw data cover
its whole holdout. Electronics, gift, multichannel, books and apparel run to at least
`5y`, so `4y` fits them all. VoD ends in January 2014 and holds only `2y`. Game ends in
October 2012 and holds `2y` and `3y`. CDNOW is not in `Datasets_full`. Its 39 + 39 weeks
cannot hold even `2y`. The `4y` and `5y` rows (except electronics `5y`) describe panels
that have not been built.

### Reproducing Valendin et al.'s Table 3

Two of Valendin et al.'s eight datasets are here, and each has a row on the paper's
split.

| Electronics (260 / 52 w) | Cohort | Clumpy | rWM | Seasonality | Cal. mean | Non-repeaters | Hold. mean | Inactive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Valendin et al., Table 3 | 3,782 | 24% | 0.5 | 0.3 | 4.5 | 23% | 0.6 | 72% |
| `5y` panel | 3,755 | 26% | 0.49 | 0.33 | 4.55 | 23% | 0.57 | 73% |

| Multichannel (52 / 338 w) | Cohort | Clumpy | rWM | Seasonality | Cal. mean | Non-repeaters | Hold. mean | Inactive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Valendin et al., Table 3 | 1,379 | 10% | 0.9 | 0.4 | 1.2 | 83% | 0.8 | 69% |
| Valendin's split, this cohort | 1,379 | 3% | 0.80 | 0.39 | 1.23 | 83% | 0.79 | 69% |

- **Electronics reproduces in every column.** That includes the two counting columns.
  Counting line items instead of purchase days would give about 2.24 times the
  transactions, so the paper's electronics counts are effectively at trip level too.
  The paper never states its transaction unit; this match is the evidence.
- **Multichannel reproduces the cohort and all five activity columns. Its clumpiness and
  rWM do not.** Computing both over the whole observation span instead of the
  calibration does not close the gap either: that gives 14% clumpy and rWM 0.85. The
  paper does not give its clumpiness implementation. Only 55 multichannel
  customers make three purchases in 2005, so rWM rests on few customers there.

### Reading the table

- **Activity separates the datasets most.** Grouped by holdout transactions per customer
  and inactivity:
  - *Frequent*: VoD, books and apparel. About one or more holdout purchase days per
    customer, and 30–59% inactive.
  - *Infrequent retail*: electronics and gift. 0.3–0.7 per customer, and 66–83% inactive.
  - *Very sparse*: multichannel and game. 0.16 or fewer per customer, and 89–98% inactive.
- **Game is not stationary.** Its holdout year holds 30% of the last calibration year's
  purchases. Its "strong" seasonality (1.8–2.6) is that decline: the measure cannot tell
  a trend from a season. By Valendin's cut-offs, nothing else is strongly seasonal.
- **Only books is regular** (rWM above 1). Electronics and game are irregular, about
  0.5–0.6. VoD and game are the clumpy panels, at 42–48%.
- **Gift `2y` grows into its holdout** (ratio 1.75). Its second year holds only 238
  purchase days, against 416 in the holdout year.
- **Clumpiness rises with the window**, because it is measured over the whole
  calibration: electronics goes from 12% at `2y` to 26% at `5y`. Compare it within one
  calibration, not across calibrations.
- **None of this predicts collapse.** Collapse is a result, and it has so far been
  measured only on the old electronics and multichannel panels
  (`docs/insights-real-panels.md`). That doc found no panel characteristic that predicts
  it. Whether the very sparse group collapses is a question for the studies on these
  panels.

## Electronics — ISMS Durables Dataset 1

**Source.** `Eletronics Retailer - 6 years/durdata1_final.csv`, documented in
`ISMS Durables Dataset 1 Documentation.docx`. The paper is Ni, Neslin & Sun (2012),
`mksc.1120.0726.pdf`.

- The file has 173,262 line items: 19,936 households buying at 1,176 stores of a US
  electronics chain between 1998-12-01 and 2004-11-30.
- One row is one product on one receipt (`ORIGINAL_TICKET_NBR`).

**Purchase row.** `TRANSACTION_TYPE` ∈ {1 product purchase, 3 service-contract
purchase, 5 discounted purchase} with `EXTENDED_PRICE > 0`. This leaves 148,798 rows.

- Types 2 and 4 are returns and type 6 is miscellaneous.
- This rule reproduces the paper's cohort exactly: 3,782 households with first
  purchase 1998-12-01 .. 1999-11-30, making 19,441 trips (`test_electronics_reproduces_the_paper_cohort`).

**Cohort.** The data open on Dec 1, which is midway through 1998 week 47, so the panel
starts at week 48 (Dec 2). The 27 households whose first purchase was on Dec 1 fall
outside the panel, leaving **3,755**.

**Windows.**

| Calibration | Training | Validation | Holdout |
|---|---|---|---|
| `2y` | 1998-12-02 .. 1999-12-01 | .. 2000-11-30 | 2000-12-01 .. 2001-12-01 |
| `3y` | 1998-12-02 .. 2000-11-30 | .. 2001-12-01 | 2001-12-02 .. 2002-12-01 |
| `5y` | 1998-12-02 .. 2002-12-01 | .. 2003-12-01 | 2003-12-02 .. 2004-11-30 |

**The `5y` panel reproduces the paper's split.** Valendin et al. (Table 3) calibrate
on 260 weeks and hold out the final year, ending on the last day of data. On the
`5y` panel:

| Statistic | Paper | `5y` panel |
|---|---|---|
| Cohort | 3,782 | 3,755 |
| Calibration mean transactions | 4.5 | 4.55 |
| Calibration non-repeaters | 23% | 23% |
| Holdout weeks | 53 | 52 |
| Holdout mean transactions | 0.6 | 0.57 (2,153 in total) |
| Holdout inactive | 72% | 73% |

The holdout is 52 buckets rather than 53 weeks because week 51 absorbs the year's
last days (ADR-0009).

**Counts.** Weekly counts reach 5.

**Static covariates.** They are constant within a household in the raw file.

| Column | Codes | Cohort distribution |
|---|---|---|
| `income` | 1–9 as given (higher = richer; the documentation gives no bands), 0 = missing | 0: 387, 1–9: 219/102/250/298/325/742/546/305/581 |
| `gender` | head of household: 0 = unknown (`U`), 1 = M, 2 = F | 559 / 2,076 / 1,120 |
| `age_band` | age of head: 0 = missing, 1 = <35, 2 = 35–44, 3 = 45–54, 4 = 55–64, 5 = 65+ | 361 / 519 / 799 / 932 / 624 / 520 |
| `children` | 0 = missing, 1 = N, 2 = Y | 1,989 / 615 / 1,151 |
| `first_spend` | log1p of first-trip spend (median $200) | |

**Not used.**

- The online flag: only one cohort household made its first purchase online, so it
  would be constant.
- Store id, product category/brand, individual gender and child-age dummies. The first
  is too granular; the rest are behaviour or redundant with `children`.

**Versus the old panel.** `Dataset_clean/electronics_customer_week_panel.csv` had 829
households and counted line items. Its cohort was Q1 1999 with missing incomes
dropped. The new panel is 4.5× larger, at trip level, on the paper's cohort, and keeps
missing values as a code.

## Gift — DMEF MultiChannel Gift Company

**Source.** `Gift retailer/original_data/`, documented in
`DMEF MultiChannel Gift Dataset.doc` and the `DMEF Demo Codes Reference.xls`. This is
a US food-gift company with stores, catalog and web channels.

| File | Rows | Used for |
|---|---|---|
| `DMEFExtractOrdersV01.CSV` | 241,366 orders | one row per order or store trip (2001-01-01 .. 2008-01-01), with `OrderMethod` |
| `DMEFExtractLinesV01.CSV` | 618,661 line items | summed per order for spend |
| `DMEFExtractSummaryV01.CSV` | 100,051 customers | `AcqDate`, `AgeCode`, `IncCode` |

`gift.csv` in the same folder is a line-level derivative (`prep_gift.R`) and is not
read.

**Purchase row.** Every order with positive spend. There are 185,957 such orders; no
line is ≤ $0. Some customers place several orders on one day (8,756 customer-days), and
these collapse to one.

**Cohort — the left-censoring fix.** The transactions start in 2001, but many customers
are older. The builder keeps a customer iff both conditions hold:

- their first retained order falls in 2001-03-01 .. 2001-05-31 (the old panel's
  window);
- their `AcqDate` (the month the company added them to its database) is 2001-03 or
  later.

That gives **918** of the old panel's 2,062. The other 1,144 were acquired before March
2001, some as early as 1988, so their "first" order in the file is not their first.
The new cohort is a strict subset of the old one (`test_gift_cohort_is_a_subset_of_the_old_panel`). The 108 customers whose
`AcqDate` falls after their first order are kept: the database entry lagged the
purchase, so they are new.

**Windows.** The panel starts at 2001 week 8 (Feb 25), the old panel's start.

| Calibration | Training | Validation | Holdout |
|---|---|---|---|
| `2y` | 2001-02-25 .. 2002-02-24 | .. 2003-02-24 | 2003-02-25 .. 2004-02-24 |
| `3y` | 2001-02-25 .. 2003-02-24 | .. 2004-02-24 | 2004-02-25 .. 2005-02-24 |

**Counts.** Weekly counts reach 3.

**Static covariates.**

| Column | Codes | Cohort distribution |
|---|---|---|
| `age_code` | vendor overlay code 1–7 (the Demo Codes Reference gives the bands), 0 = unmatched | 382 unmatched, 1–7: 5/40/130/140/120/66/35 |
| `income_code` | vendor overlay code 1–9, 0 = unmatched | 418 unmatched, 1–9: 21/6/28/30/34/92/107/61/121 |
| `first_method` | channel of the first order: 0 = store (`ST`), 1 = internet (`I`), 2 = phone (`P`), 3 = mail (`M`) | 444 / 210 / 250 / 14 |
| `first_spend` | log1p of first-day order spend (median $31) | |

**Not used.**

- Catalog and e-mail contacts. The contact file covers 2005–2007 only, after every
  window here.
- Seasonal RFM summaries (behaviour), store distance, psychographics. The last two are
  mostly unmatched.

## Multichannel — specialty catalog retailer

**Source.** `Specialty Multichannel Retailer - 11.5 years/special.csv`, documented in
`data_description.pdf`.

- The file has 226,129 order lines for 100,000 random US customers whose first
  purchase is on or after 2004-12-16. Data run to 2012-09-17, about 7.75 years despite
  the folder name.
- One row is one order line.

**Purchase row.** Order lines with `EXT_PRICE > 0`. This drops 177 free lines and
leaves 225,952. A cancelled order still counts, because the customer placed it; the
cancellation is the retailer's, mostly a stock-out.

**Cohort.** First purchase 2005-01-01 .. 2005-03-31, the old panel's window: **1,379**
customers. The 203 customers who first bought in Dec 2004 are already customers at the
panel start, so they are excluded. The old panel had 1,402, all first seen in 2005, but
it came from the CLVTools `.Rdata` extract, whose `Id` column is corrupted. So the
23-customer difference cannot be traced id by id. `test_cohort_sizes_are_close_to_the_old_ones` holds the two within 5%.

**Windows.** The panel starts on 2005-01-01.

| Calibration | Training | Validation | Holdout |
|---|---|---|---|
| `2y` | 2005 | 2006 | 2007 |
| `3y` | 2005–2006 | 2007 | 2008 |

**Counts.** Weekly counts reach 2. This is a sparse panel: 11% of customers buy in the
2y holdout.

**Static covariates.**

| Column | Codes | Cohort distribution |
|---|---|---|
| `first_channel` | channel of the first order: 0 = mail (`ML`), 1 = phone (`PH`), 2 = web (`WE`) | 77 / 644 / 658 |
| `first_division` | division credited with the first order: 0 = id 1, 1 = id 5. The PDF documents 01 = catalog and 02 = web, but the file holds 1 and 5. | 777 / 602 |
| `first_spend` | log1p of first-day spend (median $75) | |

**Not used.** ZIP code (too granular), offer id, payment method, returns and
back-orders.

## Books — German book retailer (Kaggle)

**Source.** `German Book retailer/Kaggle Files/orders.csv`.

- The file has 353,687 line items (`id, orddate, ordnum, category, qty, price`) for
  16,781 customers, 2007-11-04 .. 2014-11-24.
- `customer.csv` holds only a Kaggle train/test split and target, which are not used.

**Purchase row.** Lines with `price > 0`. The 10,749 zero-price lines are free items,
and a day made only of them (867 days) is not a sale.

**Cohort.** First purchase 2008-01-01 .. 2008-03-31, the window of the old
`Book Store data.Rdata`: **1,218** customers, exactly the old count.

**Windows.** The panel starts on 2008-01-01.

| Calibration | Training | Validation | Holdout |
|---|---|---|---|
| `2y` | 2008 | 2009 | 2010 |
| `3y` | 2008–2009 | 2010 | 2011 |

**Counts.** Weekly counts reach 2. This is the most active panel: 70% of customers buy
in the 2y holdout.

**Static covariates.**

| Column | Codes |
|---|---|
| `first_category` | category of the most expensive line on the first purchase day. The 30 codes in the file (`BOOK_CATEGORIES`: 1, 3, 5, …, 50, 99) map to 0–29 in ascending order. They are undocumented, and 99 looks like a catch-all. |
| `first_spend` | log1p of first-day spend (median €31) |

## Apparel — Zitzlsperger, 10 years

**Source.** `Apparel (Zitzlsperger) - 10 years/Apparel.txt` (identical to the copy in
`Data.zip`).

- The file has no header: `id, dd.mm.yyyy, amount`.
- It has 2,632,383 rows for 600,152 customers, 1996-08-07 .. 2006-11-23.
- It is already one row per customer-day.
- `Data.zip` also holds two unrelated datasets (`OnlineCommunity.txt`, `B2C.txt`),
  which are not used.

**Purchase row.** `amount > 0`. This drops the 1,006 zero-amount rows.

**Cohort.** The source's `clvtools.R` cohort is everyone whose first purchase was
before 1997. The data open inside 1996 week 31, so the panel starts at week 32
(Aug 11), and the one customer who first bought on Aug 7 falls outside it. That leaves
**14,358** customers.

**Windows.**

| Calibration | Training | Validation | Holdout |
|---|---|---|---|
| `2y` | 1996-08-11 .. 1997-08-11 | .. 1998-08-11 | 1998-08-12 .. 1999-08-11 |
| `3y` | 1996-08-11 .. 1998-08-11 | .. 1999-08-11 | 1999-08-12 .. 2000-08-10 |

**Counts.** Weekly counts reach 4.

**Static covariates.** Only `first_spend` (median €92). The file holds nothing else.

## VoD — video on demand, 5.5 years

**Source.** `Video-on-Demand - 5.5 years/vod.csv`.

- The file has 3,977,862 rentals (`Id, Date, Price`) for 218,860 customers, 2008-07-24
  .. 2014-01-31.
- One row is one rental.

**Purchase row.** Every rental, including the 481,263 free ones. On a streaming
service a zero-price rental is consumption like any other.

**Cohort.** `vod.R` sets aside the pre-2011 cohorts as behaving differently. First
purchase 2011-01-01 .. 2011-03-31 gives **4,843** customers.

**Windows.** The data end in January 2014, so only **`2y`** fits: 2011 / 2012 / 2013.

**Counts.** Weekly counts reach 8. This is the heaviest-count panel, with 2+ purchase
days in about 3% of customer-weeks.

**Static covariates.** Only `first_spend` (log1p of first-day rental spend, median 7.5).

## Game — Timik.pl virtual world

**Source.** `Online Game - 6 years/transactions.csv`, from "A multilayer network dataset
of interaction and influence spreading in a virtual world" (Scientific Data, 2017).

- The file has 538,597 virtual-currency transfers (`datetime; sender; receiver;
  amount`), 2007-10-28 .. 2012-10-04.
- As in the source's `load.R`, the **receiver** is the customer.

**Purchase row.** Every receipt, except:

- receipts by the two platform accounts 16453 (campaign activator, 30,971 receipts) and
  509053 (4,001);
- 8,658 self-transfers;
- 7 transfers of amount ≤ 0.

**Cohort.** First receipt 2008-01-01 .. 2008-03-31: **6,173** receivers.

**Windows.** The panel starts on 2008-01-01.

| Calibration | Training | Validation | Holdout |
|---|---|---|---|
| `2y` | 2008 | 2009 | 2010 |
| `3y` | 2008–2009 | 2010 | 2011 |

**Counts.** Weekly counts reach 8.

**Caveat.** The platform winds down. Only 4.5% of the cohort receives anything in the
2y holdout and 1.7% in the 3y holdout (258 transactions). Treat forecasts here as a
dying-platform stress test, not a steady-state customer base.

**Static covariates.** Only `first_spend` (log1p of first-day amount received, median
15 units of virtual currency).

**Not used.** Messages, logins, visits, friendships and campaigns. All are behaviour.

## How the gift panel was built over time: Valendin, Markus and this project

Five pipelines turn transactions into weekly per-customer counts. Gift is the one dataset
all of them touch, so it is the comparison here. v3 is the builder of the 2-year
`Dataset_clean/` gift panel used by the benchmark runs (`docs/benchmarks.md`).

| Pipeline | Where it lives |
|---|---|
| **Valendin et al.** | `Original_paper_model/banking_transactions_demo.ipynb` — the reference notebook, run on Czech bank data |
| **Markus** | `Markus/Markus models/TransformerMA-main/r_code/gift/Gift-preparation.r`, then `python_code/scripts/data/data_loader.py` and `data_preprocessor.py` |
| **v1** | `~/ML Projects/thesis/Gift/Gift.ipynb`, `Gift2.ipynb`, `Gift 3.ipynb` — the Valendin notebook adapted to gift |
| **v2** | `notebooks/archive/dataset_building.ipynb`, then `prepare_dataset` (the `Study_historique/Gift` runs) |
| **v3 (current)** | `scripts/build_rdata_panel.py`, then `prepare_dataset` |

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

**What differs, and what it does to a comparison.**

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

**Caveats.**

- `gift.csv` and Markus's processed `20250417_1159_gift_processed_co_1_70.csv` are not on
  disk. His column is read from his code and his saved plots, not re-run.
- The exact v2 gift window dates are known only as "2 years / 1 year", from the run
  folder names.
