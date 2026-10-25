# Are the literature's datasets comparable to ours?

A literature note for the review. Every paper in Part A of `docs/litterature_review.md` is
placed on the same descriptors `docs/datasets.md` uses for the thesis panels (cohort size,
time unit, calibration and holdout length, mean calibration transactions, share of
non-repeaters, share inactive in the holdout), plus the forecast target and the split type.
The point is to say, paper by paper, whether a reported win or loss was obtained on data
like ours.

**Method.** Descriptors were read from each paper's own data table or data section (the
Valendin et al. Table 3 reproduction and our panel statistics are in `docs/datasets.md`
§Descriptive statistics). "n/r" means the paper does not report it. Where a paper uses one
of our panels, that is said. Every number is the paper's claim.

## 1. Our panels, for reference (2y calibration, 52-week holdout; `docs/datasets.md`)

| Panel | Customers | Cal. weeks | Cal. mean trans. | Non-repeaters | Inactive in holdout | Holdout zero cells |
|---|---:|---:|---:|---:|---:|---:|
| electronics | 3,755 | 104 | 2.51 | 44% | 66% | 98.8% |
| gift | 918 | 104 | 1.52 | 70% | 78% | 99.2% |
| multichannel | 1,379 | 104 | 1.42 | 74% | 89% | 99.7% |
| books | 1,218 | 104 | 4.83 | 10% | 30% | 96.5% |
| apparel | 14,358 | 104 | 3.17 | 31% | 52% | 98.0% |
| vod | 4,843 | 104 | 18.83 | 12% | 44% | 90.1% |
| game | 6,173 | 104 | 4.89 | 46% | 96% | 99.8% |

Range: 918 to 14,358 customers; one to two transactions per customer per year on most
panels; 10% to 74% never repeat in calibration; 30% to 96% are silent in the holdout; 90%
to 99.8% of customer-weeks in the holdout are zero. The target is the weekly count.

## 2. The literature's datasets

Time unit is weekly unless stated. "Temporal" split = calibration window then holdout window
for every customer (ours); "customer-wise" = random customers held out.

| Study | Target | Customers | Time unit | Calibration | Holdout | Cal. mean trans. | Non-repeaters | Inactive in holdout | Split | Comparable to ours? |
|---|---|---:|---|---|---|---:|---:|---:|---|---|
| Valendin et al. (2022), 8 panels | weekly count | 1,379 to 31,216 | week (one monthly check) | 39 to 260 w | 39 to 338 w | 1.2 to 43.7 | 6% to 83% | 46% to 81% | temporal | **Yes.** Two of the eight are our electronics and multichannel cohorts (Table 3 reproduced in `docs/datasets.md`); the rest span our range |
| Bachmann et al. (2021), 3 retailers | count (conditional expectation) | 1,402; 829; 1,071 | continuous (weekly covariates) | 1 y; 2 y; 2 y | 6.7 y; 4 y; 4 y | 1.24; 3.09; 1.49 | 82%; 33%; 69% | n/r | temporal | **Yes.** Same multichannel and electronics sources as ours; holdouts much longer |
| Näf et al. (2026), 3 retailers | cumulative revenue | 4,785; 4,661; 5,843 | continuous | 2 y | 1 to 4 y | 4.15; 4.28; 3.53 | 55%; 48%; 28% | n/r | temporal | **Yes on data** (Retailer B is the ISMS electronics source), **no on target** (money) |
| Kurbiel (2024) | weekly count (regression head) | 23,570; 2,454 | week | 39 w; 52 w | 39 w; 52 w | CDNOW: 2.1 (Valendin's figure, same cohort and window); app: n/r | CDNOW 59%; app n/r | CDNOW 70%; app n/r | temporal windows, but customers split at random 90/10 and 70/30; training set resampled | CDNOW yes; the app panel is undescribed |
| Kiyakoglu & Aydin (2024) | weekly count (logins ≤ 7; transactions) | 53,576; 4,993 | week | 49 w; 80 w | 25 w; 25 w | n/r ("a significant portion of users had zero logins during most of the calibration period") | n/r | n/r; holdout mean 2.3 logins and 1.7 transactions per customer over 25 w | temporal | Partly: sparsity similar to ours, holdout half as long, first panel is logins not purchases |
| Gadgil et al. (2023) | 3-month count | 5,850 | n/a (features) | about 2 y of history | 3 months | about 68 transactions per customer over the two years (397,924 / 5,850); wholesale clients | n/r | n/r | n/r | **No.** Dense wholesaler data, about 30 times our activity; one short horizon |
| Xie (2019) | holdout count and active status | 23,570; 24,000 | n/a (RFM) | first half (about 9 months) | second half | CDNOW 7.1 CDs per customer over 18 months; E-tailing 4.9 orders | n/r | n/r | customer-wise 60/40, then temporal within | CDNOW yes; split differs |
| Xie (2020) | holdout count; inactivity | 5,000; 5,000; 1,525 | day | 91 combinations, holdout ≤ calibration, weekly steps | up to about 1 y | GROCERY 6.9 per customer over 2 y; others n/r | n/r | n/r | temporal | Partly: same public sources (CDNOW, grocery), but a feed-forward net on RFM features, many short windows |
| Bauer & Jannach (2021) | revenue over 4 w / 6 months | 3,284; 1,073,872 | day; week | about 6 months; several years | equal halves; last 6 months | UKRetail 44 records per customer per year (wholesalers removed); Children 9.5 records per customer | n/r | n/r | temporal | **No.** One panel is dense wholesale data, the other is 75 to 1,000 times our cohorts; money target |
| Chou et al. (2022) | repurchase next quarter (binary) | 532,410 | quarter | 10 quarters | 1 quarter | 2.3 transactions per member over 33 months | n/r | 80% to 90% of member-quarters are no-repurchase | temporal | **No.** 40 to 600 times our cohorts, quarterly unit, yes/no target |
| Sarkar & De Bruyn (2021) | donation next month (binary); amount | 6,134; 61,928 | month | up to 8 y | 12 one-month tasks; one campaign | n/r | n/r | campaign response 10.4% (6,456 / 61,928) | temporal (rolling) | Partly: sparse and small like ours, but one-step-ahead yes/no |
| Sarkar & De Bruyn (2026) | distribution of donor value | 21,939 | n/r | n/r (9 y of data) | n/r | n/r | n/r | n/r | n/r | Unknown; abstract only |
| Chen, Guitart et al. (2018) | 365-day spend | 2,505 paying, churned players | day | since install (up to 2.5 y) | 365 d | n/r; "whales" 2% of players make up to 50% of revenue | n/a (all paid) | n/a (all churned) | customer-wise 80/20 | **No.** Selected payers who churned, revenue, customer-wise split |
| Chen, Ng, Smyl & Steininger (2024) | revenue 1 to 26 w | n/r (Uber, Meta) | week | up to 4 y | 1 to 26 w | n/r | n/r | n/r | n/r | **No.** Proprietary, undescribed |
| Dew & Ansari (2018) | daily spend incidence; evaluated on the cohort's daily spend | 10,000 per game, installed in a 30-day window and spent at least once | day | all but last 30 d | 30 d | n/r | 0% by selection | n/r | 8,000 / 2,000 customer-wise plus 30-day temporal | Partly: non-contractual and sparse, but spenders only, one-month holdout, aggregate metric |
| Malthouse (2009); Jamal & Zhang (2009) | 2-year donation amount | 21,166 | n/a | 4.5 y | 2 y | n/r | 54% one-time donors | n/r | temporal | Same cohort as Valendin's Charity panel; money target |
| Mena et al. (2019) | churn in 12 months (binary) | about 385,000 | month | 36 months | 12 months | n/r | n/r | 0.24% churners | temporal, under-sampled | **No.** Contractual, binary, 30 times our cohorts |
| Kim, McCarthy & Lee (2026) | cohort-week acquisition, repeat orders, order value | 966 firms (cohort aggregates) | week | 27 months | n/r | n/r | n/a | n/a | temporal | **No.** Aggregates, not customers |

## 3. Reading

- **Count-target papers on data like ours:** Valendin et al. (2022), Bachmann et al. (2021),
  Kurbiel (2024, CDNOW only), Kiyakoglu & Aydin (2024, roughly), Xie (2019, 2020, CDNOW and
  grocery). That is the set whose results bear directly on the thesis question. Their verdict
  is split: the LSTM wins on average in Valendin, loses on CDNOW in Kurbiel, flips with the
  metric in Kiyakoglu & Aydin, and the Pareto/NBD wins the count task in both Xie papers.
- **Every large-margin deep-learning win is on a different target or a different kind of
  panel:** revenue (Bauer & Jannach, Näf, Chen et al. 2018, Chen, Ng et al.), a yes/no event
  (Chou, Mena, Sarkar & De Bruyn 2021), dense wholesale data (Gadgil, Bauer & Jannach
  UKRetail), or cohorts 30 to 1,000 times larger than ours (Chou, Mena, Bauer & Jannach
  Children). None of these transfers to a weekly count on a few thousand customers.
- **Two things no paper in the set reports:** the share of zero customer-periods in the
  holdout (ours is 90% to 99.8%), and any spread across repeated training runs. Both should be
  stated as gaps when the thesis presents its own protocol.
