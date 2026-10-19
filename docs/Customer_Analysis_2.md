# Customer analysis 2: which customers the LSTMs cannot tell apart

The aggregate tables of `docs/insights-real-panels.md` score a forecast over the whole
cohort. This doc breaks the stored forecasts apart by **customer group** to see *which*
customers the two LSTMs get wrong, and whether that depends on the panel: sparsity,
calibration length, the share of customers who bought only once.

The answer, in one line: **both LSTMs make customers look more alike than they are.**
They forecast too much for customers who bought once or went quiet, and too little for
recent, frequent buyers. How far depends on the panel. It is worst with a short
calibration window (CDNOW) or a panel that is mostly one-timers (multichannel), and it
shrinks with a longer calibration window.

Nothing was trained for this doc. Every number is read from forecasts already on disk.

## Contents

1. [Set-up](#1-set-up)
2. [Summary](#2-summary)
3. [One-timers are over-forecast](#3-one-timers-are-over-forecast)
4. [Recent and recent heavy buyers are under-forecast](#4-recent-and-recent-heavy-buyers-are-under-forecast)
5. [Separation: how far apart the best and worst customers are put](#5-separation-how-far-apart-the-best-and-worst-customers-are-put)
6. [Is recency taken into account?](#6-is-recency-taken-into-account)
7. [Panel patterns](#7-panel-patterns)
8. [Every contrast against Pareto/NBD](#8-every-contrast-against-paretonbd)
9. [What is not an LSTM failure](#9-what-is-not-an-lstm-failure)
10. [Enhancements to try](#10-enhancements-to-try)
11. [The customer groups](#11-the-customer-groups)
12. [Reproducing](#12-reproducing)
13. [Appendix: customers and transactions per group](#appendix-customers-and-transactions-per-group)

## 1. Set-up

### The forecasts

| window | ValendinLSTM | LSTM + `bounded32` | Pareto/NBD (reference) |
|---|---|---|---|
| CDNOW, electronics, gift, multichannel, 2-year windows | family N, 20 runs | family O, 100 runs | family N, 20 seeded fits |
| electronics, gift, multichannel, 3-year windows | **never run** (0/20, `docs/studies-run.md` §4) | family P, 100 runs | family Q, 20 seeded fits |
| electronic_5y (the paper's split) | family W, 20 runs | **substitute:** `ar_bounded_52 / nofloor` (family X), 20 runs, because `bounded32` was never run there | family W, 20 seeded fits |

A run is one complete Optuna search plus refit and forecast (`docs/statistical-protocol.md`
§1). A forecast is the mean of 500 simulated paths per customer. CDNOW's "2-year windows"
are its 39 calibration and 39 holdout weeks (ADR-0009). Electronics at 2 and 3 years is
the old line-item panel (829 customers), not the paper's cohort. electronic_5y is the
paper's cohort (`docs/benchmarks.md`).

The Appendix also carries a **Transformer** column, on the two windows where one was run
with the same inputs as the LSTM column. That way the two columns differ only in
architecture:
- **electronics 2y:** `ar_bounded-no_cluster` (family H), 20 runs, 200 paths, the same
  activity flags as `bounded32`;
- **electronic_5y:** `ar_bounded_52 / nofloor` (family Y), 20 runs, matching the LSTM
  substitute.

The Transformer was never run on gift, multichannel, CDNOW's current window or any
3-year window (`insights-real-panels.md` §11). Summed over all customers, its rows
reproduce §12 there too: +2.5% bias and Spearman 0.201 on electronics 2y; +12.5% and
0.401 on electronic_5y.

Check against the published numbers: summed over all customers, every model's bias and
Spearman reproduce `docs/insights-real-panels.md` §12 exactly. For example, electronics 2y
`bounded32` gives −1.7% and 0.263; ValendinLSTM gives +46.0% and 0.032.

### The customer groups

Each group is defined from **calibration data only**, from two numbers per customer:

- **F (frequency):** the customer's number of transactions in the calibration window.
- **R (recency):** weeks between the customer's last calibration purchase and the start of
  the holdout. R = 0 means they bought in the final calibration week.

These are customer groups in the `CONTEXT.md` sense: a way of reading a forecast, not an
input to one. Every customer in a panel's cohort bought at least once in calibration, so
F ≥ 1 always.

| group | definition | why it is interesting |
|---|---|---|
| **one-timers** | F = 1 | The largest group on most panels (13–73% of customers). Their history carries almost no information. |
| **recent buyers** | R ≤ 4 (bought in the last 5 calibration weeks) | The customers most likely to buy again soon. |
| **recent heavy buyers** ("hot") | F ≥ 5 **and** R ≤ 13 | The best customers: frequent and still active. |
| **dormant one-timers** ("cold") | F = 1 **and** R > 52 (silent for over a year). **On CDNOW: F = 1 and R between 14 and 38**, because its calibration is only 39 weeks and all its one-timers bought in the first 13 weeks | The worst customers: one purchase, long ago. |
| **repeat buyers by recency band** | F 2–4 or F ≥ 5, crossed with R ≤ 13, 14–52, > 52 | Holds frequency roughly fixed, so any difference in forecast between bands is the recency effect. |

### The metrics

- **actual tx / customer**: the group's total holdout transactions divided by its number
  of customers.
- **predicted tx / customer**: the same with the model's forecast holdout total, averaged
  over runs.
- **pred ÷ act**: the group's predicted total over its actual total. 1.00× is right on;
  2× is double; 0.5× is half. Each run gives one ratio. The cell shows the mean over runs
  with its 95% percentile-bootstrap interval.
- **Separation** (§5): predicted tx per customer in the hot group divided by the same for
  the cold group.
- **AUC** and **ρ** within a group: how well the forecast orders customers *inside* the
  group. AUC is whether the forecast tells holdout buyers from non-buyers (0.5 = chance);
  ρ is Spearman of predicted against actual holdout totals.
- Per-customer **MAE** is not used. On groups that are mostly zeros it rewards
  forecasting low. The error measure is the **RMSE of per-customer holdout totals**,
  whose optimum is the mean.

Comparative claims (§8) follow `docs/statistical-protocol.md`: Δ of means with a 95%
percentile-bootstrap interval from `evaluation.effects.effect`, resampled independently.
**Bold** marks an interval that excludes 0. Pareto/NBD's 20 seeded fits barely differ, so
its intervals are tiny; "supported" here means robust to the randomness of neural training
and search.

## 2. Summary

1. **One-timers are over-forecast by both LSTMs on every window**, by 1.3× to 4.3×.
   Pareto/NBD under-forecasts them (0.1×–1.15×). On CDNOW and multichannel they supply
   most of the LSTMs' positive aggregate bias. (§3)
2. **Recent and recent heavy buyers are under-forecast** on CDNOW, gift and multichannel:
   0.16×–0.78×, mostly below Pareto/NBD. **Not on electronic_5y**, where both LSTMs are
   within 10% of the actual. On the old electronics panel Pareto/NBD under-forecasts them
   even more. (§4)
3. **The gap between the best and worst customers is squeezed.** Actual: 4–21×.
   `bounded32`: 1.9–7.9×. ValendinLSTM: 1.0× on electronics 2y and multichannel 2y, so
   there it gives every customer the same forecast. (§5)
4. **Recency is used inconsistently.** ValendinLSTM ignores it where it collapses.
   `bounded32` *reverses* it on CDNOW. It shrinks the recency gradient on gift 2y and
   multichannel 2y, and tracks it well on gift 3y and electronic_5y. (§6)
5. **Panel pattern:** the squeeze is worst with a short calibration window and with a
   high share of one-timers. On the same panel, **going from 2 to 3 years of calibration
   lifts the separation captured** on all three panels (supported). With four panels,
   sparsity, one-timer share and purchases per customer cannot be told apart. (§7)
6. **The customer groups to report** are frequency, recency within frequency, overdue as a
   reading of recency, momentum, and Valendin et al.'s groups. Clumpiness and regularity
   were dropped: our histories are too short for them, by the papers' own requirements.
   (§11)
7. **Momentum (rising vs declining buyers) is the one buying pattern with information
   beyond frequency and recency.** Pareto/NBD cannot use it, and the LSTMs use it only with
   five years of history. Frequent buyers who have gone quiet are over-forecast by both
   model families. (§11.3, §11.4)
8. **Valendin et al.'s At Risk result replicates on CDNOW, gift and multichannel, not on
   electronics.**
   - The LSTMs forecast fewer purchases than Pareto/NBD for frequent customers who do not
     come back. The gap is widest for the most frequent: multichannel 5+, 0.25 against
     0.99 per customer.
   - Customers who grow (Opportunity) are missed by every model: 2–34% of what they buy.
   (§11.5)
9. **The Transformer squeezes customers like the LSTM** on the two windows where it ran.
   On electronic_5y it over-forecasts long-silent and declining customers more than the
   LSTM does: overdue > 3× 1.37× against 1.09×, declining 1.28× against 1.08×, 5+
   purchases silent > 52 weeks 1.39× against 1.01×. (Appendix)

## 3. One-timers are over-forecast

Each cell: predicted tx / customer → **pred ÷ act** [95% CI over runs].

| panel | n customers (% of cohort) | actual tx / customer | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---:|---:|---|---|---|
| cdnow 2y | 1417 (60%) | 0.24 | 0.37 → **1.50×** [1.32, 1.71] | 1.04 → **4.25×** [3.87, 4.61] | 0.17 → **0.68×** [0.67, 0.69] |
| electronics 2y | 150 (18%) | 0.82 | 2.58 → **3.14×** [2.99, 3.27] | 1.40 → **1.71×** [1.66, 1.75] | 0.15 → **0.18×** [0.18, 0.18] |
| electronics 3y | 111 (13%) | 1.30 | not run | 0.93 → **0.72×** [0.69, 0.75] | 0.12 → **0.10×** [0.10, 0.10] |
| electronic_5y | 861 (23%) | 0.15 | 0.24 → **1.62×** [1.52, 1.72] | 0.27 → **1.86×** [1.77, 1.94] | 0.06 → **0.43×** [0.43, 0.43] |
| gift 2y | 1058 (51%) | 0.23 | 0.31 → **1.34×** [1.24, 1.42] | 0.37 → **1.59×** [1.53, 1.65] | 0.22 → **0.94×** [0.94, 0.95] |
| gift 3y | 891 (43%) | 0.15 | not run | 0.24 → **1.63×** [1.57, 1.69] | 0.13 → **0.92×** [0.91, 0.92] |
| multichannel 2y | 1028 (73%) | 0.09 | 0.27 → **2.87×** [2.48, 3.28] | 0.18 → **1.87×** [1.82, 1.93] | 0.08 → **0.86×** [0.85, 0.87] |
| multichannel 3y | 951 (68%) | 0.05 | not run | 0.11 → **2.37×** [2.28, 2.45] | 0.06 → **1.15×** [1.14, 1.16] |

How to read a row, with CDNOW: its 1,417 one-timers made 0.24 holdout transactions each
on average. `bounded32` forecast 1.04 each, 4.25 times too many. Pareto/NBD forecast 0.17
(0.68×).

- **Electronics 2y and 3y are the exception.** One-timers there buy 0.82 and 1.30 per
  customer in the holdout. That is far more than on any other panel, and Pareto/NBD
  misses it badly (0.18× and 0.10×). Only there is an LSTM below 1× (`bounded32` 3y,
  0.72×).
- **Ordering customers inside this group is hopeless for every model.** Within one-timers
  ρ ≈ 0 and AUC ≈ 0.50 for all three. A single purchase gives nothing to rank by. The
  failure here is the **level**, not the ordering.

**Where the aggregate bias comes from.** Contribution of each group to the total bias, in
percentage points of the panel's actual holdout total. The three groups add to the total.

| panel | model | total bias | one-timers | F 2–4 | F ≥ 5 |
|---|---|---:|---:|---:|---:|
| cdnow 2y | ValendinLSTM | −24 | +9 | −16 | −17 |
| cdnow 2y | LSTM + bounded32 | +37 | **+59** | −4 | −18 |
| cdnow 2y | Pareto/NBD | −14 | −6 | −3 | −5 |
| electronics 2y | ValendinLSTM | +46 | +18 | +30 | −2 |
| electronics 2y | LSTM + bounded32 | −2 | +6 | +7 | −15 |
| electronics 2y | Pareto/NBD | −63 | −7 | −17 | −39 |
| electronics 3y | LSTM + bounded32 | −13 | −3 | +6 | −16 |
| electronics 3y | Pareto/NBD | −65 | −8 | −7 | −50 |
| electronic_5y | ValendinLSTM | +14 | +4 | +7 | +4 |
| electronic_5y | LSTM + bounded52 | +9 | +5 | +6 | −2 |
| electronic_5y | Pareto/NBD | −17 | −3 | −3 | −10 |
| gift 2y | ValendinLSTM | −16 | +7 | −15 | −8 |
| gift 2y | LSTM + bounded32 | −19 | +13 | −18 | −13 |
| gift 2y | Pareto/NBD | −11 | −1 | −6 | −4 |
| gift 3y | LSTM + bounded32 | −5 | +8 | −1 | −12 |
| gift 3y | Pareto/NBD | −8 | −1 | +0 | −7 |
| multichannel 2y | ValendinLSTM | +67 | **+80** | −1 | −12 |
| multichannel 2y | LSTM + bounded32 | +17 | **+37** | −8 | −12 |
| multichannel 2y | Pareto/NBD | +6 | −6 | +15 | −2 |
| multichannel 3y | LSTM + bounded32 | +33 | **+36** | +10 | −13 |
| multichannel 3y | Pareto/NBD | +23 | +4 | +20 | −1 |

A small total bias is often cancellation. On gift 3y, `bounded32`'s −5% is +8 from
one-timers against −12 from heavy buyers. This extends §9 of the insights doc ("a good
bias can be cancellation") from weeks to customers.

## 4. Recent and recent heavy buyers are under-forecast

### Recent buyers: last purchase 0–4 weeks before the holdout

These customers bought in the last five calibration weeks, whatever their frequency.

| panel | n customers (% of cohort) | actual tx / customer | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---:|---:|---|---|---|
| cdnow 2y | 194 (8%) | 3.71 | 2.24 → **0.61×** [0.53, 0.69] | 1.87 → **0.50×** [0.49, 0.52] | 3.19 → **0.86×** [0.85, 0.87] |
| electronics 2y | 99 (12%) | 4.33 | 2.63 → **0.61×** [0.58, 0.63] | 2.92 → **0.67×** [0.64, 0.71] | 1.65 → **0.38×** [0.38, 0.38] |
| electronics 3y | 91 (11%) | 4.47 | not run | 3.52 → **0.79×** [0.75, 0.82] | 1.39 → **0.31×** [0.31, 0.31] |
| electronic_5y | 237 (6%) | 2.22 | 2.36 → **1.07×** [1.03, 1.10] | 2.08 → **0.94×** [0.90, 0.97] | 1.59 → **0.72×** [0.72, 0.72] |
| gift 2y | 65 (3%) | 2.20 | 1.07 → **0.49×** [0.45, 0.52] | 0.74 → **0.34×** [0.32, 0.35] | 1.19 → **0.54×** [0.54, 0.54] |
| gift 3y | 65 (3%) | 2.48 | not run | 1.31 → **0.53×** [0.51, 0.54] | 1.41 → **0.57×** [0.57, 0.57] |
| multichannel 2y | 48 (3%) | 0.58 | 0.27 → **0.47×** [0.40, 0.54] | 0.33 → **0.56×** [0.53, 0.59] | 0.58 → **1.00×** [0.99, 1.00] |
| multichannel 3y | 41 (3%) | 0.61 | not run | 0.47 → **0.77×** [0.74, 0.81] | 0.64 → **1.05×** [1.04, 1.05] |

### Recent heavy buyers: F ≥ 5 and last purchase ≤ 13 weeks before the holdout

The narrower "best customer" group: at least five calibration purchases and still active
in the last quarter of calibration.

| panel | n customers (% of cohort) | actual tx / customer | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---:|---:|---|---|---|
| cdnow 2y | 157 (7%) | 4.65 | 2.76 → **0.59×** [0.52, 0.67] | 2.45 → **0.53×** [0.50, 0.55] | 4.17 → **0.90×** [0.89, 0.91] |
| electronics 2y | 110 (13%) | 4.59 | 2.62 → **0.57×** [0.55, 0.59] | 2.91 → **0.63×** [0.60, 0.67] | 1.81 → **0.39×** [0.39, 0.40] |
| electronics 3y | 119 (14%) | 4.82 | not run | 3.57 → **0.74×** [0.71, 0.77] | 1.50 → **0.31×** [0.31, 0.31] |
| electronic_5y | 339 (9%) | 2.34 | 2.38 → **1.02×** [0.98, 1.06] | 2.13 → **0.91×** [0.88, 0.94] | 1.77 → **0.76×** [0.76, 0.76] |
| gift 2y | 93 (5%) | 2.35 | 1.39 → **0.59×** [0.54, 0.63] | 0.91 → **0.39×** [0.37, 0.40] | 1.68 → **0.71×** [0.71, 0.72] |
| gift 3y | 171 (8%) | 2.16 | not run | 1.43 → **0.66×** [0.65, 0.68] | 1.52 → **0.70×** [0.70, 0.70] |
| multichannel 2y | 15 (1%) | 1.73 | 0.27 → **0.16×** [0.14, 0.18] | 0.33 → **0.19×** [0.18, 0.21] | 1.29 → **0.75×** [0.74, 0.75] |
| multichannel 3y | 20 (1%) | 1.00 | not run | 0.61 → **0.61×** [0.58, 0.64] | 1.20 → **1.20×** [1.19, 1.21] |

Reading across the panels:

- **CDNOW, gift and multichannel:** both LSTMs give the best customers about half their
  actual or less, and less than Pareto/NBD does. Multichannel 2y is extreme: 15 customers
  who buy 1.73 times each are forecast 0.27–0.33, the same as everyone else.
- **electronic_5y is the exception:** both LSTMs land within 10% (0.91×–1.07×) and beat
  Pareto/NBD (0.72×–0.76×). It has the longest calibration (260 weeks).
- **Electronics 2y/3y:** the LSTMs under-forecast (0.57×–0.79×), but Pareto/NBD
  under-forecasts more (0.31×–0.39×). There it misses the whole panel's level (−63%).
- Under-forecasting the best customers is partly unavoidable: their holdout is
  regression to the mean. Pareto/NBD sets the bar for what calibration data allow.

**The forecast also fades too fast.** Forecast for the last quarter of the holdout
divided by forecast for the first quarter, recent heavy buyers. Below the actual means
the model expects them to drop off faster than they do.

| panel | actual | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---:|---:|---:|---:|
| cdnow 2y | 0.61 | 0.34 | 0.34 | 0.76 |
| electronics 2y | 1.25 | 0.79 | 0.53 | 0.92 |
| electronics 3y | 2.30 | not run | 0.88 | 0.95 |
| electronic_5y | 0.62 | 0.64 | 0.60 | 0.93 |
| gift 2y | 1.25 | 1.29 | 1.37 | 0.95 |
| gift 3y | 2.01 | not run | 1.41 | 0.95 |
| multichannel 2y | 0.67 | 0.64 | 0.53 | 0.91 |
| multichannel 3y | 1.67 | not run | 0.73 | 0.93 |

And the mirror image for one-timers, whose forecast *rises* through the holdout on CDNOW
and gift:

| panel | actual | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---:|---:|---:|---:|
| cdnow 2y | 0.80 | 1.27 | **3.61** | 0.76 |
| electronics 2y | 0.74 | 0.84 | 1.23 | 0.92 |
| electronics 3y | 0.89 | not run | 1.08 | 0.95 |
| electronic_5y | 0.57 | 0.55 | 0.52 | 0.93 |
| gift 2y | 1.50 | 2.29 | 2.95 | 0.95 |
| gift 3y | 2.00 | not run | 1.85 | 0.95 |
| multichannel 2y | 2.29 | 0.67 | 1.24 | 0.91 |
| multichannel 3y | 2.50 | not run | 1.35 | 0.93 |

Quarterly actuals in small groups are noisy, so read these two tables for direction only.
On CDNOW, electronics 2y and multichannel 3y, the simulation pulls the two extremes
toward each other as it runs: hot customers cool faster than they do, and cold ones warm
up when they do not. This fits the mechanism measured in `docs/absorbing-death-state.md`
§4.3. A customer's identity travels through the simulation in the recency features, and
feeding back sampled purchases erodes it.

## 5. Separation: how far apart the best and worst customers are put

This table compares **two specific groups** from §1 on each window:

- **hot:** recent heavy buyers, F ≥ 5 and R ≤ 13;
- **cold:** dormant one-timers, F = 1 and R > 52 (on CDNOW F = 1 and R 14–38, its
  longest-silent one-timers).

For each, it gives predicted (or actual) transactions **per customer**, and divides hot by
cold. The result says how many times more a hot customer is expected to buy than a cold
one.

| panel | cold group (n) | hot group (n) | **actual**: hot / cold | ValendinLSTM: hot / cold | LSTM + bounded32: hot / cold | Pareto/NBD: hot / cold |
|---|---|---|---|---|---|---|
| cdnow 2y | F1, R 14–38 (1417) | F5+, R ≤ 13 (157) | 4.65 / 0.24 = **19.0×** | 2.76 / 0.37 = **7.9×** | 2.45 / 1.04 = **3.1×** | 4.17 / 0.17 = **25.0×** |
| electronics 2y | F1, R > 52 (150) | F5+, R ≤ 13 (110) | 4.59 / 0.82 = **5.6×** | 2.62 / 2.58 = **1.0×** | 2.91 / 1.40 = **2.1×** | 1.81 / 0.15 = **12.4×** |
| electronics 3y | F1, R > 52 (111) | F5+, R ≤ 13 (119) | 4.82 / 1.30 = **3.7×** | not run | 3.57 / 0.93 = **3.9×** | 1.50 / 0.12 = **12.0×** |
| electronic_5y | F1, R > 52 (861) | F5+, R ≤ 13 (339) | 2.34 / 0.15 = **16.0×** | 2.38 / 0.24 = **10.2×** | 2.13 / 0.27 = **7.9×** | 1.77 / 0.06 = **28.4×** |
| gift 2y | F1, R > 52 (1058) | F5+, R ≤ 13 (93) | 2.35 / 0.23 = **10.3×** | 1.39 / 0.31 = **4.6×** | 0.91 / 0.37 = **2.6×** | 1.68 / 0.22 = **7.8×** |
| gift 3y | F1, R > 52 (891) | F5+, R ≤ 13 (171) | 2.16 / 0.15 = **14.8×** | not run | 1.43 / 0.24 = **6.2×** | 1.52 / 0.13 = **11.4×** |
| multichannel 2y | F1, R > 52 (1028) | F5+, R ≤ 13 (15) | 1.73 / 0.09 = **18.4×** | 0.27 / 0.27 = **1.0×** | 0.33 / 0.18 = **1.9×** | 1.29 / 0.08 = **15.9×** |
| multichannel 3y | F1, R > 52 (951) | F5+, R ≤ 13 (20) | 1.00 / 0.05 = **20.7×** | not run | 0.61 / 0.11 = **5.5×** | 1.20 / 0.06 = **21.6×** |

How to read a row, with multichannel 2y: its 15 hot customers made 1.73 holdout
transactions each and its 1,028 cold customers 0.09, so a hot customer bought 18.4 times
as much. Pareto/NBD puts them 15.9× apart. `bounded32` puts them only 1.9× apart (0.33
against 0.18). ValendinLSTM forecasts 0.27 for both: it cannot tell the panel's best
customers from its worst.

- The ideal is the actual column. Too low means the model squeezes customers together;
  too high means it exaggerates the difference.
- **Every LSTM row is below the actual except `bounded32` on electronics 3y** (3.9×
  against 3.7×). Pareto/NBD over-separates on electronics and electronic_5y, because it
  under-forecasts dormant customers who do come back.
- The two groups are 1–14% (hot) and 13–73% (cold) of each panel. Together they
  bracket the range of customers; everyone else lies between them.

## 6. Is recency taken into account?

**The test.** Hold frequency roughly fixed and vary only recency. A model that uses
recency should forecast less for each longer-silent band, as the actuals do. One that
ignores it gives the same number in every band. Predicted tx per customer:

**Repeat buyers with 2–4 calibration purchases**

| panel | recency band | n | **actual** | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---|---:|---:|---:|---:|---:|
| cdnow 2y | R ≤ 13 | 299 | 1.40 | 0.83 | **0.71** | 1.46 |
| cdnow 2y | R 14–38 | 448 | 0.76 | 0.47 | **1.06** | 0.58 |
| electronics 2y | R ≤ 13 | 27 | 2.37 | 2.61 | 2.17 | 0.89 |
| electronics 2y | R 14–52 | 48 | 0.96 | 2.58 | 1.62 | 0.75 |
| electronics 2y | R > 52 | 234 | 1.07 | 2.58 | 1.43 | 0.22 |
| electronics 3y | R ≤ 13 | 12 | 0.67 | not run | 1.89 | 0.66 |
| electronics 3y | R 14–52 | 43 | 0.67 | not run | 1.25 | 0.58 |
| electronics 3y | R > 52 | 204 | 0.72 | not run | 0.97 | 0.21 |
| electronic_5y | R ≤ 13 | 100 | 0.58 | 0.90 | 0.81 | 0.55 |
| electronic_5y | R 14–52 | 254 | 0.50 | 0.58 | 0.52 | 0.48 |
| electronic_5y | R > 52 | 1287 | 0.25 | 0.32 | 0.33 | 0.20 |
| gift 2y | R ≤ 13 | 277 | 1.17 | 0.70 | 0.57 | 0.79 |
| gift 2y | R 14–52 | 255 | 0.76 | 0.53 | 0.52 | 0.69 |
| gift 2y | R > 52 | 336 | 0.37 | 0.41 | 0.42 | 0.52 |
| gift 3y | R ≤ 13 | 184 | 0.83 | not run | 0.84 | 0.66 |
| gift 3y | R 14–52 | 193 | 0.61 | not run | 0.53 | 0.58 |
| gift 3y | R > 52 | 490 | 0.30 | not run | 0.31 | 0.38 |
| multichannel 2y | R ≤ 13 | 72 | 0.39 | 0.27 | 0.30 | 0.52 |
| multichannel 2y | R 14–52 | 111 | 0.30 | 0.27 | 0.21 | 0.44 |
| multichannel 2y | R > 52 | 165 | 0.21 | 0.27 | 0.19 | 0.26 |
| multichannel 3y | R ≤ 13 | 57 | 0.25 | not run | 0.41 | 0.45 |
| multichannel 3y | R 14–52 | 71 | 0.34 | not run | 0.29 | 0.38 |
| multichannel 3y | R > 52 | 276 | 0.15 | not run | 0.19 | 0.22 |

**Repeat buyers with 5 or more calibration purchases**

| panel | recency band | n | **actual** | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---|---:|---:|---:|---:|---:|
| cdnow 2y | R ≤ 13 | 157 | 4.65 | 2.76 | 2.45 | 4.17 |
| cdnow 2y | R 14–38 | 36 | 1.72 | 0.90 | 1.61 | 1.11 |
| electronics 2y | R ≤ 13 | 110 | 4.59 | 2.62 | 2.91 | 1.81 |
| electronics 2y | R 14–52 | 133 | 2.35 | 2.58 | 1.90 | 1.23 |
| electronics 2y | R > 52 | 127 | 1.31 | 2.58 | 1.50 | 0.36 |
| electronics 3y | R ≤ 13 | 119 | 4.82 | not run | 3.57 | 1.50 |
| electronics 3y | R 14–52 | 161 | 2.91 | not run | 2.11 | 1.22 |
| electronics 3y | R > 52 | 179 | 0.96 | not run | 1.12 | 0.40 |
| electronic_5y | R ≤ 13 | 339 | 2.34 | 2.38 | 2.13 | 1.77 |
| electronic_5y | R 14–52 | 420 | 1.10 | 1.23 | 1.16 | 1.20 |
| electronic_5y | R > 52 | 494 | 0.54 | 0.56 | 0.54 | 0.42 |
| gift 2y | R ≤ 13 | 93 | 2.35 | 1.39 | 0.91 | 1.68 |
| gift 2y | R 14–52 | 34 | 1.18 | 1.16 | 0.80 | 1.51 |
| gift 2y | R > 52 | 9 | 0.44 | 0.86 | 0.61 | 1.03 |
| gift 3y | R ≤ 13 | 171 | 2.16 | not run | 1.43 | 1.52 |
| gift 3y | R 14–52 | 94 | 1.14 | not run | 1.04 | 1.33 |
| gift 3y | R > 52 | 39 | 0.36 | not run | 0.51 | 0.79 |
| multichannel 2y | R ≤ 13 | 15 | 1.73 | 0.27 | 0.33 | 1.29 |
| multichannel 2y | R 14–52 | 7 | 1.14 | 0.27 | 0.25 | 1.07 |
| multichannel 2y | R > 52 | 4 | 0.25 | 0.27 | 0.21 | 0.69 |
| multichannel 3y | R ≤ 13 | 20 | 1.00 | not run | 0.61 | 1.20 |
| multichannel 3y | R 14–52 | 16 | 1.38 | not run | 0.49 | 0.94 |
| multichannel 3y | R > 52 | 11 | 0.45 | not run | 0.39 | 0.52 |

Bands with under about 20 customers (multichannel F ≥ 5, gift F ≥ 5 R > 52) are too
small to read on their own.

**Four behaviours, depending on model and panel:**

| behaviour | where | evidence |
|---|---|---|
| **recency ignored**: same forecast in every band | ValendinLSTM on electronics 2y and multichannel 2y | 2.61 / 2.58 / 2.58 and 0.27 / 0.27 / 0.27 |
| **recency reversed**: longer-silent customers get *more* | `bounded32` on CDNOW | F 2–4: 0.71 for R ≤ 13 against 1.06 for R 14–38, where the actual is 1.40 against 0.76 |
| **recency used but flattened**: right direction, too small a gap | `bounded32` on gift 2y, multichannel 2y, electronics 2y | gift 2y F 2–4: 0.57 → 0.42 (1.4×) where the actual is 1.17 → 0.37 (3.2×) |
| **recency used well** | `bounded32` on gift 3y and electronic_5y; ValendinLSTM on electronic_5y | gift 3y F 2–4: 0.84 / 0.53 / 0.31 against 0.83 / 0.61 / 0.30 |

So "recency isn't taken into account" is right for the collapsed ValendinLSTM runs. For
`bounded32` on CDNOW it is stronger than that: the model uses recency the wrong way
round. That is worse than ignoring it, and it shows inside the group:

| F 2–4 customers: AUC (holdout buyer vs non-buyer) / Spearman | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---|---|---|
| cdnow 2y | 0.60 / 0.18 | **0.43 / −0.12** | 0.63 / 0.23 |
| electronics 2y | 0.50 / 0.00 | 0.54 / 0.07 | 0.56 / 0.10 |
| electronics 3y | not run | 0.54 / 0.04 | 0.55 / 0.05 |
| electronic_5y | 0.60 / 0.13 | 0.60 / 0.13 | 0.60 / 0.14 |
| gift 2y | 0.60 / 0.18 | 0.57 / 0.13 | 0.58 / 0.15 |
| gift 3y | not run | 0.61 / 0.19 | 0.62 / 0.19 |
| multichannel 2y | 0.52 / 0.02 | 0.52 / 0.03 | 0.56 / 0.09 |
| multichannel 3y | not run | 0.58 / 0.12 | 0.59 / 0.12 |

An AUC of 0.43 means that, among CDNOW's 2–4-purchase customers, `bounded32` ranks the
ones who will *not* buy above the ones who will more often than chance.

**What the CDNOW forecast does week by week.** Mean weekly forecast for CDNOW's 1,417
one-timers (median over the 100 `bounded32` runs, ×1000):

- `bounded32`: 8 in the first holdout week, rising steadily to **38** about 28 weeks in
  (spring 1998), then falling to 23.
- ValendinLSTM: flat at 7–12 throughout.
- Actual: about 6 per week, with no trend.

77 of the 100 `bounded32` runs over-forecast this group by more than 2×. So it is
systematic, not a few bad runs.

**A likely cause (not yet isolated).** `bounded32` sees recency only through flags:
"bought in the last 2, 4, 8, 16 and 32 weeks". After 32 silent weeks every flag reads 0,
and a customer silent for 33 weeks looks the same as one silent for two years. CDNOW
calibrates on only 39 weeks, and its whole cohort made its first purchase in the first 13.
So "all flags 0 but has bought before" barely occurs in training: only in the last few
calibration weeks, for customers who bought in the first few. The holdout is full of that
state, and the forecast there extrapolates. The holdout also reaches calendar weeks
(`week_sin`, `week_cos`) that a 39-week calibration never covered. Separating the two
needs the ablation in §10, item 5.

## 7. Panel patterns

The question: under which panel patterns does the LSTM lose the ability to tell customer
groups apart?

**Separation captured** = the model's hot/cold separation (§5) divided by the actual one.
100% is ideal; below it the model squeezes customers together; above it the model
exaggerates.

| panel | customers | calibration weeks | zero cells, calibration | tx / customer, calibration | one-timers | ≥ 5 tx | silent > 52 w at cut-off | actual separation | separation captured: V / B / P | Spearman: V / B / P | forecast CV: V / B / P |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| cdnow 2y | 2357 | 39 | 95.0% | 2.03 | 60% | 8% | 0% | 19.0× | 41% / **16%** / 131% | 0.40 / 0.09 / 0.45 | 1.34 / 0.66 / 1.71 |
| electronics 2y | 829 | 104 | 97.5% | 5.65 | 18% | 45% | 62% | 5.6× | **18%** / 37% / 221% | 0.03 / 0.26 / 0.31 | 0.05 / 0.36 / 1.11 |
| electronics 3y | 829 | 156 | 97.9% | 7.32 | 13% | 55% | 60% | 3.7× | – / 104% / 324% | – / 0.32 / 0.33 | – / 0.73 / 1.10 |
| electronic_5y | 3755 | 260 | 98.4% | 4.55 | 23% | 33% | 70% | 16.0× | 64% / 50% / 177% | 0.41 / 0.40 / 0.40 | 1.27 / 1.14 / 1.40 |
| gift 2y | 2062 | 104 | 98.1% | 2.04 | 51% | 7% | 68% | 10.3× | 45% / 25% / 76% | 0.37 / 0.33 / 0.38 | 0.64 / 0.37 / 0.83 |
| gift 3y | 2062 | 156 | 98.4% | 2.60 | 43% | 15% | 69% | 14.8× | – / 42% / 77% | – / 0.42 / 0.44 | – / 0.87 / 1.06 |
| multichannel 2y | 1402 | 104 | 98.6% | 1.44 | **73%** | 2% | 85% | 18.4× | **6%** / **10%** / 87% | 0.00 / 0.10 / 0.19 | 0.09 / 0.23 / 1.21 |
| multichannel 3y | 1402 | 156 | 99.0% | 1.60 | 68% | 3% | 88% | 20.7× | – / 26% / 105% | – / 0.24 / 0.25 | – / 0.69 / 1.41 |

V = ValendinLSTM, B = LSTM + `bounded32` (`bounded52` on electronic_5y), P = Pareto/NBD.
Forecast CV is the spread of per-customer forecasts (`insights-real-panels.md` §3).

**Calibration length: the one pattern that can be tested on a fixed panel.** Same panel,
same model, 2 vs 3 years of calibration (independent runs):

| panel | captured, 2y | captured, 3y | Δ log2 captured (3y − 2y) [95% CI] |
|---|---:|---:|---|
| electronics | 37% | 104% | **+1.52 [+1.43, +1.60]** |
| gift | 25% | 42% | **+0.76 [+0.67, +0.85]** |
| multichannel | 10% | 26% | **+1.39 [+1.26, +1.52]** |

On all three panels, a longer calibration window lets `bounded32` reproduce more of the
actual difference between the best and worst customers. Caveat: the 3-year forecasts
score a later holdout year, so the actual separation also changes. Normalising by it is
what makes the rows comparable.

**Across panels, descriptively only:**

- **Short calibration** (CDNOW, 39 weeks): `bounded32` captures 16% and reverses
  recency (§6), although CDNOW is the least sparse panel (95% zeros).
- **Mostly one-timers** (multichannel, 68–73%, about 1.5 transactions per customer): the
  worst separation on 2 years (6% and 10%) and the weakest gains from 3 years.
- **Long calibration with many repeat buyers** (electronic_5y, gift 3y): the best LSTM
  separation (42–64%). Spearman is level with Pareto/NBD's.
- **Zero-cell share alone does not order the panels.** Electronic_5y (98.4%) does better
  than CDNOW (95.0%).

**What cannot be said yet.** There are four panels, and their properties move together:
multichannel is at once the sparsest, the one with the most one-timers and the one with
the fewest purchases per customer. With four points the effect of each property cannot
be told apart. That needs either the synthetic grid, where one property can be varied at
a time, or subsampling a real panel. For example: dropping one-timers from gift to raise
its repeat share, or truncating electronic_5y's calibration to 2 and 3 years. Both keep
the customers' behaviour and change one property.

## 8. Every contrast against Pareto/NBD

Δ = model − Pareto/NBD, 95% percentile bootstrap, independent (n runs vs 20 fits).

- **Separation** and **pred ÷ act** are compared on a log2 scale, so +1 means twice
  Pareto/NBD's ratio and −1 means half of it.
- **RMSE** is of per-customer holdout totals inside the group. Positive means the model
  is worse.

| panel | model (n) | log2 separation | log2 pred÷act, one-timers | log2 pred÷act, recent buyers | RMSE, recent buyers | RMSE, one-timers |
|---|---|---|---|---|---|---|
| cdnow 2y | ValendinLSTM (20) | **−1.73 [−1.91, −1.55]** | **+1.08 [+0.90, +1.26]** | **−0.56 [−0.73, −0.38]** | **+0.44 [+0.31, +0.57]** | **+0.01 [+0.01, +0.03]** |
| cdnow 2y | LSTM + bounded32 (100) | **−3.25 [−3.40, −3.09]** | **+2.45 [+2.28, +2.60]** | **−0.80 [−0.86, −0.74]** | **+0.47 [+0.42, +0.51]** | **+0.40 [+0.34, +0.45]** |
| electronics 2y | ValendinLSTM (20) | **−3.60 [−3.62, −3.59]** | **+4.13 [+4.05, +4.19]** | **+0.67 [+0.60, +0.73]** | **−0.17 [−0.20, −0.13]** | **+0.55 [+0.47, +0.61]** |
| electronics 2y | LSTM + bounded32 (100) | **−2.62 [−2.69, −2.54]** | **+3.24 [+3.20, +3.29]** | **+0.77 [+0.70, +0.85]** | **−0.33 [−0.38, −0.29]** | **−0.02 [−0.03, −0.01]** |
| electronics 3y | LSTM + bounded32 (100) | **−1.65 [−1.70, −1.60]** | **+2.86 [+2.80, +2.93]** | **+1.31 [+1.25, +1.37]** | **−0.80 [−0.83, −0.77]** | **−0.20 [−0.20, −0.19]** |
| electronic_5y | ValendinLSTM (20) | **−1.48 [−1.56, −1.41]** | **+1.91 [+1.82, +2.00]** | **+0.56 [+0.52, +0.62]** | **−0.31 [−0.34, −0.28]** | **+0.00 [+0.00, +0.01]** |
| electronic_5y | LSTM + bounded52 (20) | **−1.85 [−1.92, −1.78]** | **+2.11 [+2.04, +2.18]** | **+0.38 [+0.33, +0.43]** | **−0.20 [−0.26, −0.14]** | **+0.01 [+0.01, +0.01]** |
| gift 2y | ValendinLSTM (20) | **−0.79 [−0.91, −0.68]** | **+0.49 [+0.39, +0.58]** | **−0.18 [−0.30, −0.07]** | **+0.12 [+0.09, +0.15]** | **+0.01 [+0.00, +0.01]** |
| gift 2y | LSTM + bounded32 (100) | **−1.64 [−1.72, −1.57]** | **+0.73 [+0.67, +0.78]** | **−0.71 [−0.76, −0.66]** | **+0.31 [+0.30, +0.33]** | **+0.02 [+0.01, +0.02]** |
| gift 3y | LSTM + bounded32 (100) | **−0.90 [−0.95, −0.85]** | **+0.81 [+0.76, +0.86]** | **−0.12 [−0.16, −0.08]** | **+0.51 [+0.48, +0.53]** | **+0.01 [+0.01, +0.01]** |
| multichannel 2y | ValendinLSTM (20) | **−3.98 [−4.00, −3.96]** | **+1.66 [+1.45, +1.87]** | **−1.16 [−1.38, −0.95]** | **+0.08 [+0.07, +0.09]** | **+0.05 [+0.03, +0.07]** |
| multichannel 2y | LSTM + bounded32 (100) | **−3.17 [−3.26, −3.07]** | **+1.11 [+1.06, +1.15]** | **−0.90 [−0.99, −0.82]** | **+0.06 [+0.06, +0.06]** | **+0.01 [+0.01, +0.01]** |
| multichannel 3y | LSTM + bounded32 (100) | **−2.05 [−2.14, −1.96]** | **+1.02 [+0.97, +1.07]** | **−0.48 [−0.55, −0.41]** | **+0.12 [+0.11, +0.12]** | **+0.01 [+0.01, +0.01]** |

- Lower separation and higher one-timer level than Pareto/NBD are supported on every
  window, for both models.
- On electronics and electronic_5y the LSTMs are **better** than Pareto/NBD on recent
  buyers (negative RMSE Δ), because Pareto/NBD under-forecasts them more.
- On one-timers the RMSE differences are tiny everywhere except CDNOW `bounded32`
  (+0.40). Over-forecasting a group with near-zero outcomes costs little per customer,
  but it adds up in the aggregate bias (§3).

## 9. What is not an LSTM failure

- **Customers who grow:** those who bought more in the holdout than in calibration (the
  "Opportunity" group of `evaluation.segment_analysis`). Every model forecasts 0.02×–0.34×
  of what they do. Nothing in their history signals it.
- **The holdout's heaviest buyers**, picked by holdout outcome, are under-forecast by every
  model. That is regression to the mean.
- **Ordering one-timers among themselves.** ρ ≈ 0 for all three models.
- **The class cap on the old electronics panel.** The model cannot sample more than 6
  purchases a week, and 80 of 1,467 (2y) and 120 of 1,541 (3y) holdout transactions sit
  above that in some customer-week. Its F ≥ 10 group is forecast at 0.64×–0.72×. No other
  panel exceeds its cap in the holdout.

## 10. Enhancements to try

Ordered by how directly the evidence above points to them.

1. **A fixed customer summary that the simulation cannot overwrite.** Give each customer
   frozen channels computed from calibration only, carried unchanged through the holdout
   like the behavioural cluster. Candidates: R, F, tenure and mean gap between purchases,
   or Pareto/NBD's per-customer rate and P(alive).
   - Targets §4–§6: the squeeze and the flattened or reversed recency.
   - The AR channel carries 86–93% of a customer's identity, and the simulation corrupts
     it (`absorbing-death-state.md` §4.3). A frozen channel cannot be corrupted.
   - `kmeans_8` already shows a persistent channel undoes the collapse. A continuous
     version avoids the information lost by binning, which is a likely reason it lowered
     ranking on electronic_5y.
   - Success: separation captured (§7) rising toward 100% and recent buyers moving toward
     1×.
2. **A learned per-customer hazard of being gone**, as designed in
   `absorbing-death-state.md` §8.
   - Targets §3: one-timers supply most of the positive bias on CDNOW and multichannel.
   - It must be learned per panel, not a fixed rule: electronics one-timers really do come
     back (0.82–1.30 holdout tx each). That is why the crude latch destroyed the level
     there.
   - It does nothing for under-forecast active buyers, so it complements item 1.
3. **A recency encoding that keeps distinguishing long silences.**
   - Targets §6: `bounded32` cannot tell 33 weeks from 2 years, and reverses recency on
     CDNOW.
   - Use `ar_saturating` (owed on gift and multichannel, `insights-real-panels.md` §11), or
     `log` / `ratio`, which rank at 0.44 on CDNOW against `bounded32`'s 0.085.
   - A rule of thumb to test: no flag deeper than about half the calibration window.
   - This is a config change, not a code change.
4. **Give the model a momentum signal it cannot lose.** Add a frozen, calibration-only
   input that tells the model whether each customer is speeding up or slowing down: the
   §11.4 momentum share, or the purchase rate in the second half of their tenure divided
   by the first.
   - Targets §11.4: rising customers are under-forecast at 0.47–0.86×, and on CDNOW and
     gift 2y the LSTM reproduces almost none of the actual rising − declining gap.
   - Targets §11.3: frequent customers gone quiet are over-forecast. This is where a
     sequence model should beat Pareto/NBD by design, and it currently does not.
   - It would also complement item 2: a fading customer should get a higher hazard.
   - Success: the within-cell gap of §11.4 moves toward the actual, and rising customers
     toward 1×.
5. **Train on the model's own simulated histories.** During the refit, recompute the AR
   features from sampled counts, so the model learns states it will meet in the holdout.
   It targets both sides of §4's drift, at higher cost than items 1–4.
6. **Cheap checks.**
   - `bounded32` on CDNOW without `week_sin` / `week_cos`, to tell §6's two candidate
     causes apart.
   - An open top class ("6 or more") for the old electronics panel (§9).

## 11. The customer groups

§3–§8 used groups chosen to expose the failure. This section settles the groups to report
on every window. Each candidate was judged on three questions:

1. **Does the literature use it?** Each paper was read at its methods section where
   reachable. Anything that could only be read as an abstract, or not at all, is said so.
2. **Does it make sense on our panels?** Our setting:
   - weekly counts;
   - one acquisition cohort per panel, whose first purchases fall in the first 13 weeks
     (52 on electronic_5y);
   - 39–260 calibration weeks;
   - a median of 1–4 purchases per customer.

   Most published measures were built for richer histories.
3. **Does it carry information on our data?** Three tests:
   - **coverage:** the share of customers it can be computed for;
   - **stability:** do the first and second halves of a customer's calibration agree?
   - **added information:** inside the same frequency × recency cell, does it separate
     holdout buyers from non-buyers?

Kept: frequency (§11.1), recency within frequency (§11.2), overdue as a reading of recency
(§11.3), momentum (§11.4) and the groups of Valendin et al. (§11.5). Considered and dropped:
clumpiness and regularity (§11.6). Acquisition characteristics (channel, first spend) are
left out: the trained panels do not carry them, and they answer a different question
(`docs/cohort-analysis.md` §1.5).

Conventions for this section:
- A purchase week is a week with at least one transaction.
- † marks a group with fewer than 30 customers.
- "Return rate" is the share of the group with at least one holdout transaction.
- Model cells read predicted tx / customer → **pred ÷ act** [95% CI over runs].
- Tables computed from actuals alone have 95% intervals from a seeded bootstrap over
  customers. They describe the panel; they are not model comparisons.

| grouping | literature | fits our panels | carries information here | verdict |
|---|---|---|---|---|
| frequency | standard BTYD validation | yes | yes | keep |
| recency within frequency | used descriptively | yes; CDNOW has no > 52-week band | yes | keep |
| overdue | no published precedent | needs ≥ 2 purchase weeks | mostly repeats recency | keep as a reading of recency |
| momentum | no published measure; the idea is supported | needs ≥ 2 purchase weeks | yes, beyond frequency and recency | keep |
| Valendin et al. groups | the paper we reproduce | yes (defined by the holdout outcome) | — | keep, flagged |
| clumpiness | published, built for digital daily data | poorly: too few events | weak, and confounded with leaving | drop |
| regularity | published; needs ~10 purchases per customer | no | unstable, no added information | drop |

### 11.1 Frequency: 1 / 2 / 3–4 / 5+ calibration purchases

**Definition.** Calibration transactions 1, 2, 3–4 or 5+, the same as repeat purchases
x = 0, 1, 2–3, 4+ in BTYD notation.

**Literature: the field's standard check.**
- Fader, Hardie & Lee (2005a) validate BG/NBD on CDNOW by "conditional expectations": the
  average holdout purchases of customers grouped by calibration repeat purchases, x = 0, 1,
  …, 6, 7+ ([PDF](http://www.brucehardie.com/papers/018/fader_et_al_mksc_05.pdf), Fig. 3).
  Their zero-repeat class made 18% of all holdout transactions, "second only to the 7+
  group".
- Valendin et al. (2022, doi
  [10.1016/j.ijresmar.2022.02.007](https://doi.org/10.1016/j.ijresmar.2022.02.007),
  App. B.3, Fig. 11) reuse the same plot for the LSTM.
- Vanderveld et al. (2016, KDD, doi
  [10.1145/2939672.2939693](https://doi.org/10.1145/2939672.2939693)) build "one-time
  buyer", "sporadic buyer" and "power user" cohorts at Groupon and report every metric per
  cohort. One-time and sporadic buyers had the weakest correlations (0.22, 0.23), and
  sporadic buyers "the noisiest purchasing behavior".
- Our cut points are our own: Vanderveld et al. do not publish theirs, and BTYD plots use
  every value of x.

**Our panels.** The ladder covers every customer, and both the return rate and the ranking
signal rise along it:

| window | 1 purchase | 2 | 3–4 | 5+ |
|---|---|---|---|---|
| cdnow 2y | 14% · AUC 0.56 (1417) | 35% · AUC 0.62 (435) | 58% · AUC 0.66 (312) | 78% · AUC 0.67 (193) |
| electronics 2y | 26% · AUC 0.44 (150) | 31% · AUC 0.51 (134) | 36% · AUC 0.60 (175) | 49% · AUC 0.71 (370) |
| electronics 3y | 27% · AUC 0.43 (111) | 16% · AUC 0.58 (109) | 26% · AUC 0.52 (150) | 42% · AUC 0.72 (459) |
| electronic_5y | 11% · AUC 0.53 (861) | 15% · AUC 0.61 (690) | 23% · AUC 0.61 (951) | 48% · AUC 0.70 (1253) |
| gift 2y | 16% · AUC 0.50 (1058) | 35% · AUC 0.62 (453) | 53% · AUC 0.64 (415) | 71% · AUC 0.76 (136) |
| gift 3y | 11% · AUC 0.52 (891) | 24% · AUC 0.62 (413) | 38% · AUC 0.63 (454) | 67% · AUC 0.67 (304) |
| multichannel 2y | 7% · AUC 0.48 (1028) | 17% · AUC 0.57 (242) | 29% · AUC 0.48 (106) | 58% · AUC 0.77 (26†) |
| multichannel 3y | 4% · AUC 0.38 (951) | 13% · AUC 0.60 (262) | 23% · AUC 0.56 (142) | 57% · AUC 0.67 (47) |

Each cell gives:
- the return rate;
- the AUC of recency inside the group (fewer weeks silent ranks higher; 0.5 = chance),
  computed from actuals alone;
- n.

Three readings:
- **One-timers cannot be ranked among themselves** (AUC 0.38–0.56). In a single
  acquisition cohort their one purchase is the acquisition purchase, so their history is a
  single number.
- **The second purchase adds a weak but real signal** (0.57–0.62 on six windows).
- The signal is strongest with 5+ purchases (0.67–0.77).
- **The old electronics panel breaks the ladder**: there one-timers come back more often
  than two-timers. That panel counts line items rather than trips (§1).

**Forecasts by frequency:**

| window | group | n | return rate | actual tx / customer | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---|---:|---:|---:|---|---|---|
| cdnow 2y | 1 | 1417 | 14% | 0.24 | 0.37 → **1.50×** [1.32, 1.71] | 1.04 → **4.25×** [3.87, 4.61] | 0.17 → **0.68×** [0.67, 0.69] |
| cdnow 2y | 2 | 435 | 35% | 0.71 | 0.46 → **0.65×** [0.57, 0.74] | 0.87 → **1.22×** [1.13, 1.31] | 0.63 → **0.89×** [0.88, 0.90] |
| cdnow 2y | 3-4 | 312 | 58% | 1.44 | 0.83 → **0.58×** [0.50, 0.67] | 0.99 → **0.69×** [0.65, 0.73] | 1.35 → **0.94×** [0.93, 0.95] |
| cdnow 2y | 5+ | 193 | 78% | 4.10 | 2.41 → **0.59×** [0.52, 0.66] | 2.29 → **0.56×** [0.53, 0.58] | 3.60 → **0.88×** [0.87, 0.89] |
| electronics 2y | 1 | 150 | 26% | 0.82 | 2.58 → **3.14×** [2.99, 3.27] | 1.40 → **1.71×** [1.66, 1.75] | 0.15 → **0.18×** [0.18, 0.18] |
| electronics 2y | 2 | 134 | 31% | 1.05 | 2.58 → **2.45×** [2.33, 2.55] | 1.46 → **1.39×** [1.34, 1.43] | 0.26 → **0.25×** [0.25, 0.25] |
| electronics 2y | 3-4 | 175 | 36% | 1.25 | 2.58 → **2.06×** [1.97, 2.15] | 1.57 → **1.25×** [1.21, 1.30] | 0.44 → **0.35×** [0.35, 0.35] |
| electronics 2y | 5+ | 370 | 49% | 2.66 | 2.59 → **0.97×** [0.93, 1.01] | 2.06 → **0.78×** [0.74, 0.81] | 1.10 → **0.41×** [0.41, 0.42] |
| electronics 3y | 1 | 111 | 27% | 1.30 | not run | 0.93 → **0.72×** [0.69, 0.75] | 0.12 → **0.10×** [0.10, 0.10] |
| electronics 3y | 2 | 109 | 16% | 0.50 | not run | 1.00 → **2.01×** [1.92, 2.11] | 0.23 → **0.46×** [0.46, 0.47] |
| electronics 3y | 3-4 | 150 | 26% | 0.87 | not run | 1.11 → **1.28×** [1.22, 1.34] | 0.34 → **0.39×** [0.39, 0.40] |
| electronics 3y | 5+ | 459 | 42% | 2.64 | not run | 2.10 → **0.80×** [0.76, 0.83] | 0.97 → **0.37×** [0.37, 0.37] |
| electronic_5y | 1 | 861 | 11% | 0.15 | 0.24 → **1.62×** [1.52, 1.72] | 0.27 → **1.86×** [1.77, 1.94] | 0.06 → **0.43×** [0.43, 0.43] |
| electronic_5y | 2 | 690 | 15% | 0.24 | 0.31 → **1.32×** [1.25, 1.39] | 0.32 → **1.34×** [1.29, 1.39] | 0.17 → **0.72×** [0.72, 0.73] |
| electronic_5y | 3-4 | 951 | 23% | 0.36 | 0.45 → **1.26×** [1.20, 1.32] | 0.43 → **1.20×** [1.15, 1.24] | 0.33 → **0.91×** [0.91, 0.92] |
| electronic_5y | 5+ | 1253 | 48% | 1.21 | 1.28 → **1.05×** [1.01, 1.09] | 1.18 → **0.97×** [0.94, 1.00] | 1.05 → **0.86×** [0.86, 0.87] |
| gift 2y | 1 | 1058 | 16% | 0.23 | 0.31 → **1.34×** [1.24, 1.42] | 0.37 → **1.59×** [1.53, 1.65] | 0.22 → **0.94×** [0.94, 0.95] |
| gift 2y | 2 | 453 | 35% | 0.50 | 0.41 → **0.81×** [0.73, 0.91] | 0.43 → **0.86×** [0.82, 0.90] | 0.49 → **0.97×** [0.96, 0.97] |
| gift 2y | 3-4 | 415 | 53% | 1.00 | 0.68 → **0.68×** [0.59, 0.77] | 0.57 → **0.57×** [0.54, 0.60] | 0.84 → **0.85×** [0.85, 0.85] |
| gift 2y | 5+ | 136 | 71% | 1.93 | 1.30 → **0.67×** [0.61, 0.72] | 0.86 → **0.45×** [0.43, 0.46] | 1.59 → **0.82×** [0.82, 0.83] |
| gift 3y | 1 | 891 | 11% | 0.15 | not run | 0.24 → **1.63×** [1.57, 1.69] | 0.13 → **0.92×** [0.91, 0.92] |
| gift 3y | 2 | 413 | 24% | 0.31 | not run | 0.34 → **1.13×** [1.09, 1.16] | 0.34 → **1.10×** [1.10, 1.10] |
| gift 3y | 3-4 | 454 | 38% | 0.65 | not run | 0.59 → **0.91×** [0.89, 0.93] | 0.62 → **0.96×** [0.96, 0.97] |
| gift 3y | 5+ | 304 | 67% | 1.62 | not run | 1.19 → **0.74×** [0.72, 0.76] | 1.37 → **0.85×** [0.84, 0.85] |
| multichannel 2y | 1 | 1028 | 7% | 0.09 | 0.27 → **2.87×** [2.48, 3.28] | 0.18 → **1.87×** [1.82, 1.93] | 0.08 → **0.86×** [0.85, 0.87] |
| multichannel 2y | 2 | 242 | 17% | 0.22 | 0.27 → **1.24×** [1.07, 1.42] | 0.21 → **0.97×** [0.92, 1.02] | 0.29 → **1.30×** [1.29, 1.32] |
| multichannel 2y | 3-4 | 106 | 29% | 0.41 | 0.27 → **0.67×** [0.58, 0.77] | 0.25 → **0.61×** [0.57, 0.65] | 0.57 → **1.41×** [1.40, 1.42] |
| multichannel 2y | 5+ | 26† | 58% | 1.35 | 0.27 → **0.20×** [0.17, 0.23] | 0.29 → **0.22×** [0.20, 0.23] | 1.14 → **0.85×** [0.84, 0.85] |
| multichannel 3y | 1 | 951 | 4% | 0.05 | not run | 0.11 → **2.37×** [2.28, 2.45] | 0.06 → **1.15×** [1.14, 1.16] |
| multichannel 3y | 2 | 262 | 13% | 0.16 | not run | 0.18 → **1.16×** [1.10, 1.22] | 0.21 → **1.32×** [1.31, 1.33] |
| multichannel 3y | 3-4 | 142 | 23% | 0.27 | not run | 0.35 → **1.28×** [1.21, 1.35] | 0.42 → **1.55×** [1.54, 1.56] |
| multichannel 3y | 5+ | 47 | 57% | 1.00 | not run | 0.52 → **0.52×** [0.49, 0.55] | 0.95 → **0.95×** [0.95, 0.96] |

- **The squeeze reads straight off the ladder.** On CDNOW, gift and multichannel 2y the
  LSTMs' pred ÷ act falls steadily from over-forecast at 1 purchase to under-forecast at
  5+, for example `bounded32` on gift 2y: 1.59 → 0.86 → 0.57 → 0.45. Pareto/NBD stays
  within 0.68–0.97× on CDNOW and gift.
- **Multichannel 2y:** ValendinLSTM's 0.27 for every step is the collapse.
- **Pareto/NBD's own weakness shows on multichannel:** it over-forecasts the middle of the
  ladder (1.30–1.55×).
- **Inside each group (AUC of the forecast):**
  - ValendinLSTM is at chance on electronics 2y at every step (0.49–0.53).
  - `bounded32` on CDNOW is below chance at 2 and 3–4 purchases (0.41, 0.45): the recency
    reversal of §6.
  - Elsewhere all three models are within about 0.05 of each other. The full table is in
    `results/tables_groups.md`.

### 11.2 Recency within frequency

**Definition.** Repeat buyers (2, 3–4, 5+) by weeks silent at the end of calibration: ≤ 13,
14–52, > 52. One-timers are not split, because on these panels a one-timer's only purchase
is the acquisition purchase, so they are all equally long-silent. CDNOW has no > 52 band.

**Literature.** Fader, Hardie & Lee (2005b, "RFM and CLV: using iso-value curves", JMR
42(4); [working paper](http://www.brucehardie.com/papers/rfm_clv_2005-02-16.pdf), Fig. 1)
lay out CDNOW's holdout value on a recency × frequency grid. They use it to describe
customers, not to score forecasts. The exact bins of their grid were not verified. Scoring
forecasts on such a grid is our use.

**Our panels.** This is the test that exposed §6's behaviours. The full table, every cell
on every window, is in `results/tables_groups.md`. Excerpt, predicted tx / customer:

| window | frequency | recency | n | actual | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---|---|---:|---:|---:|---:|---:|
| cdnow 2y | 2 | R ≤ 13 | 125 | 0.94 | 0.56 | **0.49** | 1.02 |
| cdnow 2y | 2 | R 14–38 | 310 | 0.62 | 0.42 | **1.02** | 0.48 |
| cdnow 2y | 3–4 | R ≤ 13 | 174 | 1.73 | 1.03 | **0.86** | 1.78 |
| cdnow 2y | 3–4 | R 14–38 | 138 | 1.07 | 0.58 | **1.15** | 0.80 |
| gift 2y | 3–4 | R ≤ 13 | 183 | 1.34 | 0.80 | 0.61 | 0.92 |
| gift 2y | 3–4 | R 14–52 | 139 | 0.94 | 0.63 | 0.57 | 0.84 |
| gift 2y | 3–4 | R > 52 | 93 | 0.41 | 0.51 | 0.46 | 0.70 |
| gift 3y | 3–4 | R ≤ 13 | 139 | 0.94 | not run | 0.90 | 0.74 |
| gift 3y | 3–4 | R 14–52 | 120 | 0.72 | not run | 0.59 | 0.68 |
| gift 3y | 3–4 | R > 52 | 195 | 0.39 | not run | 0.36 | 0.50 |
| electronic_5y | 5+ | R ≤ 13 | 339 | 2.34 | 2.38 | 2.13 | 1.77 |
| electronic_5y | 5+ | R 14–52 | 420 | 1.10 | 1.23 | 1.16 | 1.20 |
| electronic_5y | 5+ | R > 52 | 494 | 0.54 | 0.56 | 0.54 | 0.42 |

How each model handles recency here:
- **`bounded32` on CDNOW reverses recency at both frequency levels** (bold).
- **On gift it flattens recency at 2 years** (0.61 → 0.46 where the actual is 1.34 → 0.41)
  **and tracks it at 3 years** (0.90 / 0.59 / 0.36 against 0.94 / 0.72 / 0.39).
- **On electronic_5y both LSTMs track it** more closely than Pareto/NBD.
- **On the old electronics and multichannel 2y**, many cells have fewer than 30 customers,
  so read those windows by frequency (§11.1) rather than by cell.

### 11.3 Overdue: a reading of recency

**Definition.** For customers with at least 2 purchase weeks:
- **usual gap** = (last purchase week − first purchase week) ÷ (purchase weeks − 1);
- **overdue ratio** = weeks silent ÷ usual gap;
- **groups:** on schedule (≤ 1), overdue 1–3×, overdue > 3×.

Example: two customers each silent for 20 weeks. One bought every 15 weeks (ratio 1.3,
overdue 1–3×); the other every 2 weeks (ratio 10, overdue > 3×).

**Literature: no published precedent for this grouping.**
- **The closest idea:** Platzer & Reutterer (2016, Pareto/GGG, doi
  [10.1287/mksc.2015.0963](https://doi.org/10.1287/mksc.2015.0963)). In their simulation
  the largest gains over Pareto/NBD (+41% MAE) are for regular, frequent customers whose
  last purchase was over 8 weeks ago. Their abstract names "previously very frequently
  active" customers with "a longer purchase hiatus".
- **The best-known silence heuristic is different:** Wübben & Wangenheim (2008, doi
  [10.1509/jmkg.72.3.082](https://doi.org/10.1509/jmkg.72.3.082)) use a fixed calendar
  window, not the customer's own gap.

**Our panels: once frequency is fixed, overdue is almost entirely recency.** The test
compares overdue > 3× with on-schedule customers inside the same frequency × recency cell:

| window | Δ return rate, overdue > 3× − on schedule, pts [95% CI] | customers in comparable cells |
|---|---|---:|
| cdnow 2y | −23.5 [−43.7, −3.0] | 213 |
| electronics 2y | +24.0 [−14.7, +54.3] | 60 |
| electronics 3y | no cell has ≥ 10 on each side | 0 |
| electronic_5y | +3.1 [−5.1, +10.0] | 1072 |
| gift 2y | no cell has ≥ 10 on each side | 0 |
| gift 3y | +0.2 [−12.4, +13.1] | 205 |
| multichannel 2y | no cell has ≥ 10 on each side | 0 |
| multichannel 3y | −9.7 [−26.9, +4.3] | 121 |

On three windows no cell even holds both kinds of customer, and only CDNOW shows a
difference beyond recency. So overdue is kept as a readable way of *describing* recency,
not as independent information.

**Forecasts by overdue group:**

| window | group | n | return rate | actual tx / customer | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---|---:|---:|---:|---|---|---|
| cdnow 2y | on schedule | 389 | 65% | 2.55 | 1.46 → **0.57×** [0.51, 0.65] | 1.26 → **0.49×** [0.47, 0.51] | 2.23 → **0.88×** [0.87, 0.88] |
| cdnow 2y | overdue 1–3× | 222 | 54% | 1.41 | 0.83 → **0.59×** [0.53, 0.66] | 1.04 → **0.74×** [0.69, 0.80] | 1.54 → **1.09×** [1.08, 1.11] |
| cdnow 2y | overdue > 3× | 317 | 36% | 0.78 | 0.53 → **0.68×** [0.60, 0.78] | 1.24 → **1.59×** [1.48, 1.70] | 0.56 → **0.73×** [0.71, 0.74] |
| electronics 2y | on schedule | 222 | 57% | 2.98 | 2.61 → **0.88×** [0.83, 0.91] | 2.36 → **0.79×** [0.75, 0.83] | 1.31 → **0.44×** [0.44, 0.44] |
| electronics 2y | overdue 1–3× | 132 | 44% | 2.16 | 2.58 → **1.19×** [1.14, 1.24] | 1.77 → **0.82×** [0.78, 0.86] | 1.08 → **0.50×** [0.50, 0.50] |
| electronics 2y | overdue > 3× | 156 | 31% | 1.26 | 2.58 → **2.05×** [1.96, 2.13] | 1.53 → **1.22×** [1.16, 1.27] | 0.40 → **0.32×** [0.31, 0.32] |
| electronics 3y | on schedule | 273 | 48% | 3.05 | not run | 2.48 → **0.81×** [0.78, 0.85] | 1.09 → **0.36×** [0.36, 0.36] |
| electronics 3y | overdue 1–3× | 149 | 40% | 2.09 | not run | 1.65 → **0.79×** [0.75, 0.82] | 0.93 → **0.45×** [0.44, 0.45] |
| electronics 3y | overdue > 3× | 181 | 17% | 0.85 | not run | 1.16 → **1.37×** [1.31, 1.44] | 0.39 → **0.46×** [0.46, 0.47] |
| electronic_5y | on schedule | 1125 | 44% | 1.10 | 1.21 → **1.10×** [1.06, 1.15] | 1.10 → **1.01×** [0.97, 1.04] | 0.93 → **0.85×** [0.85, 0.85] |
| electronic_5y | overdue 1–3× | 751 | 32% | 0.59 | 0.67 → **1.13×** [1.09, 1.18] | 0.63 → **1.08×** [1.04, 1.11] | 0.66 → **1.12×** [1.12, 1.13] |
| electronic_5y | overdue > 3× | 972 | 20% | 0.35 | 0.39 → **1.10×** [1.04, 1.16] | 0.39 → **1.09×** [1.05, 1.13] | 0.20 → **0.57×** [0.57, 0.57] |
| gift 2y | on schedule | 537 | 59% | 1.26 | 0.76 → **0.60×** [0.55, 0.66] | 0.61 → **0.49×** [0.47, 0.51] | 0.90 → **0.71×** [0.71, 0.71] |
| gift 2y | overdue 1–3× | 268 | 41% | 0.60 | 0.51 → **0.86×** [0.76, 0.96] | 0.47 → **0.79×** [0.76, 0.82] | 0.68 → **1.14×** [1.14, 1.15] |
| gift 2y | overdue > 3× | 195 | 24% | 0.34 | 0.49 → **1.42×** [1.25, 1.62] | 0.46 → **1.35×** [1.27, 1.42] | 0.63 → **1.83×** [1.81, 1.84] |
| gift 3y | on schedule | 629 | 51% | 1.07 | not run | 0.89 → **0.83×** [0.81, 0.85] | 0.87 → **0.81×** [0.81, 0.81] |
| gift 3y | overdue 1–3× | 257 | 38% | 0.61 | not run | 0.47 → **0.77×** [0.75, 0.79] | 0.69 → **1.13×** [1.12, 1.13] |
| gift 3y | overdue > 3× | 282 | 21% | 0.27 | not run | 0.32 → **1.18×** [1.14, 1.23] | 0.40 → **1.48×** [1.47, 1.49] |
| multichannel 2y | on schedule | 164 | 26% | 0.46 | 0.27 → **0.59×** [0.51, 0.68] | 0.27 → **0.58×** [0.55, 0.62] | 0.56 → **1.22×** [1.21, 1.23] |
| multichannel 2y | overdue 1–3× | 102 | 28% | 0.38 | 0.27 → **0.71×** [0.61, 0.81] | 0.20 → **0.53×** [0.49, 0.56] | 0.39 → **1.02×** [1.01, 1.03] |
| multichannel 2y | overdue > 3× | 97 | 14% | 0.14 | 0.27 → **1.87×** [1.62, 2.14] | 0.20 → **1.36×** [1.28, 1.44] | 0.28 → **1.95×** [1.92, 1.99] |
| multichannel 3y | on schedule | 188 | 28% | 0.40 | not run | 0.35 → **0.86×** [0.82, 0.89] | 0.48 → **1.18×** [1.17, 1.18] |
| multichannel 3y | overdue 1–3× | 124 | 17% | 0.23 | not run | 0.24 → **1.01×** [0.96, 1.06] | 0.34 → **1.47×** [1.46, 1.48] |
| multichannel 3y | overdue > 3× | 129 | 16% | 0.17 | not run | 0.20 → **1.19×** [1.11, 1.28] | 0.20 → **1.20×** [1.19, 1.21] |

**Overdue > 3× customers are over-forecast:**
- by `bounded32` on 7 of 8 windows (1.09–1.59×);
- by Pareto/NBD on gift and multichannel (1.20–1.95×).

**On-schedule customers are under-forecast** by the LSTMs on CDNOW, gift and multichannel
(0.49–0.86×).

These over-forecast customers are frequent buyers who have gone quiet. Pareto/NBD cannot
tell their silence apart from an irregular buyer's pause: that is the Pareto/GGG argument
above. A sequence model could in principle, but these LSTMs do not.

### 11.4 Momentum: rising vs declining buyers

**Definition.** For customers with at least 2 purchase weeks: the share of their purchase
weeks that fall in the second half of their own tenure (first purchase → end of
calibration). **Rising** if at least 0.5, **declining** otherwise.

| | purchase weeks | tenure | share in second half | group |
|---|---|---|---|---|
| customer A | 2 and 90 | weeks 2–103 | 1/2 | rising |
| customer B | 2, 5 and 9 | weeks 2–103 | 0/3 | declining |

For a two-timer, "rising" simply means the second purchase came in the second half of
their tenure.

**Literature: no published measure of this kind; it is our construct.** The idea it
captures, that a customer's buying rate changes over time, is well supported:
- Romero, van der Lans & Wierenga (2013, doi
  [10.1016/j.intmar.2013.04.003](https://doi.org/10.1016/j.intmar.2013.04.003); abstract
  only) state that standard stochastic CLV models assume purchase behaviour constant over
  time and defection permanent. That is the assumption momentum tests.
- Netzer, Lattin & Srinivasan (2008, doi
  [10.1287/mksc.1070.0294](https://doi.org/10.1287/mksc.1070.0294)) model donors moving
  between latent "dormant", "occasional" and "active" states (yearly alumni giving).
- Miguéis et al. (2012, Ghent University working paper 2012/790,
  [PDF](http://wps-feb.ugent.be/Papers/wp_12_790.pdf)) define grocery churn as "partial and
  progressive defection": quarterly spend falling below 40% of a reference quarter.
- None of these forms rising/declining groups from calibration history, and none reports
  forecast accuracy for such groups. No peer-reviewed RFM extension with a trend term was
  found.

**Our panels: the only pattern measure with information beyond frequency and recency.**
Inside the same frequency × recency cell, rising customers return more often. Δ return
rate in points, at three cut-offs [95% CI]:

| window | cut-off 0.4 | 0.5 | 0.6 |
|---|---|---|---|
| cdnow 2y | +16.8 [+4.8, +28.6] | +16.2 [+5.0, +27.9] | +9.6 [−0.6, +19.9] |
| electronics 2y | +0.9 [−10.7, +13.0] | +2.6 [−8.6, +14.4] | +2.6 [−14.5, +17.7] |
| electronics 3y | −4.2 [−16.9, +8.2] | −11.4 [−23.0, +0.4] | +5.9 [−8.6, +18.3] |
| electronic_5y | +6.5 [+2.8, +10.2] | +5.3 [+1.5, +9.5] | +5.7 [+0.2, +11.6] |
| gift 2y | +9.2 [−0.4, +18.3] | +10.1 [+0.9, +19.1] | +6.6 [−5.1, +17.5] |
| gift 3y | +6.8 [−0.2, +13.7] | +6.4 [−0.3, +13.9] | +11.3 [+3.1, +19.0] |
| multichannel 2y | +4.1 [−13.3, +18.8] | +4.1 [−12.8, +19.4] | too few |
| multichannel 3y | +11.6 [+1.2, +23.6] | +11.6 [−0.1, +25.6] | +10.6 [−9.8, +34.3] |

- **The sign is positive on six of the eight windows at every cut-off.** The intervals
  exclude 0 on CDNOW and electronic_5y at most cut-offs, and on gift and multichannel 3y at
  some. So the result does not hinge on 0.5.
- **The old electronics panel is the exception again.**

**Do the models pick it up?** Gap in holdout tx / customer, rising − declining, within the
same cells. Actual: CI over customers. Models: mean [95% CI over runs].

| window | rising / declining | actual | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---|---|---|---|---|
| cdnow 2y | 277 / 651 | +0.73 [+0.19, +1.39] | +0.42 [+0.35, +0.51] | +0.01 [−0.03, +0.04] | +0.19 [+0.19, +0.20] |
| electronics 2y | 159 / 351 | +0.76 [−0.24, +1.69] | +0.01 [+0.00, +0.02] | +0.09 [+0.07, +0.11] | −0.12 [−0.12, −0.11] |
| electronics 3y | 221 / 382 | −0.47 [−2.04, +1.00] | not run | +0.18 [+0.15, +0.21] | −0.17 [−0.18, −0.17] |
| electronic_5y | 925 / 1923 | +0.17 [+0.06, +0.28] | +0.23 [+0.23, +0.24] | +0.19 [+0.18, +0.20] | +0.12 [+0.12, +0.12] |
| gift 2y | 360 / 640 | +0.61 [+0.25, +0.94] | +0.13 [+0.11, +0.15] | +0.09 [+0.09, +0.10] | +0.03 [+0.03, +0.04] |
| gift 3y | 500 / 668 | +0.32 [+0.15, +0.48] | not run | +0.17 [+0.16, +0.17] | +0.09 [+0.09, +0.09] |
| multichannel 2y | 129 / 234 | +0.07 [−0.14, +0.31] | −0.00 [−0.00, +0.00] | +0.02 [+0.01, +0.02] | +0.04 [+0.04, +0.04] |
| multichannel 3y | 152 / 289 | +0.15 [+0.01, +0.30] | not run | +0.03 [+0.02, +0.03] | +0.07 [+0.07, +0.07] |

- **Pareto/NBD reproduces little of the gap.** It reads only frequency, recency and
  tenure, and what it shows is what those still vary inside a cell.
- **The LSTMs could reproduce it, since they read the whole sequence:**
  - **electronic_5y:** they do, with +0.19 and +0.23 against an actual +0.17;
  - **gift 3y:** partly, +0.17 against +0.32;
  - **CDNOW and gift 2y:** hardly, `bounded32` +0.01 and +0.09 against +0.73 and +0.61.
- **Momentum is information that is in the history but goes unused on short or sparse
  windows.**

**Forecasts by momentum group:**

| window | group | n | return rate | actual tx / customer | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---|---:|---:|---:|---|---|---|
| cdnow 2y | rising | 277 | 61% | 2.06 | 1.31 → **0.64×** [0.56, 0.72] | 1.09 → **0.53×** [0.51, 0.55] | 1.82 → **0.88×** [0.88, 0.89] |
| cdnow 2y | declining | 651 | 49% | 1.50 | 0.86 → **0.57×** [0.51, 0.64] | 1.24 → **0.83×** [0.78, 0.87] | 1.36 → **0.90×** [0.89, 0.92] |
| electronics 2y | rising | 159 | 57% | 3.29 | 2.61 → **0.79×** [0.76, 0.82] | 2.31 → **0.70×** [0.67, 0.74] | 1.26 → **0.38×** [0.38, 0.39] |
| electronics 2y | declining | 351 | 40% | 1.76 | 2.58 → **1.46×** [1.40, 1.52] | 1.79 → **1.01×** [0.97, 1.06] | 0.84 → **0.47×** [0.47, 0.48] |
| electronics 3y | rising | 221 | 45% | 2.86 | not run | 2.45 → **0.86×** [0.82, 0.89] | 1.08 → **0.38×** [0.38, 0.38] |
| electronics 3y | declining | 382 | 32% | 1.74 | not run | 1.55 → **0.89×** [0.85, 0.93] | 0.71 → **0.40×** [0.40, 0.41] |
| electronic_5y | rising | 925 | 43% | 1.08 | 1.19 → **1.10×** [1.06, 1.14] | 1.07 → **0.99×** [0.96, 1.03] | 0.86 → **0.80×** [0.80, 0.80] |
| electronic_5y | declining | 1923 | 27% | 0.53 | 0.59 → **1.12×** [1.07, 1.17] | 0.57 → **1.08×** [1.05, 1.12] | 0.49 → **0.93×** [0.92, 0.93] |
| gift 2y | rising | 360 | 56% | 1.28 | 0.72 → **0.56×** [0.51, 0.61] | 0.60 → **0.47×** [0.45, 0.48] | 0.81 → **0.63×** [0.63, 0.63] |
| gift 2y | declining | 640 | 42% | 0.69 | 0.59 → **0.86×** [0.77, 0.96] | 0.52 → **0.75×** [0.72, 0.78] | 0.77 → **1.12×** [1.12, 1.13] |
| gift 3y | rising | 500 | 51% | 1.14 | not run | 0.89 → **0.78×** [0.76, 0.80] | 0.86 → **0.76×** [0.76, 0.76] |
| gift 3y | declining | 668 | 33% | 0.51 | not run | 0.49 → **0.96×** [0.94, 0.99] | 0.61 → **1.19×** [1.19, 1.19] |
| multichannel 2y | rising | 129 | 26% | 0.49 | 0.27 → **0.56×** [0.48, 0.64] | 0.26 → **0.53×** [0.50, 0.57] | 0.50 → **1.01×** [1.01, 1.02] |
| multichannel 2y | declining | 234 | 22% | 0.28 | 0.27 → **0.98×** [0.84, 1.12] | 0.21 → **0.76×** [0.72, 0.81] | 0.40 → **1.46×** [1.44, 1.47] |
| multichannel 3y | rising | 152 | 28% | 0.41 | not run | 0.32 → **0.77×** [0.73, 0.80] | 0.43 → **1.04×** [1.03, 1.04] |
| multichannel 3y | declining | 289 | 18% | 0.22 | not run | 0.25 → **1.13×** [1.07, 1.20] | 0.32 → **1.46×** [1.45, 1.47] |

The two kinds of model fail on opposite sides:
- **Pareto/NBD over-forecasts declining customers** on gift and multichannel (1.12–1.46×):
  it assumes their rate is unchanged.
- **The LSTMs under-forecast rising customers** on every window except electronic_5y
  (0.47–0.86×).

### 11.5 The groups of Valendin et al. (2022)

**Definitions, from the paper (§4.1, verified in the open-access text).** Both are defined
by the holdout outcome:
- **At Risk:** "individuals with at least average historical transaction frequency who do
  not return to make a single transaction in the holdout period."
- **Opportunity:** "customers who make transactions more frequently in the holdout than
  during the calibration period."

Valendin et al. also split At Risk by calibration-frequency percentile. Here it is split by
the §11.1 frequency groups instead, so the groups line up.

These groups are kept for comparability with the paper we reproduce, not as a way to see
what a model could have known. The paper reports that the LSTM's advantage on At Risk grows
with how frequent those customers had been, and that Pareto/NBD and Pareto/GGG "fail to
identify any" Opportunity customers in 5 of their 8 datasets.

**At Risk: predicted tx / customer (the actual is 0, so lower is better).**

| window | group | n | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---|---:|---:|---:|---:|
| cdnow 2y | At Risk, 3–4 | 130 | 0.72 | 1.03 | 1.16 |
| cdnow 2y | At Risk, 5+ | 43 | 1.71 | 1.84 | 2.74 |
| electronics 2y | At Risk, 5+ | 140 | 2.58 | 1.83 | 0.85 |
| electronics 3y | At Risk, 5+ | 136 | not run | 2.05 | 0.97 |
| electronic_5y | At Risk, 5+ | 651 | 0.90 | 0.86 | 0.77 |
| gift 2y | At Risk, 3–4 | 197 | 0.64 | 0.54 | 0.81 |
| gift 2y | At Risk, 5+ | 39 | 1.17 | 0.80 | 1.48 |
| gift 3y | At Risk, 3–4 | 280 | not run | 0.54 | 0.60 |
| gift 3y | At Risk, 5+ | 100 | not run | 0.96 | 1.15 |
| multichannel 2y | At Risk, 2 | 202 | 0.27 | 0.21 | 0.28 |
| multichannel 2y | At Risk, 3–4 | 75 | 0.27 | 0.25 | 0.56 |
| multichannel 2y | At Risk, 5+ | 11† | 0.27 | 0.25 | 0.99 |
| multichannel 3y | At Risk, 2 | 227 | not run | 0.18 | 0.20 |
| multichannel 3y | At Risk, 3–4 | 110 | not run | 0.34 | 0.42 |
| multichannel 3y | At Risk, 5+ | 20† | not run | 0.48 | 0.87 |

Which frequency levels appear depends on the panel's average frequency. On electronics,
"at least average" means at least 6 purchases.

Model − Pareto/NBD, predicted tx / customer on all At Risk customers (negative = closer to
the true 0):

| window | ValendinLSTM | LSTM + bounded32 |
|---|---|---|
| cdnow 2y | **−0.59 [−0.70, −0.46]** | **−0.32 [−0.39, −0.25]** |
| electronics 2y | **+1.73 [+1.61, +1.83]** | **+0.98 [+0.90, +1.07]** |
| electronics 3y | not run | **+1.08 [+0.99, +1.17]** |
| electronic_5y | **+0.13 [+0.10, +0.17]** | **+0.09 [+0.06, +0.12]** |
| gift 2y | **−0.20 [−0.28, −0.11]** | **−0.34 [−0.37, −0.31]** |
| gift 3y | not run | **−0.09 [−0.11, −0.07]** |
| multichannel 2y | **−0.11 [−0.15, −0.07]** | **−0.16 [−0.17, −0.15]** |
| multichannel 3y | not run | **−0.06 [−0.07, −0.05]** |

**Valendin's At Risk finding replicates on CDNOW, gift and multichannel but not on
electronics.**
- On CDNOW, gift and multichannel the LSTMs forecast less for churned customers than
  Pareto/NBD does.
- The gap is widest for the most frequent of them: multichannel 5+, 0.25 against 0.99.
- On electronics Pareto/NBD forecasts less, but only because it under-forecasts the whole
  panel (−63%).
- A model that forecasts zero for everyone would win this table. Read it only beside the
  Opportunity table below.

**Opportunity:**

| window | n | actual tx / customer | ValendinLSTM | LSTM + bounded32 | Pareto/NBD |
|---|---:|---:|---|---|---|
| cdnow 2y | 171 | 5.12 | 0.90 → **0.18×** [0.16, 0.20] | 1.22 → **0.24×** [0.23, 0.25] | 1.13 → **0.22×** [0.22, 0.22] |
| electronics 2y | 81 | 7.54 | 2.58 → **0.34×** [0.33, 0.36] | 1.62 → **0.22×** [0.21, 0.22] | 0.46 → **0.06×** [0.06, 0.06] |
| electronics 3y | 71 | 10.17 | not run | 1.62 → **0.16×** [0.15, 0.17] | 0.55 → **0.05×** [0.05, 0.05] |
| electronic_5y | 45 | 4.27 | 0.44 → **0.10×** [0.10, 0.11] | 0.43 → **0.10×** [0.10, 0.10] | 0.24 → **0.06×** [0.06, 0.06] |
| gift 2y | 89 | 3.99 | 0.50 → **0.12×** [0.11, 0.13] | 0.47 → **0.12×** [0.11, 0.12] | 0.51 → **0.13×** [0.13, 0.13] |
| gift 3y | 39 | 3.15 | not run | 0.39 → **0.12×** [0.12, 0.13] | 0.30 → **0.09×** [0.09, 0.09] |
| multichannel 2y | 20† | 2.50 | 0.27 → **0.11×** [0.09, 0.12] | 0.19 → **0.07×** [0.07, 0.08] | 0.11 → **0.04×** [0.04, 0.04] |
| multichannel 3y | 4† | 2.25 | not run | 0.12 → **0.05×** [0.05, 0.05] | 0.05 → **0.02×** [0.02, 0.02] |

Every model reaches only 2–34% of what growing customers actually buy. The LSTMs are at
or above Pareto/NBD on every window, and well above it on electronics.

### 11.6 Considered and dropped: clumpiness and regularity

Both are published, well-cited measures of buying pattern. Neither works on our panels,
and the reasons come from the papers themselves as much as from our data.

| window | customers with ≥ 3 purchase weeks | burstiness B: first-half vs second-half Spearman (customers) | clumpy (C = 1) share of repeaters | within-cell Δ return, clumpy − not, pts [95% CI] |
|---|---:|---|---:|---|
| cdnow 2y | 20% | 0.01 (63) | 9% | −8.5 [−18.4, +2.5] |
| electronics 2y | 37% | 0.30 (38) | 28% | −11.7 [−22.8, −0.3] |
| electronics 3y | 48% | −0.03 (76) | 30% | −1.9 [−10.6, +7.2] |
| electronic_5y | 56% | 0.11 (479) | 27% | −5.2 [−8.8, −1.8] |
| gift 2y | 26% | 0.25 (38) | 10% | −4.1 [−14.2, +6.4] |
| gift 3y | 36% | 0.17 (93) | 16% | −0.4 [−7.2, +5.7] |
| multichannel 2y | 9% | – (4) | 20% | −10.5 [−21.5, +2.8] |
| multichannel 3y | 13% | 0.30 (12) | 25% | +2.1 [−7.0, +11.2] |

**Regularity** (even vs uneven gaps between purchases):
- **What the literature does.** Platzer & Reutterer (2016) model gaps between purchases as
  Gamma(k): k = 1 is random (Poisson) timing and larger k is more regular. Reutterer,
  Platzer & Schröder (2021, doi
  [10.1016/j.ijresmar.2020.09.002](https://doi.org/10.1016/j.ijresmar.2020.09.002)) use a
  cohort-wide Erlang-k. Neither paper groups real customers by regularity.
- **Why it fails here, by their own requirements:**
  - Per-customer estimates need "sufficient transactions made by a customer (usually 10 or
    more)".
  - The Wheat & Morrison estimator needs at least 3 transactions and gives one value for
    the whole cohort.
  - On CDNOW, k "varies narrowly around 1, which confirms the validity of assuming a
    Poisson purchase process".
  - Regularity helps only when the cohort's k̂ is about 1.5 or higher.
- **Our data agree.** The measure applies to 9–56% of customers. The Goh & Barabási (2008,
  doi [10.1209/0295-5075/81/48002](https://doi.org/10.1209/0295-5075/81/48002)) burstiness
  B of a customer's first half does not predict that of their second half (−0.03 to 0.30,
  often on fewer than 100 customers). It is not a stable trait at this depth of history.

**Clumpiness** (purchases bunched together):
- **What the literature does.** Zhang, Bradlow & Small (2013, doi
  [10.1080/02664763.2013.818627](https://doi.org/10.1080/02664763.2013.818627); 2015, doi
  [10.1287/mksc.2014.0873](https://doi.org/10.1287/mksc.2014.0873)) score bunching on the
  gaps between events, including the gap from the window's start to the first event and
  from the last event to the window's end. A customer is flagged clumpy (C = 1) when the
  score exceeds the 95th percentile of the same number of events placed at random. It was
  built on Hulu viewing days: 120 days, 8.7 events per customer.
- **Why it fails here, by the authors' own warnings:**
  - With few events "the influence of one shock is large".
  - There is "masking between clumpiness … and attrition": a customer who buys early and
    then stops looks clumpy, because the end-of-window gap is long.
  - Bradlow notes that for "historically purchased goods, clumpiness really isn't there".
- **Our data match that warning.** Flagged customers are *less* likely to return within the
  same frequency × recency cell (−0.4 to −11.7 points; supported on electronics 2y and
  electronic_5y). The authors' rejoinder reports that clumpy customers buy *more*. Here the
  flag mostly picks up customers who have left.
- **The formula used here** is the normalised Hp of the `clumpi` reference implementation.
  It was checked against Rao's commentary examples, not against the 2015 paper, which was
  not reachable.

### 11.7 What the groups add to the picture

1. **The squeeze runs along the frequency ladder** (§11.1): over-forecast at 1 purchase,
   under-forecast at 5+, on every 2-year window except electronics.
2. **Frequent buyers who have gone quiet (overdue > 3×) are over-forecast by both model
   families**, and steady buyers (on schedule) are under-forecast by the LSTMs (§11.3).
3. **Momentum is the one pattern in the history that adds information**, and it is a blind
   spot for both models on short or sparse windows: Pareto/NBD by construction, the LSTMs
   in practice. With five years of history the LSTMs do capture it (§11.4).
4. **Valendin et al.'s At Risk result replicates on three of the four panels**, and no
   model gets near growing customers (§11.5).

## 12. Reproducing

From the repo root with the project venv:

```
python .scratch/customer-analysis-2/collect.py .scratch/customer-analysis-2/results/groups.pkl
python .scratch/customer-analysis-2/tables.py  .scratch/customer-analysis-2/results/groups.pkl
python .scratch/customer-analysis-2/tables_groups.py .scratch/customer-analysis-2/results/groups.pkl
python .scratch/customer-analysis-2/tables_appendix.py .scratch/customer-analysis-2/results/groups.pkl
```

- `collect.py` reads every stored forecast in §1 and writes one row per (window, model,
  run, grouping, group).
- `tables.py` prints the tables of §3–§8; its output is in
  `.scratch/customer-analysis-2/results/tables.md`.
- `tables_groups.py` prints every table of §11, including the actuals-only tests (seeded
  bootstraps and seeded clumpiness thresholds); its output is in
  `.scratch/customer-analysis-2/results/tables_groups.md`.
- `tables_appendix.py` prints the Appendix; its output is in
  `.scratch/customer-analysis-2/results/tables_appendix.md`.
- The CDNOW week-by-week figures in §6 came from a one-off reading of the same prediction
  files; they are the only numbers here that no script prints.

## Appendix: customers and transactions per group

For every window, every customer group of §11:
- how many customers it holds;
- how many holdout transactions they actually made;
- how many each model predicted: the mean over runs, with predicted ÷ actual in brackets.

**Window choices.**
- **No 4-year calibration has ever been run**, so the windows are 2 and 3 years, plus the
  5-year electronic_5y split as the longest available.
- Every holdout is one year, **except CDNOW**: its calibration is 39 weeks and its holdout
  39 weeks (ADR-0009). CDNOW is included for completeness.

**Model columns.**
- A model column is absent where that model was never run on the window.
- **LSTM + bounded32** is `ar_bounded_52 / nofloor` on electronic_5y (§1).
- **Transformer** has the same inputs as the LSTM column, on the two windows where it ran
  (§1).

**Reading the rows.**
- The groups inside one block partition the cohort, except "Recency within frequency",
  which leaves out one-timers (the "1 purchase" row).
- "One purchase week" includes the few customers whose several purchases all fell in one
  week, so it can exceed "1 purchase".
- At Risk customers made no holdout purchase by definition, so their cells show the
  predicted count only.
- † marks fewer than 30 customers; its ratio is unreliable.

#### CDNOW — 39 weeks calibration, 39 weeks holdout

| customer group | customers | actual tx | Pareto/NBD | ValendinLSTM | LSTM + bounded32 |
|---|---:|---:|---:|---:|---:|
| *All customers* | | | | | |
| all | 2357 | 1895 | 1627 (0.86×) | 1445 (0.76×) | 2599 (1.37×) |
| *Frequency (calibration purchases)* | | | | | |
| 1 purchase | 1417 | 346 | 236 (0.68×) | 520 (1.50×) | 1472 (4.25×) |
| 2 purchases | 435 | 309 | 276 (0.89×) | 200 (0.65×) | 377 (1.22×) |
| 3-4 purchases | 312 | 448 | 420 (0.94×) | 260 (0.58×) | 308 (0.69×) |
| 5+ purchases | 193 | 792 | 695 (0.88×) | 465 (0.59×) | 442 (0.56×) |
| *Recency within frequency (weeks silent)* | | | | | |
| 2 purchases, silent ≤ 13 w | 125 | 117 | 127 (1.09×) | 70 (0.60×) | 62 (0.53×) |
| 2 purchases, silent 14–38 w | 310 | 192 | 148 (0.77×) | 130 (0.68×) | 315 (1.64×) |
| 3-4 purchases, silent ≤ 13 w | 174 | 301 | 310 (1.03×) | 180 (0.60×) | 150 (0.50×) |
| 3-4 purchases, silent 14–38 w | 138 | 147 | 111 (0.75×) | 80 (0.54×) | 159 (1.08×) |
| 5+ purchases, silent ≤ 13 w | 157 | 730 | 655 (0.90×) | 433 (0.59×) | 384 (0.53×) |
| 5+ purchases, silent 14–38 w | 36 | 62 | 40 (0.64×) | 32 (0.52×) | 58 (0.93×) |
| *Overdue (silence ÷ own usual gap)* | | | | | |
| on schedule | 389 | 991 | 869 (0.88×) | 568 (0.57×) | 488 (0.49×) |
| overdue 1-3x | 222 | 312 | 342 (1.09×) | 185 (0.59×) | 232 (0.74×) |
| overdue >3x | 317 | 246 | 179 (0.73×) | 168 (0.68×) | 392 (1.59×) |
| *Momentum* | | | | | |
| rising | 277 | 570 | 504 (0.88×) | 362 (0.64×) | 302 (0.53×) |
| declining | 651 | 979 | 885 (0.90×) | 558 (0.57×) | 810 (0.83×) |
| one purchase week | 1429 | 346 | 238 (0.69×) | 525 (1.52×) | 1486 (4.30×) |
| *Valendin et al. (2022) groups* | | | | | |
| At Risk | 173 | 0 | 268 | 167 | 213 |
| Opportunity | 171 | 875 | 194 (0.22×) | 155 (0.18×) | 209 (0.24×) |
| Other | 2013 | 1020 | 1165 (1.14×) | 1124 (1.10×) | 2177 (2.13×) |

#### electronics — 2 years calibration, 1 year holdout

| customer group | customers | actual tx | Pareto/NBD | ValendinLSTM | LSTM + bounded32 | Transformer |
|---|---:|---:|---:|---:|---:|---:|
| *All customers* | | | | | | |
| all | 829 | 1467 | 541 (0.37×) | 2142 (1.46×) | 1443 (0.98×) | 1504 (1.03×) |
| *Frequency (calibration purchases)* | | | | | | |
| 1 purchase | 150 | 123 | 22 (0.18×) | 386 (3.14×) | 210 (1.71×) | 240 (1.95×) |
| 2 purchases | 134 | 141 | 35 (0.25×) | 346 (2.45×) | 196 (1.39×) | 221 (1.57×) |
| 3-4 purchases | 175 | 219 | 77 (0.35×) | 452 (2.06×) | 274 (1.25×) | 300 (1.37×) |
| 5+ purchases | 370 | 984 | 408 (0.41×) | 959 (0.97×) | 763 (0.78×) | 743 (0.76×) |
| *Recency within frequency (weeks silent)* | | | | | | |
| 2 purchases, silent ≤ 13 w | 7† | 29 | 5 (0.19×) | 18 (0.63×) | 15 (0.50×) | 14 (0.49×) |
| 2 purchases, silent 14–52 w | 12† | 2 | 8 (4.02×) | 31 (15.45×) | 19 (9.33×) | 20 (10.04×) |
| 2 purchases, silent > 52 w | 115 | 110 | 21 (0.19×) | 296 (2.69×) | 162 (1.48×) | 187 (1.70×) |
| 3-4 purchases, silent ≤ 13 w | 20† | 35 | 19 (0.54×) | 52 (1.49×) | 44 (1.25×) | 42 (1.20×) |
| 3-4 purchases, silent 14–52 w | 36 | 44 | 28 (0.64×) | 93 (2.12×) | 59 (1.35×) | 63 (1.43×) |
| 3-4 purchases, silent > 52 w | 119 | 140 | 30 (0.21×) | 307 (2.19×) | 171 (1.22×) | 195 (1.39×) |
| 5+ purchases, silent ≤ 13 w | 110 | 505 | 199 (0.39×) | 289 (0.57×) | 320 (0.63×) | 271 (0.54×) |
| 5+ purchases, silent 14–52 w | 133 | 312 | 163 (0.52×) | 343 (1.10×) | 252 (0.81×) | 254 (0.81×) |
| 5+ purchases, silent > 52 w | 127 | 167 | 45 (0.27×) | 327 (1.96×) | 191 (1.14×) | 218 (1.30×) |
| *Overdue (silence ÷ own usual gap)* | | | | | | |
| on schedule | 222 | 661 | 290 (0.44×) | 578 (0.88×) | 523 (0.79×) | 476 (0.72×) |
| overdue 1-3x | 132 | 285 | 143 (0.50×) | 340 (1.19×) | 234 (0.82×) | 245 (0.86×) |
| overdue >3x | 156 | 196 | 62 (0.32×) | 402 (2.05×) | 239 (1.22×) | 273 (1.39×) |
| *Momentum* | | | | | | |
| rising | 159 | 523 | 201 (0.38×) | 414 (0.79×) | 368 (0.70×) | 330 (0.63×) |
| declining | 351 | 619 | 294 (0.47×) | 906 (1.46×) | 628 (1.01×) | 663 (1.07×) |
| one purchase week | 319 | 325 | 47 (0.14×) | 822 (2.53×) | 447 (1.38×) | 511 (1.57×) |
| *Valendin et al. (2022) groups* | | | | | | |
| At Risk | 140 | 0 | 119 | 362 | 256 | 263 |
| Opportunity | 81 | 611 | 37 (0.06×) | 209 (0.34×) | 132 (0.22×) | 141 (0.23×) |
| Other | 608 | 856 | 385 (0.45×) | 1571 (1.84×) | 1055 (1.23×) | 1100 (1.28×) |

#### gift — 2 years calibration, 1 year holdout

| customer group | customers | actual tx | Pareto/NBD | ValendinLSTM | LSTM + bounded32 |
|---|---:|---:|---:|---:|---:|
| *All customers* | | | | | |
| all | 2062 | 1146 | 1015 (0.89×) | 967 (0.84×) | 934 (0.81×) |
| *Frequency (calibration purchases)* | | | | | |
| 1 purchase | 1058 | 243 | 229 (0.94×) | 325 (1.34×) | 386 (1.59×) |
| 2 purchases | 453 | 227 | 220 (0.97×) | 185 (0.81×) | 195 (0.86×) |
| 3-4 purchases | 415 | 413 | 350 (0.85×) | 280 (0.68×) | 235 (0.57×) |
| 5+ purchases | 136 | 263 | 217 (0.82×) | 176 (0.67×) | 117 (0.45×) |
| *Recency within frequency (weeks silent)* | | | | | |
| 2 purchases, silent ≤ 13 w | 94 | 78 | 50 (0.64×) | 48 (0.61×) | 45 (0.57×) |
| 2 purchases, silent 14–52 w | 116 | 63 | 59 (0.94×) | 48 (0.76×) | 53 (0.84×) |
| 2 purchases, silent > 52 w | 243 | 86 | 110 (1.28×) | 90 (1.04×) | 98 (1.13×) |
| 3-4 purchases, silent ≤ 13 w | 183 | 245 | 168 (0.68×) | 146 (0.60×) | 112 (0.46×) |
| 3-4 purchases, silent 14–52 w | 139 | 130 | 117 (0.90×) | 87 (0.67×) | 80 (0.61×) |
| 3-4 purchases, silent > 52 w | 93 | 38 | 65 (1.71×) | 47 (1.24×) | 43 (1.14×) |
| 5+ purchases, silent ≤ 13 w | 93 | 219 | 156 (0.71×) | 129 (0.59×) | 85 (0.39×) |
| 5+ purchases, silent 14–52 w | 34 | 40 | 51 (1.28×) | 39 (0.99×) | 27 (0.68×) |
| 5+ purchases, silent > 52 w | 9† | 4 | 9 (2.31×) | 8 (1.93×) | 5 (1.37×) |
| *Overdue (silence ÷ own usual gap)* | | | | | |
| on schedule | 537 | 675 | 481 (0.71×) | 408 (0.60×) | 329 (0.49×) |
| overdue 1-3x | 268 | 160 | 183 (1.14×) | 137 (0.86×) | 126 (0.79×) |
| overdue >3x | 195 | 67 | 122 (1.83×) | 95 (1.42×) | 90 (1.35×) |
| *Momentum* | | | | | |
| rising | 360 | 462 | 292 (0.63×) | 260 (0.56×) | 215 (0.47×) |
| declining | 640 | 440 | 493 (1.12×) | 380 (0.86×) | 331 (0.75×) |
| one purchase week | 1062 | 244 | 230 (0.94×) | 326 (1.34×) | 388 (1.59×) |
| *Valendin et al. (2022) groups* | | | | | |
| At Risk | 236 | 0 | 218 | 171 | 138 |
| Opportunity | 89 | 355 | 45 (0.13×) | 44 (0.12×) | 42 (0.12×) |
| Other | 1737 | 791 | 752 (0.95×) | 752 (0.95×) | 754 (0.95×) |

#### multichannel — 2 years calibration, 1 year holdout

| customer group | customers | actual tx | Pareto/NBD | ValendinLSTM | LSTM + bounded32 |
|---|---:|---:|---:|---:|---:|
| *All customers* | | | | | |
| all | 1402 | 228 | 243 (1.06×) | 380 (1.67×) | 267 (1.17×) |
| *Frequency (calibration purchases)* | | | | | |
| 1 purchase | 1028 | 97 | 83 (0.86×) | 278 (2.87×) | 182 (1.87×) |
| 2 purchases | 242 | 53 | 69 (1.30×) | 66 (1.24×) | 51 (0.97×) |
| 3-4 purchases | 106 | 43 | 61 (1.41×) | 29 (0.67×) | 26 (0.61×) |
| 5+ purchases | 26† | 35 | 30 (0.85×) | 7 (0.20×) | 8 (0.22×) |
| *Recency within frequency (weeks silent)* | | | | | |
| 2 purchases, silent ≤ 13 w | 41 | 20 | 16 (0.78×) | 11 (0.56×) | 12 (0.60×) |
| 2 purchases, silent 14–52 w | 64 | 7 | 22 (3.10×) | 17 (2.48×) | 13 (1.89×) |
| 2 purchases, silent > 52 w | 137 | 26 | 32 (1.22×) | 37 (1.43×) | 26 (1.01×) |
| 3-4 purchases, silent ≤ 13 w | 31 | 8 | 22 (2.71×) | 8 (1.05×) | 10 (1.24×) |
| 3-4 purchases, silent 14–52 w | 47 | 26 | 27 (1.06×) | 13 (0.49×) | 10 (0.40×) |
| 3-4 purchases, silent > 52 w | 28† | 9 | 11 (1.27×) | 8 (0.84×) | 6 (0.63×) |
| 5+ purchases, silent ≤ 13 w | 15† | 26 | 19 (0.75×) | 4 (0.16×) | 5 (0.19×) |
| 5+ purchases, silent 14–52 w | 7† | 8 | 8 (0.94×) | 2 (0.23×) | 2 (0.22×) |
| 5+ purchases, silent > 52 w | 4† | 1 | 3 (2.75×) | 1 (1.08×) | 1 (0.84×) |
| *Overdue (silence ÷ own usual gap)* | | | | | |
| on schedule | 164 | 75 | 91 (1.22×) | 45 (0.59×) | 44 (0.58×) |
| overdue 1-3x | 102 | 39 | 40 (1.02×) | 28 (0.71×) | 20 (0.53×) |
| overdue >3x | 97 | 14 | 27 (1.95×) | 26 (1.87×) | 19 (1.36×) |
| *Momentum* | | | | | |
| rising | 129 | 63 | 64 (1.01×) | 35 (0.56×) | 34 (0.53×) |
| declining | 234 | 65 | 95 (1.46×) | 63 (0.98×) | 49 (0.76×) |
| one purchase week | 1039 | 100 | 84 (0.84×) | 281 (2.81×) | 184 (1.84×) |
| *Valendin et al. (2022) groups* | | | | | |
| At Risk | 288 | 0 | 110 | 78 | 64 |
| Opportunity | 20† | 50 | 2 (0.04×) | 5 (0.11×) | 4 (0.07×) |
| Other | 1094 | 178 | 130 (0.73×) | 297 (1.67×) | 199 (1.12×) |

#### electronics — 3 years calibration, 1 year holdout

| customer group | customers | actual tx | Pareto/NBD | LSTM + bounded32 |
|---|---:|---:|---:|---:|
| *All customers* | | | | |
| all | 829 | 1541 | 536 (0.35×) | 1343 (0.87×) |
| *Frequency (calibration purchases)* | | | | |
| 1 purchase | 111 | 144 | 14 (0.10×) | 103 (0.72×) |
| 2 purchases | 109 | 54 | 25 (0.46×) | 109 (2.01×) |
| 3-4 purchases | 150 | 130 | 51 (0.39×) | 166 (1.28×) |
| 5+ purchases | 459 | 1213 | 445 (0.37×) | 965 (0.80×) |
| *Recency within frequency (weeks silent)* | | | | |
| 2 purchases, silent ≤ 13 w | 1† | 0 | 1 | 2 |
| 2 purchases, silent 14–52 w | 15† | 21 | 8 (0.37×) | 18 (0.87×) |
| 2 purchases, silent > 52 w | 93 | 33 | 17 (0.51×) | 89 (2.69×) |
| 3-4 purchases, silent ≤ 13 w | 11† | 8 | 7 (0.92×) | 21 (2.63×) |
| 3-4 purchases, silent 14–52 w | 28† | 8 | 17 (2.13×) | 35 (4.41×) |
| 3-4 purchases, silent > 52 w | 111 | 114 | 27 (0.23×) | 110 (0.96×) |
| 5+ purchases, silent ≤ 13 w | 119 | 574 | 179 (0.31×) | 424 (0.74×) |
| 5+ purchases, silent 14–52 w | 161 | 468 | 196 (0.42×) | 340 (0.73×) |
| 5+ purchases, silent > 52 w | 179 | 171 | 71 (0.41×) | 201 (1.17×) |
| *Overdue (silence ÷ own usual gap)* | | | | |
| on schedule | 273 | 833 | 297 (0.36×) | 676 (0.81×) |
| overdue 1-3x | 149 | 312 | 139 (0.45×) | 246 (0.79×) |
| overdue >3x | 181 | 153 | 71 (0.46×) | 210 (1.37×) |
| *Momentum* | | | | |
| rising | 221 | 632 | 238 (0.38×) | 541 (0.86×) |
| declining | 382 | 666 | 269 (0.40×) | 591 (0.89×) |
| one purchase week | 226 | 243 | 28 (0.12×) | 211 (0.87×) |
| *Valendin et al. (2022) groups* | | | | |
| At Risk | 136 | 0 | 133 | 279 |
| Opportunity | 71 | 722 | 39 (0.05×) | 115 (0.16×) |
| Other | 622 | 819 | 364 (0.44×) | 949 (1.16×) |

#### gift — 3 years calibration, 1 year holdout

| customer group | customers | actual tx | Pareto/NBD | LSTM + bounded32 |
|---|---:|---:|---:|---:|
| *All customers* | | | | |
| all | 2062 | 1040 | 956 (0.92×) | 983 (0.95×) |
| *Frequency (calibration purchases)* | | | | |
| 1 purchase | 891 | 130 | 119 (0.92×) | 212 (1.63×) |
| 2 purchases | 413 | 126 | 139 (1.10×) | 142 (1.13×) |
| 3-4 purchases | 454 | 293 | 283 (0.96×) | 267 (0.91×) |
| 5+ purchases | 304 | 491 | 416 (0.85×) | 363 (0.74×) |
| *Recency within frequency (weeks silent)* | | | | |
| 2 purchases, silent ≤ 13 w | 45 | 23 | 19 (0.83×) | 29 (1.24×) |
| 2 purchases, silent 14–52 w | 73 | 31 | 30 (0.96×) | 31 (0.99×) |
| 2 purchases, silent > 52 w | 295 | 72 | 90 (1.25×) | 83 (1.15×) |
| 3-4 purchases, silent ≤ 13 w | 139 | 130 | 102 (0.79×) | 126 (0.97×) |
| 3-4 purchases, silent 14–52 w | 120 | 87 | 82 (0.94×) | 71 (0.82×) |
| 3-4 purchases, silent > 52 w | 195 | 76 | 98 (1.29×) | 70 (0.92×) |
| 5+ purchases, silent ≤ 13 w | 171 | 370 | 260 (0.70×) | 245 (0.66×) |
| 5+ purchases, silent 14–52 w | 94 | 107 | 125 (1.17×) | 98 (0.91×) |
| 5+ purchases, silent > 52 w | 39 | 14 | 31 (2.21×) | 20 (1.42×) |
| *Overdue (silence ÷ own usual gap)* | | | | |
| on schedule | 629 | 675 | 545 (0.81×) | 559 (0.83×) |
| overdue 1-3x | 257 | 158 | 178 (1.13×) | 121 (0.77×) |
| overdue >3x | 282 | 77 | 114 (1.48×) | 91 (1.18×) |
| *Momentum* | | | | |
| rising | 500 | 570 | 432 (0.76×) | 444 (0.78×) |
| declining | 668 | 340 | 405 (1.19×) | 326 (0.96×) |
| one purchase week | 894 | 130 | 119 (0.92×) | 213 (1.64×) |
| *Valendin et al. (2022) groups* | | | | |
| At Risk | 380 | 0 | 281 | 247 |
| Opportunity | 39 | 123 | 12 (0.09×) | 15 (0.12×) |
| Other | 1643 | 917 | 663 (0.72×) | 721 (0.79×) |

#### multichannel — 3 years calibration, 1 year holdout

| customer group | customers | actual tx | Pareto/NBD | LSTM + bounded32 |
|---|---:|---:|---:|---:|
| *All customers* | | | | |
| all | 1402 | 173 | 212 (1.23×) | 231 (1.33×) |
| *Frequency (calibration purchases)* | | | | |
| 1 purchase | 951 | 46 | 53 (1.15×) | 109 (2.37×) |
| 2 purchases | 262 | 41 | 54 (1.32×) | 47 (1.16×) |
| 3-4 purchases | 142 | 39 | 60 (1.55×) | 50 (1.28×) |
| 5+ purchases | 47 | 47 | 45 (0.95×) | 24 (0.52×) |
| *Recency within frequency (weeks silent)* | | | | |
| 2 purchases, silent ≤ 13 w | 22† | 4 | 7 (1.70×) | 7 (1.69×) |
| 2 purchases, silent 14–52 w | 39 | 11 | 11 (1.01×) | 9 (0.79×) |
| 2 purchases, silent > 52 w | 201 | 26 | 36 (1.39×) | 32 (1.23×) |
| 3-4 purchases, silent ≤ 13 w | 35 | 10 | 19 (1.87×) | 17 (1.65×) |
| 3-4 purchases, silent 14–52 w | 32 | 13 | 16 (1.22×) | 12 (0.93×) |
| 3-4 purchases, silent > 52 w | 75 | 16 | 26 (1.61×) | 21 (1.33×) |
| 5+ purchases, silent ≤ 13 w | 20† | 20 | 24 (1.20×) | 12 (0.61×) |
| 5+ purchases, silent 14–52 w | 16† | 22 | 15 (0.68×) | 8 (0.36×) |
| 5+ purchases, silent > 52 w | 11† | 5 | 6 (1.14×) | 4 (0.85×) |
| *Overdue (silence ÷ own usual gap)* | | | | |
| on schedule | 188 | 76 | 90 (1.18×) | 65 (0.86×) |
| overdue 1-3x | 124 | 29 | 43 (1.47×) | 29 (1.01×) |
| overdue >3x | 129 | 22 | 26 (1.20×) | 26 (1.19×) |
| *Momentum* | | | | |
| rising | 152 | 63 | 65 (1.04×) | 48 (0.77×) |
| declining | 289 | 64 | 93 (1.46×) | 72 (1.13×) |
| one purchase week | 961 | 46 | 53 (1.16×) | 110 (2.39×) |
| *Valendin et al. (2022) groups* | | | | |
| At Risk | 357 | 0 | 109 | 88 |
| Opportunity | 4† | 9 | 0 (0.02×) | 0 (0.05×) |
| Other | 1041 | 164 | 102 (0.62×) | 142 (0.87×) |

#### electronic_5y — 5 years calibration, 1 year holdout

| customer group | customers | actual tx | Pareto/NBD | ValendinLSTM | LSTM + bounded32 | Transformer |
|---|---:|---:|---:|---:|---:|---:|
| *All customers* | | | | | | |
| all | 3755 | 2153 | 1796 (0.83×) | 2451 (1.14×) | 2340 (1.09×) | 2422 (1.12×) |
| *Frequency (calibration purchases)* | | | | | | |
| 1 purchase | 861 | 126 | 54 (0.43×) | 204 (1.62×) | 234 (1.86×) | 172 (1.37×) |
| 2 purchases | 690 | 163 | 118 (0.72×) | 215 (1.32×) | 218 (1.34×) | 198 (1.21×) |
| 3-4 purchases | 951 | 344 | 314 (0.91×) | 432 (1.26×) | 413 (1.20×) | 424 (1.23×) |
| 5+ purchases | 1253 | 1520 | 1311 (0.86×) | 1600 (1.05×) | 1476 (0.97×) | 1627 (1.07×) |
| *Recency within frequency (weeks silent)* | | | | | | |
| 2 purchases, silent ≤ 13 w | 17† | 8 | 6 (0.76×) | 13 (1.57×) | 11 (1.33×) | 9 (1.09×) |
| 2 purchases, silent 14–52 w | 61 | 26 | 21 (0.80×) | 29 (1.10×) | 25 (0.95×) | 20 (0.78×) |
| 2 purchases, silent > 52 w | 612 | 129 | 91 (0.71×) | 174 (1.35×) | 183 (1.42×) | 169 (1.31×) |
| 3-4 purchases, silent ≤ 13 w | 83 | 50 | 49 (0.97×) | 78 (1.56×) | 70 (1.40×) | 59 (1.19×) |
| 3-4 purchases, silent 14–52 w | 193 | 101 | 101 (1.00×) | 119 (1.18×) | 107 (1.06×) | 98 (0.97×) |
| 3-4 purchases, silent > 52 w | 675 | 193 | 164 (0.85×) | 235 (1.22×) | 236 (1.22×) | 267 (1.38×) |
| 5+ purchases, silent ≤ 13 w | 339 | 793 | 600 (0.76×) | 808 (1.02×) | 723 (0.91×) | 715 (0.90×) |
| 5+ purchases, silent 14–52 w | 420 | 462 | 502 (1.09×) | 516 (1.12×) | 486 (1.05×) | 542 (1.17×) |
| 5+ purchases, silent > 52 w | 494 | 265 | 208 (0.79×) | 275 (1.04×) | 266 (1.01×) | 370 (1.39×) |
| *Overdue (silence ÷ own usual gap)* | | | | | | |
| on schedule | 1125 | 1232 | 1049 (0.85×) | 1359 (1.10×) | 1242 (1.01×) | 1184 (0.96×) |
| overdue 1-3x | 751 | 441 | 495 (1.12×) | 499 (1.13×) | 475 (1.08×) | 584 (1.32×) |
| overdue >3x | 972 | 344 | 196 (0.57×) | 378 (1.10×) | 376 (1.09×) | 472 (1.37×) |
| *Momentum* | | | | | | |
| rising | 925 | 1001 | 799 (0.80×) | 1098 (1.10×) | 994 (0.99×) | 937 (0.94×) |
| declining | 1923 | 1016 | 940 (0.93×) | 1138 (1.12×) | 1100 (1.08×) | 1303 (1.28×) |
| one purchase week | 907 | 136 | 57 (0.42×) | 215 (1.58×) | 246 (1.81×) | 182 (1.34×) |
| *Valendin et al. (2022) groups* | | | | | | |
| At Risk | 651 | 0 | 500 | 585 | 558 | 640 |
| Opportunity | 45 | 192 | 11 (0.06×) | 20 (0.10×) | 19 (0.10×) | 16 (0.08×) |
| Other | 3059 | 1961 | 1286 (0.66×) | 1846 (0.94×) | 1763 (0.90×) | 1765 (0.90×) |
