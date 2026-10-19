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
11. [Next: choosing the customer groups](#11-next-choosing-the-customer-groups)
12. [Reproducing](#12-reproducing)

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
4. **Train on the model's own simulated histories.** During the refit, recompute the AR
   features from sampled counts, so the model learns states it will meet in the holdout.
   It targets both sides of §4's drift, at higher cost than items 1–3.
5. **Cheap checks.**
   - `bounded32` on CDNOW without `week_sin` / `week_cos`, to tell §6's two candidate
     causes apart.
   - An open top class ("6 or more") for the old electronics panel (§9).

## 11. Next: choosing the customer groups

Open, to settle together. The groups here (F × R, one-timers, hot/cold) were chosen to
expose the failure. A thesis chapter needs a small set justified by the literature
(`docs/cohort-analysis.md` §1) and by the four panels, and stable enough to report on
every window. To be discussed.

## 12. Reproducing

From the repo root with the project venv:

```
python .scratch/customer-analysis-2/collect.py .scratch/customer-analysis-2/results/groups.pkl
python .scratch/customer-analysis-2/tables.py  .scratch/customer-analysis-2/results/groups.pkl
```

`collect.py` reads every stored forecast in §1 and writes one row per (window, model, run,
grouping, group). `tables.py` prints every table in this doc;
`.scratch/customer-analysis-2/results/tables.md` holds its output. The CDNOW week-by-week
figures in §6 came from a one-off reading of the same prediction files; they are the only
numbers here that `tables.py` does not print.
