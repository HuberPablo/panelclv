# Insights from the Pareto/NBD synthetic grid studies

As of 2026-09-28. Exported from the Claude Doc
<https://claude.ai/code/artifact/bff00b0b-93dd-4be8-8f71-5ea53e3d965b>; all numbers were
recomputed from the stored forecasts under `Studies/seasonal_4x4x10*`.

## How the datasets were made

Every panel was simulated from a known Pareto/NBD process, so the true model is known and
Pareto/NBD is the correct model by construction. That makes it the ceiling the neural models
are measured against.

**The process, per customer** (`data_preparation/pareto_nbd_simulation.py`):

- Purchase rate λ ~ Gamma(r, α). While alive, weekly purchases are Poisson(λ).
- Dropout rate μ ~ Gamma(s, β). Lifetime τ ~ Exponential(μ). After τ the customer never buys again.
- A fixed seasonal multiplier scales every customer's weekly rate: peaks at weeks 12, 25, 30
  and 47, amplitude 1.5, width 3 weeks. It averages to 1 over the year, so the annual rate is
  unchanged.

**The grid** (`grids/seasonal_4x4x10.py`): two axes, set in plain terms and turned into α
and β by the generator.

| Setting | Values |
| --- | --- |
| Mean weekly purchase rate (r/α) | 0.01, 0.05, 0.10, 0.30 |
| Churn rate: share of customers dropped out by week 52 | 20%, 40%, 60%, 80% |
| Shape parameters | r = s = 2.0 |
| Replicate panels per cell | 10 (different seeds, base seed 42) |
| Total panels | 4 × 4 × 10 = 160 |
| Customers per panel | 1,000 (a second grid, `seasonal_4x4x10_n3000`, uses 3,000) |
| Length | 156 weeks, 1999–2001 |

**Windows.** 1999 trains, 2000 is the validation window (the temporal split of ADR-0001),
2001 is the holdout. Customers with no purchase in calibration are dropped, as in a real
cohort. Weekly counts are capped at 6, which gives the models a 7-class softmax head.

**What the models see.** In the baseline arm, only each customer's weekly count plus the
calendar (year index and week sin/cos). There are no covariates; the other arms add
encodings of the customer's own history (next section). The true λ, μ and τ are saved per
customer but never shown to a model.

## What was trained and how it was scored

Three models ran on all 160 panels: an LSTM, a Transformer and the Pareto/NBD benchmark.
The frozen ValendinLSTM did not run on this grid. It reads only embedded columns (count and
week), and every arm here, the `no_ar` baseline included, carries the continuous calendar
columns (year index, week sin/cos), which it refuses (F11). It did run without covariates
on the real panels (CDNOW, electronics), where the calendar can be left out.

**Architectures and search** (`grids/seasonal_4x4x10.py`). Each neural study is one
100-trial Optuna search on one panel, then a refit and a 200-path Monte Carlo forecast.

| Model | Searched hyperparameters | Training |
| --- | --- | --- |
| LSTM | batch {32, 64, 128, 256}; learning rate 1e-4–1e-2 (log); embedding {64, 128, 256}; hidden {32, 64, 128}; dense {32, 64, 128}; dropout {0, 0.2, 0.4} | up to 100 epochs, patience 7, cross-entropy |
| Transformer | batch {32, 64, 128, 256}; d_model {32, 64, 128}; heads {2, 4, 8}; layers 1–3; dropout {0, 0.1, 0.2, 0.3}; learning rate 1e-4–3e-3 (log) | same |
| Pareto/NBD | none: one hierarchical-Bayes MCMC fit (BTYDplus defaults) | — |

Both neural models use the `valendin` embedder. The `projected` embedder was declared but
cut on cost.

**Arms: how a customer's history is fed in.** Three AR encodings crossed with two cluster
settings = 6 arms per neural model, 12 neural trees plus Pareto/NBD, 2,080 studies (vast.ai,
4–6 September 2026).

| Arm part | Value | What the model gets |
| --- | --- | --- |
| AR encoding | `no_ar` | count and calendar only (the baseline) |
| | `ar_unbounded` | recency, frequency and age as counters: Pareto/NBD's own sufficient statistic (t_x, x, T) |
| | `ar_bounded` | the same history as 0/1 flags: active in the last 2 / 4 / 8 / 16 / 32 weeks, plus has-bought-before |
| Cluster | `no_cluster` / `kmeans_8` | none, or a fixed k-means label (K = 8) on (t_x, x, T) |

A second grid with 3,000 customers per panel (17–18 September) re-ran two LSTM arms
(`no_ar` and `ar_bounded`, no cluster) and Pareto/NBD. The Transformer was not run there.

**Metrics.** One study per panel, so every spread below is across panels, not training runs.

| Metric | What it measures | Good value |
| --- | --- | --- |
| RMSE (customer totals) | error in each customer's holdout-year total | low |
| Bias % | total forecast vs total actual over the holdout year | 0 |
| MAPE (aggregate) | error in the weekly total, week by week | low |
| Win rate | share of panels where a model's MAPE beats Pareto/NBD's on the same panel | high |
| Spearman | rank correlation of each customer's predicted and actual holdout-year total, over the panel's customers | 1 |
| Validation CE | cross-entropy per customer-week of the chosen trial on the 2000 validation window (the Optuna objective, one step ahead with the true counts fed in). Neural models only: Pareto/NBD emits no class distribution. Depends on the panel's rate and churn, so compare within one cell | low |
| Shape correlation | correlation of predicted and actual weekly totals; ignores level | 1 |
| Alive ratio R_A / dead leakage L_D | forecast volume before / after each customer's true death week, over true volume (uses the hidden truth) | 1 / 0 |

The Results tables show five metrics, each as the mean over panels with a 95% t-interval
across panels in brackets; the later sections show **RMSE on customer totals / bias % /
MAPE** as means only. RMSE is
computed on each customer's holdout-year total, the paper's definition. It grows with the
purchase rate (about 0.6 at rate 0.01, about 5 at rate 0.30), so it is compared within one
rate only. The per-week RMSE in `results.csv` is not used: 10 of 13 trees score 0.18 on it.

**Tests.** Wilcoxon signed-rank, paired on the same panels (per rate: 40 panels). Mann–Whitney
U where panels differ (1,000 vs 3,000 customers). The effect is the median of per-panel
differences.

## Results

The ranking depends on both the purchase rate and the churn rate. On MAPE, Pareto/NBD leads
every cell at rates 0.01–0.05 and at rate 0.10 with churn 20–40%; the LSTM with bounded flags
leads at rate 0.10 with churn 60–80% and at rate 0.30. The Transformer leads no cell. On
Spearman, Pareto/NBD leads all 16 cells, each time significantly. Two Transformer
`ar_unbounded` studies whose forecasts did not match their stored results are left out.

**Setting for this section.** Seasonal panels (4 peaks, amplitude 1.5), 1,000 customers, 13
trees (Pareto/NBD and 6 arms each of the LSTM and the Transformer). Four purchase rates ×
four churn rates (20, 40, 60 and 80% of customers dropped out by week 52), 10 panels per
cell. A metric cell is the mean over panels with a 95% t-interval across panels. RMSE is on
customer totals and grows with the rate, and CE depends on the panel, so compare both within
one table only. "Beats P/NBD" is the share of the table's panels where the tree's MAPE beats
Pareto/NBD's on the same panel.

### By purchase rate and churn

One table per rate × churn cell, 10 panels per row (9 where a study was left out).

**Rate 0.01, churn 20%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| **Pareto/NBD** | — | 0.79 [0.75, 0.83] | +12 [+9, +16] | 48 [45, 51] | 0.22 [0.19, 0.25] | — | — |
| LSTM | `ar_bounded` | 0.86 [0.80, 0.93] | +44 [+17, +71] | 65 [47, 82] | −0.03 [−0.10, 0.03] | 0.078 [0.074, 0.082] | 30% |
| LSTM | `ar_bounded` + `kmeans_8` | 0.88 [0.84, 0.93] | +56 [+36, +76] | 72 [61, 84] | 0.13 [0.09, 0.17] | 0.077 [0.073, 0.080] | 0% |
| LSTM | `no_ar` | 0.87 [0.82, 0.92] | +59 [+43, +76] | 71 [58, 83] | 0.01 [−0.02, 0.04] | 0.079 [0.076, 0.083] | 0% |
| LSTM | `no_ar` + `kmeans_8` | 0.91 [0.85, 0.97] | +77 [+52, +102] | 86 [66, 107] | 0.14 [0.11, 0.17] | 0.078 [0.075, 0.082] | 0% |
| LSTM | `ar_unbounded` | 6.70 [0.14, 13.27] | +502 [−91, +1096] | 513 [−77, 1104] | −0.16 [−0.21, −0.11] | 0.078 [0.074, 0.082] | 10% |
| LSTM | `ar_unbounded` + `kmeans_8` | 5.09 [2.01, 8.17] | +235 [+72, +397] | 250 [89, 411] | −0.08 [−0.14, −0.03] | 0.075 [0.071, 0.079] | 0% |
| Transformer | `ar_bounded` | 0.85 [0.79, 0.92] | +36 [+4, +67] | 62 [44, 81] | 0.03 [−0.00, 0.06] | 0.077 [0.074, 0.081] | 30% |
| Transformer | `ar_bounded` + `kmeans_8` | 0.90 [0.82, 0.99] | +23 [−31, +78] | 73 [41, 105] | 0.16 [0.13, 0.20] | 0.074 [0.070, 0.078] | 20% |
| Transformer | `no_ar` | 0.84 [0.78, 0.90] | +19 [−12, +51] | 58 [47, 69] | 0.05 [0.02, 0.08] | 0.077 [0.074, 0.081] | 30% |
| Transformer | `no_ar` + `kmeans_8` | 0.86 [0.80, 0.92] | −3 [−34, +28] | 55 [47, 64] | 0.12 [0.08, 0.16] | 0.077 [0.074, 0.081] | 40% |
| Transformer | `ar_unbounded` | 0.83 [0.79, 0.87] | +7 [−13, +28] | 49 [42, 57] | −0.05 [−0.10, 0.01] | 0.077 [0.073, 0.081] | 60% |
| Transformer | `ar_unbounded` + `kmeans_8` | 0.87 [0.82, 0.92] | −13 [−51, +26] | 65 [55, 75] | 0.03 [−0.04, 0.10] | 0.072 [0.068, 0.077] | 0% |

**Rate 0.01, churn 40%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| **Pareto/NBD** | — | 0.68 [0.64, 0.72] | +34 [+26, +42] | 66 [62, 70] | 0.25 [0.23, 0.28] | — | — |
| LSTM | `ar_bounded` | 0.78 [0.74, 0.83] | +114 [+96, +133] | 121 [104, 137] | −0.09 [−0.13, −0.06] | 0.067 [0.064, 0.071] | 0% |
| LSTM | `ar_bounded` + `kmeans_8` | 0.80 [0.75, 0.84] | +116 [+95, +136] | 121 [103, 140] | 0.11 [0.07, 0.15] | 0.067 [0.064, 0.071] | 0% |
| LSTM | `no_ar` | 0.80 [0.76, 0.84] | +132 [+110, +154] | 136 [116, 157] | 0.01 [−0.03, 0.05] | 0.068 [0.064, 0.072] | 0% |
| LSTM | `no_ar` + `kmeans_8` | 0.81 [0.78, 0.84] | +136 [+120, +152] | 140 [125, 155] | 0.14 [0.10, 0.17] | 0.067 [0.063, 0.071] | 0% |
| LSTM | `ar_unbounded` | 1.89 [0.28, 3.50] | +182 [+104, +260] | 189 [111, 266] | −0.23 [−0.25, −0.20] | 0.068 [0.064, 0.071] | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 11.37 [3.70, 19.04] | +1300 [+83, +2517] | 1304 [88, 2519] | −0.12 [−0.20, −0.03] | 0.066 [0.062, 0.070] | 0% |
| Transformer | `ar_bounded` | 0.71 [0.68, 0.75] | +27 [−16, +69] | 75 [62, 89] | −0.02 [−0.05, 0.01] | 0.065 [0.061, 0.069] | 40% |
| Transformer | `ar_bounded` + `kmeans_8` | 0.80 [0.66, 0.94] | +64 [−7, +134] | 102 [51, 153] | 0.16 [0.12, 0.19] | 0.062 [0.059, 0.066] | 40% |
| Transformer | `no_ar` | 0.75 [0.67, 0.83] | +61 [−2, +123] | 96 [54, 138] | 0.02 [−0.04, 0.07] | 0.066 [0.062, 0.069] | 60% |
| Transformer | `no_ar` + `kmeans_8` | 0.92 [0.72, 1.12] | +117 [+16, +218] | 148 [68, 228] | 0.12 [0.08, 0.17] | 0.065 [0.061, 0.069] | 20% |
| Transformer | `ar_unbounded` | 0.72 [0.68, 0.76] | +18 [−34, +70] | 78 [52, 103] | 0.00 [−0.09, 0.09] | 0.065 [0.061, 0.069] | 60% |
| Transformer | `ar_unbounded` + `kmeans_8` | 0.75 [0.71, 0.79] | +1 [−55, +57] | 83 [60, 106] | −0.00 [−0.06, 0.06] | 0.061 [0.057, 0.064] | 20% |

**Rate 0.01, churn 60%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| **Pareto/NBD** | — | 0.57 [0.53, 0.61] | +41 [+30, +52] | 88 [83, 94] | 0.26 [0.22, 0.31] | — | — |
| LSTM | `ar_bounded` | 0.71 [0.64, 0.77] | +190 [+137, +242] | 196 [146, 246] | −0.04 [−0.12, 0.03] | 0.051 [0.048, 0.055] | 0% |
| LSTM | `ar_bounded` + `kmeans_8` | 0.79 [0.73, 0.84] | +241 [+210, +272] | 245 [216, 274] | 0.07 [0.03, 0.11] | 0.051 [0.048, 0.054] | 0% |
| LSTM | `no_ar` | 0.83 [0.78, 0.88] | +290 [+259, +320] | 291 [262, 321] | 0.01 [−0.04, 0.06] | 0.052 [0.048, 0.056] | 0% |
| LSTM | `no_ar` + `kmeans_8` | 0.82 [0.75, 0.88] | +271 [+232, +311] | 273 [234, 312] | 0.16 [0.11, 0.21] | 0.051 [0.047, 0.054] | 0% |
| LSTM | `ar_unbounded` | 3.39 [0.81, 5.97] | +514 [+302, +726] | 514 [302, 726] | −0.20 [−0.26, −0.14] | 0.052 [0.048, 0.056] | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 5.48 [0.80, 10.16] | +900 [+270, +1530] | 904 [276, 1531] | −0.04 [−0.15, 0.06] | 0.051 [0.047, 0.054] | 0% |
| Transformer | `ar_bounded` | 0.60 [0.56, 0.64] | +50 [+10, +91] | 98 [78, 119] | 0.06 [0.01, 0.11] | 0.049 [0.045, 0.053] | 40% |
| Transformer | `ar_bounded` + `kmeans_8` | 0.63 [0.56, 0.70] | +43 [−14, +99] | 103 [74, 132] | 0.18 [0.11, 0.24] | 0.047 [0.043, 0.050] | 50% |
| Transformer | `no_ar` | 0.63 [0.59, 0.67] | +96 [+33, +159] | 129 [86, 173] | 0.07 [0.03, 0.10] | 0.050 [0.046, 0.054] | 30% |
| Transformer | `no_ar` + `kmeans_8` | 0.64 [0.59, 0.70] | +65 [−2, +132] | 120 [85, 155] | 0.16 [0.11, 0.22] | 0.047 [0.043, 0.051] | 40% |
| Transformer | `ar_unbounded` | 0.75 [0.59, 0.91] | +68 [+4, +132] | 127 [79, 175] | 0.06 [0.01, 0.10] | 0.049 [0.045, 0.053] | 50% |
| Transformer | `ar_unbounded` + `kmeans_8` | 0.68 [0.57, 0.79] | +68 [−1, +137] | 120 [74, 166] | 0.10 [0.04, 0.15] | 0.044 [0.041, 0.048] | 30% |

**Rate 0.01, churn 80%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| **Pareto/NBD** | — | 0.42 [0.38, 0.47] | +54 [+29, +80] | 159 [132, 186] | 0.20 [0.15, 0.25] | — | — |
| LSTM | `ar_bounded` | 0.71 [0.60, 0.82] | +576 [+278, +874] | 591 [304, 878] | −0.07 [−0.10, −0.03] | 0.034 [0.030, 0.038] | 0% |
| LSTM | `ar_bounded` + `kmeans_8` | 0.63 [0.52, 0.74] | +429 [+255, +603] | 445 [280, 611] | 0.07 [0.01, 0.13] | 0.033 [0.030, 0.037] | 0% |
| LSTM | `no_ar` | 0.86 [0.77, 0.94] | +755 [+467, +1044] | 759 [474, 1045] | −0.02 [−0.06, 0.02] | 0.034 [0.030, 0.038] | 0% |
| LSTM | `no_ar` + `kmeans_8` | 0.85 [0.78, 0.92] | +725 [+483, +968] | 729 [489, 969] | 0.11 [0.06, 0.17] | 0.033 [0.030, 0.037] | 0% |
| LSTM | `ar_unbounded` | 1.53 [0.44, 2.62] | +821 [+430, +1212] | 838 [463, 1213] | −0.12 [−0.16, −0.08] | 0.034 [0.031, 0.038] | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 6.26 [0.11, 12.42] | +1993 [+282, +3704] | 2015 [311, 3719] | −0.02 [−0.09, 0.05] | 0.033 [0.030, 0.036] | 0% |
| Transformer | `ar_bounded` | 0.44 [0.37, 0.50] | +89 [+34, +144] | 176 [145, 206] | 0.08 [0.01, 0.15] | 0.031 [0.027, 0.034] | 40% |
| Transformer | `ar_bounded` + `kmeans_8` | 0.51 [0.41, 0.60] | +109 [+31, +187] | 196 [143, 249] | 0.11 [0.04, 0.17] | 0.028 [0.025, 0.032] | 40% |
| Transformer | `no_ar` | 0.44 [0.39, 0.50] | +121 [+50, +192] | 197 [148, 246] | 0.07 [0.02, 0.13] | 0.031 [0.028, 0.035] | 20% |
| Transformer | `no_ar` + `kmeans_8` | 0.48 [0.40, 0.55] | +104 [−3, +210] | 195 [114, 276] | 0.15 [0.11, 0.19] | 0.030 [0.027, 0.034] | 60% |
| Transformer | `ar_unbounded` | 0.70 [0.17, 1.23] | +264 [−37, +566] | 345 [54, 636] | 0.00 [−0.10, 0.10] | 0.031 [0.027, 0.034] | 30% |
| Transformer | `ar_unbounded` + `kmeans_8` | 0.51 [0.41, 0.60] | +120 [+4, +236] | 216 [136, 295] | 0.09 [0.02, 0.16] | 0.027 [0.024, 0.031] | 30% |

**Rate 0.05, churn 20%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| **Pareto/NBD** | — | 1.92 [1.88, 1.96] | +8 [+5, +11] | 35 [33, 37] | 0.58 [0.55, 0.60] | — | — |
| LSTM | `ar_bounded` | 2.29 [2.22, 2.35] | +47 [+37, +56] | 48 [40, 57] | 0.50 [0.47, 0.54] | 0.174 [0.172, 0.176] | 10% |
| LSTM | `ar_bounded` + `kmeans_8` | 2.20 [2.10, 2.29] | +29 [+24, +34] | 39 [36, 42] | 0.45 [0.42, 0.47] | 0.169 [0.165, 0.172] | 20% |
| LSTM | `no_ar` | 2.33 [2.17, 2.49] | +42 [+31, +53] | 45 [35, 54] | 0.29 [0.10, 0.47] | 0.175 [0.172, 0.178] | 20% |
| LSTM | `no_ar` + `kmeans_8` | 2.26 [2.14, 2.38] | +40 [+31, +50] | 47 [39, 54] | 0.46 [0.42, 0.50] | 0.169 [0.166, 0.172] | 0% |
| LSTM | `ar_unbounded` | 14.23 [10.96, 17.50] | +240 [+173, +307] | 240 [173, 307] | 0.44 [0.42, 0.47] | 0.177 [0.174, 0.180] | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 11.36 [8.81, 13.92] | +188 [+144, +233] | 189 [145, 233] | 0.46 [0.42, 0.50] | 0.174 [0.171, 0.177] | 0% |
| Transformer | `ar_bounded` | 2.56 [2.38, 2.75] | +61 [+36, +87] | 68 [52, 84] | 0.45 [0.41, 0.49] | 0.174 [0.171, 0.176] | 20% |
| Transformer | `ar_bounded` + `kmeans_8` | 2.65 [2.46, 2.83] | +77 [+61, +93] | 79 [65, 93] | 0.51 [0.47, 0.54] | 0.168 [0.164, 0.171] | 0% |
| Transformer | `no_ar` | 2.70 [2.49, 2.90] | +72 [+52, +91] | 75 [59, 91] | 0.31 [0.19, 0.43] | 0.175 [0.173, 0.178] | 10% |
| Transformer | `no_ar` + `kmeans_8` | 2.44 [2.22, 2.66] | +62 [+38, +86] | 67 [48, 87] | 0.49 [0.45, 0.53] | 0.169 [0.166, 0.172] | 20% |
| Transformer | `ar_unbounded` | 2.69 [2.42, 2.95] | +90 [+65, +115] | 91 [68, 114] | 0.52 [0.50, 0.54] | 0.173 [0.171, 0.176] | 10% |
| Transformer | `ar_unbounded` + `kmeans_8` | 2.66 [2.35, 2.97] | +61 [+33, +88] | 69 [52, 86] | 0.47 [0.40, 0.54] | 0.168 [0.165, 0.171] | 20% |

**Rate 0.05, churn 40%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| **Pareto/NBD** | — | 1.63 [1.60, 1.66] | +15 [+10, +20] | 39 [36, 41] | 0.60 [0.58, 0.62] | — | — |
| LSTM | `ar_bounded` | 1.88 [1.82, 1.94] | +52 [+41, +63] | 56 [46, 65] | 0.54 [0.51, 0.57] | 0.138 [0.135, 0.141] | 10% |
| LSTM | `ar_bounded` + `kmeans_8` | 2.00 [1.90, 2.10] | +61 [+45, +77] | 64 [49, 79] | 0.45 [0.41, 0.48] | 0.135 [0.131, 0.139] | 0% |
| LSTM | `no_ar` | 2.24 [2.13, 2.36] | +98 [+84, +112] | 99 [85, 112] | 0.20 [0.02, 0.37] | 0.144 [0.141, 0.148] | 0% |
| LSTM | `no_ar` + `kmeans_8` | 2.09 [2.04, 2.14] | +81 [+69, +93] | 82 [71, 94] | 0.46 [0.41, 0.51] | 0.137 [0.134, 0.140] | 0% |
| LSTM | `ar_unbounded` | 10.38 [7.37, 13.38] | +313 [+241, +386] | 313 [241, 386] | 0.39 [0.35, 0.43] | 0.145 [0.141, 0.148] | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 9.40 [3.61, 15.20] | +221 [+60, +381] | 229 [75, 384] | 0.37 [0.31, 0.44] | 0.140 [0.136, 0.144] | 30% |
| Transformer | `ar_bounded` | 1.91 [1.68, 2.14] | +53 [+13, +93] | 65 [31, 98] | 0.55 [0.52, 0.58] | 0.138 [0.135, 0.141] | 30% |
| Transformer | `ar_bounded` + `kmeans_8` | 2.00 [1.88, 2.11] | +48 [+28, +67] | 56 [41, 71] | 0.51 [0.49, 0.54] | 0.133 [0.130, 0.137] | 30% |
| Transformer | `no_ar` | 2.28 [2.09, 2.47] | +108 [+78, +138] | 110 [81, 138] | 0.42 [0.37, 0.46] | 0.142 [0.139, 0.145] | 0% |
| Transformer | `no_ar` + `kmeans_8` | 2.37 [2.16, 2.59] | +105 [+71, +139] | 107 [74, 139] | 0.48 [0.44, 0.52] | 0.134 [0.130, 0.137] | 0% |
| Transformer | `ar_unbounded` | 2.30 [1.82, 2.79] | +101 [+66, +135] | 103 [71, 135] | 0.51 [0.43, 0.60] | 0.138 [0.135, 0.142] | 10% |
| Transformer | `ar_unbounded` + `kmeans_8` | 2.15 [1.92, 2.37] | +68 [+28, +107] | 78 [46, 111] | 0.46 [0.37, 0.55] | 0.134 [0.130, 0.137] | 20% |

**Rate 0.05, churn 60%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| **Pareto/NBD** | — | 1.32 [1.26, 1.38] | +9 [0, +18] | 42 [39, 45] | 0.56 [0.55, 0.58] | — | — |
| LSTM | `ar_bounded` | 1.43 [1.35, 1.52] | +60 [+41, +79] | 67 [55, 79] | 0.52 [0.50, 0.54] | 0.100 [0.097, 0.103] | 10% |
| LSTM | `ar_bounded` + `kmeans_8` | 1.80 [1.68, 1.92] | +108 [+83, +133] | 110 [86, 134] | 0.43 [0.39, 0.47] | 0.098 [0.095, 0.101] | 0% |
| LSTM | `no_ar` | 1.95 [1.84, 2.07] | +157 [+135, +180] | 158 [135, 180] | 0.03 [−0.00, 0.06] | 0.109 [0.106, 0.112] | 0% |
| LSTM | `no_ar` + `kmeans_8` | 1.99 [1.86, 2.12] | +172 [+149, +195] | 172 [149, 195] | 0.38 [0.31, 0.46] | 0.102 [0.099, 0.104] | 0% |
| LSTM | `ar_unbounded` | 10.76 [7.97, 13.54] | +462 [+281, +642] | 462 [281, 642] | 0.35 [0.32, 0.39] | 0.108 [0.105, 0.112] | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 8.50 [4.22, 12.79] | +357 [+110, +603] | 359 [114, 604] | 0.36 [0.27, 0.46] | 0.104 [0.100, 0.107] | 0% |
| Transformer | `ar_bounded` | 1.47 [1.37, 1.57] | +66 [+30, +103] | 77 [49, 105] | 0.52 [0.50, 0.53] | 0.100 [0.096, 0.103] | 20% |
| Transformer | `ar_bounded` + `kmeans_8` | 2.02 [1.78, 2.26] | +152 [+99, +205] | 155 [104, 205] | 0.47 [0.43, 0.51] | 0.096 [0.092, 0.099] | 0% |
| Transformer | `no_ar` | 1.87 [1.65, 2.08] | +144 [+78, +210] | 149 [87, 210] | 0.37 [0.31, 0.43] | 0.105 [0.101, 0.108] | 10% |
| Transformer | `no_ar` + `kmeans_8` | 1.91 [1.69, 2.12] | +131 [+77, +185] | 134 [84, 185] | 0.48 [0.46, 0.50] | 0.096 [0.093, 0.099] | 0% |
| Transformer | `ar_unbounded` | 1.73 [1.58, 1.88] | +121 [+79, +164] | 124 [84, 164] | 0.49 [0.44, 0.54] | 0.100 [0.097, 0.104] | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 1.90 [1.46, 2.34] | +98 [+32, +163] | 103 [40, 167] | 0.43 [0.35, 0.52] | 0.094 [0.091, 0.098] | 0% |

**Rate 0.05, churn 80%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| **Pareto/NBD** | — | 0.89 [0.85, 0.93] | +13 [−3, +29] | 58 [52, 65] | 0.44 [0.41, 0.47] | — | — |
| LSTM | `ar_bounded` | 0.97 [0.92, 1.02] | +19 [−3, +41] | 60 [50, 70] | 0.40 [0.36, 0.44] | 0.061 [0.056, 0.066] | 70% |
| LSTM | `ar_bounded` + `kmeans_8` | 1.58 [1.30, 1.86] | +287 [+159, +414] | 290 [165, 415] | 0.36 [0.33, 0.39] | 0.061 [0.056, 0.065] | 0% |
| LSTM | `no_ar` | 1.88 [1.71, 2.05] | +497 [+441, +553] | 497 [441, 553] | −0.01 [−0.03, 0.02] | 0.070 [0.064, 0.075] | 0% |
| LSTM | `no_ar` + `kmeans_8` | 1.99 [1.87, 2.11] | +506 [+431, +582] | 506 [431, 582] | 0.34 [0.29, 0.39] | 0.064 [0.059, 0.069] | 0% |
| LSTM | `ar_unbounded` | 12.14 [6.41, 17.87] | +1314 [+687, +1941] | 1315 [689, 1941] | 0.26 [0.19, 0.34] | 0.067 [0.061, 0.073] | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 11.44 [7.38, 15.51] | +908 [+509, +1307] | 919 [535, 1304] | 0.32 [0.26, 0.38] | 0.065 [0.060, 0.070] | 0% |
| Transformer | `ar_bounded` | 1.04 [0.98, 1.10] | +76 [+15, +137] | 102 [58, 146] | 0.40 [0.37, 0.44] | 0.061 [0.056, 0.066] | 20% |
| Transformer | `ar_bounded` + `kmeans_8` | 1.11 [1.06, 1.16] | +62 [+3, +120] | 94 [55, 134] | 0.37 [0.33, 0.42] | 0.058 [0.053, 0.062] | 30% |
| Transformer | `no_ar` | 1.13 [1.06, 1.19] | +110 [+55, +166] | 122 [76, 168] | 0.32 [0.27, 0.36] | 0.066 [0.061, 0.070] | 20% |
| Transformer | `no_ar` + `kmeans_8` | 1.29 [1.07, 1.50] | +161 [+49, +273] | 181 [85, 278] | 0.36 [0.29, 0.42] | 0.059 [0.054, 0.063] | 10% |
| Transformer | `ar_unbounded` | 1.09 [0.97, 1.22] | +70 [+12, +128] | 100 [54, 146] | 0.19 [0.02, 0.37] | 0.062 [0.057, 0.066] | 10% |
| Transformer | `ar_unbounded` + `kmeans_8` | 1.15 [0.86, 1.45] | +63 [−9, +135] | 97 [39, 155] | 0.38 [0.35, 0.41] | 0.057 [0.052, 0.062] | 30% |

**Rate 0.10, churn 20%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| **Pareto/NBD** | — | 2.91 [2.85, 2.96] | +3 [+1, +5] | 31 [30, 32] | 0.72 [0.71, 0.74] | — | — |
| LSTM | `ar_bounded` | 3.32 [3.17, 3.46] | +29 [+16, +43] | 32 [22, 43] | 0.69 [0.68, 0.71] | 0.261 [0.254, 0.268] | 40% |
| LSTM | `ar_bounded` + `kmeans_8` | 3.46 [3.38, 3.54] | +28 [+23, +34] | 33 [28, 38] | 0.60 [0.56, 0.65] | 0.254 [0.246, 0.263] | 40% |
| LSTM | `no_ar` | 3.28 [3.13, 3.44] | +31 [+23, +40] | 33 [25, 41] | 0.67 [0.65, 0.69] | 0.262 [0.254, 0.270] | 40% |
| LSTM | `no_ar` + `kmeans_8` | 3.57 [3.44, 3.69] | +35 [+27, +44] | 39 [33, 45] | 0.58 [0.55, 0.62] | 0.258 [0.250, 0.267] | 20% |
| LSTM | `ar_unbounded` | 21.15 [14.51, 27.78] | +301 [+199, +403] | 301 [199, 403] | 0.60 [0.58, 0.62] | 0.272 [0.264, 0.280] | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 13.08 [7.26, 18.91] | +141 [+66, +217] | 145 [72, 218] | 0.58 [0.53, 0.62] | 0.265 [0.257, 0.272] | 0% |
| Transformer | `ar_bounded` | 3.74 [3.50, 3.98] | +43 [+18, +69] | 52 [33, 71] | 0.68 [0.66, 0.69] | 0.262 [0.254, 0.270] | 40% |
| Transformer | `ar_bounded` + `kmeans_8` | 3.71 [3.39, 4.03] | +36 [+16, +57] | 45 [31, 59] | 0.65 [0.59, 0.71] | 0.255 [0.246, 0.263] | 20% |
| Transformer | `no_ar` | 4.11 [3.80, 4.41] | +54 [+32, +77] | 58 [41, 75] | 0.58 [0.56, 0.60] | 0.266 [0.258, 0.275] | 20% |
| Transformer | `no_ar` + `kmeans_8` | 3.98 [3.61, 4.36] | +54 [+34, +74] | 57 [41, 74] | 0.61 [0.57, 0.65] | 0.256 [0.248, 0.264] | 10% |
| Transformer | `ar_unbounded` | 5.00 [4.03, 5.97] | +101 [+69, +133] | 102 [70, 133] | 0.65 [0.63, 0.67] | 0.262 [0.254, 0.271] | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 3.86 [3.53, 4.18] | +38 [+20, +56] | 47 [36, 57] | 0.63 [0.56, 0.70] | 0.256 [0.247, 0.265] | 20% |

**Rate 0.10, churn 40%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| **Pareto/NBD** | — | 2.47 [2.37, 2.57] | +1 [−3, +6] | 32 [30, 33] | 0.73 [0.72, 0.74] | — | — |
| LSTM | `ar_bounded` | 2.79 [2.66, 2.92] | +28 [+16, +41] | 34 [25, 44] | 0.69 [0.68, 0.71] | 0.198 [0.193, 0.203] | 60% |
| LSTM | `ar_bounded` + `kmeans_8` | 3.15 [2.98, 3.33] | +52 [+44, +59] | 53 [47, 60] | 0.64 [0.62, 0.65] | 0.193 [0.189, 0.198] | 0% |
| LSTM | `no_ar` | 3.11 [2.57, 3.65] | +57 [+23, +91] | 58 [24, 92] | 0.64 [0.56, 0.71] | 0.205 [0.199, 0.210] | 20% |
| LSTM | `no_ar` + `kmeans_8` | 3.42 [3.30, 3.55] | +74 [+67, +81] | 74 [68, 81] | 0.57 [0.52, 0.62] | 0.198 [0.194, 0.203] | 0% |
| LSTM | `ar_unbounded` | 18.03 [11.59, 24.46] | +375 [+270, +479] | 375 [270, 479] | 0.57 [0.55, 0.59] | 0.213 [0.209, 0.218] | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 9.72 [4.50, 14.94] | +167 [+68, +266] | 174 [81, 267] | 0.56 [0.51, 0.62] | 0.203 [0.197, 0.209] | 0% |
| Transformer | `ar_bounded` | 2.87 [2.56, 3.18] | +27 [+6, +47] | 41 [30, 51] | 0.64 [0.53, 0.76] | 0.199 [0.194, 0.204] | 40% |
| Transformer | `ar_bounded` + `kmeans_8` | 3.42 [3.02, 3.81] | +51 [+26, +76] | 58 [39, 76] | 0.65 [0.64, 0.67] | 0.192 [0.187, 0.197] | 10% |
| Transformer | `no_ar` | 3.69 [3.40, 3.99] | +86 [+45, +127] | 94 [62, 125] | 0.58 [0.57, 0.59] | 0.206 [0.201, 0.211] | 10% |
| Transformer | `no_ar` + `kmeans_8` | 3.99 [3.60, 4.39] | +93 [+72, +115] | 95 [76, 114] | 0.63 [0.61, 0.64] | 0.195 [0.191, 0.200] | 0% |
| Transformer | `ar_unbounded` | 3.89 [3.10, 4.69] | +89 [+60, +119] | 91 [62, 119] | 0.64 [0.58, 0.69] | 0.200 [0.195, 0.206] | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 3.74 [3.16, 4.33] | +72 [+41, +104] | 77 [49, 104] | 0.62 [0.58, 0.67] | 0.192 [0.187, 0.198] | 10% |

**Rate 0.10, churn 60%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| **Pareto/NBD** | — | 2.00 [1.93, 2.07] | +3 [−5, +12] | 37 [35, 39] | 0.65 [0.64, 0.67] | — | — |
| LSTM | `ar_bounded` | 2.12 [2.04, 2.20] | +20 [+11, +29] | 33 [29, 37] | 0.63 [0.61, 0.64] | 0.140 [0.135, 0.144] | 70% |
| LSTM | `ar_bounded` + `kmeans_8` | 2.79 [2.66, 2.93] | +92 [+69, +114] | 93 [72, 115] | 0.53 [0.49, 0.57] | 0.139 [0.135, 0.143] | 0% |
| LSTM | `no_ar` | 2.43 [2.22, 2.64] | +61 [+27, +95] | 69 [40, 97] | 0.60 [0.58, 0.61] | 0.149 [0.142, 0.155] | 30% |
| LSTM | `no_ar` + `kmeans_8` | 3.12 [2.93, 3.30] | +132 [+101, +164] | 132 [101, 164] | 0.51 [0.46, 0.55] | 0.146 [0.141, 0.150] | 0% |
| LSTM | `ar_unbounded` | 7.56 [3.71, 11.41] | +299 [+138, +461] | 307 [154, 461] | 0.52 [0.46, 0.58] | 0.154 [0.145, 0.162] | 10% |
| LSTM | `ar_unbounded` + `kmeans_8` | 6.06 [3.10, 9.03] | +133 [+21, +244] | 142 [35, 249] | 0.55 [0.50, 0.60] | 0.146 [0.141, 0.151] | 10% |
| Transformer | `ar_bounded` | 2.74 [2.27, 3.22] | +104 [+45, +162] | 115 [68, 161] | 0.61 [0.59, 0.63] | 0.140 [0.136, 0.144] | 10% |
| Transformer | `ar_bounded` + `kmeans_8` | 3.05 [2.66, 3.45] | +106 [+63, +148] | 108 [68, 149] | 0.57 [0.55, 0.59] | 0.135 [0.132, 0.139] | 0% |
| Transformer | `no_ar` | 2.72 [2.41, 3.03] | +98 [+51, +144] | 103 [61, 145] | 0.52 [0.49, 0.55] | 0.150 [0.146, 0.154] | 10% |
| Transformer | `no_ar` + `kmeans_8` | 3.01 [2.69, 3.33] | +106 [+74, +139] | 109 [78, 139] | 0.56 [0.55, 0.58] | 0.139 [0.135, 0.143] | 0% |
| Transformer | `ar_unbounded` | 3.57 [2.92, 4.23] | +159 [+113, +204] | 159 [114, 204] | 0.57 [0.54, 0.61] | 0.143 [0.139, 0.147] | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 2.83 [2.50, 3.15] | +70 [+37, +102] | 76 [49, 104] | 0.56 [0.52, 0.61] | 0.136 [0.132, 0.140] | 10% |

**Rate 0.10, churn 80%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| **Pareto/NBD** | — | 1.37 [1.23, 1.50] | +7 [−11, +26] | 49 [41, 58] | 0.51 [0.49, 0.53] | — | — |
| LSTM | `ar_bounded` | 1.45 [1.28, 1.62] | +4 [−10, +18] | 45 [39, 51] | 0.47 [0.45, 0.49] | 0.077 [0.072, 0.081] | 90% |
| LSTM | `ar_bounded` + `kmeans_8` | 2.04 [1.87, 2.20] | +131 [+82, +181] | 134 [86, 182] | 0.45 [0.42, 0.48] | 0.076 [0.071, 0.080] | 0% |
| LSTM | `no_ar` | 2.02 [1.66, 2.38] | +224 [+71, +377] | 233 [87, 380] | 0.27 [0.13, 0.42] | 0.091 [0.085, 0.097] | 20% |
| LSTM | `no_ar` + `kmeans_8` | 2.42 [2.24, 2.61] | +309 [+202, +415] | 309 [203, 415] | 0.38 [0.33, 0.43] | 0.086 [0.082, 0.090] | 0% |
| LSTM | `ar_unbounded` | 4.09 [1.42, 6.77] | +285 [+51, +519] | 307 [87, 526] | 0.43 [0.37, 0.48] | 0.085 [0.079, 0.090] | 10% |
| LSTM | `ar_unbounded` + `kmeans_8` | 3.98 [1.30, 6.66] | +146 [−75, +368] | 189 [−15, 393] | 0.45 [0.42, 0.48] | 0.078 [0.072, 0.084] | 10% |
| Transformer | `ar_bounded` | 1.57 [1.44, 1.69] | +82 [+20, +145] | 102 [54, 150] | 0.47 [0.45, 0.49] | 0.078 [0.074, 0.083] | 30% |
| Transformer | `ar_bounded` + `kmeans_8` | 2.31 [2.02, 2.60] | +175 [+118, +233] | 177 [121, 233] | 0.42 [0.37, 0.47] | 0.075 [0.071, 0.079] | 0% |
| Transformer | `no_ar` | 2.20 [2.01, 2.40] | +265 [+178, +352] | 265 [179, 352] | 0.41 [0.39, 0.43] | 0.088 [0.083, 0.092] | 0% |
| Transformer | `no_ar` + `kmeans_8` | 2.39 [2.07, 2.71] | +201 [+116, +287] | 203 [120, 287] | 0.43 [0.40, 0.46] | 0.077 [0.072, 0.081] | 0% |
| Transformer | `ar_unbounded` | 2.70 [2.14, 3.27] | +245 [+182, +307] | 245 [182, 307] | 0.45 [0.42, 0.49] | 0.080 [0.075, 0.085] | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 2.21 [1.74, 2.68] | +142 [+73, +211] | 150 [88, 213] | 0.44 [0.40, 0.47] | 0.075 [0.071, 0.079] | 10% |

**Rate 0.30, churn 20%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| **Pareto/NBD** | — | 6.93 [6.68, 7.17] | −14 [−16, −13] | 30 [30, 31] | 0.86 [0.85, 0.87] | — | — |
| LSTM | `ar_bounded` | 6.49 [6.19, 6.79] | +10 [+7, +13] | 12 [10, 14] | 0.85 [0.84, 0.86] | 0.493 [0.484, 0.501] | 100% |
| LSTM | `ar_bounded` + `kmeans_8` | 7.96 [7.61, 8.31] | +25 [+21, +29] | 26 [22, 29] | 0.79 [0.77, 0.80] | 0.490 [0.481, 0.498] | 90% |
| LSTM | `no_ar` | 6.92 [6.58, 7.26] | +17 [+12, +23] | 19 [15, 23] | 0.85 [0.84, 0.86] | 0.497 [0.490, 0.505] | 100% |
| LSTM | `no_ar` + `kmeans_8` | 8.31 [8.00, 8.61] | +27 [+24, +31] | 28 [25, 31] | 0.75 [0.73, 0.76] | 0.493 [0.485, 0.502] | 70% |
| LSTM | `ar_unbounded` | 35.67 [30.14, 41.20] | +268 [+226, +309] | 268 [226, 309] | 0.73 [0.72, 0.74] | 0.546 [0.537, 0.555] | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 15.60 [9.19, 22.02] | +76 [+38, +113] | 80 [47, 114] | 0.75 [0.72, 0.78] | 0.513 [0.501, 0.525] | 30% |
| Transformer | `ar_bounded` | 8.06 [6.74, 9.37] | +25 [+4, +46] | 37 [23, 51] | 0.81 [0.79, 0.82] | 0.501 [0.492, 0.509] | 60% |
| Transformer | `ar_bounded` + `kmeans_8` | 8.08 [7.58, 8.58] | +25 [+16, +33] | 32 [27, 37] | 0.76 [0.70, 0.82] | 0.494 [0.486, 0.503] | 40% |
| Transformer | `no_ar` | 8.30 [7.14, 9.46] | +20 [−3, +42] | 36 [21, 51] | 0.71 [0.69, 0.74] | 0.517 [0.510, 0.524] | 80% |
| Transformer | `no_ar` + `kmeans_8` | 8.20 [7.59, 8.80] | +21 [+11, +31] | 30 [24, 36] | 0.76 [0.75, 0.77] | 0.505 [0.496, 0.513] | 70% |
| Transformer | `ar_unbounded` | 12.13 [9.21, 15.05] | +69 [+41, +97] | 71 [45, 98] | 0.76 [0.71, 0.82] | 0.511 [0.503, 0.518] | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 8.84 [7.83, 9.86] | +30 [+17, +43] | 37 [31, 44] | 0.80 [0.79, 0.80] | 0.499 [0.490, 0.508] | 20% |

**Rate 0.30, churn 40%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| **Pareto/NBD** | — | 5.99 [5.73, 6.24] | −14 [−16, −12] | 31 [30, 32] | 0.83 [0.82, 0.84] | — | — |
| LSTM | `ar_bounded` | 5.66 [5.41, 5.91] | +2 [−3, +7] | 13 [11, 14] | 0.82 [0.81, 0.83] | 0.355 [0.349, 0.362] | 100% |
| LSTM | `ar_bounded` + `kmeans_8` | 7.67 [7.23, 8.12] | +44 [+36, +51] | 45 [38, 51] | 0.75 [0.74, 0.77] | 0.358 [0.352, 0.365] | 0% |
| LSTM | `no_ar` | 5.74 [5.48, 6.01] | +7 [+2, +13] | 14 [13, 16] | 0.81 [0.80, 0.82] | 0.361 [0.355, 0.367] | 100% |
| LSTM | `no_ar` + `kmeans_8` | 8.24 [7.68, 8.81] | +53 [+47, +59] | 53 [48, 59] | 0.74 [0.73, 0.76] | 0.361 [0.354, 0.367] | 0% |
| LSTM | `ar_unbounded` | 18.04 [9.50, 26.58] | +151 [+41, +261] | 161 [59, 263] | 0.74 [0.71, 0.77] | 0.400 [0.384, 0.415] | 10% |
| LSTM | `ar_unbounded` + `kmeans_8` | 10.25 [5.99, 14.51] | +43 [−3, +89] | 62 [26, 98] | 0.75 [0.72, 0.78] | 0.380 [0.371, 0.389] | 30% |
| Transformer | `ar_bounded` | 6.88 [5.92, 7.84] | +23 [−4, +50] | 42 [28, 56] | 0.70 [0.59, 0.82] | 0.364 [0.356, 0.371] | 40% |
| Transformer | `ar_bounded` + `kmeans_8` | 7.90 [7.49, 8.31] | +45 [+32, +57] | 47 [37, 57] | 0.74 [0.73, 0.76] | 0.357 [0.350, 0.364] | 10% |
| Transformer | `no_ar` | 7.34 [6.73, 7.96] | +28 [+13, +43] | 36 [26, 46] | 0.68 [0.65, 0.70] | 0.386 [0.379, 0.393] | 40% |
| Transformer | `no_ar` + `kmeans_8` | 8.52 [7.79, 9.25] | +53 [+40, +66] | 55 [44, 66] | 0.70 [0.69, 0.72] | 0.373 [0.367, 0.379] | 10% |
| Transformer | `ar_unbounded` | 10.23 [8.28, 12.19] | +81 [+46, +115] | 86 [56, 115] | 0.69 [0.61, 0.76] | 0.376 [0.368, 0.384] | 10% |
| Transformer | `ar_unbounded` + `kmeans_8` | 7.85 [6.74, 8.97] | +21 [0, +42] | 41 [29, 53] | 0.74 [0.69, 0.78] | 0.363 [0.357, 0.369] | 40% |

**Rate 0.30, churn 60%** (9–10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| **Pareto/NBD** | — | 4.72 [4.56, 4.88] | −12 [−17, −7] | 32 [31, 33] | 0.77 [0.76, 0.77] | — | — |
| LSTM | `ar_bounded` | 4.60 [4.41, 4.79] | −7 [−15, +2] | 19 [16, 22] | 0.70 [0.69, 0.72] | 0.232 [0.226, 0.238] | 100% |
| LSTM | `ar_bounded` + `kmeans_8` | 6.73 [6.35, 7.11] | +71 [+62, +80] | 71 [63, 79] | 0.66 [0.64, 0.67] | 0.236 [0.229, 0.242] | 0% |
| LSTM | `no_ar` | 4.73 [4.47, 4.99] | −1 [−10, +7] | 18 [14, 22] | 0.70 [0.69, 0.72] | 0.237 [0.230, 0.243] | 100% |
| LSTM | `no_ar` + `kmeans_8` | 7.35 [6.91, 7.80] | +90 [+79, +101] | 90 [79, 101] | 0.64 [0.62, 0.65] | 0.242 [0.234, 0.250] | 0% |
| LSTM | `ar_unbounded` | 7.04 [6.02, 8.06] | +56 [+32, +81] | 64 [45, 82] | 0.66 [0.64, 0.68] | 0.259 [0.249, 0.269] | 10% |
| LSTM | `ar_unbounded` + `kmeans_8` | 6.49 [5.34, 7.63] | +11 [−26, +48] | 54 [35, 73] | 0.66 [0.64, 0.67] | 0.253 [0.244, 0.263] | 10% |
| Transformer | `ar_bounded` | 5.81 [5.14, 6.48] | +59 [+24, +94] | 67 [39, 95] | 0.67 [0.65, 0.69] | 0.238 [0.231, 0.245] | 30% |
| Transformer | `ar_bounded` + `kmeans_8` | 7.33 [6.13, 8.53] | +83 [+55, +112] | 85 [57, 112] | 0.64 [0.62, 0.66] | 0.234 [0.227, 0.240] | 0% |
| Transformer | `no_ar` | 6.19 [5.64, 6.74] | +40 [+5, +74] | 57 [36, 78] | 0.59 [0.57, 0.60] | 0.265 [0.257, 0.272] | 30% |
| Transformer | `no_ar` + `kmeans_8` | 6.74 [6.24, 7.24] | +65 [+47, +82] | 65 [48, 82] | 0.61 [0.60, 0.62] | 0.249 [0.242, 0.256] | 0% |
| Transformer | `ar_unbounded` | 7.95 [7.03, 8.88] | +96 [+65, +127] | 97 [66, 127] | 0.63 [0.59, 0.67] | 0.251 [0.240, 0.262] | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 7.05 [5.85, 8.24] | +54 [+20, +87] | 65 [40, 91] | 0.64 [0.61, 0.66] | 0.241 [0.234, 0.248] | 10% |

**Rate 0.30, churn 80%** (9–10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| **Pareto/NBD** | — | 3.03 [2.81, 3.25] | −12 [−20, −3] | 35 [33, 37] | 0.70 [0.68, 0.72] | — | — |
| LSTM | `ar_bounded` | 3.01 [2.81, 3.22] | −13 [−26, +1] | 35 [32, 39] | 0.51 [0.48, 0.53] | 0.117 [0.109, 0.125] | 50% |
| LSTM | `ar_bounded` + `kmeans_8` | 4.08 [3.84, 4.31] | +83 [+63, +103] | 84 [65, 103] | 0.49 [0.46, 0.51] | 0.118 [0.110, 0.126] | 0% |
| LSTM | `no_ar` | 3.26 [2.99, 3.54] | −15 [−24, −7] | 32 [27, 37] | 0.51 [0.48, 0.53] | 0.121 [0.115, 0.127] | 70% |
| LSTM | `no_ar` + `kmeans_8` | 5.34 [4.71, 5.97] | +142 [+108, +175] | 142 [108, 175] | 0.48 [0.46, 0.49] | 0.127 [0.117, 0.136] | 0% |
| LSTM | `ar_unbounded` | 4.63 [3.03, 6.23] | +50 [0, +101] | 70 [29, 111] | 0.48 [0.46, 0.50] | 0.131 [0.124, 0.138] | 50% |
| LSTM | `ar_unbounded` + `kmeans_8` | 3.89 [3.44, 4.33] | 0 [−21, +22] | 47 [39, 54] | 0.48 [0.46, 0.51] | 0.126 [0.117, 0.135] | 10% |
| Transformer | `ar_bounded` | 3.18 [2.93, 3.43] | +46 [+28, +64] | 54 [43, 66] | 0.48 [0.46, 0.51] | 0.119 [0.111, 0.126] | 20% |
| Transformer | `ar_bounded` + `kmeans_8` | 4.54 [3.62, 5.46] | +87 [+46, +128] | 91 [52, 129] | 0.47 [0.43, 0.50] | 0.118 [0.110, 0.125] | 10% |
| Transformer | `no_ar` | 4.06 [3.56, 4.55] | +86 [+30, +141] | 96 [49, 144] | 0.44 [0.42, 0.46] | 0.141 [0.132, 0.150] | 20% |
| Transformer | `no_ar` + `kmeans_8` | 4.96 [4.18, 5.75] | +119 [+82, +155] | 120 [84, 155] | 0.45 [0.43, 0.48] | 0.129 [0.120, 0.138] | 0% |
| Transformer | `ar_unbounded` | 4.84 [3.29, 6.40] | +112 [+44, +180] | 118 [55, 181] | 0.43 [0.40, 0.47] | 0.128 [0.118, 0.138] | 11% |
| Transformer | `ar_unbounded` + `kmeans_8` | 4.26 [3.79, 4.73] | +39 [+16, +62] | 56 [46, 66] | 0.47 [0.44, 0.50] | 0.123 [0.113, 0.132] | 0% |

### By purchase rate, churn pooled

The same tables with the four churn levels pooled, 40 panels per row.

**Rate 0.01**, churn 20–80% pooled (40 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| **Pareto/NBD** | — | 0.62 [0.57, 0.66] | +35 [+27, +43] | 90 [75, 105] | 0.23 [0.21, 0.25] | — | — |
| LSTM | `ar_bounded` | 0.77 [0.73, 0.81] | +231 [+138, +324] | 243 [151, 335] | −0.06 [−0.08, −0.03] | 0.058 [0.052, 0.063] | 8% |
| LSTM | `ar_bounded` + `kmeans_8` | 0.77 [0.73, 0.82] | +210 [+150, +270] | 221 [162, 280] | 0.10 [0.07, 0.12] | 0.057 [0.051, 0.063] | 0% |
| LSTM | `no_ar` | 0.84 [0.81, 0.87] | +309 [+201, +417] | 314 [208, 421] | 0.00 [−0.01, 0.02] | 0.058 [0.053, 0.064] | 0% |
| LSTM | `no_ar` + `kmeans_8` | 0.85 [0.82, 0.88] | +302 [+204, +400] | 307 [210, 404] | 0.14 [0.12, 0.16] | 0.057 [0.052, 0.063] | 0% |
| LSTM | `ar_unbounded` | 3.38 [1.67, 5.08] | +505 [+329, +681] | 513 [339, 688] | −0.18 [−0.20, −0.15] | 0.058 [0.052, 0.063] | 2% |
| LSTM | `ar_unbounded` + `kmeans_8` | 7.05 [4.49, 9.62] | +1107 [+592, +1622] | 1118 [604, 1632] | −0.07 [−0.10, −0.03] | 0.056 [0.051, 0.061] | 0% |
| Transformer | `ar_bounded` | 0.65 [0.60, 0.70] | +50 [+30, +71] | 103 [86, 120] | 0.04 [0.01, 0.06] | 0.056 [0.050, 0.061] | 38% |
| Transformer | `ar_bounded` + `kmeans_8` | 0.71 [0.64, 0.78] | +60 [+30, +90] | 119 [95, 142] | 0.15 [0.13, 0.18] | 0.053 [0.047, 0.059] | 38% |
| Transformer | `no_ar` | 0.67 [0.61, 0.72] | +74 [+46, +102] | 120 [96, 144] | 0.05 [0.03, 0.07] | 0.056 [0.050, 0.062] | 35% |
| Transformer | `no_ar` + `kmeans_8` | 0.73 [0.65, 0.80] | +71 [+32, +109] | 130 [99, 160] | 0.14 [0.12, 0.16] | 0.055 [0.049, 0.061] | 40% |
| Transformer | `ar_unbounded` | 0.75 [0.63, 0.87] | +89 [+14, +165] | 150 [76, 224] | 0.00 [−0.03, 0.04] | 0.055 [0.050, 0.061] | 50% |
| Transformer | `ar_unbounded` + `kmeans_8` | 0.70 [0.65, 0.76] | +44 [+7, +81] | 121 [93, 149] | 0.05 [0.02, 0.08] | 0.051 [0.045, 0.057] | 20% |

**Rate 0.05**, churn 20–80% pooled (40 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| **Pareto/NBD** | — | 1.44 [1.31, 1.57] | +11 [+7, +16] | 44 [40, 47] | 0.54 [0.52, 0.57] | — | — |
| LSTM | `ar_bounded` | 1.64 [1.48, 1.80] | +44 [+36, +53] | 58 [53, 63] | 0.49 [0.47, 0.51] | 0.118 [0.104, 0.132] | 25% |
| LSTM | `ar_bounded` + `kmeans_8` | 1.89 [1.79, 2.00] | +121 [+78, +164] | 126 [84, 168] | 0.42 [0.40, 0.44] | 0.116 [0.102, 0.129] | 5% |
| LSTM | `no_ar` | 2.10 [2.02, 2.19] | +199 [+140, +258] | 200 [141, 258] | 0.13 [0.06, 0.20] | 0.125 [0.112, 0.137] | 5% |
| LSTM | `no_ar` + `kmeans_8` | 2.08 [2.02, 2.14] | +200 [+138, +262] | 202 [141, 263] | 0.41 [0.38, 0.44] | 0.118 [0.105, 0.131] | 0% |
| LSTM | `ar_unbounded` | 11.88 [10.14, 13.61] | +582 [+384, +781] | 582 [384, 781] | 0.36 [0.33, 0.39] | 0.124 [0.111, 0.138] | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 10.18 [8.27, 12.08] | +418 [+276, +561] | 424 [283, 565] | 0.38 [0.35, 0.41] | 0.121 [0.107, 0.134] | 8% |
| Transformer | `ar_bounded` | 1.74 [1.55, 1.94] | +64 [+45, +83] | 78 [63, 93] | 0.48 [0.46, 0.50] | 0.118 [0.104, 0.132] | 22% |
| Transformer | `ar_bounded` + `kmeans_8` | 1.94 [1.75, 2.13] | +85 [+62, +107] | 96 [77, 115] | 0.47 [0.44, 0.49] | 0.113 [0.100, 0.127] | 15% |
| Transformer | `no_ar` | 1.99 [1.79, 2.20] | +109 [+87, +130] | 114 [94, 134] | 0.35 [0.32, 0.39] | 0.122 [0.109, 0.135] | 10% |
| Transformer | `no_ar` + `kmeans_8` | 2.00 [1.82, 2.18] | +115 [+84, +145] | 122 [94, 151] | 0.45 [0.43, 0.48] | 0.114 [0.101, 0.128] | 8% |
| Transformer | `ar_unbounded` | 1.95 [1.72, 2.19] | +95 [+77, +114] | 105 [89, 121] | 0.43 [0.37, 0.49] | 0.118 [0.105, 0.132] | 8% |
| Transformer | `ar_unbounded` + `kmeans_8` | 1.97 [1.74, 2.19] | +72 [+48, +96] | 87 [66, 107] | 0.44 [0.40, 0.47] | 0.113 [0.100, 0.127] | 18% |

**Rate 0.10**, churn 20–80% pooled (40 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| **Pareto/NBD** | — | 2.19 [2.00, 2.38] | +4 [−1, +8] | 37 [34, 40] | 0.65 [0.62, 0.68] | — | — |
| LSTM | `ar_bounded` | 2.42 [2.18, 2.65] | +20 [+14, +27] | 36 [32, 40] | 0.62 [0.59, 0.65] | 0.169 [0.147, 0.191] | 65% |
| LSTM | `ar_bounded` + `kmeans_8` | 2.86 [2.68, 3.04] | +76 [+58, +93] | 78 [61, 95] | 0.56 [0.53, 0.58] | 0.166 [0.144, 0.187] | 10% |
| LSTM | `no_ar` | 2.71 [2.49, 2.93] | +93 [+51, +136] | 98 [57, 140] | 0.55 [0.48, 0.61] | 0.177 [0.156, 0.197] | 28% |
| LSTM | `no_ar` + `kmeans_8` | 3.13 [2.98, 3.29] | +138 [+96, +179] | 139 [97, 180] | 0.51 [0.48, 0.54] | 0.172 [0.151, 0.193] | 5% |
| LSTM | `ar_unbounded` | 12.71 [9.51, 15.90] | +315 [+245, +385] | 322 [256, 389] | 0.53 [0.50, 0.56] | 0.181 [0.158, 0.204] | 5% |
| LSTM | `ar_unbounded` + `kmeans_8` | 8.21 [6.01, 10.41] | +147 [+87, +207] | 162 [106, 219] | 0.54 [0.51, 0.56] | 0.173 [0.151, 0.196] | 5% |
| Transformer | `ar_bounded` | 2.73 [2.44, 3.01] | +64 [+42, +86] | 77 [59, 96] | 0.60 [0.56, 0.63] | 0.170 [0.148, 0.192] | 30% |
| Transformer | `ar_bounded` + `kmeans_8` | 3.12 [2.89, 3.35] | +92 [+68, +117] | 97 [74, 120] | 0.57 [0.54, 0.61] | 0.164 [0.142, 0.186] | 8% |
| Transformer | `no_ar` | 3.18 [2.91, 3.45] | +126 [+90, +161] | 130 [96, 164] | 0.52 [0.50, 0.55] | 0.178 [0.156, 0.199] | 10% |
| Transformer | `no_ar` + `kmeans_8` | 3.34 [3.08, 3.61] | +114 [+87, +141] | 116 [90, 143] | 0.56 [0.53, 0.58] | 0.167 [0.145, 0.188] | 2% |
| Transformer | `ar_unbounded` | 3.79 [3.37, 4.21] | +148 [+121, +176] | 149 [122, 176] | 0.58 [0.55, 0.61] | 0.171 [0.149, 0.193] | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 3.16 [2.87, 3.45] | +81 [+59, +103] | 88 [67, 108] | 0.56 [0.53, 0.60] | 0.165 [0.143, 0.187] | 12% |

**Rate 0.30**, churn 20–80% pooled (38–40 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| **Pareto/NBD** | — | 5.17 [4.68, 5.65] | −13 [−15, −11] | 32 [31, 33] | 0.79 [0.77, 0.81] | — | — |
| LSTM | `ar_bounded` | 4.94 [4.51, 5.38] | −2 [−7, +3] | 20 [16, 23] | 0.72 [0.68, 0.76] | 0.299 [0.254, 0.345] | 88% |
| LSTM | `ar_bounded` + `kmeans_8` | 6.61 [6.09, 7.13] | +56 [+47, +65] | 56 [48, 65] | 0.67 [0.63, 0.71] | 0.300 [0.255, 0.345] | 22% |
| LSTM | `no_ar` | 5.16 [4.71, 5.62] | +2 [−3, +7] | 21 [18, 24] | 0.72 [0.67, 0.76] | 0.304 [0.258, 0.349] | 92% |
| LSTM | `no_ar` + `kmeans_8` | 7.31 [6.87, 7.75] | +78 [+62, +94] | 78 [62, 94] | 0.65 [0.61, 0.69] | 0.306 [0.262, 0.350] | 18% |
| LSTM | `ar_unbounded` | 16.34 [11.80, 20.89] | +131 [+91, +171] | 141 [104, 178] | 0.65 [0.62, 0.69] | 0.334 [0.284, 0.384] | 18% |
| LSTM | `ar_unbounded` + `kmeans_8` | 9.06 [6.86, 11.26] | +33 [+14, +51] | 61 [49, 73] | 0.66 [0.62, 0.70] | 0.318 [0.271, 0.365] | 20% |
| Transformer | `ar_bounded` | 5.98 [5.28, 6.68] | +38 [+26, +51] | 50 [42, 59] | 0.66 [0.62, 0.71] | 0.305 [0.259, 0.351] | 38% |
| Transformer | `ar_bounded` + `kmeans_8` | 6.96 [6.38, 7.54] | +60 [+46, +74] | 64 [51, 77] | 0.65 [0.61, 0.69] | 0.301 [0.255, 0.346] | 15% |
| Transformer | `no_ar` | 6.47 [5.87, 7.08] | +43 [+26, +61] | 56 [42, 71] | 0.60 [0.57, 0.64] | 0.327 [0.282, 0.373] | 42% |
| Transformer | `no_ar` + `kmeans_8` | 7.11 [6.57, 7.64] | +64 [+50, +79] | 67 [54, 81] | 0.63 [0.59, 0.67] | 0.314 [0.269, 0.359] | 20% |
| Transformer | `ar_unbounded` | 8.92 [7.66, 10.17] | +89 [+70, +108] | 92 [75, 110] | 0.63 [0.59, 0.68] | 0.323 [0.276, 0.371] | 5% |
| Transformer | `ar_unbounded` + `kmeans_8` | 7.00 [6.30, 7.70] | +36 [+25, +47] | 50 [42, 57] | 0.66 [0.62, 0.70] | 0.306 [0.261, 0.352] | 18% |

### The three best trees per cell, by MAPE

Ranked by mean MAPE over the cell's 10 panels (lowest first); MAPE carries its 95% interval,
the other metrics are means. "p vs rank 1" is a paired Wilcoxon test of that tree against
the cell's leader on the same panels.

| Rate | Churn | Rank | Model | Arm | **MAPE** | Spearman | RMSE | Bias % | Val. CE | p vs rank 1 |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | ---: |
| 0.01 | 20% | 1 | **Pareto/NBD** | — | 48 [45, 51] | 0.22 | 0.79 | +12 | — | — |
|  |  | 2 | Transformer | `ar_unbounded` | 49 [42, 57] | −0.05 | 0.83 | +7 | 0.077 | 0.695 |
|  |  | 3 | Transformer | `no_ar` + `kmeans_8` | 55 [47, 64] | 0.12 | 0.86 | −3 | 0.077 | 0.160 |
| 0.01 | 40% | 1 | **Pareto/NBD** | — | 66 [62, 70] | 0.25 | 0.68 | +34 | — | — |
|  |  | 2 | Transformer | `ar_bounded` | 75 [62, 89] | −0.02 | 0.71 | +27 | 0.065 | 0.193 |
|  |  | 3 | Transformer | `ar_unbounded` | 78 [52, 103] | 0.00 | 0.72 | +18 | 0.065 | 0.770 |
| 0.01 | 60% | 1 | **Pareto/NBD** | — | 88 [83, 94] | 0.26 | 0.57 | +41 | — | — |
|  |  | 2 | Transformer | `ar_bounded` | 98 [78, 119] | 0.06 | 0.60 | +50 | 0.049 | 0.492 |
|  |  | 3 | Transformer | `ar_bounded` + `kmeans_8` | 103 [74, 132] | 0.18 | 0.63 | +43 | 0.047 | 0.625 |
| 0.01 | 80% | 1 | **Pareto/NBD** | — | 159 [132, 186] | 0.20 | 0.42 | +54 | — | — |
|  |  | 2 | Transformer | `ar_bounded` | 176 [145, 206] | 0.08 | 0.44 | +89 | 0.031 | 0.492 |
|  |  | 3 | Transformer | `no_ar` + `kmeans_8` | 195 [114, 276] | 0.15 | 0.48 | +104 | 0.030 | 0.922 |
| 0.05 | 20% | 1 | **Pareto/NBD** | — | 35 [33, 37] | 0.58 | 1.92 | +8 | — | — |
|  |  | 2 | LSTM | `ar_bounded` + `kmeans_8` | 39 [36, 42] | 0.45 | 2.20 | +29 | 0.169 | 0.014 |
|  |  | 3 | LSTM | `no_ar` | 45 [35, 54] | 0.29 | 2.33 | +42 | 0.175 | 0.049 |
| 0.05 | 40% | 1 | **Pareto/NBD** | — | 39 [36, 41] | 0.60 | 1.63 | +15 | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 56 [46, 65] | 0.54 | 1.88 | +52 | 0.138 | 0.004 |
|  |  | 3 | Transformer | `ar_bounded` + `kmeans_8` | 56 [41, 71] | 0.51 | 2.00 | +48 | 0.133 | 0.049 |
| 0.05 | 60% | 1 | **Pareto/NBD** | — | 42 [39, 45] | 0.56 | 1.32 | +9 | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 67 [55, 79] | 0.52 | 1.43 | +60 | 0.100 | 0.006 |
|  |  | 3 | Transformer | `ar_bounded` | 77 [49, 105] | 0.52 | 1.47 | +66 | 0.100 | 0.014 |
| 0.05 | 80% | 1 | **Pareto/NBD** | — | 58 [52, 65] | 0.44 | 0.89 | +13 | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 60 [50, 70] | 0.40 | 0.97 | +19 | 0.061 | 0.922 |
|  |  | 3 | Transformer | `ar_bounded` + `kmeans_8` | 94 [55, 134] | 0.37 | 1.11 | +62 | 0.058 | 0.105 |
| 0.10 | 20% | 1 | **Pareto/NBD** | — | 31 [30, 32] | 0.72 | 2.91 | +3 | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 32 [22, 43] | 0.69 | 3.32 | +29 | 0.261 | 0.770 |
|  |  | 3 | LSTM | `ar_bounded` + `kmeans_8` | 33 [28, 38] | 0.60 | 3.46 | +28 | 0.254 | 0.492 |
| 0.10 | 40% | 1 | **Pareto/NBD** | — | 32 [30, 33] | 0.73 | 2.47 | +1 | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 34 [25, 44] | 0.69 | 2.79 | +28 | 0.198 | 0.770 |
|  |  | 3 | Transformer | `ar_bounded` | 41 [30, 51] | 0.64 | 2.87 | +27 | 0.199 | 0.160 |
| 0.10 | 60% | 1 | LSTM | `ar_bounded` | 33 [29, 37] | 0.63 | 2.12 | +20 | 0.140 | — |
|  |  | 2 | **Pareto/NBD** | — | 37 [35, 39] | 0.65 | 2.00 | +3 | — | 0.049 |
|  |  | 3 | LSTM | `no_ar` | 69 [40, 97] | 0.60 | 2.43 | +61 | 0.149 | 0.014 |
| 0.10 | 80% | 1 | LSTM | `ar_bounded` | 45 [39, 51] | 0.47 | 1.45 | +4 | 0.077 | — |
|  |  | 2 | **Pareto/NBD** | — | 49 [41, 58] | 0.51 | 1.37 | +7 | — | 0.037 |
|  |  | 3 | Transformer | `ar_bounded` | 102 [54, 150] | 0.47 | 1.57 | +82 | 0.078 | 0.006 |
| 0.30 | 20% | 1 | LSTM | `ar_bounded` | 12 [10, 14] | 0.85 | 6.49 | +10 | 0.493 | — |
|  |  | 2 | LSTM | `no_ar` | 19 [15, 23] | 0.85 | 6.92 | +17 | 0.497 | 0.014 |
|  |  | 3 | LSTM | `ar_bounded` + `kmeans_8` | 26 [22, 29] | 0.79 | 7.96 | +25 | 0.490 | 0.002 |
| 0.30 | 40% | 1 | LSTM | `ar_bounded` | 13 [11, 14] | 0.82 | 5.66 | +2 | 0.355 | — |
|  |  | 2 | LSTM | `no_ar` | 14 [13, 16] | 0.81 | 5.74 | +7 | 0.361 | 0.064 |
|  |  | 3 | **Pareto/NBD** | — | 31 [30, 32] | 0.83 | 5.99 | −14 | — | 0.002 |
| 0.30 | 60% | 1 | LSTM | `no_ar` | 18 [14, 22] | 0.70 | 4.73 | −1 | 0.237 | — |
|  |  | 2 | LSTM | `ar_bounded` | 19 [16, 22] | 0.70 | 4.60 | −7 | 0.232 | 0.625 |
|  |  | 3 | **Pareto/NBD** | — | 32 [31, 33] | 0.77 | 4.72 | −12 | — | 0.002 |
| 0.30 | 80% | 1 | LSTM | `no_ar` | 32 [27, 37] | 0.51 | 3.26 | −15 | 0.121 | — |
|  |  | 2 | **Pareto/NBD** | — | 35 [33, 37] | 0.70 | 3.03 | −12 | — | 0.131 |
|  |  | 3 | LSTM | `ar_bounded` | 35 [32, 39] | 0.51 | 3.01 | −13 | 0.117 | 0.020 |

### The three best trees per cell, by Spearman

Ranked by mean per-customer Spearman (highest first); layout as above.

| Rate | Churn | Rank | Model | Arm | MAPE | **Spearman** | RMSE | Bias % | Val. CE | p vs rank 1 |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | ---: |
| 0.01 | 20% | 1 | **Pareto/NBD** | — | 48 | 0.22 [0.19, 0.25] | 0.79 | +12 | — | — |
|  |  | 2 | Transformer | `ar_bounded` + `kmeans_8` | 73 | 0.16 [0.13, 0.20] | 0.90 | +23 | 0.074 | 0.010 |
|  |  | 3 | LSTM | `no_ar` + `kmeans_8` | 86 | 0.14 [0.11, 0.17] | 0.91 | +77 | 0.078 | 0.002 |
| 0.01 | 40% | 1 | **Pareto/NBD** | — | 66 | 0.25 [0.23, 0.28] | 0.68 | +34 | — | — |
|  |  | 2 | Transformer | `ar_bounded` + `kmeans_8` | 102 | 0.16 [0.12, 0.19] | 0.80 | +64 | 0.062 | 0.002 |
|  |  | 3 | LSTM | `no_ar` + `kmeans_8` | 140 | 0.14 [0.10, 0.17] | 0.81 | +136 | 0.067 | 0.002 |
| 0.01 | 60% | 1 | **Pareto/NBD** | — | 88 | 0.26 [0.22, 0.31] | 0.57 | +41 | — | — |
|  |  | 2 | Transformer | `ar_bounded` + `kmeans_8` | 103 | 0.18 [0.11, 0.24] | 0.63 | +43 | 0.047 | 0.004 |
|  |  | 3 | Transformer | `no_ar` + `kmeans_8` | 120 | 0.16 [0.11, 0.22] | 0.64 | +65 | 0.047 | 0.002 |
| 0.01 | 80% | 1 | **Pareto/NBD** | — | 159 | 0.20 [0.15, 0.25] | 0.42 | +54 | — | — |
|  |  | 2 | Transformer | `no_ar` + `kmeans_8` | 195 | 0.15 [0.11, 0.19] | 0.48 | +104 | 0.030 | 0.020 |
|  |  | 3 | LSTM | `no_ar` + `kmeans_8` | 729 | 0.11 [0.06, 0.17] | 0.85 | +725 | 0.033 | 0.006 |
| 0.05 | 20% | 1 | **Pareto/NBD** | — | 35 | 0.58 [0.55, 0.60] | 1.92 | +8 | — | — |
|  |  | 2 | Transformer | `ar_unbounded` | 91 | 0.52 [0.50, 0.54] | 2.69 | +90 | 0.173 | 0.002 |
|  |  | 3 | Transformer | `ar_bounded` + `kmeans_8` | 79 | 0.51 [0.47, 0.54] | 2.65 | +77 | 0.168 | 0.002 |
| 0.05 | 40% | 1 | **Pareto/NBD** | — | 39 | 0.60 [0.58, 0.62] | 1.63 | +15 | — | — |
|  |  | 2 | Transformer | `ar_bounded` | 65 | 0.55 [0.52, 0.58] | 1.91 | +53 | 0.138 | 0.002 |
|  |  | 3 | LSTM | `ar_bounded` | 56 | 0.54 [0.51, 0.57] | 1.88 | +52 | 0.138 | 0.002 |
| 0.05 | 60% | 1 | **Pareto/NBD** | — | 42 | 0.56 [0.55, 0.58] | 1.32 | +9 | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 67 | 0.52 [0.50, 0.54] | 1.43 | +60 | 0.100 | 0.002 |
|  |  | 3 | Transformer | `ar_bounded` | 77 | 0.52 [0.50, 0.53] | 1.47 | +66 | 0.100 | 0.002 |
| 0.05 | 80% | 1 | **Pareto/NBD** | — | 58 | 0.44 [0.41, 0.47] | 0.89 | +13 | — | — |
|  |  | 2 | Transformer | `ar_bounded` | 102 | 0.40 [0.37, 0.44] | 1.04 | +76 | 0.061 | 0.004 |
|  |  | 3 | LSTM | `ar_bounded` | 60 | 0.40 [0.36, 0.44] | 0.97 | +19 | 0.061 | 0.004 |
| 0.10 | 20% | 1 | **Pareto/NBD** | — | 31 | 0.72 [0.71, 0.74] | 2.91 | +3 | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 32 | 0.69 [0.68, 0.71] | 3.32 | +29 | 0.261 | 0.002 |
|  |  | 3 | Transformer | `ar_bounded` | 52 | 0.68 [0.66, 0.69] | 3.74 | +43 | 0.262 | 0.002 |
| 0.10 | 40% | 1 | **Pareto/NBD** | — | 32 | 0.73 [0.72, 0.74] | 2.47 | +1 | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 34 | 0.69 [0.68, 0.71] | 2.79 | +28 | 0.198 | 0.002 |
|  |  | 3 | Transformer | `ar_bounded` + `kmeans_8` | 58 | 0.65 [0.64, 0.67] | 3.42 | +51 | 0.192 | 0.002 |
| 0.10 | 60% | 1 | **Pareto/NBD** | — | 37 | 0.65 [0.64, 0.67] | 2.00 | +3 | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 33 | 0.63 [0.61, 0.64] | 2.12 | +20 | 0.140 | 0.002 |
|  |  | 3 | Transformer | `ar_bounded` | 115 | 0.61 [0.59, 0.63] | 2.74 | +104 | 0.140 | 0.002 |
| 0.10 | 80% | 1 | **Pareto/NBD** | — | 49 | 0.51 [0.49, 0.53] | 1.37 | +7 | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 45 | 0.47 [0.45, 0.49] | 1.45 | +4 | 0.077 | 0.002 |
|  |  | 3 | Transformer | `ar_bounded` | 102 | 0.47 [0.45, 0.49] | 1.57 | +82 | 0.078 | 0.002 |
| 0.30 | 20% | 1 | **Pareto/NBD** | — | 30 | 0.86 [0.85, 0.87] | 6.93 | −14 | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 12 | 0.85 [0.84, 0.86] | 6.49 | +10 | 0.493 | 0.002 |
|  |  | 3 | LSTM | `no_ar` | 19 | 0.85 [0.84, 0.86] | 6.92 | +17 | 0.497 | 0.002 |
| 0.30 | 40% | 1 | **Pareto/NBD** | — | 31 | 0.83 [0.82, 0.84] | 5.99 | −14 | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 13 | 0.82 [0.81, 0.83] | 5.66 | +2 | 0.355 | 0.002 |
|  |  | 3 | LSTM | `no_ar` | 14 | 0.81 [0.80, 0.82] | 5.74 | +7 | 0.361 | 0.002 |
| 0.30 | 60% | 1 | **Pareto/NBD** | — | 32 | 0.77 [0.76, 0.77] | 4.72 | −12 | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 19 | 0.70 [0.69, 0.72] | 4.60 | −7 | 0.232 | 0.002 |
|  |  | 3 | LSTM | `no_ar` | 18 | 0.70 [0.69, 0.72] | 4.73 | −1 | 0.237 | 0.002 |
| 0.30 | 80% | 1 | **Pareto/NBD** | — | 35 | 0.70 [0.68, 0.72] | 3.03 | −12 | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 35 | 0.51 [0.48, 0.53] | 3.01 | −13 | 0.117 | 0.002 |
|  |  | 3 | LSTM | `no_ar` | 32 | 0.51 [0.48, 0.53] | 3.26 | −15 | 0.121 | 0.002 |

### Reading

- **MAPE leader by cell.** Pareto/NBD leads all 8 cells at rates 0.01–0.05 and the two
  low-churn cells at rate 0.10. LSTM `ar_bounded` leads at rate 0.10 with churn 60–80% and at
  rate 0.30 with churn 20–40%; LSTM `no_ar` leads at rate 0.30 with churn 60–80%.
- **Most MAPE leads are not separated.** The leader beats the runner-up significantly
  (p < 0.05, 10 panels) in only 6 of 16 cells: Pareto/NBD at rate 0.05 with churn 20–60%, the
  LSTM over Pareto/NBD at rate 0.10 with churn 60–80%, and LSTM `ar_bounded` over `no_ar` at
  rate 0.30, churn 20%. At rate 0.01 the Transformer arms are level with Pareto/NBD.
- **Spearman leader by cell.** Pareto/NBD leads all 16 cells and beats the runner-up
  significantly in every one (p ≤ 0.02). From rate 0.05 up the runner-up is almost always an
  `ar_bounded` arm. On sparse panels it is a `kmeans_8` arm: the cluster label raises the
  neural Spearman (LSTM `no_ar` 0.00 → 0.14 at rate 0.01, churn pooled) while leaving bias
  and MAPE no better.
- **Good MAPE does not mean good ranking.** At rate 0.30 and churn 80% the LSTM leads on MAPE
  (32) but ranks customers at 0.51 against Pareto/NBD's 0.70. Over rates, the median paired
  Spearman gap between Pareto/NBD and the best neural arm (`ar_bounded`) is 0.29 (LSTM) and
  0.20 (Transformer) at rate 0.01 and 0.03–0.09 at rates 0.05–0.30 (Wilcoxon, 40 panels per
  rate, all p < 10⁻¹⁰).
- **Validation CE.** The CE gaps between arms (third decimal) are small beside the spread
  across panels, so the intervals overlap throughout. Paired on the same panel, a lower CE
  across the 12 neural arms goes with a better holdout: mean within-panel rank correlation of
  CE with MAPE +0.40, with RMSE +0.36 and with Spearman −0.41 (Wilcoxon over 160 panels,
  all p < 10⁻²²). The ordering is loose at the top: the `kmeans_8` arms often have a lower CE
  and a worse MAPE than their `no_cluster` counterparts.

**Shape and death detection** (same setting; from `studies.synthetic_grid`,
`docs/insights-arm-sweep.md` §6–7). Shape correlation of predicted vs actual weekly totals,
rates pooled: Pareto/NBD −0.07 to +0.09, LSTM `ar_bounded` 0.29–0.69, Transformer
`ar_bounded` 0.28–0.51, falling as churn rises. At rate 0.30 only: Pareto/NBD's alive ratio
R_A is 0.58–0.74 and dead leakage L_D 0.11–0.29; LSTM `ar_bounded` R_A 0.60–0.93, L_D
0.17–0.27; Transformer `ar_bounded` L_D 0.23–0.66.

## Impact of cohort size

Tripling the cohort from 1,000 to 3,000 customers halves the LSTM's bias and MAPE on sparse
and mid-rate panels. The gain is largest with bounded flags. Pareto/NBD improves on MAPE only.

**Setting.** Seasonal panels (4 peaks, amplitude 1.5), 156 weeks, 1,000 vs 3,000 generated
customers. The 3,000 grid is identical otherwise: same rates, churn levels, seed and training
budget. The Transformer was not run at 3,000. Each cell is RMSE on customer totals / bias % /
MAPE, mean over 40 panels.

**By purchase rate** (the 4 churn levels pooled):

| Model | Customers | Rate 0.01 | 0.05 | 0.10 | 0.30 |
| --- | ---: | --- | --- | --- | --- |
| Pareto/NBD | 1,000 | 0.62 / +35 / 90 | 1.44 / +11 / 44 | 2.19 / +4 / 37 | 5.17 / −13 / 32 |
| Pareto/NBD | 3,000 | 0.61 / +19 / 56 | 1.44 / +7 / 34 | 2.17 / 0 / 32 | 5.12 / −13 / 30 |
| LSTM `no_ar` | 1,000 | 0.84 / +309 / 314 | 2.10 / +199 / 200 | 2.71 / +93 / 98 | 5.17 / +2 / 21 |
| LSTM `no_ar` | 3,000 | 0.78 / +219 / 219 | 1.78 / +117 / 119 | 2.35 / +5 / 24 | 4.83 / −5 / 14 |
| LSTM `ar_bounded` | 1,000 | 0.77 / +231 / 243 | 1.64 / +44 / 58 | 2.42 / +20 / 36 | 4.94 / −2 / 20 |
| LSTM `ar_bounded` | 3,000 | 0.64 / +65 / 83 | 1.53 / +18 / 31 | 2.25 / +1 / 18 | 4.77 / −11 / 16 |

**By churn** (the 4 rates pooled; RMSE here mixes rates, so compare it within a column only):

| Model | Customers | Churn 20% | 40% | 60% | 80% |
| --- | ---: | --- | --- | --- | --- |
| Pareto/NBD | 1,000 | 3.14 / +2 / 36 | 2.69 / +9 / 42 | 2.15 / +10 / 50 | 1.43 / +16 / 75 |
| Pareto/NBD | 3,000 | 3.12 / +1 / 32 | 2.66 / +5 / 35 | 2.11 / +5 / 39 | 1.44 / +1 / 46 |
| LSTM `no_ar` | 1,000 | 3.35 / +37 / 42 | 2.98 / +74 / 77 | 2.49 / +127 / 134 | 2.01 / +365 / 380 |
| LSTM `no_ar` | 3,000 | 3.05 / +33 / 34 | 2.74 / +57 / 59 | 2.26 / +89 / 99 | 1.69 / +156 / 184 |
| LSTM `ar_bounded` | 1,000 | 3.24 / +32 / 39 | 2.78 / +49 / 56 | 2.22 / +66 / 79 | 1.54 / +147 / 183 |
| LSTM `ar_bounded` | 3,000 | 2.96 / +15 / 21 | 2.60 / +15 / 26 | 2.12 / +20 / 39 | 1.51 / +24 / 63 |

**Tests** (Mann–Whitney, 40 vs 40 panels per rate, since the panels differ):

- **LSTM `ar_bounded`:** |bias| and MAPE improve at rates 0.01, 0.05 and 0.10 (all
  p < 10⁻⁵). At rate 0.30 there is no change in |bias| (p = 1) and a small MAPE gain
  (p = 0.04). RMSE improves only at rate 0.01 (p = 2×10⁻⁴).
- **LSTM `no_ar`:** improves at rates 0.05 and 0.10 on all three metrics (p ≤ 0.02). No
  significant change at rate 0.01 on any metric: without the flags, extra customers do not
  help the sparsest panels.
- **Pareto/NBD:** MAPE improves at every rate (p ≤ 0.001) and |bias| at rates 0.01–0.10.
  RMSE does not move (p ≥ 0.8).

**Reading.** Cohort size matters most where each customer buys least. The bounded flags need
data to learn what silence means: at 1,000 customers they cut the LSTM's bias at churn 80%
from +365 to +147; at 3,000, from +156 to +24. The churn trend in the neural bias is partly a
cohort-size effect: at 1,000 generated customers, churn 80% panels keep only about 510 active
customers against about 800 at churn 20%.

## Impact of seasonality

Seasonality is the reason the LSTM beats Pareto/NBD on dense panels. Ignoring it costs
Pareto/NBD 14–17 MAPE points at rates 0.05–0.30. Given the true seasonal pattern,
Pareto/NBD's MAPE at rate 0.30 drops to 17, below the LSTM's 20.

**What can be measured.** Every grid on disk has the same seasonality: peaks at weeks 12, 25,
30 and 47, amplitude 1.5, width 3. Peak weeks carry 2.2× the off-season volume. The older
grids reuse seed 42 and are the same panels. No run compares seasonal with non-seasonal
panels, so the neural models' sensitivity to seasonality is not measured.

**Counterfactual for Pareto/NBD.** Pareto/NBD has no seasonal term, so its weekly forecast is
a smooth decay. Its 160 stored forecasts were rescored after multiplying each holdout week by
the true seasonal multiplier, rescaled to mean 1 so the yearly total barely changes. Each
cell is RMSE / bias % / MAPE, mean over 40 panels, 1,000 customers:

| Pareto/NBD forecast | Rate 0.01 | 0.05 | 0.10 | 0.30 |
| --- | --- | --- | --- | --- |
| As fitted (no seasonality) | 0.62 / +35 / 90 | 1.44 / +11 / 44 | 2.19 / +4 / 37 | 5.17 / −13 / 32 |
| With true seasonality | 0.62 / +35 / 83 | 1.44 / +11 / 30 | 2.19 / +3 / 21 | 5.18 / −14 / 17 |
| LSTM `ar_bounded`, for reference | 0.77 / +231 / 243 | 1.64 / +44 / 58 | 2.42 / +20 / 36 | 4.94 / −2 / 20 |

The MAPE gain is significant at every rate (Wilcoxon, paired, p ≤ 4×10⁻⁸). By churn, it
shrinks from 16 points at churn 20% (36 → 20) to 9 at churn 80% (75 → 66).

**Reading.**

- Seasonality moves MAPE only. RMSE on customer totals and bias measure yearly volume, which
  seasonality does not change.
- Given the season, Pareto/NBD beats the LSTM on MAPE at every rate on these panels. So the
  LSTM's dense-panel MAPE wins (claim 1) come from modelling the season, not from modelling
  customers better.
- The share of MAPE that is shape error (MAPE minus |bias|) tells the same story: 19–55
  points for Pareto/NBD, 9–12 for the LSTM with flags.
- **Open:** a grid without seasonality (amplitude 0), run for Pareto/NBD and the neural
  models, would measure the neural side directly.

## Impact of AR features

Bounded flags help both models on all three metrics, most on mid-rate panels (0.05–0.10) and
not at all on |bias| at rate 0.30. Unbounded counters make the LSTM several times worse at
every rate, and hurt the Transformer only on dense panels.

**Setting.** Seasonal panels, 1,000 customers, no cluster label. `no_ar` = count and calendar
only; `ar_bounded` = 0/1 flags for activity in the last 2 / 4 / 8 / 16 / 32 weeks plus
has-bought-before; `ar_unbounded` = recency, frequency and age as counters. Each cell is RMSE
on customer totals / bias % / MAPE, mean over 40 panels.

**By purchase rate** (the 4 churn levels pooled):

| Model | AR features | Rate 0.01 | 0.05 | 0.10 | 0.30 |
| --- | --- | --- | --- | --- | --- |
| Pareto/NBD | — | 0.62 / +35 / 90 | 1.44 / +11 / 44 | 2.19 / +4 / 37 | 5.17 / −13 / 32 |
| LSTM | none | 0.84 / +309 / 314 | 2.10 / +199 / 200 | 2.71 / +93 / 98 | 5.17 / +2 / 21 |
| LSTM | bounded | 0.77 / +231 / 243 | 1.64 / +44 / 58 | 2.42 / +20 / 36 | 4.94 / −2 / 20 |
| LSTM | unbounded | 3.38 / +505 / 513 | 11.88 / +582 / 582 | 12.71 / +315 / 322 | 16.35 / +131 / 141 |
| Transformer | none | 0.67 / +74 / 120 | 1.99 / +109 / 114 | 3.18 / +126 / 130 | 6.47 / +43 / 56 |
| Transformer | bounded | 0.65 / +50 / 103 | 1.74 / +64 / 78 | 2.73 / +64 / 77 | 5.98 / +38 / 50 |
| Transformer | unbounded | 0.75 / +89 / 150 | 1.95 / +95 / 105 | 3.79 / +148 / 149 | 8.92 / +89 / 92 |

**By churn** (the 4 rates pooled; compare RMSE within a column only):

| Model | AR features | Churn 20% | 40% | 60% | 80% |
| --- | --- | --- | --- | --- | --- |
| Pareto/NBD | — | 3.14 / +2 / 36 | 2.69 / +9 / 42 | 2.15 / +10 / 50 | 1.43 / +16 / 75 |
| LSTM | none | 3.35 / +37 / 42 | 2.98 / +74 / 77 | 2.49 / +127 / 134 | 2.01 / +365 / 380 |
| LSTM | bounded | 3.24 / +32 / 39 | 2.78 / +49 / 56 | 2.22 / +66 / 79 | 1.54 / +147 / 183 |
| LSTM | unbounded | 19.44 / +328 / 330 | 12.08 / +255 / 259 | 7.19 / +333 / 337 | 5.60 / +618 / 632 |
| Transformer | none | 3.99 / +41 / 57 | 3.52 / +71 / 84 | 2.85 / +94 / 110 | 1.96 / +145 / 170 |
| Transformer | bounded | 3.80 / +41 / 55 | 3.09 / +32 / 56 | 2.65 / +70 / 89 | 1.56 / +73 / 109 |
| Transformer | unbounded | 5.16 / +67 / 78 | 4.29 / +72 / 89 | 3.39 / +111 / 128 | 2.27 / +174 / 204 |

**Tests: none → bounded** (Wilcoxon, paired on the same 40 panels per rate; median change in
|bias| / MAPE / RMSE):

| Model | Rate 0.01 | 0.05 | 0.10 | 0.30 |
| --- | --- | --- | --- | --- |
| LSTM | −36 / −24 / −0.04, all p ≤ 0.002 | −60 / −56 / −0.42, all p < 10⁻⁷ | −21 / −16 / −0.12, all p ≤ 10⁻⁴ | −2 / 0 / −0.18; only RMSE significant (p = 10⁻⁵) |
| Transformer | no significant change | −33 / −22 / −0.13, all p ≤ 0.01 | −53 / −45 / −0.44, all p ≤ 7×10⁻⁴ | 0 / 0 / −0.67; only RMSE significant (p = 0.009) |

**Tests: none → unbounded.** LSTM worse on all three metrics at every rate (all p ≤ 0.002).
Transformer worse only at rate 0.30 (|bias| +44, MAPE +38, RMSE +1.6, all p ≤ 2×10⁻⁴), with
no significant change at rates 0.01 and 0.05.

**Reading.**

- The flags matter where silence is informative but the model cannot learn it alone:
  mid-rate panels. On the sparsest panels silence says little; on the densest the plain LSTM
  already gets the level right (bias +2%).
- Unbounded counters keep growing through the holdout, past any value seen in training, and
  the LSTM's forecast climbs with them. Its RMSE rises from 2.1–5.2 to 11.9–16.4 at rates
  0.05–0.30.
- Even with flags, the neural bias at churn 80% (+147 LSTM, +73 Transformer) stays far above
  Pareto/NBD's +16 at 1,000 customers. The cohort-size section shows that gap mostly closes
  for the LSTM at 3,000.

## Claims and whether the evidence supports them

Of 13 claims, five are supported under stated conditions, four only in part, and four are
not supported. All tests were recomputed from the stored forecasts.

**Setting.** Seasonal panels (4 peaks, amplitude 1.5), 1,000 customers, churn levels pooled,
unless a row says otherwise. Evidence gives the median paired change in |bias| / MAPE / RMSE
on customer totals. Most verdicts depend on the purchase rate, so every claim is tested per
rate.

**Tests.** Wilcoxon signed-rank, paired on the same 40 panels per rate. Mann–Whitney U where
panels differ (1,000 vs 3,000 customers). Spearman within each rate × churn cell for
hyperparameters. Significant at p < 0.05.

| # | Claim | Evidence | Verdict |
| --- | --- | --- | --- |
| 1 | Pareto/NBD beats the neural models. | vs LSTM `ar_bounded` (positive = LSTM worse): rate 0.01 +94 / +68 / +0.11 (all p < 10⁻¹⁰); 0.05 +37 / +15 / +0.19 (p ≤ 6×10⁻⁷); 0.10 +16 (p = 7×10⁻⁵) / −3 (n.s.) / +0.18 (p = 10⁻¹¹); 0.30 −3 (p = 0.04) / −15 (p = 4×10⁻⁹) / −0.17 (p = 10⁻⁵). vs Transformer `ar_bounded`: Pareto/NBD better on all three at every rate (p ≤ 0.03). | **Partly.** True against the Transformer everywhere and the LSTM at rates ≤ 0.10. At rate 0.30 the LSTM wins on all three metrics. |
| 2 | Neural error rises with churn. | 1,000 customers: bias and MAPE rise with churn in all 12 neural trees (LSTM `no_ar` +37 → +365 and 42 → 380). 3,000 customers, LSTM `ar_bounded`: bias +15 → +24 but MAPE still 21 → 63. RMSE falls with churn for every model because there is less volume, so it cannot be compared across churn. | **Partly.** Holds at 1,000 customers; with flags at 3,000 the bias trend is nearly gone but MAPE still rises. |
| 3 | Bounded AR flags help. | LSTM: better on all three at rates 0.01, 0.05, 0.10 (p ≤ 0.002); at 0.30 only RMSE (−0.18, p = 10⁻⁵). Transformer: better on all three at 0.05 and 0.10 (p ≤ 0.01); at 0.30 only RMSE (p = 0.009); nothing at 0.01. | **Supported** at rates 0.05–0.10 for both models (and 0.01 for the LSTM). On dense panels only RMSE improves. |
| 4 | Unbounded counters hurt. | LSTM worse on all three at every rate (p ≤ 0.002); RMSE rises to 3.4–16.4. Transformer worse at rate 0.30 (+44 / +38 / +1.6, p ≤ 2×10⁻⁴) and on RMSE at 0.10 (p = 0.004); no change at 0.01–0.05. | **Supported** for the LSTM at every rate; for the Transformer on dense panels only. |
| 5 | A k-means cluster label hurts. | LSTM, added to `no_ar`: worse on all three at 0.10 and 0.30 (p ≤ 9×10⁻⁵), no change at 0.01–0.05. Added to `ar_bounded`: worse at 0.05–0.30 (p ≤ 3×10⁻⁴). Transformer: worse at 0.30 (`no_ar` all three p ≤ 0.006), otherwise mostly RMSE only. Under `ar_unbounded` it helps both (pooled p ≤ 7×10⁻⁵). | **Supported** at rates ≥ 0.10 (LSTM from 0.05). No effect on sparse panels; helps only by displacing broken counters. |
| 6 | One architecture is better overall. | LSTM → Transformer, both `ar_bounded` (positive = Transformer worse): 0.01 −87 / −63 / −0.08 (p < 10⁻⁷); 0.05 +14 (n.s.) / +12 / +0.05 (p = 0.03); 0.10 +37 / +21 / +0.22 (p ≤ 6×10⁻⁵); 0.30 +25 / +23 / +0.56 (p ≤ 6×10⁻⁷). Same pattern for `no_ar`. Pooled over rates: p = 0.2. | **Not supported.** The Transformer is better on the sparsest panels, the LSTM from rate 0.10 up. A pooled test hides both. |
| 7 | Neural models capture seasonality; Pareto/NBD cannot. | Shape correlation LSTM `ar_bounded` 0.29–0.69 vs Pareto/NBD −0.07 to +0.09. Giving Pareto/NBD the true season cuts its MAPE by 14–17 points at rates 0.05–0.30 (p < 10⁻¹¹); RMSE and bias unchanged. | **Supported** for this one seasonal pattern. |
| 8 | Pareto/NBD's low bias means it is accurate per customer. | Rate 0.30, churn 20%: it serves living customers 74% of their volume (R_A 0.743) and leaks 11% onto dead ones (L_D 0.111). Its RMSE at rate 0.30 (5.17) is no better than the LSTM's (4.94). | **Not supported.** Two errors cancel in the total. |
| 9 | Neural models cannot detect a customer who has stopped. | Rate 0.30, churn 60–80%: LSTM `ar_bounded` dead leakage equals Pareto/NBD's (p = 0.63, 0.38). Rate 0.01: leakage 0.42 → 5.48 across churn. | **Partly.** True on sparse panels; on dense panels the flags work. |
| 10 | A bigger hyperparameter search helps. | `no_ar`, 10 (LSTM) / 20 (Transformer) trials → 100: at rate 0.30, LSTM −14 / −10 / −0.90 and Transformer −17 / −14 / −0.75 (all p ≤ 0.004). At rates 0.01–0.10 \|bias\| and MAPE do not change significantly; RMSE improves in 3 of 6 cases. | **Partly.** Clear on dense panels only. The archived run did not record its embedder. |
| 11 | Specific hyperparameters drive the error. | LSTM `no_ar` batch size vs \|bias\|: ρ = +0.48 pooled (p = 8×10⁻¹¹), +0.04 within cells (p = 0.66). The search simply picks larger batches on sparse panels. Only within-cell effect: Transformer `no_ar` layers, ρ = +0.18 (p = 0.02). | **Not supported.** The error follows the panel regime, not the chosen settings. |
| 12 | More customers improve the neural forecast. | LSTM `ar_bounded`, 1,000 → 3,000: \|bias\| and MAPE better at rates 0.01–0.10 (p < 10⁻⁵), RMSE only at 0.01; little change at 0.30. LSTM `no_ar`: better at 0.05–0.10 on all three, not at 0.01. | **Supported** at rates ≤ 0.10 (flags) or 0.05–0.10 (no flags). The Transformer was not run at 3,000. |
| 13 | RMSE can rank these models. | Per customer-week RMSE: 10 of 13 trees score 0.18. RMSE on customer totals does separate models within a rate (0.30: 4.94 to 16.35) but grows 8× from rate 0.01 to 0.30, so it cannot be pooled across rates. | **Not supported** for the per-week RMSE; the customer-total RMSE works within one rate. |

## What this does not show

- **Run-to-run noise.** Each panel has one study, so a cell's spread mixes panel variation
  with unseeded training variation. The paired tests are unaffected; single-cell spreads are
  not panel variation alone.
- **Transformer at 3,000 customers.** Not run, so claim 12 is LSTM-only.
- **Other encodings.** Only none / bounded / unbounded history and K = 8 were tested. The flag
  bins {2, 4, 8, 16, 32} were not tuned, and the `projected` embedder was not run.
- **Changing seasonality.** It is fixed across the grid, so nothing here says how the models
  behave under other seasonal patterns.
- **Transfer to real data.** The generator is a Pareto/NBD, so the benchmark is correct by
  construction here. On real panels there is no known ceiling (see
  `docs/benchmarks-real-panels.md`).
