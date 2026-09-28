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
one table only. The three "Beats P/NBD" columns compare the tree with Pareto/NBD on the
same panel, one metric each, and give the share of the table's panels where the tree wins:
lower MAPE, smaller |bias| (closer to 0, either sign), higher Spearman.

**Marks.** In each RMSE, Bias %, MAPE and Spearman column the best tree is **bold**: the
lowest mean over the table's panels (for bias, of the per-panel |bias|, so a tree whose
panels err +50 and −50 is not unbiased), or the highest mean Spearman. A † marks every tree
not significantly different from it: paired Wilcoxon on the same panels, p ≥ 0.05,
uncorrected. A paired test can tie a tree whose mean is far off the best when a few panels
blow up its mean while it wins the rest; Transformer `ar_unbounded` at rate 0.01 pooled is
the example. Val. CE and the "Beats" shares are not marked.

### By purchase rate and churn

One table per rate × churn cell, 10 panels per row (9 where a study was left out).

**Rate 0.01, churn 20%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **0.79 [0.75, 0.83]** | **+12 [+9, +16]** | **48 [45, 51]** | **0.22 [0.19, 0.25]** | — | — | — | — |
| LSTM | `no_ar` | 0.87 [0.82, 0.92] | +59 [+43, +76] | 71 [58, 83] | 0.01 [−0.02, 0.04] | 0.079 [0.076, 0.083] | 0% | 0% | 0% |
| LSTM | `ar_unbounded` | 6.70 [0.14, 13.27] | +502 [−91, +1096] | 513 [−77, 1104] | −0.16 [−0.21, −0.11] | 0.078 [0.074, 0.082] | 10% | 10% | 0% |
| LSTM | `ar_bounded` | 0.86 [0.80, 0.93] | +44 [+17, +71] | 65 [47, 82] | −0.03 [−0.10, 0.03] | 0.078 [0.074, 0.082] | 30% | 10% | 0% |
| LSTM | `ar_bounded` + `kmeans_8` | 0.88 [0.84, 0.93] | +56 [+36, +76] | 72 [61, 84] | 0.13 [0.09, 0.17] | 0.077 [0.073, 0.080] | 0% | 0% | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 5.09 [2.01, 8.17] | +235 [+72, +397] | 250 [89, 411] | −0.08 [−0.14, −0.03] | 0.075 [0.071, 0.079] | 0% | 0% | 0% |
| LSTM | `no_ar` + `kmeans_8` | 0.91 [0.85, 0.97] | +77 [+52, +102] | 86 [66, 107] | 0.14 [0.11, 0.17] | 0.078 [0.075, 0.082] | 0% | 0% | 0% |
| Transformer | `no_ar` | 0.84 [0.78, 0.90] | +19 [−12, +51] † | 58 [47, 69] † | 0.05 [0.02, 0.08] | 0.077 [0.074, 0.081] | 30% | 30% | 0% |
| Transformer | `ar_unbounded` | 0.83 [0.79, 0.87] | +7 [−13, +28] † | 49 [42, 57] † | −0.05 [−0.10, 0.01] | 0.077 [0.073, 0.081] | 60% | 60% | 0% |
| Transformer | `ar_bounded` | 0.85 [0.79, 0.92] | +36 [+4, +67] | 62 [44, 81] † | 0.03 [−0.00, 0.06] | 0.077 [0.074, 0.081] | 30% | 10% | 0% |
| Transformer | `ar_bounded` + `kmeans_8` | 0.90 [0.82, 0.99] | +23 [−31, +78] | 73 [41, 105] | 0.16 [0.13, 0.20] | 0.074 [0.070, 0.078] | 20% | 0% | 20% |
| Transformer | `ar_unbounded` + `kmeans_8` | 0.87 [0.82, 0.92] | −13 [−51, +26] | 65 [55, 75] | 0.03 [−0.04, 0.10] | 0.072 [0.068, 0.077] | 0% | 20% | 0% |
| Transformer | `no_ar` + `kmeans_8` | 0.86 [0.80, 0.92] | −3 [−34, +28] | 55 [47, 64] † | 0.12 [0.08, 0.16] | 0.077 [0.074, 0.081] | 40% | 20% | 0% |

**Rate 0.01, churn 40%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **0.68 [0.64, 0.72]** | **+34 [+26, +42]** | **66 [62, 70]** | **0.25 [0.23, 0.28]** | — | — | — | — |
| LSTM | `no_ar` | 0.80 [0.76, 0.84] | +132 [+110, +154] | 136 [116, 157] | 0.01 [−0.03, 0.05] | 0.068 [0.064, 0.072] | 0% | 0% | 0% |
| LSTM | `ar_unbounded` | 1.89 [0.28, 3.50] | +182 [+104, +260] | 189 [111, 266] | −0.23 [−0.25, −0.20] | 0.068 [0.064, 0.071] | 0% | 0% | 0% |
| LSTM | `ar_bounded` | 0.78 [0.74, 0.83] | +114 [+96, +133] | 121 [104, 137] | −0.09 [−0.13, −0.06] | 0.067 [0.064, 0.071] | 0% | 0% | 0% |
| LSTM | `ar_bounded` + `kmeans_8` | 0.80 [0.75, 0.84] | +116 [+95, +136] | 121 [103, 140] | 0.11 [0.07, 0.15] | 0.067 [0.064, 0.071] | 0% | 0% | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 11.37 [3.70, 19.04] | +1300 [+83, +2517] | 1304 [88, 2519] | −0.12 [−0.20, −0.03] | 0.066 [0.062, 0.070] | 0% | 0% | 0% |
| LSTM | `no_ar` + `kmeans_8` | 0.81 [0.78, 0.84] | +136 [+120, +152] | 140 [125, 155] | 0.14 [0.10, 0.17] | 0.067 [0.063, 0.071] | 0% | 0% | 0% |
| Transformer | `no_ar` | 0.75 [0.67, 0.83] | +61 [−2, +123] † | 96 [54, 138] † | 0.02 [−0.04, 0.07] | 0.066 [0.062, 0.069] | 60% | 60% | 0% |
| Transformer | `ar_unbounded` | 0.72 [0.68, 0.76] | +18 [−34, +70] † | 78 [52, 103] † | 0.00 [−0.09, 0.09] | 0.065 [0.061, 0.069] | 60% | 50% | 10% |
| Transformer | `ar_bounded` | 0.71 [0.68, 0.75] | +27 [−16, +69] | 75 [62, 89] † | −0.02 [−0.05, 0.01] | 0.065 [0.061, 0.069] | 40% | 20% | 0% |
| Transformer | `ar_bounded` + `kmeans_8` | 0.80 [0.66, 0.94] | +64 [−7, +134] † | 102 [51, 153] † | 0.16 [0.12, 0.19] | 0.062 [0.059, 0.066] | 40% | 40% | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 0.75 [0.71, 0.79] | +1 [−55, +57] † | 83 [60, 106] † | −0.00 [−0.06, 0.06] | 0.061 [0.057, 0.064] | 20% | 30% | 0% |
| Transformer | `no_ar` + `kmeans_8` | 0.92 [0.72, 1.12] | +117 [+16, +218] | 148 [68, 228] | 0.12 [0.08, 0.17] | 0.065 [0.061, 0.069] | 20% | 20% | 0% |

**Rate 0.01, churn 60%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **0.57 [0.53, 0.61]** | **+41 [+30, +52]** | **88 [83, 94]** | **0.26 [0.22, 0.31]** | — | — | — | — |
| LSTM | `no_ar` | 0.83 [0.78, 0.88] | +290 [+259, +320] | 291 [262, 321] | 0.01 [−0.04, 0.06] | 0.052 [0.048, 0.056] | 0% | 0% | 0% |
| LSTM | `ar_unbounded` | 3.39 [0.81, 5.97] | +514 [+302, +726] | 514 [302, 726] | −0.20 [−0.26, −0.14] | 0.052 [0.048, 0.056] | 0% | 0% | 0% |
| LSTM | `ar_bounded` | 0.71 [0.64, 0.77] | +190 [+137, +242] | 196 [146, 246] | −0.04 [−0.12, 0.03] | 0.051 [0.048, 0.055] | 0% | 0% | 0% |
| LSTM | `ar_bounded` + `kmeans_8` | 0.79 [0.73, 0.84] | +241 [+210, +272] | 245 [216, 274] | 0.07 [0.03, 0.11] | 0.051 [0.048, 0.054] | 0% | 0% | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 5.48 [0.80, 10.16] | +900 [+270, +1530] | 904 [276, 1531] | −0.04 [−0.15, 0.06] | 0.051 [0.047, 0.054] | 0% | 0% | 0% |
| LSTM | `no_ar` + `kmeans_8` | 0.82 [0.75, 0.88] | +271 [+232, +311] | 273 [234, 312] | 0.16 [0.11, 0.21] | 0.051 [0.047, 0.054] | 0% | 0% | 0% |
| Transformer | `no_ar` | 0.63 [0.59, 0.67] | +96 [+33, +159] † | 129 [86, 173] † | 0.07 [0.03, 0.10] | 0.050 [0.046, 0.054] | 30% | 30% | 0% |
| Transformer | `ar_unbounded` | 0.75 [0.59, 0.91] | +68 [+4, +132] † | 127 [79, 175] † | 0.06 [0.01, 0.10] | 0.049 [0.045, 0.053] | 50% | 40% | 0% |
| Transformer | `ar_bounded` | 0.60 [0.56, 0.64] † | +50 [+10, +91] † | 98 [78, 119] † | 0.06 [0.01, 0.11] | 0.049 [0.045, 0.053] | 40% | 40% | 0% |
| Transformer | `ar_bounded` + `kmeans_8` | 0.63 [0.56, 0.70] | +43 [−14, +99] † | 103 [74, 132] † | 0.18 [0.11, 0.24] | 0.047 [0.043, 0.050] | 50% | 50% | 10% |
| Transformer | `ar_unbounded` + `kmeans_8` | 0.68 [0.57, 0.79] | +68 [−1, +137] † | 120 [74, 166] † | 0.10 [0.04, 0.15] | 0.044 [0.041, 0.048] | 30% | 20% | 0% |
| Transformer | `no_ar` + `kmeans_8` | 0.64 [0.59, 0.70] | +65 [−2, +132] † | 120 [85, 155] † | 0.16 [0.11, 0.22] | 0.047 [0.043, 0.051] | 40% | 20% | 0% |

**Rate 0.01, churn 80%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **0.42 [0.38, 0.47]** | **+54 [+29, +80]** | **159 [132, 186]** | **0.20 [0.15, 0.25]** | — | — | — | — |
| LSTM | `no_ar` | 0.86 [0.77, 0.94] | +755 [+467, +1044] | 759 [474, 1045] | −0.02 [−0.06, 0.02] | 0.034 [0.030, 0.038] | 0% | 0% | 0% |
| LSTM | `ar_unbounded` | 1.53 [0.44, 2.62] | +821 [+430, +1212] | 838 [463, 1213] | −0.12 [−0.16, −0.08] | 0.034 [0.031, 0.038] | 0% | 0% | 0% |
| LSTM | `ar_bounded` | 0.71 [0.60, 0.82] | +576 [+278, +874] | 591 [304, 878] | −0.07 [−0.10, −0.03] | 0.034 [0.030, 0.038] | 0% | 0% | 0% |
| LSTM | `ar_bounded` + `kmeans_8` | 0.63 [0.52, 0.74] | +429 [+255, +603] | 445 [280, 611] | 0.07 [0.01, 0.13] | 0.033 [0.030, 0.037] | 0% | 0% | 10% |
| LSTM | `ar_unbounded` + `kmeans_8` | 6.26 [0.11, 12.42] | +1993 [+282, +3704] | 2015 [311, 3719] | −0.02 [−0.09, 0.05] | 0.033 [0.030, 0.036] | 0% | 0% | 0% |
| LSTM | `no_ar` + `kmeans_8` | 0.85 [0.78, 0.92] | +725 [+483, +968] | 729 [489, 969] | 0.11 [0.06, 0.17] | 0.033 [0.030, 0.037] | 0% | 0% | 10% |
| Transformer | `no_ar` | 0.44 [0.39, 0.50] † | +121 [+50, +192] | 197 [148, 246] | 0.07 [0.02, 0.13] | 0.031 [0.028, 0.035] | 20% | 30% | 0% |
| Transformer | `ar_unbounded` | 0.70 [0.17, 1.23] † | +264 [−37, +566] † | 345 [54, 636] † | 0.00 [−0.10, 0.10] | 0.031 [0.027, 0.034] | 30% | 30% | 0% |
| Transformer | `ar_bounded` | 0.44 [0.37, 0.50] † | +89 [+34, +144] † | 176 [145, 206] † | 0.08 [0.01, 0.15] | 0.031 [0.027, 0.034] | 40% | 40% | 10% |
| Transformer | `ar_bounded` + `kmeans_8` | 0.51 [0.41, 0.60] † | +109 [+31, +187] † | 196 [143, 249] † | 0.11 [0.04, 0.17] | 0.028 [0.025, 0.032] | 40% | 40% | 20% |
| Transformer | `ar_unbounded` + `kmeans_8` | 0.51 [0.41, 0.60] | +120 [+4, +236] | 216 [136, 295] † | 0.09 [0.02, 0.16] | 0.027 [0.024, 0.031] | 30% | 20% | 10% |
| Transformer | `no_ar` + `kmeans_8` | 0.48 [0.40, 0.55] † | +104 [−3, +210] † | 195 [114, 276] † | 0.15 [0.11, 0.19] | 0.030 [0.027, 0.034] | 60% | 50% | 20% |

**Rate 0.05, churn 20%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **1.92 [1.88, 1.96]** | **+8 [+5, +11]** | **35 [33, 37]** | **0.58 [0.55, 0.60]** | — | — | — | — |
| LSTM | `no_ar` | 2.33 [2.17, 2.49] | +42 [+31, +53] | 45 [35, 54] | 0.29 [0.10, 0.47] | 0.175 [0.172, 0.178] | 20% | 0% | 0% |
| LSTM | `ar_unbounded` | 14.23 [10.96, 17.50] | +240 [+173, +307] | 240 [173, 307] | 0.44 [0.42, 0.47] | 0.177 [0.174, 0.180] | 0% | 0% | 0% |
| LSTM | `ar_bounded` | 2.29 [2.22, 2.35] | +47 [+37, +56] | 48 [40, 57] | 0.50 [0.47, 0.54] | 0.174 [0.172, 0.176] | 10% | 0% | 0% |
| LSTM | `ar_bounded` + `kmeans_8` | 2.20 [2.10, 2.29] | +29 [+24, +34] | 39 [36, 42] | 0.45 [0.42, 0.47] | 0.169 [0.165, 0.172] | 20% | 0% | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 11.36 [8.81, 13.92] | +188 [+144, +233] | 189 [145, 233] | 0.46 [0.42, 0.50] | 0.174 [0.171, 0.177] | 0% | 0% | 0% |
| LSTM | `no_ar` + `kmeans_8` | 2.26 [2.14, 2.38] | +40 [+31, +50] | 47 [39, 54] | 0.46 [0.42, 0.50] | 0.169 [0.166, 0.172] | 0% | 0% | 0% |
| Transformer | `no_ar` | 2.70 [2.49, 2.90] | +72 [+52, +91] | 75 [59, 91] | 0.31 [0.19, 0.43] | 0.175 [0.173, 0.178] | 10% | 10% | 0% |
| Transformer | `ar_unbounded` | 2.69 [2.42, 2.95] | +90 [+65, +115] | 91 [68, 114] | 0.52 [0.50, 0.54] | 0.173 [0.171, 0.176] | 10% | 0% | 0% |
| Transformer | `ar_bounded` | 2.56 [2.38, 2.75] | +61 [+36, +87] | 68 [52, 84] | 0.45 [0.41, 0.49] | 0.174 [0.171, 0.176] | 20% | 10% | 0% |
| Transformer | `ar_bounded` + `kmeans_8` | 2.65 [2.46, 2.83] | +77 [+61, +93] | 79 [65, 93] | 0.51 [0.47, 0.54] | 0.168 [0.164, 0.171] | 0% | 0% | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 2.66 [2.35, 2.97] | +61 [+33, +88] | 69 [52, 86] | 0.47 [0.40, 0.54] | 0.168 [0.165, 0.171] | 20% | 10% | 0% |
| Transformer | `no_ar` + `kmeans_8` | 2.44 [2.22, 2.66] | +62 [+38, +86] | 67 [48, 87] | 0.49 [0.45, 0.53] | 0.169 [0.166, 0.172] | 20% | 0% | 0% |

**Rate 0.05, churn 40%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **1.63 [1.60, 1.66]** | **+15 [+10, +20]** | **39 [36, 41]** | **0.60 [0.58, 0.62]** | — | — | — | — |
| LSTM | `no_ar` | 2.24 [2.13, 2.36] | +98 [+84, +112] | 99 [85, 112] | 0.20 [0.02, 0.37] | 0.144 [0.141, 0.148] | 0% | 0% | 0% |
| LSTM | `ar_unbounded` | 10.38 [7.37, 13.38] | +313 [+241, +386] | 313 [241, 386] | 0.39 [0.35, 0.43] | 0.145 [0.141, 0.148] | 0% | 0% | 0% |
| LSTM | `ar_bounded` | 1.88 [1.82, 1.94] | +52 [+41, +63] | 56 [46, 65] | 0.54 [0.51, 0.57] | 0.138 [0.135, 0.141] | 10% | 0% | 0% |
| LSTM | `ar_bounded` + `kmeans_8` | 2.00 [1.90, 2.10] | +61 [+45, +77] | 64 [49, 79] | 0.45 [0.41, 0.48] | 0.135 [0.131, 0.139] | 0% | 0% | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 9.40 [3.61, 15.20] | +221 [+60, +381] | 229 [75, 384] | 0.37 [0.31, 0.44] | 0.140 [0.136, 0.144] | 30% | 20% | 0% |
| LSTM | `no_ar` + `kmeans_8` | 2.09 [2.04, 2.14] | +81 [+69, +93] | 82 [71, 94] | 0.46 [0.41, 0.51] | 0.137 [0.134, 0.140] | 0% | 0% | 0% |
| Transformer | `no_ar` | 2.28 [2.09, 2.47] | +108 [+78, +138] | 110 [81, 138] | 0.42 [0.37, 0.46] | 0.142 [0.139, 0.145] | 0% | 0% | 0% |
| Transformer | `ar_unbounded` | 2.30 [1.82, 2.79] | +101 [+66, +135] | 103 [71, 135] | 0.51 [0.43, 0.60] | 0.138 [0.135, 0.142] | 10% | 0% | 0% |
| Transformer | `ar_bounded` | 1.91 [1.68, 2.14] | +53 [+13, +93] | 65 [31, 98] † | 0.55 [0.52, 0.58] | 0.138 [0.135, 0.141] | 30% | 20% | 0% |
| Transformer | `ar_bounded` + `kmeans_8` | 2.00 [1.88, 2.11] | +48 [+28, +67] | 56 [41, 71] | 0.51 [0.49, 0.54] | 0.133 [0.130, 0.137] | 30% | 0% | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 2.15 [1.92, 2.37] | +68 [+28, +107] | 78 [46, 111] | 0.46 [0.37, 0.55] | 0.134 [0.130, 0.137] | 20% | 30% | 0% |
| Transformer | `no_ar` + `kmeans_8` | 2.37 [2.16, 2.59] | +105 [+71, +139] | 107 [74, 139] | 0.48 [0.44, 0.52] | 0.134 [0.130, 0.137] | 0% | 0% | 0% |

**Rate 0.05, churn 60%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **1.32 [1.26, 1.38]** | **+9 [0, +18]** | **42 [39, 45]** | **0.56 [0.55, 0.58]** | — | — | — | — |
| LSTM | `no_ar` | 1.95 [1.84, 2.07] | +157 [+135, +180] | 158 [135, 180] | 0.03 [−0.00, 0.06] | 0.109 [0.106, 0.112] | 0% | 0% | 0% |
| LSTM | `ar_unbounded` | 10.76 [7.97, 13.54] | +462 [+281, +642] | 462 [281, 642] | 0.35 [0.32, 0.39] | 0.108 [0.105, 0.112] | 0% | 0% | 0% |
| LSTM | `ar_bounded` | 1.43 [1.35, 1.52] | +60 [+41, +79] | 67 [55, 79] | 0.52 [0.50, 0.54] | 0.100 [0.097, 0.103] | 10% | 10% | 0% |
| LSTM | `ar_bounded` + `kmeans_8` | 1.80 [1.68, 1.92] | +108 [+83, +133] | 110 [86, 134] | 0.43 [0.39, 0.47] | 0.098 [0.095, 0.101] | 0% | 0% | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 8.50 [4.22, 12.79] | +357 [+110, +603] | 359 [114, 604] | 0.36 [0.27, 0.46] | 0.104 [0.100, 0.107] | 0% | 0% | 0% |
| LSTM | `no_ar` + `kmeans_8` | 1.99 [1.86, 2.12] | +172 [+149, +195] | 172 [149, 195] | 0.38 [0.31, 0.46] | 0.102 [0.099, 0.104] | 0% | 0% | 0% |
| Transformer | `no_ar` | 1.87 [1.65, 2.08] | +144 [+78, +210] | 149 [87, 210] | 0.37 [0.31, 0.43] | 0.105 [0.101, 0.108] | 10% | 10% | 0% |
| Transformer | `ar_unbounded` | 1.73 [1.58, 1.88] | +121 [+79, +164] | 124 [84, 164] | 0.49 [0.44, 0.54] | 0.100 [0.097, 0.104] | 0% | 0% | 0% |
| Transformer | `ar_bounded` | 1.47 [1.37, 1.57] | +66 [+30, +103] | 77 [49, 105] | 0.52 [0.50, 0.53] | 0.100 [0.096, 0.103] | 20% | 0% | 0% |
| Transformer | `ar_bounded` + `kmeans_8` | 2.02 [1.78, 2.26] | +152 [+99, +205] | 155 [104, 205] | 0.47 [0.43, 0.51] | 0.096 [0.092, 0.099] | 0% | 0% | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 1.90 [1.46, 2.34] | +98 [+32, +163] | 103 [40, 167] | 0.43 [0.35, 0.52] | 0.094 [0.091, 0.098] | 0% | 0% | 0% |
| Transformer | `no_ar` + `kmeans_8` | 1.91 [1.69, 2.12] | +131 [+77, +185] | 134 [84, 185] | 0.48 [0.46, 0.50] | 0.096 [0.093, 0.099] | 0% | 0% | 0% |

**Rate 0.05, churn 80%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **0.89 [0.85, 0.93]** | **+13 [−3, +29]** | **58 [52, 65]** | **0.44 [0.41, 0.47]** | — | — | — | — |
| LSTM | `no_ar` | 1.88 [1.71, 2.05] | +497 [+441, +553] | 497 [441, 553] | −0.01 [−0.03, 0.02] | 0.070 [0.064, 0.075] | 0% | 0% | 0% |
| LSTM | `ar_unbounded` | 12.14 [6.41, 17.87] | +1314 [+687, +1941] | 1315 [689, 1941] | 0.26 [0.19, 0.34] | 0.067 [0.061, 0.073] | 0% | 0% | 0% |
| LSTM | `ar_bounded` | 0.97 [0.92, 1.02] | +19 [−3, +41] † | 60 [50, 70] † | 0.40 [0.36, 0.44] | 0.061 [0.056, 0.066] | 70% | 40% | 10% |
| LSTM | `ar_bounded` + `kmeans_8` | 1.58 [1.30, 1.86] | +287 [+159, +414] | 290 [165, 415] | 0.36 [0.33, 0.39] | 0.061 [0.056, 0.065] | 0% | 0% | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 11.44 [7.38, 15.51] | +908 [+509, +1307] | 919 [535, 1304] | 0.32 [0.26, 0.38] | 0.065 [0.060, 0.070] | 0% | 0% | 0% |
| LSTM | `no_ar` + `kmeans_8` | 1.99 [1.87, 2.11] | +506 [+431, +582] | 506 [431, 582] | 0.34 [0.29, 0.39] | 0.064 [0.059, 0.069] | 0% | 0% | 0% |
| Transformer | `no_ar` | 1.13 [1.06, 1.19] | +110 [+55, +166] | 122 [76, 168] | 0.32 [0.27, 0.36] | 0.066 [0.061, 0.070] | 20% | 20% | 0% |
| Transformer | `ar_unbounded` | 1.09 [0.97, 1.22] | +70 [+12, +128] | 100 [54, 146] | 0.19 [0.02, 0.37] | 0.062 [0.057, 0.066] | 10% | 10% | 0% |
| Transformer | `ar_bounded` | 1.04 [0.98, 1.10] | +76 [+15, +137] | 102 [58, 146] | 0.40 [0.37, 0.44] | 0.061 [0.056, 0.066] | 20% | 10% | 10% |
| Transformer | `ar_bounded` + `kmeans_8` | 1.11 [1.06, 1.16] | +62 [+3, +120] | 94 [55, 134] † | 0.37 [0.33, 0.42] | 0.058 [0.053, 0.062] | 30% | 30% | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 1.15 [0.86, 1.45] | +63 [−9, +135] † | 97 [39, 155] † | 0.38 [0.35, 0.41] | 0.057 [0.052, 0.062] | 30% | 30% | 0% |
| Transformer | `no_ar` + `kmeans_8` | 1.29 [1.07, 1.50] | +161 [+49, +273] | 181 [85, 278] | 0.36 [0.29, 0.42] | 0.059 [0.054, 0.063] | 10% | 20% | 0% |

**Rate 0.10, churn 20%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **2.91 [2.85, 2.96]** | **+3 [+1, +5]** | **31 [30, 32]** | **0.72 [0.71, 0.74]** | — | — | — | — |
| LSTM | `no_ar` | 3.28 [3.13, 3.44] | +31 [+23, +40] | 33 [25, 41] † | 0.67 [0.65, 0.69] | 0.262 [0.254, 0.270] | 40% | 0% | 0% |
| LSTM | `ar_unbounded` | 21.15 [14.51, 27.78] | +301 [+199, +403] | 301 [199, 403] | 0.60 [0.58, 0.62] | 0.272 [0.264, 0.280] | 0% | 0% | 0% |
| LSTM | `ar_bounded` | 3.32 [3.17, 3.46] | +29 [+16, +43] | 32 [22, 43] † | 0.69 [0.68, 0.71] | 0.261 [0.254, 0.268] | 40% | 0% | 0% |
| LSTM | `ar_bounded` + `kmeans_8` | 3.46 [3.38, 3.54] | +28 [+23, +34] | 33 [28, 38] † | 0.60 [0.56, 0.65] | 0.254 [0.246, 0.263] | 40% | 0% | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 13.08 [7.26, 18.91] | +141 [+66, +217] | 145 [72, 218] | 0.58 [0.53, 0.62] | 0.265 [0.257, 0.272] | 0% | 0% | 0% |
| LSTM | `no_ar` + `kmeans_8` | 3.57 [3.44, 3.69] | +35 [+27, +44] | 39 [33, 45] | 0.58 [0.55, 0.62] | 0.258 [0.250, 0.267] | 20% | 0% | 0% |
| Transformer | `no_ar` | 4.11 [3.80, 4.41] | +54 [+32, +77] | 58 [41, 75] | 0.58 [0.56, 0.60] | 0.266 [0.258, 0.275] | 20% | 0% | 0% |
| Transformer | `ar_unbounded` | 5.00 [4.03, 5.97] | +101 [+69, +133] | 102 [70, 133] | 0.65 [0.63, 0.67] | 0.262 [0.254, 0.271] | 0% | 0% | 0% |
| Transformer | `ar_bounded` | 3.74 [3.50, 3.98] | +43 [+18, +69] | 52 [33, 71] † | 0.68 [0.66, 0.69] | 0.262 [0.254, 0.270] | 40% | 0% | 0% |
| Transformer | `ar_bounded` + `kmeans_8` | 3.71 [3.39, 4.03] | +36 [+16, +57] | 45 [31, 59] | 0.65 [0.59, 0.71] | 0.255 [0.246, 0.263] | 20% | 10% | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 3.86 [3.53, 4.18] | +38 [+20, +56] | 47 [36, 57] | 0.63 [0.56, 0.70] | 0.256 [0.247, 0.265] | 20% | 0% | 0% |
| Transformer | `no_ar` + `kmeans_8` | 3.98 [3.61, 4.36] | +54 [+34, +74] | 57 [41, 74] | 0.61 [0.57, 0.65] | 0.256 [0.248, 0.264] | 10% | 0% | 0% |

**Rate 0.10, churn 40%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **2.47 [2.37, 2.57]** | **+1 [−3, +6]** | **32 [30, 33]** | **0.73 [0.72, 0.74]** | — | — | — | — |
| LSTM | `no_ar` | 3.11 [2.57, 3.65] | +57 [+23, +91] | 58 [24, 92] | 0.64 [0.56, 0.71] | 0.205 [0.199, 0.210] | 20% | 0% | 0% |
| LSTM | `ar_unbounded` | 18.03 [11.59, 24.46] | +375 [+270, +479] | 375 [270, 479] | 0.57 [0.55, 0.59] | 0.213 [0.209, 0.218] | 0% | 0% | 0% |
| LSTM | `ar_bounded` | 2.79 [2.66, 2.92] | +28 [+16, +41] | 34 [25, 44] † | 0.69 [0.68, 0.71] | 0.198 [0.193, 0.203] | 60% | 10% | 0% |
| LSTM | `ar_bounded` + `kmeans_8` | 3.15 [2.98, 3.33] | +52 [+44, +59] | 53 [47, 60] | 0.64 [0.62, 0.65] | 0.193 [0.189, 0.198] | 0% | 0% | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 9.72 [4.50, 14.94] | +167 [+68, +266] | 174 [81, 267] | 0.56 [0.51, 0.62] | 0.203 [0.197, 0.209] | 0% | 0% | 0% |
| LSTM | `no_ar` + `kmeans_8` | 3.42 [3.30, 3.55] | +74 [+67, +81] | 74 [68, 81] | 0.57 [0.52, 0.62] | 0.198 [0.194, 0.203] | 0% | 0% | 0% |
| Transformer | `no_ar` | 3.69 [3.40, 3.99] | +86 [+45, +127] | 94 [62, 125] | 0.58 [0.57, 0.59] | 0.206 [0.201, 0.211] | 10% | 0% | 0% |
| Transformer | `ar_unbounded` | 3.89 [3.10, 4.69] | +89 [+60, +119] | 91 [62, 119] | 0.64 [0.58, 0.69] | 0.200 [0.195, 0.206] | 0% | 0% | 0% |
| Transformer | `ar_bounded` | 2.87 [2.56, 3.18] | +27 [+6, +47] | 41 [30, 51] † | 0.64 [0.53, 0.76] | 0.199 [0.194, 0.204] | 40% | 0% | 0% |
| Transformer | `ar_bounded` + `kmeans_8` | 3.42 [3.02, 3.81] | +51 [+26, +76] | 58 [39, 76] | 0.65 [0.64, 0.67] | 0.192 [0.187, 0.197] | 10% | 0% | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 3.74 [3.16, 4.33] | +72 [+41, +104] | 77 [49, 104] | 0.62 [0.58, 0.67] | 0.192 [0.187, 0.198] | 10% | 0% | 0% |
| Transformer | `no_ar` + `kmeans_8` | 3.99 [3.60, 4.39] | +93 [+72, +115] | 95 [76, 114] | 0.63 [0.61, 0.64] | 0.195 [0.191, 0.200] | 0% | 0% | 0% |

**Rate 0.10, churn 60%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **2.00 [1.93, 2.07]** | **+3 [−5, +12]** | 37 [35, 39] | **0.65 [0.64, 0.67]** | — | — | — | — |
| LSTM | `no_ar` | 2.43 [2.22, 2.64] | +61 [+27, +95] | 69 [40, 97] | 0.60 [0.58, 0.61] | 0.149 [0.142, 0.155] | 30% | 20% | 0% |
| LSTM | `ar_unbounded` | 7.56 [3.71, 11.41] | +299 [+138, +461] | 307 [154, 461] | 0.52 [0.46, 0.58] | 0.154 [0.145, 0.162] | 10% | 10% | 0% |
| LSTM | `ar_bounded` | 2.12 [2.04, 2.20] | +20 [+11, +29] | **33 [29, 37]** | 0.63 [0.61, 0.64] | 0.140 [0.135, 0.144] | 70% | 10% | 0% |
| LSTM | `ar_bounded` + `kmeans_8` | 2.79 [2.66, 2.93] | +92 [+69, +114] | 93 [72, 115] | 0.53 [0.49, 0.57] | 0.139 [0.135, 0.143] | 0% | 0% | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 6.06 [3.10, 9.03] | +133 [+21, +244] | 142 [35, 249] | 0.55 [0.50, 0.60] | 0.146 [0.141, 0.151] | 10% | 0% | 0% |
| LSTM | `no_ar` + `kmeans_8` | 3.12 [2.93, 3.30] | +132 [+101, +164] | 132 [101, 164] | 0.51 [0.46, 0.55] | 0.146 [0.141, 0.150] | 0% | 0% | 0% |
| Transformer | `no_ar` | 2.72 [2.41, 3.03] | +98 [+51, +144] | 103 [61, 145] | 0.52 [0.49, 0.55] | 0.150 [0.146, 0.154] | 10% | 10% | 0% |
| Transformer | `ar_unbounded` | 3.57 [2.92, 4.23] | +159 [+113, +204] | 159 [114, 204] | 0.57 [0.54, 0.61] | 0.143 [0.139, 0.147] | 0% | 0% | 0% |
| Transformer | `ar_bounded` | 2.74 [2.27, 3.22] | +104 [+45, +162] | 115 [68, 161] | 0.61 [0.59, 0.63] | 0.140 [0.136, 0.144] | 10% | 0% | 0% |
| Transformer | `ar_bounded` + `kmeans_8` | 3.05 [2.66, 3.45] | +106 [+63, +148] | 108 [68, 149] | 0.57 [0.55, 0.59] | 0.135 [0.132, 0.139] | 0% | 0% | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 2.83 [2.50, 3.15] | +70 [+37, +102] | 76 [49, 104] | 0.56 [0.52, 0.61] | 0.136 [0.132, 0.140] | 10% | 10% | 0% |
| Transformer | `no_ar` + `kmeans_8` | 3.01 [2.69, 3.33] | +106 [+74, +139] | 109 [78, 139] | 0.56 [0.55, 0.58] | 0.139 [0.135, 0.143] | 0% | 0% | 0% |

**Rate 0.10, churn 80%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **1.37 [1.23, 1.50]** | +7 [−11, +26] † | 49 [41, 58] | **0.51 [0.49, 0.53]** | — | — | — | — |
| LSTM | `no_ar` | 2.02 [1.66, 2.38] | +224 [+71, +377] | 233 [87, 380] | 0.27 [0.13, 0.42] | 0.091 [0.085, 0.097] | 20% | 20% | 0% |
| LSTM | `ar_unbounded` | 4.09 [1.42, 6.77] | +285 [+51, +519] | 307 [87, 526] | 0.43 [0.37, 0.48] | 0.085 [0.079, 0.090] | 10% | 10% | 0% |
| LSTM | `ar_bounded` | 1.45 [1.28, 1.62] | **+4 [−10, +18]** | **45 [39, 51]** | 0.47 [0.45, 0.49] | 0.077 [0.072, 0.081] | 90% | 80% | 0% |
| LSTM | `ar_bounded` + `kmeans_8` | 2.04 [1.87, 2.20] | +131 [+82, +181] | 134 [86, 182] | 0.45 [0.42, 0.48] | 0.076 [0.071, 0.080] | 0% | 0% | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 3.98 [1.30, 6.66] | +146 [−75, +368] | 189 [−15, 393] | 0.45 [0.42, 0.48] | 0.078 [0.072, 0.084] | 10% | 20% | 0% |
| LSTM | `no_ar` + `kmeans_8` | 2.42 [2.24, 2.61] | +309 [+202, +415] | 309 [203, 415] | 0.38 [0.33, 0.43] | 0.086 [0.082, 0.090] | 0% | 0% | 0% |
| Transformer | `no_ar` | 2.20 [2.01, 2.40] | +265 [+178, +352] | 265 [179, 352] | 0.41 [0.39, 0.43] | 0.088 [0.083, 0.092] | 0% | 0% | 0% |
| Transformer | `ar_unbounded` | 2.70 [2.14, 3.27] | +245 [+182, +307] | 245 [182, 307] | 0.45 [0.42, 0.49] | 0.080 [0.075, 0.085] | 0% | 0% | 0% |
| Transformer | `ar_bounded` | 1.57 [1.44, 1.69] | +82 [+20, +145] | 102 [54, 150] | 0.47 [0.45, 0.49] | 0.078 [0.074, 0.083] | 30% | 20% | 0% |
| Transformer | `ar_bounded` + `kmeans_8` | 2.31 [2.02, 2.60] | +175 [+118, +233] | 177 [121, 233] | 0.42 [0.37, 0.47] | 0.075 [0.071, 0.079] | 0% | 0% | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 2.21 [1.74, 2.68] | +142 [+73, +211] | 150 [88, 213] | 0.44 [0.40, 0.47] | 0.075 [0.071, 0.079] | 10% | 0% | 0% |
| Transformer | `no_ar` + `kmeans_8` | 2.39 [2.07, 2.71] | +201 [+116, +287] | 203 [120, 287] | 0.43 [0.40, 0.46] | 0.077 [0.072, 0.081] | 0% | 0% | 0% |

**Rate 0.30, churn 20%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | 6.93 [6.68, 7.17] | −14 [−16, −13] | 30 [30, 31] | **0.86 [0.85, 0.87]** | — | — | — | — |
| LSTM | `no_ar` | 6.92 [6.58, 7.26] | +17 [+12, +23] † | 19 [15, 23] | 0.85 [0.84, 0.86] | 0.497 [0.490, 0.505] | 100% | 20% | 0% |
| LSTM | `ar_unbounded` | 35.67 [30.14, 41.20] | +268 [+226, +309] | 268 [226, 309] | 0.73 [0.72, 0.74] | 0.546 [0.537, 0.555] | 0% | 0% | 0% |
| LSTM | `ar_bounded` | **6.49 [6.19, 6.79]** | **+10 [+7, +13]** | **12 [10, 14]** | 0.85 [0.84, 0.86] | 0.493 [0.484, 0.501] | 100% | 80% | 0% |
| LSTM | `ar_bounded` + `kmeans_8` | 7.96 [7.61, 8.31] | +25 [+21, +29] | 26 [22, 29] | 0.79 [0.77, 0.80] | 0.490 [0.481, 0.498] | 90% | 0% | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 15.60 [9.19, 22.02] | +76 [+38, +113] | 80 [47, 114] | 0.75 [0.72, 0.78] | 0.513 [0.501, 0.525] | 30% | 20% | 0% |
| LSTM | `no_ar` + `kmeans_8` | 8.31 [8.00, 8.61] | +27 [+24, +31] | 28 [25, 31] | 0.75 [0.73, 0.76] | 0.493 [0.485, 0.502] | 70% | 0% | 0% |
| Transformer | `no_ar` | 8.30 [7.14, 9.46] | +20 [−3, +42] † | 36 [21, 51] | 0.71 [0.69, 0.74] | 0.517 [0.510, 0.524] | 80% | 50% | 0% |
| Transformer | `ar_unbounded` | 12.13 [9.21, 15.05] | +69 [+41, +97] | 71 [45, 98] | 0.76 [0.71, 0.82] | 0.511 [0.503, 0.518] | 0% | 0% | 0% |
| Transformer | `ar_bounded` | 8.06 [6.74, 9.37] | +25 [+4, +46] † | 37 [23, 51] | 0.81 [0.79, 0.82] | 0.501 [0.492, 0.509] | 60% | 50% | 0% |
| Transformer | `ar_bounded` + `kmeans_8` | 8.08 [7.58, 8.58] | +25 [+16, +33] | 32 [27, 37] | 0.76 [0.70, 0.82] | 0.494 [0.486, 0.503] | 40% | 20% | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 8.84 [7.83, 9.86] | +30 [+17, +43] | 37 [31, 44] | 0.80 [0.79, 0.80] | 0.499 [0.490, 0.508] | 20% | 30% | 0% |
| Transformer | `no_ar` + `kmeans_8` | 8.20 [7.59, 8.80] | +21 [+11, +31] | 30 [24, 36] | 0.76 [0.75, 0.77] | 0.505 [0.496, 0.513] | 70% | 40% | 0% |

**Rate 0.30, churn 40%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | 5.99 [5.73, 6.24] | −14 [−16, −12] | 31 [30, 32] | **0.83 [0.82, 0.84]** | — | — | — | — |
| LSTM | `no_ar` | 5.74 [5.48, 6.01] † | +7 [+2, +13] † | 14 [13, 16] † | 0.81 [0.80, 0.82] | 0.361 [0.355, 0.367] | 100% | 80% | 0% |
| LSTM | `ar_unbounded` | 18.04 [9.50, 26.58] | +151 [+41, +261] | 161 [59, 263] | 0.74 [0.71, 0.77] | 0.400 [0.384, 0.415] | 10% | 0% | 0% |
| LSTM | `ar_bounded` | **5.66 [5.41, 5.91]** | **+2 [−3, +7]** | **13 [11, 14]** | 0.82 [0.81, 0.83] | 0.355 [0.349, 0.362] | 100% | 90% | 0% |
| LSTM | `ar_bounded` + `kmeans_8` | 7.67 [7.23, 8.12] | +44 [+36, +51] | 45 [38, 51] | 0.75 [0.74, 0.77] | 0.358 [0.352, 0.365] | 0% | 0% | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 10.25 [5.99, 14.51] | +43 [−3, +89] | 62 [26, 98] | 0.75 [0.72, 0.78] | 0.380 [0.371, 0.389] | 30% | 40% | 0% |
| LSTM | `no_ar` + `kmeans_8` | 8.24 [7.68, 8.81] | +53 [+47, +59] | 53 [48, 59] | 0.74 [0.73, 0.76] | 0.361 [0.354, 0.367] | 0% | 0% | 0% |
| Transformer | `no_ar` | 7.34 [6.73, 7.96] | +28 [+13, +43] | 36 [26, 46] | 0.68 [0.65, 0.70] | 0.386 [0.379, 0.393] | 40% | 20% | 0% |
| Transformer | `ar_unbounded` | 10.23 [8.28, 12.19] | +81 [+46, +115] | 86 [56, 115] | 0.69 [0.61, 0.76] | 0.376 [0.368, 0.384] | 10% | 10% | 0% |
| Transformer | `ar_bounded` | 6.88 [5.92, 7.84] | +23 [−4, +50] | 42 [28, 56] | 0.70 [0.59, 0.82] | 0.364 [0.356, 0.371] | 40% | 30% | 0% |
| Transformer | `ar_bounded` + `kmeans_8` | 7.90 [7.49, 8.31] | +45 [+32, +57] | 47 [37, 57] | 0.74 [0.73, 0.76] | 0.357 [0.350, 0.364] | 10% | 10% | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 7.85 [6.74, 8.97] | +21 [0, +42] | 41 [29, 53] | 0.74 [0.69, 0.78] | 0.363 [0.357, 0.369] | 40% | 30% | 0% |
| Transformer | `no_ar` + `kmeans_8` | 8.52 [7.79, 9.25] | +53 [+40, +66] | 55 [44, 66] | 0.70 [0.69, 0.72] | 0.373 [0.367, 0.379] | 10% | 0% | 0% |

**Rate 0.30, churn 60%** (9–10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | 4.72 [4.56, 4.88] | −12 [−17, −7] † | 32 [31, 33] | **0.77 [0.76, 0.77]** | — | — | — | — |
| LSTM | `no_ar` | 4.73 [4.47, 4.99] | **−1 [−10, +7]** | **18 [14, 22]** | 0.70 [0.69, 0.72] | 0.237 [0.230, 0.243] | 100% | 70% | 0% |
| LSTM | `ar_unbounded` | 7.04 [6.02, 8.06] | +56 [+32, +81] | 64 [45, 82] | 0.66 [0.64, 0.68] | 0.259 [0.249, 0.269] | 10% | 10% | 0% |
| LSTM | `ar_bounded` | **4.60 [4.41, 4.79]** | −7 [−15, +2] † | 19 [16, 22] † | 0.70 [0.69, 0.72] | 0.232 [0.226, 0.238] | 100% | 60% | 0% |
| LSTM | `ar_bounded` + `kmeans_8` | 6.73 [6.35, 7.11] | +71 [+62, +80] | 71 [63, 79] | 0.66 [0.64, 0.67] | 0.236 [0.229, 0.242] | 0% | 0% | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 6.49 [5.34, 7.63] | +11 [−26, +48] | 54 [35, 73] | 0.66 [0.64, 0.67] | 0.253 [0.244, 0.263] | 10% | 20% | 0% |
| LSTM | `no_ar` + `kmeans_8` | 7.35 [6.91, 7.80] | +90 [+79, +101] | 90 [79, 101] | 0.64 [0.62, 0.65] | 0.242 [0.234, 0.250] | 0% | 0% | 0% |
| Transformer | `no_ar` | 6.19 [5.64, 6.74] | +40 [+5, +74] | 57 [36, 78] | 0.59 [0.57, 0.60] | 0.265 [0.257, 0.272] | 30% | 20% | 0% |
| Transformer | `ar_unbounded` | 7.95 [7.03, 8.88] | +96 [+65, +127] | 97 [66, 127] | 0.63 [0.59, 0.67] | 0.251 [0.240, 0.262] | 0% | 0% | 0% |
| Transformer | `ar_bounded` | 5.81 [5.14, 6.48] | +59 [+24, +94] | 67 [39, 95] | 0.67 [0.65, 0.69] | 0.238 [0.231, 0.245] | 30% | 10% | 0% |
| Transformer | `ar_bounded` + `kmeans_8` | 7.33 [6.13, 8.53] | +83 [+55, +112] | 85 [57, 112] | 0.64 [0.62, 0.66] | 0.234 [0.227, 0.240] | 0% | 0% | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 7.05 [5.85, 8.24] | +54 [+20, +87] | 65 [40, 91] | 0.64 [0.61, 0.66] | 0.241 [0.234, 0.248] | 10% | 30% | 0% |
| Transformer | `no_ar` + `kmeans_8` | 6.74 [6.24, 7.24] | +65 [+47, +82] | 65 [48, 82] | 0.61 [0.60, 0.62] | 0.249 [0.242, 0.256] | 0% | 0% | 0% |

**Rate 0.30, churn 80%** (9–10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | 3.03 [2.81, 3.25] † | **−12 [−20, −3]** | 35 [33, 37] † | **0.70 [0.68, 0.72]** | — | — | — | — |
| LSTM | `no_ar` | 3.26 [2.99, 3.54] | −15 [−24, −7] † | **32 [27, 37]** | 0.51 [0.48, 0.53] | 0.121 [0.115, 0.127] | 70% | 30% | 0% |
| LSTM | `ar_unbounded` | 4.63 [3.03, 6.23] | +50 [0, +101] † | 70 [29, 111] † | 0.48 [0.46, 0.50] | 0.131 [0.124, 0.138] | 50% | 40% | 0% |
| LSTM | `ar_bounded` | **3.01 [2.81, 3.22]** | −13 [−26, +1] † | 35 [32, 39] | 0.51 [0.48, 0.53] | 0.117 [0.109, 0.125] | 50% | 40% | 0% |
| LSTM | `ar_bounded` + `kmeans_8` | 4.08 [3.84, 4.31] | +83 [+63, +103] | 84 [65, 103] | 0.49 [0.46, 0.51] | 0.118 [0.110, 0.126] | 0% | 0% | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 3.89 [3.44, 4.33] | 0 [−21, +22] | 47 [39, 54] | 0.48 [0.46, 0.51] | 0.126 [0.117, 0.135] | 10% | 30% | 0% |
| LSTM | `no_ar` + `kmeans_8` | 5.34 [4.71, 5.97] | +142 [+108, +175] | 142 [108, 175] | 0.48 [0.46, 0.49] | 0.127 [0.117, 0.136] | 0% | 0% | 0% |
| Transformer | `no_ar` | 4.06 [3.56, 4.55] | +86 [+30, +141] | 96 [49, 144] | 0.44 [0.42, 0.46] | 0.141 [0.132, 0.150] | 20% | 20% | 0% |
| Transformer | `ar_unbounded` | 4.84 [3.29, 6.40] | +112 [+44, +180] | 118 [55, 181] | 0.43 [0.40, 0.47] | 0.128 [0.118, 0.138] | 11% | 11% | 0% |
| Transformer | `ar_bounded` | 3.18 [2.93, 3.43] † | +46 [+28, +64] | 54 [43, 66] | 0.48 [0.46, 0.51] | 0.119 [0.111, 0.126] | 20% | 20% | 0% |
| Transformer | `ar_bounded` + `kmeans_8` | 4.54 [3.62, 5.46] | +87 [+46, +128] | 91 [52, 129] | 0.47 [0.43, 0.50] | 0.118 [0.110, 0.125] | 10% | 10% | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 4.26 [3.79, 4.73] | +39 [+16, +62] | 56 [46, 66] | 0.47 [0.44, 0.50] | 0.123 [0.113, 0.132] | 0% | 10% | 0% |
| Transformer | `no_ar` + `kmeans_8` | 4.96 [4.18, 5.75] | +119 [+82, +155] | 120 [84, 155] | 0.45 [0.43, 0.48] | 0.129 [0.120, 0.138] | 0% | 0% | 0% |

### By purchase rate, churn pooled

The same tables with the four churn levels pooled, 40 panels per row.

**Rate 0.01**, churn 20–80% pooled (40 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **0.62 [0.57, 0.66]** | **+35 [+27, +43]** | **90 [75, 105]** | **0.23 [0.21, 0.25]** | — | — | — | — |
| LSTM | `no_ar` | 0.84 [0.81, 0.87] | +309 [+201, +417] | 314 [208, 421] | 0.00 [−0.01, 0.02] | 0.058 [0.053, 0.064] | 0% | 0% | 0% |
| LSTM | `ar_unbounded` | 3.38 [1.67, 5.08] | +505 [+329, +681] | 513 [339, 688] | −0.18 [−0.20, −0.15] | 0.058 [0.052, 0.063] | 2% | 2% | 0% |
| LSTM | `ar_bounded` | 0.77 [0.73, 0.81] | +231 [+138, +324] | 243 [151, 335] | −0.06 [−0.08, −0.03] | 0.058 [0.052, 0.063] | 8% | 2% | 0% |
| LSTM | `ar_bounded` + `kmeans_8` | 0.77 [0.73, 0.82] | +210 [+150, +270] | 221 [162, 280] | 0.10 [0.07, 0.12] | 0.057 [0.051, 0.063] | 0% | 0% | 2% |
| LSTM | `ar_unbounded` + `kmeans_8` | 7.05 [4.49, 9.62] | +1107 [+592, +1622] | 1118 [604, 1632] | −0.07 [−0.10, −0.03] | 0.056 [0.051, 0.061] | 0% | 0% | 0% |
| LSTM | `no_ar` + `kmeans_8` | 0.85 [0.82, 0.88] | +302 [+204, +400] | 307 [210, 404] | 0.14 [0.12, 0.16] | 0.057 [0.052, 0.063] | 0% | 0% | 2% |
| Transformer | `no_ar` | 0.67 [0.61, 0.72] | +74 [+46, +102] | 120 [96, 144] | 0.05 [0.03, 0.07] | 0.056 [0.050, 0.062] | 35% | 38% | 0% |
| Transformer | `ar_unbounded` | 0.75 [0.63, 0.87] | +89 [+14, +165] | 150 [76, 224] † | 0.00 [−0.03, 0.04] | 0.055 [0.050, 0.061] | 50% | 45% | 2% |
| Transformer | `ar_bounded` | 0.65 [0.60, 0.70] | +50 [+30, +71] | 103 [86, 120] | 0.04 [0.01, 0.06] | 0.056 [0.050, 0.061] | 38% | 28% | 2% |
| Transformer | `ar_bounded` + `kmeans_8` | 0.71 [0.64, 0.78] | +60 [+30, +90] | 119 [95, 142] | 0.15 [0.13, 0.18] | 0.053 [0.047, 0.059] | 38% | 32% | 12% |
| Transformer | `ar_unbounded` + `kmeans_8` | 0.70 [0.65, 0.76] | +44 [+7, +81] | 121 [93, 149] | 0.05 [0.02, 0.08] | 0.051 [0.045, 0.057] | 20% | 22% | 2% |
| Transformer | `no_ar` + `kmeans_8` | 0.73 [0.65, 0.80] | +71 [+32, +109] | 130 [99, 160] | 0.14 [0.12, 0.16] | 0.055 [0.049, 0.061] | 40% | 28% | 5% |

**Rate 0.05**, churn 20–80% pooled (40 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **1.44 [1.31, 1.57]** | **+11 [+7, +16]** | **44 [40, 47]** | **0.54 [0.52, 0.57]** | — | — | — | — |
| LSTM | `no_ar` | 2.10 [2.02, 2.19] | +199 [+140, +258] | 200 [141, 258] | 0.13 [0.06, 0.20] | 0.125 [0.112, 0.137] | 5% | 0% | 0% |
| LSTM | `ar_unbounded` | 11.88 [10.14, 13.61] | +582 [+384, +781] | 582 [384, 781] | 0.36 [0.33, 0.39] | 0.124 [0.111, 0.138] | 0% | 0% | 0% |
| LSTM | `ar_bounded` | 1.64 [1.48, 1.80] | +44 [+36, +53] | 58 [53, 63] | 0.49 [0.47, 0.51] | 0.118 [0.104, 0.132] | 25% | 12% | 2% |
| LSTM | `ar_bounded` + `kmeans_8` | 1.89 [1.79, 2.00] | +121 [+78, +164] | 126 [84, 168] | 0.42 [0.40, 0.44] | 0.116 [0.102, 0.129] | 5% | 0% | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 10.18 [8.27, 12.08] | +418 [+276, +561] | 424 [283, 565] | 0.38 [0.35, 0.41] | 0.121 [0.107, 0.134] | 8% | 5% | 0% |
| LSTM | `no_ar` + `kmeans_8` | 2.08 [2.02, 2.14] | +200 [+138, +262] | 202 [141, 263] | 0.41 [0.38, 0.44] | 0.118 [0.105, 0.131] | 0% | 0% | 0% |
| Transformer | `no_ar` | 1.99 [1.79, 2.20] | +109 [+87, +130] | 114 [94, 134] | 0.35 [0.32, 0.39] | 0.122 [0.109, 0.135] | 10% | 10% | 0% |
| Transformer | `ar_unbounded` | 1.95 [1.72, 2.19] | +95 [+77, +114] | 105 [89, 121] | 0.43 [0.37, 0.49] | 0.118 [0.105, 0.132] | 8% | 2% | 0% |
| Transformer | `ar_bounded` | 1.74 [1.55, 1.94] | +64 [+45, +83] | 78 [63, 93] | 0.48 [0.46, 0.50] | 0.118 [0.104, 0.132] | 22% | 10% | 2% |
| Transformer | `ar_bounded` + `kmeans_8` | 1.94 [1.75, 2.13] | +85 [+62, +107] | 96 [77, 115] | 0.47 [0.44, 0.49] | 0.113 [0.100, 0.127] | 15% | 8% | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 1.97 [1.74, 2.19] | +72 [+48, +96] | 87 [66, 107] | 0.44 [0.40, 0.47] | 0.113 [0.100, 0.127] | 18% | 18% | 0% |
| Transformer | `no_ar` + `kmeans_8` | 2.00 [1.82, 2.18] | +115 [+84, +145] | 122 [94, 151] | 0.45 [0.43, 0.48] | 0.114 [0.101, 0.128] | 8% | 5% | 0% |

**Rate 0.10**, churn 20–80% pooled (40 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **2.19 [2.00, 2.38]** | **+4 [−1, +8]** | 37 [34, 40] † | **0.65 [0.62, 0.68]** | — | — | — | — |
| LSTM | `no_ar` | 2.71 [2.49, 2.93] | +93 [+51, +136] | 98 [57, 140] | 0.55 [0.48, 0.61] | 0.177 [0.156, 0.197] | 28% | 10% | 0% |
| LSTM | `ar_unbounded` | 12.71 [9.51, 15.90] | +315 [+245, +385] | 322 [256, 389] | 0.53 [0.50, 0.56] | 0.181 [0.158, 0.204] | 5% | 5% | 0% |
| LSTM | `ar_bounded` | 2.42 [2.18, 2.65] | +20 [+14, +27] | **36 [32, 40]** | 0.62 [0.59, 0.65] | 0.169 [0.147, 0.191] | 65% | 25% | 0% |
| LSTM | `ar_bounded` + `kmeans_8` | 2.86 [2.68, 3.04] | +76 [+58, +93] | 78 [61, 95] | 0.56 [0.53, 0.58] | 0.166 [0.144, 0.187] | 10% | 0% | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 8.21 [6.01, 10.41] | +147 [+87, +207] | 162 [106, 219] | 0.54 [0.51, 0.56] | 0.173 [0.151, 0.196] | 5% | 5% | 0% |
| LSTM | `no_ar` + `kmeans_8` | 3.13 [2.98, 3.29] | +138 [+96, +179] | 139 [97, 180] | 0.51 [0.48, 0.54] | 0.172 [0.151, 0.193] | 5% | 0% | 0% |
| Transformer | `no_ar` | 3.18 [2.91, 3.45] | +126 [+90, +161] | 130 [96, 164] | 0.52 [0.50, 0.55] | 0.178 [0.156, 0.199] | 10% | 2% | 0% |
| Transformer | `ar_unbounded` | 3.79 [3.37, 4.21] | +148 [+121, +176] | 149 [122, 176] | 0.58 [0.55, 0.61] | 0.171 [0.149, 0.193] | 0% | 0% | 0% |
| Transformer | `ar_bounded` | 2.73 [2.44, 3.01] | +64 [+42, +86] | 77 [59, 96] | 0.60 [0.56, 0.63] | 0.170 [0.148, 0.192] | 30% | 5% | 0% |
| Transformer | `ar_bounded` + `kmeans_8` | 3.12 [2.89, 3.35] | +92 [+68, +117] | 97 [74, 120] | 0.57 [0.54, 0.61] | 0.164 [0.142, 0.186] | 8% | 2% | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 3.16 [2.87, 3.45] | +81 [+59, +103] | 88 [67, 108] | 0.56 [0.53, 0.60] | 0.165 [0.143, 0.187] | 12% | 2% | 0% |
| Transformer | `no_ar` + `kmeans_8` | 3.34 [3.08, 3.61] | +114 [+87, +141] | 116 [90, 143] | 0.56 [0.53, 0.58] | 0.167 [0.145, 0.188] | 2% | 0% | 0% |

**Rate 0.30**, churn 20–80% pooled (38–40 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | 5.17 [4.68, 5.65] | −13 [−15, −11] | 32 [31, 33] | **0.79 [0.77, 0.81]** | — | — | — | — |
| LSTM | `no_ar` | 5.16 [4.71, 5.62] | +2 [−3, +7] † | 21 [18, 24] † | 0.72 [0.67, 0.76] | 0.304 [0.258, 0.349] | 92% | 50% | 0% |
| LSTM | `ar_unbounded` | 16.34 [11.80, 20.89] | +131 [+91, +171] | 141 [104, 178] | 0.65 [0.62, 0.69] | 0.334 [0.284, 0.384] | 18% | 12% | 0% |
| LSTM | `ar_bounded` | **4.94 [4.51, 5.38]** | **−2 [−7, +3]** | **20 [16, 23]** | 0.72 [0.68, 0.76] | 0.299 [0.254, 0.345] | 88% | 68% | 0% |
| LSTM | `ar_bounded` + `kmeans_8` | 6.61 [6.09, 7.13] | +56 [+47, +65] | 56 [48, 65] | 0.67 [0.63, 0.71] | 0.300 [0.255, 0.345] | 22% | 0% | 0% |
| LSTM | `ar_unbounded` + `kmeans_8` | 9.06 [6.86, 11.26] | +33 [+14, +51] | 61 [49, 73] | 0.66 [0.62, 0.70] | 0.318 [0.271, 0.365] | 20% | 28% | 0% |
| LSTM | `no_ar` + `kmeans_8` | 7.31 [6.87, 7.75] | +78 [+62, +94] | 78 [62, 94] | 0.65 [0.61, 0.69] | 0.306 [0.262, 0.350] | 18% | 0% | 0% |
| Transformer | `no_ar` | 6.47 [5.87, 7.08] | +43 [+26, +61] | 56 [42, 71] | 0.60 [0.57, 0.64] | 0.327 [0.282, 0.373] | 42% | 28% | 0% |
| Transformer | `ar_unbounded` | 8.92 [7.66, 10.17] | +89 [+70, +108] | 92 [75, 110] | 0.63 [0.59, 0.68] | 0.323 [0.276, 0.371] | 5% | 5% | 0% |
| Transformer | `ar_bounded` | 5.98 [5.28, 6.68] | +38 [+26, +51] | 50 [42, 59] | 0.66 [0.62, 0.71] | 0.305 [0.259, 0.351] | 38% | 28% | 0% |
| Transformer | `ar_bounded` + `kmeans_8` | 6.96 [6.38, 7.54] | +60 [+46, +74] | 64 [51, 77] | 0.65 [0.61, 0.69] | 0.301 [0.255, 0.346] | 15% | 10% | 0% |
| Transformer | `ar_unbounded` + `kmeans_8` | 7.00 [6.30, 7.70] | +36 [+25, +47] | 50 [42, 57] | 0.66 [0.62, 0.70] | 0.306 [0.261, 0.352] | 18% | 25% | 0% |
| Transformer | `no_ar` + `kmeans_8` | 7.11 [6.57, 7.64] | +64 [+50, +79] | 67 [54, 81] | 0.63 [0.59, 0.67] | 0.314 [0.269, 0.359] | 20% | 10% | 0% |

### The three best trees per cell, by MAPE

Ranked by mean MAPE over the cell's 10 panels (lowest first); MAPE carries its 95% interval,
the other metrics are means. "p vs rank 1" is a paired Wilcoxon test of that tree against
the cell's leader on the same panels.

| Rate | Churn | Rank | Model | Arm | **MAPE** | Spearman | RMSE | Bias % | Val. CE | p vs rank 1 | Beats P/NBD: MAPE / \|bias\| / Spearman |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| 0.01 | 20% | 1 | **Pareto/NBD** | — | 48 [45, 51] | 0.22 | 0.79 | +12 | — | — | — |
|  |  | 2 | Transformer | `ar_unbounded` | 49 [42, 57] | −0.05 | 0.83 | +7 | 0.077 | 0.695 | 60% / 60% / 0% |
|  |  | 3 | Transformer | `no_ar` + `kmeans_8` | 55 [47, 64] | 0.12 | 0.86 | −3 | 0.077 | 0.160 | 40% / 20% / 0% |
| 0.01 | 40% | 1 | **Pareto/NBD** | — | 66 [62, 70] | 0.25 | 0.68 | +34 | — | — | — |
|  |  | 2 | Transformer | `ar_bounded` | 75 [62, 89] | −0.02 | 0.71 | +27 | 0.065 | 0.193 | 40% / 20% / 0% |
|  |  | 3 | Transformer | `ar_unbounded` | 78 [52, 103] | 0.00 | 0.72 | +18 | 0.065 | 0.770 | 60% / 50% / 10% |
| 0.01 | 60% | 1 | **Pareto/NBD** | — | 88 [83, 94] | 0.26 | 0.57 | +41 | — | — | — |
|  |  | 2 | Transformer | `ar_bounded` | 98 [78, 119] | 0.06 | 0.60 | +50 | 0.049 | 0.492 | 40% / 40% / 0% |
|  |  | 3 | Transformer | `ar_bounded` + `kmeans_8` | 103 [74, 132] | 0.18 | 0.63 | +43 | 0.047 | 0.625 | 50% / 50% / 10% |
| 0.01 | 80% | 1 | **Pareto/NBD** | — | 159 [132, 186] | 0.20 | 0.42 | +54 | — | — | — |
|  |  | 2 | Transformer | `ar_bounded` | 176 [145, 206] | 0.08 | 0.44 | +89 | 0.031 | 0.492 | 40% / 40% / 10% |
|  |  | 3 | Transformer | `no_ar` + `kmeans_8` | 195 [114, 276] | 0.15 | 0.48 | +104 | 0.030 | 0.922 | 60% / 50% / 20% |
| 0.05 | 20% | 1 | **Pareto/NBD** | — | 35 [33, 37] | 0.58 | 1.92 | +8 | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` + `kmeans_8` | 39 [36, 42] | 0.45 | 2.20 | +29 | 0.169 | 0.014 | 20% / 0% / 0% |
|  |  | 3 | LSTM | `no_ar` | 45 [35, 54] | 0.29 | 2.33 | +42 | 0.175 | 0.049 | 20% / 0% / 0% |
| 0.05 | 40% | 1 | **Pareto/NBD** | — | 39 [36, 41] | 0.60 | 1.63 | +15 | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 56 [46, 65] | 0.54 | 1.88 | +52 | 0.138 | 0.004 | 10% / 0% / 0% |
|  |  | 3 | Transformer | `ar_bounded` + `kmeans_8` | 56 [41, 71] | 0.51 | 2.00 | +48 | 0.133 | 0.049 | 30% / 0% / 0% |
| 0.05 | 60% | 1 | **Pareto/NBD** | — | 42 [39, 45] | 0.56 | 1.32 | +9 | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 67 [55, 79] | 0.52 | 1.43 | +60 | 0.100 | 0.006 | 10% / 10% / 0% |
|  |  | 3 | Transformer | `ar_bounded` | 77 [49, 105] | 0.52 | 1.47 | +66 | 0.100 | 0.014 | 20% / 0% / 0% |
| 0.05 | 80% | 1 | **Pareto/NBD** | — | 58 [52, 65] | 0.44 | 0.89 | +13 | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 60 [50, 70] | 0.40 | 0.97 | +19 | 0.061 | 0.922 | 70% / 40% / 10% |
|  |  | 3 | Transformer | `ar_bounded` + `kmeans_8` | 94 [55, 134] | 0.37 | 1.11 | +62 | 0.058 | 0.105 | 30% / 30% / 0% |
| 0.10 | 20% | 1 | **Pareto/NBD** | — | 31 [30, 32] | 0.72 | 2.91 | +3 | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 32 [22, 43] | 0.69 | 3.32 | +29 | 0.261 | 0.770 | 40% / 0% / 0% |
|  |  | 3 | LSTM | `ar_bounded` + `kmeans_8` | 33 [28, 38] | 0.60 | 3.46 | +28 | 0.254 | 0.492 | 40% / 0% / 0% |
| 0.10 | 40% | 1 | **Pareto/NBD** | — | 32 [30, 33] | 0.73 | 2.47 | +1 | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 34 [25, 44] | 0.69 | 2.79 | +28 | 0.198 | 0.770 | 60% / 10% / 0% |
|  |  | 3 | Transformer | `ar_bounded` | 41 [30, 51] | 0.64 | 2.87 | +27 | 0.199 | 0.160 | 40% / 0% / 0% |
| 0.10 | 60% | 1 | LSTM | `ar_bounded` | 33 [29, 37] | 0.63 | 2.12 | +20 | 0.140 | — | 70% / 10% / 0% |
|  |  | 2 | **Pareto/NBD** | — | 37 [35, 39] | 0.65 | 2.00 | +3 | — | 0.049 | — |
|  |  | 3 | LSTM | `no_ar` | 69 [40, 97] | 0.60 | 2.43 | +61 | 0.149 | 0.014 | 30% / 20% / 0% |
| 0.10 | 80% | 1 | LSTM | `ar_bounded` | 45 [39, 51] | 0.47 | 1.45 | +4 | 0.077 | — | 90% / 80% / 0% |
|  |  | 2 | **Pareto/NBD** | — | 49 [41, 58] | 0.51 | 1.37 | +7 | — | 0.037 | — |
|  |  | 3 | Transformer | `ar_bounded` | 102 [54, 150] | 0.47 | 1.57 | +82 | 0.078 | 0.006 | 30% / 20% / 0% |
| 0.30 | 20% | 1 | LSTM | `ar_bounded` | 12 [10, 14] | 0.85 | 6.49 | +10 | 0.493 | — | 100% / 80% / 0% |
|  |  | 2 | LSTM | `no_ar` | 19 [15, 23] | 0.85 | 6.92 | +17 | 0.497 | 0.014 | 100% / 20% / 0% |
|  |  | 3 | LSTM | `ar_bounded` + `kmeans_8` | 26 [22, 29] | 0.79 | 7.96 | +25 | 0.490 | 0.002 | 90% / 0% / 0% |
| 0.30 | 40% | 1 | LSTM | `ar_bounded` | 13 [11, 14] | 0.82 | 5.66 | +2 | 0.355 | — | 100% / 90% / 0% |
|  |  | 2 | LSTM | `no_ar` | 14 [13, 16] | 0.81 | 5.74 | +7 | 0.361 | 0.064 | 100% / 80% / 0% |
|  |  | 3 | **Pareto/NBD** | — | 31 [30, 32] | 0.83 | 5.99 | −14 | — | 0.002 | — |
| 0.30 | 60% | 1 | LSTM | `no_ar` | 18 [14, 22] | 0.70 | 4.73 | −1 | 0.237 | — | 100% / 70% / 0% |
|  |  | 2 | LSTM | `ar_bounded` | 19 [16, 22] | 0.70 | 4.60 | −7 | 0.232 | 0.625 | 100% / 60% / 0% |
|  |  | 3 | **Pareto/NBD** | — | 32 [31, 33] | 0.77 | 4.72 | −12 | — | 0.002 | — |
| 0.30 | 80% | 1 | LSTM | `no_ar` | 32 [27, 37] | 0.51 | 3.26 | −15 | 0.121 | — | 70% / 30% / 0% |
|  |  | 2 | **Pareto/NBD** | — | 35 [33, 37] | 0.70 | 3.03 | −12 | — | 0.131 | — |
|  |  | 3 | LSTM | `ar_bounded` | 35 [32, 39] | 0.51 | 3.01 | −13 | 0.117 | 0.020 | 50% / 40% / 0% |

### The three best trees per cell, by Spearman

Ranked by mean per-customer Spearman (highest first); layout as above.

| Rate | Churn | Rank | Model | Arm | MAPE | **Spearman** | RMSE | Bias % | Val. CE | p vs rank 1 | Beats P/NBD: MAPE / \|bias\| / Spearman |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |
| 0.01 | 20% | 1 | **Pareto/NBD** | — | 48 | 0.22 [0.19, 0.25] | 0.79 | +12 | — | — | — |
|  |  | 2 | Transformer | `ar_bounded` + `kmeans_8` | 73 | 0.16 [0.13, 0.20] | 0.90 | +23 | 0.074 | 0.010 | 20% / 0% / 20% |
|  |  | 3 | LSTM | `no_ar` + `kmeans_8` | 86 | 0.14 [0.11, 0.17] | 0.91 | +77 | 0.078 | 0.002 | 0% / 0% / 0% |
| 0.01 | 40% | 1 | **Pareto/NBD** | — | 66 | 0.25 [0.23, 0.28] | 0.68 | +34 | — | — | — |
|  |  | 2 | Transformer | `ar_bounded` + `kmeans_8` | 102 | 0.16 [0.12, 0.19] | 0.80 | +64 | 0.062 | 0.002 | 40% / 40% / 0% |
|  |  | 3 | LSTM | `no_ar` + `kmeans_8` | 140 | 0.14 [0.10, 0.17] | 0.81 | +136 | 0.067 | 0.002 | 0% / 0% / 0% |
| 0.01 | 60% | 1 | **Pareto/NBD** | — | 88 | 0.26 [0.22, 0.31] | 0.57 | +41 | — | — | — |
|  |  | 2 | Transformer | `ar_bounded` + `kmeans_8` | 103 | 0.18 [0.11, 0.24] | 0.63 | +43 | 0.047 | 0.004 | 50% / 50% / 10% |
|  |  | 3 | Transformer | `no_ar` + `kmeans_8` | 120 | 0.16 [0.11, 0.22] | 0.64 | +65 | 0.047 | 0.002 | 40% / 20% / 0% |
| 0.01 | 80% | 1 | **Pareto/NBD** | — | 159 | 0.20 [0.15, 0.25] | 0.42 | +54 | — | — | — |
|  |  | 2 | Transformer | `no_ar` + `kmeans_8` | 195 | 0.15 [0.11, 0.19] | 0.48 | +104 | 0.030 | 0.020 | 60% / 50% / 20% |
|  |  | 3 | LSTM | `no_ar` + `kmeans_8` | 729 | 0.11 [0.06, 0.17] | 0.85 | +725 | 0.033 | 0.006 | 0% / 0% / 10% |
| 0.05 | 20% | 1 | **Pareto/NBD** | — | 35 | 0.58 [0.55, 0.60] | 1.92 | +8 | — | — | — |
|  |  | 2 | Transformer | `ar_unbounded` | 91 | 0.52 [0.50, 0.54] | 2.69 | +90 | 0.173 | 0.002 | 10% / 0% / 0% |
|  |  | 3 | Transformer | `ar_bounded` + `kmeans_8` | 79 | 0.51 [0.47, 0.54] | 2.65 | +77 | 0.168 | 0.002 | 0% / 0% / 0% |
| 0.05 | 40% | 1 | **Pareto/NBD** | — | 39 | 0.60 [0.58, 0.62] | 1.63 | +15 | — | — | — |
|  |  | 2 | Transformer | `ar_bounded` | 65 | 0.55 [0.52, 0.58] | 1.91 | +53 | 0.138 | 0.002 | 30% / 20% / 0% |
|  |  | 3 | LSTM | `ar_bounded` | 56 | 0.54 [0.51, 0.57] | 1.88 | +52 | 0.138 | 0.002 | 10% / 0% / 0% |
| 0.05 | 60% | 1 | **Pareto/NBD** | — | 42 | 0.56 [0.55, 0.58] | 1.32 | +9 | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 67 | 0.52 [0.50, 0.54] | 1.43 | +60 | 0.100 | 0.002 | 10% / 10% / 0% |
|  |  | 3 | Transformer | `ar_bounded` | 77 | 0.52 [0.50, 0.53] | 1.47 | +66 | 0.100 | 0.002 | 20% / 0% / 0% |
| 0.05 | 80% | 1 | **Pareto/NBD** | — | 58 | 0.44 [0.41, 0.47] | 0.89 | +13 | — | — | — |
|  |  | 2 | Transformer | `ar_bounded` | 102 | 0.40 [0.37, 0.44] | 1.04 | +76 | 0.061 | 0.004 | 20% / 10% / 10% |
|  |  | 3 | LSTM | `ar_bounded` | 60 | 0.40 [0.36, 0.44] | 0.97 | +19 | 0.061 | 0.004 | 70% / 40% / 10% |
| 0.10 | 20% | 1 | **Pareto/NBD** | — | 31 | 0.72 [0.71, 0.74] | 2.91 | +3 | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 32 | 0.69 [0.68, 0.71] | 3.32 | +29 | 0.261 | 0.002 | 40% / 0% / 0% |
|  |  | 3 | Transformer | `ar_bounded` | 52 | 0.68 [0.66, 0.69] | 3.74 | +43 | 0.262 | 0.002 | 40% / 0% / 0% |
| 0.10 | 40% | 1 | **Pareto/NBD** | — | 32 | 0.73 [0.72, 0.74] | 2.47 | +1 | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 34 | 0.69 [0.68, 0.71] | 2.79 | +28 | 0.198 | 0.002 | 60% / 10% / 0% |
|  |  | 3 | Transformer | `ar_bounded` + `kmeans_8` | 58 | 0.65 [0.64, 0.67] | 3.42 | +51 | 0.192 | 0.002 | 10% / 0% / 0% |
| 0.10 | 60% | 1 | **Pareto/NBD** | — | 37 | 0.65 [0.64, 0.67] | 2.00 | +3 | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 33 | 0.63 [0.61, 0.64] | 2.12 | +20 | 0.140 | 0.002 | 70% / 10% / 0% |
|  |  | 3 | Transformer | `ar_bounded` | 115 | 0.61 [0.59, 0.63] | 2.74 | +104 | 0.140 | 0.002 | 10% / 0% / 0% |
| 0.10 | 80% | 1 | **Pareto/NBD** | — | 49 | 0.51 [0.49, 0.53] | 1.37 | +7 | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 45 | 0.47 [0.45, 0.49] | 1.45 | +4 | 0.077 | 0.002 | 90% / 80% / 0% |
|  |  | 3 | Transformer | `ar_bounded` | 102 | 0.47 [0.45, 0.49] | 1.57 | +82 | 0.078 | 0.002 | 30% / 20% / 0% |
| 0.30 | 20% | 1 | **Pareto/NBD** | — | 30 | 0.86 [0.85, 0.87] | 6.93 | −14 | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 12 | 0.85 [0.84, 0.86] | 6.49 | +10 | 0.493 | 0.002 | 100% / 80% / 0% |
|  |  | 3 | LSTM | `no_ar` | 19 | 0.85 [0.84, 0.86] | 6.92 | +17 | 0.497 | 0.002 | 100% / 20% / 0% |
| 0.30 | 40% | 1 | **Pareto/NBD** | — | 31 | 0.83 [0.82, 0.84] | 5.99 | −14 | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 13 | 0.82 [0.81, 0.83] | 5.66 | +2 | 0.355 | 0.002 | 100% / 90% / 0% |
|  |  | 3 | LSTM | `no_ar` | 14 | 0.81 [0.80, 0.82] | 5.74 | +7 | 0.361 | 0.002 | 100% / 80% / 0% |
| 0.30 | 60% | 1 | **Pareto/NBD** | — | 32 | 0.77 [0.76, 0.77] | 4.72 | −12 | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 19 | 0.70 [0.69, 0.72] | 4.60 | −7 | 0.232 | 0.002 | 100% / 60% / 0% |
|  |  | 3 | LSTM | `no_ar` | 18 | 0.70 [0.69, 0.72] | 4.73 | −1 | 0.237 | 0.002 | 100% / 70% / 0% |
| 0.30 | 80% | 1 | **Pareto/NBD** | — | 35 | 0.70 [0.68, 0.72] | 3.03 | −12 | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 35 | 0.51 [0.48, 0.53] | 3.01 | −13 | 0.117 | 0.002 | 50% / 40% / 0% |
|  |  | 3 | LSTM | `no_ar` | 32 | 0.51 [0.48, 0.53] | 3.26 | −15 | 0.121 | 0.002 | 70% / 30% / 0% |

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
- **Wins on each metric.** A neural tree beats Pareto/NBD on Spearman in 14 of 1,918
  panel comparisons: 12 on rate-0.01 panels, where both rank customers at only 0.1–0.3, and
  2 at rate 0.05, churn 80%, by less than 0.01. Wins
  on |bias| are rarer than wins on MAPE: at rate 0.30 LSTM `ar_bounded` beats Pareto/NBD's
  MAPE on 88% of panels but its |bias| on 68%, and at rate 0.10, churn 60%, on 70% against
  10%. A lower MAPE there comes from the weekly shape, not the yearly level.
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
| LSTM | unbounded | 3.38 / +505 / 513 | 11.88 / +582 / 582 | 12.71 / +315 / 322 | 16.35 / +131 / 141 |
| LSTM | bounded | 0.77 / +231 / 243 | 1.64 / +44 / 58 | 2.42 / +20 / 36 | 4.94 / −2 / 20 |
| Transformer | none | 0.67 / +74 / 120 | 1.99 / +109 / 114 | 3.18 / +126 / 130 | 6.47 / +43 / 56 |
| Transformer | unbounded | 0.75 / +89 / 150 | 1.95 / +95 / 105 | 3.79 / +148 / 149 | 8.92 / +89 / 92 |
| Transformer | bounded | 0.65 / +50 / 103 | 1.74 / +64 / 78 | 2.73 / +64 / 77 | 5.98 / +38 / 50 |

**By churn** (the 4 rates pooled; compare RMSE within a column only):

| Model | AR features | Churn 20% | 40% | 60% | 80% |
| --- | --- | --- | --- | --- | --- |
| Pareto/NBD | — | 3.14 / +2 / 36 | 2.69 / +9 / 42 | 2.15 / +10 / 50 | 1.43 / +16 / 75 |
| LSTM | none | 3.35 / +37 / 42 | 2.98 / +74 / 77 | 2.49 / +127 / 134 | 2.01 / +365 / 380 |
| LSTM | unbounded | 19.44 / +328 / 330 | 12.08 / +255 / 259 | 7.19 / +333 / 337 | 5.60 / +618 / 632 |
| LSTM | bounded | 3.24 / +32 / 39 | 2.78 / +49 / 56 | 2.22 / +66 / 79 | 1.54 / +147 / 183 |
| Transformer | none | 3.99 / +41 / 57 | 3.52 / +71 / 84 | 2.85 / +94 / 110 | 1.96 / +145 / 170 |
| Transformer | unbounded | 5.16 / +67 / 78 | 4.29 / +72 / 89 | 3.39 / +111 / 128 | 2.27 / +174 / 204 |
| Transformer | bounded | 3.80 / +41 / 55 | 3.09 / +32 / 56 | 2.65 / +70 / 89 | 1.56 / +73 / 109 |

**Tests: none → bounded** (claim 3 below breaks these down by churn and cohort size, with
intervals; Wilcoxon, paired on the same 40 panels per rate; median change in
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

Every claim below is tested per rate × churn cell, per rate with churn pooled, over the
whole grid and, where the data allow, at both cohort sizes. Most verdicts depend on those
conditions, so each claim states where it holds and where it does not. All numbers are
recomputed from the stored forecasts by `.scratch/synthetic-grid/metrics_by_cell.py` and
`claims.py`.

**Setting.** Seasonal panels (4 peaks, amplitude 1.5), 1,000 customers unless a table says
3,000. Four purchase rates × four churn rates (20, 40, 60 and 80% of customers dropped out
by week 52), 10 panels per cell, 40 per rate, 160 over the grid.

**How to read the tests.**

- **Effect Δ.** Every comparison is written "A → B" and Δ is B minus A. For MAPE, |bias| and
  RMSE a negative Δ means B is better; for Spearman, R_A and shape correlation a positive Δ
  does. MAPE and |bias| are in percentage points.
- **Paired comparisons** (two trees on the same panels): Δ is the Hodges–Lehmann estimate,
  the median of the averages of all pairs of per-panel differences, with its exact 95%
  interval. The test is Wilcoxon signed-rank. **Unpaired comparisons** (1,000 vs 3,000
  customers, which are different panels): the two-sample Hodges–Lehmann shift with its
  exact interval, and the Mann–Whitney U test.
- Δ is a median effect, so it can differ from the difference of the means shown as
  "A → B": a few panels with bias in the hundreds move the means far more than the median.
- **Bold** means significant at a 5% false discovery rate (Benjamini–Hochberg) within the
  claim's table. With 10 panels per cell the smallest possible Wilcoxon p is 0.002, so a
  single cell can only reach significance when all 10 panels move the same way.
- **Grids.** One grid per model and context: rows are purchase rates, columns churn levels,
  and each cell reads "Δ MAPE / Δ |bias| / Δ Spearman". The intervals and p-values behind
  every cell are in the collapsed tables under each grid.

### Summary

| # | Claim | Verdict | Holds | Does not hold |
| --- | --- | --- | --- | --- |
| 1 | Pareto/NBD beats the neural models. | **Partly** | Ranking (Spearman): every tree, every cell, both cohort sizes. Level (MAPE, \|bias\|): against the Transformer at every rate; against the LSTM at rates 0.01–0.05. | LSTM `ar_bounded` on MAPE at rate 0.30 (churn 20–60%) at 1,000 customers, and at rates 0.05–0.30 at 3,000. |
| 2 | Neural error rises with churn. | **Partly** | \|bias\| of the flagless LSTM at rates 0.01–0.05, both cohort sizes; MAPE of nearly every tree. | Not neural-specific for MAPE: Pareto/NBD's also rises (ρ +0.70 to +0.97). The flags remove the \|bias\| trend at rates 0.05–0.10. |
| 3 | Bounded AR flags help. | **Supported, conditionally** | LSTM: rates 0.01–0.10, from churn 40–60% up, by 34–439 MAPE points per significant cell; at 3,000 customers at every churn level on the sparsest panels. Transformer: rates 0.05–0.10. | Rate 0.30 (level unchanged, only RMSE improves); churn 20%; the Transformer at rate 0.01. |
| 4 | Unbounded counters hurt. | **Supported for the LSTM, partly for the Transformer** | LSTM: every rate, +98 to +260 MAPE points. Transformer: rate 0.30. | Transformer at rates 0.01–0.10 (one cell aside); ranking improves for both models at rate 0.05. |
| 5 | A k-means cluster label hurts. | **Partly** | LSTM level at rates ≥ 0.10 (≥ 0.05 with flags). | It raises Spearman on sparse panels, and it helps whenever the unbounded counters are present. |
| 6 | One architecture is better overall. | **Not supported** | — | The Transformer is better at rate 0.01 and with unbounded counters; the LSTM at rates 0.10–0.30 with `no_ar` or flags. |
| 7 | Neural models capture seasonality; Pareto/NBD cannot. | **Supported** | Shape correlation 0.11–0.96 (LSTM) against −0.09 to +0.13 (Pareto/NBD); the true season cuts Pareto/NBD's MAPE by 5–20 points in every cell. | — |
| 8 | Pareto/NBD's low bias means it is accurate per customer. | **Not supported** | — | In every cell it serves living customers only 51–87% of their volume and leaks 11–104% onto dead ones. |
| 9 | Neural models cannot detect a customer who has stopped. | **Partly** | Transformer everywhere; LSTM on rates 0.01–0.10. | LSTM `ar_bounded` at rate 0.30 with churn 60–80%, and at rate 0.10 with churn 80%: its leakage equals Pareto/NBD's. |
| 10 | A bigger hyperparameter search helps. | **Partly** | Rate 0.30 for both models; the Transformer's ranking at rates 0.05–0.30. | Level at rates 0.01–0.10. |
| 11 | Specific hyperparameters drive the error. | **Not supported** | — | No within-cell correlation survives the FDR correction. |
| 12 | More customers improve the neural forecast. | **Supported, conditionally** | LSTM `ar_bounded`: every cell at rates 0.01–0.10. LSTM `no_ar`: rates 0.05–0.30. | LSTM `no_ar` at rate 0.01; LSTM `ar_bounded` at rate 0.30 above churn 20%. The Transformer was not run at 3,000. |
| 13 | RMSE can rank these models. | **Not supported** for per-week RMSE; **supported** within one rate for customer-total RMSE | Customer-total RMSE ranks trees like MAPE (ρ +0.88 to +0.97 within each rate). | Per-week RMSE puts 4–10 of 13 trees on the same value to two decimals. |

### 1. Pareto/NBD beats the neural models

**Verdict: partly.** Pareto/NBD ranks customers better than every neural tree in every
cell, at both cohort sizes. On the level of the forecast it beats the Transformer
everywhere and the LSTM on sparse panels, but the LSTM with bounded flags beats it on MAPE
on dense panels, and at 3,000 customers from rate 0.05 up.

- **Ranking.** Δ Spearman is negative in 94 of the 96 cells (16 cells × six neural
  contexts), zero to two decimals in the other two, and significant in 95. The gap is largest at rate 0.01 (−0.11 to −0.34) and at rate
  0.30 with churn 80% (−0.19 to −0.26).
- **Level, LSTM `ar_bounded`.** Pareto/NBD is better at rate 0.01 (MAPE +11 to +394 points
  for the LSTM) and at rate 0.05 with churn 20–60%. At rate 0.10 MAPE is level except at
  churn 80% (LSTM better by 4), while |bias| favours Pareto/NBD at churn 20–60%. At rate
  0.30 with churn 20–60% the LSTM is better by 13–18 MAPE points and up to 8 |bias| points.
- **Level, Transformer.** Pareto/NBD is better at every rate pooled (MAPE +9 to +32 with
  flags, +17 to +79 without).
- **At 3,000 customers** the LSTM `ar_bounded` beats Pareto/NBD on MAPE at rates 0.05–0.30
  (−4 to −15 points rate-pooled) but still has the larger |bias| at rates 0.01–0.10.

Δ = neural minus Pareto/NBD: positive MAPE/|bias| or negative Spearman means Pareto/NBD
is better.

**LSTM `ar_bounded`, 1,000 customers**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **+11** / **+35** / **−0.25** | **+53** / **+79** / **−0.34** | **+97** / **+132** / **−0.31** | **+394** / **+481** / **−0.27** | **+81** / **+113** / **−0.29** |
| 0.05 | **+13** / **+39** / **−0.07** | **+17** / **+36** / **−0.06** | **+26** / **+54** / **−0.04** | 0 / +4 / **−0.04** | **+14** / **+33** / **−0.05** |
| 0.10 | +1 / **+28** / **−0.03** | +2 / **+22** / **−0.04** | −4 / **+14** / **−0.03** | **−4** / −6 / **−0.04** | −3 / **+14** / **−0.03** |
| 0.30 | **−18** / **−4** / **−0.01** | **−18** / **−8** / **−0.01** | **−13** / −2 / **−0.06** | +1 / +6 / **−0.19** | **−13** / **−3** / **−0.06** |
| all |  |  |  |  | **+9** / **+26** / **−0.09** |

**LSTM `no_ar`, 1,000 customers**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **+21** / **+45** / **−0.20** | **+66** / **+94** / **−0.24** | **+199** / **+248** / **−0.24** | **+557** / **+660** / **−0.22** | **+147** / **+187** / **−0.23** |
| 0.05 | +10 / **+33** / **−0.32** | **+61** / **+83** / **−0.38** | **+115** / **+144** / **−0.52** | **+439** / **+477** / **−0.44** | **+105** / **+133** / **−0.46** |
| 0.10 | +1 / **+28** / **−0.05** | **+15** / **+40** / **−0.05** | **+31** / **+50** / **−0.05** | **+148** / **+175** / **−0.24** | **+25** / **+49** / **−0.06** |
| 0.30 | **−10** / +5 / **−0.01** | **−16** / −5 / **−0.02** | **−14** / −4 / **−0.06** | −3 / +2 / **−0.19** | **−11** / −1 / **−0.06** |
| all |  |  |  |  | **+55** / **+79** / **−0.19** |

**Transformer `ar_bounded`, 1,000 customers**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | +9 / **+27** / **−0.19** | +10 / **+23** / **−0.28** | +7 / +21 / **−0.20** | +7 / +26 / **−0.11** | **+9** / **+25** / **−0.20** |
| 0.05 | **+36** / **+63** / **−0.12** | +12 / **+27** / **−0.05** | **+31** / **+56** / **−0.05** | **+31** / **+52** / **−0.03** | **+26** / **+48** / **−0.06** |
| 0.10 | +22 / **+42** / **−0.05** | +10 / **+28** / **−0.04** | **+70** / **+101** / **−0.05** | **+47** / **+66** / **−0.04** | **+32** / **+53** / **−0.04** |
| 0.30 | +5 / +10 / **−0.05** | +8 / +20 / **−0.06** | **+34** / **+49** / **−0.10** | **+20** / **+34** / **−0.21** | **+15** / **+27** / **−0.12** |
| all |  |  |  |  | **+20** / **+38** / **−0.09** |

**Transformer `no_ar`, 1,000 customers**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | +7 / +25 / **−0.17** | +28 / +44 / **−0.25** | +33 / +57 / **−0.20** | **+32** / +67 / **−0.12** | **+23** / **+45** / **−0.18** |
| 0.05 | **+40** / **+66** / **−0.28** | **+73** / **+97** / **−0.19** | **+103** / **+129** / **−0.20** | **+62** / **+88** / **−0.12** | **+64** / **+90** / **−0.17** |
| 0.10 | **+26** / **+53** / **−0.14** | **+60** / **+85** / **−0.15** | **+59** / **+78** / **−0.13** | **+209** / **+237** / **−0.10** | **+79** / **+104** / **−0.13** |
| 0.30 | −2 / +3 / **−0.14** | +4 / **+13** / **−0.16** | +20 / **+36** / **−0.19** | **+52** / **+63** / **−0.26** | **+17** / **+27** / **−0.18** |
| all |  |  |  |  | **+42** / **+64** / **−0.16** |

**LSTM `ar_bounded`, 3,000 customers**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | +1 / **+15** / **−0.18** | +5 / **+17** / **−0.20** | **+28** / **+41** / **−0.21** | **+59** / **+95** / **−0.19** | **+14** / **+29** / **−0.19** |
| 0.05 | −5 / **+17** / **−0.04** | −4 / **+15** / **−0.05** | −5 / +11 / **−0.04** | −2 / +8 / **−0.04** | **−4** / **+13** / **−0.04** |
| 0.10 | **−17** / **+9** / **−0.02** | **−19** / **+3** / **−0.01** | **−15** / **+4** / **−0.02** | −4 / +7 / **−0.04** | **−14** / **+6** / **−0.02** |
| 0.30 | **−24** / **−12** / 0 | **−20** / **−7** / **−0.01** | **−12** / 0 / **−0.07** | −1 / **+15** / **−0.19** | **−15** / −2 / **−0.05** |
| all |  |  |  |  | **−6** / **+9** / **−0.07** |

**LSTM `no_ar`, 3,000 customers**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **+47** / **+73** / **−0.22** | **+83** / **+106** / **−0.30** | **+198** / **+238** / **−0.28** | **+319** / **+379** / **−0.27** | **+155** / **+189** / **−0.27** |
| 0.05 | −6 / **+17** / **−0.07** | **+37** / **+60** / **−0.13** | **+80** / **+105** / **−0.24** | **+252** / **+282** / **−0.42** | **+57** / **+81** / **−0.16** |
| 0.10 | **−12** / **+15** / **−0.04** | −2 / **+27** / **−0.04** | **−15** / **+5** / **−0.02** | −3 / **+15** / **−0.04** | **−8** / **+15** / **−0.03** |
| 0.30 | **−23** / **−9** / **0** | **−22** / **−10** / **−0.01** | **−18** / **−6** / **−0.07** | −3 / **+12** / **−0.19** | **−18** / **−5** / **−0.05** |
| all |  |  |  |  | **+28** / **+49** / **−0.13** |

<details><summary>Intervals and p-values by rate, churn pooled</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI], p | Δ \|bias\| [95% CI], p | Δ RMSE [95% CI], p | Δ Spearman [95% CI], p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM `ar_bounded`, 1,000 customers | 0.01 | pooled | 40 | 90 → 243 | 35 → 232 | **+81 [+52, +134], p <10⁻⁴** | **+113 [+81, +185], p <10⁻⁴** | **+0.12 [+0.09, +0.17], p <10⁻⁴** | **−0.29 [−0.33, −0.25], p <10⁻⁴** |
| LSTM `ar_bounded`, 1,000 customers | 0.05 | pooled | 40 | 44 → 58 | 14 → 46 | **+14 [+9, +18], p <10⁻⁴** | **+33 [+25, +42], p <10⁻⁴** | **+0.20 [+0.15, +0.25], p <10⁻⁴** | **−0.05 [−0.06, −0.04], p <10⁻⁴** |
| LSTM `ar_bounded`, 1,000 customers | 0.10 | pooled | 40 | 37 → 36 | 9 → 24 | −3 [−5, +1], p 0.109 | **+14 [+8, +21], p <10⁻⁴** | **+0.20 [+0.15, +0.27], p <10⁻⁴** | **−0.03 [−0.04, −0.03], p <10⁻⁴** |
| LSTM `ar_bounded`, 1,000 customers | 0.30 | pooled | 40 | 32 → 20 | 13 → 11 | **−13 [−16, −10], p <10⁻⁴** | **−3 [−6, 0], p 0.037** | **−0.21 [−0.30, −0.12], p <10⁻⁴** | **−0.06 [−0.10, −0.03], p <10⁻⁴** |
| LSTM `ar_bounded`, 1,000 customers | all | pooled | 160 | 51 → 89 | 18 → 78 | **+9 [+4, +14], p 0.0003** | **+26 [+20, +34], p <10⁻⁴** | **+0.11 [+0.08, +0.14], p <10⁻⁴** | **−0.09 [−0.12, −0.06], p <10⁻⁴** |
| LSTM `no_ar`, 1,000 customers | 0.01 | pooled | 40 | 90 → 314 | 35 → 309 | **+147 [+105, +247], p <10⁻⁴** | **+187 [+142, +307], p <10⁻⁴** | **+0.20 [+0.15, +0.26], p <10⁻⁴** | **−0.23 [−0.25, −0.20], p <10⁻⁴** |
| LSTM `no_ar`, 1,000 customers | 0.05 | pooled | 40 | 44 → 200 | 14 → 199 | **+105 [+67, +223], p <10⁻⁴** | **+133 [+92, +255], p <10⁻⁴** | **+0.65 [+0.57, +0.73], p <10⁻⁴** | **−0.46 [−0.52, −0.33], p <10⁻⁴** |
| LSTM `no_ar`, 1,000 customers | 0.10 | pooled | 40 | 37 → 98 | 9 → 93 | **+25 [+9, +71], p <10⁻⁴** | **+49 [+33, +92], p <10⁻⁴** | **+0.44 [+0.33, +0.58], p <10⁻⁴** | **−0.06 [−0.09, −0.05], p <10⁻⁴** |
| LSTM `no_ar`, 1,000 customers | 0.30 | pooled | 40 | 32 → 21 | 13 → 13 | **−11 [−13, −9], p <10⁻⁴** | −1 [−4, +2], p 0.675 | 0 [−0.10, +0.10], p 0.995 | **−0.06 [−0.10, −0.03], p <10⁻⁴** |
| LSTM `no_ar`, 1,000 customers | all | pooled | 160 | 51 → 158 | 18 → 154 | **+55 [+36, +81], p <10⁻⁴** | **+79 [+58, +109], p <10⁻⁴** | **+0.33 [+0.27, +0.39], p <10⁻⁴** | **−0.19 [−0.23, −0.15], p <10⁻⁴** |
| Transformer `ar_bounded`, 1,000 customers | 0.01 | pooled | 40 | 90 → 103 | 35 → 63 | **+9 [+1, +19], p 0.028** | **+25 [+10, +39], p 0.001** | **+0.03 [+0.02, +0.04], p <10⁻⁴** | **−0.20 [−0.23, −0.16], p <10⁻⁴** |
| Transformer `ar_bounded`, 1,000 customers | 0.05 | pooled | 40 | 44 → 78 | 14 → 68 | **+26 [+16, +45], p <10⁻⁴** | **+48 [+32, +66], p <10⁻⁴** | **+0.24 [+0.15, +0.43], p <10⁻⁴** | **−0.06 [−0.07, −0.04], p <10⁻⁴** |
| Transformer `ar_bounded`, 1,000 customers | 0.10 | pooled | 40 | 37 → 77 | 9 → 70 | **+32 [+17, +50], p <10⁻⁴** | **+53 [+37, +72], p <10⁻⁴** | **+0.46 [+0.36, +0.62], p <10⁻⁴** | **−0.04 [−0.05, −0.04], p <10⁻⁴** |
| Transformer `ar_bounded`, 1,000 customers | 0.30 | pooled | 40 | 32 → 50 | 13 → 42 | **+15 [+8, +23], p 0.0004** | **+27 [+16, +40], p <10⁻⁴** | **+0.59 [+0.29, +1.12], p <10⁻⁴** | **−0.12 [−0.14, −0.08], p <10⁻⁴** |
| Transformer `ar_bounded`, 1,000 customers | all | pooled | 160 | 51 → 77 | 18 → 61 | **+20 [+15, +26], p <10⁻⁴** | **+38 [+31, +46], p <10⁻⁴** | **+0.28 [+0.20, +0.37], p <10⁻⁴** | **−0.09 [−0.12, −0.08], p <10⁻⁴** |
| Transformer `no_ar`, 1,000 customers | 0.01 | pooled | 40 | 90 → 120 | 35 → 84 | **+23 [+7, +38], p 0.001** | **+45 [+20, +69], p 0.0007** | **+0.04 [+0.02, +0.07], p <10⁻⁴** | **−0.18 [−0.21, −0.15], p <10⁻⁴** |
| Transformer `no_ar`, 1,000 customers | 0.05 | pooled | 40 | 44 → 114 | 14 → 109 | **+64 [+45, +89], p <10⁻⁴** | **+90 [+68, +114], p <10⁻⁴** | **+0.54 [+0.44, +0.64], p <10⁻⁴** | **−0.17 [−0.20, −0.15], p <10⁻⁴** |
| Transformer `no_ar`, 1,000 customers | 0.10 | pooled | 40 | 37 → 130 | 9 → 128 | **+79 [+51, +115], p <10⁻⁴** | **+104 [+75, +139], p <10⁻⁴** | **+0.99 [+0.84, +1.14], p <10⁻⁴** | **−0.13 [−0.14, −0.12], p <10⁻⁴** |
| Transformer `no_ar`, 1,000 customers | 0.30 | pooled | 40 | 32 → 56 | 13 → 48 | **+17 [+3, +32], p 0.003** | **+27 [+12, +44], p <10⁻⁴** | **+1.09 [+0.91, +1.38], p <10⁻⁴** | **−0.18 [−0.20, −0.16], p <10⁻⁴** |
| Transformer `no_ar`, 1,000 customers | all | pooled | 160 | 51 → 105 | 18 → 92 | **+42 [+33, +53], p <10⁻⁴** | **+64 [+53, +78], p <10⁻⁴** | **+0.64 [+0.54, +0.74], p <10⁻⁴** | **−0.16 [−0.18, −0.15], p <10⁻⁴** |
| LSTM `ar_bounded`, 3,000 customers | 0.01 | pooled | 40 | 56 → 83 | 21 → 65 | **+14 [+6, +29], p <10⁻⁴** | **+29 [+18, +49], p <10⁻⁴** | **+0.03 [+0.03, +0.04], p <10⁻⁴** | **−0.19 [−0.22, −0.17], p <10⁻⁴** |
| LSTM `ar_bounded`, 3,000 customers | 0.05 | pooled | 40 | 34 → 31 | 8 → 22 | **−4 [−7, 0], p 0.045** | **+13 [+8, +18], p <10⁻⁴** | **+0.09 [+0.07, +0.10], p <10⁻⁴** | **−0.04 [−0.05, −0.04], p <10⁻⁴** |
| LSTM `ar_bounded`, 3,000 customers | 0.10 | pooled | 40 | 32 → 18 | 4 → 9 | **−14 [−17, −11], p <10⁻⁴** | **+6 [+3, +8], p <10⁻⁴** | **+0.07 [+0.06, +0.09], p <10⁻⁴** | **−0.02 [−0.03, −0.02], p <10⁻⁴** |
| LSTM `ar_bounded`, 3,000 customers | 0.30 | pooled | 40 | 30 → 16 | 13 → 12 | **−15 [−18, −11], p <10⁻⁴** | −2 [−7, +2], p 0.320 | **−0.33 [−0.47, −0.19], p <10⁻⁴** | **−0.05 [−0.09, −0.03], p <10⁻⁴** |
| LSTM `ar_bounded`, 3,000 customers | all | pooled | 160 | 38 → 37 | 12 → 27 | **−6 [−8, −4], p <10⁻⁴** | **+9 [+7, +12], p <10⁻⁴** | **+0.04 [+0.03, +0.05], p 0.0001** | **−0.07 [−0.09, −0.05], p <10⁻⁴** |
| LSTM `no_ar`, 3,000 customers | 0.01 | pooled | 40 | 56 → 219 | 21 → 219 | **+155 [+111, +197], p <10⁻⁴** | **+189 [+140, +235], p <10⁻⁴** | **+0.17 [+0.15, +0.20], p <10⁻⁴** | **−0.27 [−0.28, −0.25], p <10⁻⁴** |
| LSTM `no_ar`, 3,000 customers | 0.05 | pooled | 40 | 34 → 119 | 8 → 118 | **+57 [+28, +120], p <10⁻⁴** | **+81 [+49, +146], p <10⁻⁴** | **+0.34 [+0.27, +0.41], p <10⁻⁴** | **−0.16 [−0.27, −0.10], p <10⁻⁴** |
| LSTM `no_ar`, 3,000 customers | 0.10 | pooled | 40 | 32 → 24 | 4 → 19 | **−8 [−12, −6], p <10⁻⁴** | **+15 [+11, +18], p <10⁻⁴** | **+0.18 [+0.16, +0.20], p <10⁻⁴** | **−0.03 [−0.04, −0.03], p <10⁻⁴** |
| LSTM `no_ar`, 3,000 customers | 0.30 | pooled | 40 | 30 → 14 | 13 → 10 | **−18 [−20, −14], p <10⁻⁴** | **−5 [−8, −1], p 0.037** | **−0.29 [−0.41, −0.17], p <10⁻⁴** | **−0.05 [−0.10, −0.03], p <10⁻⁴** |
| LSTM `no_ar`, 3,000 customers | all | pooled | 160 | 38 → 94 | 12 → 91 | **+28 [+13, +44], p 0.0004** | **+49 [+34, +66], p <10⁻⁴** | **+0.15 [+0.12, +0.18], p <10⁻⁴** | **−0.13 [−0.15, −0.11], p <10⁻⁴** |

</details>

<details><summary>Intervals and p-values per rate × churn cell</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI], p | Δ \|bias\| [95% CI], p | Δ RMSE [95% CI], p | Δ Spearman [95% CI], p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM `ar_bounded`, 1,000 customers | 0.01 | 20% | 10 | 48 → 65 | 12 → 47 | **+11 [+2, +39], p 0.037** | **+35 [+11, +55], p 0.010** | **+0.06 [+0.03, +0.12], p 0.002** | **−0.25 [−0.34, −0.17], p 0.002** |
| LSTM `ar_bounded`, 1,000 customers | 0.01 | 40% | 10 | 66 → 121 | 34 → 114 | **+53 [+38, +71], p 0.002** | **+79 [+63, +95], p 0.002** | **+0.10 [+0.07, +0.14], p 0.002** | **−0.34 [−0.40, −0.28], p 0.002** |
| LSTM `ar_bounded`, 1,000 customers | 0.01 | 60% | 10 | 88 → 196 | 41 → 190 | **+97 [+63, +162], p 0.002** | **+132 [+93, +213], p 0.002** | **+0.13 [+0.09, +0.21], p 0.004** | **−0.31 [−0.40, −0.20], p 0.002** |
| LSTM `ar_bounded`, 1,000 customers | 0.01 | 80% | 10 | 159 → 591 | 54 → 576 | **+394 [+138, +688], p 0.002** | **+481 [+199, +796], p 0.002** | **+0.29 [+0.15, +0.42], p 0.002** | **−0.27 [−0.33, −0.20], p 0.002** |
| LSTM `ar_bounded`, 1,000 customers | 0.05 | 20% | 10 | 35 → 48 | 8 → 47 | **+13 [+4, +21], p 0.004** | **+39 [+30, +47], p 0.002** | **+0.36 [+0.29, +0.44], p 0.002** | **−0.07 [−0.11, −0.04], p 0.002** |
| LSTM `ar_bounded`, 1,000 customers | 0.05 | 40% | 10 | 39 → 56 | 15 → 52 | **+17 [+8, +26], p 0.004** | **+36 [+26, +47], p 0.002** | **+0.25 [+0.21, +0.29], p 0.002** | **−0.06 [−0.08, −0.04], p 0.002** |
| LSTM `ar_bounded`, 1,000 customers | 0.05 | 60% | 10 | 42 → 67 | 12 → 61 | **+26 [+12, +36], p 0.006** | **+54 [+25, +63], p 0.004** | **+0.12 [+0.05, +0.17], p 0.006** | **−0.04 [−0.06, −0.03], p 0.002** |
| LSTM `ar_bounded`, 1,000 customers | 0.05 | 80% | 10 | 58 → 60 | 21 → 27 | 0 [−2, +11], p 0.922 | +4 [−6, +17], p 0.492 | **+0.08 [+0.04, +0.12], p 0.002** | **−0.04 [−0.06, −0.02], p 0.004** |
| LSTM `ar_bounded`, 1,000 customers | 0.10 | 20% | 10 | 31 → 32 | 3 → 30 | +1 [−9, +14], p 0.770 | **+28 [+15, +40], p 0.002** | **+0.38 [+0.27, +0.55], p 0.002** | **−0.03 [−0.04, −0.02], p 0.002** |
| LSTM `ar_bounded`, 1,000 customers | 0.10 | 40% | 10 | 32 → 34 | 5 → 28 | +2 [−7, +11], p 0.770 | **+22 [+11, +35], p 0.004** | **+0.29 [+0.19, +0.47], p 0.002** | **−0.04 [−0.05, −0.02], p 0.002** |
| LSTM `ar_bounded`, 1,000 customers | 0.10 | 60% | 10 | 37 → 33 | 9 → 21 | −4 [−8, 0], p 0.049 | **+14 [+5, +18], p 0.006** | **+0.11 [+0.07, +0.15], p 0.002** | **−0.03 [−0.04, −0.01], p 0.002** |
| LSTM `ar_bounded`, 1,000 customers | 0.10 | 80% | 10 | 49 → 45 | 21 → 15 | **−4 [−9, −1], p 0.037** | −6 [−16, +7], p 0.232 | **+0.08 [+0.02, +0.14], p 0.014** | **−0.04 [−0.06, −0.03], p 0.002** |
| LSTM `ar_bounded`, 1,000 customers | 0.30 | 20% | 10 | 30 → 12 | 14 → 10 | **−18 [−21, −16], p 0.002** | **−4 [−9, −1], p 0.037** | **−0.47 [−0.70, −0.15], p 0.010** | **−0.01 [−0.01, 0], p 0.002** |
| LSTM `ar_bounded`, 1,000 customers | 0.30 | 40% | 10 | 31 → 13 | 14 → 6 | **−18 [−20, −16], p 0.002** | **−8 [−12, −4], p 0.004** | **−0.35 [−0.43, −0.19], p 0.002** | **−0.01 [−0.02, −0.01], p 0.002** |
| LSTM `ar_bounded`, 1,000 customers | 0.30 | 60% | 10 | 32 → 19 | 13 → 10 | **−13 [−16, −9], p 0.002** | −2 [−9, +4], p 0.625 | −0.13 [−0.21, −0.01], p 0.049 | **−0.06 [−0.08, −0.05], p 0.002** |
| LSTM `ar_bounded`, 1,000 customers | 0.30 | 80% | 10 | 35 → 35 | 13 → 19 | +1 [−2, +3], p 0.922 | +6 [−4, +18], p 0.432 | −0.01 [−0.11, +0.09], p 0.625 | **−0.19 [−0.21, −0.18], p 0.002** |
| LSTM `no_ar`, 1,000 customers | 0.01 | 20% | 10 | 48 → 71 | 12 → 59 | **+21 [+13, +37], p 0.002** | **+45 [+34, +61], p 0.002** | **+0.07 [+0.05, +0.10], p 0.002** | **−0.20 [−0.26, −0.15], p 0.002** |
| LSTM `no_ar`, 1,000 customers | 0.01 | 40% | 10 | 66 → 136 | 34 → 132 | **+66 [+52, +89], p 0.002** | **+94 [+83, +114], p 0.002** | **+0.12 [+0.10, +0.15], p 0.002** | **−0.24 [−0.28, −0.20], p 0.002** |
| LSTM `no_ar`, 1,000 customers | 0.01 | 60% | 10 | 88 → 291 | 41 → 290 | **+199 [+177, +230], p 0.002** | **+248 [+220, +279], p 0.002** | **+0.26 [+0.22, +0.31], p 0.002** | **−0.24 [−0.32, −0.20], p 0.002** |
| LSTM `no_ar`, 1,000 customers | 0.01 | 80% | 10 | 159 → 759 | 54 → 755 | **+557 [+343, +866], p 0.002** | **+660 [+437, +979], p 0.002** | **+0.44 [+0.30, +0.57], p 0.002** | **−0.22 [−0.28, −0.16], p 0.002** |
| LSTM `no_ar`, 1,000 customers | 0.05 | 20% | 10 | 35 → 45 | 8 → 42 | +10 [+1, +17], p 0.049 | **+33 [+22, +42], p 0.002** | **+0.40 [+0.22, +0.58], p 0.002** | **−0.32 [−0.55, −0.08], p 0.002** |
| LSTM `no_ar`, 1,000 customers | 0.05 | 40% | 10 | 39 → 99 | 15 → 98 | **+61 [+46, +74], p 0.002** | **+83 [+66, +99], p 0.002** | **+0.63 [+0.50, +0.72], p 0.002** | **−0.38 [−0.61, −0.12], p 0.002** |
| LSTM `no_ar`, 1,000 customers | 0.05 | 60% | 10 | 42 → 158 | 12 → 157 | **+115 [+92, +138], p 0.002** | **+144 [+123, +168], p 0.002** | **+0.64 [+0.55, +0.73], p 0.002** | **−0.52 [−0.57, −0.51], p 0.002** |
| LSTM `no_ar`, 1,000 customers | 0.05 | 80% | 10 | 58 → 497 | 21 → 497 | **+439 [+376, +502], p 0.002** | **+477 [+414, +542], p 0.002** | **+1.00 [+0.81, +1.17], p 0.002** | **−0.44 [−0.49, −0.40], p 0.002** |
| LSTM `no_ar`, 1,000 customers | 0.10 | 20% | 10 | 31 → 33 | 3 → 31 | +1 [−6, +12], p 0.770 | **+28 [+20, +39], p 0.002** | **+0.37 [+0.24, +0.50], p 0.002** | **−0.05 [−0.06, −0.04], p 0.002** |
| LSTM `no_ar`, 1,000 customers | 0.10 | 40% | 10 | 32 → 58 | 5 → 57 | **+15 [+3, +75], p 0.020** | **+40 [+26, +101], p 0.002** | **+0.39 [+0.24, +1.45], p 0.002** | **−0.05 [−0.19, −0.04], p 0.002** |
| LSTM `no_ar`, 1,000 customers | 0.10 | 60% | 10 | 37 → 69 | 9 → 61 | **+31 [+2, +61], p 0.037** | **+50 [+16, +89], p 0.010** | **+0.41 [+0.27, +0.62], p 0.002** | **−0.05 [−0.08, −0.03], p 0.002** |
| LSTM `no_ar`, 1,000 customers | 0.10 | 80% | 10 | 49 → 233 | 21 → 225 | **+148 [+49, +320], p 0.010** | **+175 [+74, +324], p 0.010** | **+0.66 [+0.18, +1.02], p 0.002** | **−0.24 [−0.41, −0.06], p 0.002** |
| LSTM `no_ar`, 1,000 customers | 0.30 | 20% | 10 | 30 → 19 | 14 → 17 | **−10 [−15, −7], p 0.002** | +5 [−4, +9], p 0.432 | +0.05 [−0.31, +0.23], p 0.625 | **−0.01 [−0.01, 0], p 0.002** |
| LSTM `no_ar`, 1,000 customers | 0.30 | 40% | 10 | 31 → 14 | 14 → 9 | **−16 [−19, −13], p 0.002** | −5 [−10, +1], p 0.084 | **−0.25 [−0.36, −0.12], p 0.006** | **−0.02 [−0.02, −0.01], p 0.002** |
| LSTM `no_ar`, 1,000 customers | 0.30 | 60% | 10 | 32 → 18 | 13 → 9 | **−14 [−17, −11], p 0.002** | −4 [−10, +1], p 0.131 | −0.01 [−0.16, +0.21], p 0.922 | **−0.06 [−0.08, −0.05], p 0.002** |
| LSTM `no_ar`, 1,000 customers | 0.30 | 80% | 10 | 35 → 32 | 13 → 17 | −3 [−7, +1], p 0.131 | +2 [−3, +11], p 0.232 | **+0.23 [+0.08, +0.41], p 0.014** | **−0.19 [−0.21, −0.18], p 0.002** |
| Transformer `ar_bounded`, 1,000 customers | 0.01 | 20% | 10 | 48 → 62 | 12 → 44 | +9 [0, +36], p 0.064 | **+27 [+10, +57], p 0.010** | **+0.05 [+0.02, +0.10], p 0.002** | **−0.19 [−0.25, −0.14], p 0.002** |
| Transformer `ar_bounded`, 1,000 customers | 0.01 | 40% | 10 | 66 → 75 | 34 → 58 | +10 [−4, +24], p 0.193 | **+23 [+4, +42], p 0.020** | **+0.03 [+0.02, +0.05], p 0.002** | **−0.28 [−0.31, −0.24], p 0.002** |
| Transformer `ar_bounded`, 1,000 customers | 0.01 | 60% | 10 | 88 → 98 | 41 → 60 | +7 [−13, +35], p 0.492 | +21 [−21, +58], p 0.275 | +0.03 [0, +0.05], p 0.064 | **−0.20 [−0.26, −0.15], p 0.002** |
| Transformer `ar_bounded`, 1,000 customers | 0.01 | 80% | 10 | 159 → 176 | 54 → 90 | +7 [−20, +59], p 0.492 | +26 [−34, +107], p 0.375 | +0.01 [−0.02, +0.05], p 0.492 | **−0.11 [−0.22, −0.03], p 0.014** |
| Transformer `ar_bounded`, 1,000 customers | 0.05 | 20% | 10 | 35 → 68 | 8 → 63 | **+36 [+18, +50], p 0.010** | **+63 [+30, +76], p 0.006** | **+0.66 [+0.43, +0.82], p 0.002** | **−0.12 [−0.16, −0.09], p 0.002** |
| Transformer `ar_bounded`, 1,000 customers | 0.05 | 40% | 10 | 39 → 65 | 15 → 54 | +12 [−4, +60], p 0.105 | **+27 [+1, +82], p 0.037** | **+0.17 [+0.08, +0.52], p 0.002** | **−0.05 [−0.07, −0.03], p 0.002** |
| Transformer `ar_bounded`, 1,000 customers | 0.05 | 60% | 10 | 42 → 77 | 12 → 69 | **+31 [+10, +65], p 0.014** | **+56 [+26, +89], p 0.002** | **+0.13 [+0.08, +0.25], p 0.002** | **−0.05 [−0.06, −0.03], p 0.002** |
| Transformer `ar_bounded`, 1,000 customers | 0.05 | 80% | 10 | 58 → 102 | 21 → 84 | **+31 [+2, +90], p 0.027** | **+52 [+11, +117], p 0.014** | **+0.14 [+0.06, +0.22], p 0.002** | **−0.03 [−0.05, −0.02], p 0.004** |
| Transformer `ar_bounded`, 1,000 customers | 0.10 | 20% | 10 | 31 → 52 | 3 → 45 | +22 [−3, +40], p 0.084 | **+42 [+12, +67], p 0.002** | **+0.80 [+0.57, +1.07], p 0.002** | **−0.05 [−0.06, −0.04], p 0.002** |
| Transformer `ar_bounded`, 1,000 customers | 0.10 | 40% | 10 | 32 → 41 | 5 → 33 | +10 [−3, +20], p 0.160 | **+28 [+10, +47], p 0.002** | **+0.36 [+0.18, +0.63], p 0.002** | **−0.04 [−0.28, −0.03], p 0.002** |
| Transformer `ar_bounded`, 1,000 customers | 0.10 | 60% | 10 | 37 → 115 | 9 → 113 | **+70 [+33, +118], p 0.004** | **+101 [+60, +144], p 0.002** | **+0.56 [+0.34, +1.23], p 0.002** | **−0.05 [−0.07, −0.03], p 0.002** |
| Transformer `ar_bounded`, 1,000 customers | 0.10 | 80% | 10 | 49 → 102 | 21 → 90 | **+47 [+4, +92], p 0.027** | **+66 [+19, +116], p 0.010** | **+0.21 [+0.07, +0.30], p 0.010** | **−0.04 [−0.05, −0.03], p 0.002** |
| Transformer `ar_bounded`, 1,000 customers | 0.30 | 20% | 10 | 30 → 37 | 14 → 27 | +5 [−7, +18], p 0.625 | +10 [−10, +31], p 0.432 | +0.89 [−0.18, +2.51], p 0.131 | **−0.05 [−0.07, −0.04], p 0.002** |
| Transformer `ar_bounded`, 1,000 customers | 0.30 | 40% | 10 | 31 → 42 | 14 → 35 | +8 [−3, +24], p 0.322 | +20 [+1, +39], p 0.049 | +0.80 [+0.01, +1.85], p 0.049 | **−0.06 [−0.23, −0.04], p 0.002** |
| Transformer `ar_bounded`, 1,000 customers | 0.30 | 60% | 10 | 32 → 67 | 13 → 62 | **+34 [+2, +62], p 0.027** | **+49 [+14, +81], p 0.010** | **+1.09 [+0.34, +1.79], p 0.006** | **−0.10 [−0.12, −0.08], p 0.002** |
| Transformer `ar_bounded`, 1,000 customers | 0.30 | 80% | 10 | 35 → 54 | 13 → 46 | **+20 [+6, +33], p 0.010** | **+34 [+8, +55], p 0.010** | +0.16 [−0.08, +0.35], p 0.084 | **−0.21 [−0.24, −0.20], p 0.002** |
| Transformer `no_ar`, 1,000 customers | 0.01 | 20% | 10 | 48 → 58 | 12 → 37 | +7 [−1, +23], p 0.105 | +25 [−1, +49], p 0.064 | **+0.04 [+0.02, +0.08], p 0.004** | **−0.17 [−0.22, −0.12], p 0.002** |
| Transformer `no_ar`, 1,000 customers | 0.01 | 40% | 10 | 66 → 96 | 34 → 76 | +28 [−9, +76], p 0.557 | +44 [−11, +91], p 0.557 | **+0.06 [+0.01, +0.14], p 0.004** | **−0.25 [−0.30, −0.17], p 0.002** |
| Transformer `no_ar`, 1,000 customers | 0.01 | 60% | 10 | 88 → 129 | 41 → 102 | +33 [−3, +86], p 0.084 | +57 [−3, +118], p 0.064 | **+0.06 [+0.01, +0.11], p 0.014** | **−0.20 [−0.24, −0.15], p 0.002** |
| Transformer `no_ar`, 1,000 customers | 0.01 | 80% | 10 | 159 → 197 | 54 → 121 | **+32 [+3, +71], p 0.037** | +67 [+2, +131], p 0.049 | +0.02 [−0.01, +0.05], p 0.232 | **−0.12 [−0.19, −0.08], p 0.002** |
| Transformer `no_ar`, 1,000 customers | 0.05 | 20% | 10 | 35 → 75 | 8 → 72 | **+40 [+22, +55], p 0.004** | **+66 [+41, +83], p 0.004** | **+0.77 [+0.55, +1.01], p 0.002** | **−0.28 [−0.41, −0.14], p 0.002** |
| Transformer `no_ar`, 1,000 customers | 0.05 | 40% | 10 | 39 → 110 | 15 → 108 | **+73 [+42, +102], p 0.002** | **+97 [+63, +124], p 0.002** | **+0.67 [+0.42, +0.88], p 0.002** | **−0.19 [−0.23, −0.13], p 0.002** |
| Transformer `no_ar`, 1,000 customers | 0.05 | 60% | 10 | 42 → 149 | 12 → 144 | **+103 [+19, +189], p 0.004** | **+129 [+40, +221], p 0.004** | **+0.54 [+0.29, +0.86], p 0.002** | **−0.20 [−0.25, −0.14], p 0.002** |
| Transformer `no_ar`, 1,000 customers | 0.05 | 80% | 10 | 58 → 122 | 21 → 111 | **+62 [+21, +106], p 0.010** | **+88 [+40, +147], p 0.010** | **+0.24 [+0.17, +0.30], p 0.002** | **−0.12 [−0.16, −0.09], p 0.002** |
| Transformer `no_ar`, 1,000 customers | 0.10 | 20% | 10 | 31 → 58 | 3 → 56 | **+26 [+9, +44], p 0.010** | **+53 [+34, +73], p 0.002** | **+1.26 [+0.86, +1.49], p 0.002** | **−0.14 [−0.17, −0.11], p 0.002** |
| Transformer `no_ar`, 1,000 customers | 0.10 | 40% | 10 | 32 → 94 | 5 → 91 | **+60 [+26, +96], p 0.004** | **+85 [+47, +122], p 0.002** | **+1.22 [+0.84, +1.60], p 0.002** | **−0.15 [−0.16, −0.14], p 0.002** |
| Transformer `no_ar`, 1,000 customers | 0.10 | 60% | 10 | 37 → 103 | 9 → 99 | **+59 [+25, +109], p 0.004** | **+78 [+47, +135], p 0.004** | **+0.65 [+0.46, +1.04], p 0.002** | **−0.13 [−0.15, −0.11], p 0.002** |
| Transformer `no_ar`, 1,000 customers | 0.10 | 80% | 10 | 49 → 265 | 21 → 265 | **+209 [+137, +303], p 0.002** | **+237 [+175, +325], p 0.002** | **+0.85 [+0.64, +1.04], p 0.002** | **−0.10 [−0.12, −0.08], p 0.002** |
| Transformer `no_ar`, 1,000 customers | 0.30 | 20% | 10 | 30 → 36 | 14 → 24 | −2 [−5, +25], p 0.432 | +3 [−8, +31], p 0.557 | **+0.83 [+0.48, +3.02], p 0.002** | **−0.14 [−0.17, −0.12], p 0.002** |
| Transformer `no_ar`, 1,000 customers | 0.30 | 40% | 10 | 31 → 36 | 14 → 29 | +4 [−5, +17], p 0.557 | **+13 [+3, +30], p 0.037** | **+1.17 [+0.86, +2.11], p 0.002** | **−0.16 [−0.18, −0.13], p 0.002** |
| Transformer `no_ar`, 1,000 customers | 0.30 | 60% | 10 | 32 → 57 | 13 → 50 | +20 [+1, +50], p 0.049 | **+36 [+9, +71], p 0.020** | **+1.33 [+1.06, +1.96], p 0.002** | **−0.19 [−0.19, −0.16], p 0.002** |
| Transformer `no_ar`, 1,000 customers | 0.30 | 80% | 10 | 35 → 96 | 13 → 88 | **+52 [+14, +107], p 0.014** | **+63 [+19, +127], p 0.014** | **+0.90 [+0.66, +1.68], p 0.002** | **−0.26 [−0.28, −0.24], p 0.002** |
| LSTM `ar_bounded`, 3,000 customers | 0.01 | 20% | 10 | 37 → 38 | 11 → 25 | +1 [−4, +5], p 0.770 | **+15 [+4, +22], p 0.014** | **+0.03 [+0.02, +0.04], p 0.002** | **−0.18 [−0.20, −0.16], p 0.002** |
| LSTM `ar_bounded`, 3,000 customers | 0.01 | 40% | 10 | 46 → 52 | 22 → 37 | +5 [−1, +12], p 0.064 | **+17 [+6, +26], p 0.014** | **+0.04 [+0.03, +0.04], p 0.002** | **−0.20 [−0.28, −0.13], p 0.002** |
| LSTM `ar_bounded`, 3,000 customers | 0.01 | 60% | 10 | 62 → 89 | 30 → 72 | **+28 [+5, +46], p 0.014** | **+41 [+19, +65], p 0.002** | **+0.03 [+0.02, +0.05], p 0.002** | **−0.21 [−0.26, −0.11], p 0.004** |
| LSTM `ar_bounded`, 3,000 customers | 0.01 | 80% | 10 | 78 → 152 | 22 → 127 | **+59 [+17, +136], p 0.004** | **+95 [+29, +165], p 0.002** | **+0.04 [+0.02, +0.11], p 0.002** | **−0.19 [−0.31, −0.15], p 0.002** |
| LSTM `ar_bounded`, 3,000 customers | 0.05 | 20% | 10 | 32 → 26 | 7 → 24 | −5 [−13, +2], p 0.131 | **+17 [+9, +25], p 0.002** | **+0.11 [+0.07, +0.14], p 0.002** | **−0.04 [−0.05, −0.03], p 0.002** |
| LSTM `ar_bounded`, 3,000 customers | 0.05 | 40% | 10 | 33 → 29 | 10 → 25 | −4 [−14, +6], p 0.492 | **+15 [+3, +28], p 0.020** | **+0.11 [+0.06, +0.16], p 0.002** | **−0.05 [−0.07, −0.04], p 0.002** |
| LSTM `ar_bounded`, 3,000 customers | 0.05 | 60% | 10 | 34 → 33 | 7 → 22 | −5 [−12, +11], p 0.557 | +11 [−1, +30], p 0.105 | **+0.08 [+0.04, +0.12], p 0.002** | **−0.04 [−0.05, −0.02], p 0.002** |
| LSTM `ar_bounded`, 3,000 customers | 0.05 | 80% | 10 | 39 → 37 | 8 → 16 | −2 [−5, +1], p 0.160 | +8 [−3, +19], p 0.131 | **+0.05 [+0.03, +0.08], p 0.002** | **−0.04 [−0.05, −0.03], p 0.002** |
| LSTM `ar_bounded`, 3,000 customers | 0.10 | 20% | 10 | 30 → 14 | 2 → 11 | **−17 [−21, −12], p 0.002** | **+9 [+3, +15], p 0.014** | **+0.10 [+0.06, +0.16], p 0.002** | **−0.02 [−0.02, −0.01], p 0.002** |
| LSTM `ar_bounded`, 3,000 customers | 0.10 | 40% | 10 | 30 → 12 | 2 → 6 | **−19 [−22, −16], p 0.002** | **+3 [0, +7], p 0.037** | **+0.06 [+0.03, +0.09], p 0.002** | **−0.01 [−0.02, −0.01], p 0.004** |
| LSTM `ar_bounded`, 3,000 customers | 0.10 | 60% | 10 | 32 → 16 | 3 → 7 | **−15 [−18, −13], p 0.002** | **+4 [0, +8], p 0.027** | **+0.07 [+0.04, +0.10], p 0.004** | **−0.02 [−0.02, −0.01], p 0.004** |
| LSTM `ar_bounded`, 3,000 customers | 0.10 | 80% | 10 | 35 → 32 | 7 → 14 | −4 [−7, 0], p 0.105 | +7 [+1, +13], p 0.049 | **+0.07 [+0.03, +0.10], p 0.002** | **−0.04 [−0.05, −0.04], p 0.002** |
| LSTM `ar_bounded`, 3,000 customers | 0.30 | 20% | 10 | 30 → 6 | 15 → 3 | **−24 [−25, −23], p 0.002** | **−12 [−14, −10], p 0.002** | **−0.85 [−1.04, −0.74], p 0.002** | 0 [0, +0.01], p 0.049 |
| LSTM `ar_bounded`, 3,000 customers | 0.30 | 40% | 10 | 30 → 10 | 14 → 7 | **−20 [−22, −18], p 0.002** | **−7 [−9, −5], p 0.002** | **−0.45 [−0.54, −0.36], p 0.002** | **−0.01 [−0.01, −0.01], p 0.002** |
| LSTM `ar_bounded`, 3,000 customers | 0.30 | 60% | 10 | 30 → 18 | 13 → 13 | **−12 [−16, −9], p 0.002** | 0 [−5, +6], p 0.770 | **−0.15 [−0.24, −0.04], p 0.002** | **−0.07 [−0.08, −0.06], p 0.002** |
| LSTM `ar_bounded`, 3,000 customers | 0.30 | 80% | 10 | 31 → 31 | 12 → 24 | −1 [−6, +6], p 0.846 | **+15 [+5, +19], p 0.010** | +0.12 [−0.02, +0.20], p 0.084 | **−0.19 [−0.20, −0.18], p 0.002** |
| LSTM `no_ar`, 3,000 customers | 0.01 | 20% | 10 | 37 → 84 | 11 → 83 | **+47 [+39, +56], p 0.002** | **+73 [+65, +80], p 0.002** | **+0.11 [+0.09, +0.12], p 0.002** | **−0.22 [−0.24, −0.20], p 0.002** |
| LSTM `no_ar`, 3,000 customers | 0.01 | 40% | 10 | 46 → 132 | 22 → 131 | **+83 [+75, +102], p 0.002** | **+106 [+98, +130], p 0.002** | **+0.14 [+0.13, +0.18], p 0.002** | **−0.30 [−0.33, −0.25], p 0.002** |
| LSTM `no_ar`, 3,000 customers | 0.01 | 60% | 10 | 62 → 261 | 30 → 261 | **+198 [+157, +238], p 0.002** | **+238 [+189, +274], p 0.002** | **+0.23 [+0.18, +0.28], p 0.002** | **−0.28 [−0.32, −0.23], p 0.002** |
| LSTM `no_ar`, 3,000 customers | 0.01 | 80% | 10 | 78 → 399 | 22 → 399 | **+319 [+250, +393], p 0.002** | **+379 [+310, +444], p 0.002** | **+0.22 [+0.18, +0.26], p 0.002** | **−0.27 [−0.31, −0.24], p 0.002** |
| LSTM `no_ar`, 3,000 customers | 0.05 | 20% | 10 | 32 → 26 | 7 → 25 | −6 [−11, 0], p 0.064 | **+17 [+12, +25], p 0.002** | **+0.13 [+0.10, +0.21], p 0.002** | **−0.07 [−0.09, −0.05], p 0.002** |
| LSTM `no_ar`, 3,000 customers | 0.05 | 40% | 10 | 33 → 69 | 10 → 69 | **+37 [+21, +51], p 0.002** | **+60 [+45, +73], p 0.002** | **+0.32 [+0.23, +0.41], p 0.002** | **−0.13 [−0.15, −0.10], p 0.002** |
| LSTM `no_ar`, 3,000 customers | 0.05 | 60% | 10 | 34 → 105 | 7 → 102 | **+80 [+38, +111], p 0.020** | **+105 [+54, +139], p 0.006** | **+0.39 [+0.19, +0.56], p 0.002** | **−0.24 [−0.38, −0.06], p 0.002** |
| LSTM `no_ar`, 3,000 customers | 0.05 | 80% | 10 | 39 → 277 | 8 → 274 | **+252 [+130, +327], p 0.002** | **+282 [+154, +361], p 0.002** | **+0.58 [+0.32, +0.68], p 0.002** | **−0.42 [−0.45, −0.25], p 0.002** |
| LSTM `no_ar`, 3,000 customers | 0.10 | 20% | 10 | 30 → 18 | 2 → 17 | **−12 [−16, −10], p 0.002** | **+15 [+13, +17], p 0.002** | **+0.21 [+0.18, +0.24], p 0.002** | **−0.04 [−0.05, −0.03], p 0.002** |
| LSTM `no_ar`, 3,000 customers | 0.10 | 40% | 10 | 30 → 29 | 2 → 27 | −2 [−8, +5], p 0.695 | **+27 [+18, +33], p 0.002** | **+0.21 [+0.18, +0.26], p 0.002** | **−0.04 [−0.05, −0.03], p 0.002** |
| LSTM `no_ar`, 3,000 customers | 0.10 | 60% | 10 | 32 → 17 | 3 → 8 | **−15 [−16, −13], p 0.002** | **+5 [+2, +9], p 0.010** | **+0.15 [+0.10, +0.19], p 0.002** | **−0.02 [−0.03, −0.01], p 0.006** |
| LSTM `no_ar`, 3,000 customers | 0.10 | 80% | 10 | 35 → 32 | 7 → 22 | −3 [−7, +1], p 0.193 | **+15 [+9, +20], p 0.004** | **+0.13 [+0.10, +0.18], p 0.002** | **−0.04 [−0.05, −0.03], p 0.002** |
| LSTM `no_ar`, 3,000 customers | 0.30 | 20% | 10 | 30 → 8 | 15 → 6 | **−23 [−24, −20], p 0.002** | **−9 [−12, −7], p 0.002** | **−0.75 [−0.84, −0.62], p 0.002** | **0 [0, +0.01], p 0.006** |
| LSTM `no_ar`, 3,000 customers | 0.30 | 40% | 10 | 30 → 9 | 14 → 4 | **−22 [−23, −20], p 0.002** | **−10 [−13, −7], p 0.002** | **−0.40 [−0.50, −0.20], p 0.004** | **−0.01 [−0.01, −0.01], p 0.002** |
| LSTM `no_ar`, 3,000 customers | 0.30 | 60% | 10 | 30 → 12 | 13 → 7 | **−18 [−20, −16], p 0.002** | **−6 [−10, −2], p 0.020** | **−0.17 [−0.30, −0.06], p 0.006** | **−0.07 [−0.08, −0.06], p 0.002** |
| LSTM `no_ar`, 3,000 customers | 0.30 | 80% | 10 | 31 → 28 | 12 → 23 | −3 [−8, +2], p 0.160 | **+12 [+3, +18], p 0.002** | +0.12 [0, +0.25], p 0.049 | **−0.19 [−0.20, −0.18], p 0.002** |

</details>

### 2. Neural error rises with churn

**Verdict: partly.** MAPE rises with churn for nearly every tree, Pareto/NBD included, so
that half of the claim is not specific to the neural models. What is specific is the size
of the |bias| rise without flags: the LSTM `no_ar` goes from 59 to 755 at rate 0.01 and
from 42 to 497 at rate 0.05, against 12 to 54 and 8 to 21 for Pareto/NBD. With bounded
flags the |bias| trend disappears at rates 0.05–0.10 at both cohort sizes. It stays at rate
0.01 and, at 3,000 customers, reappears at rate 0.30 (3 → 24).

Each row gives the mean at churn 20 / 40 / 60 / 80% and the Spearman correlation of the
metric with churn over the rate's 40 panels. RMSE is left out: it falls with churn for
every tree because there is less volume.

| Tree | Customers | Rate | \|bias\| at churn 20 / 40 / 60 / 80% | ρ(\|bias\|, churn), p | MAPE at churn 20 / 40 / 60 / 80% | ρ(MAPE, churn), p |
| --- | --- | --- | --- | --- | --- | --- |
| Pareto/NBD | 1,000 | 0.01 | 12 / 34 / 41 / 54 | **+0.61, p <10⁻⁴** | 48 / 66 / 88 / 159 | **+0.97, p <10⁻⁴** |
| Pareto/NBD | 1,000 | 0.05 | 8 / 15 / 12 / 21 | +0.30, p 0.058 | 35 / 39 / 42 / 58 | **+0.86, p <10⁻⁴** |
| Pareto/NBD | 1,000 | 0.10 | 3 / 5 / 9 / 21 | **+0.60, p <10⁻⁴** | 31 / 32 / 37 / 49 | **+0.87, p <10⁻⁴** |
| Pareto/NBD | 1,000 | 0.30 | 14 / 14 / 13 / 13 | −0.16, p 0.322 | 30 / 31 / 32 / 35 | **+0.70, p <10⁻⁴** |
| LSTM `no_ar` | 1,000 | 0.01 | 59 / 132 / 290 / 755 | **+0.94, p <10⁻⁴** | 71 / 136 / 291 / 759 | **+0.95, p <10⁻⁴** |
| LSTM `no_ar` | 1,000 | 0.05 | 42 / 98 / 157 / 497 | **+0.95, p <10⁻⁴** | 45 / 99 / 158 / 497 | **+0.95, p <10⁻⁴** |
| LSTM `no_ar` | 1,000 | 0.10 | 31 / 57 / 61 / 225 | **+0.46, p 0.003** | 33 / 58 / 69 / 233 | **+0.64, p <10⁻⁴** |
| LSTM `no_ar` | 1,000 | 0.30 | 17 / 9 / 9 / 17 | −0.08, p 0.643 | 19 / 14 / 18 / 32 | **+0.47, p 0.002** |
| LSTM `ar_bounded` | 1,000 | 0.01 | 47 / 114 / 190 / 576 | **+0.88, p <10⁻⁴** | 65 / 121 / 196 / 591 | **+0.91, p <10⁻⁴** |
| LSTM `ar_bounded` | 1,000 | 0.05 | 47 / 52 / 61 / 27 | −0.22, p 0.171 | 48 / 56 / 67 / 60 | **+0.34, p 0.029** |
| LSTM `ar_bounded` | 1,000 | 0.10 | 30 / 28 / 21 / 15 | **−0.37, p 0.018** | 32 / 34 / 33 / 45 | **+0.36, p 0.023** |
| LSTM `ar_bounded` | 1,000 | 0.30 | 10 / 6 / 10 / 19 | +0.29, p 0.067 | 12 / 13 / 19 / 35 | **+0.88, p <10⁻⁴** |
| Transformer `no_ar` | 1,000 | 0.01 | 37 / 76 / 102 / 121 | **+0.40, p 0.011** | 58 / 96 / 129 / 197 | **+0.76, p <10⁻⁴** |
| Transformer `no_ar` | 1,000 | 0.05 | 72 / 108 / 144 / 111 | +0.21, p 0.204 | 75 / 110 / 149 / 122 | +0.26, p 0.103 |
| Transformer `no_ar` | 1,000 | 0.10 | 56 / 91 / 99 / 265 | **+0.66, p <10⁻⁴** | 58 / 94 / 103 / 265 | **+0.68, p <10⁻⁴** |
| Transformer `no_ar` | 1,000 | 0.30 | 24 / 29 / 50 / 88 | **+0.45, p 0.004** | 36 / 36 / 57 / 96 | **+0.59, p <10⁻⁴** |
| Transformer `ar_bounded` | 1,000 | 0.01 | 44 / 58 / 60 / 90 | +0.23, p 0.149 | 62 / 75 / 98 / 176 | **+0.82, p <10⁻⁴** |
| Transformer `ar_bounded` | 1,000 | 0.05 | 63 / 54 / 69 / 84 | +0.07, p 0.669 | 68 / 65 / 77 / 102 | +0.20, p 0.226 |
| Transformer `ar_bounded` | 1,000 | 0.10 | 45 / 33 / 113 / 90 | **+0.35, p 0.027** | 52 / 41 / 115 / 102 | **+0.49, p 0.001** |
| Transformer `ar_bounded` | 1,000 | 0.30 | 27 / 35 / 62 / 46 | +0.29, p 0.067 | 37 / 42 / 67 / 54 | **+0.40, p 0.011** |
| Pareto/NBD | 3,000 | 0.01 | 11 / 22 / 30 / 22 | **+0.38, p 0.017** | 37 / 46 / 62 / 78 | **+0.91, p <10⁻⁴** |
| Pareto/NBD | 3,000 | 0.05 | 7 / 10 / 7 / 8 | −0.04, p 0.794 | 32 / 33 / 34 / 39 | **+0.45, p 0.003** |
| Pareto/NBD | 3,000 | 0.10 | 2 / 2 / 3 / 7 | **+0.44, p 0.004** | 30 / 30 / 32 / 35 | **+0.69, p <10⁻⁴** |
| Pareto/NBD | 3,000 | 0.30 | 15 / 14 / 13 / 12 | −0.30, p 0.058 | 30 / 30 / 30 / 31 | +0.25, p 0.120 |
| LSTM `no_ar` | 3,000 | 0.01 | 83 / 131 / 261 / 399 | **+0.95, p <10⁻⁴** | 84 / 132 / 261 / 399 | **+0.95, p <10⁻⁴** |
| LSTM `no_ar` | 3,000 | 0.05 | 25 / 69 / 102 / 274 | **+0.69, p <10⁻⁴** | 26 / 69 / 105 / 277 | **+0.75, p <10⁻⁴** |
| LSTM `no_ar` | 3,000 | 0.10 | 17 / 27 / 8 / 22 | −0.07, p 0.669 | 18 / 29 / 17 / 32 | **+0.35, p 0.027** |
| LSTM `no_ar` | 3,000 | 0.30 | 6 / 4 / 7 / 23 | **+0.61, p <10⁻⁴** | 8 / 9 / 12 / 28 | **+0.84, p <10⁻⁴** |
| LSTM `ar_bounded` | 3,000 | 0.01 | 25 / 37 / 72 / 127 | **+0.60, p <10⁻⁴** | 38 / 52 / 89 / 152 | **+0.89, p <10⁻⁴** |
| LSTM `ar_bounded` | 3,000 | 0.05 | 24 / 25 / 22 / 16 | −0.25, p 0.117 | 26 / 29 / 33 / 37 | **+0.35, p 0.026** |
| LSTM `ar_bounded` | 3,000 | 0.10 | 11 / 6 / 7 / 14 | +0.21, p 0.191 | 14 / 12 / 16 / 32 | **+0.69, p <10⁻⁴** |
| LSTM `ar_bounded` | 3,000 | 0.30 | 3 / 7 / 13 / 24 | **+0.79, p <10⁻⁴** | 6 / 10 / 18 / 31 | **+0.93, p <10⁻⁴** |

### 3. Bounded AR flags help

**Verdict: supported, conditionally.** The flags improve the LSTM's level on panels with
purchase rates 0.01–0.10, and its ranking at 0.05–0.10. The gain grows with churn, is absent at churn 20% and
at rate 0.30, and is larger with more customers. For the Transformer they help less, only
at rates 0.05–0.10.

**How much, LSTM at 1,000 customers.**

- Over the grid: Δ MAPE −28 points [−44, −17] and Δ |bias| −32 [−48, −20]; the mean MAPE
  falls from 158 to 89.
- By rate: −44 MAPE points at rate 0.01, −85 at 0.05, −26 at 0.10, and no change at 0.30
  (−1 [−3, +1], p = 0.34). RMSE on customer totals improves at every rate, by 0.05–0.45.
- By churn: at rate 0.05 the change is +5 (not significant), −42, −91 and −439 MAPE
  points at churn 20, 40, 60 and 80%. The same rise with churn holds at rates 0.01 and 0.10.
  Where silence is common, the flags are what lets the model learn that silence means
  dropout.
- Ranking improves at rates 0.05–0.10 (+0.41 and +0.03 Spearman) but gets worse at rate
  0.01 (−0.07 [−0.09, −0.04]).

**How much, Transformer at 1,000 customers.**

- Over the grid: Δ MAPE −20 points [−31, −9] and Δ |bias| −24 [−38, −12]; the mean MAPE
  falls from 105 to 77.
- By rate: significant at 0.05 (−35) and 0.10 (−48), nothing at 0.01 or 0.30 on the level.
  Per cell, the MAPE gain is significant only at rate 0.10, churn 40% (−56) and 80% (−166).
- Ranking improves from rate 0.05 up in most cells, by 0.04–0.14.

**How much, LSTM at 3,000 customers.**

- The flags help more: −32 MAPE points over the grid, and −125 at rate 0.01, where they are
  significant in every churn cell (−46 to −251).
- At 3,000 customers the flags also improve the ranking at rate 0.01 (+0.07), where they
  worsened it at 1,000.
- At rate 0.30 they cost 2 MAPE points [0, 4] (p = 0.03), and 6 at churn 60%.

**LSTM, 1,000 customers**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | −9 / −16 / −0.05 | −10 / −11 / **−0.09** | **−97** / **−102** / −0.07 | −147 / −148 / −0.05 | **−44** / **−48** / **−0.07** |
| 0.05 | +5 / +6 / **+0.25** | **−42** / **−45** / **+0.30** | **−91** / **−97** / **+0.48** | **−439** / **−468** / **+0.40** | **−85** / **−92** / **+0.41** |
| 0.10 | 0 / −1 / **+0.02** | −13 / −18 / **+0.02** | **−34** / −38 / **+0.03** | **−150** / **−173** / **+0.19** | **−26** / **−31** / **+0.03** |
| 0.30 | **−8** / −10 / 0 | −2 / −3 / 0 | 0 / +2 / 0 | **+3** / +1 / 0 | −1 / −2 / 0 |
| all |  |  |  |  | **−28** / **−32** / **+0.03** |

**Transformer, 1,000 customers**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | +1 / +5 / −0.02 | −17 / −17 / −0.03 | −23 / −34 / −0.01 | −10 / −26 / +0.01 | −9 / −12 / −0.02 |
| 0.05 | −6 / −5 / +0.14 | −54 / −62 / **+0.13** | −73 / −78 / **+0.14** | −26 / −39 / **+0.09** | **−35** / **−42** / **+0.12** |
| 0.10 | −7 / −9 / **+0.10** | **−56** / **−62** / +0.11 | +10 / +15 / **+0.09** | **−166** / **−168** / **+0.06** | **−48** / **−52** / **+0.09** |
| 0.30 | 0 / +1 / **+0.09** | +6 / +6 / +0.08 | +10 / +16 / **+0.08** | −31 / −30 / **+0.04** | −1 / 0 / **+0.08** |
| all |  |  |  |  | **−20** / **−24** / **+0.07** |

**LSTM, 3,000 customers**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **−46** / **−59** / +0.04 | **−78** / **−88** / +0.09 | **−172** / **−194** / +0.08 | **−251** / **−270** / +0.08 | **−125** / **−142** / **+0.07** |
| 0.05 | −2 / −3 / **+0.03** | **−40** / **−44** / **+0.08** | **−80** / **−80** / **+0.20** | **−255** / **−277** / **+0.39** | **−60** / **−67** / **+0.12** |
| 0.10 | −4 / −7 / **+0.02** | **−18** / **−22** / **+0.02** | −1 / −1 / 0 | 0 / **−8** / 0 | **−4** / **−8** / **+0.01** |
| 0.30 | −1 / −3 / 0 | +1 / +3 / 0 | **+6** / +7 / 0 | +2 / +1 / 0 | **+2** / +2 / 0 |
| all |  |  |  |  | **−32** / **−38** / **+0.03** |

**Intervals and p-values by rate, churn pooled**

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI], p | Δ \|bias\| [95% CI], p | Δ RMSE [95% CI], p | Δ Spearman [95% CI], p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM, 1,000 customers | 0.01 | pooled | 40 | 314 → 243 | 309 → 232 | **−44 [−78, −16], p 0.0005** | **−48 [−88, −20], p 0.0004** | **−0.05 [−0.09, −0.02], p 0.002** | **−0.07 [−0.09, −0.04], p 0.0001** |
| LSTM, 1,000 customers | 0.05 | pooled | 40 | 200 → 58 | 199 → 46 | **−85 [−211, −46], p <10⁻⁴** | **−92 [−227, −49], p <10⁻⁴** | **−0.45 [−0.57, −0.34], p <10⁻⁴** | **+0.41 [+0.27, +0.47], p <10⁻⁴** |
| LSTM, 1,000 customers | 0.10 | pooled | 40 | 98 → 36 | 93 → 24 | **−26 [−70, −10], p <10⁻⁴** | **−31 [−78, −13], p <10⁻⁴** | **−0.20 [−0.37, −0.09], p 0.0001** | **+0.03 [+0.02, +0.06], p <10⁻⁴** |
| LSTM, 1,000 customers | 0.30 | pooled | 40 | 21 → 20 | 13 → 11 | −1 [−3, +1], p 0.340 | −2 [−5, +1], p 0.237 | **−0.20 [−0.32, −0.11], p <10⁻⁴** | 0 [0, 0], p 0.192 |
| LSTM, 1,000 customers | all | pooled | 160 | 158 → 89 | 154 → 78 | **−28 [−44, −17], p <10⁻⁴** | **−32 [−48, −20], p <10⁻⁴** | **−0.21 [−0.28, −0.16], p <10⁻⁴** | **+0.03 [+0.01, +0.08], p <10⁻⁴** |
| Transformer, 1,000 customers | 0.01 | pooled | 40 | 120 → 103 | 84 → 63 | −9 [−24, +2], p 0.135 | −12 [−34, +7], p 0.277 | −0.01 [−0.03, 0], p 0.242 | −0.02 [−0.03, 0], p 0.068 |
| Transformer, 1,000 customers | 0.05 | pooled | 40 | 114 → 78 | 109 → 68 | **−35 [−60, −9], p 0.010** | **−42 [−71, −12], p 0.009** | **−0.21 [−0.39, −0.09], p <10⁻⁴** | **+0.12 [+0.09, +0.15], p <10⁻⁴** |
| Transformer, 1,000 customers | 0.10 | pooled | 40 | 130 → 77 | 128 → 70 | **−48 [−78, −20], p 0.0007** | **−52 [−85, −23], p 0.0003** | **−0.49 [−0.66, −0.28], p <10⁻⁴** | **+0.09 [+0.07, +0.10], p <10⁻⁴** |
| Transformer, 1,000 customers | 0.30 | pooled | 40 | 56 → 50 | 48 → 42 | −1 [−13, +9], p 0.775 | 0 [−16, +14], p 0.984 | **−0.59 [−0.84, −0.22], p 0.009** | **+0.08 [+0.06, +0.09], p <10⁻⁴** |
| Transformer, 1,000 customers | all | pooled | 160 | 105 → 77 | 92 → 61 | **−20 [−31, −9], p <10⁻⁴** | **−24 [−38, −12], p <10⁻⁴** | **−0.30 [−0.38, −0.17], p <10⁻⁴** | **+0.07 [+0.05, +0.08], p <10⁻⁴** |
| LSTM, 3,000 customers | 0.01 | pooled | 40 | 219 → 83 | 219 → 65 | **−125 [−162, −94], p <10⁻⁴** | **−142 [−180, −106], p <10⁻⁴** | **−0.13 [−0.16, −0.11], p <10⁻⁴** | **+0.07 [+0.04, +0.09], p <10⁻⁴** |
| LSTM, 3,000 customers | 0.05 | pooled | 40 | 119 → 31 | 118 → 22 | **−60 [−123, −31], p <10⁻⁴** | **−67 [−137, −34], p <10⁻⁴** | **−0.24 [−0.33, −0.16], p <10⁻⁴** | **+0.12 [+0.06, +0.22], p <10⁻⁴** |
| LSTM, 3,000 customers | 0.10 | pooled | 40 | 24 → 18 | 19 → 9 | **−4 [−8, −2], p 0.0001** | **−8 [−13, −5], p <10⁻⁴** | **−0.10 [−0.12, −0.08], p <10⁻⁴** | **+0.01 [0, +0.02], p 0.0003** |
| LSTM, 3,000 customers | 0.30 | pooled | 40 | 14 → 16 | 10 → 12 | **+2 [0, +4], p 0.033** | +2 [0, +4], p 0.109 | **−0.04 [−0.10, 0], p 0.029** | 0 [0, 0], p 0.068 |
| LSTM, 3,000 customers | all | pooled | 160 | 94 → 37 | 91 → 27 | **−32 [−47, −21], p <10⁻⁴** | **−38 [−54, −25], p <10⁻⁴** | **−0.12 [−0.14, −0.10], p <10⁻⁴** | **+0.03 [+0.02, +0.04], p <10⁻⁴** |

<details><summary>Intervals and p-values per rate × churn cell</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI], p | Δ \|bias\| [95% CI], p | Δ RMSE [95% CI], p | Δ Spearman [95% CI], p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM, 1,000 customers | 0.01 | 20% | 10 | 71 → 65 | 59 → 47 | −9 [−21, +7], p 0.375 | −16 [−36, +7], p 0.375 | −0.01 [−0.04, +0.04], p 0.557 | −0.05 [−0.11, +0.01], p 0.131 |
| LSTM, 1,000 customers | 0.01 | 40% | 10 | 136 → 121 | 132 → 114 | −10 [−40, +6], p 0.131 | −11 [−45, +6], p 0.160 | −0.01 [−0.06, +0.02], p 0.375 | **−0.09 [−0.15, −0.05], p 0.002** |
| LSTM, 1,000 customers | 0.01 | 60% | 10 | 291 → 196 | 290 → 190 | **−97 [−143, −38], p 0.014** | **−102 [−152, −39], p 0.014** | **−0.14 [−0.20, −0.04], p 0.014** | −0.07 [−0.13, +0.03], p 0.275 |
| LSTM, 1,000 customers | 0.01 | 80% | 10 | 759 → 591 | 755 → 576 | −147 [−307, −7], p 0.049 | −148 [−338, −5], p 0.049 | −0.14 [−0.28, +0.01], p 0.064 | −0.05 [−0.11, 0], p 0.131 |
| LSTM, 1,000 customers | 0.05 | 20% | 10 | 45 → 48 | 42 → 47 | +5 [−3, +11], p 0.275 | +6 [−3, +12], p 0.160 | −0.03 [−0.17, +0.10], p 0.492 | **+0.25 [+0.03, +0.48], p 0.004** |
| LSTM, 1,000 customers | 0.05 | 40% | 10 | 99 → 56 | 98 → 52 | **−42 [−54, −31], p 0.002** | **−45 [−58, −33], p 0.002** | **−0.39 [−0.48, −0.22], p 0.004** | **+0.30 [+0.06, +0.54], p 0.002** |
| LSTM, 1,000 customers | 0.05 | 60% | 10 | 158 → 67 | 157 → 61 | **−91 [−122, −59], p 0.002** | **−97 [−135, −60], p 0.002** | **−0.49 [−0.66, −0.38], p 0.002** | **+0.48 [+0.47, +0.52], p 0.002** |
| LSTM, 1,000 customers | 0.05 | 80% | 10 | 497 → 60 | 497 → 27 | **−439 [−497, −373], p 0.002** | **−468 [−533, −404], p 0.002** | **−0.91 [−1.09, −0.73], p 0.002** | **+0.40 [+0.36, +0.45], p 0.002** |
| LSTM, 1,000 customers | 0.10 | 20% | 10 | 33 → 32 | 31 → 30 | 0 [−13, +11], p 1.000 | −1 [−15, +14], p 0.922 | +0.02 [−0.14, +0.20], p 0.432 | **+0.02 [0, +0.04], p 0.014** |
| LSTM, 1,000 customers | 0.10 | 40% | 10 | 58 → 34 | 57 → 28 | −13 [−73, +2], p 0.160 | −18 [−78, +1], p 0.105 | −0.10 [−1.16, 0], p 0.084 | **+0.02 [+0.01, +0.15], p 0.006** |
| LSTM, 1,000 customers | 0.10 | 60% | 10 | 69 → 33 | 61 → 21 | **−34 [−67, −5], p 0.014** | −38 [−78, −2], p 0.027 | **−0.27 [−0.52, −0.16], p 0.002** | **+0.03 [+0.01, +0.05], p 0.010** |
| LSTM, 1,000 customers | 0.10 | 80% | 10 | 233 → 45 | 225 → 15 | **−150 [−332, −48], p 0.006** | **−173 [−348, −61], p 0.006** | **−0.58 [−0.98, −0.12], p 0.010** | **+0.19 [+0.01, +0.36], p 0.020** |
| LSTM, 1,000 customers | 0.30 | 20% | 10 | 19 → 12 | 17 → 10 | **−8 [−12, −2], p 0.014** | −10 [−15, 0], p 0.105 | −0.49 [−0.77, −0.07], p 0.027 | 0 [−0.01, +0.01], p 0.432 |
| LSTM, 1,000 customers | 0.30 | 40% | 10 | 14 → 13 | 9 → 6 | −2 [−4, 0], p 0.064 | −3 [−6, +1], p 0.131 | −0.09 [−0.22, +0.05], p 0.160 | 0 [0, +0.01], p 0.193 |
| LSTM, 1,000 customers | 0.30 | 60% | 10 | 18 → 19 | 9 → 10 | 0 [−2, +5], p 0.625 | +2 [−4, +8], p 0.625 | **−0.12 [−0.24, −0.04], p 0.010** | 0 [0, +0.01], p 0.557 |
| LSTM, 1,000 customers | 0.30 | 80% | 10 | 32 → 35 | 17 → 19 | **+3 [0, +6], p 0.020** | +1 [−6, +10], p 0.557 | −0.24 [−0.44, −0.04], p 0.027 | 0 [−0.01, +0.01], p 0.770 |
| Transformer, 1,000 customers | 0.01 | 20% | 10 | 58 → 62 | 37 → 44 | +1 [−8, +18], p 0.922 | +5 [−12, +29], p 0.625 | 0 [−0.01, +0.04], p 0.770 | −0.02 [−0.04, −0.01], p 0.037 |
| Transformer, 1,000 customers | 0.01 | 40% | 10 | 96 → 75 | 76 → 58 | −17 [−61, +14], p 0.557 | −17 [−64, +28], p 0.625 | −0.03 [−0.10, +0.02], p 0.557 | −0.03 [−0.08, +0.01], p 0.105 |
| Transformer, 1,000 customers | 0.01 | 60% | 10 | 129 → 98 | 102 → 60 | −23 [−91, +4], p 0.105 | −34 [−114, +5], p 0.131 | −0.02 [−0.09, 0], p 0.105 | −0.01 [−0.04, +0.04], p 0.432 |
| Transformer, 1,000 customers | 0.01 | 80% | 10 | 197 → 176 | 121 → 90 | −10 [−70, +28], p 0.375 | −26 [−122, +51], p 0.695 | −0.01 [−0.03, +0.02], p 0.846 | +0.01 [−0.05, +0.07], p 0.695 |
| Transformer, 1,000 customers | 0.05 | 20% | 10 | 75 → 68 | 72 → 63 | −6 [−27, +17], p 0.770 | −5 [−39, +20], p 0.846 | −0.05 [−0.42, +0.12], p 0.492 | +0.14 [0, +0.30], p 0.049 |
| Transformer, 1,000 customers | 0.05 | 40% | 10 | 110 → 65 | 108 → 54 | −54 [−97, +2], p 0.064 | −62 [−104, 0], p 0.049 | −0.36 [−0.77, −0.09], p 0.049 | **+0.13 [+0.09, +0.17], p 0.002** |
| Transformer, 1,000 customers | 0.05 | 60% | 10 | 149 → 77 | 144 → 69 | −73 [−150, +10], p 0.160 | −78 [−165, +14], p 0.160 | **−0.41 [−0.72, −0.12], p 0.002** | **+0.14 [+0.09, +0.20], p 0.002** |
| Transformer, 1,000 customers | 0.05 | 80% | 10 | 122 → 102 | 111 → 84 | −26 [−95, +43], p 0.492 | −39 [−119, +49], p 0.432 | −0.08 [−0.16, −0.02], p 0.037 | **+0.09 [+0.06, +0.12], p 0.002** |
| Transformer, 1,000 customers | 0.10 | 20% | 10 | 58 → 52 | 56 → 45 | −7 [−31, +21], p 0.557 | −9 [−41, +20], p 0.432 | −0.46 [−0.77, +0.03], p 0.084 | **+0.10 [+0.07, +0.12], p 0.002** |
| Transformer, 1,000 customers | 0.10 | 40% | 10 | 94 → 41 | 91 → 33 | **−56 [−87, −20], p 0.020** | **−62 [−96, −21], p 0.020** | **−0.86 [−1.29, −0.37], p 0.010** | +0.11 [−0.14, +0.13], p 0.084 |
| Transformer, 1,000 customers | 0.10 | 60% | 10 | 103 → 115 | 99 → 113 | +10 [−47, +73], p 0.625 | +15 [−47, +73], p 0.695 | −0.02 [−0.46, +0.58], p 0.922 | **+0.09 [+0.05, +0.12], p 0.004** |
| Transformer, 1,000 customers | 0.10 | 80% | 10 | 265 → 102 | 265 → 90 | **−166 [−215, −117], p 0.002** | **−168 [−227, −118], p 0.002** | **−0.63 [−0.88, −0.41], p 0.002** | **+0.06 [+0.04, +0.09], p 0.002** |
| Transformer, 1,000 customers | 0.30 | 20% | 10 | 36 → 37 | 24 → 27 | 0 [−14, +21], p 0.922 | +1 [−24, +32], p 0.922 | −0.45 [−1.64, +1.54], p 0.492 | **+0.09 [+0.07, +0.11], p 0.002** |
| Transformer, 1,000 customers | 0.30 | 40% | 10 | 36 → 42 | 29 → 35 | +6 [−11, +25], p 0.432 | +6 [−16, +29], p 0.695 | −0.59 [−1.78, +0.77], p 0.375 | +0.08 [−0.09, +0.11], p 0.432 |
| Transformer, 1,000 customers | 0.30 | 60% | 10 | 57 → 67 | 50 → 62 | +10 [−19, +38], p 0.492 | +16 [−22, +47], p 0.375 | −0.50 [−0.86, +0.31], p 0.375 | **+0.08 [+0.06, +0.10], p 0.002** |
| Transformer, 1,000 customers | 0.30 | 80% | 10 | 96 → 54 | 88 → 46 | −31 [−96, +2], p 0.064 | −30 [−100, +13], p 0.131 | **−0.77 [−1.47, −0.46], p 0.004** | **+0.04 [+0.01, +0.07], p 0.020** |
| LSTM, 3,000 customers | 0.01 | 20% | 10 | 84 → 38 | 83 → 25 | **−46 [−56, −35], p 0.002** | **−59 [−77, −43], p 0.002** | **−0.08 [−0.10, −0.06], p 0.002** | +0.04 [+0.01, +0.07], p 0.037 |
| LSTM, 3,000 customers | 0.01 | 40% | 10 | 132 → 52 | 131 → 37 | **−78 [−98, −67], p 0.002** | **−88 [−121, −77], p 0.002** | **−0.11 [−0.14, −0.09], p 0.002** | +0.09 [+0.01, +0.16], p 0.027 |
| LSTM, 3,000 customers | 0.01 | 60% | 10 | 261 → 89 | 261 → 72 | **−172 [−218, −131], p 0.002** | **−194 [−244, −140], p 0.002** | **−0.19 [−0.25, −0.14], p 0.002** | +0.08 [+0.02, +0.16], p 0.027 |
| LSTM, 3,000 customers | 0.01 | 80% | 10 | 399 → 152 | 399 → 127 | **−251 [−317, −169], p 0.002** | **−270 [−350, −189], p 0.002** | **−0.16 [−0.22, −0.12], p 0.002** | +0.08 [−0.02, +0.12], p 0.131 |
| LSTM, 3,000 customers | 0.05 | 20% | 10 | 26 → 26 | 25 → 24 | −2 [−7, +7], p 0.625 | −3 [−11, +7], p 0.695 | −0.03 [−0.10, +0.01], p 0.232 | **+0.03 [+0.01, +0.05], p 0.020** |
| LSTM, 3,000 customers | 0.05 | 40% | 10 | 69 → 29 | 69 → 25 | **−40 [−54, −24], p 0.004** | **−44 [−62, −28], p 0.002** | **−0.21 [−0.28, −0.13], p 0.002** | **+0.08 [+0.04, +0.10], p 0.002** |
| LSTM, 3,000 customers | 0.05 | 60% | 10 | 105 → 33 | 102 → 22 | **−80 [−112, −40], p 0.014** | **−80 [−123, −43], p 0.010** | **−0.30 [−0.48, −0.12], p 0.010** | **+0.20 [+0.03, +0.34], p 0.020** |
| LSTM, 3,000 customers | 0.05 | 80% | 10 | 277 → 37 | 274 → 16 | **−255 [−327, −131], p 0.004** | **−277 [−344, −143], p 0.004** | **−0.52 [−0.65, −0.24], p 0.002** | **+0.39 [+0.21, +0.41], p 0.002** |
| LSTM, 3,000 customers | 0.10 | 20% | 10 | 18 → 14 | 17 → 11 | −4 [−7, 0], p 0.049 | −7 [−11, −1], p 0.027 | **−0.11 [−0.16, −0.04], p 0.006** | **+0.02 [+0.01, +0.03], p 0.002** |
| LSTM, 3,000 customers | 0.10 | 40% | 10 | 29 → 12 | 27 → 6 | **−18 [−23, −12], p 0.002** | **−22 [−30, −14], p 0.002** | **−0.14 [−0.20, −0.11], p 0.002** | **+0.02 [+0.01, +0.03], p 0.002** |
| LSTM, 3,000 customers | 0.10 | 60% | 10 | 17 → 16 | 8 → 7 | −1 [−3, +1], p 0.322 | −1 [−6, +4], p 0.695 | **−0.09 [−0.14, −0.02], p 0.020** | 0 [−0.01, +0.01], p 0.695 |
| LSTM, 3,000 customers | 0.10 | 80% | 10 | 32 → 32 | 22 → 14 | 0 [−4, +3], p 1.000 | **−8 [−14, −2], p 0.020** | **−0.06 [−0.11, −0.03], p 0.002** | 0 [−0.01, 0], p 0.049 |
| LSTM, 3,000 customers | 0.30 | 20% | 10 | 8 → 6 | 6 → 3 | −1 [−3, 0], p 0.232 | −3 [−4, 0], p 0.049 | **−0.14 [−0.22, −0.05], p 0.010** | 0 [0, 0], p 0.322 |
| LSTM, 3,000 customers | 0.30 | 40% | 10 | 9 → 10 | 4 → 7 | +1 [0, +3], p 0.131 | +3 [−1, +6], p 0.105 | −0.07 [−0.18, +0.05], p 0.193 | 0 [0, 0], p 0.922 |
| LSTM, 3,000 customers | 0.30 | 60% | 10 | 12 → 18 | 7 → 13 | **+6 [+2, +9], p 0.014** | +7 [+1, +12], p 0.027 | +0.03 [−0.04, +0.09], p 0.432 | 0 [0, 0], p 0.770 |
| LSTM, 3,000 customers | 0.30 | 80% | 10 | 28 → 31 | 23 → 24 | +2 [−3, +8], p 0.232 | +1 [−6, +9], p 0.922 | −0.03 [−0.13, +0.06], p 0.695 | 0 [0, +0.01], p 0.160 |

</details>

### 4. Unbounded counters hurt

**Verdict: supported for the LSTM, partly for the Transformer.** The counters make the
LSTM's level worse at every rate (+98 to +260 MAPE points, rate pooled) because they keep
growing through the holdout, past any value seen in training. The Transformer is hurt only
at rate 0.30 (+37 MAPE, +43 |bias|). Ranking does not follow the level: the counters raise
both models' Spearman at rate 0.05 (+0.24 LSTM, +0.11 Transformer).

**LSTM**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | +75 / +75 / **−0.18** | +15 / +16 / **−0.23** | **+148** / **+152** / **−0.22** | +76 / +62 / **−0.09** | **+98** / **+98** / **−0.18** |
| 0.05 | **+191** / **+194** / +0.18 | **+211** / **+211** / +0.18 | **+264** / **+264** / **+0.32** | +803 / +802 / **+0.27** | **+260** / **+261** / **+0.24** |
| 0.10 | **+249** / **+249** / **−0.07** | **+293** / **+294** / −0.09 | +224 / +220 / **−0.08** | +12 / +20 / +0.15 | **+237** / **+238** / **−0.06** |
| 0.30 | **+252** / **+256** / **−0.12** | **+146** / **+148** / **−0.08** | **+47** / **+48** / **−0.04** | +36 / +39 / **−0.03** | **+115** / **+115** / **−0.06** |
| all |  |  |  |  | **+172** / **+173** / **−0.05** |

**Transformer**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | −4 / −11 / **−0.10** | −5 / −10 / −0.02 | −3 / −28 / −0.03 | +15 / +27 / −0.06 | −4 / −14 / **−0.05** |
| 0.05 | +29 / +29 / **+0.20** | −6 / −8 / +0.14 | −20 / −25 / **+0.13** | −20 / −37 / −0.13 | −8 / −10 / **+0.11** |
| 0.10 | **+37** / **+38** / **+0.07** | −4 / −4 / +0.08 | +73 / +75 / **+0.05** | −31 / −31 / +0.05 | +19 / +21 / **+0.06** |
| 0.30 | +35 / +47 / +0.07 | **+47** / **+54** / +0.02 | +34 / +42 / +0.04 | +18 / +22 / −0.01 | **+37** / **+43** / **+0.03** |
| all |  |  |  |  | +10 / +9 / **+0.04** |

<details><summary>Intervals and p-values by rate, churn pooled</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI], p | Δ \|bias\| [95% CI], p | Δ RMSE [95% CI], p | Δ Spearman [95% CI], p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM | 0.01 | pooled | 40 | 314 → 513 | 309 → 505 | **+98 [+29, +175], p 0.0010** | **+98 [+28, +179], p 0.002** | **+0.85 [+0.12, +3.11], p <10⁻⁴** | **−0.18 [−0.21, −0.15], p <10⁻⁴** |
| LSTM | 0.05 | pooled | 40 | 200 → 582 | 199 → 582 | **+260 [+185, +401], p <10⁻⁴** | **+261 [+186, +402], p <10⁻⁴** | **+9.75 [+8.02, +11.40], p <10⁻⁴** | **+0.24 [+0.17, +0.32], p <10⁻⁴** |
| LSTM | 0.10 | pooled | 40 | 98 → 322 | 93 → 318 | **+237 [+147, +312], p <10⁻⁴** | **+238 [+143, +317], p <10⁻⁴** | **+9.05 [+6.00, +12.42], p <10⁻⁴** | **−0.06 [−0.09, −0.01], p 0.027** |
| LSTM | 0.30 | pooled | 40 | 21 → 141 | 13 → 135 | **+115 [+60, +157], p <10⁻⁴** | **+115 [+64, +160], p <10⁻⁴** | **+11.26 [+3.10, +15.81], p <10⁻⁴** | **−0.06 [−0.08, −0.05], p <10⁻⁴** |
| LSTM | all | pooled | 160 | 158 → 390 | 154 → 385 | **+172 [+138, +208], p <10⁻⁴** | **+173 [+139, +211], p <10⁻⁴** | **+6.80 [+5.62, +8.48], p <10⁻⁴** | **−0.05 [−0.07, −0.02], p 0.013** |
| Transformer | 0.01 | pooled | 40 | 120 → 150 | 84 → 107 | −4 [−20, +5], p 0.248 | −14 [−36, +5], p 0.162 | 0 [−0.01, +0.02], p 0.889 | **−0.05 [−0.08, −0.01], p 0.010** |
| Transformer | 0.05 | pooled | 40 | 114 → 105 | 109 → 97 | −8 [−31, +15], p 0.501 | −10 [−36, +15], p 0.413 | −0.06 [−0.20, +0.07], p 0.361 | **+0.11 [+0.05, +0.14], p 0.005** |
| Transformer | 0.10 | pooled | 40 | 130 → 149 | 128 → 148 | +19 [−10, +47], p 0.162 | +21 [−8, +49], p 0.132 | **+0.48 [+0.13, +0.90], p 0.004** | **+0.06 [+0.05, +0.07], p <10⁻⁴** |
| Transformer | 0.30 | pooled | 38 | 58 → 92 | 49 → 89 | **+37 [+21, +53], p 0.0001** | **+43 [+24, +61], p 0.0002** | **+2.11 [+1.23, +3.16], p <10⁻⁴** | **+0.03 [+0.01, +0.05], p 0.018** |
| Transformer | all | pooled | 158 | 106 → 124 | 93 → 111 | +10 [0, +21], p 0.060 | +9 [−3, +22], p 0.136 | **+0.28 [+0.13, +0.51], p <10⁻⁴** | **+0.04 [+0.02, +0.05], p 0.0004** |

</details>

<details><summary>Intervals and p-values per rate × churn cell</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI], p | Δ \|bias\| [95% CI], p | Δ RMSE [95% CI], p | Δ Spearman [95% CI], p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM | 0.01 | 20% | 10 | 71 → 513 | 59 → 502 | +75 [−2, +1232], p 0.084 | +75 [−2, +1236], p 0.084 | **+3.47 [+0.12, +13.47], p 0.002** | **−0.18 [−0.22, −0.12], p 0.002** |
| LSTM | 0.01 | 40% | 10 | 136 → 189 | 132 → 182 | +15 [−23, +141], p 0.432 | +16 [−29, +144], p 0.432 | +0.08 [+0.01, +3.13], p 0.027 | **−0.23 [−0.27, −0.20], p 0.002** |
| LSTM | 0.01 | 60% | 10 | 291 → 514 | 290 → 514 | **+148 [+41, +462], p 0.006** | **+152 [+43, +461], p 0.006** | **+1.70 [+0.08, +4.30], p 0.002** | **−0.22 [−0.28, −0.16], p 0.002** |
| LSTM | 0.01 | 80% | 10 | 759 → 838 | 755 → 821 | +76 [−114, +259], p 0.432 | +62 [−160, +259], p 0.432 | +0.25 [−0.09, +2.22], p 0.275 | **−0.09 [−0.16, −0.05], p 0.004** |
| LSTM | 0.05 | 20% | 10 | 45 → 240 | 42 → 240 | **+191 [+121, +277], p 0.002** | **+194 [+123, +279], p 0.002** | **+11.75 [+8.83, +14.39], p 0.002** | +0.18 [−0.06, +0.41], p 0.432 |
| LSTM | 0.05 | 40% | 10 | 99 → 313 | 98 → 313 | **+211 [+135, +290], p 0.002** | **+211 [+136, +291], p 0.002** | **+8.05 [+5.16, +11.28], p 0.002** | +0.18 [−0.02, +0.38], p 0.064 |
| LSTM | 0.05 | 60% | 10 | 158 → 462 | 157 → 462 | **+264 [+141, +506], p 0.002** | **+264 [+141, +506], p 0.002** | **+8.80 [+5.86, +11.48], p 0.002** | **+0.32 [+0.26, +0.38], p 0.002** |
| LSTM | 0.05 | 80% | 10 | 497 → 1315 | 497 → 1314 | +803 [+160, +1508], p 0.027 | +802 [+160, +1508], p 0.027 | **+10.53 [+4.26, +16.51], p 0.004** | **+0.27 [+0.17, +0.36], p 0.002** |
| LSTM | 0.10 | 20% | 10 | 33 → 301 | 31 → 301 | **+249 [+178, +375], p 0.002** | **+249 [+178, +377], p 0.002** | **+17.52 [+11.38, +24.75], p 0.002** | **−0.07 [−0.08, −0.06], p 0.002** |
| LSTM | 0.10 | 40% | 10 | 58 → 375 | 57 → 375 | **+293 [+231, +465], p 0.002** | **+294 [+233, +465], p 0.002** | **+13.72 [+9.21, +22.21], p 0.002** | −0.09 [−0.13, +0.02], p 0.131 |
| LSTM | 0.10 | 60% | 10 | 69 → 307 | 61 → 301 | +224 [+44, +429], p 0.027 | +220 [+55, +444], p 0.037 | **+4.34 [+1.48, +9.07], p 0.006** | **−0.08 [−0.14, −0.02], p 0.020** |
| LSTM | 0.10 | 80% | 10 | 233 → 307 | 225 → 295 | +12 [−205, +349], p 0.922 | +20 [−208, +359], p 0.922 | +1.26 [−0.31, +5.78], p 0.232 | +0.15 [−0.06, +0.37], p 0.193 |
| LSTM | 0.30 | 20% | 10 | 19 → 268 | 17 → 268 | **+252 [+204, +291], p 0.002** | **+256 [+207, +295], p 0.002** | **+29.38 [+22.36, +33.68], p 0.002** | **−0.12 [−0.13, −0.10], p 0.002** |
| LSTM | 0.30 | 40% | 10 | 14 → 161 | 9 → 158 | **+146 [+27, +264], p 0.002** | **+148 [+29, +266], p 0.002** | **+11.86 [+2.56, +22.03], p 0.002** | **−0.08 [−0.11, −0.04], p 0.002** |
| LSTM | 0.30 | 60% | 10 | 18 → 64 | 9 → 57 | **+47 [+26, +64], p 0.002** | **+48 [+26, +74], p 0.002** | **+2.23 [+1.18, +3.27], p 0.002** | **−0.04 [−0.06, −0.02], p 0.002** |
| LSTM | 0.30 | 80% | 10 | 32 → 70 | 17 → 55 | +36 [−3, +85], p 0.232 | +39 [−9, +93], p 0.375 | +0.88 [−0.08, +3.17], p 0.131 | **−0.03 [−0.03, −0.02], p 0.002** |
| Transformer | 0.01 | 20% | 10 | 58 → 49 | 37 → 19 | −4 [−21, +2], p 0.193 | −11 [−43, +2], p 0.084 | 0 [−0.04, +0.01], p 1.000 | **−0.10 [−0.17, −0.03], p 0.020** |
| Transformer | 0.01 | 40% | 10 | 96 → 78 | 76 → 55 | −5 [−55, +6], p 0.232 | −10 [−60, +10], p 0.275 | −0.01 [−0.09, +0.01], p 0.432 | −0.02 [−0.10, +0.07], p 0.625 |
| Transformer | 0.01 | 60% | 10 | 129 → 127 | 102 → 79 | −3 [−40, +38], p 0.625 | −28 [−73, +47], p 0.193 | +0.12 [−0.02, +0.28], p 0.492 | −0.03 [−0.07, +0.06], p 0.846 |
| Transformer | 0.01 | 80% | 10 | 197 → 345 | 121 → 273 | +15 [−49, +610], p 0.846 | +27 [−79, +569], p 0.625 | +0.01 [−0.02, +1.15], p 0.770 | −0.06 [−0.19, +0.03], p 0.160 |
| Transformer | 0.05 | 20% | 10 | 75 → 91 | 72 → 90 | +29 [−24, +52], p 0.375 | +29 [−27, +58], p 0.322 | +0.07 [−0.49, +0.40], p 0.695 | **+0.20 [+0.09, +0.35], p 0.002** |
| Transformer | 0.05 | 40% | 10 | 110 → 103 | 108 → 101 | −6 [−62, +46], p 0.770 | −8 [−65, +46], p 0.770 | −0.06 [−0.58, +0.65], p 0.770 | +0.14 [−0.04, +0.17], p 0.084 |
| Transformer | 0.05 | 60% | 10 | 149 → 124 | 144 → 121 | −20 [−85, +39], p 0.492 | −25 [−85, +46], p 0.557 | −0.15 [−0.41, +0.10], p 0.193 | **+0.13 [+0.01, +0.20], p 0.020** |
| Transformer | 0.05 | 80% | 10 | 122 → 100 | 111 → 77 | −20 [−57, +15], p 0.275 | −37 [−80, +23], p 0.131 | −0.04 [−0.13, +0.07], p 0.275 | −0.13 [−0.31, +0.07], p 0.275 |
| Transformer | 0.10 | 20% | 10 | 58 → 102 | 56 → 101 | **+37 [+7, +87], p 0.020** | **+38 [+9, +88], p 0.020** | +0.55 [−0.09, +2.36], p 0.064 | **+0.07 [+0.04, +0.10], p 0.004** |
| Transformer | 0.10 | 40% | 10 | 94 → 91 | 91 → 89 | −4 [−47, +36], p 0.922 | −4 [−49, +45], p 0.922 | −0.19 [−0.66, +1.24], p 0.695 | +0.08 [−0.02, +0.09], p 0.084 |
| Transformer | 0.10 | 60% | 10 | 103 → 159 | 99 → 159 | +73 [−10, +114], p 0.105 | +75 [−5, +121], p 0.084 | +0.84 [+0.15, +1.56], p 0.037 | **+0.05 [+0.02, +0.08], p 0.010** |
| Transformer | 0.10 | 80% | 10 | 265 → 245 | 265 → 245 | −31 [−85, +51], p 0.432 | −31 [−85, +53], p 0.492 | +0.36 [+0.01, +1.14], p 0.049 | +0.05 [+0.01, +0.07], p 0.027 |
| Transformer | 0.30 | 20% | 10 | 36 → 71 | 24 → 69 | +35 [+6, +64], p 0.027 | +47 [+5, +83], p 0.027 | +3.93 [+0.33, +7.24], p 0.037 | +0.07 [−0.03, +0.09], p 0.084 |
| Transformer | 0.30 | 40% | 10 | 36 → 86 | 29 → 81 | **+47 [+22, +79], p 0.006** | **+54 [+19, +84], p 0.010** | **+2.79 [+1.01, +4.77], p 0.014** | +0.02 [−0.06, +0.09], p 0.375 |
| Transformer | 0.30 | 60% | 9 | 59 → 97 | 52 → 96 | +34 [−1, +81], p 0.055 | +42 [+2, +90], p 0.039 | **+1.76 [+0.79, +2.73], p 0.004** | +0.04 [−0.01, +0.09], p 0.129 |
| Transformer | 0.30 | 80% | 9 | 104 → 118 | 97 → 112 | +18 [−41, +68], p 0.570 | +22 [−53, +80], p 0.652 | +0.60 [−0.38, +2.31], p 0.203 | −0.01 [−0.04, +0.03], p 0.496 |

</details>

### 5. A k-means cluster label hurts

**Verdict: partly.** The label's effect depends on the model, the AR encoding and the rate.

- **LSTM `no_ar`:** no change in level at rates 0.01–0.05; worse at 0.10 (+34 MAPE) and
  0.30 (+53). It raises Spearman at 0.01–0.05 (+0.13, +0.31) and lowers it at 0.10–0.30
  (−0.07).
- **LSTM `ar_bounded`:** worse level from rate 0.05 up (+33 to +37 MAPE), better ranking at
  rate 0.01 only (+0.16).
- **Transformer `no_ar` and `ar_bounded`:** level mostly unchanged per cell; worse at rate
  0.30 without flags (+11 MAPE) and over the grid with flags (+12). Spearman rises on sparse
  panels (+0.09 to +0.13 at rate 0.01).
- **With `ar_unbounded`** the label helps both models (−74 and −31 MAPE over the grid),
  most at rates 0.05–0.30. It helps only by displacing the broken counters.

**LSTM `no_ar`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | +17 / +21 / **+0.12** | −2 / −2 / **+0.13** | −29 / −29 / **+0.16** | −30 / −30 / **+0.13** | +2 / +4 / **+0.13** |
| 0.05 | +1 / −2 / +0.19 | −17 / −18 / +0.25 | +10 / +10 / **+0.37** | +17 / +17 / **+0.35** | 0 / −1 / **+0.31** |
| 0.10 | +6 / +4 / **−0.09** | +27 / +28 / −0.10 | **+64** / **+75** / **−0.08** | +70 / +81 / +0.11 | **+34** / **+37** / **−0.07** |
| 0.30 | **+8** / **+8** / **−0.10** | **+39** / **+44** / **−0.07** | **+69** / **+79** / **−0.06** | **+110** / **+129** / **−0.03** | **+53** / **+60** / **−0.07** |
| all |  |  |  |  | **+24** / **+27** / **+0.06** |

**LSTM `ar_bounded`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | +9 / +12 / **+0.17** | −2 / −1 / **+0.21** | +55 / +56 / +0.12 | −126 / −134 / **+0.15** | +9 / +12 / **+0.16** |
| 0.05 | −9 / **−20** / **−0.06** | +6 / +7 / **−0.08** | **+37** / **+42** / **−0.09** | **+215** / **+255** / **−0.02** | **+33** / **+37** / **−0.06** |
| 0.10 | +1 / −1 / **−0.08** | **+20** / **+24** / **−0.05** | **+58** / **+71** / **−0.09** | **+80** / **+104** / −0.03 | **+37** / **+47** / **−0.06** |
| 0.30 | **+15** / **+17** / **−0.06** | **+32** / **+37** / **−0.06** | **+52** / **+62** / **−0.05** | **+48** / **+60** / **−0.02** | **+36** / **+43** / **−0.05** |
| all |  |  |  |  | **+30** / **+35** / **−0.03** |

**LSTM `ar_unbounded`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | −14 / −21 / **+0.07** | +600 / +601 / **+0.10** | +358 / +354 / **+0.17** | +551 / +551 / +0.11 | +213 / +216 / **+0.11** |
| 0.05 | −47 / −47 / +0.02 | −105 / −113 / −0.01 | −161 / −164 / +0.06 | −379 / −394 / +0.07 | **−133** / **−137** / +0.03 |
| 0.10 | **−110** / **−112** / −0.02 | **−196** / **−201** / 0 | −182 / −194 / +0.04 | −98 / −111 / +0.02 | **−166** / **−171** / 0 |
| 0.30 | **−183** / **−187** / +0.02 | −98 / −98 / +0.01 | −9 / −16 / 0 | −14 / −15 / 0 | **−73** / **−79** / +0.01 |
| all |  |  |  |  | **−74** / **−78** / **+0.03** |

**Transformer `no_ar`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | −4 / −5 / +0.07 | +43 / +47 / +0.11 | −10 / −13 / +0.11 | −19 / −37 / **+0.07** | −1 / −2 / **+0.09** |
| 0.05 | −7 / −8 / **+0.17** | −11 / −12 / +0.06 | −22 / −20 / **+0.11** | +60 / +48 / +0.05 | −3 / −5 / **+0.08** |
| 0.10 | 0 / 0 / +0.03 | −4 / −3 / **+0.05** | +9 / +10 / +0.04 | −59 / −61 / +0.03 | −8 / −8 / **+0.04** |
| 0.30 | −2 / 0 / **+0.04** | **+18** / **+25** / **+0.03** | +11 / +20 / +0.02 | +26 / +35 / +0.01 | **+11** / **+19** / **+0.03** |
| all |  |  |  |  | +1 / +2 / **+0.05** |

**Transformer `ar_bounded`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | +5 / +11 / **+0.14** | +6 / +3 / **+0.17** | −6 / −3 / **+0.11** | +15 / +24 / +0.03 | +5 / +7 / **+0.13** |
| 0.05 | +14 / +16 / **+0.05** | −1 / +3 / **−0.03** | +88 / **+96** / −0.04 | −3 / −4 / −0.03 | +18 / +23 / −0.02 |
| 0.10 | −7 / −8 / −0.01 | +9 / +16 / −0.03 | −3 / −3 / −0.04 | **+66** / **+86** / −0.04 | +16 / +20 / **−0.03** |
| 0.30 | −1 / −1 / **−0.02** | +3 / +7 / −0.01 | +17 / +22 / −0.02 | +32 / +39 / −0.01 | +12 / +18 / **−0.02** |
| all |  |  |  |  | **+12** / **+17** / 0 |

**Transformer `ar_unbounded`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | +16 / +28 / +0.08 | +7 / +12 / 0 | −14 / +3 / +0.05 | −27 / −49 / +0.08 | +1 / +6 / **+0.05** |
| 0.05 | −24 / −29 / −0.03 | −26 / −31 / −0.06 | −31 / −33 / −0.04 | −3 / −4 / +0.19 | **−22** / **−26** / −0.02 |
| 0.10 | **−55** / **−64** / 0 | −17 / −21 / −0.01 | **−70** / **−82** / +0.01 | **−94** / **−103** / −0.01 | **−60** / **−68** / 0 |
| 0.30 | **−38** / **−45** / +0.01 | **−46** / **−54** / +0.03 | −29 / −39 / +0.01 | −46 / −59 / +0.03 | **−39** / **−47** / **+0.02** |
| all |  |  |  |  | **−31** / **−35** / +0.01 |

<details><summary>Intervals and p-values by rate, churn pooled</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI], p | Δ \|bias\| [95% CI], p | Δ RMSE [95% CI], p | Δ Spearman [95% CI], p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM `no_ar` | 0.01 | pooled | 40 | 314 → 307 | 309 → 302 | +2 [−17, +24], p 0.755 | +4 [−19, +27], p 0.715 | +0.01 [−0.02, +0.05], p 0.545 | **+0.13 [+0.11, +0.16], p <10⁻⁴** |
| LSTM `no_ar` | 0.05 | pooled | 40 | 200 → 202 | 199 → 200 | 0 [−11, +15], p 0.973 | −1 [−13, +14], p 0.858 | −0.03 [−0.12, +0.06], p 0.452 | **+0.31 [+0.21, +0.37], p <10⁻⁴** |
| LSTM `no_ar` | 0.10 | pooled | 40 | 98 → 139 | 93 → 138 | **+34 [+16, +57], p <10⁻⁴** | **+37 [+17, +64], p <10⁻⁴** | **+0.45 [+0.32, +0.60], p <10⁻⁴** | **−0.07 [−0.09, −0.03], p 0.020** |
| LSTM `no_ar` | 0.30 | pooled | 40 | 21 → 78 | 13 → 78 | **+53 [+39, +68], p <10⁻⁴** | **+60 [+44, +78], p <10⁻⁴** | **+2.15 [+1.88, +2.44], p <10⁻⁴** | **−0.07 [−0.08, −0.06], p <10⁻⁴** |
| LSTM `no_ar` | all | pooled | 160 | 158 → 181 | 154 → 179 | **+24 [+15, +33], p <10⁻⁴** | **+27 [+17, +37], p <10⁻⁴** | **+0.41 [+0.27, +0.62], p <10⁻⁴** | **+0.06 [+0.03, +0.10], p <10⁻⁴** |
| LSTM `ar_bounded` | 0.01 | pooled | 40 | 243 → 221 | 232 → 211 | +9 [−15, +29], p 0.368 | +12 [−17, +34], p 0.390 | +0.02 [−0.02, +0.05], p 0.242 | **+0.16 [+0.13, +0.19], p <10⁻⁴** |
| LSTM `ar_bounded` | 0.05 | pooled | 40 | 58 → 126 | 46 → 121 | **+33 [+14, +71], p <10⁻⁴** | **+37 [+13, +88], p 0.0003** | **+0.22 [+0.12, +0.34], p <10⁻⁴** | **−0.06 [−0.09, −0.05], p <10⁻⁴** |
| LSTM `ar_bounded` | 0.10 | pooled | 40 | 36 → 78 | 24 → 76 | **+37 [+24, +51], p <10⁻⁴** | **+47 [+30, +64], p <10⁻⁴** | **+0.44 [+0.35, +0.54], p <10⁻⁴** | **−0.06 [−0.08, −0.05], p <10⁻⁴** |
| LSTM `ar_bounded` | 0.30 | pooled | 40 | 20 → 56 | 11 → 56 | **+36 [+28, +42], p <10⁻⁴** | **+43 [+34, +51], p <10⁻⁴** | **+1.67 [+1.45, +1.90], p <10⁻⁴** | **−0.05 [−0.06, −0.04], p <10⁻⁴** |
| LSTM `ar_bounded` | all | pooled | 160 | 89 → 120 | 78 → 116 | **+30 [+23, +36], p <10⁻⁴** | **+35 [+27, +44], p <10⁻⁴** | **+0.46 [+0.33, +0.60], p <10⁻⁴** | **−0.03 [−0.04, −0.01], p 0.008** |
| LSTM `ar_unbounded` | 0.01 | pooled | 40 | 513 → 1118 | 505 → 1107 | +213 [−10, +734], p 0.068 | +216 [−7, +740], p 0.070 | +1.97 [+0.02, +6.54], p 0.037 | **+0.11 [+0.07, +0.15], p <10⁻⁴** |
| LSTM `ar_unbounded` | 0.05 | pooled | 40 | 582 → 424 | 582 → 420 | **−133 [−215, −55], p 0.002** | **−137 [−221, −56], p 0.002** | −2.12 [−4.09, +0.39], p 0.100 | +0.03 [−0.02, +0.06], p 0.216 |
| LSTM `ar_unbounded` | 0.10 | pooled | 40 | 322 → 162 | 318 → 153 | **−166 [−258, −79], p 0.0001** | **−171 [−265, −79], p 0.0002** | **−3.85 [−6.69, −1.07], p 0.006** | 0 [−0.02, +0.03], p 0.889 |
| LSTM `ar_unbounded` | 0.30 | pooled | 40 | 141 → 61 | 135 → 48 | **−73 [−114, −31], p 0.0001** | **−79 [−118, −41], p <10⁻⁴** | **−5.31 [−11.50, −1.08], p 0.0005** | +0.01 [0, +0.02], p 0.242 |
| LSTM `ar_unbounded` | all | pooled | 160 | 390 → 441 | 385 → 432 | **−74 [−111, −34], p <10⁻⁴** | **−78 [−116, −39], p <10⁻⁴** | **−1.77 [−2.99, −0.56], p 0.003** | **+0.03 [+0.02, +0.05], p <10⁻⁴** |
| Transformer `no_ar` | 0.01 | pooled | 40 | 120 → 130 | 84 → 91 | −1 [−22, +22], p 0.900 | −2 [−38, +30], p 0.921 | +0.01 [−0.02, +0.07], p 0.590 | **+0.09 [+0.06, +0.12], p <10⁻⁴** |
| Transformer `no_ar` | 0.05 | pooled | 40 | 114 → 122 | 109 → 116 | −3 [−29, +28], p 0.847 | −5 [−34, +32], p 0.847 | 0 [−0.14, +0.14], p 0.952 | **+0.08 [+0.06, +0.11], p <10⁻⁴** |
| Transformer `no_ar` | 0.10 | pooled | 40 | 130 → 116 | 128 → 114 | −8 [−27, +9], p 0.361 | −8 [−29, +10], p 0.375 | +0.19 [0, +0.37], p 0.055 | **+0.04 [+0.02, +0.05], p <10⁻⁴** |
| Transformer `no_ar` | 0.30 | pooled | 40 | 56 → 67 | 48 → 64 | **+11 [+4, +20], p 0.006** | **+19 [+6, +30], p 0.005** | **+0.64 [+0.32, +1.09], p 0.0002** | **+0.03 [+0.02, +0.04], p <10⁻⁴** |
| Transformer `no_ar` | all | pooled | 160 | 105 → 109 | 92 → 96 | +1 [−8, +9], p 0.837 | +2 [−10, +13], p 0.749 | **+0.16 [+0.07, +0.25], p 0.0002** | **+0.05 [+0.04, +0.06], p <10⁻⁴** |
| Transformer `ar_bounded` | 0.01 | pooled | 40 | 103 → 119 | 63 → 79 | +5 [−6, +21], p 0.390 | +7 [−11, +33], p 0.420 | **+0.03 [0, +0.08], p 0.019** | **+0.13 [+0.09, +0.16], p <10⁻⁴** |
| Transformer `ar_bounded` | 0.05 | pooled | 40 | 78 → 96 | 68 → 88 | +18 [−1, +41], p 0.070 | +23 [−2, +49], p 0.077 | **+0.18 [+0.08, +0.30], p 0.0005** | −0.02 [−0.03, 0], p 0.042 |
| Transformer `ar_bounded` | 0.10 | pooled | 40 | 77 → 97 | 70 → 93 | +16 [+1, +34], p 0.038 | +20 [+1, +43], p 0.041 | **+0.43 [+0.22, +0.65], p 0.0010** | **−0.03 [−0.04, −0.02], p <10⁻⁴** |
| Transformer `ar_bounded` | 0.30 | pooled | 40 | 50 → 64 | 42 → 60 | +12 [−3, +26], p 0.106 | +18 [0, +34], p 0.053 | **+1.08 [+0.52, +1.54], p 0.0007** | **−0.02 [−0.04, −0.01], p 0.006** |
| Transformer `ar_bounded` | all | pooled | 160 | 77 → 94 | 61 → 80 | **+12 [+5, +21], p 0.0009** | **+17 [+7, +27], p 0.0008** | **+0.30 [+0.20, +0.43], p <10⁻⁴** | 0 [−0.01, +0.02], p 0.789 |
| Transformer `ar_unbounded` | 0.01 | pooled | 40 | 150 → 121 | 107 → 83 | +1 [−22, +12], p 0.900 | +6 [−29, +28], p 0.715 | +0.02 [−0.02, +0.04], p 0.227 | **+0.05 [+0.01, +0.09], p 0.017** |
| Transformer `ar_unbounded` | 0.05 | pooled | 40 | 105 → 87 | 97 → 76 | **−22 [−38, −8], p 0.005** | **−26 [−44, −9], p 0.007** | +0.01 [−0.12, +0.14], p 0.879 | −0.02 [−0.05, +0.02], p 0.170 |
| Transformer `ar_unbounded` | 0.10 | pooled | 40 | 149 → 88 | 148 → 81 | **−60 [−82, −39], p <10⁻⁴** | **−68 [−91, −43], p <10⁻⁴** | **−0.58 [−0.97, −0.21], p 0.002** | 0 [−0.02, +0.01], p 0.493 |
| Transformer `ar_unbounded` | 0.30 | pooled | 38 | 92 → 49 | 89 → 38 | **−39 [−53, −25], p <10⁻⁴** | **−47 [−64, −31], p <10⁻⁴** | **−1.42 [−2.69, −0.58], p 0.0003** | **+0.02 [+0.01, +0.04], p 0.003** |
| Transformer `ar_unbounded` | all | pooled | 158 | 124 → 87 | 111 → 70 | **−31 [−40, −21], p <10⁻⁴** | **−35 [−47, −24], p <10⁻⁴** | **−0.24 [−0.41, −0.10], p 0.0004** | +0.01 [0, +0.02], p 0.150 |

</details>

<details><summary>Intervals and p-values per rate × churn cell</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI], p | Δ \|bias\| [95% CI], p | Δ RMSE [95% CI], p | Δ Spearman [95% CI], p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM `no_ar` | 0.01 | 20% | 10 | 71 → 86 | 59 → 77 | +17 [−5, +36], p 0.160 | +21 [−9, +41], p 0.160 | +0.04 [−0.02, +0.10], p 0.193 | **+0.12 [+0.08, +0.17], p 0.002** |
| LSTM `no_ar` | 0.01 | 40% | 10 | 136 → 140 | 132 → 136 | −2 [−12, +22], p 1.000 | −2 [−14, +25], p 0.846 | 0 [−0.02, +0.05], p 0.922 | **+0.13 [+0.08, +0.17], p 0.004** |
| LSTM `no_ar` | 0.01 | 60% | 10 | 291 → 273 | 290 → 271 | −29 [−68, +35], p 0.695 | −29 [−70, +35], p 0.695 | −0.01 [−0.11, +0.09], p 0.625 | **+0.16 [+0.09, +0.21], p 0.004** |
| LSTM `no_ar` | 0.01 | 80% | 10 | 759 → 729 | 755 → 725 | −30 [−150, +109], p 0.846 | −30 [−156, +113], p 0.846 | 0 [−0.14, +0.14], p 1.000 | **+0.13 [+0.09, +0.18], p 0.002** |
| LSTM `no_ar` | 0.05 | 20% | 10 | 45 → 47 | 42 → 40 | +1 [−10, +16], p 1.000 | −2 [−15, +15], p 0.695 | −0.12 [−0.29, +0.13], p 0.432 | +0.19 [−0.01, +0.39], p 0.131 |
| LSTM `no_ar` | 0.05 | 40% | 10 | 99 → 82 | 98 → 81 | −17 [−29, −5], p 0.027 | −18 [−30, −6], p 0.027 | **−0.19 [−0.25, −0.04], p 0.010** | +0.25 [+0.03, +0.46], p 0.020 |
| LSTM `no_ar` | 0.05 | 60% | 10 | 158 → 172 | 157 → 172 | +10 [−11, +40], p 0.193 | +10 [−11, +40], p 0.193 | +0.03 [−0.10, +0.16], p 0.695 | **+0.37 [+0.28, +0.40], p 0.002** |
| LSTM `no_ar` | 0.05 | 80% | 10 | 497 → 506 | 497 → 506 | +17 [−66, +90], p 0.846 | +17 [−66, +90], p 0.846 | +0.13 [−0.12, +0.35], p 0.232 | **+0.35 [+0.29, +0.41], p 0.002** |
| LSTM `no_ar` | 0.10 | 20% | 10 | 33 → 39 | 31 → 35 | +6 [−2, +14], p 0.193 | +4 [−6, +15], p 0.557 | **+0.28 [+0.06, +0.48], p 0.010** | **−0.09 [−0.13, −0.05], p 0.002** |
| LSTM `no_ar` | 0.10 | 40% | 10 | 58 → 74 | 57 → 74 | +27 [−35, +46], p 0.084 | +28 [−35, +47], p 0.084 | +0.53 [−0.50, +0.77], p 0.105 | −0.10 [−0.16, +0.04], p 0.232 |
| LSTM `no_ar` | 0.10 | 60% | 10 | 69 → 132 | 61 → 132 | **+64 [+29, +98], p 0.006** | **+75 [+33, +107], p 0.006** | **+0.65 [+0.38, +1.01], p 0.002** | **−0.08 [−0.14, −0.04], p 0.002** |
| LSTM `no_ar` | 0.10 | 80% | 10 | 233 → 309 | 225 → 309 | +70 [−1, +147], p 0.064 | +81 [−1, +163], p 0.064 | +0.39 [+0.07, +0.81], p 0.020 | +0.11 [−0.06, +0.27], p 0.232 |
| LSTM `no_ar` | 0.30 | 20% | 10 | 19 → 28 | 17 → 27 | **+8 [+4, +14], p 0.004** | **+8 [+4, +17], p 0.004** | **+1.41 [+0.91, +1.78], p 0.002** | **−0.10 [−0.13, −0.09], p 0.002** |
| LSTM `no_ar` | 0.30 | 40% | 10 | 14 → 53 | 9 → 53 | **+39 [+33, +45], p 0.002** | **+44 [+39, +50], p 0.002** | **+2.44 [+1.98, +2.96], p 0.002** | **−0.07 [−0.08, −0.06], p 0.002** |
| LSTM `no_ar` | 0.30 | 60% | 10 | 18 → 90 | 9 → 90 | **+69 [+63, +85], p 0.002** | **+79 [+73, +90], p 0.002** | **+2.69 [+2.05, +3.18], p 0.002** | **−0.06 [−0.08, −0.05], p 0.002** |
| LSTM `no_ar` | 0.30 | 80% | 10 | 32 → 142 | 17 → 142 | **+110 [+78, +147], p 0.002** | **+129 [+92, +161], p 0.002** | **+2.02 [+1.23, +2.79], p 0.002** | **−0.03 [−0.04, −0.02], p 0.002** |
| LSTM `ar_bounded` | 0.01 | 20% | 10 | 65 → 72 | 47 → 58 | +9 [−13, +28], p 0.275 | +12 [−19, +43], p 0.322 | +0.03 [−0.04, +0.08], p 0.375 | **+0.17 [+0.08, +0.25], p 0.006** |
| LSTM `ar_bounded` | 0.01 | 40% | 10 | 121 → 121 | 114 → 116 | −2 [−25, +30], p 0.922 | −1 [−29, +33], p 1.000 | +0.01 [−0.04, +0.07], p 0.557 | **+0.21 [+0.14, +0.25], p 0.002** |
| LSTM `ar_bounded` | 0.01 | 60% | 10 | 196 → 245 | 190 → 241 | +55 [−4, +101], p 0.064 | +56 [−10, +109], p 0.064 | +0.09 [0, +0.15], p 0.064 | +0.12 [0, +0.21], p 0.037 |
| LSTM `ar_bounded` | 0.01 | 80% | 10 | 591 → 445 | 576 → 429 | −126 [−349, +58], p 0.375 | −134 [−349, +71], p 0.375 | −0.07 [−0.20, +0.05], p 0.322 | **+0.15 [+0.06, +0.22], p 0.010** |
| LSTM `ar_bounded` | 0.05 | 20% | 10 | 48 → 39 | 47 → 29 | −9 [−16, −3], p 0.027 | **−20 [−26, −9], p 0.006** | −0.10 [−0.18, +0.01], p 0.105 | **−0.06 [−0.08, −0.03], p 0.004** |
| LSTM `ar_bounded` | 0.05 | 40% | 10 | 56 → 64 | 52 → 61 | +6 [−4, +24], p 0.232 | +7 [−6, +26], p 0.275 | +0.09 [0, +0.24], p 0.037 | **−0.08 [−0.13, −0.05], p 0.004** |
| LSTM `ar_bounded` | 0.05 | 60% | 10 | 67 → 110 | 61 → 108 | **+37 [+26, +75], p 0.002** | **+42 [+29, +79], p 0.002** | **+0.35 [+0.26, +0.48], p 0.002** | **−0.09 [−0.14, −0.04], p 0.004** |
| LSTM `ar_bounded` | 0.05 | 80% | 10 | 60 → 290 | 27 → 287 | **+215 [+116, +353], p 0.002** | **+255 [+146, +381], p 0.002** | **+0.57 [+0.29, +0.91], p 0.002** | **−0.02 [−0.08, 0], p 0.014** |
| LSTM `ar_bounded` | 0.10 | 20% | 10 | 32 → 33 | 30 → 28 | +1 [−9, +10], p 0.770 | −1 [−14, +11], p 0.846 | +0.15 [−0.05, +0.34], p 0.275 | **−0.08 [−0.12, −0.05], p 0.002** |
| LSTM `ar_bounded` | 0.10 | 40% | 10 | 34 → 53 | 28 → 52 | **+20 [+10, +29], p 0.006** | **+24 [+11, +36], p 0.006** | **+0.38 [+0.20, +0.53], p 0.002** | **−0.05 [−0.08, −0.04], p 0.002** |
| LSTM `ar_bounded` | 0.10 | 60% | 10 | 33 → 93 | 21 → 92 | **+58 [+41, +78], p 0.002** | **+71 [+54, +89], p 0.002** | **+0.66 [+0.52, +0.85], p 0.002** | **−0.09 [−0.14, −0.05], p 0.002** |
| LSTM `ar_bounded` | 0.10 | 80% | 10 | 45 → 134 | 15 → 131 | **+80 [+53, +142], p 0.002** | **+104 [+77, +170], p 0.002** | **+0.54 [+0.41, +0.80], p 0.002** | −0.03 [−0.04, 0], p 0.027 |
| LSTM `ar_bounded` | 0.30 | 20% | 10 | 12 → 26 | 10 → 25 | **+15 [+10, +18], p 0.002** | **+17 [+10, +19], p 0.002** | **+1.48 [+1.09, +1.82], p 0.002** | **−0.06 [−0.08, −0.05], p 0.002** |
| LSTM `ar_bounded` | 0.30 | 40% | 10 | 13 → 45 | 6 → 44 | **+32 [+25, +39], p 0.002** | **+37 [+29, +45], p 0.002** | **+2.03 [+1.64, +2.40], p 0.002** | **−0.06 [−0.08, −0.05], p 0.002** |
| LSTM `ar_bounded` | 0.30 | 60% | 10 | 19 → 71 | 10 → 71 | **+52 [+41, +63], p 0.002** | **+62 [+47, +74], p 0.002** | **+2.11 [+1.76, +2.52], p 0.002** | **−0.05 [−0.05, −0.04], p 0.002** |
| LSTM `ar_bounded` | 0.30 | 80% | 10 | 35 → 84 | 19 → 83 | **+48 [+28, +67], p 0.002** | **+60 [+41, +86], p 0.002** | **+1.06 [+0.78, +1.36], p 0.002** | **−0.02 [−0.03, −0.01], p 0.002** |
| LSTM `ar_unbounded` | 0.01 | 20% | 10 | 513 → 250 | 502 → 235 | −14 [−931, +68], p 0.695 | −21 [−938, +81], p 0.695 | −0.36 [−7.69, +3.37], p 0.846 | **+0.07 [+0.03, +0.12], p 0.014** |
| LSTM `ar_unbounded` | 0.01 | 40% | 10 | 189 → 1304 | 182 → 1300 | +600 [+80, +2282], p 0.020 | +601 [+80, +2289], p 0.020 | **+7.92 [+1.82, +15.96], p 0.010** | **+0.10 [+0.03, +0.19], p 0.014** |
| LSTM `ar_unbounded` | 0.01 | 60% | 10 | 514 → 904 | 514 → 900 | +358 [−182, +941], p 0.432 | +354 [−200, +940], p 0.432 | +0.61 [−2.62, +6.74], p 0.770 | **+0.17 [+0.06, +0.28], p 0.006** |
| LSTM `ar_unbounded` | 0.01 | 80% | 10 | 838 → 2015 | 821 → 1993 | +551 [−108, +2672], p 0.492 | +551 [−128, +2656], p 0.432 | +1.23 [−0.24, +8.68], p 0.232 | +0.11 [0, +0.21], p 0.064 |
| LSTM `ar_unbounded` | 0.05 | 20% | 10 | 240 → 189 | 240 → 188 | −47 [−120, +13], p 0.160 | −47 [−121, +13], p 0.160 | −2.90 [−5.95, −0.09], p 0.049 | +0.02 [−0.01, +0.05], p 0.232 |
| LSTM `ar_unbounded` | 0.05 | 40% | 10 | 313 → 229 | 313 → 221 | −105 [−239, +47], p 0.275 | −113 [−252, +47], p 0.232 | −1.45 [−7.10, +5.22], p 0.770 | −0.01 [−0.10, +0.06], p 0.846 |
| LSTM `ar_unbounded` | 0.05 | 60% | 10 | 462 → 359 | 462 → 357 | −161 [−271, +193], p 0.160 | −164 [−278, +190], p 0.160 | −3.28 [−7.79, +3.29], p 0.275 | +0.06 [−0.09, +0.10], p 0.922 |
| LSTM `ar_unbounded` | 0.05 | 80% | 10 | 1315 → 919 | 1314 → 914 | −379 [−1056, +175], p 0.131 | −394 [−1068, +178], p 0.105 | −1.22 [−6.05, +5.00], p 0.695 | +0.07 [−0.03, +0.16], p 0.232 |
| LSTM `ar_unbounded` | 0.10 | 20% | 10 | 301 → 145 | 301 → 141 | **−110 [−312, −48], p 0.002** | **−112 [−326, −48], p 0.002** | −6.34 [−15.59, −0.10], p 0.049 | −0.02 [−0.07, +0.02], p 0.492 |
| LSTM `ar_unbounded` | 0.10 | 40% | 10 | 375 → 174 | 375 → 167 | **−196 [−318, −79], p 0.004** | **−201 [−334, −79], p 0.004** | −8.36 [−14.84, −1.55], p 0.020 | 0 [−0.07, +0.05], p 0.922 |
| LSTM `ar_unbounded` | 0.10 | 60% | 10 | 307 → 142 | 301 → 133 | −182 [−417, +85], p 0.193 | −194 [−432, +75], p 0.193 | −0.63 [−7.44, +4.63], p 0.695 | +0.04 [−0.05, +0.11], p 0.557 |
| LSTM `ar_unbounded` | 0.10 | 80% | 10 | 307 → 189 | 295 → 171 | −98 [−389, +154], p 0.375 | −111 [−419, +172], p 0.557 | −0.49 [−3.02, +3.42], p 0.846 | +0.02 [−0.04, +0.07], p 0.695 |
| LSTM `ar_unbounded` | 0.30 | 20% | 10 | 268 → 80 | 268 → 76 | **−183 [−233, −134], p 0.002** | **−187 [−238, −136], p 0.002** | **−21.33 [−28.92, −11.76], p 0.004** | +0.02 [−0.01, +0.05], p 0.193 |
| LSTM `ar_unbounded` | 0.30 | 40% | 10 | 161 → 62 | 158 → 50 | −98 [−195, +2], p 0.084 | −98 [−206, −4], p 0.049 | −6.79 [−14.69, +0.73], p 0.105 | +0.01 [−0.03, +0.06], p 0.846 |
| LSTM `ar_unbounded` | 0.30 | 60% | 10 | 64 → 54 | 57 → 39 | −9 [−33, +10], p 0.432 | −16 [−49, +13], p 0.232 | −0.56 [−1.66, +0.48], p 0.232 | 0 [−0.02, +0.02], p 0.922 |
| LSTM `ar_unbounded` | 0.30 | 80% | 10 | 70 → 47 | 55 → 26 | −14 [−73, +13], p 0.625 | −15 [−83, +13], p 0.557 | −0.02 [−2.68, +0.45], p 1.000 | 0 [−0.01, +0.01], p 0.492 |
| Transformer `no_ar` | 0.01 | 20% | 10 | 58 → 55 | 37 → 35 | −4 [−17, +10], p 0.770 | −5 [−40, +30], p 0.846 | 0 [−0.04, +0.13], p 1.000 | +0.07 [+0.02, +0.13], p 0.027 |
| Transformer `no_ar` | 0.01 | 40% | 10 | 96 → 148 | 76 → 134 | +43 [−37, +153], p 0.275 | +47 [−53, +167], p 0.232 | +0.15 [−0.07, +0.41], p 0.275 | +0.11 [+0.03, +0.20], p 0.027 |
| Transformer `no_ar` | 0.01 | 60% | 10 | 129 → 120 | 102 → 87 | −10 [−60, +38], p 0.625 | −13 [−71, +38], p 0.557 | 0 [−0.04, +0.08], p 1.000 | +0.11 [+0.03, +0.17], p 0.027 |
| Transformer `no_ar` | 0.01 | 80% | 10 | 197 → 195 | 121 → 109 | −19 [−77, +93], p 0.625 | −37 [−141, +121], p 0.625 | 0 [−0.05, +0.18], p 1.000 | **+0.07 [+0.05, +0.16], p 0.002** |
| Transformer `no_ar` | 0.05 | 20% | 10 | 75 → 67 | 72 → 62 | −7 [−35, +19], p 0.432 | −8 [−44, +20], p 0.557 | −0.25 [−0.58, +0.02], p 0.084 | **+0.17 [+0.04, +0.32], p 0.006** |
| Transformer `no_ar` | 0.05 | 40% | 10 | 110 → 107 | 108 → 105 | −11 [−51, +45], p 0.695 | −12 [−55, +48], p 0.695 | 0 [−0.15, +0.45], p 1.000 | +0.06 [+0.02, +0.11], p 0.027 |
| Transformer `no_ar` | 0.05 | 60% | 10 | 149 → 134 | 144 → 131 | −22 [−95, +60], p 0.557 | −20 [−107, +66], p 0.695 | +0.04 [−0.32, +0.32], p 1.000 | **+0.11 [+0.06, +0.16], p 0.002** |
| Transformer `no_ar` | 0.05 | 80% | 10 | 122 → 181 | 111 → 167 | +60 [−53, +175], p 0.432 | +48 [−77, +188], p 0.432 | +0.17 [−0.11, +0.47], p 0.432 | +0.05 [−0.05, +0.11], p 0.064 |
| Transformer `no_ar` | 0.10 | 20% | 10 | 58 → 57 | 56 → 54 | 0 [−28, +21], p 1.000 | 0 [−34, +26], p 1.000 | −0.17 [−0.79, +0.44], p 0.557 | +0.03 [−0.02, +0.09], p 0.275 |
| Transformer `no_ar` | 0.10 | 40% | 10 | 94 → 95 | 91 → 93 | −4 [−17, +29], p 0.695 | −3 [−17, +31], p 0.846 | +0.31 [−0.02, +0.59], p 0.064 | **+0.05 [+0.03, +0.07], p 0.004** |
| Transformer `no_ar` | 0.10 | 60% | 10 | 103 → 109 | 99 → 106 | +9 [−40, +49], p 0.770 | +10 [−33, +58], p 0.625 | +0.36 [−0.19, +0.69], p 0.160 | +0.04 [+0.01, +0.07], p 0.027 |
| Transformer `no_ar` | 0.10 | 80% | 10 | 265 → 203 | 265 → 201 | −59 [−139, +9], p 0.064 | −61 [−141, +8], p 0.064 | +0.14 [−0.09, +0.58], p 0.160 | +0.03 [−0.03, +0.05], p 0.105 |
| Transformer `no_ar` | 0.30 | 20% | 10 | 36 → 30 | 24 → 21 | −2 [−28, +6], p 0.695 | 0 [−28, +20], p 0.922 | +0.14 [−1.94, +0.88], p 0.770 | **+0.04 [+0.03, +0.06], p 0.002** |
| Transformer `no_ar` | 0.30 | 40% | 10 | 36 → 55 | 29 → 53 | **+18 [+7, +30], p 0.010** | **+25 [+9, +45], p 0.014** | +1.19 [+0.17, +2.12], p 0.027 | **+0.03 [+0.01, +0.05], p 0.010** |
| Transformer `no_ar` | 0.30 | 60% | 10 | 57 → 65 | 50 → 65 | +11 [−22, +36], p 0.322 | +20 [−19, +47], p 0.322 | +0.58 [−0.22, +1.38], p 0.160 | +0.02 [+0.01, +0.04], p 0.020 |
| Transformer `no_ar` | 0.30 | 80% | 10 | 96 → 120 | 88 → 119 | +26 [−1, +52], p 0.064 | +35 [−1, +65], p 0.064 | **+0.74 [+0.40, +1.43], p 0.002** | +0.01 [0, +0.03], p 0.084 |
| Transformer `ar_bounded` | 0.01 | 20% | 10 | 62 → 73 | 44 → 59 | +5 [−24, +50], p 0.557 | +11 [−30, +53], p 0.557 | +0.03 [−0.06, +0.17], p 0.492 | **+0.14 [+0.08, +0.19], p 0.002** |
| Transformer `ar_bounded` | 0.01 | 40% | 10 | 75 → 102 | 58 → 80 | +6 [−21, +88], p 0.770 | +3 [−38, +93], p 0.922 | +0.03 [−0.03, +0.22], p 0.432 | **+0.17 [+0.14, +0.21], p 0.002** |
| Transformer `ar_bounded` | 0.01 | 60% | 10 | 98 → 103 | 60 → 67 | −6 [−26, +47], p 0.432 | −3 [−51, +72], p 0.922 | 0 [−0.01, +0.12], p 0.492 | **+0.11 [+0.04, +0.17], p 0.006** |
| Transformer `ar_bounded` | 0.01 | 80% | 10 | 176 → 196 | 90 → 112 | +15 [−20, +81], p 0.193 | +24 [−32, +88], p 0.432 | **+0.08 [+0.01, +0.14], p 0.004** | +0.03 [−0.11, +0.16], p 0.432 |
| Transformer `ar_bounded` | 0.05 | 20% | 10 | 68 → 79 | 63 → 77 | +14 [−8, +25], p 0.193 | +16 [−8, +35], p 0.193 | +0.10 [−0.11, +0.28], p 0.375 | **+0.05 [+0.02, +0.10], p 0.014** |
| Transformer `ar_bounded` | 0.05 | 40% | 10 | 65 → 56 | 54 → 48 | −1 [−47, +21], p 0.846 | +3 [−52, +33], p 0.922 | +0.10 [−0.15, +0.34], p 0.432 | **−0.03 [−0.06, −0.01], p 0.010** |
| Transformer `ar_bounded` | 0.05 | 60% | 10 | 77 → 155 | 69 → 152 | +88 [+16, +124], p 0.020 | **+96 [+12, +142], p 0.014** | **+0.58 [+0.29, +0.79], p 0.006** | −0.04 [−0.08, −0.02], p 0.020 |
| Transformer `ar_bounded` | 0.05 | 80% | 10 | 102 → 94 | 84 → 73 | −3 [−82, +71], p 0.922 | −4 [−107, +88], p 0.922 | +0.07 [−0.02, +0.16], p 0.105 | −0.03 [−0.06, 0], p 0.020 |
| Transformer `ar_bounded` | 0.10 | 20% | 10 | 52 → 45 | 45 → 38 | −7 [−34, +19], p 0.625 | −8 [−40, +24], p 0.770 | −0.01 [−0.49, +0.42], p 1.000 | −0.01 [−0.12, 0], p 0.105 |
| Transformer `ar_bounded` | 0.10 | 40% | 10 | 41 → 58 | 33 → 54 | +9 [+3, +42], p 0.049 | +16 [+3, +53], p 0.049 | +0.53 [+0.09, +1.00], p 0.027 | −0.03 [−0.06, +0.21], p 0.084 |
| Transformer `ar_bounded` | 0.10 | 60% | 10 | 115 → 108 | 113 → 106 | −3 [−82, +63], p 0.846 | −3 [−82, +67], p 0.846 | +0.45 [−0.45, +0.99], p 0.322 | −0.04 [−0.06, 0], p 0.020 |
| Transformer `ar_bounded` | 0.10 | 80% | 10 | 102 → 177 | 90 → 175 | **+66 [+32, +134], p 0.010** | **+86 [+33, +140], p 0.014** | **+0.73 [+0.42, +1.06], p 0.002** | −0.04 [−0.09, −0.01], p 0.027 |
| Transformer `ar_bounded` | 0.30 | 20% | 10 | 37 → 32 | 27 → 25 | −1 [−20, +10], p 0.846 | −1 [−24, +23], p 1.000 | +0.60 [−1.44, +1.44], p 0.922 | **−0.02 [−0.14, 0], p 0.014** |
| Transformer `ar_bounded` | 0.30 | 40% | 10 | 42 → 47 | 35 → 45 | +3 [−19, +27], p 0.625 | +7 [−25, +39], p 0.492 | +0.95 [−0.05, +2.18], p 0.064 | −0.01 [−0.05, +0.16], p 0.846 |
| Transformer `ar_bounded` | 0.30 | 60% | 10 | 67 → 85 | 62 → 83 | +17 [−27, +65], p 0.492 | +22 [−25, +72], p 0.322 | +1.20 [−0.05, +2.94], p 0.064 | −0.02 [−0.05, +0.01], p 0.131 |
| Transformer `ar_bounded` | 0.30 | 80% | 10 | 54 → 91 | 46 → 87 | +32 [+1, +72], p 0.049 | +39 [+1, +84], p 0.049 | **+1.25 [+0.55, +2.04], p 0.004** | −0.01 [−0.05, +0.01], p 0.232 |
| Transformer `ar_unbounded` | 0.01 | 20% | 10 | 49 → 65 | 19 → 46 | +16 [+2, +28], p 0.027 | +28 [−5, +59], p 0.084 | +0.04 [+0.01, +0.08], p 0.027 | +0.08 [−0.01, +0.16], p 0.084 |
| Transformer `ar_unbounded` | 0.01 | 40% | 10 | 78 → 83 | 55 → 60 | +7 [−33, +44], p 0.695 | +12 [−55, +66], p 0.922 | +0.03 [−0.04, +0.10], p 0.232 | 0 [−0.06, +0.07], p 1.000 |
| Transformer `ar_unbounded` | 0.01 | 60% | 10 | 127 → 120 | 79 → 83 | −14 [−64, +53], p 0.846 | +3 [−61, +65], p 0.922 | −0.08 [−0.29, +0.20], p 0.846 | +0.05 [−0.06, +0.13], p 0.322 |
| Transformer `ar_unbounded` | 0.01 | 80% | 10 | 345 → 216 | 273 → 145 | −27 [−523, +16], p 0.160 | −49 [−489, +33], p 0.375 | −0.01 [−0.96, +0.05], p 0.846 | +0.08 [−0.02, +0.22], p 0.105 |
| Transformer `ar_unbounded` | 0.05 | 20% | 10 | 91 → 69 | 90 → 64 | −24 [−48, +6], p 0.105 | −29 [−56, +8], p 0.131 | −0.03 [−0.41, +0.45], p 0.922 | −0.03 [−0.15, −0.01], p 0.020 |
| Transformer `ar_unbounded` | 0.05 | 40% | 10 | 103 → 78 | 101 → 68 | −26 [−52, +1], p 0.084 | −31 [−65, −5], p 0.037 | −0.04 [−0.77, +0.24], p 0.846 | −0.06 [−0.20, +0.12], p 0.084 |
| Transformer `ar_unbounded` | 0.05 | 60% | 10 | 124 → 103 | 121 → 98 | −31 [−74, +50], p 0.232 | −33 [−78, +51], p 0.232 | +0.05 [−0.19, +0.83], p 0.695 | −0.04 [−0.16, +0.04], p 0.193 |
| Transformer `ar_unbounded` | 0.05 | 80% | 10 | 100 → 97 | 77 → 72 | −3 [−83, +92], p 0.625 | −4 [−95, +92], p 0.770 | −0.01 [−0.20, +0.51], p 1.000 | +0.19 [0, +0.36], p 0.049 |
| Transformer `ar_unbounded` | 0.10 | 20% | 10 | 102 → 47 | 101 → 39 | **−55 [−91, −16], p 0.010** | **−64 [−105, −16], p 0.010** | −0.98 [−2.50, +0.05], p 0.064 | 0 [−0.13, +0.04], p 1.000 |
| Transformer `ar_unbounded` | 0.10 | 40% | 10 | 91 → 77 | 89 → 72 | −17 [−59, +32], p 0.432 | −21 [−69, +33], p 0.375 | −0.08 [−1.30, +0.88], p 0.846 | −0.01 [−0.11, +0.08], p 0.275 |
| Transformer `ar_unbounded` | 0.10 | 60% | 10 | 159 → 76 | 159 → 70 | **−70 [−126, −36], p 0.002** | **−82 [−132, −42], p 0.002** | −0.71 [−1.37, −0.17], p 0.027 | +0.01 [−0.08, +0.05], p 0.846 |
| Transformer `ar_unbounded` | 0.10 | 80% | 10 | 245 → 150 | 245 → 142 | **−94 [−147, −33], p 0.002** | **−103 [−160, −35], p 0.002** | −0.39 [−1.04, +0.01], p 0.084 | −0.01 [−0.07, +0.02], p 0.695 |
| Transformer `ar_unbounded` | 0.30 | 20% | 10 | 71 → 37 | 69 → 30 | **−38 [−72, −3], p 0.014** | **−45 [−87, −6], p 0.010** | −3.41 [−6.50, +0.11], p 0.084 | +0.01 [0, +0.11], p 0.131 |
| Transformer `ar_unbounded` | 0.30 | 40% | 10 | 86 → 41 | 81 → 28 | **−46 [−73, −19], p 0.004** | **−54 [−85, −23], p 0.006** | **−2.31 [−4.15, −0.70], p 0.014** | +0.03 [−0.02, +0.13], p 0.193 |
| Transformer `ar_unbounded` | 0.30 | 60% | 9 | 97 → 62 | 96 → 51 | −29 [−74, +1], p 0.074 | −39 [−98, +1], p 0.055 | −1.03 [−2.25, +0.20], p 0.129 | +0.01 [−0.02, +0.04], p 0.734 |
| Transformer `ar_unbounded` | 0.30 | 80% | 9 | 118 → 58 | 112 → 45 | −46 [−130, −1], p 0.039 | −59 [−152, −1], p 0.039 | −0.17 [−2.79, +0.83], p 0.652 | +0.03 [+0.01, +0.05], p 0.020 |

</details>

### 6. One architecture is better overall

**Verdict: not supported.** Over the grid the two architectures are level with `no_ar`
(−4 MAPE, p = 0.62), and the LSTM is slightly better with flags (+9 for the Transformer,
p = 0.02). Both hide opposite effects by rate:

- **Rate 0.01:** the Transformer is better with every arm (−69 to −493 MAPE points), and
  mostly ranks customers better too.
- **Rates 0.10–0.30:** the LSTM is better with `no_ar` and `ar_bounded` (+27 to +34 MAPE
  for the Transformer) and ranks customers better at rate 0.30.
- **With unbounded counters** the Transformer is better at every rate but 0.30: it
  tolerates the counters and the LSTM does not.

Δ = Transformer minus LSTM on the same panels.

**`no_ar`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | −13 / **−22** / +0.04 | −45 / −58 / +0.01 | **−157** / **−181** / +0.06 | **−521** / **−579** / **+0.10** | **−123** / **−152** / **+0.05** |
| 0.05 | **+29** / +31 / +0.05 | +12 / +12 / +0.20 | −9 / −12 / **+0.34** | **−370** / **−384** / **+0.33** | −47 / −52 / **+0.30** |
| 0.10 | +25 / +25 / **−0.09** | +41 / +37 / −0.10 | +27 / +30 / **−0.07** | +25 / +43 / +0.14 | **+32** / **+34** / **−0.07** |
| 0.30 | **+11** / +1 / **−0.14** | **+20** / **+20** / **−0.14** | **+35** / **+39** / **−0.12** | **+55** / **+66** / **−0.07** | **+28** / **+28** / **−0.12** |
| all |  |  |  |  | −4 / −10 / +0.01 |

**`no_ar` + `kmeans_8`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **−31** / **−42** / 0 | −17 / −24 / −0.01 | **−154** / **−194** / 0 | **−488** / **−565** / +0.04 | **−125** / **−150** / +0.01 |
| 0.05 | +21 / +23 / +0.03 | +23 / +23 / +0.01 | −43 / −43 / **+0.07** | **−330** / **−340** / +0.03 | **−40** / **−42** / **+0.04** |
| 0.10 | +20 / +18 / +0.04 | +22 / +23 / +0.05 | −23 / −24 / **+0.05** | **−96** / **−97** / +0.05 | −11 / −13 / **+0.05** |
| 0.30 | −1 / −8 / +0.01 | +3 / −1 / **−0.04** | **−25** / **−26** / **−0.03** | −24 / −24 / **−0.02** | **−7** / **−12** / **−0.02** |
| all |  |  |  |  | **−31** / **−37** / **+0.01** |

**`ar_bounded`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | −2 / +2 / **+0.07** | **−48** / **−61** / **+0.07** | **−81** / **−115** / **+0.10** | **−345** / **−415** / **+0.16** | **−69** / **−94** / **+0.09** |
| 0.05 | +22 / +23 / −0.05 | +1 / −7 / +0.01 | +6 / 0 / 0 | +31 / +53 / +0.01 | **+13** / +14 / 0 |
| 0.10 | +14 / +14 / **−0.02** | +6 / +3 / 0 | **+74** / **+86** / **−0.01** | **+48** / **+66** / 0 | **+34** / **+40** / **−0.01** |
| 0.30 | **+22** / +17 / **−0.04** | **+28** / **+29** / **−0.06** | **+43** / **+51** / **−0.04** | **+20** / +28 / **−0.02** | **+27** / **+29** / **−0.04** |
| all |  |  |  |  | **+9** / +8 / 0 |

**`ar_bounded` + `kmeans_8`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | −9 / −8 / +0.03 | −46 / −65 / +0.06 | **−145** / **−184** / **+0.10** | **−228** / **−302** / +0.05 | **−85** / **−113** / **+0.06** |
| 0.05 | **+42** / **+49** / **+0.06** | −6 / −12 / **+0.07** | +41 / +44 / **+0.04** | **−195** / **−215** / +0.01 | −2 / −4 / **+0.05** |
| 0.10 | +9 / +9 / +0.05 | +4 / +2 / +0.01 | +6 / +5 / +0.04 | **+41** / +42 / −0.02 | **+13** / +14 / **+0.02** |
| 0.30 | +6 / 0 / −0.01 | +6 / +6 / −0.01 | +8 / +7 / −0.01 | −1 / −2 / −0.01 | +5 / +2 / **−0.01** |
| all |  |  |  |  | −6 / −10 / **+0.02** |

**`ar_unbounded`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **−140** / **−183** / **+0.11** | **−96** / **−122** / **+0.21** | **−335** / **−382** / **+0.25** | −570 / −617 / +0.15 | **−269** / **−301** / **+0.18** |
| 0.05 | **−131** / **−131** / **+0.08** | **−211** / **−213** / +0.15 | **−285** / **−287** / **+0.14** | **−1188** / **−1213** / −0.05 | **−264** / **−267** / **+0.10** |
| 0.10 | **−193** / **−193** / **+0.05** | **−249** / **−252** / +0.08 | −159 / −142 / +0.05 | +63 / +64 / +0.02 | **−180** / **−176** / **+0.05** |
| 0.30 | **−193** / **−195** / +0.04 | −64 / −59 / −0.04 | +31 / +40 / −0.02 | +61 / +76 / **−0.04** | −47 / −44 / −0.02 |
| all |  |  |  |  | **−180** / **−189** / **+0.07** |

**`ar_unbounded` + `kmeans_8`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **−111** / **−128** / **+0.11** | **−750** / **−759** / +0.11 | **−699** / **−743** / +0.13 | **−1680** / **−1763** / +0.10 | **−493** / **−529** / **+0.12** |
| 0.05 | **−131** / **−133** / +0.03 | **−122** / −131 / +0.10 | −169 / −171 / +0.08 | **−871** / **−882** / +0.07 | **−189** / **−194** / **+0.06** |
| 0.10 | **−106** / **−107** / +0.06 | −98 / −97 / +0.05 | −50 / −48 / +0.02 | +9 / +14 / 0 | **−60** / **−60** / **+0.03** |
| 0.30 | −43 / −48 / **+0.04** | −9 / −11 / 0 | +13 / +21 / −0.02 | +11 / +16 / −0.02 | −1 / −3 / 0 |
| all |  |  |  |  | **−113** / **−117** / **+0.04** |

<details><summary>Intervals and p-values by rate, churn pooled</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI], p | Δ \|bias\| [95% CI], p | Δ RMSE [95% CI], p | Δ Spearman [95% CI], p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `no_ar` | 0.01 | pooled | 40 | 314 → 120 | 309 → 84 | **−123 [−224, −74], p <10⁻⁴** | **−152 [−265, −90], p <10⁻⁴** | **−0.15 [−0.22, −0.10], p <10⁻⁴** | **+0.05 [+0.03, +0.07], p 0.0002** |
| `no_ar` | 0.05 | pooled | 40 | 200 → 114 | 199 → 109 | −47 [−153, +13], p 0.154 | −52 [−159, +10], p 0.121 | −0.11 [−0.28, +0.07], p 0.216 | **+0.30 [+0.16, +0.35], p <10⁻⁴** |
| `no_ar` | 0.10 | pooled | 40 | 98 → 130 | 93 → 128 | **+32 [+8, +55], p 0.005** | **+34 [+8, +58], p 0.011** | **+0.52 [+0.26, +0.69], p <10⁻⁴** | **−0.07 [−0.09, −0.03], p 0.016** |
| `no_ar` | 0.30 | pooled | 40 | 21 → 56 | 13 → 48 | **+28 [+18, +44], p <10⁻⁴** | **+28 [+15, +45], p <10⁻⁴** | **+1.15 [+0.93, +1.42], p <10⁻⁴** | **−0.12 [−0.13, −0.10], p <10⁻⁴** |
| `no_ar` | all | pooled | 160 | 158 → 105 | 154 → 92 | −4 [−30, +9], p 0.616 | −10 [−37, +6], p 0.230 | **+0.29 [+0.16, +0.43], p <10⁻⁴** | +0.01 [−0.02, +0.06], p 0.657 |
| `no_ar` + `kmeans_8` | 0.01 | pooled | 40 | 307 → 130 | 302 → 91 | **−125 [−219, −67], p <10⁻⁴** | **−150 [−264, −88], p <10⁻⁴** | **−0.12 [−0.21, −0.07], p 0.0002** | +0.01 [−0.02, +0.03], p 0.554 |
| `no_ar` + `kmeans_8` | 0.05 | pooled | 40 | 202 → 122 | 200 → 116 | **−40 [−115, −4], p 0.025** | **−42 [−122, −5], p 0.024** | −0.06 [−0.22, +0.09], p 0.460 | **+0.04 [+0.02, +0.06], p 0.0002** |
| `no_ar` + `kmeans_8` | 0.10 | pooled | 40 | 139 → 116 | 138 → 114 | −11 [−37, +13], p 0.485 | −13 [−39, +13], p 0.361 | +0.21 [0, +0.41], p 0.047 | **+0.05 [+0.02, +0.07], p <10⁻⁴** |
| `no_ar` + `kmeans_8` | 0.30 | pooled | 40 | 78 → 67 | 78 → 64 | **−7 [−20, −2], p 0.011** | **−12 [−22, −4], p 0.002** | −0.20 [−0.52, +0.07], p 0.125 | **−0.02 [−0.03, −0.01], p <10⁻⁴** |
| `no_ar` + `kmeans_8` | all | pooled | 160 | 181 → 109 | 179 → 96 | **−31 [−49, −18], p <10⁻⁴** | **−37 [−56, −22], p <10⁻⁴** | −0.06 [−0.14, +0.03], p 0.150 | **+0.01 [+0.01, +0.03], p 0.002** |
| `ar_bounded` | 0.01 | pooled | 40 | 243 → 103 | 232 → 63 | **−69 [−117, −45], p <10⁻⁴** | **−94 [−153, −62], p <10⁻⁴** | **−0.08 [−0.14, −0.06], p <10⁻⁴** | **+0.09 [+0.07, +0.12], p <10⁻⁴** |
| `ar_bounded` | 0.05 | pooled | 40 | 58 → 78 | 46 → 68 | **+13 [+1, +31], p 0.031** | +14 [−3, +35], p 0.125 | **+0.07 [+0.01, +0.17], p 0.026** | 0 [−0.02, +0.01], p 0.685 |
| `ar_bounded` | 0.10 | pooled | 40 | 36 → 77 | 24 → 70 | **+34 [+18, +51], p <10⁻⁴** | **+40 [+22, +58], p <10⁻⁴** | **+0.23 [+0.13, +0.40], p <10⁻⁴** | **−0.01 [−0.02, 0], p 0.006** |
| `ar_bounded` | 0.30 | pooled | 40 | 20 → 50 | 11 → 42 | **+27 [+19, +36], p <10⁻⁴** | **+29 [+18, +40], p <10⁻⁴** | **+0.78 [+0.46, +1.29], p <10⁻⁴** | **−0.04 [−0.05, −0.03], p <10⁻⁴** |
| `ar_bounded` | all | pooled | 160 | 89 → 77 | 78 → 61 | **+9 [+2, +17], p 0.020** | +8 [−3, +18], p 0.162 | **+0.16 [+0.09, +0.23], p <10⁻⁴** | 0 [−0.01, +0.01], p 0.674 |
| `ar_bounded` + `kmeans_8` | 0.01 | pooled | 40 | 221 → 119 | 211 → 79 | **−85 [−128, −46], p <10⁻⁴** | **−113 [−168, −62], p <10⁻⁴** | **−0.09 [−0.12, −0.04], p 0.008** | **+0.06 [+0.03, +0.08], p 0.0001** |
| `ar_bounded` + `kmeans_8` | 0.05 | pooled | 40 | 126 → 96 | 121 → 88 | −2 [−39, +22], p 0.952 | −4 [−47, +23], p 0.889 | +0.08 [−0.08, +0.22], p 0.307 | **+0.05 [+0.03, +0.06], p <10⁻⁴** |
| `ar_bounded` + `kmeans_8` | 0.10 | pooled | 40 | 78 → 97 | 76 → 93 | **+13 [+2, +28], p 0.020** | +14 [0, +30], p 0.051 | **+0.23 [+0.04, +0.39], p 0.014** | **+0.02 [0, +0.03], p 0.026** |
| `ar_bounded` + `kmeans_8` | 0.30 | pooled | 40 | 56 → 64 | 56 → 60 | +5 [−4, +12], p 0.216 | +2 [−8, +10], p 0.725 | +0.23 [−0.03, +0.56], p 0.068 | **−0.01 [−0.02, 0], p 0.003** |
| `ar_bounded` + `kmeans_8` | all | pooled | 160 | 120 → 94 | 116 → 80 | −6 [−16, +3], p 0.166 | −10 [−22, 0], p 0.062 | **+0.09 [+0.01, +0.17], p 0.029** | **+0.02 [+0.01, +0.03], p <10⁻⁴** |
| `ar_unbounded` | 0.01 | pooled | 40 | 513 → 150 | 505 → 107 | **−269 [−480, −156], p <10⁻⁴** | **−301 [−511, −188], p <10⁻⁴** | **−1.02 [−3.11, −0.29], p <10⁻⁴** | **+0.18 [+0.15, +0.22], p <10⁻⁴** |
| `ar_unbounded` | 0.05 | pooled | 40 | 582 → 105 | 582 → 97 | **−264 [−526, −194], p <10⁻⁴** | **−267 [−529, −198], p <10⁻⁴** | **−9.78 [−11.56, −8.01], p <10⁻⁴** | **+0.10 [+0.06, +0.13], p 0.0006** |
| `ar_unbounded` | 0.10 | pooled | 40 | 322 → 149 | 318 → 148 | **−180 [−250, −94], p <10⁻⁴** | **−176 [−250, −83], p <10⁻⁴** | **−7.71 [−11.38, −4.77], p <10⁻⁴** | **+0.05 [+0.03, +0.07], p 0.0001** |
| `ar_unbounded` | 0.30 | pooled | 38 | 144 → 92 | 138 → 89 | −47 [−101, +9], p 0.099 | −44 [−101, +12], p 0.122 | **−7.26 [−12.93, −0.42], p 0.021** | −0.02 [−0.04, +0.01], p 0.243 |
| `ar_unbounded` | all | pooled | 158 | 394 → 124 | 389 → 111 | **−180 [−224, −143], p <10⁻⁴** | **−189 [−234, −150], p <10⁻⁴** | **−6.17 [−7.68, −4.77], p <10⁻⁴** | **+0.07 [+0.05, +0.09], p <10⁻⁴** |
| `ar_unbounded` + `kmeans_8` | 0.01 | pooled | 40 | 1118 → 121 | 1107 → 83 | **−493 [−1049, −252], p <10⁻⁴** | **−529 [−1084, −269], p <10⁻⁴** | **−5.46 [−8.23, −1.90], p <10⁻⁴** | **+0.12 [+0.07, +0.17], p <10⁻⁴** |
| `ar_unbounded` + `kmeans_8` | 0.05 | pooled | 40 | 424 → 87 | 420 → 76 | **−189 [−493, −103], p <10⁻⁴** | **−194 [−502, −107], p <10⁻⁴** | **−8.06 [−10.13, −6.08], p <10⁻⁴** | **+0.06 [+0.03, +0.10], p 0.001** |
| `ar_unbounded` + `kmeans_8` | 0.10 | pooled | 40 | 162 → 88 | 153 → 81 | **−60 [−115, −11], p 0.009** | **−60 [−116, −9], p 0.017** | **−4.28 [−7.25, −2.30], p 0.0001** | **+0.03 [+0.01, +0.05], p 0.012** |
| `ar_unbounded` + `kmeans_8` | 0.30 | pooled | 40 | 61 → 50 | 48 → 39 | −1 [−19, +9], p 0.795 | −3 [−23, +14], p 0.775 | −0.40 [−2.48, +0.36], p 0.334 | 0 [−0.02, +0.01], p 0.921 |
| `ar_unbounded` + `kmeans_8` | all | pooled | 160 | 441 → 86 | 432 → 70 | **−113 [−155, −75], p <10⁻⁴** | **−117 [−165, −80], p <10⁻⁴** | **−4.83 [−5.81, −3.71], p <10⁻⁴** | **+0.04 [+0.03, +0.06], p <10⁻⁴** |

</details>

<details><summary>Intervals and p-values per rate × churn cell</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI], p | Δ \|bias\| [95% CI], p | Δ RMSE [95% CI], p | Δ Spearman [95% CI], p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `no_ar` | 0.01 | 20% | 10 | 71 → 58 | 59 → 37 | −13 [−22, −2], p 0.027 | **−22 [−39, −7], p 0.020** | −0.02 [−0.05, −0.01], p 0.027 | +0.04 [+0.01, +0.07], p 0.037 |
| `no_ar` | 0.01 | 40% | 10 | 136 → 96 | 132 → 76 | −45 [−87, +13], p 0.084 | −58 [−116, +7], p 0.084 | −0.05 [−0.13, +0.03], p 0.193 | +0.01 [−0.04, +0.06], p 0.770 |
| `no_ar` | 0.01 | 60% | 10 | 291 → 129 | 290 → 102 | **−157 [−218, −109], p 0.002** | **−181 [−263, −119], p 0.002** | **−0.20 [−0.26, −0.14], p 0.002** | +0.06 [0, +0.12], p 0.084 |
| `no_ar` | 0.01 | 80% | 10 | 759 → 197 | 755 → 121 | **−521 [−829, −312], p 0.002** | **−579 [−917, −354], p 0.002** | **−0.43 [−0.54, −0.29], p 0.002** | **+0.10 [+0.04, +0.14], p 0.006** |
| `no_ar` | 0.05 | 20% | 10 | 45 → 75 | 42 → 72 | **+29 [+10, +49], p 0.010** | +31 [+2, +49], p 0.037 | +0.33 [+0.02, +0.67], p 0.027 | +0.05 [−0.29, +0.38], p 0.846 |
| `no_ar` | 0.05 | 40% | 10 | 99 → 110 | 98 → 108 | +12 [−18, +42], p 0.432 | +12 [−22, +42], p 0.432 | +0.06 [−0.20, +0.26], p 0.846 | +0.20 [+0.01, +0.41], p 0.049 |
| `no_ar` | 0.05 | 60% | 10 | 158 → 149 | 157 → 144 | −9 [−91, +77], p 0.625 | −12 [−101, +77], p 0.695 | −0.07 [−0.37, +0.25], p 0.432 | **+0.34 [+0.30, +0.39], p 0.002** |
| `no_ar` | 0.05 | 80% | 10 | 497 → 122 | 497 → 111 | **−370 [−443, −312], p 0.002** | **−384 [−452, −324], p 0.002** | **−0.76 [−0.93, −0.57], p 0.002** | **+0.33 [+0.26, +0.38], p 0.002** |
| `no_ar` | 0.10 | 20% | 10 | 33 → 58 | 31 → 56 | +25 [+1, +49], p 0.049 | +25 [0, +50], p 0.064 | **+0.88 [+0.52, +1.19], p 0.002** | **−0.09 [−0.11, −0.06], p 0.002** |
| `no_ar` | 0.10 | 40% | 10 | 58 → 94 | 57 → 91 | +41 [0, +83], p 0.049 | +37 [−8, +85], p 0.193 | +0.71 [+0.07, +1.06], p 0.049 | −0.10 [−0.11, +0.05], p 0.105 |
| `no_ar` | 0.10 | 60% | 10 | 69 → 103 | 61 → 99 | +27 [+6, +71], p 0.037 | +30 [+4, +72], p 0.037 | **+0.24 [+0.08, +0.63], p 0.010** | **−0.07 [−0.11, −0.03], p 0.002** |
| `no_ar` | 0.10 | 80% | 10 | 233 → 265 | 225 → 265 | +25 [−103, +165], p 0.625 | +43 [−103, +168], p 0.557 | +0.17 [−0.34, +0.70], p 0.432 | +0.14 [−0.07, +0.34], p 0.232 |
| `no_ar` | 0.30 | 20% | 10 | 19 → 36 | 17 → 24 | **+11 [+4, +34], p 0.002** | +1 [−13, +28], p 0.846 | **+0.98 [+0.55, +3.02], p 0.002** | **−0.14 [−0.15, −0.12], p 0.002** |
| `no_ar` | 0.30 | 40% | 10 | 14 → 36 | 9 → 29 | **+20 [+11, +31], p 0.002** | **+20 [+6, +34], p 0.020** | **+1.46 [+1.09, +2.35], p 0.002** | **−0.14 [−0.16, −0.11], p 0.002** |
| `no_ar` | 0.30 | 60% | 10 | 18 → 57 | 9 → 50 | **+35 [+16, +65], p 0.002** | **+39 [+15, +71], p 0.010** | **+1.37 [+0.95, +1.97], p 0.002** | **−0.12 [−0.13, −0.09], p 0.002** |
| `no_ar` | 0.30 | 80% | 10 | 32 → 96 | 17 → 88 | **+55 [+18, +103], p 0.010** | **+66 [+17, +121], p 0.020** | **+0.73 [+0.40, +1.24], p 0.002** | **−0.07 [−0.08, −0.05], p 0.002** |
| `no_ar` + `kmeans_8` | 0.01 | 20% | 10 | 86 → 55 | 77 → 35 | **−31 [−50, −11], p 0.004** | **−42 [−74, −14], p 0.010** | **−0.05 [−0.08, −0.02], p 0.002** | 0 [−0.04, +0.01], p 0.492 |
| `no_ar` + `kmeans_8` | 0.01 | 40% | 10 | 140 → 148 | 136 → 134 | −17 [−63, +89], p 0.695 | −24 [−78, +83], p 0.695 | +0.08 [−0.10, +0.32], p 0.557 | −0.01 [−0.08, +0.05], p 1.000 |
| `no_ar` + `kmeans_8` | 0.01 | 60% | 10 | 273 → 120 | 271 → 87 | **−154 [−204, −96], p 0.002** | **−194 [−249, −112], p 0.002** | **−0.17 [−0.28, −0.07], p 0.010** | 0 [−0.05, +0.06], p 1.000 |
| `no_ar` + `kmeans_8` | 0.01 | 80% | 10 | 729 → 195 | 725 → 109 | **−488 [−782, −365], p 0.002** | **−565 [−893, −435], p 0.002** | **−0.39 [−0.50, −0.24], p 0.002** | +0.04 [+0.01, +0.08], p 0.049 |
| `no_ar` + `kmeans_8` | 0.05 | 20% | 10 | 47 → 67 | 40 → 62 | +21 [−1, +40], p 0.064 | +23 [−4, +47], p 0.064 | +0.15 [+0.01, +0.38], p 0.049 | +0.03 [−0.02, +0.08], p 0.275 |
| `no_ar` + `kmeans_8` | 0.05 | 40% | 10 | 82 → 107 | 81 → 105 | +23 [−15, +61], p 0.432 | +23 [−21, +62], p 0.492 | +0.28 [+0.03, +0.52], p 0.037 | +0.01 [−0.01, +0.05], p 0.131 |
| `no_ar` + `kmeans_8` | 0.05 | 60% | 10 | 172 → 134 | 172 → 131 | −43 [−76, +6], p 0.064 | −43 [−86, +6], p 0.064 | −0.11 [−0.30, +0.13], p 0.432 | **+0.07 [+0.04, +0.18], p 0.002** |
| `no_ar` + `kmeans_8` | 0.05 | 80% | 10 | 506 → 181 | 506 → 167 | **−330 [−430, −211], p 0.002** | **−340 [−452, −227], p 0.002** | **−0.75 [−0.95, −0.45], p 0.002** | +0.03 [−0.06, +0.08], p 0.232 |
| `no_ar` + `kmeans_8` | 0.10 | 20% | 10 | 39 → 57 | 35 → 54 | +20 [+3, +36], p 0.027 | +18 [−2, +42], p 0.064 | +0.41 [0, +0.87], p 0.064 | +0.04 [−0.04, +0.08], p 0.131 |
| `no_ar` + `kmeans_8` | 0.10 | 40% | 10 | 74 → 95 | 74 → 93 | +22 [+1, +39], p 0.049 | +23 [−5, +40], p 0.064 | **+0.66 [+0.11, +0.97], p 0.014** | +0.05 [0, +0.11], p 0.049 |
| `no_ar` + `kmeans_8` | 0.10 | 60% | 10 | 132 → 109 | 132 → 106 | −23 [−69, +21], p 0.275 | −24 [−69, +21], p 0.275 | −0.11 [−0.48, +0.24], p 0.557 | **+0.05 [+0.01, +0.10], p 0.010** |
| `no_ar` + `kmeans_8` | 0.10 | 80% | 10 | 309 → 203 | 309 → 201 | **−96 [−176, −31], p 0.010** | **−97 [−176, −32], p 0.010** | −0.07 [−0.40, +0.38], p 0.695 | +0.05 [−0.01, +0.11], p 0.084 |
| `no_ar` + `kmeans_8` | 0.30 | 20% | 10 | 28 → 30 | 27 → 21 | −1 [−3, +8], p 0.625 | −8 [−16, +3], p 0.232 | −0.25 [−0.63, +0.51], p 0.375 | +0.01 [0, +0.05], p 0.064 |
| `no_ar` + `kmeans_8` | 0.30 | 40% | 10 | 53 → 55 | 53 → 53 | +3 [−12, +14], p 0.846 | −1 [−13, +13], p 1.000 | +0.39 [−0.39, +0.96], p 0.375 | **−0.04 [−0.06, −0.02], p 0.002** |
| `no_ar` + `kmeans_8` | 0.30 | 60% | 10 | 90 → 65 | 90 → 65 | **−25 [−40, −9], p 0.006** | **−26 [−42, −9], p 0.006** | −0.57 [−1.18, +0.01], p 0.064 | **−0.03 [−0.04, −0.02], p 0.002** |
| `no_ar` + `kmeans_8` | 0.30 | 80% | 10 | 142 → 120 | 142 → 119 | −24 [−59, +14], p 0.193 | −24 [−61, +15], p 0.193 | −0.26 [−0.99, +0.19], p 0.232 | **−0.02 [−0.03, −0.01], p 0.004** |
| `ar_bounded` | 0.01 | 20% | 10 | 65 → 62 | 47 → 44 | −2 [−22, +21], p 0.846 | +2 [−30, +25], p 0.922 | −0.01 [−0.07, +0.05], p 0.492 | **+0.07 [+0.02, +0.11], p 0.014** |
| `ar_bounded` | 0.01 | 40% | 10 | 121 → 75 | 114 → 58 | **−48 [−66, −24], p 0.006** | **−61 [−88, −24], p 0.006** | **−0.07 [−0.11, −0.03], p 0.010** | **+0.07 [+0.03, +0.11], p 0.004** |
| `ar_bounded` | 0.01 | 60% | 10 | 196 → 98 | 190 → 60 | **−81 [−147, −62], p 0.002** | **−115 [−176, −84], p 0.002** | **−0.09 [−0.18, −0.06], p 0.002** | **+0.10 [+0.05, +0.16], p 0.004** |
| `ar_bounded` | 0.01 | 80% | 10 | 591 → 176 | 576 → 90 | **−345 [−686, −135], p 0.002** | **−415 [−802, −159], p 0.002** | **−0.27 [−0.43, −0.12], p 0.002** | **+0.16 [+0.07, +0.23], p 0.006** |
| `ar_bounded` | 0.05 | 20% | 10 | 48 → 68 | 47 → 63 | +22 [0, +39], p 0.064 | +23 [−9, +42], p 0.275 | **+0.30 [+0.07, +0.47], p 0.020** | −0.05 [−0.08, −0.01], p 0.027 |
| `ar_bounded` | 0.05 | 40% | 10 | 56 → 65 | 52 → 54 | +1 [−23, +47], p 1.000 | −7 [−36, +46], p 0.922 | −0.07 [−0.18, +0.28], p 0.557 | +0.01 [0, +0.02], p 0.131 |
| `ar_bounded` | 0.05 | 60% | 10 | 67 → 77 | 61 → 69 | +6 [−19, +46], p 0.770 | 0 [−30, +46], p 1.000 | +0.03 [−0.07, +0.14], p 0.492 | 0 [−0.02, +0.02], p 0.770 |
| `ar_bounded` | 0.05 | 80% | 10 | 60 → 102 | 27 → 84 | +31 [−3, +88], p 0.084 | +53 [−2, +122], p 0.064 | **+0.05 [+0.02, +0.11], p 0.014** | +0.01 [−0.02, +0.03], p 0.492 |
| `ar_bounded` | 0.10 | 20% | 10 | 32 → 52 | 30 → 45 | +14 [+2, +40], p 0.037 | +14 [−7, +37], p 0.322 | **+0.44 [+0.07, +0.73], p 0.020** | **−0.02 [−0.03, −0.01], p 0.014** |
| `ar_bounded` | 0.10 | 40% | 10 | 34 → 41 | 28 → 33 | +6 [−8, +19], p 0.432 | +3 [−15, +28], p 0.695 | +0.02 [−0.16, +0.41], p 0.846 | 0 [−0.25, +0.01], p 0.770 |
| `ar_bounded` | 0.10 | 60% | 10 | 33 → 115 | 21 → 113 | **+74 [+38, +126], p 0.002** | **+86 [+49, +139], p 0.002** | **+0.44 [+0.24, +1.10], p 0.004** | **−0.01 [−0.04, 0], p 0.006** |
| `ar_bounded` | 0.10 | 80% | 10 | 45 → 102 | 15 → 90 | **+48 [+9, +103], p 0.006** | **+66 [+28, +119], p 0.006** | +0.09 [0, +0.25], p 0.064 | 0 [−0.02, +0.02], p 0.922 |
| `ar_bounded` | 0.30 | 20% | 10 | 12 → 37 | 10 → 27 | **+22 [+13, +39], p 0.002** | +17 [−2, +38], p 0.275 | **+1.13 [+0.25, +2.98], p 0.010** | **−0.04 [−0.06, −0.03], p 0.002** |
| `ar_bounded` | 0.30 | 40% | 10 | 13 → 42 | 6 → 35 | **+28 [+14, +44], p 0.002** | **+29 [+8, +48], p 0.006** | **+0.96 [+0.36, +2.14], p 0.004** | **−0.06 [−0.22, −0.03], p 0.002** |
| `ar_bounded` | 0.30 | 60% | 10 | 19 → 67 | 10 → 62 | **+43 [+18, +73], p 0.002** | **+51 [+22, +80], p 0.002** | **+1.19 [+0.45, +1.85], p 0.002** | **−0.04 [−0.05, −0.02], p 0.002** |
| `ar_bounded` | 0.30 | 80% | 10 | 35 → 54 | 19 → 46 | **+20 [+6, +33], p 0.020** | +28 [+4, +51], p 0.027 | +0.19 [−0.03, +0.36], p 0.084 | **−0.02 [−0.05, −0.01], p 0.002** |
| `ar_bounded` + `kmeans_8` | 0.01 | 20% | 10 | 72 → 73 | 58 → 59 | −9 [−28, +52], p 0.322 | −8 [−37, +53], p 0.432 | 0 [−0.08, +0.14], p 1.000 | +0.03 [−0.01, +0.07], p 0.105 |
| `ar_bounded` + `kmeans_8` | 0.01 | 40% | 10 | 121 → 102 | 116 → 80 | −46 [−68, +51], p 0.432 | −65 [−101, +36], p 0.275 | −0.07 [−0.11, +0.15], p 0.432 | +0.06 [−0.01, +0.09], p 0.084 |
| `ar_bounded` + `kmeans_8` | 0.01 | 60% | 10 | 245 → 103 | 241 → 67 | **−145 [−189, −88], p 0.002** | **−184 [−242, −100], p 0.004** | **−0.17 [−0.25, −0.06], p 0.020** | **+0.10 [+0.04, +0.17], p 0.014** |
| `ar_bounded` + `kmeans_8` | 0.01 | 80% | 10 | 445 → 196 | 429 → 112 | **−228 [−419, −116], p 0.004** | **−302 [−479, −151], p 0.004** | −0.12 [−0.23, −0.04], p 0.027 | +0.05 [−0.06, +0.13], p 0.275 |
| `ar_bounded` + `kmeans_8` | 0.05 | 20% | 10 | 39 → 79 | 29 → 77 | **+42 [+26, +52], p 0.002** | **+49 [+34, +62], p 0.002** | **+0.43 [+0.30, +0.62], p 0.002** | **+0.06 [+0.03, +0.09], p 0.006** |
| `ar_bounded` + `kmeans_8` | 0.05 | 40% | 10 | 64 → 56 | 61 → 48 | −6 [−34, +14], p 0.625 | −12 [−41, +14], p 0.432 | +0.01 [−0.18, +0.20], p 0.770 | **+0.07 [+0.03, +0.10], p 0.010** |
| `ar_bounded` + `kmeans_8` | 0.05 | 60% | 10 | 110 → 155 | 108 → 152 | +41 [+4, +99], p 0.049 | +44 [+4, +102], p 0.049 | +0.25 [−0.02, +0.41], p 0.064 | **+0.04 [+0.01, +0.08], p 0.020** |
| `ar_bounded` + `kmeans_8` | 0.05 | 80% | 10 | 290 → 94 | 287 → 73 | **−195 [−307, −81], p 0.002** | **−215 [−332, −90], p 0.002** | **−0.45 [−0.76, −0.19], p 0.002** | +0.01 [−0.01, +0.04], p 0.232 |
| `ar_bounded` + `kmeans_8` | 0.10 | 20% | 10 | 33 → 45 | 28 → 38 | +9 [−2, +30], p 0.131 | +9 [−11, +34], p 0.375 | +0.23 [−0.15, +0.61], p 0.322 | +0.05 [−0.03, +0.10], p 0.105 |
| `ar_bounded` + `kmeans_8` | 0.10 | 40% | 10 | 53 → 58 | 52 → 54 | +4 [−18, +25], p 0.625 | +2 [−23, +25], p 0.770 | +0.25 [−0.17, +0.72], p 0.432 | +0.01 [0, +0.03], p 0.049 |
| `ar_bounded` + `kmeans_8` | 0.10 | 60% | 10 | 93 → 108 | 92 → 106 | +6 [−21, +57], p 0.695 | +5 [−24, +58], p 0.695 | +0.13 [−0.11, +0.71], p 0.375 | +0.04 [+0.01, +0.07], p 0.027 |
| `ar_bounded` + `kmeans_8` | 0.10 | 80% | 10 | 134 → 177 | 131 → 175 | **+41 [+10, +78], p 0.020** | +42 [+6, +82], p 0.027 | +0.27 [−0.04, +0.56], p 0.084 | −0.02 [−0.07, 0], p 0.105 |
| `ar_bounded` + `kmeans_8` | 0.30 | 20% | 10 | 26 → 32 | 25 → 25 | +6 [+1, +12], p 0.037 | 0 [−10, +9], p 0.922 | +0.13 [−0.49, +0.66], p 0.432 | −0.01 [−0.12, +0.01], p 0.322 |
| `ar_bounded` + `kmeans_8` | 0.30 | 40% | 10 | 45 → 47 | 44 → 45 | +6 [−9, +13], p 0.432 | +6 [−15, +13], p 0.557 | +0.29 [−0.51, +0.73], p 0.232 | −0.01 [−0.03, +0.01], p 0.193 |
| `ar_bounded` + `kmeans_8` | 0.30 | 60% | 10 | 71 → 85 | 71 → 83 | +8 [−17, +48], p 0.557 | +7 [−18, +48], p 0.625 | +0.20 [−0.51, +2.19], p 0.557 | −0.01 [−0.03, 0], p 0.193 |
| `ar_bounded` + `kmeans_8` | 0.30 | 80% | 10 | 84 → 91 | 83 → 87 | −1 [−35, +51], p 1.000 | −2 [−40, +48], p 0.922 | +0.51 [−0.40, +1.17], p 0.232 | −0.01 [−0.05, 0], p 0.064 |
| `ar_unbounded` | 0.01 | 20% | 10 | 513 → 49 | 502 → 19 | **−140 [−1254, −16], p 0.010** | **−183 [−1283, −34], p 0.010** | **−3.48 [−13.49, −0.12], p 0.002** | **+0.11 [+0.05, +0.17], p 0.004** |
| `ar_unbounded` | 0.01 | 40% | 10 | 189 → 78 | 182 → 55 | **−96 [−203, −21], p 0.010** | **−122 [−227, −32], p 0.010** | **−0.21 [−3.19, −0.06], p 0.010** | **+0.21 [+0.15, +0.33], p 0.002** |
| `ar_unbounded` | 0.01 | 60% | 10 | 514 → 127 | 514 → 79 | **−335 [−624, −227], p 0.002** | **−382 [−670, −280], p 0.002** | **−1.93 [−4.46, −0.18], p 0.006** | **+0.25 [+0.21, +0.33], p 0.002** |
| `ar_unbounded` | 0.01 | 80% | 10 | 838 → 345 | 821 → 273 | −570 [−994, −16], p 0.049 | −617 [−1044, −16], p 0.037 | −0.57 [−2.53, +0.24], p 0.105 | +0.15 [+0.02, +0.22], p 0.027 |
| `ar_unbounded` | 0.05 | 20% | 10 | 240 → 91 | 240 → 90 | **−131 [−236, −78], p 0.002** | **−131 [−239, −78], p 0.002** | **−11.77 [−13.96, −8.21], p 0.002** | **+0.08 [+0.06, +0.10], p 0.002** |
| `ar_unbounded` | 0.05 | 40% | 10 | 313 → 103 | 313 → 101 | **−211 [−308, −114], p 0.002** | **−213 [−310, −115], p 0.002** | **−7.98 [−11.15, −5.11], p 0.002** | +0.15 [+0.02, +0.20], p 0.049 |
| `ar_unbounded` | 0.05 | 60% | 10 | 462 → 124 | 462 → 121 | **−285 [−561, −191], p 0.002** | **−287 [−564, −191], p 0.002** | **−9.16 [−11.71, −6.06], p 0.002** | **+0.14 [+0.08, +0.19], p 0.004** |
| `ar_unbounded` | 0.05 | 80% | 10 | 1315 → 100 | 1314 → 77 | **−1188 [−1909, −507], p 0.002** | **−1213 [−1920, −524], p 0.002** | **−11.28 [−17.15, −4.90], p 0.002** | −0.05 [−0.25, +0.11], p 0.770 |
| `ar_unbounded` | 0.10 | 20% | 10 | 301 → 102 | 301 → 101 | **−193 [−308, −98], p 0.002** | **−193 [−308, −98], p 0.002** | **−16.03 [−23.63, −8.96], p 0.002** | **+0.05 [+0.02, +0.07], p 0.010** |
| `ar_unbounded` | 0.10 | 40% | 10 | 375 → 91 | 375 → 89 | **−249 [−404, −190], p 0.002** | **−252 [−404, −190], p 0.002** | **−12.76 [−21.61, −7.48], p 0.002** | +0.08 [−0.01, +0.11], p 0.084 |
| `ar_unbounded` | 0.10 | 60% | 10 | 307 → 159 | 301 → 159 | −159 [−329, +19], p 0.131 | −142 [−329, +37], p 0.160 | −3.40 [−8.04, −0.55], p 0.037 | +0.05 [0, +0.12], p 0.084 |
| `ar_unbounded` | 0.10 | 80% | 10 | 307 → 245 | 295 → 245 | +63 [−256, +141], p 1.000 | +64 [−255, +168], p 1.000 | −0.71 [−4.77, +0.94], p 0.557 | +0.02 [−0.02, +0.07], p 0.432 |
| `ar_unbounded` | 0.30 | 20% | 10 | 268 → 71 | 268 → 69 | **−193 [−267, −129], p 0.002** | **−195 [−268, −129], p 0.002** | **−23.63 [−29.88, −16.25], p 0.002** | +0.04 [−0.05, +0.07], p 0.084 |
| `ar_unbounded` | 0.30 | 40% | 10 | 161 → 86 | 158 → 81 | −64 [−189, +36], p 0.232 | −59 [−192, +40], p 0.232 | −7.66 [−16.67, +1.52], p 0.131 | −0.04 [−0.14, +0.05], p 0.432 |
| `ar_unbounded` | 0.30 | 60% | 9 | 66 → 97 | 60 → 96 | +31 [0, +60], p 0.039 | +40 [+7, +66], p 0.039 | +0.78 [−0.04, +1.53], p 0.074 | −0.02 [−0.07, +0.02], p 0.074 |
| `ar_unbounded` | 0.30 | 80% | 9 | 65 → 118 | 49 → 112 | +61 [−41, +146], p 0.203 | +76 [−44, +164], p 0.203 | +0.87 [−2.69, +3.63], p 0.426 | **−0.04 [−0.07, −0.02], p 0.004** |
| `ar_unbounded` + `kmeans_8` | 0.01 | 20% | 10 | 250 → 65 | 235 → 46 | **−111 [−319, −32], p 0.006** | **−128 [−317, −33], p 0.010** | **−4.70 [−7.79, −0.84], p 0.002** | **+0.11 [+0.06, +0.17], p 0.004** |
| `ar_unbounded` + `kmeans_8` | 0.01 | 40% | 10 | 1304 → 83 | 1300 → 60 | **−750 [−2380, −141], p 0.004** | **−759 [−2409, −150], p 0.004** | **−10.02 [−18.15, −1.98], p 0.004** | +0.11 [−0.01, +0.25], p 0.084 |
| `ar_unbounded` + `kmeans_8` | 0.01 | 60% | 10 | 904 → 120 | 900 → 83 | **−699 [−1393, −164], p 0.002** | **−743 [−1423, −189], p 0.002** | **−2.64 [−8.23, −0.25], p 0.002** | +0.13 [+0.01, +0.26], p 0.037 |
| `ar_unbounded` + `kmeans_8` | 0.01 | 80% | 10 | 2015 → 216 | 1993 → 145 | **−1680 [−3315, −180], p 0.010** | **−1763 [−3350, −210], p 0.010** | **−1.79 [−11.44, −0.23], p 0.014** | +0.10 [+0.01, +0.22], p 0.037 |
| `ar_unbounded` + `kmeans_8` | 0.05 | 20% | 10 | 189 → 69 | 188 → 64 | **−131 [−163, −66], p 0.004** | **−133 [−173, −68], p 0.004** | **−9.19 [−11.23, −5.93], p 0.002** | +0.03 [−0.09, +0.06], p 0.232 |
| `ar_unbounded` + `kmeans_8` | 0.05 | 40% | 10 | 229 → 78 | 221 → 68 | **−122 [−330, −6], p 0.020** | −131 [−328, −4], p 0.049 | **−5.97 [−12.71, −0.55], p 0.002** | +0.10 [−0.01, +0.16], p 0.064 |
| `ar_unbounded` + `kmeans_8` | 0.05 | 60% | 10 | 359 → 103 | 357 → 98 | −169 [−506, −23], p 0.027 | −171 [−518, −29], p 0.027 | **−6.60 [−11.51, −1.91], p 0.004** | +0.08 [−0.06, +0.20], p 0.275 |
| `ar_unbounded` + `kmeans_8` | 0.05 | 80% | 10 | 919 → 97 | 914 → 72 | **−871 [−1243, −474], p 0.006** | **−882 [−1289, −472], p 0.010** | **−10.34 [−14.45, −6.36], p 0.004** | +0.07 [+0.01, +0.11], p 0.027 |
| `ar_unbounded` + `kmeans_8` | 0.10 | 20% | 10 | 145 → 47 | 141 → 39 | **−106 [−179, −12], p 0.010** | **−107 [−185, −15], p 0.014** | **−9.45 [−15.53, −2.24], p 0.010** | +0.06 [−0.05, +0.13], p 0.084 |
| `ar_unbounded` + `kmeans_8` | 0.10 | 40% | 10 | 174 → 77 | 167 → 72 | −98 [−203, −11], p 0.037 | −97 [−203, −2], p 0.037 | −5.91 [−11.54, −0.12], p 0.037 | +0.05 [−0.01, +0.12], p 0.105 |
| `ar_unbounded` + `kmeans_8` | 0.10 | 60% | 10 | 142 → 76 | 133 → 70 | −50 [−188, +51], p 0.432 | −48 [−203, +61], p 0.625 | −3.52 [−7.46, +0.21], p 0.105 | +0.02 [−0.09, +0.07], p 0.375 |
| `ar_unbounded` + `kmeans_8` | 0.10 | 80% | 10 | 189 → 150 | 171 → 142 | +9 [−231, +105], p 0.922 | +14 [−214, +126], p 0.770 | −0.24 [−4.55, +0.78], p 0.625 | 0 [−0.04, +0.03], p 0.846 |
| `ar_unbounded` + `kmeans_8` | 0.30 | 20% | 10 | 80 → 37 | 76 → 30 | −43 [−85, −2], p 0.049 | −48 [−92, 0], p 0.049 | −5.55 [−13.28, −1.36], p 0.027 | **+0.04 [+0.01, +0.08], p 0.014** |
| `ar_unbounded` + `kmeans_8` | 0.30 | 40% | 10 | 62 → 41 | 50 → 28 | −9 [−71, +3], p 0.375 | −11 [−76, +19], p 0.492 | −0.92 [−8.36, +1.32], p 0.375 | 0 [−0.06, +0.04], p 0.922 |
| `ar_unbounded` + `kmeans_8` | 0.30 | 60% | 10 | 54 → 65 | 39 → 55 | +13 [−3, +29], p 0.160 | +21 [−10, +41], p 0.193 | +0.48 [−0.12, +1.39], p 0.232 | −0.02 [−0.04, 0], p 0.193 |
| `ar_unbounded` + `kmeans_8` | 0.30 | 80% | 10 | 47 → 56 | 26 → 43 | +11 [−4, +22], p 0.131 | +16 [−5, +43], p 0.193 | +0.33 [−0.17, +0.99], p 0.232 | −0.02 [−0.04, 0], p 0.131 |

</details>

### 7. Neural models capture seasonality; Pareto/NBD cannot

**Verdict: supported** for this one seasonal pattern.

- **Shape.** The LSTM's predicted weekly totals correlate with the actual ones at 0.11–0.96
  and the Transformer's at 0.09–0.62, against −0.09 to +0.13 for Pareto/NBD. The
  correlation rises with the rate and falls with churn.
- **Counterfactual.** Multiplying Pareto/NBD's forecast by the true seasonal curve (rescaled
  to mean 1) lowers its MAPE by 5–20 points, significant in 15 of 16 cells. |bias| and
  Spearman do not move (under 1 point and 0.005): the season changes the weekly shape, not
  the yearly total or the order of customers.
- **Given the season, Pareto/NBD beats the LSTM** on MAPE in every cell at rates 0.01–0.10
  (+7 to +399 points for the LSTM). At rate 0.30 the two are level (+2, not significant).
  So the LSTM's dense-panel MAPE wins in claim 1 come from modelling the season.

Shape correlation, mean over panels: Pareto/NBD / LSTM `ar_bounded` / Transformer `ar_bounded`

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | −0.02 / +0.26 / +0.32 | 0 / +0.29 / +0.22 | −0.02 / +0.14 / +0.11 | +0.05 / +0.11 / +0.09 | 0 / +0.20 / +0.19 |
| 0.05 | −0.09 / +0.70 / +0.50 | +0.04 / +0.64 / +0.51 | +0.06 / +0.53 / +0.35 | +0.06 / +0.21 / +0.23 | +0.02 / +0.52 / +0.40 |
| 0.10 | −0.07 / +0.84 / +0.62 | +0.01 / +0.73 / +0.50 | +0.10 / +0.67 / +0.41 | +0.11 / +0.36 / +0.30 | +0.04 / +0.65 / +0.46 |
| 0.30 | −0.08 / +0.96 / +0.60 | +0.01 / +0.91 / +0.55 | +0.12 / +0.83 / +0.53 | +0.13 / +0.49 / +0.47 | +0.05 / +0.80 / +0.54 |

**Pareto/NBD → with true season**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **−9** / **0** / 0 | **−8** / **0** / 0 | **−5** / **−1** / 0 | −5 / **−1** / 0 | **−7** / **−1** / **0** |
| 0.05 | **−20** / **0** / 0 | **−14** / **−1** / 0 | **−14** / **−1** / 0 | **−7** / −1 / 0 | **−14** / **−1** / 0 |
| 0.10 | **−20** / 0 / 0 | **−18** / 0 / 0 | **−16** / −1 / 0 | **−11** / 0 / 0 | **−17** / 0 / **0** |
| 0.30 | **−15** / **0** / **0** | **−16** / **+1** / **0** | **−16** / **+1** / 0 | **−11** / +1 / 0 | **−15** / **+1** / **0** |
| all |  |  |  |  | **−13** / **0** / **0** |

**Pareto/NBD with season → LSTM `ar_bounded`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **+22** / **+35** / **−0.25** | **+61** / **+79** / **−0.34** | **+99** / **+133** / **−0.31** | **+399** / **+482** / **−0.27** | **+85** / **+114** / **−0.29** |
| 0.05 | **+33** / **+39** / **−0.07** | **+32** / **+37** / **−0.06** | **+38** / **+55** / **−0.04** | **+7** / +5 / **−0.04** | **+28** / **+34** / **−0.05** |
| 0.10 | **+22** / **+28** / **−0.03** | **+20** / **+22** / **−0.04** | **+11** / **+14** / **−0.03** | **+7** / −7 / **−0.04** | **+14** / **+14** / **−0.03** |
| 0.30 | **−3** / **−5** / **−0.01** | −2 / **−8** / **−0.01** | +3 / −3 / **−0.06** | **+11** / +6 / **−0.19** | +2 / **−4** / **−0.06** |
| all |  |  |  |  | **+23** / **+27** / **−0.09** |

<details><summary>Intervals and p-values by rate, churn pooled</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI], p | Δ \|bias\| [95% CI], p | Δ RMSE [95% CI], p | Δ Spearman [95% CI], p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pareto/NBD → with true season | 0.01 | pooled | 40 | 90 → 83 | 35 → 35 | **−7 [−9, −5], p <10⁻⁴** | **−1 [−1, 0], p <10⁻⁴** | **0 [0, 0], p <10⁻⁴** | **0 [0, 0], p 0.0009** |
| Pareto/NBD → with true season | 0.05 | pooled | 40 | 44 → 30 | 14 → 14 | **−14 [−16, −12], p <10⁻⁴** | **−1 [−1, 0], p 0.0004** | **0 [0, 0], p 0.0010** | 0 [0, 0], p 0.192 |
| Pareto/NBD → with true season | 0.10 | pooled | 40 | 37 → 21 | 9 → 9 | **−17 [−18, −15], p <10⁻⁴** | 0 [−1, 0], p 0.166 | **0 [0, 0], p 0.025** | **0 [0, 0], p 0.010** |
| Pareto/NBD → with true season | 0.30 | pooled | 40 | 32 → 17 | 13 → 14 | **−15 [−16, −14], p <10⁻⁴** | **+1 [0, +1], p <10⁻⁴** | **+0.01 [+0.01, +0.02], p <10⁻⁴** | **0 [0, 0], p 0.009** |
| Pareto/NBD → with true season | all | pooled | 160 | 51 → 38 | 18 → 18 | **−13 [−14, −12], p <10⁻⁴** | **0 [0, 0], p 0.002** | **0 [0, 0], p 0.0004** | **0 [0, 0], p <10⁻⁴** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.01 | pooled | 40 | 83 → 243 | 35 → 232 | **+85 [+60, +138], p <10⁻⁴** | **+114 [+82, +186], p <10⁻⁴** | **+0.12 [+0.09, +0.17], p <10⁻⁴** | **−0.29 [−0.33, −0.25], p <10⁻⁴** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.05 | pooled | 40 | 30 → 58 | 14 → 46 | **+28 [+22, +33], p <10⁻⁴** | **+34 [+25, +42], p <10⁻⁴** | **+0.20 [+0.15, +0.25], p <10⁻⁴** | **−0.05 [−0.06, −0.04], p <10⁻⁴** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.10 | pooled | 40 | 21 → 36 | 9 → 24 | **+14 [+11, +18], p <10⁻⁴** | **+14 [+8, +21], p <10⁻⁴** | **+0.20 [+0.15, +0.26], p <10⁻⁴** | **−0.03 [−0.04, −0.03], p <10⁻⁴** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.30 | pooled | 40 | 17 → 20 | 14 → 11 | +2 [−1, +4], p 0.216 | **−4 [−7, −1], p 0.019** | **−0.22 [−0.32, −0.13], p <10⁻⁴** | **−0.06 [−0.10, −0.03], p <10⁻⁴** |
| Pareto/NBD with season → LSTM `ar_bounded` | all | pooled | 160 | 38 → 89 | 18 → 78 | **+23 [+18, +28], p <10⁻⁴** | **+27 [+21, +35], p <10⁻⁴** | **+0.11 [+0.08, +0.14], p <10⁻⁴** | **−0.09 [−0.12, −0.06], p <10⁻⁴** |

</details>

<details><summary>Intervals and p-values per rate × churn cell</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI], p | Δ \|bias\| [95% CI], p | Δ RMSE [95% CI], p | Δ Spearman [95% CI], p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pareto/NBD → with true season | 0.01 | 20% | 10 | 48 → 39 | 12 → 12 | **−9 [−11, −7], p 0.002** | **0 [0, 0], p 0.002** | **0 [0, 0], p 0.010** | 0 [0, 0], p 0.625 |
| Pareto/NBD → with true season | 0.01 | 40% | 10 | 66 → 58 | 34 → 34 | **−8 [−11, −4], p 0.004** | **0 [−1, 0], p 0.002** | **0 [0, 0], p 0.002** | 0 [0, 0], p 0.049 |
| Pareto/NBD → with true season | 0.01 | 60% | 10 | 88 → 83 | 41 → 40 | **−5 [−9, −1], p 0.027** | **−1 [−1, −1], p 0.002** | **0 [0, 0], p 0.010** | 0 [0, 0], p 0.131 |
| Pareto/NBD → with true season | 0.01 | 80% | 10 | 159 → 153 | 54 → 53 | −5 [−13, +1], p 0.084 | **−1 [−1, −1], p 0.002** | **0 [0, 0], p 0.002** | 0 [0, 0], p 0.055 |
| Pareto/NBD → with true season | 0.05 | 20% | 10 | 35 → 16 | 8 → 8 | **−20 [−21, −18], p 0.002** | **0 [0, 0], p 0.002** | **0 [0, 0], p 0.020** | 0 [0, 0], p 0.770 |
| Pareto/NBD → with true season | 0.05 | 40% | 10 | 39 → 24 | 15 → 15 | **−14 [−16, −13], p 0.002** | **−1 [−1, −1], p 0.002** | **0 [0, 0], p 0.002** | 0 [0, 0], p 0.770 |
| Pareto/NBD → with true season | 0.05 | 60% | 10 | 42 → 29 | 12 → 12 | **−14 [−16, −11], p 0.002** | **−1 [−1, 0], p 0.027** | 0 [0, 0], p 0.193 | 0 [0, 0], p 1.000 |
| Pareto/NBD → with true season | 0.05 | 80% | 10 | 58 → 51 | 21 → 20 | **−7 [−11, −3], p 0.002** | −1 [−2, 0], p 0.375 | 0 [0, 0], p 0.695 | 0 [0, 0], p 0.049 |
| Pareto/NBD → with true season | 0.10 | 20% | 10 | 31 → 11 | 3 → 3 | **−20 [−21, −19], p 0.002** | 0 [0, 0], p 0.105 | **0 [0, 0], p 0.020** | 0 [0, 0], p 0.232 |
| Pareto/NBD → with true season | 0.10 | 40% | 10 | 32 → 13 | 5 → 5 | **−18 [−20, −16], p 0.002** | 0 [−1, 0], p 0.492 | **0 [0, 0], p 0.010** | 0 [0, 0], p 0.322 |
| Pareto/NBD → with true season | 0.10 | 60% | 10 | 37 → 21 | 9 → 8 | **−16 [−18, −13], p 0.002** | −1 [−1, 0], p 0.105 | 0 [0, 0], p 0.557 | 0 [0, 0], p 0.557 |
| Pareto/NBD → with true season | 0.10 | 80% | 10 | 49 → 38 | 21 → 20 | **−11 [−14, −8], p 0.002** | 0 [−2, +1], p 0.846 | 0 [0, 0], p 0.922 | 0 [0, 0], p 0.105 |
| Pareto/NBD → with true season | 0.30 | 20% | 10 | 30 → 15 | 14 → 15 | **−15 [−17, −14], p 0.002** | **0 [0, 0], p 0.002** | **+0.01 [+0.01, +0.01], p 0.002** | **0 [0, 0], p 0.037** |
| Pareto/NBD → with true season | 0.30 | 40% | 10 | 31 → 15 | 14 → 14 | **−16 [−17, −14], p 0.002** | **+1 [0, +1], p 0.002** | **+0.02 [+0.01, +0.02], p 0.002** | **0 [0, 0], p 0.010** |
| Pareto/NBD → with true season | 0.30 | 60% | 10 | 32 → 16 | 13 → 13 | **−16 [−18, −13], p 0.002** | **+1 [0, +1], p 0.006** | **+0.02 [+0.01, +0.02], p 0.002** | 0 [0, 0], p 0.922 |
| Pareto/NBD → with true season | 0.30 | 80% | 10 | 35 → 23 | 13 → 14 | **−11 [−14, −9], p 0.002** | +1 [0, +1], p 0.322 | **+0.01 [+0.01, +0.02], p 0.002** | 0 [0, 0], p 0.770 |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.01 | 20% | 10 | 39 → 65 | 12 → 47 | **+22 [+10, +46], p 0.002** | **+35 [+12, +56], p 0.010** | **+0.06 [+0.03, +0.12], p 0.002** | **−0.25 [−0.34, −0.17], p 0.002** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.01 | 40% | 10 | 58 → 121 | 34 → 114 | **+61 [+46, +77], p 0.002** | **+79 [+64, +95], p 0.002** | **+0.10 [+0.07, +0.14], p 0.002** | **−0.34 [−0.40, −0.28], p 0.002** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.01 | 60% | 10 | 83 → 196 | 40 → 190 | **+99 [+66, +167], p 0.002** | **+133 [+93, +214], p 0.002** | **+0.13 [+0.09, +0.21], p 0.004** | **−0.31 [−0.40, −0.20], p 0.002** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.01 | 80% | 10 | 153 → 591 | 53 → 576 | **+399 [+146, +693], p 0.002** | **+482 [+199, +797], p 0.002** | **+0.29 [+0.15, +0.42], p 0.002** | **−0.27 [−0.33, −0.20], p 0.002** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.05 | 20% | 10 | 16 → 48 | 8 → 47 | **+33 [+24, +42], p 0.002** | **+39 [+30, +47], p 0.002** | **+0.36 [+0.29, +0.44], p 0.002** | **−0.07 [−0.11, −0.04], p 0.002** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.05 | 40% | 10 | 24 → 56 | 15 → 52 | **+32 [+22, +40], p 0.002** | **+37 [+27, +48], p 0.002** | **+0.25 [+0.21, +0.29], p 0.002** | **−0.06 [−0.08, −0.04], p 0.002** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.05 | 60% | 10 | 29 → 67 | 12 → 61 | **+38 [+26, +49], p 0.002** | **+55 [+25, +63], p 0.004** | **+0.12 [+0.05, +0.18], p 0.006** | **−0.04 [−0.06, −0.03], p 0.002** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.05 | 80% | 10 | 51 → 60 | 20 → 27 | **+7 [+2, +20], p 0.006** | +5 [−7, +19], p 0.375 | **+0.08 [+0.05, +0.11], p 0.002** | **−0.04 [−0.06, −0.02], p 0.004** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.10 | 20% | 10 | 11 → 32 | 3 → 30 | **+22 [+12, +34], p 0.002** | **+28 [+15, +40], p 0.002** | **+0.37 [+0.27, +0.55], p 0.002** | **−0.03 [−0.04, −0.02], p 0.002** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.10 | 40% | 10 | 13 → 34 | 5 → 28 | **+20 [+12, +29], p 0.002** | **+22 [+11, +35], p 0.004** | **+0.29 [+0.19, +0.47], p 0.002** | **−0.04 [−0.05, −0.02], p 0.002** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.10 | 60% | 10 | 21 → 33 | 8 → 21 | **+11 [+8, +15], p 0.002** | **+14 [+5, +19], p 0.006** | **+0.11 [+0.07, +0.15], p 0.002** | **−0.03 [−0.04, −0.01], p 0.002** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.10 | 80% | 10 | 38 → 45 | 20 → 15 | **+7 [+1, +13], p 0.037** | −7 [−16, +6], p 0.275 | **+0.08 [+0.02, +0.14], p 0.014** | **−0.04 [−0.06, −0.03], p 0.002** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.30 | 20% | 10 | 15 → 12 | 15 → 10 | **−3 [−6, 0], p 0.037** | **−5 [−9, −1], p 0.020** | **−0.48 [−0.71, −0.16], p 0.010** | **−0.01 [−0.01, 0], p 0.002** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.30 | 40% | 10 | 15 → 13 | 14 → 6 | −2 [−5, +1], p 0.064 | **−8 [−12, −4], p 0.004** | **−0.37 [−0.45, −0.20], p 0.002** | **−0.01 [−0.02, −0.01], p 0.002** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.30 | 60% | 10 | 16 → 19 | 13 → 10 | +3 [0, +5], p 0.160 | −3 [−10, +3], p 0.432 | **−0.14 [−0.24, −0.02], p 0.027** | **−0.06 [−0.08, −0.05], p 0.002** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.30 | 80% | 10 | 23 → 35 | 14 → 19 | **+11 [+8, +17], p 0.002** | +6 [−4, +17], p 0.625 | −0.02 [−0.12, +0.08], p 0.375 | **−0.19 [−0.21, −0.18], p 0.002** |

</details>

### 8. Pareto/NBD's low bias means it is accurate per customer

**Verdict: not supported.** In every cell Pareto/NBD under-serves customers who are still
alive (alive ratio R_A 0.51–0.87, below 1 in all 16 cells) and assigns volume to customers
who have already died (dead leakage L_D 0.11–1.04, above 0 in all 16). The two errors
cancel in the total, which is why its |bias| is small. Both worsen with churn: at rate 0.01
and churn 80% it puts more volume on dead customers (L_D 1.04) than the living ones should
get. R_A and L_D use the generator's hidden truth (each customer's λ and death week).

| Rate | Churn | n | \|bias\| [95% CI] | R_A [95% CI], p vs 1 | L_D [95% CI], p vs 0 | RMSE [95% CI] |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 20% | 10 | 12 [9, 16] | **0.85 [0.81, 0.89], p 0.002** | **0.28 [0.27, 0.28], p 0.002** | 0.79 [0.75, 0.83] |
| 0.01 | 40% | 10 | 34 [26, 42] | **0.77 [0.73, 0.80], p 0.002** | **0.55 [0.52, 0.59], p 0.002** | 0.68 [0.64, 0.72] |
| 0.01 | 60% | 10 | 41 [30, 52] | **0.65 [0.60, 0.71], p 0.002** | **0.76 [0.71, 0.81], p 0.002** | 0.57 [0.53, 0.61] |
| 0.01 | 80% | 10 | 54 [29, 80] | **0.51 [0.46, 0.55], p 0.002** | **1.04 [0.85, 1.23], p 0.002** | 0.42 [0.38, 0.47] |
| 0.05 | 20% | 10 | 8 [5, 11] | **0.87 [0.85, 0.89], p 0.002** | **0.22 [0.21, 0.24], p 0.002** | 1.92 [1.88, 1.96] |
| 0.05 | 40% | 10 | 15 [10, 20] | **0.77 [0.74, 0.79], p 0.002** | **0.39 [0.37, 0.41], p 0.002** | 1.63 [1.60, 1.66] |
| 0.05 | 60% | 10 | 12 [7, 18] | **0.65 [0.62, 0.69], p 0.002** | **0.44 [0.40, 0.49], p 0.002** | 1.32 [1.26, 1.38] |
| 0.05 | 80% | 10 | 21 [10, 32] | **0.56 [0.50, 0.62], p 0.002** | **0.53 [0.45, 0.61], p 0.002** | 0.89 [0.85, 0.93] |
| 0.10 | 20% | 10 | 3 [1, 5] | **0.84 [0.83, 0.85], p 0.002** | **0.18 [0.17, 0.19], p 0.002** | 2.91 [2.85, 2.96] |
| 0.10 | 40% | 10 | 5 [4, 7] | **0.74 [0.72, 0.77], p 0.002** | **0.27 [0.26, 0.29], p 0.002** | 2.47 [2.37, 2.57] |
| 0.10 | 60% | 10 | 9 [3, 14] | **0.67 [0.64, 0.70], p 0.002** | **0.35 [0.29, 0.41], p 0.002** | 2.00 [1.93, 2.07] |
| 0.10 | 80% | 10 | 21 [9, 32] | **0.60 [0.55, 0.65], p 0.002** | **0.42 [0.33, 0.51], p 0.002** | 1.37 [1.23, 1.50] |
| 0.30 | 20% | 10 | 14 [13, 16] | **0.74 [0.74, 0.75], p 0.002** | **0.11 [0.10, 0.12], p 0.002** | 6.93 [6.68, 7.17] |
| 0.30 | 40% | 10 | 14 [12, 16] | **0.68 [0.67, 0.69], p 0.002** | **0.19 [0.17, 0.20], p 0.002** | 5.99 [5.73, 6.24] |
| 0.30 | 60% | 10 | 13 [9, 16] | **0.63 [0.62, 0.65], p 0.002** | **0.25 [0.22, 0.28], p 0.002** | 4.72 [4.56, 4.88] |
| 0.30 | 80% | 10 | 13 [6, 20] | **0.58 [0.56, 0.61], p 0.002** | **0.29 [0.24, 0.35], p 0.002** | 3.03 [2.81, 3.25] |
| 0.01 | pooled | 40 | 35 [27, 43] | **0.70 [0.65, 0.74], p <10⁻⁴** | **0.66 [0.56, 0.76], p <10⁻⁴** | 0.62 [0.57, 0.66] |
| 0.05 | pooled | 40 | 14 [11, 17] | **0.71 [0.67, 0.75], p <10⁻⁴** | **0.40 [0.35, 0.44], p <10⁻⁴** | 1.44 [1.31, 1.57] |
| 0.10 | pooled | 40 | 9 [6, 13] | **0.71 [0.68, 0.75], p <10⁻⁴** | **0.31 [0.27, 0.34], p <10⁻⁴** | 2.19 [2.00, 2.38] |
| 0.30 | pooled | 40 | 13 [12, 15] | **0.66 [0.64, 0.68], p <10⁻⁴** | **0.21 [0.18, 0.23], p <10⁻⁴** | 5.17 [4.68, 5.65] |
| all | pooled | 160 | 18 [15, 21] | **0.70 [0.68, 0.71], p <10⁻⁴** | **0.39 [0.35, 0.43], p <10⁻⁴** | 2.35 [2.05, 2.65] |

### 9. Neural models cannot detect a customer who has stopped

**Verdict: partly.** On dead leakage L_D, lower is better.

- **LSTM `ar_bounded`** leaks more than Pareto/NBD at rates 0.01–0.10 (+0.13 to +4.44), most
  at rate 0.01, churn 80%. At rate 0.30 with churn 60–80% and at rate 0.10 with churn 80%
  its leakage equals Pareto/NBD's: there the flags work.
- **Transformer `ar_bounded`** leaks more than Pareto/NBD in every cell (+0.06 to +0.78),
  and significantly in 13 of 16.

**LSTM**: Δ L_D / Δ R_A

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **+0.14** / +0.19 | **+0.54** / **+0.26** | **+1.21** / **+0.25** | **+4.44** / **+0.52** | **+0.97** / **+0.27** |
| 0.05 | **+0.20** / **+0.19** | **+0.35** / +0.03 | **+0.49** / +0.04 | **+0.27** / **−0.22** | **+0.31** / +0.02 |
| 0.10 | **+0.13** / **+0.13** | **+0.22** / +0.04 | **+0.17** / −0.01 | +0.09 / **−0.13** | **+0.16** / +0.01 |
| 0.30 | **+0.05** / **+0.18** | **+0.03** / **+0.13** | +0.01 / +0.04 | −0.02 / +0.01 | **+0.02** / **+0.09** |
| all |  |  |  |  | **+0.23** / **+0.08** |

**Transformer**: Δ L_D / Δ R_A

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **+0.12** / +0.13 | +0.06 / −0.16 | +0.26 / **−0.17** | +0.44 / −0.13 | **+0.19** / −0.09 |
| 0.05 | **+0.32** / **+0.32** | **+0.29** / +0.04 | **+0.50** / −0.03 | **+0.73** / **−0.20** | **+0.40** / +0.02 |
| 0.10 | **+0.20** / +0.21 | **+0.27** / 0 | **+0.78** / +0.20 | **+0.66** / +0.07 | **+0.43** / **+0.11** |
| 0.30 | **+0.12** / **+0.27** | **+0.18** / +0.18 | **+0.37** / **+0.35** | **+0.37** / **+0.21** | **+0.24** / **+0.25** |
| all |  |  |  |  | **+0.30** / **+0.08** |

<details><summary>Intervals and p-values by rate, churn pooled</summary>

| Comparison | Rate | Churn | n | L_D A → B | R_A A → B | Δ L_D [95% CI], p | Δ R_A [95% CI], p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM | 0.01 | pooled | 40 | 0.66 → 2.25 | 0.70 → 1.01 | **+0.97 [+0.61, +1.51], p <10⁻⁴** | **+0.27 [+0.19, +0.41], p <10⁻⁴** |
| LSTM | 0.05 | pooled | 40 | 0.40 → 0.72 | 0.71 → 0.72 | **+0.31 [+0.26, +0.37], p <10⁻⁴** | +0.02 [−0.05, +0.07], p 0.554 |
| LSTM | 0.10 | pooled | 40 | 0.31 → 0.47 | 0.71 → 0.72 | **+0.16 [+0.13, +0.20], p <10⁻⁴** | +0.01 [−0.04, +0.05], p 0.725 |
| LSTM | 0.30 | pooled | 40 | 0.21 → 0.23 | 0.66 → 0.75 | **+0.02 [+0.01, +0.04], p 0.010** | **+0.09 [+0.06, +0.13], p <10⁻⁴** |
| LSTM | all | pooled | 160 | 0.39 → 0.92 | 0.70 → 0.80 | **+0.23 [+0.19, +0.29], p <10⁻⁴** | **+0.08 [+0.06, +0.11], p <10⁻⁴** |
| Transformer | 0.01 | pooled | 40 | 0.66 → 0.91 | 0.70 → 0.62 | **+0.19 [+0.07, +0.31], p 0.002** | −0.09 [−0.17, +0.01], p 0.068 |
| Transformer | 0.05 | pooled | 40 | 0.40 → 0.91 | 0.71 → 0.73 | **+0.40 [+0.30, +0.57], p <10⁻⁴** | +0.02 [−0.07, +0.09], p 0.666 |
| Transformer | 0.10 | pooled | 40 | 0.31 → 0.79 | 0.71 → 0.83 | **+0.43 [+0.29, +0.58], p <10⁻⁴** | **+0.11 [+0.03, +0.20], p 0.008** |
| Transformer | 0.30 | pooled | 40 | 0.21 → 0.47 | 0.66 → 0.91 | **+0.24 [+0.18, +0.31], p <10⁻⁴** | **+0.25 [+0.17, +0.33], p <10⁻⁴** |
| Transformer | all | pooled | 160 | 0.39 → 0.77 | 0.70 → 0.77 | **+0.30 [+0.26, +0.36], p <10⁻⁴** | **+0.08 [+0.03, +0.12], p 0.001** |

</details>

<details><summary>Intervals and p-values per rate × churn cell</summary>

| Comparison | Rate | Churn | n | L_D A → B | R_A A → B | Δ L_D [95% CI], p | Δ R_A [95% CI], p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM | 0.01 | 20% | 10 | 0.28 → 0.42 | 0.85 → 1.03 | **+0.14 [+0.07, +0.21], p 0.010** | +0.19 [+0.01, +0.36], p 0.037 |
| LSTM | 0.01 | 40% | 10 | 0.55 → 1.09 | 0.77 → 1.03 | **+0.54 [+0.44, +0.63], p 0.002** | **+0.26 [+0.17, +0.34], p 0.002** |
| LSTM | 0.01 | 60% | 10 | 0.76 → 2.01 | 0.65 → 0.95 | **+1.21 [+0.78, +1.79], p 0.002** | **+0.25 [+0.10, +0.49], p 0.006** |
| LSTM | 0.01 | 80% | 10 | 1.04 → 5.48 | 0.51 → 1.03 | **+4.44 [+1.97, +6.33], p 0.002** | **+0.52 [+0.16, +0.89], p 0.006** |
| LSTM | 0.05 | 20% | 10 | 0.22 → 0.42 | 0.87 → 1.05 | **+0.20 [+0.17, +0.23], p 0.002** | **+0.19 [+0.12, +0.25], p 0.002** |
| LSTM | 0.05 | 40% | 10 | 0.39 → 0.73 | 0.77 → 0.80 | **+0.35 [+0.29, +0.40], p 0.002** | +0.03 [−0.03, +0.08], p 0.275 |
| LSTM | 0.05 | 60% | 10 | 0.44 → 0.92 | 0.65 → 0.69 | **+0.49 [+0.37, +0.60], p 0.002** | +0.04 [0, +0.08], p 0.064 |
| LSTM | 0.05 | 80% | 10 | 0.53 → 0.80 | 0.56 → 0.34 | **+0.27 [+0.14, +0.36], p 0.002** | **−0.22 [−0.26, −0.17], p 0.002** |
| LSTM | 0.10 | 20% | 10 | 0.18 → 0.32 | 0.84 → 0.97 | **+0.13 [+0.10, +0.18], p 0.002** | **+0.13 [+0.03, +0.21], p 0.020** |
| LSTM | 0.10 | 40% | 10 | 0.27 → 0.51 | 0.74 → 0.78 | **+0.22 [+0.15, +0.32], p 0.002** | +0.04 [−0.04, +0.12], p 0.275 |
| LSTM | 0.10 | 60% | 10 | 0.35 → 0.53 | 0.67 → 0.67 | **+0.17 [+0.13, +0.21], p 0.002** | −0.01 [−0.04, +0.03], p 0.770 |
| LSTM | 0.10 | 80% | 10 | 0.42 → 0.52 | 0.60 → 0.47 | +0.09 [−0.05, +0.22], p 0.105 | **−0.13 [−0.20, −0.04], p 0.010** |
| LSTM | 0.30 | 20% | 10 | 0.11 → 0.16 | 0.74 → 0.93 | **+0.05 [+0.04, +0.07], p 0.002** | **+0.18 [+0.17, +0.21], p 0.002** |
| LSTM | 0.30 | 40% | 10 | 0.19 → 0.22 | 0.68 → 0.80 | **+0.03 [+0.02, +0.05], p 0.002** | **+0.13 [+0.09, +0.16], p 0.002** |
| LSTM | 0.30 | 60% | 10 | 0.25 → 0.26 | 0.63 → 0.68 | +0.01 [−0.02, +0.04], p 0.625 | +0.04 [−0.01, +0.10], p 0.131 |
| LSTM | 0.30 | 80% | 10 | 0.29 → 0.27 | 0.58 → 0.59 | −0.02 [−0.07, +0.03], p 0.375 | +0.01 [−0.05, +0.07], p 0.557 |
| Transformer | 0.01 | 20% | 10 | 0.28 → 0.39 | 0.85 → 0.97 | **+0.12 [+0.02, +0.20], p 0.020** | +0.13 [−0.10, +0.37], p 0.275 |
| Transformer | 0.01 | 40% | 10 | 0.55 → 0.63 | 0.77 → 0.61 | +0.06 [−0.20, +0.31], p 0.375 | −0.16 [−0.43, +0.07], p 0.232 |
| Transformer | 0.01 | 60% | 10 | 0.76 → 1.04 | 0.65 → 0.51 | +0.26 [−0.08, +0.60], p 0.131 | **−0.17 [−0.32, −0.02], p 0.027** |
| Transformer | 0.01 | 80% | 10 | 1.04 → 1.59 | 0.51 → 0.37 | +0.44 [+0.02, +1.25], p 0.049 | −0.13 [−0.28, 0], p 0.049 |
| Transformer | 0.05 | 20% | 10 | 0.22 → 0.51 | 0.87 → 1.11 | **+0.32 [+0.19, +0.39], p 0.002** | **+0.32 [+0.07, +0.38], p 0.010** |
| Transformer | 0.05 | 40% | 10 | 0.39 → 0.75 | 0.77 → 0.79 | **+0.29 [+0.11, +0.62], p 0.002** | +0.04 [−0.14, +0.21], p 0.922 |
| Transformer | 0.05 | 60% | 10 | 0.44 → 1.02 | 0.65 → 0.65 | **+0.50 [+0.31, +0.91], p 0.002** | −0.03 [−0.13, +0.08], p 0.846 |
| Transformer | 0.05 | 80% | 10 | 0.53 → 1.36 | 0.56 → 0.37 | **+0.73 [+0.24, +1.41], p 0.004** | **−0.20 [−0.33, −0.06], p 0.020** |
| Transformer | 0.10 | 20% | 10 | 0.18 → 0.39 | 0.84 → 1.04 | **+0.20 [+0.13, +0.29], p 0.002** | +0.21 [0, +0.39], p 0.037 |
| Transformer | 0.10 | 40% | 10 | 0.27 → 0.54 | 0.74 → 0.74 | **+0.27 [+0.12, +0.41], p 0.002** | 0 [−0.15, +0.12], p 1.000 |
| Transformer | 0.10 | 60% | 10 | 0.35 → 1.14 | 0.67 → 0.87 | **+0.78 [+0.44, +1.14], p 0.004** | +0.20 [−0.02, +0.41], p 0.064 |
| Transformer | 0.10 | 80% | 10 | 0.42 → 1.07 | 0.60 → 0.65 | **+0.66 [+0.27, +1.00], p 0.004** | +0.07 [−0.11, +0.22], p 0.322 |
| Transformer | 0.30 | 20% | 10 | 0.11 → 0.23 | 0.74 → 1.02 | **+0.12 [+0.08, +0.16], p 0.002** | **+0.27 [+0.10, +0.44], p 0.004** |
| Transformer | 0.30 | 40% | 10 | 0.19 → 0.37 | 0.68 → 0.86 | **+0.18 [+0.09, +0.29], p 0.004** | +0.18 [+0.01, +0.38], p 0.049 |
| Transformer | 0.30 | 60% | 10 | 0.25 → 0.62 | 0.63 → 0.98 | **+0.37 [+0.15, +0.57], p 0.002** | **+0.35 [+0.18, +0.52], p 0.004** |
| Transformer | 0.30 | 80% | 10 | 0.29 → 0.66 | 0.58 → 0.79 | **+0.37 [+0.26, +0.47], p 0.002** | **+0.21 [+0.12, +0.30], p 0.002** |

</details>

### 10. A bigger hyperparameter search helps

**Verdict: partly.** From the archived 10-trial (LSTM) and 20-trial (Transformer) `no_ar`
runs on the same panels to the 100-trial runs, the level improves only at rate 0.30 (LSTM
−11 MAPE / −15 |bias|, Transformer −22 / −29). At rates 0.01–0.10 no rate-pooled MAPE or
|bias| change is significant. The Transformer's ranking improves at rates 0.05–0.30 (+0.03
to +0.18). The archived run did not record its embedder, so the two runs may also differ
in that.

**LSTM, 10 → 100 trials**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | −2 / −3 / 0 | −67 / −71 / +0.01 | −33 / −33 / 0 | +47 / +45 / −0.01 | −21 / −22 / 0 |
| 0.05 | −6 / −8 / −0.03 | +4 / +4 / −0.01 | −32 / −33 / −0.03 | +78 / +78 / 0 | −3 / −4 / −0.01 |
| 0.10 | −7 / −4 / +0.03 | −16 / −17 / +0.01 | −34 / −42 / 0 | −45 / −53 / −0.01 | −17 / −17 / +0.01 |
| 0.30 | −2 / −2 / **+0.03** | **−14** / **−18** / **+0.02** | −27 / −33 / **+0.01** | −9 / −16 / **+0.01** | **−11** / **−15** / **+0.02** |
| all |  |  |  |  | **−13** / **−16** / +0.01 |

**Transformer, 20 → 100 trials**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | −14 / −24 / +0.02 | +11 / +12 / −0.02 | +24 / +36 / +0.01 | −9 / −21 / +0.05 | 0 / −1 / +0.01 |
| 0.05 | +1 / +2 / +0.13 | −47 / −48 / +0.22 | −7 / −10 / +0.15 | −139 / −143 / +0.21 | −25 / −27 / **+0.18** |
| 0.10 | −20 / −21 / +0.02 | −12 / −10 / **+0.04** | −24 / −29 / **+0.06** | −1 / −1 / +0.04 | −10 / −10 / **+0.04** |
| 0.30 | +1 / −11 / +0.02 | −19 / −21 / +0.04 | −40 / −42 / +0.03 | −80 / −78 / +0.03 | **−22** / **−29** / **+0.03** |
| all |  |  |  |  | **−13** / **−17** / **+0.05** |

<details><summary>Intervals and p-values by rate, churn pooled</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI], p | Δ \|bias\| [95% CI], p | Δ RMSE [95% CI], p | Δ Spearman [95% CI], p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM, 10 → 100 trials | 0.01 | pooled | 40 | 324 → 314 | 320 → 309 | −21 [−44, +7], p 0.158 | −22 [−47, +7], p 0.135 | −0.04 [−0.08, 0], p 0.044 | 0 [−0.01, +0.01], p 0.609 |
| LSTM, 10 → 100 trials | 0.05 | pooled | 40 | 189 → 200 | 188 → 199 | −3 [−21, +16], p 0.685 | −4 [−21, +18], p 0.675 | −0.03 [−0.11, +0.06], p 0.527 | −0.01 [−0.05, +0.01], p 0.248 |
| LSTM, 10 → 100 trials | 0.10 | pooled | 40 | 122 → 98 | 121 → 93 | −17 [−40, +9], p 0.197 | −17 [−46, +9], p 0.183 | **−0.31 [−0.46, −0.11], p 0.004** | +0.01 [−0.02, +0.03], p 0.237 |
| LSTM, 10 → 100 trials | 0.30 | pooled | 40 | 44 → 21 | 40 → 13 | **−11 [−20, −6], p <10⁻⁴** | **−15 [−25, −8], p 0.0001** | **−0.90 [−1.15, −0.65], p <10⁻⁴** | **+0.02 [+0.01, +0.02], p <10⁻⁴** |
| LSTM, 10 → 100 trials | all | pooled | 160 | 170 → 158 | 167 → 154 | **−13 [−23, −5], p 0.002** | **−16 [−26, −6], p 0.001** | **−0.22 [−0.32, −0.14], p <10⁻⁴** | +0.01 [0, +0.01], p 0.049 |
| Transformer, 20 → 100 trials | 0.01 | pooled | 40 | 114 → 120 | 80 → 84 | 0 [−14, +18], p 0.952 | −1 [−23, +27], p 0.942 | 0 [−0.02, +0.02], p 0.973 | +0.01 [−0.01, +0.04], p 0.253 |
| Transformer, 20 → 100 trials | 0.05 | pooled | 40 | 165 → 114 | 163 → 109 | −25 [−58, +1], p 0.064 | −27 [−65, +2], p 0.066 | **−0.17 [−0.32, −0.04], p 0.011** | **+0.18 [+0.12, +0.24], p <10⁻⁴** |
| Transformer, 20 → 100 trials | 0.10 | pooled | 40 | 147 → 130 | 146 → 128 | −10 [−34, +16], p 0.368 | −10 [−41, +17], p 0.361 | −0.18 [−0.44, +0.04], p 0.121 | **+0.04 [+0.03, +0.05], p <10⁻⁴** |
| Transformer, 20 → 100 trials | 0.30 | pooled | 40 | 88 → 56 | 85 → 48 | **−22 [−49, −6], p 0.004** | **−29 [−54, −10], p 0.001** | **−0.77 [−1.13, −0.34], p 0.0006** | **+0.03 [+0.02, +0.04], p <10⁻⁴** |
| Transformer, 20 → 100 trials | all | pooled | 160 | 129 → 105 | 118 → 92 | **−13 [−23, −4], p 0.008** | **−17 [−30, −5], p 0.005** | **−0.18 [−0.30, −0.08], p <10⁻⁴** | **+0.05 [+0.03, +0.06], p <10⁻⁴** |

</details>

<details><summary>Intervals and p-values per rate × churn cell</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI], p | Δ \|bias\| [95% CI], p | Δ RMSE [95% CI], p | Δ Spearman [95% CI], p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM, 10 → 100 trials | 0.01 | 20% | 10 | 74 → 71 | 63 → 59 | −2 [−28, +22], p 0.922 | −3 [−36, +27], p 0.770 | −0.01 [−0.06, +0.04], p 0.625 | 0 [−0.02, +0.01], p 0.770 |
| LSTM, 10 → 100 trials | 0.01 | 40% | 10 | 186 → 136 | 185 → 132 | −67 [−76, −6], p 0.020 | −71 [−80, −6], p 0.020 | −0.14 [−0.16, −0.03], p 0.010 | +0.01 [−0.02, +0.03], p 0.557 |
| LSTM, 10 → 100 trials | 0.01 | 60% | 10 | 323 → 291 | 322 → 290 | −33 [−77, +22], p 0.160 | −33 [−80, +22], p 0.160 | −0.05 [−0.13, +0.02], p 0.105 | 0 [−0.03, +0.04], p 0.922 |
| LSTM, 10 → 100 trials | 0.01 | 80% | 10 | 712 → 759 | 710 → 755 | +47 [−79, +174], p 0.322 | +45 [−81, +178], p 0.322 | +0.04 [−0.07, +0.14], p 0.375 | −0.01 [−0.04, +0.01], p 0.160 |
| LSTM, 10 → 100 trials | 0.05 | 20% | 10 | 55 → 45 | 51 → 42 | −6 [−38, +15], p 0.695 | −8 [−39, +19], p 0.695 | −0.06 [−0.43, +0.09], p 0.557 | −0.03 [−0.26, +0.03], p 0.432 |
| LSTM, 10 → 100 trials | 0.05 | 40% | 10 | 93 → 99 | 93 → 98 | +4 [−22, +33], p 0.770 | +4 [−22, +34], p 0.846 | −0.02 [−0.19, +0.15], p 0.846 | −0.01 [−0.23, +0.19], p 0.695 |
| LSTM, 10 → 100 trials | 0.05 | 60% | 10 | 190 → 158 | 190 → 157 | −32 [−58, −5], p 0.020 | −33 [−58, −5], p 0.020 | −0.10 [−0.22, +0.02], p 0.084 | −0.03 [−0.21, +0.02], p 0.492 |
| LSTM, 10 → 100 trials | 0.05 | 80% | 10 | 418 → 497 | 418 → 497 | +78 [−46, +203], p 0.193 | +78 [−46, +203], p 0.193 | +0.17 [−0.12, +0.47], p 0.193 | 0 [−0.03, +0.03], p 0.846 |
| LSTM, 10 → 100 trials | 0.10 | 20% | 10 | 40 → 33 | 37 → 31 | −7 [−26, +11], p 0.322 | −4 [−27, +15], p 0.625 | **−0.51 [−0.66, −0.39], p 0.002** | +0.03 [0, +0.05], p 0.020 |
| LSTM, 10 → 100 trials | 0.10 | 40% | 10 | 69 → 58 | 68 → 57 | −16 [−47, +20], p 0.557 | −17 [−48, +20], p 0.557 | −0.29 [−0.76, +0.50], p 0.275 | +0.01 [−0.09, +0.04], p 0.492 |
| LSTM, 10 → 100 trials | 0.10 | 60% | 10 | 101 → 69 | 100 → 61 | −34 [−74, +10], p 0.160 | −42 [−86, +10], p 0.131 | −0.18 [−0.40, +0.02], p 0.105 | 0 [−0.02, +0.03], p 0.770 |
| LSTM, 10 → 100 trials | 0.10 | 80% | 10 | 279 → 233 | 277 → 225 | −45 [−204, +134], p 0.695 | −53 [−216, +136], p 0.695 | −0.25 [−0.85, +0.50], p 0.770 | −0.01 [−0.36, +0.23], p 0.922 |
| LSTM, 10 → 100 trials | 0.30 | 20% | 10 | 22 → 19 | 20 → 17 | −2 [−14, +6], p 0.695 | −2 [−14, +9], p 0.846 | **−0.59 [−0.93, −0.32], p 0.002** | **+0.03 [+0.02, +0.04], p 0.002** |
| LSTM, 10 → 100 trials | 0.30 | 40% | 10 | 33 → 14 | 32 → 9 | **−14 [−35, −9], p 0.002** | **−18 [−41, −12], p 0.002** | **−1.61 [−1.99, −1.15], p 0.002** | **+0.02 [+0.01, +0.03], p 0.002** |
| LSTM, 10 → 100 trials | 0.30 | 60% | 10 | 60 → 18 | 55 → 9 | −27 [−74, −3], p 0.020 | −33 [−84, −4], p 0.027 | −0.95 [−1.63, −0.24], p 0.010 | **+0.01 [0, +0.01], p 0.004** |
| LSTM, 10 → 100 trials | 0.30 | 80% | 10 | 62 → 32 | 55 → 17 | −9 [−111, +5], p 0.160 | −16 [−128, +4], p 0.160 | −0.54 [−1.00, −0.11], p 0.037 | **+0.01 [0, +0.01], p 0.002** |
| Transformer, 20 → 100 trials | 0.01 | 20% | 10 | 72 → 58 | 60 → 37 | −14 [−32, +4], p 0.105 | −24 [−55, +8], p 0.131 | −0.03 [−0.07, +0.01], p 0.193 | +0.02 [−0.02, +0.05], p 0.232 |
| Transformer, 20 → 100 trials | 0.01 | 40% | 10 | 84 → 96 | 67 → 76 | +11 [−32, +55], p 0.557 | +12 [−51, +71], p 0.625 | +0.03 [−0.04, +0.10], p 0.557 | −0.02 [−0.07, +0.03], p 0.492 |
| Transformer, 20 → 100 trials | 0.01 | 60% | 10 | 92 → 129 | 47 → 102 | +24 [−5, +89], p 0.105 | +36 [−8, +121], p 0.131 | +0.02 [0, +0.09], p 0.105 | +0.01 [−0.03, +0.07], p 0.557 |
| Transformer, 20 → 100 trials | 0.01 | 80% | 10 | 209 → 197 | 145 → 121 | −9 [−54, +25], p 0.625 | −21 [−89, +41], p 0.492 | 0 [−0.03, +0.01], p 0.625 | +0.05 [−0.01, +0.11], p 0.232 |
| Transformer, 20 → 100 trials | 0.05 | 20% | 10 | 76 → 75 | 74 → 72 | +1 [−21, +20], p 0.922 | +2 [−29, +21], p 0.922 | −0.01 [−0.27, +0.20], p 0.922 | +0.13 [−0.02, +0.30], p 0.131 |
| Transformer, 20 → 100 trials | 0.05 | 40% | 10 | 151 → 110 | 151 → 108 | −47 [−77, −4], p 0.037 | −48 [−79, −4], p 0.037 | −0.39 [−0.70, −0.07], p 0.037 | +0.22 [+0.08, +0.37], p 0.010 |
| Transformer, 20 → 100 trials | 0.05 | 60% | 10 | 168 → 149 | 167 → 144 | −7 [−94, +57], p 1.000 | −10 [−106, +58], p 1.000 | −0.05 [−0.50, +0.21], p 0.770 | +0.15 [+0.05, +0.24], p 0.027 |
| Transformer, 20 → 100 trials | 0.05 | 80% | 10 | 265 → 122 | 260 → 111 | −139 [−304, +29], p 0.160 | −143 [−323, +37], p 0.160 | −0.22 [−0.56, −0.02], p 0.020 | +0.21 [+0.11, +0.35], p 0.010 |
| Transformer, 20 → 100 trials | 0.10 | 20% | 10 | 77 → 58 | 76 → 56 | −20 [−48, +12], p 0.193 | −21 [−53, +13], p 0.193 | −0.34 [−1.13, +0.07], p 0.105 | +0.02 [−0.03, +0.07], p 0.492 |
| Transformer, 20 → 100 trials | 0.10 | 40% | 10 | 100 → 94 | 99 → 91 | −12 [−32, +37], p 0.492 | −10 [−41, +36], p 0.557 | −0.20 [−0.84, +0.32], p 0.432 | **+0.04 [+0.02, +0.09], p 0.004** |
| Transformer, 20 → 100 trials | 0.10 | 60% | 10 | 127 → 103 | 124 → 99 | −24 [−78, +38], p 0.375 | −29 [−86, +41], p 0.375 | −0.17 [−0.76, +0.18], p 0.432 | **+0.06 [+0.03, +0.21], p 0.004** |
| Transformer, 20 → 100 trials | 0.10 | 80% | 10 | 285 → 265 | 284 → 265 | −1 [−159, +116], p 1.000 | −1 [−159, +116], p 1.000 | +0.10 [−0.61, +0.42], p 0.695 | +0.04 [+0.01, +0.06], p 0.014 |
| Transformer, 20 → 100 trials | 0.30 | 20% | 10 | 34 → 36 | 31 → 24 | +1 [−12, +18], p 1.000 | −11 [−25, +17], p 0.375 | −0.49 [−1.35, +0.56], p 0.193 | +0.02 [0, +0.14], p 0.160 |
| Transformer, 20 → 100 trials | 0.30 | 40% | 10 | 58 → 36 | 55 → 29 | −19 [−41, −2], p 0.037 | −21 [−52, −2], p 0.049 | −0.61 [−1.36, +0.11], p 0.105 | +0.04 [+0.01, +0.07], p 0.020 |
| Transformer, 20 → 100 trials | 0.30 | 60% | 10 | 96 → 57 | 94 → 50 | −40 [−80, +4], p 0.105 | −42 [−93, +7], p 0.064 | −0.92 [−1.98, −0.02], p 0.049 | +0.03 [+0.01, +0.06], p 0.027 |
| Transformer, 20 → 100 trials | 0.30 | 80% | 10 | 166 → 96 | 160 → 88 | −80 [−167, +2], p 0.084 | −78 [−172, +4], p 0.160 | −0.91 [−1.74, +0.07], p 0.084 | +0.03 [0, +0.06], p 0.049 |

</details>

### 11. Specific hyperparameters drive the error

**Verdict: not supported.** Four pooled correlations are significant, but none survives
within cells. For example, the LSTM `no_ar` batch size correlates with |bias| at +0.48 over
the grid and +0.04 within cells: the search picks larger batches on sparse panels, where
|bias| is large anyway. The closest to a within-cell effect is the Transformer `no_ar`
layer count against MAPE (+0.21, p = 0.008), which does not survive the correction across
the table's 88 tests. "Within cells" ranks both variables inside each rate × churn cell
before correlating.

| Tree | Hyperparameter | ρ with \|bias\|, pooled | ρ with \|bias\|, within cells | ρ with MAPE, pooled | ρ with MAPE, within cells |
| --- | --- | --- | --- | --- | --- |
| LSTM `no_ar` | `batch_size` | **+0.48, p <10⁻⁴** | +0.04, p 0.655 | **+0.49, p <10⁻⁴** | +0.05, p 0.513 |
| LSTM `no_ar` | `learning_rate` | −0.15, p 0.056 | +0.12, p 0.123 | −0.16, p 0.043 | +0.12, p 0.133 |
| LSTM `no_ar` | `lstm_hidden_size` | −0.20, p 0.012 | +0.10, p 0.190 | −0.21, p 0.007 | +0.08, p 0.346 |
| LSTM `no_ar` | `dense_units` | **+0.28, p 0.0003** | −0.05, p 0.508 | **+0.28, p 0.0003** | −0.06, p 0.469 |
| LSTM `no_ar` | `dropout` | −0.05, p 0.524 | +0.14, p 0.069 | −0.05, p 0.567 | +0.13, p 0.102 |
| LSTM `ar_bounded` | `batch_size` | +0.07, p 0.390 | +0.06, p 0.485 | +0.06, p 0.427 | +0.08, p 0.300 |
| LSTM `ar_bounded` | `learning_rate` | −0.02, p 0.803 | +0.10, p 0.221 | −0.10, p 0.214 | +0.03, p 0.669 |
| LSTM `ar_bounded` | `lstm_hidden_size` | 0, p 0.969 | +0.01, p 0.892 | −0.04, p 0.616 | +0.04, p 0.582 |
| LSTM `ar_bounded` | `dense_units` | **−0.28, p 0.0003** | −0.15, p 0.061 | −0.22, p 0.006 | −0.19, p 0.014 |
| LSTM `ar_bounded` | `dropout` | −0.02, p 0.787 | +0.07, p 0.354 | +0.03, p 0.736 | +0.10, p 0.230 |
| Transformer `no_ar` | `batch_size` | −0.04, p 0.637 | −0.07, p 0.366 | −0.03, p 0.730 | −0.06, p 0.457 |
| Transformer `no_ar` | `learning_rate` | −0.09, p 0.270 | −0.03, p 0.740 | +0.05, p 0.557 | +0.03, p 0.710 |
| Transformer `no_ar` | `d_model` | −0.01, p 0.886 | +0.01, p 0.928 | −0.03, p 0.675 | +0.01, p 0.934 |
| Transformer `no_ar` | `nhead` | +0.08, p 0.285 | +0.02, p 0.798 | +0.05, p 0.552 | +0.02, p 0.843 |
| Transformer `no_ar` | `num_encoder_layers` | **+0.24, p 0.002** | +0.18, p 0.020 | **+0.29, p 0.0002** | +0.21, p 0.008 |
| Transformer `no_ar` | `dropout` | +0.17, p 0.027 | +0.08, p 0.297 | +0.13, p 0.113 | +0.09, p 0.240 |
| Transformer `ar_bounded` | `batch_size` | +0.03, p 0.699 | +0.02, p 0.818 | +0.04, p 0.639 | −0.01, p 0.872 |
| Transformer `ar_bounded` | `learning_rate` | −0.05, p 0.545 | −0.14, p 0.088 | +0.13, p 0.100 | −0.07, p 0.349 |
| Transformer `ar_bounded` | `d_model` | −0.13, p 0.098 | −0.08, p 0.325 | −0.13, p 0.093 | −0.08, p 0.325 |
| Transformer `ar_bounded` | `nhead` | −0.05, p 0.541 | −0.06, p 0.480 | −0.01, p 0.892 | −0.08, p 0.312 |
| Transformer `ar_bounded` | `num_encoder_layers` | −0.01, p 0.946 | 0, p 0.996 | −0.10, p 0.193 | +0.02, p 0.845 |
| Transformer `ar_bounded` | `dropout` | −0.01, p 0.927 | −0.04, p 0.603 | +0.03, p 0.687 | −0.06, p 0.472 |

### 12. More customers improve the neural forecast

**Verdict: supported, conditionally.** Tripling the cohort (1,000 → 3,000 customers,
different panels, so unpaired tests):

- **LSTM `ar_bounded`:** MAPE improves in every cell at rates 0.01–0.10 (−12 to −298
  points); at rate 0.30 only at churn 20% (−5). Ranking improves only at rate 0.01 (+0.12).
- **LSTM `no_ar`:** MAPE improves at rates 0.05–0.30 but not at 0.01 (−11, p = 0.74).
  Without flags, extra customers do not help the sparsest panels.
- **Pareto/NBD:** MAPE improves modestly (−1 to −22 points rate-pooled, most at high
  churn); RMSE does not move.
- The Transformer was not run at 3,000 customers.

**LSTM `no_ar`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | +16 / **+27** / −0.02 | −7 / −1 / −0.01 | −40 / −36 / −0.02 | **−282** / −281 / +0.03 | −11 / −9 / −0.01 |
| 0.05 | **−19** / **−16** / **+0.06** | **−31** / **−30** / +0.42 | **−42** / −42 / **+0.39** | **−188** / **−188** / **+0.04** | **−47** / **−48** / **+0.37** |
| 0.10 | **−14** / **−14** / +0.01 | **−15** / −16 / +0.01 | **−32** / **−36** / **+0.04** | **−186** / −197 / **+0.21** | **−24** / **−27** / +0.02 |
| 0.30 | **−12** / **−14** / **+0.01** | **−6** / −5 / +0.01 | **−6** / −2 / −0.01 | −5 / +5 / 0 | **−7** / −4 / +0.01 |
| all |  |  |  |  | **−19** / **−17** / +0.03 |

**LSTM `ar_bounded`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **−21** / −19 / +0.07 | **−66** / **−75** / **+0.18** | **−89** / **−101** / **+0.11** | **−298** / **−333** / **+0.15** | **−72** / **−83** / **+0.12** |
| 0.05 | **−22** / **−22** / +0.04 | **−25** / **−27** / +0.01 | **−35** / **−45** / +0.01 | **−22** / −6 / +0.03 | **−26** / **−26** / +0.02 |
| 0.10 | **−17** / **−20** / +0.02 | **−18** / **−20** / +0.01 | **−16** / **−13** / +0.01 | **−12** / −1 / +0.01 | **−18** / **−13** / +0.01 |
| 0.30 | **−5** / **−6** / +0.01 | −3 / 0 / +0.01 | −1 / +2 / −0.01 | −5 / +4 / 0 | −4 / 0 / +0.01 |
| all |  |  |  |  | **−18** / **−13** / +0.02 |

**Pareto/NBD**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **−11** / −2 / −0.01 | **−20** / **−14** / +0.04 | **−26** / −11 / +0.02 | **−77** / **−28** / **+0.08** | **−22** / **−11** / **+0.03** |
| 0.05 | **−3** / −1 / +0.01 | **−5** / −6 / 0 | **−10** / −6 / 0 | **−20** / −11 / **+0.03** | **−7** / **−5** / +0.01 |
| 0.10 | −1 / −1 / 0 | −2 / **−4** / −0.01 | **−5** / −4 / −0.01 | **−11** / **−11** / +0.01 | **−3** / **−3** / 0 |
| 0.30 | 0 / +1 / 0 | −1 / 0 / 0 | −2 / 0 / 0 | **−4** / 0 / 0 | **−1** / 0 / 0 |
| all |  |  |  |  | **−4** / **−4** / +0.01 |

<details><summary>Intervals and p-values by rate, churn pooled</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI], p | Δ \|bias\| [95% CI], p | Δ RMSE [95% CI], p | Δ Spearman [95% CI], p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM `no_ar` | 0.01 | pooled | 40 / 40 | 314 → 219 | 309 → 219 | −11 [−78, +43], p 0.740 | −9 [−77, +52], p 0.889 | −0.04 [−0.10, 0], p 0.077 | −0.01 [−0.03, +0.02], p 0.669 |
| LSTM `no_ar` | 0.05 | pooled | 40 / 40 | 200 → 119 | 199 → 118 | **−47 [−89, −12], p 0.009** | **−48 [−92, −12], p 0.010** | **−0.30 [−0.43, −0.17], p <10⁻⁴** | **+0.37 [+0.05, +0.44], p <10⁻⁴** |
| LSTM `no_ar` | 0.10 | pooled | 40 / 40 | 98 → 24 | 93 → 19 | **−24 [−35, −16], p <10⁻⁴** | **−27 [−39, −18], p <10⁻⁴** | **−0.30 [−0.65, −0.05], p 0.022** | +0.02 [0, +0.07], p 0.097 |
| LSTM `no_ar` | 0.30 | pooled | 40 / 40 | 21 → 14 | 13 → 10 | **−7 [−10, −5], p <10⁻⁴** | −4 [−8, 0], p 0.054 | −0.30 [−0.96, +0.22], p 0.283 | +0.01 [−0.02, +0.04], p 0.683 |
| LSTM `no_ar` | all | pooled | 160 / 160 | 158 → 94 | 154 → 91 | **−19 [−31, −9], p <10⁻⁴** | **−17 [−30, −6], p 0.002** | −0.19 [−0.51, +0.01], p 0.073 | +0.03 [−0.01, +0.08], p 0.113 |
| LSTM `ar_bounded` | 0.01 | pooled | 40 / 40 | 243 → 83 | 232 → 65 | **−72 [−110, −41], p <10⁻⁴** | **−83 [−117, −48], p <10⁻⁴** | **−0.11 [−0.19, −0.06], p 0.0002** | **+0.12 [+0.09, +0.16], p <10⁻⁴** |
| LSTM `ar_bounded` | 0.05 | pooled | 40 / 40 | 58 → 31 | 46 → 22 | **−26 [−33, −20], p <10⁻⁴** | **−26 [−35, −17], p <10⁻⁴** | −0.10 [−0.33, +0.09], p 0.271 | +0.02 [−0.01, +0.04], p 0.111 |
| LSTM `ar_bounded` | 0.10 | pooled | 40 / 40 | 36 → 18 | 24 → 9 | **−18 [−22, −13], p <10⁻⁴** | **−13 [−19, −8], p <10⁻⁴** | −0.17 [−0.52, +0.13], p 0.209 | +0.01 [−0.01, +0.04], p 0.301 |
| LSTM `ar_bounded` | 0.30 | pooled | 40 / 40 | 20 → 16 | 11 → 12 | −4 [−7, 0], p 0.044 | 0 [−4, +4], p 0.958 | −0.17 [−0.77, +0.30], p 0.580 | +0.01 [−0.03, +0.04], p 0.704 |
| LSTM `ar_bounded` | all | pooled | 160 / 160 | 89 → 37 | 78 → 27 | **−18 [−23, −11], p <10⁻⁴** | **−13 [−20, −7], p <10⁻⁴** | −0.12 [−0.35, +0.12], p 0.300 | +0.02 [−0.01, +0.07], p 0.203 |
| Pareto/NBD | 0.01 | pooled | 40 / 40 | 90 → 56 | 35 → 21 | **−22 [−35, −11], p <10⁻⁴** | **−11 [−19, −3], p 0.004** | −0.01 [−0.08, +0.06], p 0.806 | **+0.03 [+0.01, +0.05], p 0.006** |
| Pareto/NBD | 0.05 | pooled | 40 / 40 | 44 → 34 | 14 → 8 | **−7 [−10, −4], p <10⁻⁴** | **−5 [−8, −1], p 0.006** | −0.01 [−0.17, +0.19], p 0.836 | +0.01 [−0.01, +0.02], p 0.373 |
| Pareto/NBD | 0.10 | pooled | 40 / 40 | 37 → 32 | 9 → 4 | **−3 [−5, −1], p 0.0004** | **−3 [−5, −1], p 0.002** | −0.02 [−0.27, +0.26], p 0.836 | 0 [−0.02, +0.02], p 0.799 |
| Pareto/NBD | 0.30 | pooled | 40 / 40 | 32 → 30 | 13 → 13 | **−1 [−2, 0], p 0.001** | 0 [−2, +2], p 0.821 | −0.03 [−0.71, +0.55], p 0.912 | 0 [−0.02, +0.02], p 0.973 |
| Pareto/NBD | all | pooled | 160 / 160 | 51 → 38 | 18 → 12 | **−4 [−6, −2], p <10⁻⁴** | **−4 [−6, −2], p 0.0003** | −0.01 [−0.25, +0.24], p 0.902 | +0.01 [−0.03, +0.05], p 0.672 |

</details>

<details><summary>Intervals and p-values per rate × churn cell</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI], p | Δ \|bias\| [95% CI], p | Δ RMSE [95% CI], p | Δ Spearman [95% CI], p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM `no_ar` | 0.01 | 20% | 10 / 10 | 71 → 84 | 59 → 83 | +16 [+4, +27], p 0.026 | **+27 [+11, +40], p 0.006** | +0.04 [−0.03, +0.08], p 0.140 | −0.02 [−0.05, +0.02], p 0.273 |
| LSTM `no_ar` | 0.01 | 40% | 10 / 10 | 136 → 132 | 132 → 131 | −7 [−22, +17], p 0.623 | −1 [−21, +21], p 0.850 | +0.02 [−0.02, +0.07], p 0.186 | −0.01 [−0.06, +0.03], p 0.473 |
| LSTM `no_ar` | 0.01 | 60% | 10 / 10 | 291 → 261 | 290 → 261 | −40 [−76, +18], p 0.140 | −36 [−76, +22], p 0.186 | −0.05 [−0.13, +0.03], p 0.186 | −0.02 [−0.07, +0.04], p 0.850 |
| LSTM `no_ar` | 0.01 | 80% | 10 / 10 | 759 → 399 | 755 → 399 | **−282 [−576, −54], p 0.021** | −281 [−580, −47], p 0.026 | **−0.24 [−0.34, −0.13], p 0.0008** | +0.03 [−0.02, +0.09], p 0.186 |
| LSTM `no_ar` | 0.05 | 20% | 10 / 10 | 45 → 26 | 42 → 25 | **−19 [−30, −7], p 0.005** | **−16 [−29, −4], p 0.014** | **−0.30 [−0.51, −0.06], p 0.005** | **+0.06 [+0.02, +0.51], p 0.011** |
| LSTM `no_ar` | 0.05 | 40% | 10 / 10 | 99 → 69 | 98 → 69 | **−31 [−50, −10], p 0.011** | **−30 [−49, −9], p 0.011** | **−0.30 [−0.41, −0.16], p 0.001** | +0.42 [0, +0.48], p 0.038 |
| LSTM `no_ar` | 0.05 | 60% | 10 / 10 | 158 → 105 | 157 → 102 | **−42 [−106, −4], p 0.021** | −42 [−114, −4], p 0.026 | −0.27 [−0.51, −0.03], p 0.026 | **+0.39 [+0.03, +0.48], p 0.011** |
| LSTM `no_ar` | 0.05 | 80% | 10 / 10 | 497 → 277 | 497 → 274 | **−188 [−330, −117], p 0.001** | **−188 [−330, −117], p 0.001** | **−0.40 [−0.61, −0.21], p 0.0008** | **+0.04 [+0.01, +0.08], p 0.021** |
| LSTM `no_ar` | 0.10 | 20% | 10 / 10 | 33 → 18 | 31 → 17 | **−14 [−24, −6], p 0.003** | **−14 [−22, −5], p 0.014** | **−0.23 [−0.38, −0.10], p 0.021** | +0.01 [0, +0.03], p 0.076 |
| LSTM `no_ar` | 0.10 | 40% | 10 / 10 | 58 → 29 | 57 → 27 | **−15 [−32, −4], p 0.021** | −16 [−34, −3], p 0.026 | −0.19 [−0.46, +0.01], p 0.076 | +0.01 [−0.01, +0.04], p 0.521 |
| LSTM `no_ar` | 0.10 | 60% | 10 / 10 | 69 → 17 | 61 → 8 | **−32 [−92, −22], p 0.0002** | **−36 [−99, −24], p 0.011** | **−0.18 [−0.41, −0.07], p 0.006** | **+0.04 [+0.01, +0.05], p 0.003** |
| LSTM `no_ar` | 0.10 | 80% | 10 / 10 | 233 → 32 | 225 → 22 | **−186 [−278, −33], p 0.0003** | −197 [−290, −36], p 0.026 | **−0.63 [−0.88, −0.06], p 0.009** | **+0.21 [+0.02, +0.40], p 0.009** |
| LSTM `no_ar` | 0.30 | 20% | 10 / 10 | 19 → 8 | 17 → 6 | **−12 [−16, −7], p 0.0006** | **−14 [−18, −8], p 0.011** | **−0.69 [−1.05, −0.32], p 0.002** | **+0.01 [0, +0.02], p 0.021** |
| LSTM `no_ar` | 0.30 | 40% | 10 / 10 | 14 → 9 | 9 → 4 | **−6 [−9, −3], p 0.0008** | −5 [−9, 0], p 0.045 | −0.26 [−0.56, +0.02], p 0.064 | +0.01 [0, +0.02], p 0.121 |
| LSTM `no_ar` | 0.30 | 60% | 10 / 10 | 18 → 12 | 9 → 7 | **−6 [−8, −2], p 0.004** | −2 [−8, +4], p 0.521 | −0.21 [−0.62, +0.06], p 0.104 | −0.01 [−0.02, +0.01], p 0.212 |
| LSTM `no_ar` | 0.30 | 80% | 10 / 10 | 32 → 28 | 17 → 23 | −5 [−10, +3], p 0.121 | +5 [−2, +15], p 0.104 | −0.06 [−0.41, +0.28], p 0.678 | 0 [−0.02, +0.02], p 0.571 |
| LSTM `ar_bounded` | 0.01 | 20% | 10 / 10 | 65 → 38 | 47 → 25 | **−21 [−37, −12], p 0.001** | −19 [−40, +1], p 0.054 | −0.03 [−0.09, +0.01], p 0.162 | +0.07 [0, +0.14], p 0.045 |
| LSTM `ar_bounded` | 0.01 | 40% | 10 / 10 | 121 → 52 | 114 → 37 | **−66 [−85, −49], p 0.0002** | **−75 [−97, −50], p 0.0002** | **−0.07 [−0.10, −0.04], p 0.003** | **+0.18 [+0.10, +0.25], p 0.0008** |
| LSTM `ar_bounded` | 0.01 | 60% | 10 / 10 | 196 → 89 | 190 → 72 | **−89 [−152, −54], p 0.0004** | **−101 [−162, −58], p 0.0006** | **−0.12 [−0.20, −0.06], p 0.0008** | **+0.11 [+0.03, +0.23], p 0.007** |
| LSTM `ar_bounded` | 0.01 | 80% | 10 / 10 | 591 → 152 | 576 → 127 | **−298 [−665, −146], p 0.0008** | **−333 [−691, −124], p 0.004** | **−0.26 [−0.38, −0.10], p 0.002** | **+0.15 [+0.04, +0.23], p 0.017** |
| LSTM `ar_bounded` | 0.05 | 20% | 10 / 10 | 48 → 26 | 47 → 24 | **−22 [−33, −11], p 0.001** | **−22 [−36, −11], p 0.006** | **−0.30 [−0.36, −0.22], p 0.0002** | +0.04 [−0.01, +0.07], p 0.104 |
| LSTM `ar_bounded` | 0.05 | 40% | 10 / 10 | 56 → 29 | 52 → 25 | **−25 [−40, −13], p 0.003** | **−27 [−44, −8], p 0.003** | **−0.15 [−0.21, −0.06], p 0.004** | +0.01 [−0.03, +0.05], p 0.734 |
| LSTM `ar_bounded` | 0.05 | 60% | 10 / 10 | 67 → 33 | 61 → 22 | **−35 [−51, −17], p 0.001** | **−45 [−62, −16], p 0.014** | −0.04 [−0.13, +0.03], p 0.212 | +0.01 [−0.01, +0.04], p 0.241 |
| LSTM `ar_bounded` | 0.05 | 80% | 10 / 10 | 60 → 37 | 27 → 16 | **−22 [−28, −12], p 0.0004** | −6 [−26, +8], p 0.307 | +0.01 [−0.08, +0.10], p 0.970 | +0.03 [−0.01, +0.05], p 0.076 |
| LSTM `ar_bounded` | 0.10 | 20% | 10 / 10 | 32 → 14 | 30 → 11 | **−17 [−30, −7], p 0.003** | **−20 [−33, −3], p 0.011** | **−0.33 [−0.49, −0.20], p 0.0003** | +0.02 [0, +0.03], p 0.054 |
| LSTM `ar_bounded` | 0.10 | 40% | 10 / 10 | 34 → 12 | 28 → 6 | **−18 [−35, −13], p 0.0002** | **−20 [−39, −11], p 0.005** | **−0.21 [−0.35, −0.13], p 0.002** | +0.01 [−0.01, +0.04], p 0.385 |
| LSTM `ar_bounded` | 0.10 | 60% | 10 / 10 | 33 → 16 | 21 → 7 | **−16 [−20, −11], p 0.0002** | **−13 [−23, −6], p 0.004** | −0.04 [−0.15, +0.05], p 0.427 | +0.01 [−0.01, +0.02], p 0.427 |
| LSTM `ar_bounded` | 0.10 | 80% | 10 / 10 | 45 → 32 | 15 → 14 | **−12 [−20, −6], p 0.0006** | −1 [−11, +8], p 0.970 | −0.08 [−0.19, +0.12], p 0.385 | +0.01 [−0.02, +0.02], p 0.427 |
| LSTM `ar_bounded` | 0.30 | 20% | 10 / 10 | 12 → 6 | 10 → 3 | **−5 [−7, −3], p 0.0002** | **−6 [−10, −3], p 0.002** | **−0.35 [−0.83, −0.05], p 0.017** | +0.01 [0, +0.02], p 0.038 |
| LSTM `ar_bounded` | 0.30 | 40% | 10 / 10 | 13 → 10 | 6 → 7 | −3 [−5, −1], p 0.026 | 0 [−3, +5], p 0.910 | −0.27 [−0.49, +0.09], p 0.121 | +0.01 [−0.01, +0.02], p 0.241 |
| LSTM `ar_bounded` | 0.30 | 60% | 10 / 10 | 19 → 18 | 10 → 13 | −1 [−5, +3], p 0.678 | +2 [−4, +12], p 0.385 | −0.08 [−0.39, +0.13], p 0.678 | −0.01 [−0.02, +0.01], p 0.385 |
| LSTM `ar_bounded` | 0.30 | 80% | 10 / 10 | 35 → 31 | 19 → 24 | −5 [−12, +2], p 0.162 | +4 [−5, +16], p 0.385 | +0.16 [−0.07, +0.39], p 0.162 | 0 [−0.01, +0.03], p 0.571 |
| Pareto/NBD | 0.01 | 20% | 10 / 10 | 48 → 37 | 12 → 11 | **−11 [−17, −6], p 0.0008** | −2 [−7, +4], p 0.473 | 0 [−0.03, +0.02], p 0.791 | −0.01 [−0.03, +0.03], p 0.571 |
| Pareto/NBD | 0.01 | 40% | 10 / 10 | 66 → 46 | 34 → 22 | **−20 [−25, −13], p 0.0002** | **−14 [−22, −6], p 0.006** | 0 [−0.03, +0.04], p 1.000 | +0.04 [0, +0.07], p 0.031 |
| Pareto/NBD | 0.01 | 60% | 10 / 10 | 88 → 62 | 41 → 30 | **−26 [−33, −17], p 0.0002** | −11 [−27, +8], p 0.273 | −0.02 [−0.06, +0.02], p 0.427 | +0.02 [−0.03, +0.07], p 0.521 |
| Pareto/NBD | 0.01 | 80% | 10 / 10 | 159 → 78 | 54 → 22 | **−77 [−116, −49], p 0.0002** | **−28 [−56, −3], p 0.021** | −0.01 [−0.07, +0.05], p 0.623 | **+0.08 [+0.02, +0.16], p 0.001** |
| Pareto/NBD | 0.05 | 20% | 10 / 10 | 35 → 32 | 8 → 7 | **−3 [−6, −1], p 0.001** | −1 [−5, +2], p 0.571 | −0.02 [−0.08, +0.01], p 0.121 | +0.01 [−0.02, +0.03], p 0.473 |
| Pareto/NBD | 0.05 | 40% | 10 / 10 | 39 → 33 | 15 → 10 | **−5 [−8, −3], p 0.0003** | −6 [−10, +1], p 0.054 | 0 [−0.03, +0.05], p 0.734 | 0 [−0.02, +0.02], p 0.678 |
| Pareto/NBD | 0.05 | 60% | 10 / 10 | 42 → 34 | 12 → 7 | **−10 [−13, −5], p 0.002** | −6 [−14, +2], p 0.186 | −0.02 [−0.09, +0.04], p 0.623 | 0 [−0.01, +0.02], p 0.623 |
| Pareto/NBD | 0.05 | 80% | 10 / 10 | 58 → 39 | 21 → 8 | **−20 [−27, −11], p 0.0003** | −11 [−26, 0], p 0.045 | +0.03 [−0.04, +0.13], p 0.427 | **+0.03 [0, +0.05], p 0.021** |
| Pareto/NBD | 0.10 | 20% | 10 / 10 | 31 → 30 | 3 → 2 | −1 [−2, 0], p 0.064 | −1 [−3, +1], p 0.850 | −0.05 [−0.12, +0.02], p 0.089 | 0 [−0.01, +0.02], p 0.678 |
| Pareto/NBD | 0.10 | 40% | 10 / 10 | 32 → 30 | 5 → 2 | −2 [−3, 0], p 0.064 | **−4 [−5, −2], p 0.006** | +0.01 [−0.10, +0.12], p 0.623 | −0.01 [−0.02, +0.01], p 0.186 |
| Pareto/NBD | 0.10 | 60% | 10 / 10 | 37 → 32 | 9 → 3 | **−5 [−8, −3], p 0.0008** | −4 [−9, 0], p 0.076 | +0.01 [−0.09, +0.09], p 0.791 | −0.01 [−0.02, +0.01], p 0.273 |
| Pareto/NBD | 0.10 | 80% | 10 / 10 | 49 → 35 | 21 → 7 | **−11 [−18, −7], p 0.0003** | **−11 [−18, −2], p 0.017** | −0.06 [−0.19, +0.09], p 0.473 | +0.01 [−0.01, +0.03], p 0.345 |
| Pareto/NBD | 0.30 | 20% | 10 / 10 | 30 → 30 | 14 → 15 | 0 [−1, +1], p 0.734 | +1 [−1, +2], p 0.212 | +0.02 [−0.23, +0.25], p 0.850 | 0 [−0.01, +0.01], p 0.791 |
| Pareto/NBD | 0.30 | 40% | 10 / 10 | 31 → 30 | 14 → 14 | −1 [−2, +1], p 0.241 | 0 [−3, +3], p 0.970 | −0.11 [−0.41, +0.16], p 0.427 | 0 [−0.01, +0.02], p 0.678 |
| Pareto/NBD | 0.30 | 60% | 10 / 10 | 32 → 30 | 13 → 13 | −2 [−3, 0], p 0.038 | 0 [−4, +4], p 0.910 | −0.10 [−0.29, +0.09], p 0.473 | 0 [−0.01, 0], p 0.623 |
| Pareto/NBD | 0.30 | 80% | 10 / 10 | 35 → 31 | 13 → 12 | **−4 [−6, −2], p 0.001** | 0 [−10, +8], p 0.910 | +0.09 [−0.17, +0.30], p 0.623 | 0 [−0.02, +0.02], p 0.850 |

</details>

### 13. RMSE can rank these models

**Verdict: not supported for per-week RMSE; supported within one rate for customer-total
RMSE.** Per customer-week, RMSE puts 10 of 13 trees on the same value to two decimals at
rates 0.01–0.05, because almost every cell is a zero. On customer totals RMSE separates
the trees and orders them almost as MAPE does (ρ over the 13 trees +0.88 to +0.97). It grows
about 8-fold from rate 0.01 to 0.30, though, so it cannot be pooled across rates.

| Rate | Per-week RMSE, range over trees | Trees at the modal per-week RMSE (2 dp) | Customer-total RMSE, range | ρ(customer-total RMSE, MAPE) over trees |
| --- | --- | --- | --- | --- |
| 0.01 | 0.071–0.209 | 10 of 13 | 0.62–7.05 | +0.97 |
| 0.05 | 0.135–0.309 | 10 of 13 | 1.44–11.88 | +0.90 |
| 0.10 | 0.181–0.351 | 7 of 13 | 2.19–12.71 | +0.88 |
| 0.30 | 0.310–0.487 | 4 of 13 | 4.94–16.34 | +0.88 |

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
