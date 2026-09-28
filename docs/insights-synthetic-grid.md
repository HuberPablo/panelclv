# Insights from the Pareto/NBD synthetic grid studies

As of 2026-09-28. Exported from the Claude Doc
<https://claude.ai/code/artifact/bff00b0b-93dd-4be8-8f71-5ea53e3d965b>; all numbers were
recomputed from the stored forecasts under `Studies/seasonal_4x4x10*`.

**Revised under the statistical protocol (2026-09-28).** Every test now follows
`docs/statistical-protocol.md`: Δ is the difference of means with a 95% percentile-bootstrap
interval (`evaluation.effects.effect`), run per rate × churn cell only (paired by panel;
independent for 1,000 vs 3,000 customers). Wilcoxon, Mann–Whitney, Hodges–Lehmann,
Benjamini–Hochberg, McNemar and t-intervals are gone, and so are all p-values. Rate, churn
and whole-grid results are now described, not tested, and the pooled tables carry no marks.
Δ values changed throughout because they are now means, not medians. Verdicts that moved:

- Claim 1: Pareto/NBD ranks better in 94 of 96 cell comparisons, not all of them. At 3,000
  customers, rate 0.30, churn 20%, both LSTM arms rank slightly better (+0.003, supported).
  At 3,000 customers and rate 0.05, the LSTM's MAPE lead over Pareto/NBD (supported
  rate-pooled before) is clear in no cell.
- Claim 2: was tested by a churn correlation with FDR control; it is now a described pattern.
- Claim 4: the Transformer is also hurt at rate 0.10 (two cells), not only at rate 0.30.
- Claim 5: with the unbounded counters, the label no longer "helps whenever": it hurts the
  LSTM badly in two rate-0.01 cells.
- Claim 7: the true season lowers Pareto/NBD's MAPE in 16 of 16 cells (was 15). Against the
  seasonal Pareto/NBD at rate 0.30, the LSTM is no longer "level": it keeps a small lead at
  churn 20–40% and loses at churn 80%.
- Claim 9: the LSTM leaks more than Pareto/NBD at rate 0.10, churn 80% (was "equal"); the
  Transformer leaks more in 15 of 16 cells (was 13).
- Claim 11: was "not supported" by within-cell tests with FDR control; it is now described
  per cell with no test (a correlation over panels is not a difference of means), and the
  verdict stays "not supported". The Transformer layer count is the one consistent pattern
  (positive in 13 of 16 cells).
- Claim 12: the LSTM `no_ar` at rate 0.01 goes from "no change" to mixed (worse at churn
  20%, better at churn 80%).
- Results: the MAPE leader is clearly ahead of the runner-up in 7 of 16 cells (was 6; rate
  0.30, churn 40% is new). Fewer per-cell entries are marked tied with the best (47 of 1,024
  metric entries against 85).

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

The per-cell Results tables show five metrics, each as the mean over the cell's panels with
its 95% percentile-bootstrap interval in brackets; tables that pool cells, and the later
sections, show **RMSE on customer totals / bias % / MAPE** as means only. RMSE is
computed on each customer's holdout-year total, the paper's definition. It grows with the
purchase rate (about 0.6 at rate 0.01, about 5 at rate 0.30), so it is compared within one
rate only. The per-week RMSE in `results.csv` is not used: 10 of 13 trees score 0.18 on it.

**Tests** (`docs/statistical-protocol.md`). One panel is one replication, and every test
runs inside one rate × churn cell of 10 panels. The effect is the difference of means,
Δ = mean(B) − mean(A), with its 95% percentile-bootstrap interval from 10,000 resamples
(`panelclv.evaluation.effects.effect`), and a claim is supported when the interval excludes
0. Two trees fitted on the same panels are paired by panel. The 1,000- and 3,000-customer
grids are independent replications (claim 12 says why). Results per rate, per churn level
or over the whole grid pool cells, so they are described, never tested: the pooled mean,
and the pattern of per-cell verdicts. RMSE is descriptive only (§4 of the protocol).

## Results

The ranking depends on both the purchase rate and the churn rate. On MAPE, Pareto/NBD leads
every cell at rates 0.01–0.05 and at rate 0.10 with churn 20–40%; the LSTM with bounded flags
leads at rate 0.10 with churn 60–80% and at rate 0.30 with churn 20–40%, and the LSTM without
AR features at rate 0.30 with churn 60–80%. The Transformer leads no cell. On
Spearman, Pareto/NBD leads all 16 cells, and in every one its lead over the runner-up is
supported. Two Transformer
`ar_unbounded` studies whose forecasts did not match their stored results are left out.

**Setting for this section.** Seasonal panels (4 peaks, amplitude 1.5), 1,000 customers, 13
trees (Pareto/NBD and 6 arms each of the LSTM and the Transformer). Four purchase rates ×
four churn rates (20, 40, 60 and 80% of customers dropped out by week 52), 10 panels per
cell. A metric entry in a per-cell table is the mean over the cell's panels with its 95%
percentile-bootstrap interval. RMSE is on
customer totals and grows with the rate, and CE depends on the panel, so compare both within
one table only. The three "Beats P/NBD" columns compare the tree with Pareto/NBD on the
same panel, one metric each, and count the table's panels where the tree wins:
lower MAPE, smaller |bias| (closer to 0, either sign), higher Spearman. The counts are
description only and carry no marks.

**Best, and not clearly worse than the best** (per-cell tables only). Every metric column
of a cell's table marks its best tree in **bold**: the lowest mean over the cell's panels
for RMSE, MAPE, Val. CE and |bias| (taken per panel, so a tree that errs +50% on one panel
and −50% on another is not unbiased), and the highest mean for Spearman. A tree is
<ins>underlined</ins> when the paired 95% bootstrap interval of its difference from the
best tree, over the cell's 10 panels, contains 0. Pairing compares the two trees within
each panel, which removes the panel-to-panel differences in difficulty. Pareto/NBD has no
CE. The rate-pooled tables carry no marks: they pool four cells, and the protocol never
tests across cells.

Read an underline as "no clear difference from the best at n = 10", never as "equal". 47
of the 1,024 metric entries in the per-cell tables are underlined. The mark is per column,
so a tree underlined on MAPE can be plain on Spearman. At rate 0.30 and churn 80%, LSTM
`ar_bounded` is not underlined against LSTM `no_ar` on MAPE at 35 against 32: the paired
difference is small but consistent (+3, 95% CI +1 to +6).

### By purchase rate and churn

One table per rate × churn cell, 10 panels per row (9 where a study was left out).

**Rate 0.01, churn 20%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **0.79 [0.77, 0.83]** | **+12 [+9, +15]** | **48 [45, 51]** | **0.22 [0.19, 0.25]** | — | — | — | — |
| LSTM | `no_ar` | 0.87 [0.83, 0.91] | +59 [+47, +74] | 71 [62, 82] | 0.01 [−0.02, 0.04] | 0.079 [0.076, 0.082] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` | 6.70 [1.95, 12.65] | +502 [+93, +1058] | 513 [104, 1069] | −0.16 [−0.20, −0.12] | 0.078 [0.074, 0.081] | 1 of 10 | 1 of 10 | 0 of 10 |
| LSTM | `ar_bounded` | 0.86 [0.82, 0.92] | +44 [+23, +67] | 65 [53, 81] | −0.03 [−0.09, 0.02] | 0.078 [0.075, 0.081] | 3 of 10 | 1 of 10 | 0 of 10 |
| LSTM | `ar_bounded` + `kmeans_8` | 0.88 [0.85, 0.92] | +56 [+38, +71] | 72 [62, 82] | 0.13 [0.10, 0.17] | 0.077 [0.074, 0.080] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` + `kmeans_8` | 5.09 [2.65, 7.70] | +235 [+107, +377] | 250 [124, 390] | −0.08 [−0.13, −0.04] | 0.075 [0.072, 0.078] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `no_ar` + `kmeans_8` | 0.91 [0.87, 0.97] | +77 [+56, +98] | 86 [70, 104] | 0.14 [0.11, 0.16] | 0.078 [0.075, 0.081] | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `no_ar` | 0.84 [0.80, 0.90] | +19 [−6, +45] | 58 [50, 68] | 0.05 [0.03, 0.08] | 0.077 [0.074, 0.081] | 3 of 10 | 3 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` | 0.83 [0.81, 0.87] | <ins>+7 [−9, +26]</ins> | <ins>49 [44, 56]</ins> | −0.05 [−0.09, 0.00] | 0.077 [0.074, 0.080] | 6 of 10 | 6 of 10 | 0 of 10 |
| Transformer | `ar_bounded` | 0.85 [0.81, 0.91] | +36 [+11, +62] | 62 [50, 79] | 0.03 [0.00, 0.06] | 0.077 [0.074, 0.080] | 3 of 10 | 1 of 10 | 0 of 10 |
| Transformer | `ar_bounded` + `kmeans_8` | 0.90 [0.84, 0.98] | +23 [−17, +72] | 73 [55, 102] | 0.16 [0.14, 0.19] | 0.074 [0.071, 0.078] | 2 of 10 | 0 of 10 | 2 of 10 |
| Transformer | `ar_unbounded` + `kmeans_8` | 0.87 [0.83, 0.92] | −13 [−42, +21] | 65 [57, 73] | 0.03 [−0.03, 0.08] | **0.072 [0.068, 0.076]** | 0 of 10 | 2 of 10 | 0 of 10 |
| Transformer | `no_ar` + `kmeans_8` | 0.86 [0.81, 0.92] | −3 [−27, +22] | 55 [49, 63] | 0.12 [0.09, 0.16] | 0.077 [0.074, 0.080] | 4 of 10 | 2 of 10 | 0 of 10 |

**Rate 0.01, churn 40%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **0.68 [0.65, 0.71]** | **+34 [+26, +40]** | **66 [62, 68]** | **0.25 [0.23, 0.28]** | — | — | — | — |
| LSTM | `no_ar` | 0.80 [0.77, 0.83] | +132 [+114, +150] | 136 [120, 154] | 0.01 [−0.02, 0.04] | 0.068 [0.065, 0.071] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` | 1.89 [0.82, 3.37] | +182 [+123, +251] | 189 [130, 257] | −0.23 [−0.25, −0.21] | 0.068 [0.064, 0.071] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_bounded` | 0.78 [0.75, 0.82] | +114 [+100, +130] | 121 [108, 135] | −0.09 [−0.12, −0.06] | 0.067 [0.064, 0.070] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_bounded` + `kmeans_8` | 0.80 [0.76, 0.83] | +116 [+99, +132] | 121 [107, 136] | 0.11 [0.07, 0.14] | 0.067 [0.064, 0.070] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` + `kmeans_8` | 11.37 [5.39, 17.90] | +1300 [+412, +2406] | 1304 [418, 2409] | −0.12 [−0.19, −0.05] | 0.066 [0.063, 0.069] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `no_ar` + `kmeans_8` | 0.81 [0.78, 0.84] | +136 [+123, +149] | 140 [128, 152] | 0.14 [0.11, 0.16] | 0.067 [0.064, 0.070] | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `no_ar` | 0.75 [0.69, 0.82] | +61 [+13, +113] | 96 [65, 134] | 0.02 [−0.03, 0.06] | 0.066 [0.063, 0.068] | 6 of 10 | 6 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` | 0.72 [0.69, 0.75] | <ins>+18 [−21, +63]</ins> | <ins>78 [59, 100]</ins> | 0.00 [−0.07, 0.08] | 0.065 [0.062, 0.068] | 6 of 10 | 5 of 10 | 1 of 10 |
| Transformer | `ar_bounded` | 0.71 [0.69, 0.74] | +27 [−9, +61] | <ins>75 [65, 87]</ins> | −0.02 [−0.04, 0.00] | 0.065 [0.062, 0.068] | 4 of 10 | 2 of 10 | 0 of 10 |
| Transformer | `ar_bounded` + `kmeans_8` | 0.80 [0.70, 0.92] | <ins>+64 [+11, +126]</ins> | <ins>102 [66, 149]</ins> | 0.16 [0.13, 0.18] | 0.062 [0.060, 0.065] | 4 of 10 | 4 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` + `kmeans_8` | 0.75 [0.72, 0.79] | +1 [−40, +50] | 83 [66, 104] | 0.00 [−0.05, 0.04] | **0.061 [0.058, 0.063]** | 2 of 10 | 3 of 10 | 0 of 10 |
| Transformer | `no_ar` + `kmeans_8` | 0.92 [0.77, 1.10] | +117 [+39, +206] | 148 [88, 220] | 0.12 [0.09, 0.16] | 0.065 [0.061, 0.068] | 2 of 10 | 2 of 10 | 0 of 10 |

**Rate 0.01, churn 60%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **0.57 [0.54, 0.60]** | **+41 [+32, +50]** | **88 [83, 93]** | **0.26 [0.23, 0.30]** | — | — | — | — |
| LSTM | `no_ar` | 0.83 [0.79, 0.87] | +290 [+267, +316] | 291 [269, 317] | 0.01 [−0.03, 0.04] | 0.052 [0.049, 0.055] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` | 3.39 [1.42, 5.65] | +514 [+364, +705] | 514 [365, 705] | −0.20 [−0.25, −0.15] | 0.052 [0.049, 0.056] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_bounded` | 0.71 [0.66, 0.76] | +190 [+150, +235] | 196 [159, 240] | −0.04 [−0.10, 0.03] | 0.051 [0.049, 0.054] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_bounded` + `kmeans_8` | 0.79 [0.74, 0.83] | +241 [+216, +266] | 245 [222, 269] | 0.07 [0.04, 0.11] | 0.051 [0.048, 0.054] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` + `kmeans_8` | 5.48 [1.95, 9.60] | +900 [+424, +1465] | 904 [431, 1467] | −0.04 [−0.13, 0.04] | 0.051 [0.048, 0.054] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `no_ar` + `kmeans_8` | 0.82 [0.76, 0.87] | +271 [+240, +305] | 273 [243, 306] | 0.16 [0.12, 0.20] | 0.051 [0.048, 0.054] | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `no_ar` | 0.63 [0.60, 0.66] | +96 [+47, +149] | 129 [97, 167] | 0.07 [0.04, 0.09] | 0.050 [0.047, 0.053] | 3 of 10 | 3 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` | 0.75 [0.62, 0.89] | <ins>+68 [+18, +124]</ins> | 127 [89, 169] | 0.06 [0.02, 0.09] | 0.049 [0.046, 0.052] | 5 of 10 | 4 of 10 | 0 of 10 |
| Transformer | `ar_bounded` | 0.60 [0.57, 0.63] | <ins>+50 [+17, +83]</ins> | <ins>98 [82, 116]</ins> | 0.06 [0.02, 0.10] | 0.049 [0.046, 0.053] | 4 of 10 | 4 of 10 | 0 of 10 |
| Transformer | `ar_bounded` + `kmeans_8` | 0.63 [0.58, 0.69] | <ins>+43 [−2, +91]</ins> | <ins>103 [81, 129]</ins> | 0.18 [0.12, 0.23] | 0.047 [0.044, 0.050] | 5 of 10 | 5 of 10 | 1 of 10 |
| Transformer | `ar_unbounded` + `kmeans_8` | 0.68 [0.60, 0.78] | +68 [+17, +130] | 120 [92, 163] | 0.10 [0.05, 0.14] | **0.044 [0.042, 0.048]** | 3 of 10 | 2 of 10 | 0 of 10 |
| Transformer | `no_ar` + `kmeans_8` | 0.64 [0.60, 0.69] | +65 [+11, +119] | 120 [93, 150] | 0.16 [0.12, 0.21] | 0.047 [0.044, 0.051] | 4 of 10 | 2 of 10 | 0 of 10 |

**Rate 0.01, churn 80%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **0.42 [0.38, 0.46]** | **+54 [+35, +76]** | **159 [137, 181]** | **0.20 [0.16, 0.24]** | — | — | — | — |
| LSTM | `no_ar` | 0.86 [0.79, 0.92] | +755 [+533, +1003] | 759 [540, 1005] | −0.02 [−0.05, 0.02] | 0.034 [0.031, 0.037] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` | 1.53 [0.84, 2.57] | +821 [+501, +1141] | 838 [533, 1147] | −0.12 [−0.15, −0.08] | 0.034 [0.031, 0.037] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_bounded` | 0.71 [0.62, 0.80] | +576 [+348, +831] | 591 [374, 838] | −0.07 [−0.10, −0.04] | 0.034 [0.031, 0.037] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_bounded` + `kmeans_8` | 0.63 [0.54, 0.72] | +429 [+304, +587] | 445 [328, 598] | 0.07 [0.02, 0.12] | 0.033 [0.030, 0.036] | 0 of 10 | 0 of 10 | 1 of 10 |
| LSTM | `ar_unbounded` + `kmeans_8` | 6.26 [1.38, 11.96] | +1993 [+731, +3553] | 2015 [763, 3571] | −0.02 [−0.08, 0.04] | 0.033 [0.030, 0.035] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `no_ar` + `kmeans_8` | 0.85 [0.79, 0.91] | +725 [+542, +937] | 729 [548, 938] | 0.11 [0.07, 0.16] | 0.033 [0.030, 0.036] | 0 of 10 | 0 of 10 | 1 of 10 |
| Transformer | `no_ar` | <ins>0.44 [0.39, 0.49]</ins> | +121 [+66, +182] | 197 [159, 239] | 0.07 [0.02, 0.12] | 0.031 [0.029, 0.034] | 2 of 10 | 3 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` | 0.70 [0.44, 1.18] | +264 [+73, +541] | 345 [170, 611] | 0.00 [−0.09, 0.08] | 0.031 [0.028, 0.033] | 3 of 10 | 3 of 10 | 0 of 10 |
| Transformer | `ar_bounded` | <ins>0.44 [0.38, 0.49]</ins> | <ins>+89 [+45, +135]</ins> | <ins>176 [152, 201]</ins> | 0.08 [0.02, 0.14] | 0.031 [0.028, 0.034] | 4 of 10 | 4 of 10 | 1 of 10 |
| Transformer | `ar_bounded` + `kmeans_8` | 0.51 [0.43, 0.58] | +109 [+53, +183] | 196 [159, 246] | 0.11 [0.05, 0.16] | 0.028 [0.026, 0.031] | 4 of 10 | 4 of 10 | 2 of 10 |
| Transformer | `ar_unbounded` + `kmeans_8` | 0.51 [0.44, 0.59] | +120 [+28, +218] | 216 [153, 283] | 0.09 [0.03, 0.14] | **0.027 [0.025, 0.030]** | 3 of 10 | 2 of 10 | 1 of 10 |
| Transformer | `no_ar` + `kmeans_8` | <ins>0.48 [0.42, 0.54]</ins> | <ins>+104 [+32, +205]</ins> | <ins>195 [144, 271]</ins> | 0.15 [0.12, 0.18] | 0.030 [0.027, 0.033] | 6 of 10 | 5 of 10 | 2 of 10 |

**Rate 0.05, churn 20%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **1.92 [1.89, 1.96]** | **+8 [+6, +11]** | **35 [34, 37]** | **0.58 [0.56, 0.60]** | — | — | — | — |
| LSTM | `no_ar` | 2.33 [2.20, 2.46] | +42 [+33, +51] | 45 [37, 53] | 0.29 [0.13, 0.44] | 0.175 [0.173, 0.178] | 2 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` | 14.23 [11.69, 17.09] | +240 [+190, +299] | 240 [190, 299] | 0.44 [0.42, 0.46] | 0.177 [0.175, 0.179] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_bounded` | 2.29 [2.23, 2.33] | +47 [+39, +54] | 48 [42, 55] | 0.50 [0.47, 0.53] | 0.174 [0.172, 0.176] | 1 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_bounded` + `kmeans_8` | 2.20 [2.13, 2.28] | +29 [+24, +33] | 39 [37, 42] | 0.45 [0.43, 0.47] | <ins>0.169 [0.166, 0.172]</ins> | 2 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` + `kmeans_8` | 11.36 [9.16, 13.38] | +188 [+151, +224] | 189 [151, 224] | 0.46 [0.43, 0.49] | 0.174 [0.171, 0.177] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `no_ar` + `kmeans_8` | 2.26 [2.17, 2.36] | +40 [+33, +48] | 47 [41, 53] | 0.46 [0.43, 0.50] | 0.169 [0.167, 0.172] | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `no_ar` | 2.70 [2.53, 2.88] | +72 [+55, +87] | 75 [62, 88] | 0.31 [0.20, 0.40] | 0.175 [0.173, 0.177] | 1 of 10 | 1 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` | 2.69 [2.46, 2.89] | +90 [+68, +109] | 91 [70, 109] | 0.52 [0.50, 0.54] | 0.173 [0.171, 0.176] | 1 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_bounded` | 2.56 [2.40, 2.69] | +61 [+38, +80] | 68 [54, 80] | 0.45 [0.42, 0.48] | 0.174 [0.172, 0.176] | 2 of 10 | 1 of 10 | 0 of 10 |
| Transformer | `ar_bounded` + `kmeans_8` | 2.65 [2.50, 2.80] | +77 [+63, +89] | 79 [67, 90] | 0.51 [0.48, 0.53] | **0.168 [0.165, 0.170]** | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` + `kmeans_8` | 2.66 [2.40, 2.91] | +61 [+36, +81] | 69 [55, 82] | 0.47 [0.40, 0.52] | <ins>0.168 [0.166, 0.171]</ins> | 2 of 10 | 1 of 10 | 0 of 10 |
| Transformer | `no_ar` + `kmeans_8` | 2.44 [2.25, 2.61] | +62 [+42, +81] | 67 [51, 83] | 0.49 [0.45, 0.52] | 0.169 [0.166, 0.172] | 2 of 10 | 0 of 10 | 0 of 10 |

**Rate 0.05, churn 40%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **1.63 [1.60, 1.65]** | **+15 [+11, +19]** | **39 [37, 41]** | **0.60 [0.58, 0.61]** | — | — | — | — |
| LSTM | `no_ar` | 2.24 [2.15, 2.33] | +98 [+87, +109] | 99 [88, 110] | 0.20 [0.05, 0.34] | 0.144 [0.142, 0.147] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` | 10.38 [7.99, 12.92] | +313 [+255, +372] | 313 [255, 372] | 0.39 [0.36, 0.42] | 0.145 [0.142, 0.147] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_bounded` | 1.88 [1.82, 1.92] | +52 [+43, +61] | 56 [47, 63] | 0.54 [0.51, 0.56] | 0.138 [0.136, 0.141] | 1 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_bounded` + `kmeans_8` | 2.00 [1.93, 2.09] | +61 [+50, +75] | 64 [54, 78] | 0.45 [0.43, 0.48] | 0.135 [0.132, 0.138] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` + `kmeans_8` | 9.40 [4.97, 14.54] | +221 [+101, +363] | 229 [116, 367] | 0.37 [0.32, 0.43] | 0.140 [0.137, 0.143] | 3 of 10 | 2 of 10 | 0 of 10 |
| LSTM | `no_ar` + `kmeans_8` | 2.09 [2.05, 2.14] | +81 [+71, +91] | 82 [73, 92] | 0.46 [0.42, 0.50] | 0.137 [0.134, 0.139] | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `no_ar` | 2.28 [2.13, 2.44] | +108 [+83, +132] | 110 [85, 132] | 0.42 [0.38, 0.45] | 0.142 [0.139, 0.144] | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` | 2.30 [1.97, 2.75] | +101 [+72, +128] | 103 [77, 129] | 0.51 [0.44, 0.56] | 0.138 [0.136, 0.141] | 1 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_bounded` | 1.91 [1.74, 2.12] | +53 [+22, +88] | 65 [40, 96] | 0.55 [0.52, 0.57] | 0.138 [0.136, 0.141] | 3 of 10 | 2 of 10 | 0 of 10 |
| Transformer | `ar_bounded` + `kmeans_8` | 2.00 [1.91, 2.10] | +48 [+32, +63] | 56 [44, 68] | 0.51 [0.49, 0.54] | **0.133 [0.131, 0.136]** | 3 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` + `kmeans_8` | 2.15 [1.98, 2.34] | +68 [+36, +101] | 78 [54, 107] | 0.46 [0.38, 0.52] | <ins>0.134 [0.131, 0.137]</ins> | 2 of 10 | 3 of 10 | 0 of 10 |
| Transformer | `no_ar` + `kmeans_8` | 2.37 [2.20, 2.56] | +105 [+79, +134] | 107 [82, 135] | 0.48 [0.44, 0.51] | <ins>0.134 [0.131, 0.137]</ins> | 0 of 10 | 0 of 10 | 0 of 10 |

**Rate 0.05, churn 60%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **1.32 [1.27, 1.37]** | **+9 [+1, +16]** | **42 [40, 44]** | **0.56 [0.55, 0.57]** | — | — | — | — |
| LSTM | `no_ar` | 1.95 [1.86, 2.04] | +157 [+138, +176] | 158 [139, 176] | 0.03 [0.00, 0.06] | 0.109 [0.106, 0.112] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` | 10.76 [8.38, 12.94] | +462 [+338, +630] | 462 [338, 630] | 0.35 [0.32, 0.38] | 0.108 [0.105, 0.111] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_bounded` | 1.43 [1.37, 1.51] | +60 [+43, +74] | 67 [57, 77] | 0.52 [0.51, 0.53] | 0.100 [0.098, 0.103] | 1 of 10 | 1 of 10 | 0 of 10 |
| LSTM | `ar_bounded` + `kmeans_8` | 1.80 [1.70, 1.90] | +108 [+88, +129] | 110 [92, 131] | 0.43 [0.39, 0.46] | 0.098 [0.095, 0.101] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` + `kmeans_8` | 8.50 [5.06, 12.13] | +357 [+173, +578] | 359 [176, 580] | 0.36 [0.28, 0.43] | 0.104 [0.101, 0.107] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `no_ar` + `kmeans_8` | 1.99 [1.89, 2.10] | +172 [+153, +191] | 172 [153, 191] | 0.38 [0.31, 0.43] | 0.102 [0.099, 0.104] | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `no_ar` | 1.87 [1.69, 2.04] | +144 [+89, +197] | 149 [98, 198] | 0.37 [0.32, 0.42] | 0.105 [0.102, 0.107] | 1 of 10 | 1 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` | 1.73 [1.61, 1.85] | +121 [+87, +156] | 124 [92, 157] | 0.49 [0.44, 0.52] | 0.100 [0.098, 0.103] | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_bounded` | 1.47 [1.39, 1.55] | +66 [+36, +96] | 77 [56, 101] | 0.52 [0.50, 0.53] | 0.100 [0.097, 0.103] | 2 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_bounded` + `kmeans_8` | 2.02 [1.81, 2.21] | +152 [+108, +196] | 155 [114, 197] | 0.47 [0.44, 0.50] | <ins>0.096 [0.093, 0.098]</ins> | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` + `kmeans_8` | 1.90 [1.64, 2.31] | +98 [+55, +159] | 103 [63, 163] | 0.43 [0.36, 0.49] | **0.094 [0.092, 0.097]** | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `no_ar` + `kmeans_8` | 1.91 [1.73, 2.08] | +131 [+88, +177] | 134 [96, 178] | 0.48 [0.46, 0.50] | 0.096 [0.093, 0.098] | 0 of 10 | 0 of 10 | 0 of 10 |

**Rate 0.05, churn 80%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **0.89 [0.86, 0.92]** | **+13 [0, +26]** | **58 [53, 64]** | **0.44 [0.42, 0.47]** | — | — | — | — |
| LSTM | `no_ar` | 1.88 [1.75, 2.03] | +497 [+451, +544] | 497 [451, 544] | −0.01 [−0.03, 0.02] | 0.070 [0.065, 0.074] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` | 12.14 [7.39, 16.74] | +1314 [+789, +1810] | 1315 [789, 1810] | 0.26 [0.20, 0.33] | 0.067 [0.062, 0.072] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_bounded` | 0.97 [0.93, 1.01] | <ins>+19 [+1, +37]</ins> | <ins>60 [53, 69]</ins> | 0.40 [0.37, 0.43] | 0.061 [0.057, 0.065] | 7 of 10 | 4 of 10 | 1 of 10 |
| LSTM | `ar_bounded` + `kmeans_8` | 1.58 [1.36, 1.82] | +287 [+189, +396] | 290 [195, 397] | 0.36 [0.34, 0.38] | 0.061 [0.057, 0.064] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` + `kmeans_8` | 11.44 [7.94, 14.72] | +908 [+568, +1222] | 919 [594, 1225] | 0.32 [0.27, 0.37] | 0.065 [0.061, 0.069] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `no_ar` + `kmeans_8` | 1.99 [1.90, 2.09] | +506 [+437, +560] | 506 [437, 560] | 0.34 [0.30, 0.38] | 0.064 [0.060, 0.068] | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `no_ar` | 1.13 [1.07, 1.18] | +110 [+66, +157] | 122 [87, 162] | 0.32 [0.27, 0.35] | 0.066 [0.062, 0.070] | 2 of 10 | 2 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` | 1.09 [1.00, 1.21] | +70 [+27, +122] | 100 [69, 142] | 0.19 [0.05, 0.34] | 0.062 [0.058, 0.066] | 1 of 10 | 1 of 10 | 0 of 10 |
| Transformer | `ar_bounded` | 1.04 [0.99, 1.08] | +76 [+28, +127] | 102 [68, 141] | 0.40 [0.38, 0.44] | 0.061 [0.057, 0.065] | 2 of 10 | 1 of 10 | 1 of 10 |
| Transformer | `ar_bounded` + `kmeans_8` | 1.11 [1.06, 1.14] | +62 [+15, +111] | 94 [64, 128] | 0.37 [0.34, 0.41] | 0.058 [0.054, 0.061] | 3 of 10 | 3 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` + `kmeans_8` | 1.15 [0.98, 1.43] | +63 [+14, +129] | 97 [63, 151] | 0.38 [0.36, 0.41] | **0.057 [0.053, 0.061]** | 3 of 10 | 3 of 10 | 0 of 10 |
| Transformer | `no_ar` + `kmeans_8` | 1.29 [1.12, 1.47] | +161 [+69, +254] | 181 [103, 262] | 0.36 [0.30, 0.40] | 0.059 [0.055, 0.063] | 1 of 10 | 2 of 10 | 0 of 10 |

**Rate 0.10, churn 20%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **2.91 [2.86, 2.95]** | **+3 [+1, +4]** | **31 [30, 32]** | **0.72 [0.71, 0.73]** | — | — | — | — |
| LSTM | `no_ar` | 3.28 [3.16, 3.41] | +31 [+24, +38] | <ins>33 [26, 40]</ins> | 0.67 [0.65, 0.69] | 0.262 [0.255, 0.268] | 4 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` | 21.15 [15.92, 26.87] | +301 [+223, +391] | 301 [223, 391] | 0.60 [0.58, 0.62] | 0.272 [0.265, 0.278] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_bounded` | 3.32 [3.21, 3.45] | +29 [+17, +39] | <ins>32 [24, 41]</ins> | 0.69 [0.68, 0.70] | 0.261 [0.255, 0.266] | 4 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_bounded` + `kmeans_8` | 3.46 [3.40, 3.53] | +28 [+24, +33] | <ins>33 [29, 37]</ins> | 0.60 [0.57, 0.64] | **0.254 [0.247, 0.260]** | 4 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` + `kmeans_8` | 13.08 [8.43, 17.91] | +141 [+82, +205] | 145 [88, 206] | 0.58 [0.54, 0.61] | 0.265 [0.258, 0.270] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `no_ar` + `kmeans_8` | 3.57 [3.46, 3.66] | +35 [+28, +42] | 39 [34, 44] | 0.58 [0.55, 0.61] | 0.258 [0.251, 0.265] | 2 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `no_ar` | 4.11 [3.84, 4.34] | +54 [+35, +72] | 58 [45, 72] | 0.58 [0.56, 0.60] | 0.266 [0.259, 0.273] | 2 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` | 5.00 [4.30, 5.90] | +101 [+76, +128] | 102 [77, 129] | 0.65 [0.63, 0.66] | 0.262 [0.255, 0.268] | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_bounded` | 3.74 [3.56, 3.95] | +43 [+23, +64] | 52 [37, 68] | 0.68 [0.67, 0.69] | 0.262 [0.255, 0.268] | 4 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_bounded` + `kmeans_8` | 3.71 [3.46, 3.97] | +36 [+20, +54] | 45 [35, 58] | 0.65 [0.59, 0.68] | <ins>0.255 [0.247, 0.260]</ins> | 2 of 10 | 1 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` + `kmeans_8` | 3.86 [3.59, 4.13] | +38 [+22, +52] | 47 [38, 55] | 0.63 [0.57, 0.67] | <ins>0.256 [0.248, 0.263]</ins> | 2 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `no_ar` + `kmeans_8` | 3.98 [3.69, 4.30] | +54 [+38, +70] | 57 [44, 71] | 0.61 [0.57, 0.64] | 0.256 [0.249, 0.261] | 1 of 10 | 0 of 10 | 0 of 10 |

**Rate 0.10, churn 40%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **2.47 [2.39, 2.55]** | **+1 [−2, +5]** | **32 [30, 33]** | **0.73 [0.72, 0.74]** | — | — | — | — |
| LSTM | `no_ar` | 3.11 [2.77, 3.61] | +57 [+36, +88] | 58 [38, 89] | 0.64 [0.57, 0.69] | 0.205 [0.200, 0.209] | 2 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` | 18.03 [13.16, 23.69] | +375 [+302, +471] | 375 [302, 471] | 0.57 [0.56, 0.59] | 0.213 [0.210, 0.217] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_bounded` | 2.79 [2.68, 2.89] | +28 [+18, +39] | <ins>34 [27, 42]</ins> | 0.69 [0.68, 0.71] | 0.198 [0.194, 0.202] | 6 of 10 | 1 of 10 | 0 of 10 |
| LSTM | `ar_bounded` + `kmeans_8` | 3.15 [3.03, 3.32] | +52 [+45, +58] | 53 [48, 59] | 0.64 [0.62, 0.65] | <ins>0.193 [0.190, 0.197]</ins> | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` + `kmeans_8` | 9.72 [5.58, 14.13] | +167 [+89, +250] | 174 [101, 253] | 0.56 [0.52, 0.61] | 0.203 [0.198, 0.207] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `no_ar` + `kmeans_8` | 3.42 [3.33, 3.53] | +74 [+68, +80] | 74 [69, 80] | 0.57 [0.53, 0.61] | 0.198 [0.194, 0.202] | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `no_ar` | 3.69 [3.45, 3.93] | +86 [+51, +117] | 94 [67, 119] | 0.58 [0.57, 0.59] | 0.206 [0.202, 0.210] | 1 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` | 3.89 [3.30, 4.61] | +89 [+67, +115] | 91 [69, 116] | 0.64 [0.58, 0.67] | 0.200 [0.196, 0.205] | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_bounded` | 2.87 [2.64, 3.15] | +27 [+9, +43] | 41 [32, 50] | 0.64 [0.54, 0.70] | 0.199 [0.195, 0.203] | 4 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_bounded` + `kmeans_8` | 3.42 [3.12, 3.76] | +51 [+31, +72] | 58 [43, 74] | 0.65 [0.64, 0.66] | **0.192 [0.188, 0.196]** | 1 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` + `kmeans_8` | 3.74 [3.29, 4.24] | +72 [+47, +99] | 77 [55, 100] | 0.62 [0.58, 0.65] | <ins>0.192 [0.188, 0.197]</ins> | 1 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `no_ar` + `kmeans_8` | 3.99 [3.64, 4.28] | +93 [+74, +109] | 95 [78, 109] | 0.63 [0.61, 0.64] | 0.195 [0.192, 0.199] | 0 of 10 | 0 of 10 | 0 of 10 |

**Rate 0.10, churn 60%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **2.00 [1.95, 2.06]** | **+3 [−3, +10]** | 37 [35, 39] | **0.65 [0.64, 0.66]** | — | — | — | — |
| LSTM | `no_ar` | 2.43 [2.28, 2.61] | +61 [+33, +90] | 69 [47, 93] | 0.60 [0.58, 0.61] | 0.149 [0.143, 0.153] | 3 of 10 | 2 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` | 7.56 [4.76, 10.97] | +299 [+164, +429] | 307 [179, 431] | 0.52 [0.47, 0.57] | 0.154 [0.146, 0.160] | 1 of 10 | 1 of 10 | 0 of 10 |
| LSTM | `ar_bounded` | 2.12 [2.05, 2.18] | +20 [+13, +28] | **33 [30, 36]** | 0.63 [0.61, 0.64] | 0.140 [0.136, 0.143] | 7 of 10 | 1 of 10 | 0 of 10 |
| LSTM | `ar_bounded` + `kmeans_8` | 2.79 [2.68, 2.91] | +92 [+74, +111] | 93 [77, 112] | 0.53 [0.50, 0.56] | 0.139 [0.135, 0.141] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` + `kmeans_8` | 6.06 [3.69, 8.57] | +133 [+53, +234] | 142 [66, 240] | 0.55 [0.51, 0.59] | 0.146 [0.142, 0.151] | 1 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `no_ar` + `kmeans_8` | 3.12 [2.98, 3.28] | +132 [+110, +161] | 132 [110, 161] | 0.51 [0.46, 0.54] | 0.146 [0.142, 0.149] | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `no_ar` | 2.72 [2.49, 3.00] | +98 [+59, +137] | 103 [70, 139] | 0.52 [0.50, 0.55] | 0.150 [0.146, 0.153] | 1 of 10 | 1 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` | 3.57 [3.04, 4.12] | +159 [+124, +198] | 159 [125, 198] | 0.57 [0.54, 0.60] | 0.143 [0.139, 0.146] | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_bounded` | 2.74 [2.40, 3.18] | +104 [+55, +152] | 115 [79, 156] | 0.61 [0.59, 0.62] | 0.140 [0.136, 0.143] | 1 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_bounded` + `kmeans_8` | 3.05 [2.76, 3.40] | +106 [+73, +142] | 108 [77, 143] | 0.57 [0.55, 0.59] | **0.135 [0.132, 0.138]** | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` + `kmeans_8` | 2.83 [2.56, 3.09] | +70 [+43, +97] | 76 [55, 101] | 0.56 [0.52, 0.59] | <ins>0.136 [0.132, 0.138]</ins> | 1 of 10 | 1 of 10 | 0 of 10 |
| Transformer | `no_ar` + `kmeans_8` | 3.01 [2.75, 3.27] | +106 [+78, +132] | 109 [82, 133] | 0.56 [0.55, 0.58] | 0.139 [0.135, 0.141] | 0 of 10 | 0 of 10 | 0 of 10 |

**Rate 0.10, churn 80%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | **1.37 [1.26, 1.47]** | <ins>+7 [−7, +23]</ins> | 49 [44, 57] | **0.51 [0.49, 0.52]** | — | — | — | — |
| LSTM | `no_ar` | 2.02 [1.73, 2.31] | +224 [+111, +362] | 233 [128, 367] | 0.27 [0.16, 0.39] | 0.091 [0.085, 0.095] | 2 of 10 | 2 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` | 4.09 [2.30, 6.63] | +285 [+103, +493] | 307 [137, 503] | 0.43 [0.38, 0.47] | 0.085 [0.080, 0.089] | 1 of 10 | 1 of 10 | 0 of 10 |
| LSTM | `ar_bounded` | 1.45 [1.31, 1.58] | **+4 [−7, +15]** | **45 [40, 50]** | 0.47 [0.45, 0.49] | 0.077 [0.073, 0.081] | 9 of 10 | 8 of 10 | 0 of 10 |
| LSTM | `ar_bounded` + `kmeans_8` | 2.04 [1.89, 2.16] | +131 [+97, +177] | 134 [102, 178] | 0.45 [0.42, 0.47] | <ins>0.076 [0.072, 0.079]</ins> | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` + `kmeans_8` | 3.98 [1.79, 6.39] | +146 [+2, +355] | 189 [56, 382] | 0.45 [0.42, 0.47] | 0.078 [0.074, 0.084] | 1 of 10 | 2 of 10 | 0 of 10 |
| LSTM | `no_ar` + `kmeans_8` | 2.42 [2.27, 2.58] | +309 [+231, +404] | 309 [231, 404] | 0.38 [0.34, 0.42] | 0.086 [0.083, 0.089] | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `no_ar` | 2.20 [2.04, 2.36] | +265 [+201, +342] | 265 [202, 343] | 0.41 [0.39, 0.43] | 0.088 [0.084, 0.091] | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` | 2.70 [2.27, 3.19] | +245 [+199, +300] | 245 [199, 300] | 0.45 [0.42, 0.48] | 0.080 [0.077, 0.084] | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_bounded` | 1.57 [1.46, 1.66] | +82 [+34, +135] | 102 [68, 146] | 0.47 [0.45, 0.48] | 0.078 [0.075, 0.082] | 3 of 10 | 2 of 10 | 0 of 10 |
| Transformer | `ar_bounded` + `kmeans_8` | 2.31 [2.08, 2.56] | +175 [+130, +223] | 177 [133, 223] | 0.42 [0.38, 0.45] | **0.075 [0.072, 0.078]** | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` + `kmeans_8` | 2.21 [1.85, 2.61] | +142 [+87, +200] | 150 [103, 204] | 0.44 [0.41, 0.46] | <ins>0.075 [0.072, 0.079]</ins> | 1 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `no_ar` + `kmeans_8` | 2.39 [2.16, 2.68] | +201 [+136, +275] | 203 [140, 276] | 0.43 [0.40, 0.45] | 0.077 [0.073, 0.080] | 0 of 10 | 0 of 10 | 0 of 10 |

**Rate 0.30, churn 20%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | 6.93 [6.72, 7.13] | −14 [−16, −13] | 30 [30, 31] | **0.86 [0.85, 0.87]** | — | — | — | — |
| LSTM | `no_ar` | 6.92 [6.65, 7.21] | +17 [+12, +22] | 19 [16, 22] | 0.85 [0.85, 0.86] | 0.497 [0.491, 0.504] | 10 of 10 | 2 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` | 35.67 [30.70, 39.91] | +268 [+233, +302] | 268 [233, 302] | 0.73 [0.73, 0.74] | 0.546 [0.539, 0.553] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_bounded` | **6.49 [6.26, 6.75]** | **+10 [+7, +12]** | **12 [10, 13]** | 0.85 [0.85, 0.86] | 0.493 [0.486, 0.499] | 10 of 10 | 8 of 10 | 0 of 10 |
| LSTM | `ar_bounded` + `kmeans_8` | 7.96 [7.66, 8.24] | +25 [+22, +28] | 26 [23, 29] | 0.79 [0.78, 0.80] | **0.490 [0.483, 0.497]** | 9 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` + `kmeans_8` | 15.60 [11.32, 21.66] | +76 [+46, +107] | 80 [54, 109] | 0.75 [0.73, 0.78] | 0.513 [0.503, 0.523] | 3 of 10 | 2 of 10 | 0 of 10 |
| LSTM | `no_ar` + `kmeans_8` | 8.31 [8.05, 8.55] | +27 [+25, +30] | 28 [25, 31] | 0.75 [0.73, 0.76] | 0.493 [0.486, 0.500] | 7 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `no_ar` | 8.30 [7.55, 9.39] | +20 [+3, +40] | 36 [26, 50] | 0.71 [0.70, 0.73] | 0.517 [0.511, 0.523] | 8 of 10 | 5 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` | 12.13 [9.75, 14.58] | +69 [+47, +93] | 71 [51, 94] | 0.76 [0.71, 0.80] | 0.511 [0.504, 0.517] | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_bounded` | 8.06 [7.06, 9.18] | +25 [+9, +43] | 37 [27, 49] | 0.81 [0.79, 0.82] | 0.501 [0.494, 0.508] | 6 of 10 | 5 of 10 | 0 of 10 |
| Transformer | `ar_bounded` + `kmeans_8` | 8.08 [7.66, 8.47] | +25 [+17, +32] | 32 [29, 37] | 0.76 [0.70, 0.79] | 0.494 [0.488, 0.501] | 4 of 10 | 2 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` + `kmeans_8` | 8.84 [7.98, 9.67] | +30 [+19, +40] | 37 [32, 43] | 0.80 [0.79, 0.80] | 0.499 [0.491, 0.507] | 2 of 10 | 3 of 10 | 0 of 10 |
| Transformer | `no_ar` + `kmeans_8` | 8.20 [7.80, 8.75] | +21 [+14, +30] | 30 [26, 36] | 0.76 [0.75, 0.77] | 0.505 [0.498, 0.512] | 7 of 10 | 4 of 10 | 0 of 10 |

**Rate 0.30, churn 40%** (10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | 5.99 [5.79, 6.20] | −14 [−15, −12] | 31 [30, 32] | **0.83 [0.82, 0.84]** | — | — | — | — |
| LSTM | `no_ar` | <ins>5.74 [5.53, 5.96]</ins> | <ins>+7 [+3, +11]</ins> | 14 [13, 16] | 0.81 [0.81, 0.82] | 0.361 [0.356, 0.366] | 10 of 10 | 8 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` | 18.04 [11.28, 25.33] | +151 [+62, +243] | 161 [81, 247] | 0.74 [0.71, 0.76] | 0.400 [0.387, 0.412] | 1 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_bounded` | **5.66 [5.46, 5.87]** | **+2 [−3, +6]** | **13 [11, 14]** | 0.82 [0.81, 0.82] | **0.355 [0.350, 0.361]** | 10 of 10 | 9 of 10 | 0 of 10 |
| LSTM | `ar_bounded` + `kmeans_8` | 7.67 [7.32, 8.04] | +44 [+38, +49] | 45 [39, 50] | 0.75 [0.74, 0.77] | 0.358 [0.353, 0.364] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` + `kmeans_8` | 10.25 [7.51, 14.34] | +43 [+8, +84] | 62 [36, 95] | 0.75 [0.72, 0.77] | 0.380 [0.373, 0.387] | 3 of 10 | 4 of 10 | 0 of 10 |
| LSTM | `no_ar` + `kmeans_8` | 8.24 [7.80, 8.73] | +53 [+48, +58] | 53 [48, 58] | 0.74 [0.73, 0.76] | 0.361 [0.356, 0.366] | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `no_ar` | 7.34 [6.87, 7.88] | +28 [+16, +41] | 36 [28, 45] | 0.68 [0.66, 0.70] | 0.386 [0.380, 0.392] | 4 of 10 | 2 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` | 10.23 [8.64, 11.86] | +81 [+51, +108] | 86 [62, 109] | 0.69 [0.62, 0.75] | 0.376 [0.370, 0.383] | 1 of 10 | 1 of 10 | 0 of 10 |
| Transformer | `ar_bounded` | 6.88 [6.14, 7.73] | +23 [+2, +45] | 42 [32, 55] | 0.70 [0.60, 0.78] | 0.364 [0.357, 0.369] | 4 of 10 | 3 of 10 | 0 of 10 |
| Transformer | `ar_bounded` + `kmeans_8` | 7.90 [7.56, 8.24] | +45 [+34, +55] | 47 [39, 56] | 0.74 [0.73, 0.76] | 0.357 [0.351, 0.363] | 1 of 10 | 1 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` + `kmeans_8` | 7.85 [7.03, 8.83] | +21 [+5, +39] | 41 [33, 52] | 0.74 [0.70, 0.76] | 0.363 [0.358, 0.368] | 4 of 10 | 3 of 10 | 0 of 10 |
| Transformer | `no_ar` + `kmeans_8` | 8.52 [7.90, 9.09] | +53 [+42, +63] | 55 [45, 63] | 0.70 [0.69, 0.72] | 0.373 [0.367, 0.378] | 1 of 10 | 0 of 10 | 0 of 10 |

**Rate 0.30, churn 60%** (9–10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | 4.72 [4.60, 4.86] | <ins>−12 [−16, −8]</ins> | 32 [31, 33] | **0.77 [0.76, 0.77]** | — | — | — | — |
| LSTM | `no_ar` | 4.73 [4.53, 4.96] | **−1 [−8, +5]** | **18 [15, 21]** | 0.70 [0.69, 0.71] | 0.237 [0.231, 0.241] | 10 of 10 | 7 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` | 7.04 [6.23, 7.91] | +56 [+36, +77] | 64 [49, 80] | 0.66 [0.64, 0.67] | 0.259 [0.251, 0.268] | 1 of 10 | 1 of 10 | 0 of 10 |
| LSTM | `ar_bounded` | **4.60 [4.46, 4.77]** | <ins>−7 [−13, +1]</ins> | <ins>19 [17, 22]</ins> | 0.70 [0.69, 0.71] | **0.232 [0.227, 0.237]** | 10 of 10 | 6 of 10 | 0 of 10 |
| LSTM | `ar_bounded` + `kmeans_8` | 6.73 [6.43, 7.06] | +71 [+63, +78] | 71 [64, 78] | 0.66 [0.64, 0.67] | 0.236 [0.230, 0.240] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` + `kmeans_8` | 6.49 [5.73, 7.54] | +11 [−16, +44] | 54 [41, 71] | 0.66 [0.65, 0.67] | 0.253 [0.245, 0.261] | 1 of 10 | 2 of 10 | 0 of 10 |
| LSTM | `no_ar` + `kmeans_8` | 7.35 [6.96, 7.68] | +90 [+81, +99] | 90 [81, 99] | 0.64 [0.63, 0.65] | 0.242 [0.235, 0.248] | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `no_ar` | 6.19 [5.78, 6.68] | +40 [+11, +68] | 57 [41, 75] | 0.59 [0.57, 0.60] | 0.265 [0.258, 0.270] | 3 of 10 | 2 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` | 7.95 [7.26, 8.74] | +96 [+72, +121] | 97 [73, 122] | 0.63 [0.59, 0.66] | 0.251 [0.242, 0.259] | 0 of 9 | 0 of 9 | 0 of 9 |
| Transformer | `ar_bounded` | 5.81 [5.27, 6.37] | +59 [+31, +88] | 67 [46, 91] | 0.67 [0.65, 0.68] | 0.238 [0.232, 0.243] | 3 of 10 | 1 of 10 | 0 of 10 |
| Transformer | `ar_bounded` + `kmeans_8` | 7.33 [6.46, 8.41] | +83 [+61, +108] | 85 [64, 109] | 0.64 [0.62, 0.66] | 0.234 [0.228, 0.239] | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` + `kmeans_8` | 7.05 [6.19, 8.11] | +54 [+27, +82] | 65 [47, 88] | 0.64 [0.62, 0.66] | 0.241 [0.235, 0.247] | 1 of 10 | 3 of 10 | 0 of 10 |
| Transformer | `no_ar` + `kmeans_8` | 6.74 [6.36, 7.15] | +65 [+52, +80] | 65 [53, 80] | 0.61 [0.60, 0.62] | 0.249 [0.243, 0.255] | 0 of 10 | 0 of 10 | 0 of 10 |

**Rate 0.30, churn 80%** (9–10 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | <ins>3.03 [2.86, 3.22]</ins> | **−12 [−18, −5]** | <ins>35 [34, 36]</ins> | **0.70 [0.68, 0.72]** | — | — | — | — |
| LSTM | `no_ar` | 3.26 [3.04, 3.49] | <ins>−15 [−22, −8]</ins> | **32 [28, 36]** | 0.51 [0.49, 0.53] | 0.121 [0.116, 0.127] | 7 of 10 | 3 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` | 4.63 [3.48, 6.06] | +50 [+12, +95] | 70 [41, 107] | 0.48 [0.46, 0.50] | 0.131 [0.126, 0.137] | 5 of 10 | 4 of 10 | 0 of 10 |
| LSTM | `ar_bounded` | **3.01 [2.85, 3.19]** | <ins>−13 [−23, −1]</ins> | 35 [33, 39] | 0.51 [0.48, 0.53] | **0.117 [0.110, 0.123]** | 5 of 10 | 4 of 10 | 0 of 10 |
| LSTM | `ar_bounded` + `kmeans_8` | 4.08 [3.89, 4.26] | +83 [+67, +100] | 84 [69, 101] | 0.49 [0.46, 0.51] | 0.118 [0.111, 0.125] | 0 of 10 | 0 of 10 | 0 of 10 |
| LSTM | `ar_unbounded` + `kmeans_8` | 3.89 [3.52, 4.25] | 0 [−17, +18] | 47 [41, 53] | 0.48 [0.47, 0.50] | 0.126 [0.119, 0.133] | 1 of 10 | 3 of 10 | 0 of 10 |
| LSTM | `no_ar` + `kmeans_8` | 5.34 [4.83, 5.86] | +142 [+115, +170] | 142 [115, 170] | 0.48 [0.46, 0.49] | 0.127 [0.119, 0.134] | 0 of 10 | 0 of 10 | 0 of 10 |
| Transformer | `no_ar` | 4.06 [3.69, 4.49] | +86 [+43, +133] | 96 [62, 138] | 0.44 [0.42, 0.46] | 0.141 [0.133, 0.148] | 2 of 10 | 2 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` | 4.84 [3.90, 6.23] | +112 [+61, +168] | 118 [71, 171] | 0.43 [0.40, 0.46] | 0.128 [0.120, 0.136] | 1 of 9 | 1 of 9 | 0 of 9 |
| Transformer | `ar_bounded` | 3.18 [2.98, 3.38] | +46 [+30, +61] | 54 [45, 64] | 0.48 [0.46, 0.50] | 0.119 [0.112, 0.125] | 2 of 10 | 2 of 10 | 0 of 10 |
| Transformer | `ar_bounded` + `kmeans_8` | 4.54 [3.83, 5.33] | +87 [+56, +122] | 91 [61, 123] | 0.47 [0.44, 0.50] | 0.118 [0.112, 0.124] | 1 of 10 | 1 of 10 | 0 of 10 |
| Transformer | `ar_unbounded` + `kmeans_8` | 4.26 [3.89, 4.65] | +39 [+19, +57] | 56 [48, 65] | 0.47 [0.44, 0.49] | 0.123 [0.115, 0.130] | 0 of 10 | 1 of 10 | 0 of 10 |
| Transformer | `no_ar` + `kmeans_8` | 4.96 [4.36, 5.63] | +119 [+88, +148] | 120 [90, 148] | 0.45 [0.44, 0.47] | 0.129 [0.122, 0.137] | 0 of 10 | 0 of 10 | 0 of 10 |

### By purchase rate, churn pooled

The same tables with the four churn levels pooled, 40 panels per row. These pool four
cells, so they are description only: means, with no interval and no marks, and the
"Beats P/NBD" columns count wins over the rate's 40 panels.

**Rate 0.01**, churn 20–80% pooled (40 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | 0.62 | +35 | 90 | 0.23 | — | — | — | — |
| LSTM | `no_ar` | 0.84 | +309 | 314 | 0.00 | 0.058 | 0 of 40 | 0 of 40 | 0 of 40 |
| LSTM | `ar_unbounded` | 3.38 | +505 | 513 | −0.18 | 0.058 | 1 of 40 | 1 of 40 | 0 of 40 |
| LSTM | `ar_bounded` | 0.77 | +231 | 243 | −0.06 | 0.058 | 3 of 40 | 1 of 40 | 0 of 40 |
| LSTM | `ar_bounded` + `kmeans_8` | 0.77 | +210 | 221 | 0.10 | 0.057 | 0 of 40 | 0 of 40 | 1 of 40 |
| LSTM | `ar_unbounded` + `kmeans_8` | 7.05 | +1107 | 1118 | −0.07 | 0.056 | 0 of 40 | 0 of 40 | 0 of 40 |
| LSTM | `no_ar` + `kmeans_8` | 0.85 | +302 | 307 | 0.14 | 0.057 | 0 of 40 | 0 of 40 | 1 of 40 |
| Transformer | `no_ar` | 0.67 | +74 | 120 | 0.05 | 0.056 | 14 of 40 | 15 of 40 | 0 of 40 |
| Transformer | `ar_unbounded` | 0.75 | +89 | 150 | 0.00 | 0.055 | 20 of 40 | 18 of 40 | 1 of 40 |
| Transformer | `ar_bounded` | 0.65 | +50 | 103 | 0.04 | 0.056 | 15 of 40 | 11 of 40 | 1 of 40 |
| Transformer | `ar_bounded` + `kmeans_8` | 0.71 | +60 | 119 | 0.15 | 0.053 | 15 of 40 | 13 of 40 | 5 of 40 |
| Transformer | `ar_unbounded` + `kmeans_8` | 0.70 | +44 | 121 | 0.05 | 0.051 | 8 of 40 | 9 of 40 | 1 of 40 |
| Transformer | `no_ar` + `kmeans_8` | 0.73 | +71 | 130 | 0.14 | 0.055 | 16 of 40 | 11 of 40 | 2 of 40 |

**Rate 0.05**, churn 20–80% pooled (40 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | 1.44 | +11 | 44 | 0.54 | — | — | — | — |
| LSTM | `no_ar` | 2.10 | +199 | 200 | 0.13 | 0.125 | 2 of 40 | 0 of 40 | 0 of 40 |
| LSTM | `ar_unbounded` | 11.88 | +582 | 582 | 0.36 | 0.124 | 0 of 40 | 0 of 40 | 0 of 40 |
| LSTM | `ar_bounded` | 1.64 | +44 | 58 | 0.49 | 0.118 | 10 of 40 | 5 of 40 | 1 of 40 |
| LSTM | `ar_bounded` + `kmeans_8` | 1.89 | +121 | 126 | 0.42 | 0.116 | 2 of 40 | 0 of 40 | 0 of 40 |
| LSTM | `ar_unbounded` + `kmeans_8` | 10.18 | +418 | 424 | 0.38 | 0.121 | 3 of 40 | 2 of 40 | 0 of 40 |
| LSTM | `no_ar` + `kmeans_8` | 2.08 | +200 | 202 | 0.41 | 0.118 | 0 of 40 | 0 of 40 | 0 of 40 |
| Transformer | `no_ar` | 1.99 | +109 | 114 | 0.35 | 0.122 | 4 of 40 | 4 of 40 | 0 of 40 |
| Transformer | `ar_unbounded` | 1.95 | +95 | 105 | 0.43 | 0.118 | 3 of 40 | 1 of 40 | 0 of 40 |
| Transformer | `ar_bounded` | 1.74 | +64 | 78 | 0.48 | 0.118 | 9 of 40 | 4 of 40 | 1 of 40 |
| Transformer | `ar_bounded` + `kmeans_8` | 1.94 | +85 | 96 | 0.47 | 0.113 | 6 of 40 | 3 of 40 | 0 of 40 |
| Transformer | `ar_unbounded` + `kmeans_8` | 1.97 | +72 | 87 | 0.44 | 0.113 | 7 of 40 | 7 of 40 | 0 of 40 |
| Transformer | `no_ar` + `kmeans_8` | 2.00 | +115 | 122 | 0.45 | 0.114 | 3 of 40 | 2 of 40 | 0 of 40 |

**Rate 0.10**, churn 20–80% pooled (40 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | 2.19 | +4 | 37 | 0.65 | — | — | — | — |
| LSTM | `no_ar` | 2.71 | +93 | 98 | 0.55 | 0.177 | 11 of 40 | 4 of 40 | 0 of 40 |
| LSTM | `ar_unbounded` | 12.71 | +315 | 322 | 0.53 | 0.181 | 2 of 40 | 2 of 40 | 0 of 40 |
| LSTM | `ar_bounded` | 2.42 | +20 | 36 | 0.62 | 0.169 | 26 of 40 | 10 of 40 | 0 of 40 |
| LSTM | `ar_bounded` + `kmeans_8` | 2.86 | +76 | 78 | 0.56 | 0.166 | 4 of 40 | 0 of 40 | 0 of 40 |
| LSTM | `ar_unbounded` + `kmeans_8` | 8.21 | +147 | 162 | 0.54 | 0.173 | 2 of 40 | 2 of 40 | 0 of 40 |
| LSTM | `no_ar` + `kmeans_8` | 3.13 | +138 | 139 | 0.51 | 0.172 | 2 of 40 | 0 of 40 | 0 of 40 |
| Transformer | `no_ar` | 3.18 | +126 | 130 | 0.52 | 0.178 | 4 of 40 | 1 of 40 | 0 of 40 |
| Transformer | `ar_unbounded` | 3.79 | +148 | 149 | 0.58 | 0.171 | 0 of 40 | 0 of 40 | 0 of 40 |
| Transformer | `ar_bounded` | 2.73 | +64 | 77 | 0.60 | 0.170 | 12 of 40 | 2 of 40 | 0 of 40 |
| Transformer | `ar_bounded` + `kmeans_8` | 3.12 | +92 | 97 | 0.57 | 0.164 | 3 of 40 | 1 of 40 | 0 of 40 |
| Transformer | `ar_unbounded` + `kmeans_8` | 3.16 | +81 | 88 | 0.56 | 0.165 | 5 of 40 | 1 of 40 | 0 of 40 |
| Transformer | `no_ar` + `kmeans_8` | 3.34 | +114 | 116 | 0.56 | 0.167 | 1 of 40 | 0 of 40 | 0 of 40 |

**Rate 0.30**, churn 20–80% pooled (38–40 panels)

| Model | Arm | RMSE | Bias % | MAPE | Spearman | Val. CE | Beats P/NBD: MAPE | Beats P/NBD: \|bias\| | Beats P/NBD: Spearman |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| **Pareto/NBD** | — | 5.17 | −13 | 32 | 0.79 | — | — | — | — |
| LSTM | `no_ar` | 5.16 | +2 | 21 | 0.72 | 0.304 | 37 of 40 | 20 of 40 | 0 of 40 |
| LSTM | `ar_unbounded` | 16.34 | +131 | 141 | 0.65 | 0.334 | 7 of 40 | 5 of 40 | 0 of 40 |
| LSTM | `ar_bounded` | 4.94 | −2 | 20 | 0.72 | 0.299 | 35 of 40 | 27 of 40 | 0 of 40 |
| LSTM | `ar_bounded` + `kmeans_8` | 6.61 | +56 | 56 | 0.67 | 0.300 | 9 of 40 | 0 of 40 | 0 of 40 |
| LSTM | `ar_unbounded` + `kmeans_8` | 9.06 | +33 | 61 | 0.66 | 0.318 | 8 of 40 | 11 of 40 | 0 of 40 |
| LSTM | `no_ar` + `kmeans_8` | 7.31 | +78 | 78 | 0.65 | 0.306 | 7 of 40 | 0 of 40 | 0 of 40 |
| Transformer | `no_ar` | 6.47 | +43 | 56 | 0.60 | 0.327 | 17 of 40 | 11 of 40 | 0 of 40 |
| Transformer | `ar_unbounded` | 8.92 | +89 | 92 | 0.63 | 0.323 | 2 of 38 | 2 of 38 | 0 of 38 |
| Transformer | `ar_bounded` | 5.98 | +38 | 50 | 0.66 | 0.305 | 15 of 40 | 11 of 40 | 0 of 40 |
| Transformer | `ar_bounded` + `kmeans_8` | 6.96 | +60 | 64 | 0.65 | 0.301 | 6 of 40 | 4 of 40 | 0 of 40 |
| Transformer | `ar_unbounded` + `kmeans_8` | 7.00 | +36 | 50 | 0.66 | 0.306 | 7 of 40 | 10 of 40 | 0 of 40 |
| Transformer | `no_ar` + `kmeans_8` | 7.11 | +64 | 67 | 0.63 | 0.314 | 8 of 40 | 4 of 40 | 0 of 40 |

### The three best trees per cell, by MAPE

Ranked by mean MAPE over the cell's 10 panels (lowest first); MAPE carries its 95% bootstrap
interval, the other metrics are means. "Δ vs rank 1" is the paired difference of that tree
from the cell's leader on the same panels, with its 95% bootstrap interval; "clearly behind"
is yes when that interval excludes 0. An interval end printed as 0 is not exactly 0 before
rounding. The "Beats P/NBD" counts are over the cell's panels.

| Rate | Churn | Rank | Model | Arm | **MAPE** | Spearman | RMSE | Bias % | Val. CE | Δ vs rank 1 [95% CI] | Clearly behind rank 1 | Beats P/NBD: MAPE / \|bias\| / Spearman |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | :---: | ---: |
| 0.01 | 20% | 1 | **Pareto/NBD** | — | 48 [45, 51] | 0.22 | 0.79 | +12 | — | — | — | — |
|  |  | 2 | Transformer | `ar_unbounded` | 49 [44, 56] | −0.05 | 0.83 | +7 | 0.077 | +1 [−3, +7] | no | 6 / 6 / 0 of 10 |
|  |  | 3 | Transformer | `no_ar` + `kmeans_8` | 55 [49, 63] | 0.12 | 0.86 | −3 | 0.077 | +7 [+1, +15] | yes | 4 / 2 / 0 of 10 |
| 0.01 | 40% | 1 | **Pareto/NBD** | — | 66 [62, 68] | 0.25 | 0.68 | +34 | — | — | — | — |
|  |  | 2 | Transformer | `ar_bounded` | 75 [65, 87] | −0.02 | 0.71 | +27 | 0.065 | +10 [0, +20] | no | 4 / 2 / 0 of 10 |
|  |  | 3 | Transformer | `ar_unbounded` | 78 [59, 100] | 0.00 | 0.72 | +18 | 0.065 | +12 [−5, +33] | no | 6 / 5 / 1 of 10 |
| 0.01 | 60% | 1 | **Pareto/NBD** | — | 88 [83, 93] | 0.26 | 0.57 | +41 | — | — | — | — |
|  |  | 2 | Transformer | `ar_bounded` | 98 [82, 116] | 0.06 | 0.60 | +50 | 0.049 | +10 [−7, +29] | no | 4 / 4 / 0 of 10 |
|  |  | 3 | Transformer | `ar_bounded` + `kmeans_8` | 103 [81, 129] | 0.18 | 0.63 | +43 | 0.047 | +15 [−7, +41] | no | 5 / 5 / 1 of 10 |
| 0.01 | 80% | 1 | **Pareto/NBD** | — | 159 [137, 181] | 0.20 | 0.42 | +54 | — | — | — | — |
|  |  | 2 | Transformer | `ar_bounded` | 176 [152, 201] | 0.08 | 0.44 | +89 | 0.031 | +17 [−12, +49] | no | 4 / 4 / 1 of 10 |
|  |  | 3 | Transformer | `no_ar` + `kmeans_8` | 195 [144, 271] | 0.15 | 0.48 | +104 | 0.030 | +36 [−8, +103] | no | 6 / 5 / 2 of 10 |
| 0.05 | 20% | 1 | **Pareto/NBD** | — | 35 [34, 37] | 0.58 | 1.92 | +8 | — | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` + `kmeans_8` | 39 [37, 42] | 0.45 | 2.20 | +29 | 0.169 | +4 [+2, +6] | yes | 2 / 0 / 0 of 10 |
|  |  | 3 | LSTM | `no_ar` | 45 [37, 53] | 0.29 | 2.33 | +42 | 0.175 | +9 [+2, +17] | yes | 2 / 0 / 0 of 10 |
| 0.05 | 40% | 1 | **Pareto/NBD** | — | 39 [37, 41] | 0.60 | 1.63 | +15 | — | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 56 [47, 63] | 0.54 | 1.88 | +52 | 0.138 | +17 [+9, +24] | yes | 1 / 0 / 0 of 10 |
|  |  | 3 | Transformer | `ar_bounded` + `kmeans_8` | 56 [44, 68] | 0.51 | 2.00 | +48 | 0.133 | +17 [+6, +29] | yes | 3 / 0 / 0 of 10 |
| 0.05 | 60% | 1 | **Pareto/NBD** | — | 42 [40, 44] | 0.56 | 1.32 | +9 | — | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 67 [57, 77] | 0.52 | 1.43 | +60 | 0.100 | +24 [+15, +34] | yes | 1 / 1 / 0 of 10 |
|  |  | 3 | Transformer | `ar_bounded` | 77 [56, 101] | 0.52 | 1.47 | +66 | 0.100 | +35 [+15, +58] | yes | 2 / 0 / 0 of 10 |
| 0.05 | 80% | 1 | **Pareto/NBD** | — | 58 [53, 64] | 0.44 | 0.89 | +13 | — | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 60 [53, 69] | 0.40 | 0.97 | +19 | 0.061 | +2 [−2, +8] | no | 7 / 4 / 1 of 10 |
|  |  | 3 | Transformer | `ar_bounded` + `kmeans_8` | 94 [64, 128] | 0.37 | 1.11 | +62 | 0.058 | +36 [+9, +67] | yes | 3 / 3 / 0 of 10 |
| 0.10 | 20% | 1 | **Pareto/NBD** | — | 31 [30, 32] | 0.72 | 2.91 | +3 | — | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 32 [24, 41] | 0.69 | 3.32 | +29 | 0.261 | +1 [−7, +9] | no | 4 / 0 / 0 of 10 |
|  |  | 3 | LSTM | `ar_bounded` + `kmeans_8` | 33 [29, 37] | 0.60 | 3.46 | +28 | 0.254 | +2 [−2, +5] | no | 4 / 0 / 0 of 10 |
| 0.10 | 40% | 1 | **Pareto/NBD** | — | 32 [30, 33] | 0.73 | 2.47 | +1 | — | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 34 [27, 42] | 0.69 | 2.79 | +28 | 0.198 | +3 [−5, +11] | no | 6 / 1 / 0 of 10 |
|  |  | 3 | Transformer | `ar_bounded` | 41 [32, 50] | 0.64 | 2.87 | +27 | 0.199 | +9 [+1, +17] | yes | 4 / 0 / 0 of 10 |
| 0.10 | 60% | 1 | LSTM | `ar_bounded` | 33 [30, 36] | 0.63 | 2.12 | +20 | 0.140 | — | — | 7 / 1 / 0 of 10 |
|  |  | 2 | **Pareto/NBD** | — | 37 [35, 39] | 0.65 | 2.00 | +3 | — | +4 [+1, +7] | yes | — |
|  |  | 3 | LSTM | `no_ar` | 69 [47, 93] | 0.60 | 2.43 | +61 | 0.149 | +36 [+13, +62] | yes | 3 / 2 / 0 of 10 |
| 0.10 | 80% | 1 | LSTM | `ar_bounded` | 45 [40, 50] | 0.47 | 1.45 | +4 | 0.077 | — | — | 9 / 8 / 0 of 10 |
|  |  | 2 | **Pareto/NBD** | — | 49 [44, 57] | 0.51 | 1.37 | +7 | — | +4 [+1, +8] | yes | — |
|  |  | 3 | Transformer | `ar_bounded` | 102 [68, 146] | 0.47 | 1.57 | +82 | 0.078 | +57 [+24, +97] | yes | 3 / 2 / 0 of 10 |
| 0.30 | 20% | 1 | LSTM | `ar_bounded` | 12 [10, 13] | 0.85 | 6.49 | +10 | 0.493 | — | — | 10 / 8 / 0 of 10 |
|  |  | 2 | LSTM | `no_ar` | 19 [16, 22] | 0.85 | 6.92 | +17 | 0.497 | +8 [+3, +12] | yes | 10 / 2 / 0 of 10 |
|  |  | 3 | LSTM | `ar_bounded` + `kmeans_8` | 26 [23, 29] | 0.79 | 7.96 | +25 | 0.490 | +14 [+10, +17] | yes | 9 / 0 / 0 of 10 |
| 0.30 | 40% | 1 | LSTM | `ar_bounded` | 13 [11, 14] | 0.82 | 5.66 | +2 | 0.355 | — | — | 10 / 9 / 0 of 10 |
|  |  | 2 | LSTM | `no_ar` | 14 [13, 16] | 0.81 | 5.74 | +7 | 0.361 | +2 [0, +3] | yes | 10 / 8 / 0 of 10 |
|  |  | 3 | **Pareto/NBD** | — | 31 [30, 32] | 0.83 | 5.99 | −14 | — | +18 [+17, +20] | yes | — |
| 0.30 | 60% | 1 | LSTM | `no_ar` | 18 [15, 21] | 0.70 | 4.73 | −1 | 0.237 | — | — | 10 / 7 / 0 of 10 |
|  |  | 2 | LSTM | `ar_bounded` | 19 [17, 22] | 0.70 | 4.60 | −7 | 0.232 | +1 [−1, +4] | no | 10 / 6 / 0 of 10 |
|  |  | 3 | **Pareto/NBD** | — | 32 [31, 33] | 0.77 | 4.72 | −12 | — | +14 [+11, +16] | yes | — |
| 0.30 | 80% | 1 | LSTM | `no_ar` | 32 [28, 36] | 0.51 | 3.26 | −15 | 0.121 | — | — | 7 / 3 / 0 of 10 |
|  |  | 2 | **Pareto/NBD** | — | 35 [34, 36] | 0.70 | 3.03 | −12 | — | +3 [0, +6] | no | — |
|  |  | 3 | LSTM | `ar_bounded` | 35 [33, 39] | 0.51 | 3.01 | −13 | 0.117 | +3 [+1, +6] | yes | 5 / 4 / 0 of 10 |

### The three best trees per cell, by Spearman

Ranked by mean per-customer Spearman (highest first); layout as above.

| Rate | Churn | Rank | Model | Arm | MAPE | **Spearman** | RMSE | Bias % | Val. CE | Δ vs rank 1 [95% CI] | Clearly behind rank 1 | Beats P/NBD: MAPE / \|bias\| / Spearman |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | :---: | ---: |
| 0.01 | 20% | 1 | **Pareto/NBD** | — | 48 | 0.22 [0.19, 0.25] | 0.79 | +12 | — | — | — | — |
|  |  | 2 | Transformer | `ar_bounded` + `kmeans_8` | 73 | 0.16 [0.14, 0.19] | 0.90 | +23 | 0.074 | −0.06 [−0.09, −0.03] | yes | 2 / 0 / 2 of 10 |
|  |  | 3 | LSTM | `no_ar` + `kmeans_8` | 86 | 0.14 [0.11, 0.16] | 0.91 | +77 | 0.078 | −0.08 [−0.11, −0.06] | yes | 0 / 0 / 0 of 10 |
| 0.01 | 40% | 1 | **Pareto/NBD** | — | 66 | 0.25 [0.23, 0.28] | 0.68 | +34 | — | — | — | — |
|  |  | 2 | Transformer | `ar_bounded` + `kmeans_8` | 102 | 0.16 [0.13, 0.18] | 0.80 | +64 | 0.062 | −0.10 [−0.13, −0.06] | yes | 4 / 4 / 0 of 10 |
|  |  | 3 | LSTM | `no_ar` + `kmeans_8` | 140 | 0.14 [0.11, 0.16] | 0.81 | +136 | 0.067 | −0.12 [−0.15, −0.08] | yes | 0 / 0 / 0 of 10 |
| 0.01 | 60% | 1 | **Pareto/NBD** | — | 88 | 0.26 [0.23, 0.30] | 0.57 | +41 | — | — | — | — |
|  |  | 2 | Transformer | `ar_bounded` + `kmeans_8` | 103 | 0.18 [0.12, 0.23] | 0.63 | +43 | 0.047 | −0.09 [−0.12, −0.05] | yes | 5 / 5 / 1 of 10 |
|  |  | 3 | Transformer | `no_ar` + `kmeans_8` | 120 | 0.16 [0.12, 0.21] | 0.64 | +65 | 0.047 | −0.10 [−0.15, −0.05] | yes | 4 / 2 / 0 of 10 |
| 0.01 | 80% | 1 | **Pareto/NBD** | — | 159 | 0.20 [0.16, 0.24] | 0.42 | +54 | — | — | — | — |
|  |  | 2 | Transformer | `no_ar` + `kmeans_8` | 195 | 0.15 [0.12, 0.18] | 0.48 | +104 | 0.030 | −0.05 [−0.07, −0.02] | yes | 6 / 5 / 2 of 10 |
|  |  | 3 | LSTM | `no_ar` + `kmeans_8` | 729 | 0.11 [0.07, 0.16] | 0.85 | +725 | 0.033 | −0.09 [−0.12, −0.05] | yes | 0 / 0 / 1 of 10 |
| 0.05 | 20% | 1 | **Pareto/NBD** | — | 35 | 0.58 [0.56, 0.60] | 1.92 | +8 | — | — | — | — |
|  |  | 2 | Transformer | `ar_unbounded` | 91 | 0.52 [0.50, 0.54] | 2.69 | +90 | 0.173 | −0.06 [−0.07, −0.04] | yes | 1 / 0 / 0 of 10 |
|  |  | 3 | Transformer | `ar_bounded` + `kmeans_8` | 79 | 0.51 [0.48, 0.53] | 2.65 | +77 | 0.168 | −0.07 [−0.09, −0.06] | yes | 0 / 0 / 0 of 10 |
| 0.05 | 40% | 1 | **Pareto/NBD** | — | 39 | 0.60 [0.58, 0.61] | 1.63 | +15 | — | — | — | — |
|  |  | 2 | Transformer | `ar_bounded` | 65 | 0.55 [0.52, 0.57] | 1.91 | +53 | 0.138 | −0.05 [−0.06, −0.03] | yes | 3 / 2 / 0 of 10 |
|  |  | 3 | LSTM | `ar_bounded` | 56 | 0.54 [0.51, 0.56] | 1.88 | +52 | 0.138 | −0.06 [−0.08, −0.05] | yes | 1 / 0 / 0 of 10 |
| 0.05 | 60% | 1 | **Pareto/NBD** | — | 42 | 0.56 [0.55, 0.57] | 1.32 | +9 | — | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 67 | 0.52 [0.51, 0.53] | 1.43 | +60 | 0.100 | −0.04 [−0.05, −0.03] | yes | 1 / 1 / 0 of 10 |
|  |  | 3 | Transformer | `ar_bounded` | 77 | 0.52 [0.50, 0.53] | 1.47 | +66 | 0.100 | −0.05 [−0.05, −0.04] | yes | 2 / 0 / 0 of 10 |
| 0.05 | 80% | 1 | **Pareto/NBD** | — | 58 | 0.44 [0.42, 0.47] | 0.89 | +13 | — | — | — | — |
|  |  | 2 | Transformer | `ar_bounded` | 102 | 0.40 [0.38, 0.44] | 1.04 | +76 | 0.061 | −0.03 [−0.05, −0.02] | yes | 2 / 1 / 1 of 10 |
|  |  | 3 | LSTM | `ar_bounded` | 60 | 0.40 [0.37, 0.43] | 0.97 | +19 | 0.061 | −0.04 [−0.05, −0.03] | yes | 7 / 4 / 1 of 10 |
| 0.10 | 20% | 1 | **Pareto/NBD** | — | 31 | 0.72 [0.71, 0.73] | 2.91 | +3 | — | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 32 | 0.69 [0.68, 0.70] | 3.32 | +29 | 0.261 | −0.03 [−0.04, −0.02] | yes | 4 / 0 / 0 of 10 |
|  |  | 3 | Transformer | `ar_bounded` | 52 | 0.68 [0.67, 0.69] | 3.74 | +43 | 0.262 | −0.05 [−0.05, −0.04] | yes | 4 / 0 / 0 of 10 |
| 0.10 | 40% | 1 | **Pareto/NBD** | — | 32 | 0.73 [0.72, 0.74] | 2.47 | +1 | — | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 34 | 0.69 [0.68, 0.71] | 2.79 | +28 | 0.198 | −0.03 [−0.04, −0.03] | yes | 6 / 1 / 0 of 10 |
|  |  | 3 | Transformer | `ar_bounded` + `kmeans_8` | 58 | 0.65 [0.64, 0.66] | 3.42 | +51 | 0.192 | −0.08 [−0.09, −0.06] | yes | 1 / 0 / 0 of 10 |
| 0.10 | 60% | 1 | **Pareto/NBD** | — | 37 | 0.65 [0.64, 0.66] | 2.00 | +3 | — | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 33 | 0.63 [0.61, 0.64] | 2.12 | +20 | 0.140 | −0.03 [−0.04, −0.02] | yes | 7 / 1 / 0 of 10 |
|  |  | 3 | Transformer | `ar_bounded` | 115 | 0.61 [0.59, 0.62] | 2.74 | +104 | 0.140 | −0.05 [−0.06, −0.03] | yes | 1 / 0 / 0 of 10 |
| 0.10 | 80% | 1 | **Pareto/NBD** | — | 49 | 0.51 [0.49, 0.52] | 1.37 | +7 | — | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 45 | 0.47 [0.45, 0.49] | 1.45 | +4 | 0.077 | −0.04 [−0.05, −0.03] | yes | 9 / 8 / 0 of 10 |
|  |  | 3 | Transformer | `ar_bounded` | 102 | 0.47 [0.45, 0.48] | 1.57 | +82 | 0.078 | −0.04 [−0.05, −0.03] | yes | 3 / 2 / 0 of 10 |
| 0.30 | 20% | 1 | **Pareto/NBD** | — | 30 | 0.86 [0.85, 0.87] | 6.93 | −14 | — | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 12 | 0.85 [0.85, 0.86] | 6.49 | +10 | 0.493 | −0.01 [−0.01, 0] | yes | 10 / 8 / 0 of 10 |
|  |  | 3 | LSTM | `no_ar` | 19 | 0.85 [0.85, 0.86] | 6.92 | +17 | 0.497 | −0.01 [−0.01, 0] | yes | 10 / 2 / 0 of 10 |
| 0.30 | 40% | 1 | **Pareto/NBD** | — | 31 | 0.83 [0.82, 0.84] | 5.99 | −14 | — | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 13 | 0.82 [0.81, 0.82] | 5.66 | +2 | 0.355 | −0.01 [−0.02, −0.01] | yes | 10 / 9 / 0 of 10 |
|  |  | 3 | LSTM | `no_ar` | 14 | 0.81 [0.81, 0.82] | 5.74 | +7 | 0.361 | −0.02 [−0.02, −0.01] | yes | 10 / 8 / 0 of 10 |
| 0.30 | 60% | 1 | **Pareto/NBD** | — | 32 | 0.77 [0.76, 0.77] | 4.72 | −12 | — | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 19 | 0.70 [0.69, 0.71] | 4.60 | −7 | 0.232 | −0.07 [−0.08, −0.06] | yes | 10 / 6 / 0 of 10 |
|  |  | 3 | LSTM | `no_ar` | 18 | 0.70 [0.69, 0.71] | 4.73 | −1 | 0.237 | −0.07 [−0.08, −0.06] | yes | 10 / 7 / 0 of 10 |
| 0.30 | 80% | 1 | **Pareto/NBD** | — | 35 | 0.70 [0.68, 0.72] | 3.03 | −12 | — | — | — | — |
|  |  | 2 | LSTM | `ar_bounded` | 35 | 0.51 [0.48, 0.53] | 3.01 | −13 | 0.117 | −0.19 [−0.21, −0.18] | yes | 5 / 4 / 0 of 10 |
|  |  | 3 | LSTM | `no_ar` | 32 | 0.51 [0.49, 0.53] | 3.26 | −15 | 0.121 | −0.19 [−0.20, −0.18] | yes | 7 / 3 / 0 of 10 |

### Reading

- **MAPE leader by cell.** Pareto/NBD leads all 8 cells at rates 0.01–0.05 and the two
  low-churn cells at rate 0.10. LSTM `ar_bounded` leads at rate 0.10 with churn 60–80% and at
  rate 0.30 with churn 20–40%; LSTM `no_ar` leads at rate 0.30 with churn 60–80%.
- **Most MAPE leads are not separated.** The runner-up is clearly behind the leader (paired
  interval excluding 0, 10 panels) in only 7 of 16 cells: behind Pareto/NBD at rate 0.05
  with churn 20–60%, behind the LSTM at rate 0.10 with churn 60–80% (Pareto/NBD, by 4
  points), and LSTM `no_ar` behind LSTM `ar_bounded` at rate 0.30 with churn 20–40%. At rate
  0.01 no Transformer runner-up is clearly behind Pareto/NBD.
- **Spearman leader by cell.** Pareto/NBD leads all 16 cells, and the runner-up is clearly
  behind it in every one. From rate 0.05 up the runner-up is almost always an
  `ar_bounded` arm. On sparse panels it is a `kmeans_8` arm: the cluster label raises the
  neural Spearman (LSTM `no_ar` 0.00 → 0.14 at rate 0.01, churn pooled) while leaving bias
  and MAPE no better.
- **Wins on each metric.** A neural tree beats Pareto/NBD on Spearman in 14 of 1,918
  panel comparisons: 12 on rate-0.01 panels, where both rank customers at only 0.1–0.3, and
  2 at rate 0.05, churn 80%, by less than 0.01. Wins
  on |bias| are rarer than wins on MAPE: at rate 0.30 LSTM `ar_bounded` beats Pareto/NBD's
  MAPE on 35 of 40 panels but its |bias| on 27, and at rate 0.10, churn 60%, on 7 of 10
  against 1. A lower MAPE there comes from the weekly shape, not the yearly level. These
  counts are description; the tests are in claim 1.
- **Good MAPE does not mean good ranking.** At rate 0.30 and churn 80% the LSTM leads on MAPE
  (32) but ranks customers at 0.51 against Pareto/NBD's 0.70. The paired Spearman gap
  between Pareto/NBD and the best neural arm (`ar_bounded`) is supported in all 16 cells for
  both models (claim 1): 0.26–0.34 (LSTM) and 0.12–0.27 (Transformer) at rate 0.01, and
  0.01–0.22 at rates 0.05–0.30.
- **Validation CE.** The CE gaps between arms (third decimal) are small beside the spread
  across panels, so the intervals overlap throughout. Within one panel, a lower CE across
  the 12 neural arms goes with a better holdout. Each panel gives one rank correlation over
  its 12 arms, tested against 0 inside each cell (table below): with MAPE it is supported
  and positive in 15 of 16 cells (cell means +0.21 to +0.70), with RMSE in 15 (+0.19 to
  +0.66) and with Spearman negative in 15 (−0.20 to −0.60). The one exception to all three
  is rate 0.01, churn 20%, where no correlation is clear. The ordering is loose at the top:
  the `kmeans_8` arms often have a lower CE and a worse MAPE than their `no_cluster`
  counterparts.

<details><summary>Within-panel rank correlation of Val. CE with the holdout, mean over each cell's 10 panels</summary>

| Rate | Churn | n | ρ(CE, MAPE) [95% CI] | ρ(CE, RMSE) [95% CI] | ρ(CE, Spearman) [95% CI] |
| --- | --- | --- | --- | --- | --- |
| 0.01 | 20% | 10 | 0 [−0.13, +0.14] | −0.13 [−0.30, +0.04] | −0.11 [−0.27, +0.06] |
| 0.01 | 40% | 10 | **+0.40 [+0.27, +0.53]** | **+0.30 [+0.13, +0.46]** | **−0.26 [−0.39, −0.11]** |
| 0.01 | 60% | 10 | **+0.70 [+0.59, +0.79]** | **+0.56 [+0.45, +0.65]** | **−0.51 [−0.63, −0.39]** |
| 0.01 | 80% | 10 | **+0.69 [+0.56, +0.81]** | **+0.62 [+0.45, +0.76]** | **−0.51 [−0.64, −0.36]** |
| 0.05 | 20% | 10 | **+0.21 [+0.10, +0.32]** | **+0.30 [+0.18, +0.41]** | **−0.39 [−0.50, −0.31]** |
| 0.05 | 40% | 10 | **+0.46 [+0.39, +0.53]** | **+0.36 [+0.26, +0.47]** | **−0.38 [−0.51, −0.26]** |
| 0.05 | 60% | 10 | **+0.39 [+0.30, +0.49]** | **+0.39 [+0.28, +0.50]** | **−0.55 [−0.64, −0.45]** |
| 0.05 | 80% | 10 | **+0.67 [+0.60, +0.75]** | **+0.66 [+0.58, +0.74]** | **−0.60 [−0.72, −0.46]** |
| 0.10 | 20% | 10 | **+0.45 [+0.33, +0.57]** | **+0.45 [+0.34, +0.56]** | **−0.20 [−0.30, −0.10]** |
| 0.10 | 40% | 10 | **+0.30 [+0.14, +0.47]** | **+0.24 [+0.11, +0.37]** | **−0.30 [−0.42, −0.18]** |
| 0.10 | 60% | 10 | **+0.27 [+0.14, +0.39]** | **+0.19 [+0.08, +0.30]** | **−0.37 [−0.50, −0.25]** |
| 0.10 | 80% | 10 | **+0.49 [+0.33, +0.62]** | **+0.23 [+0.07, +0.38]** | **−0.32 [−0.47, −0.16]** |
| 0.30 | 20% | 10 | **+0.53 [+0.42, +0.63]** | **+0.45 [+0.35, +0.56]** | **−0.51 [−0.60, −0.42]** |
| 0.30 | 40% | 10 | **+0.36 [+0.21, +0.50]** | **+0.43 [+0.28, +0.57]** | **−0.55 [−0.63, −0.47]** |
| 0.30 | 60% | 10 | **+0.21 [+0.08, +0.35]** | **+0.33 [+0.18, +0.49]** | **−0.51 [−0.59, −0.43]** |
| 0.30 | 80% | 10 | **+0.28 [+0.15, +0.40]** | **+0.39 [+0.28, +0.48]** | **−0.56 [−0.69, −0.42]** |

</details>

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
budget. The Transformer was not run at 3,000. Each entry is RMSE on customer totals / bias % /
MAPE, mean over 40 panels. The entries pool four cells, so they are description only and
carry no marks; the per-cell tests are claim 12 below. A 1,000 row and a 3,000 row are
different customer populations, so they are compared as independent replications.

**By purchase rate** (the 4 churn levels pooled):

| Model | Customers | Rate 0.01 | 0.05 | 0.10 | 0.30 |
| --- | ---: | --- | --- | --- | --- |
| Pareto/NBD | 1,000 | 0.62 / +35 / 90 | 1.44 / +11 / 44 | 2.19 / +4 / 37 | 5.17 / −13 / 32 |
| Pareto/NBD | 3,000 | 0.61 / +19 / 56 | 1.44 / +7 / 34 | 2.17 / 0 / 32 | 5.12 / −13 / 30 |
| LSTM `no_ar` | 1,000 | 0.84 / +309 / 314 | 2.10 / +199 / 200 | 2.71 / +93 / 98 | 5.16 / +2 / 21 |
| LSTM `no_ar` | 3,000 | 0.78 / +219 / 219 | 1.78 / +117 / 119 | 2.35 / +5 / 24 | 4.83 / −5 / 14 |
| LSTM `ar_bounded` | 1,000 | 0.77 / +231 / 243 | 1.64 / +44 / 58 | 2.42 / +20 / 36 | 4.94 / −2 / 20 |
| LSTM `ar_bounded` | 3,000 | 0.64 / +65 / 83 | 1.53 / +18 / 31 | 2.25 / +1 / 18 | 4.77 / −11 / 16 |

**By churn** (the 4 rates pooled; RMSE here mixes rates, so compare it within a column only):

| Model | Customers | Churn 20% | 40% | 60% | 80% |
| --- | ---: | --- | --- | --- | --- |
| Pareto/NBD | 1,000 | 3.14 / +2 / 36 | 2.69 / +9 / 42 | 2.15 / +10 / 50 | 1.43 / +16 / 75 |
| Pareto/NBD | 3,000 | 3.11 / +1 / 32 | 2.66 / +5 / 35 | 2.11 / +5 / 39 | 1.44 / +1 / 46 |
| LSTM `no_ar` | 1,000 | 3.35 / +37 / 42 | 2.97 / +74 / 77 | 2.49 / +127 / 134 | 2.01 / +365 / 380 |
| LSTM `no_ar` | 3,000 | 3.05 / +33 / 34 | 2.74 / +57 / 59 | 2.26 / +89 / 99 | 1.69 / +156 / 184 |
| LSTM `ar_bounded` | 1,000 | 3.24 / +32 / 39 | 2.78 / +49 / 56 | 2.22 / +66 / 79 | 1.54 / +147 / 183 |
| LSTM `ar_bounded` | 3,000 | 2.96 / +15 / 21 | 2.60 / +15 / 26 | 2.12 / +20 / 39 | 1.51 / +24 / 63 |

**Tests** (claim 12: per cell, 10 vs 10 panels, independent bootstrap):

- **LSTM `ar_bounded`:** MAPE improves in all 12 cells at rates 0.01–0.10 and |bias| in 10
  of them (not at churn 80% of rates 0.05 and 0.10). At rate 0.30 MAPE improves at churn
  20–40% only (−5 and −3 points) and |bias| at churn 20% only.
- **LSTM `no_ar`:** MAPE and |bias| improve in every cell at rates 0.05 and 0.10, and at
  rate 0.30 with churn 20–60% (MAPE). At rate 0.01 the result is mixed: better at churn 80%
  (−360 MAPE points), worse at churn 20% (+13), no clear difference at churn 40–60%.
- **Pareto/NBD:** MAPE improves in 14 of 16 cells, all except churn 20–40% at rate 0.30, and
  |bias| in 7, all at rates 0.01–0.10. There is no clear RMSE difference in any cell.

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

The MAPE gain is supported in all 16 cells (claim 7; paired, 10 panels a cell). By churn, the
pooled mean gain shrinks from 16 points at churn 20% (36 → 20) to 9 at churn 80% (75 → 66).

**Reading.**

- Seasonality moves MAPE only. RMSE on customer totals and bias measure yearly volume, which
  seasonality does not change.
- Given the season, Pareto/NBD beats the LSTM on MAPE in 13 of 16 cells, every cell at
  rates 0.01–0.10 and churn 80% at rate 0.30. The LSTM keeps a small, supported lead at rate
  0.30 with churn 20–40% (2–3 points), and churn 60% shows no clear difference. So most of
  the LSTM's dense-panel MAPE wins (claim 1) come from modelling the season, not from
  modelling customers better; on the densest, least-churned panels a small edge remains.
- The share of MAPE that is shape error (MAPE minus |bias|) tells the same story: 19–55
  points for Pareto/NBD, 9–12 for the LSTM with flags.
- **Open:** a grid without seasonality (amplitude 0), run for Pareto/NBD and the neural
  models, would measure the neural side directly.

## Impact of AR features

Bounded flags help both models, most on mid-rate panels (0.05–0.10) and at high churn, and
barely at rate 0.30 (one LSTM cell on |bias|). Unbounded counters make the LSTM several times worse at
every rate, and hurt the Transformer only on dense panels.

**Setting.** Seasonal panels, 1,000 customers, no cluster label. `no_ar` = count and calendar
only; `ar_bounded` = 0/1 flags for activity in the last 2 / 4 / 8 / 16 / 32 weeks plus
has-bought-before; `ar_unbounded` = recency, frequency and age as counters. Each cell is RMSE
on customer totals / bias % / MAPE, mean over 40 panels. The entries pool four cells, so
they are description only and carry no marks; the tests are per cell, below.

**By purchase rate** (the 4 churn levels pooled):

| Model | AR features | Rate 0.01 | 0.05 | 0.10 | 0.30 |
| --- | --- | --- | --- | --- | --- |
| Pareto/NBD | — | 0.62 / +35 / 90 | 1.44 / +11 / 44 | 2.19 / +4 / 37 | 5.17 / −13 / 32 |
| LSTM | none | 0.84 / +309 / 314 | 2.10 / +199 / 200 | 2.71 / +93 / 98 | 5.16 / +2 / 21 |
| LSTM | unbounded | 3.38 / +505 / 513 | 11.88 / +582 / 582 | 12.71 / +315 / 322 | 16.34 / +131 / 141 |
| LSTM | bounded | 0.77 / +231 / 243 | 1.64 / +44 / 58 | 2.42 / +20 / 36 | 4.94 / −2 / 20 |
| Transformer | none | 0.67 / +74 / 120 | 1.99 / +109 / 114 | 3.18 / +126 / 130 | 6.47 / +43 / 56 |
| Transformer | unbounded | 0.75 / +89 / 150 | 1.95 / +95 / 105 | 3.79 / +148 / 149 | 8.92 / +89 / 92 |
| Transformer | bounded | 0.65 / +50 / 103 | 1.74 / +64 / 78 | 2.73 / +64 / 77 | 5.98 / +38 / 50 |

**By churn** (the 4 rates pooled; compare RMSE within a column only):

| Model | AR features | Churn 20% | 40% | 60% | 80% |
| --- | --- | --- | --- | --- | --- |
| Pareto/NBD | — | 3.14 / +2 / 36 | 2.69 / +9 / 42 | 2.15 / +10 / 50 | 1.43 / +16 / 75 |
| LSTM | none | 3.35 / +37 / 42 | 2.97 / +74 / 77 | 2.49 / +127 / 134 | 2.01 / +365 / 380 |
| LSTM | unbounded | 19.44 / +328 / 330 | 12.08 / +255 / 259 | 7.19 / +333 / 337 | 5.60 / +618 / 632 |
| LSTM | bounded | 3.24 / +32 / 39 | 2.78 / +49 / 56 | 2.22 / +66 / 79 | 1.54 / +147 / 183 |
| Transformer | none | 3.99 / +41 / 57 | 3.52 / +71 / 84 | 2.85 / +94 / 110 | 1.96 / +145 / 170 |
| Transformer | unbounded | 5.16 / +67 / 78 | 4.29 / +72 / 89 | 3.39 / +111 / 128 | 2.27 / +174 / 204 |
| Transformer | bounded | 3.80 / +41 / 55 | 3.09 / +32 / 56 | 2.65 / +70 / 89 | 1.56 / +73 / 109 |

**Tests: none → bounded** (claim 3 below gives every cell with its interval). Per rate, the
pooled mean change in |bias| / MAPE / RMSE, then in how many of the rate's four cells each
change is supported (paired, 10 panels a cell):

| Model | Rate 0.01 | 0.05 | 0.10 | 0.30 |
| --- | --- | --- | --- | --- |
| LSTM | −77 / −71 / −0.07; 2 / 2 / 2 of 4 | −152 / −142 / −0.46; 3 / 3 / 3 of 4 | −70 / −62 / −0.29; 3 / 3 / 3 of 4 | −2 / −1 / −0.22; 1 / 3 / 3 of 4 (MAPE: 2 better, 1 worse) |
| Transformer | −21 / −17 / −0.02; 0 / 1 / 1 of 4 | −41 / −36 / −0.25; 2 / 2 / 3 of 4 | −57 / −53 / −0.45; 2 / 2 / 2 of 4 | −5 / −6 / −0.49; 1 / 1 / 1 of 4 |

Every supported cell above is an improvement except the one LSTM MAPE cell at rate 0.30,
churn 80% (+3).

**Tests: none → unbounded** (claim 4). The LSTM is worse on MAPE and |bias| in 13 of 16
cells and on RMSE in all 16, at every rate. The Transformer is worse on MAPE in 5 cells, all
at rates 0.10–0.30 (pooled means at rate 0.30: |bias| +41, MAPE +36, RMSE +2.4), and in no
cell at rates 0.01–0.05.

**Reading.**

- The flags matter where silence is informative but the model cannot learn it alone:
  mid-rate panels. On the sparsest panels silence says little; on the densest the plain LSTM
  already gets the level right (bias +2%).
- Unbounded counters keep growing through the holdout, past any value seen in training, and
  the LSTM's forecast climbs with them. Its RMSE rises from 2.1–5.2 to 11.9–16.3 at rates
  0.05–0.30.
- Even with flags, the neural bias at churn 80% (+147 LSTM, +73 Transformer) stays far above
  Pareto/NBD's +16 at 1,000 customers. The cohort-size section shows that gap mostly closes
  for the LSTM at 3,000.

## Claims and whether the evidence supports them

Every claim below is tested per rate × churn cell and, where the data allow, at both cohort
sizes. Rates, churn levels and the whole grid are then described by the pattern of those
per-cell verdicts. Most verdicts depend on the conditions, so each claim states where it
holds and where it does not. All numbers are recomputed from the stored forecasts by
`.scratch/synthetic-grid/metrics_by_cell.py` and `claims.py`.

**Setting.** Seasonal panels (4 peaks, amplitude 1.5), 1,000 customers unless a table says
3,000. Four purchase rates × four churn rates (20, 40, 60 and 80% of customers dropped out
by week 52), 10 panels per cell, 40 per rate, 160 over the grid.

**How to read the tests** (`docs/statistical-protocol.md`).

- **Effect Δ.** Every comparison is written "A → B" and Δ = mean(B) − mean(A) over the cell's
  panels. For MAPE, |bias| and RMSE a negative Δ means B is better; for Spearman, R_A and
  shape correlation a positive Δ does. MAPE and |bias| are in percentage points.
- **Interval.** The 95% percentile-bootstrap interval of Δ from 10,000 resamples
  (`panelclv.evaluation.effects.effect`). **Paired** comparisons (two trees on the same
  panels) resample the panels with their pairs intact. **Independent** comparisons (1,000
  vs 3,000 customers, claim 12) resample each side separately.
- **Bold** means supported: the cell's interval excludes 0. There is no multiple-testing
  correction and no p-value. An interval containing 0 means no clear difference at n = 10,
  never "no difference".
- **Pooled entries are description.** In each grid, the "churn pooled" column and the "all"
  row (in italics) are pooled mean differences over 40 or 160 panels, with no interval and
  no verdict. The collapsed "by rate" tables give the pooled means and, per metric, how many
  of the rate's cells support a difference and in which direction ("3/4 (−)" = three of four
  cells, all with B lower).
- **Grids.** One grid per model and context: rows are purchase rates, columns churn levels,
  and each cell reads "Δ MAPE / Δ |bias| / Δ Spearman". The intervals behind every cell are
  in the collapsed tables under each grid. An interval end printed as 0 is not exactly 0
  before rounding.

### Summary

| # | Claim | Verdict | Holds | Does not hold |
| --- | --- | --- | --- | --- |
| 1 | Pareto/NBD beats the neural models. | **Partly** | Ranking (Spearman): 94 of 96 cell comparisons, both cohort sizes. Level (MAPE): against the Transformer in 12–14 of 16 cells and never the other way; against the LSTM in every cell at rate 0.01 and at rate 0.05 with churn 20–60%. | LSTM `ar_bounded` on MAPE at rate 0.10 (churn 60–80%) and rate 0.30 (churn 20–60%) at 1,000 customers, and at rates 0.10–0.30 at 3,000; the LSTM ranks slightly better (+0.003) at 3,000, rate 0.30, churn 20%. |
| 2 | Neural error rises with churn. | **Partly** (described, not tested) | \|bias\| of the flagless LSTM at rates 0.01–0.05, both cohort sizes; MAPE of most trees. | Not neural-specific for MAPE: Pareto/NBD's also rises at every step at all four rates. With the flags the LSTM's \|bias\| no longer rises steadily at rates 0.05–0.10. |
| 3 | Bounded AR flags help. | **Supported, conditionally** | LSTM: 10 of 16 cells on MAPE, at rates 0.01–0.10 from churn 40–60% up (−24 to −437 points per supported cell); at 3,000 customers every cell at rate 0.01. Transformer: 6 of 16 cells, mostly at rates 0.05–0.10. | Churn 20% below rate 0.30; rate 0.30, where the LSTM changes by a few points either way; most Transformer cells. |
| 4 | Unbounded counters hurt. | **Supported for the LSTM, partly for the Transformer** | LSTM: 13 of 16 cells, every rate, +38 to +818 MAPE points. Transformer: 5 cells, all at rates 0.10–0.30. | Transformer at rates 0.01–0.05; ranking improves for both models at rate 0.05. |
| 5 | A k-means cluster label hurts. | **Partly** | LSTM level from rate 0.10 up (from 0.05 with flags). | It raises Spearman on sparse panels; with the unbounded counters it helps the Transformer and mostly helps the LSTM, except two rate-0.01 cells where it hurts badly. |
| 6 | One architecture is better overall. | **Not supported** | — | The Transformer is better at rate 0.01 and with unbounded counters; the LSTM at rates 0.10–0.30 with `no_ar` or flags. |
| 7 | Neural models capture seasonality; Pareto/NBD cannot. | **Supported** | Shape correlation 0.11–0.96 (LSTM) against −0.09 to +0.13 (Pareto/NBD); the true season cuts Pareto/NBD's MAPE by 5–20 points, supported in every cell. | — |
| 8 | Pareto/NBD's low bias means it is accurate per customer. | **Not supported** | — | In every cell R_A < 1 and L_D > 0 are both supported: it serves living customers only 51–87% of their volume and leaks 11–104% onto dead ones. |
| 9 | Neural models cannot detect a customer who has stopped. | **Partly** | Transformer: more leakage than Pareto/NBD in 15 of 16 cells. LSTM: every cell at rates 0.01–0.10, and rate 0.30 with churn 20–40%. | LSTM `ar_bounded` at rate 0.30 with churn 60–80%: no clear difference from Pareto/NBD. |
| 10 | A bigger hyperparameter search helps. | **Partly** | Level at rate 0.30 (3 LSTM cells, 2 Transformer); the Transformer's ranking in 11 of 16 cells. | Level at rates 0.01–0.10 (a few scattered cells, one of them worse). |
| 11 | Specific hyperparameters drive the error. | **Not supported** (described, not tested) | Only as a pattern: the Transformer `no_ar` layer count correlates positively with \|bias\| and MAPE in 13 of 16 cells. | Within cells nearly every hyperparameter's correlation splits in sign across cells; the large pooled correlations are the regime choosing both variables. |
| 12 | More customers improve the neural forecast. | **Supported, conditionally** | LSTM `ar_bounded`: every cell at rates 0.01–0.10. LSTM `no_ar`: every cell at rates 0.05–0.10, most at 0.30. | LSTM `no_ar` at rate 0.01 (mixed: worse at churn 20%, better at 80%); LSTM `ar_bounded` at rate 0.30 above churn 40%. The Transformer was not run at 3,000. |
| 13 | RMSE can rank these models. | **Not supported** for per-week RMSE; **supported** within one rate for customer-total RMSE (described, not tested) | Customer-total RMSE ranks trees like MAPE (ρ +0.88 to +0.97 within each rate). | Per-week RMSE puts 4–10 of 13 trees on the same value to two decimals. |

### 1. Pareto/NBD beats the neural models

**Verdict: partly.** Pareto/NBD ranks customers better than the neural tree in 94 of the 96
cell comparisons (16 cells × six neural contexts). The other two are both LSTM arms at 3,000
customers, rate 0.30, churn 20%, which rank customers better than Pareto/NBD by +0.003 and
+0.004: supported, and small. On the level of the forecast Pareto/NBD beats the Transformer
wherever a difference is clear and the LSTM on sparse panels, but the LSTM beats it on MAPE
on dense panels, and at 3,000 customers from rate 0.10 up.

- **Ranking.** Δ Spearman is supported in all 96 cells, negative in 94. The gap is largest at
  rate 0.01 (−0.12 to −0.34) and at rate 0.30 with churn 80% (−0.19 to −0.26).
- **Level, LSTM `ar_bounded`.** Pareto/NBD is better on MAPE in every cell at rate 0.01
  (+16 to +433 points for the LSTM) and at rate 0.05 with churn 20–60% (+13 to +24); churn
  80% at rate 0.05 shows no clear difference. At rate 0.10 the LSTM is better on MAPE at
  churn 60–80% (by 4 points each), while |bias| favours Pareto/NBD at churn 20–60% (+12 to
  +27). At rate 0.30 with churn 20–60% the LSTM is better by 13–18 MAPE points, and by 5–8
  |bias| points at churn 20–40%.
- **Level, Transformer.** Pareto/NBD is better on MAPE in 12 of 16 cells against the
  Transformer with flags and 14 of 16 without, and the Transformer is better in none. The
  cells with no clear difference are rate 0.01 with churn 40–80% and rate 0.30 with churn
  20% (with flags), and rate 0.30 with churn 20–40% (without).
- **At 3,000 customers** the LSTM `ar_bounded` beats Pareto/NBD on MAPE in every cell at rate
  0.10 (−4 to −19 points) and at rate 0.30 with churn 20–60% (−13 to −24). At rate 0.05 no
  cell shows a clear difference, and at rate 0.01 Pareto/NBD is better at churn 40–80%. The
  LSTM still has the larger |bias| in 12 of 16 cells.

Δ = neural minus Pareto/NBD: positive MAPE/|bias| or negative Spearman means Pareto/NBD
is better.

**LSTM `ar_bounded`, 1,000 customers**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **+16** / **+34** / **−0.26** | **+55** / **+80** / **−0.34** | **+108** / **+149** / **−0.31** | **+433** / **+521** / **−0.27** | *+153 / +196 / −0.29* |
| 0.05 | **+13** / **+38** / **−0.07** | **+17** / **+37** / **−0.06** | **+24** / **+48** / **−0.04** | +2 / +6 / **−0.04** | *+14 / +32 / −0.06* |
| 0.10 | +1 / **+27** / **−0.03** | +3 / **+23** / **−0.03** | **−4** / **+12** / **−0.03** | **−4** / −5 / **−0.04** | *−1 / +14 / −0.03* |
| 0.30 | **−18** / **−5** / **−0.01** | **−18** / **−8** / **−0.01** | **−13** / −2 / **−0.07** | 0 / +6 / **−0.19** | *−12 / −2 / −0.07* |
| all |  |  |  |  | *+38 / +60 / −0.11* |

**LSTM `no_ar`, 1,000 customers**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **+23** / **+47** / **−0.21** | **+71** / **+98** / **−0.24** | **+203** / **+249** / **−0.25** | **+600** / **+701** / **−0.22** | *+224 / +274 / −0.23* |
| 0.05 | **+9** / **+33** / **−0.29** | **+60** / **+83** / **−0.40** | **+115** / **+145** / **−0.53** | **+439** / **+477** / **−0.44** | *+156 / +184 / −0.42* |
| 0.10 | +2 / **+28** / **−0.05** | **+26** / **+51** / **−0.09** | **+32** / **+52** / **−0.05** | **+184** / **+204** / **−0.24** | *+61 / +84 / −0.11* |
| 0.30 | **−11** / +3 / **−0.01** | **−16** / **−5** / **−0.02** | **−14** / −4 / **−0.07** | −3 / +4 / **−0.19** | *−11 / 0 / −0.07* |
| all |  |  |  |  | *+108 / +135 / −0.21* |

**Transformer `ar_bounded`, 1,000 customers**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **+14** / **+32** / **−0.19** | +10 / **+24** / **−0.27** | +10 / +20 / **−0.20** | +17 / +35 / **−0.12** | *+13 / +28 / −0.20* |
| 0.05 | **+33** / **+54** / **−0.12** | **+26** / **+39** / **−0.05** | **+35** / **+56** / **−0.05** | **+44** / **+64** / **−0.03** | *+34 / +53 / −0.06* |
| 0.10 | **+21** / **+42** / **−0.05** | **+9** / **+28** / **−0.09** | **+78** / **+104** / **−0.05** | **+53** / **+69** / **−0.04** | *+40 / +61 / −0.05* |
| 0.30 | +7 / +12 / **−0.05** | **+11** / **+21** / **−0.13** | **+35** / **+49** / **−0.10** | **+19** / **+33** / **−0.22** | *+18 / +29 / −0.12* |
| all |  |  |  |  | *+26 / +43 / −0.11* |

**Transformer `no_ar`, 1,000 customers**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **+10** / **+25** / **−0.17** | **+31** / **+42** / **−0.24** | **+41** / **+61** / **−0.20** | **+38** / **+67** / **−0.13** | *+30 / +49 / −0.18* |
| 0.05 | **+39** / **+63** / **−0.27** | **+71** / **+93** / **−0.18** | **+107** / **+131** / **−0.19** | **+64** / **+91** / **−0.12** | *+70 / +95 / −0.19* |
| 0.10 | **+27** / **+53** / **−0.14** | **+62** / **+85** / **−0.15** | **+66** / **+90** / **−0.13** | **+216** / **+244** / **−0.10** | *+93 / +118 / −0.13* |
| 0.30 | +6 / +10 / **−0.15** | +5 / **+15** / **−0.16** | **+25** / **+37** / **−0.18** | **+61** / **+75** / **−0.26** | *+24 / +34 / −0.19* |
| all |  |  |  |  | *+54 / +74 / −0.17* |

**LSTM `ar_bounded`, 3,000 customers**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | +1 / **+14** / **−0.18** | **+5** / **+15** / **−0.21** | **+27** / **+43** / **−0.19** | **+74** / **+105** / **−0.22** | *+27 / +44 / −0.20* |
| 0.05 | −5 / **+17** / **−0.04** | −4 / **+15** / **−0.05** | −1 / **+15** / **−0.04** | −2 / +8 / **−0.04** | *−3 / +14 / −0.04* |
| 0.10 | **−17** / **+9** / **−0.02** | **−19** / **+4** / **−0.01** | **−15** / **+4** / **−0.01** | **−4** / **+7** / **−0.04** | *−14 / +6 / −0.02* |
| 0.30 | **−24** / **−12** / **0** | **−20** / **−7** / **−0.01** | **−13** / +1 / **−0.07** | 0 / **+12** / **−0.19** | *−14 / −1 / −0.07* |
| all |  |  |  |  | *−1 / +16 / −0.08* |

**LSTM `no_ar`, 3,000 customers**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **+47** / **+73** / **−0.22** | **+85** / **+109** / **−0.29** | **+199** / **+231** / **−0.28** | **+322** / **+377** / **−0.28** | *+163 / +197 / −0.27* |
| 0.05 | **−5** / **+18** / **−0.07** | **+36** / **+59** / **−0.13** | **+72** / **+95** / **−0.23** | **+238** / **+265** / **−0.36** | *+85 / +109 / −0.20* |
| 0.10 | **−13** / **+15** / **−0.04** | −2 / **+26** / **−0.04** | **−15** / **+5** / **−0.02** | −3 / **+15** / **−0.04** | *−8 / +15 / −0.03* |
| 0.30 | **−23** / **−9** / **0** | **−21** / **−10** / **−0.01** | **−18** / **−6** / **−0.07** | −3 / **+11** / **−0.19** | *−16 / −3 / −0.07* |
| all |  |  |  |  | *+56 / +80 / −0.14* |

<details><summary>By rate, churn pooled (descriptive)</summary>

| Comparison | Rate | n (A / B) | MAPE A → B | \|bias\| A → B | Δ MAPE (pooled mean) · cells supported | Δ \|bias\| (pooled mean) · cells supported | Δ RMSE (pooled mean) · cells supported | Δ Spearman (pooled mean) · cells supported |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM `ar_bounded`, 1,000 customers | 0.01 | 40 / 40 | 90 → 243 | 35 → 232 | +153 · 4/4 (+) | +196 · 4/4 (+) | +0.15 · 4/4 (+) | −0.29 · 4/4 (−) |
| LSTM `ar_bounded`, 1,000 customers | 0.05 | 40 / 40 | 44 → 58 | 14 → 46 | +14 · 3/4 (+) | +32 · 3/4 (+) | +0.20 · 4/4 (+) | −0.06 · 4/4 (−) |
| LSTM `ar_bounded`, 1,000 customers | 0.10 | 40 / 40 | 37 → 36 | 9 → 24 | −1 · 2/4 (−) | +14 · 3/4 (+) | +0.23 · 4/4 (+) | −0.03 · 4/4 (−) |
| LSTM `ar_bounded`, 1,000 customers | 0.30 | 40 / 40 | 32 → 20 | 13 → 11 | −12 · 3/4 (−) | −2 · 2/4 (−) | −0.22 · 3/4 (−) | −0.07 · 4/4 (−) |
| LSTM `ar_bounded`, 1,000 customers | all | 160 / 160 | 51 → 89 | 18 → 78 | +38 · 12/16 (7+, 5−) | +60 · 12/16 (10+, 2−) | +0.09 · 15/16 (12+, 3−) | −0.11 · 16/16 (−) |
| LSTM `no_ar`, 1,000 customers | 0.01 | 40 / 40 | 90 → 314 | 35 → 309 | +224 · 4/4 (+) | +274 · 4/4 (+) | +0.22 · 4/4 (+) | −0.23 · 4/4 (−) |
| LSTM `no_ar`, 1,000 customers | 0.05 | 40 / 40 | 44 → 200 | 14 → 199 | +156 · 4/4 (+) | +184 · 4/4 (+) | +0.66 · 4/4 (+) | −0.42 · 4/4 (−) |
| LSTM `no_ar`, 1,000 customers | 0.10 | 40 / 40 | 37 → 98 | 9 → 93 | +61 · 3/4 (+) | +84 · 4/4 (+) | +0.52 · 4/4 (+) | −0.11 · 4/4 (−) |
| LSTM `no_ar`, 1,000 customers | 0.30 | 40 / 40 | 32 → 21 | 13 → 13 | −11 · 3/4 (−) | 0 · 1/4 (−) | 0 · 2/4 (1+, 1−) | −0.07 · 4/4 (−) |
| LSTM `no_ar`, 1,000 customers | all | 160 / 160 | 51 → 158 | 18 → 154 | +108 · 14/16 (11+, 3−) | +135 · 13/16 (12+, 1−) | +0.35 · 14/16 (13+, 1−) | −0.21 · 16/16 (−) |
| Transformer `ar_bounded`, 1,000 customers | 0.01 | 40 / 40 | 90 → 103 | 35 → 63 | +13 · 1/4 (+) | +28 · 2/4 (+) | +0.03 · 3/4 (+) | −0.20 · 4/4 (−) |
| Transformer `ar_bounded`, 1,000 customers | 0.05 | 40 / 40 | 44 → 78 | 14 → 68 | +34 · 4/4 (+) | +53 · 4/4 (+) | +0.30 · 4/4 (+) | −0.06 · 4/4 (−) |
| Transformer `ar_bounded`, 1,000 customers | 0.10 | 40 / 40 | 37 → 77 | 9 → 70 | +40 · 4/4 (+) | +61 · 4/4 (+) | +0.54 · 4/4 (+) | −0.05 · 4/4 (−) |
| Transformer `ar_bounded`, 1,000 customers | 0.30 | 40 / 40 | 32 → 50 | 13 → 42 | +18 · 3/4 (+) | +29 · 3/4 (+) | +0.82 · 3/4 (+) | −0.12 · 4/4 (−) |
| Transformer `ar_bounded`, 1,000 customers | all | 160 / 160 | 51 → 77 | 18 → 61 | +26 · 12/16 (+) | +43 · 13/16 (+) | +0.42 · 14/16 (+) | −0.11 · 16/16 (−) |
| Transformer `no_ar`, 1,000 customers | 0.01 | 40 / 40 | 90 → 120 | 35 → 84 | +30 · 4/4 (+) | +49 · 4/4 (+) | +0.05 · 3/4 (+) | −0.18 · 4/4 (−) |
| Transformer `no_ar`, 1,000 customers | 0.05 | 40 / 40 | 44 → 114 | 14 → 109 | +70 · 4/4 (+) | +95 · 4/4 (+) | +0.55 · 4/4 (+) | −0.19 · 4/4 (−) |
| Transformer `no_ar`, 1,000 customers | 0.10 | 40 / 40 | 37 → 130 | 9 → 128 | +93 · 4/4 (+) | +118 · 4/4 (+) | +1.00 · 4/4 (+) | −0.13 · 4/4 (−) |
| Transformer `no_ar`, 1,000 customers | 0.30 | 40 / 40 | 32 → 56 | 13 → 48 | +24 · 2/4 (+) | +34 · 3/4 (+) | +1.31 · 4/4 (+) | −0.19 · 4/4 (−) |
| Transformer `no_ar`, 1,000 customers | all | 160 / 160 | 51 → 105 | 18 → 92 | +54 · 14/16 (+) | +74 · 15/16 (+) | +0.73 · 15/16 (+) | −0.17 · 16/16 (−) |
| LSTM `ar_bounded`, 3,000 customers | 0.01 | 40 / 40 | 56 → 83 | 21 → 65 | +27 · 3/4 (+) | +44 · 4/4 (+) | +0.04 · 4/4 (+) | −0.20 · 4/4 (−) |
| LSTM `ar_bounded`, 3,000 customers | 0.05 | 40 / 40 | 34 → 31 | 8 → 22 | −3 · 0/4 | +14 · 3/4 (+) | +0.09 · 4/4 (+) | −0.04 · 4/4 (−) |
| LSTM `ar_bounded`, 3,000 customers | 0.10 | 40 / 40 | 32 → 18 | 4 → 9 | −14 · 4/4 (−) | +6 · 4/4 (+) | +0.08 · 4/4 (+) | −0.02 · 4/4 (−) |
| LSTM `ar_bounded`, 3,000 customers | 0.30 | 40 / 40 | 30 → 16 | 13 → 12 | −14 · 3/4 (−) | −1 · 3/4 (1+, 2−) | −0.34 · 3/4 (−) | −0.07 · 4/4 (1+, 3−) |
| LSTM `ar_bounded`, 3,000 customers | all | 160 / 160 | 38 → 37 | 12 → 27 | −1 · 10/16 (3+, 7−) | +16 · 14/16 (12+, 2−) | −0.03 · 15/16 (12+, 3−) | −0.08 · 16/16 (1+, 15−) |
| LSTM `no_ar`, 3,000 customers | 0.01 | 40 / 40 | 56 → 219 | 21 → 219 | +163 · 4/4 (+) | +197 · 4/4 (+) | +0.18 · 4/4 (+) | −0.27 · 4/4 (−) |
| LSTM `no_ar`, 3,000 customers | 0.05 | 40 / 40 | 34 → 119 | 8 → 118 | +85 · 4/4 (3+, 1−) | +109 · 4/4 (+) | +0.34 · 4/4 (+) | −0.20 · 4/4 (−) |
| LSTM `no_ar`, 3,000 customers | 0.10 | 40 / 40 | 32 → 24 | 4 → 19 | −8 · 2/4 (−) | +15 · 4/4 (+) | +0.18 · 4/4 (+) | −0.03 · 4/4 (−) |
| LSTM `no_ar`, 3,000 customers | 0.30 | 40 / 40 | 30 → 14 | 13 → 10 | −16 · 3/4 (−) | −3 · 4/4 (1+, 3−) | −0.29 · 4/4 (1+, 3−) | −0.07 · 4/4 (1+, 3−) |
| LSTM `no_ar`, 3,000 customers | all | 160 / 160 | 38 → 94 | 12 → 91 | +56 · 13/16 (7+, 6−) | +80 · 16/16 (13+, 3−) | +0.10 · 16/16 (13+, 3−) | −0.14 · 16/16 (1+, 15−) |

</details>

<details><summary>Intervals per rate × churn cell</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI] | Δ \|bias\| [95% CI] | Δ RMSE [95% CI] | Δ Spearman [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM `ar_bounded`, 1,000 customers | 0.01 | 20% | 10 | 48 → 65 | 12 → 47 | **+16 [+4, +33]** | **+34 [+16, +55]** | **+0.07 [+0.04, +0.11]** | **−0.26 [−0.33, −0.19]** |
| LSTM `ar_bounded`, 1,000 customers | 0.01 | 40% | 10 | 66 → 121 | 34 → 114 | **+55 [+43, +69]** | **+80 [+67, +95]** | **+0.10 [+0.08, +0.13]** | **−0.34 [−0.39, −0.30]** |
| LSTM `ar_bounded`, 1,000 customers | 0.01 | 60% | 10 | 88 → 196 | 41 → 190 | **+108 [+72, +150]** | **+149 [+104, +198]** | **+0.14 [+0.09, +0.20]** | **−0.31 [−0.38, −0.23]** |
| LSTM `ar_bounded`, 1,000 customers | 0.01 | 80% | 10 | 159 → 591 | 54 → 576 | **+433 [+229, +663]** | **+521 [+302, +768]** | **+0.29 [+0.18, +0.40]** | **−0.27 [−0.32, −0.22]** |
| LSTM `ar_bounded`, 1,000 customers | 0.05 | 20% | 10 | 35 → 48 | 8 → 47 | **+13 [+7, +19]** | **+38 [+31, +45]** | **+0.36 [+0.31, +0.42]** | **−0.07 [−0.11, −0.05]** |
| LSTM `ar_bounded`, 1,000 customers | 0.05 | 40% | 10 | 39 → 56 | 15 → 52 | **+17 [+9, +24]** | **+37 [+28, +45]** | **+0.25 [+0.21, +0.28]** | **−0.06 [−0.08, −0.05]** |
| LSTM `ar_bounded`, 1,000 customers | 0.05 | 60% | 10 | 42 → 67 | 12 → 61 | **+24 [+15, +34]** | **+48 [+32, +61]** | **+0.12 [+0.06, +0.17]** | **−0.04 [−0.05, −0.03]** |
| LSTM `ar_bounded`, 1,000 customers | 0.05 | 80% | 10 | 58 → 60 | 21 → 27 | +2 [−2, +8] | +6 [−4, +18] | **+0.08 [+0.05, +0.11]** | **−0.04 [−0.05, −0.03]** |
| LSTM `ar_bounded`, 1,000 customers | 0.10 | 20% | 10 | 31 → 32 | 3 → 30 | +1 [−7, +9] | **+27 [+17, +37]** | **+0.41 [+0.31, +0.54]** | **−0.03 [−0.04, −0.02]** |
| LSTM `ar_bounded`, 1,000 customers | 0.10 | 40% | 10 | 32 → 34 | 5 → 28 | +3 [−5, +11] | **+23 [+13, +34]** | **+0.32 [+0.22, +0.44]** | **−0.03 [−0.04, −0.03]** |
| LSTM `ar_bounded`, 1,000 customers | 0.10 | 60% | 10 | 37 → 33 | 9 → 21 | **−4 [−7, −1]** | **+12 [+7, +17]** | **+0.11 [+0.08, +0.15]** | **−0.03 [−0.04, −0.02]** |
| LSTM `ar_bounded`, 1,000 customers | 0.10 | 80% | 10 | 49 → 45 | 21 → 15 | **−4 [−8, −1]** | −5 [−15, +6] | **+0.08 [+0.04, +0.13]** | **−0.04 [−0.05, −0.03]** |
| LSTM `ar_bounded`, 1,000 customers | 0.30 | 20% | 10 | 30 → 12 | 14 → 10 | **−18 [−20, −17]** | **−5 [−8, −2]** | **−0.43 [−0.65, −0.20]** | **−0.01 [−0.01, 0]** |
| LSTM `ar_bounded`, 1,000 customers | 0.30 | 40% | 10 | 31 → 13 | 14 → 6 | **−18 [−20, −17]** | **−8 [−11, −4]** | **−0.33 [−0.45, −0.21]** | **−0.01 [−0.02, −0.01]** |
| LSTM `ar_bounded`, 1,000 customers | 0.30 | 60% | 10 | 32 → 19 | 13 → 10 | **−13 [−15, −10]** | −2 [−8, +2] | **−0.11 [−0.19, −0.03]** | **−0.07 [−0.08, −0.06]** |
| LSTM `ar_bounded`, 1,000 customers | 0.30 | 80% | 10 | 35 → 35 | 13 → 19 | 0 [−2, +3] | +6 [−2, +15] | −0.02 [−0.11, +0.08] | **−0.19 [−0.21, −0.18]** |
| LSTM `no_ar`, 1,000 customers | 0.01 | 20% | 10 | 48 → 71 | 12 → 59 | **+23 [+15, +34]** | **+47 [+37, +60]** | **+0.07 [+0.05, +0.10]** | **−0.21 [−0.26, −0.16]** |
| LSTM `no_ar`, 1,000 customers | 0.01 | 40% | 10 | 66 → 136 | 34 → 132 | **+71 [+56, +87]** | **+98 [+86, +113]** | **+0.12 [+0.10, +0.14]** | **−0.24 [−0.28, −0.21]** |
| LSTM `no_ar`, 1,000 customers | 0.01 | 60% | 10 | 88 → 291 | 41 → 290 | **+203 [+182, +227]** | **+249 [+227, +273]** | **+0.26 [+0.23, +0.29]** | **−0.25 [−0.31, −0.21]** |
| LSTM `no_ar`, 1,000 customers | 0.01 | 80% | 10 | 159 → 759 | 54 → 755 | **+600 [+401, +826]** | **+701 [+493, +933]** | **+0.43 [+0.33, +0.53]** | **−0.22 [−0.26, −0.17]** |
| LSTM `no_ar`, 1,000 customers | 0.05 | 20% | 10 | 35 → 45 | 8 → 42 | **+9 [+2, +17]** | **+33 [+25, +43]** | **+0.40 [+0.28, +0.53]** | **−0.29 [−0.44, −0.14]** |
| LSTM `no_ar`, 1,000 customers | 0.05 | 40% | 10 | 39 → 99 | 15 → 98 | **+60 [+49, +71]** | **+83 [+71, +95]** | **+0.62 [+0.52, +0.70]** | **−0.40 [−0.55, −0.25]** |
| LSTM `no_ar`, 1,000 customers | 0.05 | 60% | 10 | 42 → 158 | 12 → 157 | **+115 [+98, +134]** | **+145 [+127, +163]** | **+0.64 [+0.56, +0.71]** | **−0.53 [−0.56, −0.51]** |
| LSTM `no_ar`, 1,000 customers | 0.05 | 80% | 10 | 58 → 497 | 21 → 497 | **+439 [+392, +486]** | **+477 [+428, +525]** | **+0.99 [+0.85, +1.13]** | **−0.44 [−0.48, −0.41]** |
| LSTM `no_ar`, 1,000 customers | 0.10 | 20% | 10 | 31 → 33 | 3 → 31 | +2 [−5, +9] | **+28 [+21, +36]** | **+0.38 [+0.27, +0.48]** | **−0.05 [−0.06, −0.04]** |
| LSTM `no_ar`, 1,000 customers | 0.10 | 40% | 10 | 32 → 58 | 5 → 57 | **+26 [+6, +57]** | **+51 [+30, +83]** | **+0.64 [+0.29, +1.16]** | **−0.09 [−0.16, −0.04]** |
| LSTM `no_ar`, 1,000 customers | 0.10 | 60% | 10 | 37 → 69 | 9 → 61 | **+32 [+9, +57]** | **+52 [+24, +82]** | **+0.43 [+0.31, +0.57]** | **−0.05 [−0.07, −0.04]** |
| LSTM `no_ar`, 1,000 customers | 0.10 | 80% | 10 | 49 → 233 | 21 → 225 | **+184 [+83, +312]** | **+204 [+96, +335]** | **+0.65 [+0.38, +0.92]** | **−0.24 [−0.36, −0.12]** |
| LSTM `no_ar`, 1,000 customers | 0.30 | 20% | 10 | 30 → 19 | 14 → 17 | **−11 [−14, −8]** | +3 [−2, +7] | −0.01 [−0.24, +0.19] | **−0.01 [−0.01, 0]** |
| LSTM `no_ar`, 1,000 customers | 0.30 | 40% | 10 | 31 → 14 | 14 → 9 | **−16 [−19, −14]** | **−5 [−9, −1]** | **−0.24 [−0.34, −0.14]** | **−0.02 [−0.02, −0.01]** |
| LSTM `no_ar`, 1,000 customers | 0.30 | 60% | 10 | 32 → 18 | 13 → 9 | **−14 [−16, −11]** | −4 [−9, +1] | +0.02 [−0.11, +0.16] | **−0.07 [−0.08, −0.06]** |
| LSTM `no_ar`, 1,000 customers | 0.30 | 80% | 10 | 35 → 32 | 13 → 17 | −3 [−6, 0] | +4 [−1, +10] | **+0.23 [+0.11, +0.37]** | **−0.19 [−0.20, −0.18]** |
| Transformer `ar_bounded`, 1,000 customers | 0.01 | 20% | 10 | 48 → 62 | 12 → 44 | **+14 [+2, +30]** | **+32 [+14, +53]** | **+0.06 [+0.03, +0.09]** | **−0.19 [−0.24, −0.14]** |
| Transformer `ar_bounded`, 1,000 customers | 0.01 | 40% | 10 | 66 → 75 | 34 → 58 | +10 [0, +20] | **+24 [+10, +38]** | **+0.03 [+0.02, +0.04]** | **−0.27 [−0.30, −0.24]** |
| Transformer `ar_bounded`, 1,000 customers | 0.01 | 60% | 10 | 88 → 98 | 41 → 60 | +10 [−7, +29] | +20 [−11, +51] | **+0.03 [0, +0.05]** | **−0.20 [−0.24, −0.16]** |
| Transformer `ar_bounded`, 1,000 customers | 0.01 | 80% | 10 | 159 → 176 | 54 → 90 | +17 [−12, +49] | +35 [−17, +90] | +0.01 [−0.01, +0.04] | **−0.12 [−0.19, −0.05]** |
| Transformer `ar_bounded`, 1,000 customers | 0.05 | 20% | 10 | 35 → 68 | 8 → 63 | **+33 [+19, +45]** | **+54 [+33, +72]** | **+0.64 [+0.47, +0.78]** | **−0.12 [−0.16, −0.09]** |
| Transformer `ar_bounded`, 1,000 customers | 0.05 | 40% | 10 | 39 → 65 | 15 → 54 | **+26 [+2, +56]** | **+39 [+9, +74]** | **+0.28 [+0.11, +0.50]** | **−0.05 [−0.06, −0.03]** |
| Transformer `ar_bounded`, 1,000 customers | 0.05 | 60% | 10 | 42 → 77 | 12 → 69 | **+35 [+15, +58]** | **+56 [+32, +82]** | **+0.15 [+0.09, +0.23]** | **−0.05 [−0.05, −0.04]** |
| Transformer `ar_bounded`, 1,000 customers | 0.05 | 80% | 10 | 58 → 102 | 21 → 84 | **+44 [+11, +83]** | **+64 [+21, +113]** | **+0.15 [+0.09, +0.20]** | **−0.03 [−0.05, −0.02]** |
| Transformer `ar_bounded`, 1,000 customers | 0.10 | 20% | 10 | 31 → 52 | 3 → 45 | **+21 [+6, +37]** | **+42 [+23, +62]** | **+0.83 [+0.63, +1.05]** | **−0.05 [−0.05, −0.04]** |
| Transformer `ar_bounded`, 1,000 customers | 0.10 | 40% | 10 | 32 → 41 | 5 → 33 | **+9 [+1, +17]** | **+28 [+16, +40]** | **+0.40 [+0.23, +0.60]** | **−0.09 [−0.19, −0.03]** |
| Transformer `ar_bounded`, 1,000 customers | 0.10 | 60% | 10 | 37 → 115 | 9 → 113 | **+78 [+43, +118]** | **+104 [+68, +146]** | **+0.74 [+0.40, +1.14]** | **−0.05 [−0.06, −0.03]** |
| Transformer `ar_bounded`, 1,000 customers | 0.10 | 80% | 10 | 49 → 102 | 21 → 90 | **+53 [+21, +90]** | **+69 [+31, +110]** | **+0.20 [+0.10, +0.30]** | **−0.04 [−0.05, −0.03]** |
| Transformer `ar_bounded`, 1,000 customers | 0.30 | 20% | 10 | 30 → 37 | 14 → 27 | +7 [−4, +19] | +12 [−4, +29] | **+1.13 [+0.11, +2.25]** | **−0.05 [−0.07, −0.04]** |
| Transformer `ar_bounded`, 1,000 customers | 0.30 | 40% | 10 | 31 → 42 | 14 → 35 | **+11 [+1, +24]** | **+21 [+6, +38]** | **+0.89 [+0.22, +1.68]** | **−0.13 [−0.23, −0.05]** |
| Transformer `ar_bounded`, 1,000 customers | 0.30 | 60% | 10 | 32 → 67 | 13 → 62 | **+35 [+14, +59]** | **+49 [+24, +76]** | **+1.09 [+0.53, +1.68]** | **−0.10 [−0.12, −0.09]** |
| Transformer `ar_bounded`, 1,000 customers | 0.30 | 80% | 10 | 35 → 54 | 13 → 46 | **+19 [+9, +30]** | **+33 [+13, +51]** | +0.15 [−0.01, +0.30] | **−0.22 [−0.23, −0.20]** |
| Transformer `no_ar`, 1,000 customers | 0.01 | 20% | 10 | 48 → 58 | 12 → 37 | **+10 [+1, +20]** | **+25 [+8, +42]** | **+0.05 [+0.03, +0.07]** | **−0.17 [−0.21, −0.12]** |
| Transformer `no_ar`, 1,000 customers | 0.01 | 40% | 10 | 66 → 96 | 34 → 76 | **+31 [+1, +68]** | **+42 [+3, +87]** | **+0.07 [+0.02, +0.13]** | **−0.24 [−0.28, −0.18]** |
| Transformer `no_ar`, 1,000 customers | 0.01 | 60% | 10 | 88 → 129 | 41 → 102 | **+41 [+9, +78]** | **+61 [+16, +108]** | **+0.06 [+0.02, +0.10]** | **−0.20 [−0.23, −0.16]** |
| Transformer `no_ar`, 1,000 customers | 0.01 | 80% | 10 | 159 → 197 | 54 → 121 | **+38 [+11, +69]** | **+67 [+19, +114]** | +0.02 [0, +0.05] | **−0.13 [−0.18, −0.08]** |
| Transformer `no_ar`, 1,000 customers | 0.05 | 20% | 10 | 35 → 75 | 8 → 72 | **+39 [+26, +54]** | **+63 [+45, +80]** | **+0.77 [+0.59, +0.97]** | **−0.27 [−0.38, −0.17]** |
| Transformer `no_ar`, 1,000 customers | 0.05 | 40% | 10 | 39 → 110 | 15 → 108 | **+71 [+47, +93]** | **+93 [+68, +116]** | **+0.65 [+0.48, +0.82]** | **−0.18 [−0.22, −0.14]** |
| Transformer `no_ar`, 1,000 customers | 0.05 | 60% | 10 | 42 → 149 | 12 → 144 | **+107 [+55, +156]** | **+131 [+73, +187]** | **+0.55 [+0.35, +0.74]** | **−0.19 [−0.23, −0.15]** |
| Transformer `no_ar`, 1,000 customers | 0.05 | 80% | 10 | 58 → 122 | 21 → 111 | **+64 [+30, +101]** | **+91 [+49, +133]** | **+0.24 [+0.18, +0.28]** | **−0.12 [−0.16, −0.09]** |
| Transformer `no_ar`, 1,000 customers | 0.10 | 20% | 10 | 31 → 58 | 3 → 56 | **+27 [+14, +41]** | **+53 [+37, +69]** | **+1.20 [+0.94, +1.43]** | **−0.14 [−0.16, −0.12]** |
| Transformer `no_ar`, 1,000 customers | 0.10 | 40% | 10 | 32 → 94 | 5 → 91 | **+62 [+36, +87]** | **+85 [+55, +113]** | **+1.23 [+0.94, +1.50]** | **−0.15 [−0.16, −0.14]** |
| Transformer `no_ar`, 1,000 customers | 0.10 | 60% | 10 | 37 → 103 | 9 → 99 | **+66 [+33, +103]** | **+90 [+52, +130]** | **+0.72 [+0.51, +0.97]** | **−0.13 [−0.15, −0.11]** |
| Transformer `no_ar`, 1,000 customers | 0.10 | 80% | 10 | 49 → 265 | 21 → 265 | **+216 [+155, +286]** | **+244 [+186, +312]** | **+0.84 [+0.66, +1.00]** | **−0.10 [−0.12, −0.09]** |
| Transformer `no_ar`, 1,000 customers | 0.30 | 20% | 10 | 30 → 36 | 14 → 24 | +6 [−4, +20] | +10 [−4, +27] | **+1.38 [+0.60, +2.50]** | **−0.15 [−0.16, −0.13]** |
| Transformer `no_ar`, 1,000 customers | 0.30 | 40% | 10 | 31 → 36 | 14 → 29 | +5 [−3, +15] | **+15 [+3, +28]** | **+1.35 [+0.93, +1.94]** | **−0.16 [−0.17, −0.14]** |
| Transformer `no_ar`, 1,000 customers | 0.30 | 60% | 10 | 32 → 57 | 13 → 50 | **+25 [+9, +43]** | **+37 [+15, +60]** | **+1.47 [+1.12, +1.89]** | **−0.18 [−0.19, −0.17]** |
| Transformer `no_ar`, 1,000 customers | 0.30 | 80% | 10 | 35 → 96 | 13 → 88 | **+61 [+27, +103]** | **+75 [+32, +124]** | **+1.03 [+0.69, +1.46]** | **−0.26 [−0.28, −0.24]** |
| LSTM `ar_bounded`, 3,000 customers | 0.01 | 20% | 10 | 37 → 38 | 11 → 25 | +1 [−4, +5] | **+14 [+6, +21]** | **+0.03 [+0.02, +0.04]** | **−0.18 [−0.20, −0.16]** |
| LSTM `ar_bounded`, 3,000 customers | 0.01 | 40% | 10 | 46 → 52 | 22 → 37 | **+5 [+1, +10]** | **+15 [+7, +24]** | **+0.04 [+0.03, +0.04]** | **−0.21 [−0.26, −0.15]** |
| LSTM `ar_bounded`, 3,000 customers | 0.01 | 60% | 10 | 62 → 89 | 30 → 72 | **+27 [+13, +41]** | **+43 [+25, +61]** | **+0.03 [+0.03, +0.05]** | **−0.19 [−0.25, −0.13]** |
| LSTM `ar_bounded`, 3,000 customers | 0.01 | 80% | 10 | 78 → 152 | 22 → 127 | **+74 [+33, +127]** | **+105 [+53, +165]** | **+0.06 [+0.03, +0.09]** | **−0.22 [−0.30, −0.15]** |
| LSTM `ar_bounded`, 3,000 customers | 0.05 | 20% | 10 | 32 → 26 | 7 → 24 | −5 [−11, 0] | **+17 [+11, +23]** | **+0.11 [+0.08, +0.13]** | **−0.04 [−0.05, −0.03]** |
| LSTM `ar_bounded`, 3,000 customers | 0.05 | 40% | 10 | 33 → 29 | 10 → 25 | −4 [−11, +4] | **+15 [+5, +25]** | **+0.11 [+0.08, +0.15]** | **−0.05 [−0.06, −0.04]** |
| LSTM `ar_bounded`, 3,000 customers | 0.05 | 60% | 10 | 34 → 33 | 7 → 22 | −1 [−9, +10] | **+15 [+2, +29]** | **+0.08 [+0.06, +0.11]** | **−0.04 [−0.05, −0.03]** |
| LSTM `ar_bounded`, 3,000 customers | 0.05 | 80% | 10 | 39 → 37 | 8 → 16 | −2 [−4, +1] | +8 [−1, +16] | **+0.05 [+0.03, +0.07]** | **−0.04 [−0.05, −0.03]** |
| LSTM `ar_bounded`, 3,000 customers | 0.10 | 20% | 10 | 30 → 14 | 2 → 11 | **−17 [−20, −13]** | **+9 [+4, +14]** | **+0.11 [+0.07, +0.15]** | **−0.02 [−0.02, −0.01]** |
| LSTM `ar_bounded`, 3,000 customers | 0.10 | 40% | 10 | 30 → 12 | 2 → 6 | **−19 [−21, −16]** | **+4 [+1, +7]** | **+0.06 [+0.04, +0.09]** | **−0.01 [−0.02, −0.01]** |
| LSTM `ar_bounded`, 3,000 customers | 0.10 | 60% | 10 | 32 → 16 | 3 → 7 | **−15 [−17, −13]** | **+4 [+1, +7]** | **+0.07 [+0.04, +0.10]** | **−0.01 [−0.02, −0.01]** |
| LSTM `ar_bounded`, 3,000 customers | 0.10 | 80% | 10 | 35 → 32 | 7 → 14 | **−4 [−7, −1]** | **+7 [+1, +11]** | **+0.07 [+0.05, +0.09]** | **−0.04 [−0.05, −0.04]** |
| LSTM `ar_bounded`, 3,000 customers | 0.30 | 20% | 10 | 30 → 6 | 15 → 3 | **−24 [−24, −23]** | **−12 [−13, −10]** | **−0.87 [−0.98, −0.75]** | **0 [0, +0.01]** |
| LSTM `ar_bounded`, 3,000 customers | 0.30 | 40% | 10 | 30 → 10 | 14 → 7 | **−20 [−22, −19]** | **−7 [−9, −6]** | **−0.44 [−0.52, −0.37]** | **−0.01 [−0.01, −0.01]** |
| LSTM `ar_bounded`, 3,000 customers | 0.30 | 60% | 10 | 30 → 18 | 13 → 13 | **−13 [−15, −10]** | +1 [−4, +5] | **−0.15 [−0.22, −0.08]** | **−0.07 [−0.08, −0.07]** |
| LSTM `ar_bounded`, 3,000 customers | 0.30 | 80% | 10 | 31 → 31 | 12 → 24 | 0 [−5, +4] | **+12 [+6, +17]** | +0.09 [0, +0.16] | **−0.19 [−0.20, −0.18]** |
| LSTM `no_ar`, 3,000 customers | 0.01 | 20% | 10 | 37 → 84 | 11 → 83 | **+47 [+40, +54]** | **+73 [+66, +80]** | **+0.11 [+0.10, +0.12]** | **−0.22 [−0.23, −0.20]** |
| LSTM `no_ar`, 3,000 customers | 0.01 | 40% | 10 | 46 → 132 | 22 → 131 | **+85 [+76, +97]** | **+109 [+100, +123]** | **+0.15 [+0.13, +0.17]** | **−0.29 [−0.32, −0.26]** |
| LSTM `no_ar`, 3,000 customers | 0.01 | 60% | 10 | 62 → 261 | 30 → 261 | **+199 [+168, +232]** | **+231 [+200, +264]** | **+0.23 [+0.19, +0.27]** | **−0.28 [−0.31, −0.25]** |
| LSTM `no_ar`, 3,000 customers | 0.01 | 80% | 10 | 78 → 399 | 22 → 399 | **+322 [+269, +377]** | **+377 [+321, +434]** | **+0.22 [+0.19, +0.26]** | **−0.28 [−0.31, −0.25]** |
| LSTM `no_ar`, 3,000 customers | 0.05 | 20% | 10 | 32 → 26 | 7 → 25 | **−5 [−10, −1]** | **+18 [+13, +23]** | **+0.15 [+0.11, +0.20]** | **−0.07 [−0.08, −0.05]** |
| LSTM `no_ar`, 3,000 customers | 0.05 | 40% | 10 | 33 → 69 | 10 → 69 | **+36 [+23, +48]** | **+59 [+47, +71]** | **+0.32 [+0.25, +0.38]** | **−0.13 [−0.15, −0.11]** |
| LSTM `no_ar`, 3,000 customers | 0.05 | 60% | 10 | 34 → 105 | 7 → 102 | **+72 [+37, +103]** | **+95 [+57, +129]** | **+0.39 [+0.24, +0.52]** | **−0.23 [−0.37, −0.11]** |
| LSTM `no_ar`, 3,000 customers | 0.05 | 80% | 10 | 39 → 277 | 8 → 274 | **+238 [+154, +312]** | **+265 [+176, +344]** | **+0.52 [+0.37, +0.66]** | **−0.36 [−0.45, −0.26]** |
| LSTM `no_ar`, 3,000 customers | 0.10 | 20% | 10 | 30 → 18 | 2 → 17 | **−13 [−15, −10]** | **+15 [+13, +17]** | **+0.21 [+0.19, +0.23]** | **−0.04 [−0.05, −0.03]** |
| LSTM `no_ar`, 3,000 customers | 0.10 | 40% | 10 | 30 → 29 | 2 → 27 | −2 [−7, +4] | **+26 [+19, +32]** | **+0.21 [+0.18, +0.25]** | **−0.04 [−0.05, −0.03]** |
| LSTM `no_ar`, 3,000 customers | 0.10 | 60% | 10 | 32 → 17 | 3 → 8 | **−15 [−16, −13]** | **+5 [+2, +8]** | **+0.15 [+0.11, +0.18]** | **−0.02 [−0.03, −0.01]** |
| LSTM `no_ar`, 3,000 customers | 0.10 | 80% | 10 | 35 → 32 | 7 → 22 | −3 [−7, +1] | **+15 [+9, +19]** | **+0.14 [+0.10, +0.17]** | **−0.04 [−0.04, −0.04]** |
| LSTM `no_ar`, 3,000 customers | 0.30 | 20% | 10 | 30 → 8 | 15 → 6 | **−23 [−24, −21]** | **−9 [−11, −7]** | **−0.74 [−0.82, −0.64]** | **0 [0, 0]** |
| LSTM `no_ar`, 3,000 customers | 0.30 | 40% | 10 | 30 → 9 | 14 → 4 | **−21 [−22, −20]** | **−10 [−12, −7]** | **−0.38 [−0.50, −0.24]** | **−0.01 [−0.01, −0.01]** |
| LSTM `no_ar`, 3,000 customers | 0.30 | 60% | 10 | 30 → 12 | 13 → 7 | **−18 [−20, −16]** | **−6 [−9, −3]** | **−0.17 [−0.27, −0.08]** | **−0.07 [−0.08, −0.07]** |
| LSTM `no_ar`, 3,000 customers | 0.30 | 80% | 10 | 31 → 28 | 12 → 23 | −3 [−7, +1] | **+11 [+7, +16]** | **+0.13 [+0.04, +0.21]** | **−0.19 [−0.20, −0.18]** |

</details>

### 2. Neural error rises with churn

**Verdict: partly, as a described pattern.** A trend across churn levels runs across
cells, and the protocol tests only within a cell, so this claim is described, not tested.
MAPE rises at every churn step for most trees, Pareto/NBD included (at all four rates at
1,000 customers), so that half of the claim is not specific to the neural models. What is
specific is the size of the |bias| rise without flags: the LSTM `no_ar` goes from 59 to 755
at rate 0.01 and from 42 to 497 at rate 0.05, against 12 to 54 and 8 to 21 for Pareto/NBD.
With bounded flags the LSTM's |bias| no longer rises steadily at rates 0.05–0.10, at either
cohort size. It still rises at rate 0.01 and, at 3,000 customers, at rate 0.30 (3 → 24).

Each row gives the mean at churn 20 / 40 / 60 / 80% over the level's 10 panels, and whether
the mean rises at every step. RMSE is left out: it falls with churn for every tree because
there is less volume.

| Tree | Customers | Rate | \|bias\| at churn 20 / 40 / 60 / 80% | MAPE at churn 20 / 40 / 60 / 80% | \|bias\| rises at every step | MAPE rises at every step |
| --- | --- | --- | --- | --- | --- | --- |
| Pareto/NBD | 1,000 | 0.01 | 12 / 34 / 41 / 54 | 48 / 66 / 88 / 159 | yes | yes |
| Pareto/NBD | 1,000 | 0.05 | 8 / 15 / 12 / 21 | 35 / 39 / 42 / 58 | no | yes |
| Pareto/NBD | 1,000 | 0.10 | 3 / 5 / 9 / 21 | 31 / 32 / 37 / 49 | yes | yes |
| Pareto/NBD | 1,000 | 0.30 | 14 / 14 / 13 / 13 | 30 / 31 / 32 / 35 | no | yes |
| LSTM `no_ar` | 1,000 | 0.01 | 59 / 132 / 290 / 755 | 71 / 136 / 291 / 759 | yes | yes |
| LSTM `no_ar` | 1,000 | 0.05 | 42 / 98 / 157 / 497 | 45 / 99 / 158 / 497 | yes | yes |
| LSTM `no_ar` | 1,000 | 0.10 | 31 / 57 / 61 / 225 | 33 / 58 / 69 / 233 | yes | yes |
| LSTM `no_ar` | 1,000 | 0.30 | 17 / 9 / 9 / 17 | 19 / 14 / 18 / 32 | no | no |
| LSTM `ar_bounded` | 1,000 | 0.01 | 47 / 114 / 190 / 576 | 65 / 121 / 196 / 591 | yes | yes |
| LSTM `ar_bounded` | 1,000 | 0.05 | 47 / 52 / 61 / 27 | 48 / 56 / 67 / 60 | no | no |
| LSTM `ar_bounded` | 1,000 | 0.10 | 30 / 28 / 21 / 15 | 32 / 34 / 33 / 45 | no | no |
| LSTM `ar_bounded` | 1,000 | 0.30 | 10 / 6 / 10 / 19 | 12 / 13 / 19 / 35 | no | yes |
| Transformer `no_ar` | 1,000 | 0.01 | 37 / 76 / 102 / 121 | 58 / 96 / 129 / 197 | yes | yes |
| Transformer `no_ar` | 1,000 | 0.05 | 72 / 108 / 144 / 111 | 75 / 110 / 149 / 122 | no | no |
| Transformer `no_ar` | 1,000 | 0.10 | 56 / 91 / 99 / 265 | 58 / 94 / 103 / 265 | yes | yes |
| Transformer `no_ar` | 1,000 | 0.30 | 24 / 29 / 50 / 88 | 36 / 36 / 57 / 96 | yes | no |
| Transformer `ar_bounded` | 1,000 | 0.01 | 44 / 58 / 60 / 90 | 62 / 75 / 98 / 176 | yes | yes |
| Transformer `ar_bounded` | 1,000 | 0.05 | 63 / 54 / 69 / 84 | 68 / 65 / 77 / 102 | no | no |
| Transformer `ar_bounded` | 1,000 | 0.10 | 45 / 33 / 113 / 90 | 52 / 41 / 115 / 102 | no | no |
| Transformer `ar_bounded` | 1,000 | 0.30 | 27 / 35 / 62 / 46 | 37 / 42 / 67 / 54 | no | no |
| Pareto/NBD | 3,000 | 0.01 | 11 / 22 / 30 / 22 | 37 / 46 / 62 / 78 | no | yes |
| Pareto/NBD | 3,000 | 0.05 | 7 / 10 / 7 / 8 | 32 / 33 / 34 / 39 | no | yes |
| Pareto/NBD | 3,000 | 0.10 | 2 / 2 / 3 / 7 | 30 / 30 / 32 / 35 | no | yes |
| Pareto/NBD | 3,000 | 0.30 | 15 / 14 / 13 / 12 | 30 / 30 / 30 / 31 | no | no |
| LSTM `no_ar` | 3,000 | 0.01 | 83 / 131 / 261 / 399 | 84 / 132 / 261 / 399 | yes | yes |
| LSTM `no_ar` | 3,000 | 0.05 | 25 / 69 / 102 / 274 | 26 / 69 / 105 / 277 | yes | yes |
| LSTM `no_ar` | 3,000 | 0.10 | 17 / 27 / 8 / 22 | 18 / 29 / 17 / 32 | no | no |
| LSTM `no_ar` | 3,000 | 0.30 | 6 / 4 / 7 / 23 | 8 / 9 / 12 / 28 | no | yes |
| LSTM `ar_bounded` | 3,000 | 0.01 | 25 / 37 / 72 / 127 | 38 / 52 / 89 / 152 | yes | yes |
| LSTM `ar_bounded` | 3,000 | 0.05 | 24 / 25 / 22 / 16 | 26 / 29 / 33 / 37 | no | yes |
| LSTM `ar_bounded` | 3,000 | 0.10 | 11 / 6 / 7 / 14 | 14 / 12 / 16 / 32 | no | no |
| LSTM `ar_bounded` | 3,000 | 0.30 | 3 / 7 / 13 / 24 | 6 / 10 / 18 / 31 | yes | yes |

### 3. Bounded AR flags help

**Verdict: supported, conditionally.** The flags improve the LSTM's level at purchase rates
0.01–0.10 once churn is 40% or more (60% or more at rate 0.01), and its ranking in every
cell at rates 0.05–0.10. The gain grows with churn, is absent at churn 20% below rate 0.30,
and is larger with more customers. At rate 0.30 the effect is small and mixed. For the
Transformer they help less: the level improves in 6 of 16 cells, the ranking in 10.

**How much, LSTM at 1,000 customers.**

- MAPE improves in 10 of 16 cells: at rate 0.01 with churn 60–80% (−95 and −168 points), at
  rates 0.05 and 0.10 with churn 40–80% (−24 to −437), and at rate 0.30 with churn 20–40%
  (−8 and −2). It worsens in one, rate 0.30 with churn 80% (+3). |bias| improves in 9 cells,
  the same pattern. Over the grid the pooled mean MAPE falls from 158 to 89.
- By churn: at rate 0.05 the change is +4 (no clear difference), −43, −91 and −437 MAPE
  points at churn 20, 40, 60 and 80%. The same rise with churn holds at rates 0.01 and 0.10.
  Where silence is common, the flags are what lets the model learn that silence means
  dropout.
- Ranking improves in all eight cells at rates 0.05–0.10 (+0.02 to +0.49 Spearman) but
  worsens in two cells at rate 0.01 (−0.10 and −0.05) and does not move at rate 0.30.

**How much, Transformer at 1,000 customers.**

- MAPE improves in 6 of 16 cells, by 31–163 points: rate 0.01 at churn 60%, rate 0.05 at
  churn 40–60%, rate 0.10 at churn 40% and 80%, rate 0.30 at churn 80%. No cell worsens.
  Over the grid the pooled mean MAPE falls from 105 to 77.
- Ranking improves in 10 cells, every cell at rate 0.05 and three of four at rates 0.10 and
  0.30, by 0.04–0.15. It worsens slightly in one (rate 0.01, churn 20%, −0.02).

**How much, LSTM at 3,000 customers.**

- The flags help more on sparse panels: MAPE improves in every cell at rate 0.01 (−46 to
  −248) and at rate 0.05 with churn 40–80% (−40 to −240). The pooled mean falls from 94 to 37.
- At 3,000 customers the flags also improve the ranking at rate 0.01 (three of four cells,
  +0.04 to +0.09), where they worsened it at 1,000.
- At rate 0.30 they cost a little: MAPE +6 at churn 60%, and |bias| +3 and +7 at churn
  40–60%.

**LSTM, 1,000 customers**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | −6 / −13 / −0.05 | −16 / −17 / **−0.10** | **−95** / **−100** / −0.05 | **−168** / **−179** / **−0.05** | *−71 / −77 / −0.06* |
| 0.05 | +4 / +5 / **+0.21** | **−43** / **−46** / **+0.34** | **−91** / **−97** / **+0.49** | **−437** / **−470** / **+0.40** | *−142 / −152 / +0.36* |
| 0.10 | 0 / −1 / **+0.02** | **−24** / **−28** / **+0.06** | **−36** / **−40** / **+0.03** | **−188** / **−210** / **+0.20** | *−62 / −70 / +0.08* |
| 0.30 | **−8** / **−8** / 0 | **−2** / −3 / 0 | +1 / +1 / 0 | **+3** / +2 / 0 | *−1 / −2 / 0* |
| all |  |  |  |  | *−69 / −75 / +0.09* |

**Transformer, 1,000 customers**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | +4 / +7 / **−0.02** | −21 / −18 / −0.04 | **−31** / −41 / 0 | −21 / −31 / +0.01 | *−17 / −21 / −0.01* |
| 0.05 | −6 / −9 / **+0.14** | **−45** / **−54** / **+0.13** | **−72** / **−75** / **+0.15** | −20 / −27 / **+0.09** | *−36 / −41 / +0.13* |
| 0.10 | −6 / −11 / **+0.10** | **−53** / **−58** / +0.06 | +12 / +14 / **+0.08** | **−163** / **−175** / **+0.06** | *−53 / −57 / +0.08* |
| 0.30 | +1 / +2 / **+0.09** | +6 / +6 / +0.03 | +10 / +13 / **+0.08** | **−42** / **−42** / **+0.04** | *−6 / −5 / +0.06* |
| all |  |  |  |  | *−28 / −31 / +0.06* |

**LSTM, 3,000 customers**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **−46** / **−59** / **+0.04** | **−80** / **−94** / **+0.09** | **−172** / **−189** / **+0.09** | **−248** / **−271** / +0.06 | *−136 / −153 / +0.07* |
| 0.05 | 0 / −1 / **+0.03** | **−40** / **−44** / **+0.07** | **−73** / **−81** / **+0.19** | **−240** / **−258** / **+0.33** | *−88 / −96 / +0.16* |
| 0.10 | **−4** / **−6** / **+0.02** | **−17** / **−22** / **+0.02** | −1 / −1 / 0 | −1 / **−8** / **0** | *−6 / −9 / +0.01* |
| 0.30 | −1 / **−2** / 0 | +1 / **+3** / 0 | **+6** / **+7** / 0 | +2 / +1 / 0 | *+2 / +2 / 0* |
| all |  |  |  |  | *−57 / −64 / +0.06* |

**By rate, churn pooled (descriptive)**

| Comparison | Rate | n (A / B) | MAPE A → B | \|bias\| A → B | Δ MAPE (pooled mean) · cells supported | Δ \|bias\| (pooled mean) · cells supported | Δ RMSE (pooled mean) · cells supported | Δ Spearman (pooled mean) · cells supported |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM, 1,000 customers | 0.01 | 40 / 40 | 314 → 243 | 309 → 232 | −71 · 2/4 (−) | −77 · 2/4 (−) | −0.07 · 2/4 (−) | −0.06 · 2/4 (−) |
| LSTM, 1,000 customers | 0.05 | 40 / 40 | 200 → 58 | 199 → 46 | −142 · 3/4 (−) | −152 · 3/4 (−) | −0.46 · 3/4 (−) | +0.36 · 4/4 (+) |
| LSTM, 1,000 customers | 0.10 | 40 / 40 | 98 → 36 | 93 → 24 | −62 · 3/4 (−) | −70 · 3/4 (−) | −0.29 · 3/4 (−) | +0.08 · 4/4 (+) |
| LSTM, 1,000 customers | 0.30 | 40 / 40 | 21 → 20 | 13 → 11 | −1 · 3/4 (1+, 2−) | −2 · 1/4 (−) | −0.22 · 3/4 (−) | 0 · 0/4 |
| LSTM, 1,000 customers | all | 160 / 160 | 158 → 89 | 154 → 78 | −69 · 11/16 (1+, 10−) | −75 · 9/16 (−) | −0.26 · 11/16 (−) | +0.09 · 10/16 (8+, 2−) |
| Transformer, 1,000 customers | 0.01 | 40 / 40 | 120 → 103 | 84 → 63 | −17 · 1/4 (−) | −21 · 0/4 | −0.02 · 1/4 (−) | −0.01 · 1/4 (−) |
| Transformer, 1,000 customers | 0.05 | 40 / 40 | 114 → 78 | 109 → 68 | −36 · 2/4 (−) | −41 · 2/4 (−) | −0.25 · 3/4 (−) | +0.13 · 4/4 (+) |
| Transformer, 1,000 customers | 0.10 | 40 / 40 | 130 → 77 | 128 → 70 | −53 · 2/4 (−) | −57 · 2/4 (−) | −0.45 · 2/4 (−) | +0.08 · 3/4 (+) |
| Transformer, 1,000 customers | 0.30 | 40 / 40 | 56 → 50 | 48 → 42 | −6 · 1/4 (−) | −5 · 1/4 (−) | −0.49 · 1/4 (−) | +0.06 · 3/4 (+) |
| Transformer, 1,000 customers | all | 160 / 160 | 105 → 77 | 92 → 61 | −28 · 6/16 (−) | −31 · 5/16 (−) | −0.30 · 7/16 (−) | +0.06 · 11/16 (10+, 1−) |
| LSTM, 3,000 customers | 0.01 | 40 / 40 | 219 → 83 | 219 → 65 | −136 · 4/4 (−) | −153 · 4/4 (−) | −0.14 · 4/4 (−) | +0.07 · 3/4 (+) |
| LSTM, 3,000 customers | 0.05 | 40 / 40 | 119 → 31 | 118 → 22 | −88 · 3/4 (−) | −96 · 3/4 (−) | −0.26 · 3/4 (−) | +0.16 · 4/4 (+) |
| LSTM, 3,000 customers | 0.10 | 40 / 40 | 24 → 18 | 19 → 9 | −6 · 2/4 (−) | −9 · 3/4 (−) | −0.10 · 4/4 (−) | +0.01 · 3/4 (2+, 1−) |
| LSTM, 3,000 customers | 0.30 | 40 / 40 | 14 → 16 | 10 → 12 | +2 · 1/4 (+) | +2 · 3/4 (2+, 1−) | −0.05 · 1/4 (−) | 0 · 0/4 |
| LSTM, 3,000 customers | all | 160 / 160 | 94 → 37 | 91 → 27 | −57 · 10/16 (1+, 9−) | −64 · 13/16 (2+, 11−) | −0.14 · 12/16 (−) | +0.06 · 10/16 (9+, 1−) |

<details><summary>Intervals per rate × churn cell</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI] | Δ \|bias\| [95% CI] | Δ RMSE [95% CI] | Δ Spearman [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM, 1,000 customers | 0.01 | 20% | 10 | 71 → 65 | 59 → 47 | −6 [−19, +8] | −13 [−31, +6] | 0 [−0.03, +0.03] | −0.05 [−0.09, 0] |
| LSTM, 1,000 customers | 0.01 | 40% | 10 | 136 → 121 | 132 → 114 | −16 [−35, +1] | −17 [−39, +1] | −0.02 [−0.05, +0.01] | **−0.10 [−0.14, −0.06]** |
| LSTM, 1,000 customers | 0.01 | 60% | 10 | 291 → 196 | 290 → 190 | **−95 [−143, −45]** | **−100 [−151, −46]** | **−0.12 [−0.18, −0.05]** | −0.05 [−0.11, +0.02] |
| LSTM, 1,000 customers | 0.01 | 80% | 10 | 759 → 591 | 755 → 576 | **−168 [−303, −47]** | **−179 [−325, −50]** | **−0.15 [−0.27, −0.03]** | **−0.05 [−0.09, −0.01]** |
| LSTM, 1,000 customers | 0.05 | 20% | 10 | 45 → 48 | 42 → 47 | +4 [−1, +8] | +5 [−1, +10] | −0.04 [−0.15, +0.06] | **+0.21 [+0.07, +0.36]** |
| LSTM, 1,000 customers | 0.05 | 40% | 10 | 99 → 56 | 98 → 52 | **−43 [−53, −33]** | **−46 [−57, −36]** | **−0.37 [−0.47, −0.26]** | **+0.34 [+0.19, +0.48]** |
| LSTM, 1,000 customers | 0.05 | 60% | 10 | 158 → 67 | 157 → 61 | **−91 [−114, −69]** | **−97 [−121, −72]** | **−0.52 [−0.64, −0.42]** | **+0.49 [+0.47, +0.51]** |
| LSTM, 1,000 customers | 0.05 | 80% | 10 | 497 → 60 | 497 → 27 | **−437 [−485, −390]** | **−470 [−521, −421]** | **−0.91 [−1.05, −0.78]** | **+0.40 [+0.37, +0.44]** |
| LSTM, 1,000 customers | 0.10 | 20% | 10 | 33 → 32 | 31 → 30 | 0 [−9, +8] | −1 [−13, +10] | +0.03 [−0.10, +0.17] | **+0.02 [+0.01, +0.03]** |
| LSTM, 1,000 customers | 0.10 | 40% | 10 | 58 → 34 | 57 → 28 | **−24 [−57, −1]** | **−28 [−64, −3]** | **−0.32 [−0.80, −0.03]** | **+0.06 [+0.01, +0.12]** |
| LSTM, 1,000 customers | 0.10 | 60% | 10 | 69 → 33 | 61 → 21 | **−36 [−62, −13]** | **−40 [−70, −12]** | **−0.32 [−0.47, −0.19]** | **+0.03 [+0.01, +0.04]** |
| LSTM, 1,000 customers | 0.10 | 80% | 10 | 233 → 45 | 225 → 15 | **−188 [−319, −86]** | **−210 [−346, −99]** | **−0.57 [−0.85, −0.28]** | **+0.20 [+0.08, +0.32]** |
| LSTM, 1,000 customers | 0.30 | 20% | 10 | 19 → 12 | 17 → 10 | **−8 [−12, −3]** | **−8 [−14, −1]** | **−0.42 [−0.67, −0.13]** | 0 [0, +0.01] |
| LSTM, 1,000 customers | 0.30 | 40% | 10 | 14 → 13 | 9 → 6 | **−2 [−3, 0]** | −3 [−6, 0] | −0.09 [−0.19, +0.02] | 0 [0, +0.01] |
| LSTM, 1,000 customers | 0.30 | 60% | 10 | 18 → 19 | 9 → 10 | +1 [−1, +4] | +1 [−4, +6] | **−0.13 [−0.21, −0.06]** | 0 [0, 0] |
| LSTM, 1,000 customers | 0.30 | 80% | 10 | 32 → 35 | 17 → 19 | **+3 [+1, +6]** | +2 [−4, +9] | **−0.25 [−0.42, −0.09]** | 0 [−0.01, +0.01] |
| Transformer, 1,000 customers | 0.01 | 20% | 10 | 58 → 62 | 37 → 44 | +4 [−5, +16] | +7 [−8, +26] | +0.01 [−0.01, +0.03] | **−0.02 [−0.03, −0.01]** |
| Transformer, 1,000 customers | 0.01 | 40% | 10 | 96 → 75 | 76 → 58 | −21 [−52, +6] | −18 [−56, +15] | −0.04 [−0.09, +0.01] | −0.04 [−0.08, 0] |
| Transformer, 1,000 customers | 0.01 | 60% | 10 | 129 → 98 | 102 → 60 | **−31 [−72, −1]** | −41 [−93, +1] | **−0.03 [−0.07, 0]** | 0 [−0.04, +0.03] |
| Transformer, 1,000 customers | 0.01 | 80% | 10 | 197 → 176 | 121 → 90 | −21 [−62, +17] | −31 [−99, +28] | −0.01 [−0.03, +0.01] | +0.01 [−0.04, +0.05] |
| Transformer, 1,000 customers | 0.05 | 20% | 10 | 75 → 68 | 72 → 63 | −6 [−25, +11] | −9 [−35, +15] | −0.13 [−0.36, +0.07] | **+0.14 [+0.03, +0.27]** |
| Transformer, 1,000 customers | 0.05 | 40% | 10 | 110 → 65 | 108 → 54 | **−45 [−81, −3]** | **−54 [−94, −8]** | **−0.37 [−0.62, −0.10]** | **+0.13 [+0.10, +0.16]** |
| Transformer, 1,000 customers | 0.05 | 60% | 10 | 149 → 77 | 144 → 69 | **−72 [−126, −19]** | **−75 [−136, −17]** | **−0.40 [−0.59, −0.21]** | **+0.15 [+0.10, +0.19]** |
| Transformer, 1,000 customers | 0.05 | 80% | 10 | 122 → 102 | 111 → 84 | −20 [−76, +41] | −27 [−94, +48] | **−0.09 [−0.16, −0.03]** | **+0.09 [+0.06, +0.12]** |
| Transformer, 1,000 customers | 0.10 | 20% | 10 | 58 → 52 | 56 → 45 | −6 [−26, +14] | −11 [−36, +14] | −0.37 [−0.70, +0.03] | **+0.10 [+0.08, +0.11]** |
| Transformer, 1,000 customers | 0.10 | 40% | 10 | 94 → 41 | 91 → 33 | **−53 [−79, −25]** | **−58 [−87, −26]** | **−0.83 [−1.19, −0.45]** | +0.06 [−0.05, +0.12] |
| Transformer, 1,000 customers | 0.10 | 60% | 10 | 103 → 115 | 99 → 113 | +12 [−35, +62] | +14 [−34, +66] | +0.02 [−0.35, +0.41] | **+0.08 [+0.06, +0.11]** |
| Transformer, 1,000 customers | 0.10 | 80% | 10 | 265 → 102 | 265 → 90 | **−163 [−206, −121]** | **−175 [−218, −135]** | **−0.64 [−0.81, −0.47]** | **+0.06 [+0.04, +0.08]** |
| Transformer, 1,000 customers | 0.30 | 20% | 10 | 36 → 37 | 24 → 27 | +1 [−17, +18] | +2 [−23, +26] | −0.25 [−1.69, +1.17] | **+0.09 [+0.08, +0.11]** |
| Transformer, 1,000 customers | 0.30 | 40% | 10 | 36 → 42 | 29 → 35 | +6 [−9, +21] | +6 [−13, +25] | −0.46 [−1.51, +0.59] | +0.03 [−0.07, +0.10] |
| Transformer, 1,000 customers | 0.30 | 60% | 10 | 57 → 67 | 50 → 62 | +10 [−16, +35] | +13 [−19, +42] | −0.38 [−1.03, +0.23] | **+0.08 [+0.06, +0.09]** |
| Transformer, 1,000 customers | 0.30 | 80% | 10 | 96 → 54 | 88 → 46 | **−42 [−85, −8]** | **−42 [−90, −2]** | **−0.87 [−1.33, −0.51]** | **+0.04 [+0.02, +0.07]** |
| LSTM, 3,000 customers | 0.01 | 20% | 10 | 84 → 38 | 83 → 25 | **−46 [−56, −37]** | **−59 [−73, −45]** | **−0.08 [−0.10, −0.06]** | **+0.04 [+0.01, +0.06]** |
| LSTM, 3,000 customers | 0.01 | 40% | 10 | 132 → 52 | 131 → 37 | **−80 [−95, −68]** | **−94 [−115, −78]** | **−0.11 [−0.14, −0.09]** | **+0.09 [+0.03, +0.15]** |
| LSTM, 3,000 customers | 0.01 | 60% | 10 | 261 → 89 | 261 → 72 | **−172 [−206, −139]** | **−189 [−226, −153]** | **−0.19 [−0.23, −0.15]** | **+0.09 [+0.03, +0.14]** |
| LSTM, 3,000 customers | 0.01 | 80% | 10 | 399 → 152 | 399 → 127 | **−248 [−302, −192]** | **−271 [−332, −208]** | **−0.17 [−0.21, −0.13]** | +0.06 [−0.01, +0.11] |
| LSTM, 3,000 customers | 0.05 | 20% | 10 | 26 → 26 | 25 → 24 | 0 [−5, +6] | −1 [−7, +6] | −0.04 [−0.10, 0] | **+0.03 [+0.01, +0.05]** |
| LSTM, 3,000 customers | 0.05 | 40% | 10 | 69 → 29 | 69 → 25 | **−40 [−52, −26]** | **−44 [−57, −30]** | **−0.21 [−0.27, −0.14]** | **+0.07 [+0.05, +0.10]** |
| LSTM, 3,000 customers | 0.05 | 60% | 10 | 105 → 33 | 102 → 22 | **−73 [−102, −40]** | **−81 [−113, −44]** | **−0.30 [−0.43, −0.17]** | **+0.19 [+0.07, +0.34]** |
| LSTM, 3,000 customers | 0.05 | 80% | 10 | 277 → 37 | 274 → 16 | **−240 [−314, −156]** | **−258 [−336, −167]** | **−0.47 [−0.62, −0.30]** | **+0.33 [+0.21, +0.41]** |
| LSTM, 3,000 customers | 0.10 | 20% | 10 | 18 → 14 | 17 → 11 | **−4 [−7, 0]** | **−6 [−10, −2]** | **−0.10 [−0.14, −0.05]** | **+0.02 [+0.02, +0.03]** |
| LSTM, 3,000 customers | 0.10 | 40% | 10 | 29 → 12 | 27 → 6 | **−17 [−22, −12]** | **−22 [−28, −16]** | **−0.15 [−0.19, −0.12]** | **+0.02 [+0.01, +0.03]** |
| LSTM, 3,000 customers | 0.10 | 60% | 10 | 17 → 16 | 8 → 7 | −1 [−3, +1] | −1 [−5, +3] | **−0.08 [−0.13, −0.03]** | 0 [−0.01, +0.01] |
| LSTM, 3,000 customers | 0.10 | 80% | 10 | 32 → 32 | 22 → 14 | −1 [−4, +2] | **−8 [−13, −3]** | **−0.07 [−0.10, −0.04]** | **0 [−0.01, 0]** |
| LSTM, 3,000 customers | 0.30 | 20% | 10 | 8 → 6 | 6 → 3 | −1 [−2, 0] | **−2 [−4, −1]** | **−0.14 [−0.20, −0.06]** | 0 [0, 0] |
| LSTM, 3,000 customers | 0.30 | 40% | 10 | 9 → 10 | 4 → 7 | +1 [0, +3] | **+3 [0, +5]** | −0.06 [−0.15, +0.01] | 0 [0, 0] |
| LSTM, 3,000 customers | 0.30 | 60% | 10 | 12 → 18 | 7 → 13 | **+6 [+3, +8]** | **+7 [+2, +11]** | +0.02 [−0.03, +0.08] | 0 [0, 0] |
| LSTM, 3,000 customers | 0.30 | 80% | 10 | 28 → 31 | 23 → 24 | +2 [−2, +6] | +1 [−5, +7] | −0.04 [−0.14, +0.04] | 0 [0, +0.01] |

</details>

### 4. Unbounded counters hurt

**Verdict: supported for the LSTM, partly for the Transformer.** The counters make the
LSTM's level worse in 13 of 16 cells, at every rate (+38 to +818 MAPE points per supported
cell), because they keep growing through the holdout, past any value seen in training. The
three cells with no clear difference are at rate 0.01 (churn 40% and 80%) and rate 0.10
(churn 80%), where the LSTM without counters is already far off. The Transformer is worse
on MAPE in 5 cells, all at rates 0.10–0.30, and in none at rates 0.01–0.05. Ranking does not
follow the level: the counters raise both models' Spearman at rate 0.05 (LSTM in all four
cells, +0.15 to +0.32; Transformer in three, +0.10 to +0.21).

**LSTM**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **+443** / **+443** / **−0.17** | +52 / +50 / **−0.24** | **+223** / **+224** / **−0.21** | +79 / +66 / **−0.10** | *+199 / +196 / −0.18* |
| 0.05 | **+195** / **+198** / **+0.15** | **+214** / **+215** / **+0.19** | **+304** / **+304** / **+0.32** | **+818** / **+817** / **+0.27** | *+383 / +384 / +0.23* |
| 0.10 | **+268** / **+269** / **−0.07** | **+317** / **+318** / −0.07 | **+239** / **+240** / **−0.08** | +73 / +70 / **+0.15** | *+224 / +224 / −0.02* |
| 0.30 | **+248** / **+250** / **−0.12** | **+147** / **+149** / **−0.08** | **+46** / **+48** / **−0.04** | **+38** / **+39** / **−0.03** | *+120 / +122 / −0.07* |
| all |  |  |  |  | *+231 / +231 / −0.01* |

**Transformer**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | −9 / **−18** / **−0.10** | −18 / −21 / −0.02 | −2 / −22 / −0.01 | +148 / +152 / −0.07 | *+30 / +23 / −0.05* |
| 0.05 | +16 / +18 / **+0.21** | −6 / −8 / **+0.10** | −25 / −22 / **+0.11** | −22 / −35 / −0.12 | *−9 / −12 / +0.08* |
| 0.10 | **+43** / **+45** / **+0.07** | −3 / −1 / **+0.06** | **+56** / **+60** / **+0.05** | −20 / −20 / **+0.04** | *+19 / +21 / +0.05* |
| 0.30 | **+35** / **+45** / +0.05 | **+49** / **+53** / +0.01 | **+38** / **+44** / **+0.04** | +14 / +15 / −0.01 | *+36 / +41 / +0.03* |
| all |  |  |  |  | *+19 / +19 / +0.02* |

<details><summary>By rate, churn pooled (descriptive)</summary>

| Comparison | Rate | n (A / B) | MAPE A → B | \|bias\| A → B | Δ MAPE (pooled mean) · cells supported | Δ \|bias\| (pooled mean) · cells supported | Δ RMSE (pooled mean) · cells supported | Δ Spearman (pooled mean) · cells supported |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM | 0.01 | 40 / 40 | 314 → 513 | 309 → 505 | +199 · 2/4 (+) | +196 · 2/4 (+) | +2.54 · 4/4 (+) | −0.18 · 4/4 (−) |
| LSTM | 0.05 | 40 / 40 | 200 → 582 | 199 → 582 | +383 · 4/4 (+) | +384 · 4/4 (+) | +9.77 · 4/4 (+) | +0.23 · 4/4 (+) |
| LSTM | 0.10 | 40 / 40 | 98 → 322 | 93 → 318 | +224 · 3/4 (+) | +224 · 3/4 (+) | +10.00 · 4/4 (+) | −0.02 · 3/4 (1+, 2−) |
| LSTM | 0.30 | 40 / 40 | 21 → 141 | 13 → 135 | +120 · 4/4 (+) | +122 · 4/4 (+) | +11.18 · 4/4 (+) | −0.07 · 4/4 (−) |
| LSTM | all | 160 / 160 | 158 → 390 | 154 → 385 | +231 · 13/16 (+) | +231 · 13/16 (+) | +8.37 · 16/16 (+) | −0.01 · 15/16 (5+, 10−) |
| Transformer | 0.01 | 40 / 40 | 120 → 150 | 84 → 107 | +30 · 0/4 | +23 · 1/4 (−) | +0.08 · 1/4 (+) | −0.05 · 1/4 (−) |
| Transformer | 0.05 | 40 / 40 | 114 → 105 | 109 → 97 | −9 · 0/4 | −12 · 0/4 | −0.04 · 0/4 | +0.08 · 3/4 (+) |
| Transformer | 0.10 | 40 / 40 | 130 → 149 | 128 → 148 | +19 · 2/4 (+) | +21 · 2/4 (+) | +0.61 · 3/4 (+) | +0.05 · 4/4 (+) |
| Transformer | 0.30 | 40 / 38 | 56 → 92 | 48 → 89 | +36 · 3/4 (+) | +41 · 3/4 (+) | +2.44 · 3/4 (+) | +0.03 · 1/4 (+) |
| Transformer | all | 160 / 158 | 105 → 124 | 92 → 111 | +19 · 5/16 (+) | +19 · 6/16 (5+, 1−) | +0.71 · 7/16 (+) | +0.02 · 9/16 (8+, 1−) |

</details>

<details><summary>Intervals per rate × churn cell</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI] | Δ \|bias\| [95% CI] | Δ RMSE [95% CI] | Δ Spearman [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM | 0.01 | 20% | 10 | 71 → 513 | 59 → 502 | **+443 [+28, +997]** | **+443 [+25, +1002]** | **+5.83 [+1.08, +11.80]** | **−0.17 [−0.21, −0.13]** |
| LSTM | 0.01 | 40% | 10 | 136 → 189 | 132 → 182 | +52 [−11, +130] | +50 [−15, +129] | **+1.09 [+0.03, +2.57]** | **−0.24 [−0.27, −0.21]** |
| LSTM | 0.01 | 60% | 10 | 291 → 514 | 290 → 514 | **+223 [+75, +407]** | **+224 [+76, +408]** | **+2.56 [+0.57, +4.84]** | **−0.21 [−0.26, −0.16]** |
| LSTM | 0.01 | 80% | 10 | 759 → 838 | 755 → 821 | +79 [−91, +258] | +66 [−123, +255] | **+0.67 [0, +1.69]** | **−0.10 [−0.15, −0.06]** |
| LSTM | 0.05 | 20% | 10 | 45 → 240 | 42 → 240 | **+195 [+141, +258]** | **+198 [+144, +261]** | **+11.90 [+9.40, +14.71]** | **+0.15 [0, +0.31]** |
| LSTM | 0.05 | 40% | 10 | 99 → 313 | 98 → 313 | **+214 [+156, +273]** | **+215 [+156, +274]** | **+8.13 [+5.71, +10.70]** | **+0.19 [+0.06, +0.33]** |
| LSTM | 0.05 | 60% | 10 | 158 → 462 | 157 → 462 | **+304 [+182, +465]** | **+304 [+183, +465]** | **+8.80 [+6.45, +10.98]** | **+0.32 [+0.28, +0.37]** |
| LSTM | 0.05 | 80% | 10 | 497 → 1315 | 497 → 1314 | **+818 [+298, +1310]** | **+817 [+296, +1310]** | **+10.26 [+5.55, +14.78]** | **+0.27 [+0.19, +0.34]** |
| LSTM | 0.10 | 20% | 10 | 33 → 301 | 31 → 301 | **+268 [+192, +356]** | **+269 [+194, +358]** | **+17.86 [+12.65, +23.48]** | **−0.07 [−0.08, −0.06]** |
| LSTM | 0.10 | 40% | 10 | 58 → 375 | 57 → 375 | **+317 [+243, +416]** | **+318 [+244, +417]** | **+14.92 [+10.18, +20.54]** | −0.07 [−0.12, 0] |
| LSTM | 0.10 | 60% | 10 | 69 → 307 | 61 → 301 | **+239 [+95, +376]** | **+240 [+88, +385]** | **+5.13 [+2.30, +8.54]** | **−0.08 [−0.13, −0.03]** |
| LSTM | 0.10 | 80% | 10 | 233 → 307 | 225 → 295 | +73 [−139, +311] | +70 [−149, +316] | **+2.07 [+0.21, +4.78]** | **+0.15 [+0.01, +0.29]** |
| LSTM | 0.30 | 20% | 10 | 19 → 268 | 17 → 268 | **+248 [+213, +281]** | **+250 [+215, +283]** | **+28.75 [+23.76, +33.03]** | **−0.12 [−0.13, −0.10]** |
| LSTM | 0.30 | 40% | 10 | 14 → 161 | 9 → 158 | **+147 [+66, +233]** | **+149 [+68, +236]** | **+12.30 [+5.46, +19.71]** | **−0.08 [−0.11, −0.05]** |
| LSTM | 0.30 | 60% | 10 | 18 → 64 | 9 → 57 | **+46 [+31, +61]** | **+48 [+29, +67]** | **+2.31 [+1.53, +3.16]** | **−0.04 [−0.06, −0.03]** |
| LSTM | 0.30 | 80% | 10 | 32 → 70 | 17 → 55 | **+38 [+6, +76]** | **+39 [+2, +81]** | **+1.37 [+0.21, +2.79]** | **−0.03 [−0.03, −0.02]** |
| Transformer | 0.01 | 20% | 10 | 58 → 49 | 37 → 19 | −9 [−20, +1] | **−18 [−37, −1]** | −0.01 [−0.03, +0.01] | **−0.10 [−0.15, −0.04]** |
| Transformer | 0.01 | 40% | 10 | 96 → 78 | 76 → 55 | −18 [−45, +4] | −21 [−51, +6] | −0.03 [−0.07, +0.01] | −0.02 [−0.08, +0.05] |
| Transformer | 0.01 | 60% | 10 | 129 → 127 | 102 → 79 | −2 [−35, +33] | −22 [−67, +29] | **+0.12 [+0.01, +0.24]** | −0.01 [−0.06, +0.05] |
| Transformer | 0.01 | 80% | 10 | 197 → 345 | 121 → 273 | +148 [−30, +424] | +152 [−48, +432] | +0.26 [−0.01, +0.74] | −0.07 [−0.16, +0.01] |
| Transformer | 0.05 | 20% | 10 | 75 → 91 | 72 → 90 | +16 [−16, +44] | +18 [−17, +49] | −0.01 [−0.38, +0.31] | **+0.21 [+0.12, +0.33]** |
| Transformer | 0.05 | 40% | 10 | 110 → 103 | 108 → 101 | −6 [−49, +37] | −8 [−53, +39] | +0.02 [−0.42, +0.55] | **+0.10 [0, +0.16]** |
| Transformer | 0.05 | 60% | 10 | 149 → 124 | 144 → 121 | −25 [−75, +23] | −22 [−76, +30] | −0.14 [−0.32, +0.06] | **+0.11 [+0.04, +0.18]** |
| Transformer | 0.05 | 80% | 10 | 122 → 100 | 111 → 77 | −22 [−53, +6] | −35 [−75, +5] | −0.03 [−0.11, +0.05] | −0.12 [−0.27, +0.02] |
| Transformer | 0.10 | 20% | 10 | 58 → 102 | 56 → 101 | **+43 [+13, +78]** | **+45 [+15, +80]** | **+0.90 [+0.11, +1.97]** | **+0.07 [+0.04, +0.09]** |
| Transformer | 0.10 | 40% | 10 | 94 → 91 | 91 → 89 | −3 [−36, +33] | −1 [−37, +38] | +0.20 [−0.52, +1.09] | **+0.06 [+0.01, +0.09]** |
| Transformer | 0.10 | 60% | 10 | 103 → 159 | 99 → 159 | **+56 [+1, +105]** | **+60 [+2, +110]** | **+0.85 [+0.25, +1.45]** | **+0.05 [+0.02, +0.07]** |
| Transformer | 0.10 | 80% | 10 | 265 → 245 | 265 → 245 | −20 [−73, +38] | −20 [−73, +39] | **+0.50 [+0.06, +1.01]** | **+0.04 [+0.01, +0.07]** |
| Transformer | 0.30 | 20% | 10 | 36 → 71 | 24 → 69 | **+35 [+9, +60]** | **+45 [+13, +75]** | **+3.83 [+0.93, +6.49]** | +0.05 [−0.01, +0.09] |
| Transformer | 0.30 | 40% | 10 | 36 → 86 | 29 → 81 | **+49 [+28, +72]** | **+53 [+26, +77]** | **+2.89 [+1.40, +4.50]** | +0.01 [−0.05, +0.07] |
| Transformer | 0.30 | 60% | 9 | 59 → 97 | 52 → 96 | **+38 [+9, +70]** | **+44 [+11, +79]** | **+1.70 [+1.04, +2.42]** | **+0.04 [0, +0.08]** |
| Transformer | 0.30 | 80% | 9 | 104 → 118 | 97 → 112 | +14 [−28, +52] | +15 [−35, +63] | +0.77 [−0.11, +1.80] | −0.01 [−0.03, +0.02] |

</details>

### 5. A k-means cluster label hurts

**Verdict: partly.** The label's effect depends on the model, the AR encoding and the rate.

- **LSTM `no_ar`:** no clear change in level at rate 0.01; one better cell at rate 0.05
  (churn 40%, −17 MAPE); worse at rate 0.10 with churn 60–80% (+64, +76) and in every cell
  at rate 0.30 (+9 to +110). It raises Spearman in every cell at rates 0.01–0.05 (+0.12 to
  +0.35) and lowers it in six cells at rates 0.10–0.30.
- **LSTM `ar_bounded`:** worse level in 10 cells, most from rate 0.05 up (+14 to +229 MAPE
  per supported cell), better in two (rate 0.01 with churn 80%, −146; rate 0.05 with churn
  20%, −9). Better ranking at rate 0.01 only (all four cells), worse in the other 12.
- **Transformer `no_ar` and `ar_bounded`:** the level changes clearly in only 2 and 4 of 16
  cells (all four `ar_bounded` cells worse, +17 to +77). Spearman rises on sparse panels
  (seven of the eight rate-0.01 cells, +0.07 to +0.18).
- **With `ar_unbounded`** the label helps the Transformer in 8 cells (all at rates
  0.05–0.30) and hurts it in 1. For the LSTM it helps in 5 cells at rates 0.05–0.30 but hurts
  in 2 at rate 0.01, by more than 1,000 MAPE points each, so its pooled grid mean even
  rises (+52). Where it helps, it helps by displacing the broken counters.

**LSTM `no_ar`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | +15 / +18 / **+0.12** | +3 / +4 / **+0.13** | −18 / −18 / **+0.15** | −30 / −30 / **+0.13** | *−7 / −7 / +0.13* |
| 0.05 | +2 / −1 / **+0.17** | **−17** / **−17** / **+0.26** | +15 / +15 / **+0.35** | +9 / +9 / **+0.35** | *+2 / +1 / +0.28* |
| 0.10 | +6 / +4 / **−0.09** | +17 / +17 / −0.07 | **+64** / **+71** / **−0.09** | **+76** / **+84** / +0.11 | *+40 / +44 / −0.04* |
| 0.30 | **+9** / **+10** / **−0.11** | **+39** / **+45** / **−0.07** | **+72** / **+81** / **−0.06** | **+110** / **+125** / **−0.03** | *+57 / +65 / −0.07* |
| all |  |  |  |  | *+23 / +26 / +0.08* |

**LSTM `ar_bounded`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | +8 / +12 / **+0.17** | +1 / +1 / **+0.20** | **+48** / +51 / **+0.12** | **−146** / −147 / **+0.14** | *−22 / −21 / +0.15* |
| 0.05 | **−9** / **−18** / **−0.05** | +9 / +9 / **−0.09** | **+43** / **+47** / **−0.09** | **+229** / **+260** / **−0.04** | *+68 / +75 / −0.07* |
| 0.10 | 0 / −2 / **−0.09** | **+19** / **+23** / **−0.06** | **+60** / **+71** / **−0.09** | **+89** / **+116** / **−0.02** | *+42 / +52 / −0.07* |
| 0.30 | **+14** / **+15** / **−0.07** | **+32** / **+38** / **−0.06** | **+52** / **+60** / **−0.05** | **+49** / **+64** / **−0.02** | *+37 / +44 / −0.05* |
| all |  |  |  |  | *+31 / +38 / −0.01* |

**LSTM `ar_unbounded`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | −264 / −268 / **+0.07** | **+1115** / **+1118** / **+0.11** | +390 / +386 / **+0.16** | **+1177** / **+1172** / **+0.10** | *+605 / +602 / +0.11* |
| 0.05 | **−51** / **−52** / +0.02 | −84 / −92 / −0.01 | −103 / −105 / +0.01 | −395 / −400 / +0.05 | *−158 / −162 / +0.02* |
| 0.10 | **−156** / **−159** / −0.02 | **−201** / **−208** / −0.01 | −165 / −168 / +0.03 | −118 / −124 / +0.02 | *−160 / −165 / +0.01* |
| 0.30 | **−187** / **−192** / +0.02 | **−99** / **−108** / +0.01 | −10 / −18 / 0 | −23 / −30 / 0 | *−80 / −87 / +0.01* |
| all |  |  |  |  | *+52 / +47 / +0.04* |

**Transformer `no_ar`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | −3 / −2 / **+0.07** | +52 / +57 / **+0.11** | −9 / −15 / **+0.10** | −2 / −12 / **+0.08** | *+10 / +7 / +0.09* |
| 0.05 | −7 / −9 / **+0.18** | −3 / −4 / **+0.06** | −14 / −13 / **+0.11** | +59 / +56 / +0.04 | *+9 / +7 / +0.10* |
| 0.10 | −1 / −2 / +0.03 | +1 / +3 / **+0.05** | +6 / +8 / **+0.04** | **−62** / **−63** / +0.02 | *−14 / −14 / +0.03* |
| 0.30 | −6 / −3 / **+0.05** | **+19** / **+24** / **+0.03** | +8 / +15 / **+0.02** | +23 / +31 / **+0.01** | *+11 / +17 / +0.03* |
| all |  |  |  |  | *+4 / +4 / +0.06* |

**Transformer `ar_bounded`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | +10 / +15 / **+0.13** | +27 / +22 / **+0.18** | +5 / +7 / **+0.12** | +20 / +23 / +0.03 | *+16 / +16 / +0.11* |
| 0.05 | +11 / +14 / **+0.05** | −9 / −7 / **−0.03** | **+77** / **+83** / **−0.05** | −8 / −11 / **−0.03** | *+18 / +20 / −0.01* |
| 0.10 | −7 / −7 / **−0.03** | **+17** / **+21** / +0.01 | −6 / −7 / **−0.04** | **+75** / **+86** / **−0.05** | *+20 / +23 / −0.03* |
| 0.30 | −5 / −2 / **−0.05** | +5 / +10 / +0.04 | +17 / +21 / −0.02 | **+36** / **+41** / −0.01 | *+13 / +17 / −0.01* |
| all |  |  |  |  | *+17 / +19 / +0.02* |

**Transformer `ar_unbounded`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **+16** / **+27** / **+0.08** | +5 / +5 / 0 | −7 / +4 / +0.04 | −130 / −128 / +0.09 | *−29 / −23 / +0.05* |
| 0.05 | −22 / −25 / **−0.05** | **−25** / **−33** / −0.05 | −21 / −24 / −0.05 | −3 / −4 / **+0.19** | *−18 / −22 / +0.01* |
| 0.10 | **−55** / **−62** / −0.02 | −14 / −17 / −0.01 | **−83** / **−89** / −0.01 | **−94** / **−103** / −0.01 | *−62 / −68 / −0.01* |
| 0.30 | **−34** / **−40** / **+0.03** | **−45** / **−54** / +0.05 | **−35** / **−45** / +0.01 | **−60** / **−67** / **+0.03** | *−42 / −50 / +0.03* |
| all |  |  |  |  | *−38 / −41 / +0.02* |

<details><summary>By rate, churn pooled (descriptive)</summary>

| Comparison | Rate | n (A / B) | MAPE A → B | \|bias\| A → B | Δ MAPE (pooled mean) · cells supported | Δ \|bias\| (pooled mean) · cells supported | Δ RMSE (pooled mean) · cells supported | Δ Spearman (pooled mean) · cells supported |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM `no_ar` | 0.01 | 40 / 40 | 314 → 307 | 309 → 302 | −7 · 0/4 | −7 · 0/4 | +0.01 · 0/4 | +0.13 · 4/4 (+) |
| LSTM `no_ar` | 0.05 | 40 / 40 | 200 → 202 | 199 → 200 | +2 · 1/4 (−) | +1 · 1/4 (−) | −0.02 · 1/4 (−) | +0.28 · 4/4 (+) |
| LSTM `no_ar` | 0.10 | 40 / 40 | 98 → 139 | 93 → 138 | +40 · 2/4 (+) | +44 · 2/4 (+) | +0.42 · 3/4 (+) | −0.04 · 2/4 (−) |
| LSTM `no_ar` | 0.30 | 40 / 40 | 21 → 78 | 13 → 78 | +57 · 4/4 (+) | +65 · 4/4 (+) | +2.15 · 4/4 (+) | −0.07 · 4/4 (−) |
| LSTM `no_ar` | all | 160 / 160 | 158 → 181 | 154 → 179 | +23 · 7/16 (6+, 1−) | +26 · 7/16 (6+, 1−) | +0.64 · 8/16 (7+, 1−) | +0.08 · 14/16 (8+, 6−) |
| LSTM `ar_bounded` | 0.01 | 40 / 40 | 243 → 221 | 232 → 211 | −22 · 2/4 (1+, 1−) | −21 · 0/4 | +0.01 · 1/4 (+) | +0.15 · 4/4 (+) |
| LSTM `ar_bounded` | 0.05 | 40 / 40 | 58 → 126 | 46 → 121 | +68 · 3/4 (2+, 1−) | +75 · 3/4 (2+, 1−) | +0.25 · 4/4 (3+, 1−) | −0.07 · 4/4 (−) |
| LSTM `ar_bounded` | 0.10 | 40 / 40 | 36 → 78 | 24 → 76 | +42 · 3/4 (+) | +52 · 3/4 (+) | +0.44 · 3/4 (+) | −0.07 · 4/4 (−) |
| LSTM `ar_bounded` | 0.30 | 40 / 40 | 20 → 56 | 11 → 56 | +37 · 4/4 (+) | +44 · 4/4 (+) | +1.67 · 4/4 (+) | −0.05 · 4/4 (−) |
| LSTM `ar_bounded` | all | 160 / 160 | 89 → 120 | 78 → 116 | +31 · 12/16 (10+, 2−) | +38 · 10/16 (9+, 1−) | +0.59 · 12/16 (11+, 1−) | −0.01 · 16/16 (4+, 12−) |
| LSTM `ar_unbounded` | 0.01 | 40 / 40 | 513 → 1118 | 505 → 1107 | +605 · 2/4 (+) | +602 · 2/4 (+) | +3.67 · 2/4 (+) | +0.11 · 4/4 (+) |
| LSTM `ar_unbounded` | 0.05 | 40 / 40 | 582 → 424 | 582 → 420 | −158 · 1/4 (−) | −162 · 1/4 (−) | −1.70 · 1/4 (−) | +0.02 · 0/4 |
| LSTM `ar_unbounded` | 0.10 | 40 / 40 | 322 → 162 | 318 → 153 | −160 · 2/4 (−) | −165 · 2/4 (−) | −4.49 · 2/4 (−) | +0.01 · 0/4 |
| LSTM `ar_unbounded` | 0.30 | 40 / 40 | 141 → 61 | 135 → 48 | −80 · 2/4 (−) | −87 · 2/4 (−) | −7.29 · 2/4 (−) | +0.01 · 0/4 |
| LSTM `ar_unbounded` | all | 160 / 160 | 390 → 441 | 385 → 432 | +52 · 7/16 (2+, 5−) | +47 · 7/16 (2+, 5−) | −2.45 · 7/16 (2+, 5−) | +0.04 · 4/16 (+) |
| Transformer `no_ar` | 0.01 | 40 / 40 | 120 → 130 | 84 → 91 | +10 · 0/4 | +7 · 0/4 | +0.06 · 0/4 | +0.09 · 4/4 (+) |
| Transformer `no_ar` | 0.05 | 40 / 40 | 114 → 122 | 109 → 116 | +9 · 0/4 | +7 · 0/4 | +0.01 · 1/4 (−) | +0.10 · 3/4 (+) |
| Transformer `no_ar` | 0.10 | 40 / 40 | 130 → 116 | 128 → 114 | −14 · 1/4 (−) | −14 · 1/4 (−) | +0.16 · 1/4 (+) | +0.03 · 2/4 (+) |
| Transformer `no_ar` | 0.30 | 40 / 40 | 56 → 67 | 48 → 64 | +11 · 1/4 (+) | +17 · 1/4 (+) | +0.63 · 2/4 (+) | +0.03 · 4/4 (+) |
| Transformer `no_ar` | all | 160 / 160 | 105 → 109 | 92 → 96 | +4 · 2/16 (1+, 1−) | +4 · 2/16 (1+, 1−) | +0.22 · 4/16 (3+, 1−) | +0.06 · 13/16 (+) |
| Transformer `ar_bounded` | 0.01 | 40 / 40 | 103 → 119 | 63 → 79 | +16 · 0/4 | +16 · 0/4 | +0.06 · 1/4 (+) | +0.11 · 3/4 (+) |
| Transformer `ar_bounded` | 0.05 | 40 / 40 | 78 → 96 | 68 → 88 | +18 · 1/4 (+) | +20 · 1/4 (+) | +0.20 · 2/4 (+) | −0.01 · 4/4 (1+, 3−) |
| Transformer `ar_bounded` | 0.10 | 40 / 40 | 77 → 97 | 70 → 93 | +20 · 2/4 (+) | +23 · 2/4 (+) | +0.39 · 2/4 (+) | −0.03 · 3/4 (−) |
| Transformer `ar_bounded` | 0.30 | 40 / 40 | 50 → 64 | 42 → 60 | +13 · 1/4 (+) | +17 · 1/4 (+) | +0.98 · 3/4 (+) | −0.01 · 1/4 (−) |
| Transformer `ar_bounded` | all | 160 / 160 | 77 → 94 | 61 → 80 | +17 · 4/16 (+) | +19 · 4/16 (+) | +0.41 · 8/16 (+) | +0.02 · 11/16 (4+, 7−) |
| Transformer `ar_unbounded` | 0.01 | 40 / 40 | 150 → 121 | 107 → 83 | −29 · 1/4 (+) | −23 · 1/4 (+) | −0.05 · 1/4 (+) | +0.05 · 1/4 (+) |
| Transformer `ar_unbounded` | 0.05 | 40 / 40 | 105 → 87 | 97 → 76 | −18 · 1/4 (−) | −22 · 1/4 (−) | +0.01 · 0/4 | +0.01 · 2/4 (1+, 1−) |
| Transformer `ar_unbounded` | 0.10 | 40 / 40 | 149 → 88 | 148 → 81 | −62 · 3/4 (−) | −68 · 3/4 (−) | −0.63 · 3/4 (−) | −0.01 · 0/4 |
| Transformer `ar_unbounded` | 0.30 | 38 / 40 | 92 → 50 | 89 → 39 | −42 · 4/4 (−) | −50 · 4/4 (−) | −1.92 · 2/4 (−) | +0.03 · 2/4 (+) |
| Transformer `ar_unbounded` | all | 158 / 160 | 124 → 86 | 111 → 70 | −38 · 9/16 (1+, 8−) | −41 · 9/16 (1+, 8−) | −0.58 · 6/16 (1+, 5−) | +0.02 · 5/16 (4+, 1−) |

</details>

<details><summary>Intervals per rate × churn cell</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI] | Δ \|bias\| [95% CI] | Δ RMSE [95% CI] | Δ Spearman [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM `no_ar` | 0.01 | 20% | 10 | 71 → 86 | 59 → 77 | +15 [0, +31] | +18 [−3, +37] | +0.04 [−0.01, +0.10] | **+0.12 [+0.09, +0.16]** |
| LSTM `no_ar` | 0.01 | 40% | 10 | 136 → 140 | 132 → 136 | +3 [−8, +18] | +4 [−9, +21] | +0.01 [−0.02, +0.04] | **+0.13 [+0.08, +0.16]** |
| LSTM `no_ar` | 0.01 | 60% | 10 | 291 → 273 | 290 → 271 | −18 [−62, +26] | −18 [−63, +27] | −0.01 [−0.09, +0.06] | **+0.15 [+0.10, +0.19]** |
| LSTM `no_ar` | 0.01 | 80% | 10 | 759 → 729 | 755 → 725 | −30 [−141, +74] | −30 [−142, +77] | 0 [−0.11, +0.10] | **+0.13 [+0.09, +0.16]** |
| LSTM `no_ar` | 0.05 | 20% | 10 | 45 → 47 | 42 → 40 | +2 [−9, +13] | −1 [−14, +12] | −0.07 [−0.24, +0.12] | **+0.17 [+0.04, +0.31]** |
| LSTM `no_ar` | 0.05 | 40% | 10 | 99 → 82 | 98 → 81 | **−17 [−26, −6]** | **−17 [−27, −6]** | **−0.16 [−0.24, −0.06]** | **+0.26 [+0.12, +0.40]** |
| LSTM `no_ar` | 0.05 | 60% | 10 | 158 → 172 | 157 → 172 | +15 [−7, +36] | +15 [−7, +36] | +0.04 [−0.05, +0.14] | **+0.35 [+0.30, +0.39]** |
| LSTM `no_ar` | 0.05 | 80% | 10 | 497 → 506 | 497 → 506 | +9 [−56, +68] | +9 [−56, +68] | +0.11 [−0.08, +0.28] | **+0.35 [+0.30, +0.39]** |
| LSTM `no_ar` | 0.10 | 20% | 10 | 33 → 39 | 31 → 35 | +6 [−1, +13] | +4 [−4, +12] | **+0.28 [+0.13, +0.44]** | **−0.09 [−0.12, −0.06]** |
| LSTM `no_ar` | 0.10 | 40% | 10 | 58 → 74 | 57 → 74 | +17 [−18, +41] | +17 [−18, +43] | +0.31 [−0.23, +0.69] | −0.07 [−0.14, +0.02] |
| LSTM `no_ar` | 0.10 | 60% | 10 | 69 → 132 | 61 → 132 | **+64 [+35, +91]** | **+71 [+40, +100]** | **+0.69 [+0.45, +0.94]** | **−0.09 [−0.14, −0.05]** |
| LSTM `no_ar` | 0.10 | 80% | 10 | 233 → 309 | 225 → 309 | **+76 [+17, +137]** | **+84 [+21, +147]** | **+0.41 [+0.14, +0.68]** | +0.11 [−0.01, +0.22] |
| LSTM `no_ar` | 0.30 | 20% | 10 | 19 → 28 | 17 → 27 | **+9 [+5, +13]** | **+10 [+5, +16]** | **+1.39 [+1.05, +1.73]** | **−0.11 [−0.13, −0.09]** |
| LSTM `no_ar` | 0.30 | 40% | 10 | 14 → 53 | 9 → 53 | **+39 [+34, +44]** | **+45 [+40, +50]** | **+2.50 [+2.09, +2.89]** | **−0.07 [−0.08, −0.06]** |
| LSTM `no_ar` | 0.30 | 60% | 10 | 18 → 90 | 9 → 90 | **+72 [+64, +81]** | **+81 [+75, +88]** | **+2.62 [+2.13, +3.06]** | **−0.06 [−0.08, −0.05]** |
| LSTM `no_ar` | 0.30 | 80% | 10 | 32 → 142 | 17 → 142 | **+110 [+82, +138]** | **+125 [+95, +154]** | **+2.08 [+1.56, +2.60]** | **−0.03 [−0.04, −0.02]** |
| LSTM `ar_bounded` | 0.01 | 20% | 10 | 65 → 72 | 47 → 58 | +8 [−9, +24] | +12 [−13, +38] | +0.02 [−0.03, +0.06] | **+0.17 [+0.09, +0.24]** |
| LSTM `ar_bounded` | 0.01 | 40% | 10 | 121 → 121 | 114 → 116 | +1 [−18, +19] | +1 [−20, +22] | +0.01 [−0.02, +0.05] | **+0.20 [+0.16, +0.24]** |
| LSTM `ar_bounded` | 0.01 | 60% | 10 | 196 → 245 | 190 → 241 | **+48 [+1, +90]** | +51 [0, +96] | **+0.08 [+0.01, +0.13]** | **+0.12 [+0.04, +0.19]** |
| LSTM `ar_bounded` | 0.01 | 80% | 10 | 591 → 445 | 576 → 429 | **−146 [−315, −5]** | −147 [−322, +1] | −0.08 [−0.18, +0.01] | **+0.14 [+0.07, +0.19]** |
| LSTM `ar_bounded` | 0.05 | 20% | 10 | 48 → 39 | 47 → 29 | **−9 [−15, −4]** | **−18 [−24, −11]** | **−0.09 [−0.16, 0]** | **−0.05 [−0.08, −0.03]** |
| LSTM `ar_bounded` | 0.05 | 40% | 10 | 56 → 64 | 52 → 61 | +9 [−2, +21] | +9 [−4, +22] | **+0.12 [+0.03, +0.24]** | **−0.09 [−0.12, −0.06]** |
| LSTM `ar_bounded` | 0.05 | 60% | 10 | 67 → 110 | 61 → 108 | **+43 [+27, +64]** | **+47 [+30, +69]** | **+0.37 [+0.28, +0.46]** | **−0.09 [−0.13, −0.05]** |
| LSTM `ar_bounded` | 0.05 | 80% | 10 | 60 → 290 | 27 → 287 | **+229 [+138, +334]** | **+260 [+170, +362]** | **+0.61 [+0.37, +0.86]** | **−0.04 [−0.07, −0.01]** |
| LSTM `ar_bounded` | 0.10 | 20% | 10 | 32 → 33 | 30 → 28 | 0 [−8, +8] | −2 [−12, +8] | +0.14 [−0.03, +0.29] | **−0.09 [−0.12, −0.06]** |
| LSTM `ar_bounded` | 0.10 | 40% | 10 | 34 → 53 | 28 → 52 | **+19 [+10, +27]** | **+23 [+12, +34]** | **+0.37 [+0.22, +0.52]** | **−0.06 [−0.08, −0.04]** |
| LSTM `ar_bounded` | 0.10 | 60% | 10 | 33 → 93 | 21 → 92 | **+60 [+46, +76]** | **+71 [+57, +85]** | **+0.68 [+0.54, +0.82]** | **−0.09 [−0.13, −0.06]** |
| LSTM `ar_bounded` | 0.10 | 80% | 10 | 45 → 134 | 15 → 131 | **+89 [+59, +129]** | **+116 [+84, +157]** | **+0.59 [+0.44, +0.76]** | **−0.02 [−0.04, −0.01]** |
| LSTM `ar_bounded` | 0.30 | 20% | 10 | 12 → 26 | 10 → 25 | **+14 [+10, +17]** | **+15 [+11, +18]** | **+1.46 [+1.14, +1.77]** | **−0.07 [−0.08, −0.06]** |
| LSTM `ar_bounded` | 0.30 | 40% | 10 | 13 → 45 | 6 → 44 | **+32 [+26, +37]** | **+38 [+32, +44]** | **+2.02 [+1.68, +2.34]** | **−0.06 [−0.07, −0.05]** |
| LSTM `ar_bounded` | 0.30 | 60% | 10 | 19 → 71 | 10 → 71 | **+52 [+44, +60]** | **+60 [+51, +70]** | **+2.13 [+1.83, +2.42]** | **−0.05 [−0.05, −0.04]** |
| LSTM `ar_bounded` | 0.30 | 80% | 10 | 35 → 84 | 19 → 83 | **+49 [+33, +65]** | **+64 [+47, +85]** | **+1.06 [+0.84, +1.28]** | **−0.02 [−0.03, −0.01]** |
| LSTM `ar_unbounded` | 0.01 | 20% | 10 | 513 → 250 | 502 → 235 | −264 [−734, +100] | −268 [−742, +102] | −1.61 [−6.40, +2.58] | **+0.07 [+0.04, +0.12]** |
| LSTM `ar_unbounded` | 0.01 | 40% | 10 | 189 → 1304 | 182 → 1300 | **+1115 [+236, +2224]** | **+1118 [+238, +2226]** | **+9.47 [+3.66, +16.14]** | **+0.11 [+0.05, +0.18]** |
| LSTM `ar_unbounded` | 0.01 | 60% | 10 | 514 → 904 | 514 → 900 | +390 [−93, +926] | +386 [−97, +924] | +2.09 [−1.79, +6.49] | **+0.16 [+0.08, +0.24]** |
| LSTM `ar_unbounded` | 0.01 | 80% | 10 | 838 → 2015 | 821 → 1993 | **+1177 [+69, +2545]** | **+1172 [+67, +2535]** | **+4.73 [+0.28, +9.70]** | **+0.10 [+0.01, +0.18]** |
| LSTM `ar_unbounded` | 0.05 | 20% | 10 | 240 → 189 | 240 → 188 | **−51 [−107, −3]** | **−52 [−108, −3]** | **−2.86 [−5.16, −0.55]** | +0.02 [−0.01, +0.04] |
| LSTM `ar_unbounded` | 0.05 | 40% | 10 | 313 → 229 | 313 → 221 | −84 [−194, +39] | −92 [−207, +35] | −0.97 [−5.49, +3.90] | −0.01 [−0.07, +0.05] |
| LSTM `ar_unbounded` | 0.05 | 60% | 10 | 462 → 359 | 462 → 357 | −103 [−264, +96] | −105 [−266, +95] | −2.25 [−6.36, +2.05] | +0.01 [−0.09, +0.10] |
| LSTM `ar_unbounded` | 0.05 | 80% | 10 | 1315 → 919 | 1314 → 914 | −395 [−927, +163] | −400 [−933, +161] | −0.70 [−4.69, +3.70] | +0.05 [−0.03, +0.13] |
| LSTM `ar_unbounded` | 0.10 | 20% | 10 | 301 → 145 | 301 → 141 | **−156 [−267, −77]** | **−159 [−275, −78]** | **−8.06 [−15.48, −1.80]** | −0.02 [−0.07, +0.02] |
| LSTM `ar_unbounded` | 0.10 | 40% | 10 | 375 → 174 | 375 → 167 | **−201 [−295, −110]** | **−208 [−304, −114]** | **−8.31 [−13.49, −3.09]** | −0.01 [−0.06, +0.04] |
| LSTM `ar_unbounded` | 0.10 | 60% | 10 | 307 → 142 | 301 → 133 | −165 [−350, +40] | −168 [−361, +45] | −1.49 [−6.55, +3.35] | +0.03 [−0.03, +0.10] |
| LSTM `ar_unbounded` | 0.10 | 80% | 10 | 307 → 189 | 295 → 171 | −118 [−358, +131] | −124 [−373, +132] | −0.11 [−2.36, +2.38] | +0.02 [−0.02, +0.07] |
| LSTM `ar_unbounded` | 0.30 | 20% | 10 | 268 → 80 | 268 → 76 | **−187 [−229, −149]** | **−192 [−236, −152]** | **−20.07 [−26.58, −12.98]** | +0.02 [0, +0.04] |
| LSTM `ar_unbounded` | 0.30 | 40% | 10 | 161 → 62 | 158 → 50 | **−99 [−181, −20]** | **−108 [−193, −27]** | **−7.79 [−14.09, −1.85]** | +0.01 [−0.02, +0.05] |
| LSTM `ar_unbounded` | 0.30 | 60% | 10 | 64 → 54 | 57 → 39 | −10 [−28, +5] | −18 [−41, +3] | −0.55 [−1.33, +0.27] | 0 [−0.02, +0.01] |
| LSTM `ar_unbounded` | 0.30 | 80% | 10 | 70 → 47 | 55 → 26 | −23 [−61, +7] | −30 [−73, +5] | −0.74 [−2.11, +0.33] | 0 [0, +0.01] |
| Transformer `no_ar` | 0.01 | 20% | 10 | 58 → 55 | 37 → 35 | −3 [−14, +9] | −2 [−23, +20] | +0.02 [−0.03, +0.09] | **+0.07 [+0.03, +0.12]** |
| Transformer `no_ar` | 0.01 | 40% | 10 | 96 → 148 | 76 → 134 | +52 [−20, +133] | +57 [−27, +149] | +0.17 [−0.01, +0.37] | **+0.11 [+0.04, +0.17]** |
| Transformer `no_ar` | 0.01 | 60% | 10 | 129 → 120 | 102 → 87 | −9 [−49, +29] | −15 [−62, +32] | +0.02 [−0.03, +0.07] | **+0.10 [+0.04, +0.15]** |
| Transformer `no_ar` | 0.01 | 80% | 10 | 197 → 195 | 121 → 109 | −2 [−65, +77] | −12 [−111, +107] | +0.03 [−0.04, +0.13] | **+0.08 [+0.05, +0.13]** |
| Transformer `no_ar` | 0.05 | 20% | 10 | 75 → 67 | 72 → 62 | −7 [−26, +11] | −9 [−34, +15] | **−0.26 [−0.48, −0.04]** | **+0.18 [+0.07, +0.30]** |
| Transformer `no_ar` | 0.05 | 40% | 10 | 110 → 107 | 108 → 105 | −3 [−38, +38] | −4 [−41, +40] | +0.09 [−0.14, +0.35] | **+0.06 [+0.02, +0.10]** |
| Transformer `no_ar` | 0.05 | 60% | 10 | 149 → 134 | 144 → 131 | −14 [−77, +54] | −13 [−81, +59] | +0.04 [−0.21, +0.30] | **+0.11 [+0.07, +0.15]** |
| Transformer `no_ar` | 0.05 | 80% | 10 | 122 → 181 | 111 → 167 | +59 [−33, +156] | +56 [−48, +165] | +0.16 [−0.04, +0.37] | +0.04 [−0.02, +0.09] |
| Transformer `no_ar` | 0.10 | 20% | 10 | 58 → 57 | 56 → 54 | −1 [−21, +21] | −2 [−27, +25] | −0.12 [−0.58, +0.35] | +0.03 [−0.02, +0.07] |
| Transformer `no_ar` | 0.10 | 40% | 10 | 94 → 95 | 91 → 93 | +1 [−14, +21] | +3 [−13, +24] | **+0.30 [+0.08, +0.53]** | **+0.05 [+0.03, +0.06]** |
| Transformer `no_ar` | 0.10 | 60% | 10 | 103 → 109 | 99 → 106 | +6 [−31, +39] | +8 [−30, +42] | +0.29 [−0.07, +0.62] | **+0.04 [+0.01, +0.07]** |
| Transformer `no_ar` | 0.10 | 80% | 10 | 265 → 203 | 265 → 201 | **−62 [−115, −8]** | **−63 [−117, −8]** | +0.19 [−0.08, +0.47] | +0.02 [−0.01, +0.05] |
| Transformer `no_ar` | 0.30 | 20% | 10 | 36 → 30 | 24 → 21 | −6 [−21, +6] | −3 [−24, +16] | −0.10 [−1.35, +0.89] | **+0.05 [+0.03, +0.06]** |
| Transformer `no_ar` | 0.30 | 40% | 10 | 36 → 55 | 29 → 53 | **+19 [+9, +28]** | **+24 [+11, +37]** | **+1.18 [+0.49, +1.86]** | **+0.03 [+0.01, +0.04]** |
| Transformer `no_ar` | 0.30 | 60% | 10 | 57 → 65 | 50 → 65 | +8 [−15, +30] | +15 [−11, +39] | +0.55 [−0.10, +1.16] | **+0.02 [+0.01, +0.03]** |
| Transformer `no_ar` | 0.30 | 80% | 10 | 96 → 120 | 88 → 119 | +23 [−7, +51] | +31 [−1, +59] | **+0.91 [+0.50, +1.39]** | **+0.01 [0, +0.03]** |
| Transformer `ar_bounded` | 0.01 | 20% | 10 | 62 → 73 | 44 → 59 | +10 [−17, +45] | +15 [−22, +56] | +0.05 [−0.03, +0.15] | **+0.13 [+0.09, +0.17]** |
| Transformer `ar_bounded` | 0.01 | 40% | 10 | 75 → 102 | 58 → 80 | +27 [−14, +77] | +22 [−26, +80] | +0.08 [−0.01, +0.21] | **+0.18 [+0.15, +0.21]** |
| Transformer `ar_bounded` | 0.01 | 60% | 10 | 98 → 103 | 60 → 67 | +5 [−25, +40] | +7 [−38, +57] | +0.03 [−0.02, +0.10] | **+0.12 [+0.06, +0.17]** |
| Transformer `ar_bounded` | 0.01 | 80% | 10 | 176 → 196 | 90 → 112 | +20 [−15, +62] | +23 [−27, +75] | **+0.07 [+0.03, +0.12]** | +0.03 [−0.08, +0.13] |
| Transformer `ar_bounded` | 0.05 | 20% | 10 | 68 → 79 | 63 → 77 | +11 [−4, +23] | +14 [−3, +31] | +0.08 [−0.08, +0.23] | **+0.05 [+0.02, +0.09]** |
| Transformer `ar_bounded` | 0.05 | 40% | 10 | 65 → 56 | 54 → 48 | −9 [−39, +17] | −7 [−45, +25] | +0.09 [−0.10, +0.27] | **−0.03 [−0.06, −0.01]** |
| Transformer `ar_bounded` | 0.05 | 60% | 10 | 77 → 155 | 69 → 152 | **+77 [+30, +120]** | **+83 [+30, +130]** | **+0.55 [+0.34, +0.73]** | **−0.05 [−0.08, −0.02]** |
| Transformer `ar_bounded` | 0.05 | 80% | 10 | 102 → 94 | 84 → 73 | −8 [−68, +50] | −11 [−90, +64] | **+0.07 [0, +0.14]** | **−0.03 [−0.05, −0.01]** |
| Transformer `ar_bounded` | 0.10 | 20% | 10 | 52 → 45 | 45 → 38 | −7 [−28, +14] | −7 [−34, +19] | −0.03 [−0.39, +0.33] | **−0.03 [−0.08, 0]** |
| Transformer `ar_bounded` | 0.10 | 40% | 10 | 41 → 58 | 33 → 54 | **+17 [0, +36]** | **+21 [+1, +44]** | **+0.55 [+0.20, +0.92]** | +0.01 [−0.05, +0.12] |
| Transformer `ar_bounded` | 0.10 | 60% | 10 | 115 → 108 | 113 → 106 | −6 [−61, +46] | −7 [−62, +47] | +0.31 [−0.30, +0.86] | **−0.04 [−0.06, −0.01]** |
| Transformer `ar_bounded` | 0.10 | 80% | 10 | 102 → 177 | 90 → 175 | **+75 [+34, +120]** | **+86 [+40, +134]** | **+0.74 [+0.50, +0.99]** | **−0.05 [−0.09, −0.02]** |
| Transformer `ar_bounded` | 0.30 | 20% | 10 | 37 → 32 | 27 → 25 | −5 [−18, +7] | −2 [−22, +17] | +0.03 [−1.26, +1.16] | **−0.05 [−0.11, −0.01]** |
| Transformer `ar_bounded` | 0.30 | 40% | 10 | 42 → 47 | 35 → 45 | +5 [−11, +21] | +10 [−11, +31] | **+1.02 [+0.10, +1.87]** | +0.04 [−0.04, +0.14] |
| Transformer `ar_bounded` | 0.30 | 60% | 10 | 67 → 85 | 62 → 83 | +17 [−21, +53] | +21 [−20, +60] | **+1.52 [+0.35, +2.83]** | −0.02 [−0.05, 0] |
| Transformer `ar_bounded` | 0.30 | 80% | 10 | 54 → 91 | 46 → 87 | **+36 [+7, +68]** | **+41 [+9, +74]** | **+1.36 [+0.77, +2.02]** | −0.01 [−0.04, +0.01] |
| Transformer `ar_unbounded` | 0.01 | 20% | 10 | 49 → 65 | 19 → 46 | **+16 [+5, +26]** | **+27 [+6, +48]** | **+0.04 [+0.01, +0.07]** | **+0.08 [+0.01, +0.15]** |
| Transformer `ar_unbounded` | 0.01 | 40% | 10 | 78 → 83 | 55 → 60 | +5 [−25, +36] | +5 [−43, +52] | +0.03 [−0.02, +0.09] | 0 [−0.06, +0.05] |
| Transformer `ar_unbounded` | 0.01 | 60% | 10 | 127 → 120 | 79 → 83 | −7 [−60, +55] | +4 [−58, +75] | −0.06 [−0.26, +0.12] | +0.04 [−0.04, +0.11] |
| Transformer `ar_unbounded` | 0.01 | 80% | 10 | 345 → 216 | 273 → 145 | −130 [−352, +14] | −128 [−347, +20] | −0.20 [−0.61, +0.04] | +0.09 [−0.02, +0.20] |
| Transformer `ar_unbounded` | 0.05 | 20% | 10 | 91 → 69 | 90 → 64 | −22 [−43, +2] | −25 [−51, +2] | −0.03 [−0.34, +0.27] | **−0.05 [−0.11, −0.01]** |
| Transformer `ar_unbounded` | 0.05 | 40% | 10 | 103 → 78 | 101 → 68 | **−25 [−46, −3]** | **−33 [−56, −9]** | −0.16 [−0.59, +0.16] | −0.05 [−0.16, +0.06] |
| Transformer `ar_unbounded` | 0.05 | 60% | 10 | 124 → 103 | 121 → 98 | −21 [−64, +32] | −24 [−71, +32] | +0.17 [−0.14, +0.61] | −0.05 [−0.14, +0.02] |
| Transformer `ar_unbounded` | 0.05 | 80% | 10 | 100 → 97 | 77 → 72 | −3 [−67, +71] | −4 [−77, +76] | +0.06 [−0.18, +0.37] | **+0.19 [+0.05, +0.33]** |
| Transformer `ar_unbounded` | 0.10 | 20% | 10 | 102 → 47 | 101 → 39 | **−55 [−85, −25]** | **−62 [−95, −27]** | **−1.14 [−2.14, −0.25]** | −0.02 [−0.09, +0.03] |
| Transformer `ar_unbounded` | 0.10 | 40% | 10 | 91 → 77 | 89 → 72 | −14 [−49, +25] | −17 [−54, +25] | −0.15 [−1.04, +0.68] | −0.01 [−0.08, +0.05] |
| Transformer `ar_unbounded` | 0.10 | 60% | 10 | 159 → 76 | 159 → 70 | **−83 [−122, −50]** | **−89 [−128, −56]** | **−0.74 [−1.20, −0.28]** | −0.01 [−0.07, +0.03] |
| Transformer `ar_unbounded` | 0.10 | 80% | 10 | 245 → 150 | 245 → 142 | **−94 [−133, −57]** | **−103 [−146, −61]** | **−0.49 [−0.95, −0.12]** | −0.01 [−0.06, +0.02] |
| Transformer `ar_unbounded` | 0.30 | 20% | 10 | 71 → 37 | 69 → 30 | **−34 [−58, −12]** | **−40 [−65, −15]** | **−3.29 [−5.75, −1.06]** | **+0.03 [0, +0.08]** |
| Transformer `ar_unbounded` | 0.30 | 40% | 10 | 86 → 41 | 81 → 28 | **−45 [−64, −25]** | **−54 [−78, −30]** | **−2.38 [−3.70, −1.13]** | +0.05 [−0.01, +0.11] |
| Transformer `ar_unbounded` | 0.30 | 60% | 9 | 97 → 62 | 96 → 51 | **−35 [−66, −9]** | **−45 [−83, −13]** | −0.95 [−1.91, 0] | +0.01 [−0.01, +0.03] |
| Transformer `ar_unbounded` | 0.30 | 80% | 9 | 118 → 58 | 112 → 45 | **−60 [−114, −15]** | **−67 [−124, −16]** | −0.59 [−2.07, +0.58] | **+0.03 [+0.01, +0.05]** |

</details>

### 6. One architecture is better overall

**Verdict: not supported.** Neither architecture is better across the grid: in every arm
the cells split, with the Transformer clearly better in some and the LSTM clearly better in
others. The pooled grid means (Transformer minus LSTM: −53 MAPE with `no_ar`, −12 with
flags) are dominated by the LSTM's blow-ups at rate 0.01 and describe no typical cell.

- **Rate 0.01:** the Transformer is better on MAPE in three of four cells with `no_ar` and
  with flags (−13 to −562 points) and in all four with the counters, and ranks customers
  better in seven of the eight `no_ar` and `ar_bounded` cells.
- **Rates 0.10–0.30:** the LSTM is better on MAPE in 7 of 8 cells with `no_ar` and 7 of 8
  with flags (+17 to +82 for the Transformer), and ranks customers better in every cell at
  rate 0.30.
- **With unbounded counters** the Transformer is better in 12 of 16 cells, every cell at
  rates 0.01–0.05: it tolerates the counters and the LSTM does not. At rate 0.30 the
  advantage is gone (one cell each way).

Δ = Transformer minus LSTM on the same panels.

**`no_ar`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **−13** / **−23** / **+0.04** | −40 / **−55** / +0.01 | **−162** / **−188** / **+0.06** | **−562** / **−634** / **+0.09** | *−194 / −225 / +0.05* |
| 0.05 | **+30** / **+30** / +0.02 | +11 / +10 / **+0.22** | −9 / −13 / **+0.34** | **−375** / **−386** / **+0.32** | *−86 / −90 / +0.23* |
| 0.10 | **+25** / **+25** / **−0.09** | **+36** / +34 / −0.06 | **+34** / **+38** / **−0.07** | +32 / +40 / **+0.13** | *+32 / +34 / −0.02* |
| 0.30 | **+17** / +7 / **−0.14** | **+22** / **+20** / **−0.14** | **+39** / **+41** / **−0.11** | **+64** / **+71** / **−0.07** | *+35 / +35 / −0.11* |
| all |  |  |  |  | *−53 / −62 / +0.03* |

**`no_ar` + `kmeans_8`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **−31** / **−42** / −0.01 | +9 / −2 / −0.01 | **−153** / **−184** / 0 | **−534** / **−616** / **+0.04** | *−177 / −211 / 0* |
| 0.05 | **+21** / **+22** / +0.03 | +24 / +23 / +0.02 | **−38** / **−41** / **+0.10** | **−325** / **−339** / +0.01 | *−79 / −84 / +0.04* |
| 0.10 | **+19** / **+19** / +0.03 | **+20** / +19 / **+0.06** | −24 / −26 / **+0.06** | **−105** / **−107** / **+0.05** | *−23 / −24 / +0.05* |
| 0.30 | +2 / −6 / **+0.02** | +1 / 0 / **−0.04** | **−24** / **−25** / **−0.03** | −22 / −23 / **−0.02** | *−11 / −14 / −0.02* |
| all |  |  |  |  | *−73 / −83 / +0.02* |

**`ar_bounded`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | −2 / −2 / **+0.06** | **−45** / **−56** / **+0.07** | **−98** / **−129** / **+0.10** | **−416** / **−486** / **+0.15** | *−140 / −169 / +0.10* |
| 0.05 | **+20** / +16 / **−0.05** | +9 / +2 / +0.01 | +11 / +8 / 0 | **+42** / **+57** / +0.01 | *+20 / +21 / −0.01* |
| 0.10 | **+20** / +15 / **−0.02** | +6 / +5 / −0.05 | **+82** / **+92** / **−0.02** | **+57** / **+74** / 0 | *+41 / +46 / −0.02* |
| 0.30 | **+26** / **+17** / **−0.05** | **+29** / **+29** / **−0.11** | **+48** / **+52** / **−0.04** | **+19** / **+27** / **−0.02** | *+31 / +31 / −0.05* |
| all |  |  |  |  | *−12 / −17 / 0* |

**`ar_bounded` + `kmeans_8`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | +1 / 0 / +0.03 | −19 / −36 / **+0.05** | **−142** / **−174** / **+0.10** | **−249** / **−316** / +0.04 | *−102 / −131 / +0.06* |
| 0.05 | **+40** / **+48** / **+0.06** | −8 / −13 / **+0.07** | **+44** / **+44** / **+0.04** | **−195** / **−213** / +0.01 | *−30 / −34 / +0.04* |
| 0.10 | **+12** / +10 / +0.04 | +4 / +3 / **+0.01** | +15 / +14 / **+0.04** | **+43** / **+44** / **−0.03** | *+19 / +18 / +0.02* |
| 0.30 | **+7** / 0 / −0.03 | +2 / +1 / −0.01 | +13 / +13 / **−0.01** | +7 / +4 / **−0.02** | *+7 / +4 / −0.02* |
| all |  |  |  |  | *−27 / −36 / +0.02* |

**`ar_unbounded`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **−464** / **−484** / **+0.11** | **−111** / **−127** / **+0.23** | **−387** / **−434** / **+0.26** | **−493** / **−549** / **+0.12** | *−364 / −398 / +0.18* |
| 0.05 | **−149** / **−150** / **+0.08** | **−210** / **−212** / **+0.13** | **−337** / **−340** / **+0.13** | **−1215** / **−1237** / −0.07 | *−478 / −485 / +0.07* |
| 0.10 | **−199** / **−200** / **+0.05** | **−284** / **−285** / **+0.06** | **−148** / −142 / **+0.05** | −62 / −50 / +0.03 | *−173 / −169 / +0.05* |
| 0.30 | **−196** / **−198** / +0.03 | −76 / −77 / −0.05 | **+31** / **+36** / −0.03 | +53 / +64 / **−0.04** | *−48 / −46 / −0.02* |
| all |  |  |  |  | *−265 / −274 / +0.07* |

**`ar_unbounded` + `kmeans_8`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **−185** / **−189** / **+0.11** | **−1221** / **−1240** / **+0.11** | **−784** / **−817** / **+0.14** | **−1799** / **−1849** / **+0.11** | *−997 / −1023 / +0.12* |
| 0.05 | **−120** / **−124** / +0.01 | **−151** / **−153** / **+0.09** | **−256** / **−259** / +0.07 | **−823** / **−842** / **+0.06** | *−337 / −345 / +0.06* |
| 0.10 | **−98** / **−102** / +0.05 | **−97** / **−95** / **+0.06** | −65 / −63 / +0.01 | −39 / −29 / −0.01 | *−75 / −72 / +0.03* |
| 0.30 | **−43** / **−46** / **+0.04** | −21 / −22 / −0.01 | +12 / +16 / **−0.02** | +9 / +17 / **−0.02** | *−11 / −9 / 0* |
| all |  |  |  |  | *−355 / −362 / +0.05* |

<details><summary>By rate, churn pooled (descriptive)</summary>

| Comparison | Rate | n (A / B) | MAPE A → B | \|bias\| A → B | Δ MAPE (pooled mean) · cells supported | Δ \|bias\| (pooled mean) · cells supported | Δ RMSE (pooled mean) · cells supported | Δ Spearman (pooled mean) · cells supported |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `no_ar` | 0.01 | 40 / 40 | 314 → 120 | 309 → 84 | −194 · 3/4 (−) | −225 · 4/4 (−) | −0.17 · 3/4 (−) | +0.05 · 3/4 (+) |
| `no_ar` | 0.05 | 40 / 40 | 200 → 114 | 199 → 109 | −86 · 2/4 (1+, 1−) | −90 · 2/4 (1+, 1−) | −0.11 · 2/4 (1+, 1−) | +0.23 · 3/4 (+) |
| `no_ar` | 0.10 | 40 / 40 | 98 → 130 | 93 → 128 | +32 · 3/4 (+) | +34 · 2/4 (+) | +0.47 · 3/4 (+) | −0.02 · 3/4 (1+, 2−) |
| `no_ar` | 0.30 | 40 / 40 | 21 → 56 | 13 → 48 | +35 · 4/4 (+) | +35 · 3/4 (+) | +1.31 · 4/4 (+) | −0.11 · 4/4 (−) |
| `no_ar` | all | 160 / 160 | 158 → 105 | 154 → 92 | −53 · 12/16 (8+, 4−) | −62 · 11/16 (6+, 5−) | +0.37 · 12/16 (8+, 4−) | +0.03 · 13/16 (7+, 6−) |
| `no_ar` + `kmeans_8` | 0.01 | 40 / 40 | 307 → 130 | 302 → 91 | −177 · 3/4 (−) | −211 · 3/4 (−) | −0.12 · 3/4 (−) | 0 · 1/4 (+) |
| `no_ar` + `kmeans_8` | 0.05 | 40 / 40 | 202 → 122 | 200 → 116 | −79 · 3/4 (1+, 2−) | −84 · 3/4 (1+, 2−) | −0.08 · 3/4 (2+, 1−) | +0.04 · 1/4 (+) |
| `no_ar` + `kmeans_8` | 0.10 | 40 / 40 | 139 → 116 | 138 → 114 | −23 · 3/4 (2+, 1−) | −24 · 2/4 (1+, 1−) | +0.21 · 2/4 (+) | +0.05 · 3/4 (+) |
| `no_ar` + `kmeans_8` | 0.30 | 40 / 40 | 78 → 67 | 78 → 64 | −11 · 1/4 (−) | −14 · 1/4 (−) | −0.21 · 1/4 (−) | −0.02 · 4/4 (1+, 3−) |
| `no_ar` + `kmeans_8` | all | 160 / 160 | 181 → 109 | 179 → 96 | −73 · 10/16 (3+, 7−) | −83 · 9/16 (2+, 7−) | −0.05 · 9/16 (4+, 5−) | +0.02 · 9/16 (6+, 3−) |
| `ar_bounded` | 0.01 | 40 / 40 | 243 → 103 | 232 → 63 | −140 · 3/4 (−) | −169 · 3/4 (−) | −0.12 · 3/4 (−) | +0.10 · 4/4 (+) |
| `ar_bounded` | 0.05 | 40 / 40 | 58 → 78 | 46 → 68 | +20 · 2/4 (+) | +21 · 1/4 (+) | +0.10 · 2/4 (+) | −0.01 · 1/4 (−) |
| `ar_bounded` | 0.10 | 40 / 40 | 36 → 77 | 24 → 70 | +41 · 3/4 (+) | +46 · 2/4 (+) | +0.31 · 3/4 (+) | −0.02 · 2/4 (−) |
| `ar_bounded` | 0.30 | 40 / 40 | 20 → 50 | 11 → 42 | +31 · 4/4 (+) | +31 · 4/4 (+) | +1.04 · 4/4 (+) | −0.05 · 4/4 (−) |
| `ar_bounded` | all | 160 / 160 | 89 → 77 | 78 → 61 | −12 · 12/16 (9+, 3−) | −17 · 10/16 (7+, 3−) | +0.33 · 12/16 (9+, 3−) | 0 · 11/16 (4+, 7−) |
| `ar_bounded` + `kmeans_8` | 0.01 | 40 / 40 | 221 → 119 | 211 → 79 | −102 · 2/4 (−) | −131 · 2/4 (−) | −0.06 · 2/4 (−) | +0.06 · 2/4 (+) |
| `ar_bounded` + `kmeans_8` | 0.05 | 40 / 40 | 126 → 96 | 121 → 88 | −30 · 3/4 (2+, 1−) | −34 · 3/4 (2+, 1−) | +0.05 · 3/4 (2+, 1−) | +0.04 · 3/4 (+) |
| `ar_bounded` + `kmeans_8` | 0.10 | 40 / 40 | 78 → 97 | 76 → 93 | +19 · 2/4 (+) | +18 · 1/4 (+) | +0.26 · 1/4 (+) | +0.02 · 3/4 (2+, 1−) |
| `ar_bounded` + `kmeans_8` | 0.30 | 40 / 40 | 56 → 64 | 56 → 60 | +7 · 1/4 (+) | +4 · 0/4 | +0.35 · 0/4 | −0.02 · 2/4 (−) |
| `ar_bounded` + `kmeans_8` | all | 160 / 160 | 120 → 94 | 116 → 80 | −27 · 8/16 (5+, 3−) | −36 · 6/16 (3+, 3−) | +0.15 · 6/16 (3+, 3−) | +0.02 · 10/16 (7+, 3−) |
| `ar_unbounded` | 0.01 | 40 / 40 | 513 → 150 | 505 → 107 | −364 · 4/4 (−) | −398 · 4/4 (−) | −2.63 · 3/4 (−) | +0.18 · 4/4 (+) |
| `ar_unbounded` | 0.05 | 40 / 40 | 582 → 105 | 582 → 97 | −478 · 4/4 (−) | −485 · 4/4 (−) | −9.92 · 4/4 (−) | +0.07 · 3/4 (+) |
| `ar_unbounded` | 0.10 | 40 / 40 | 322 → 149 | 318 → 148 | −173 · 3/4 (−) | −169 · 2/4 (−) | −8.91 · 3/4 (−) | +0.05 · 3/4 (+) |
| `ar_unbounded` | 0.30 | 40 / 38 | 141 → 92 | 135 → 89 | −48 · 2/4 (1+, 1−) | −46 · 2/4 (1+, 1−) | −7.43 · 3/4 (1+, 2−) | −0.02 · 1/4 (−) |
| `ar_unbounded` | all | 160 / 158 | 390 → 124 | 385 → 111 | −265 · 13/16 (1+, 12−) | −274 · 12/16 (1+, 11−) | −7.29 · 13/16 (1+, 12−) | +0.07 · 11/16 (10+, 1−) |
| `ar_unbounded` + `kmeans_8` | 0.01 | 40 / 40 | 1118 → 121 | 1107 → 83 | −997 · 4/4 (−) | −1023 · 4/4 (−) | −6.35 · 4/4 (−) | +0.12 · 4/4 (+) |
| `ar_unbounded` + `kmeans_8` | 0.05 | 40 / 40 | 424 → 87 | 420 → 76 | −337 · 4/4 (−) | −345 · 4/4 (−) | −8.21 · 4/4 (−) | +0.06 · 2/4 (+) |
| `ar_unbounded` + `kmeans_8` | 0.10 | 40 / 40 | 162 → 88 | 153 → 81 | −75 · 2/4 (−) | −72 · 2/4 (−) | −5.05 · 3/4 (−) | +0.03 · 1/4 (+) |
| `ar_unbounded` + `kmeans_8` | 0.30 | 40 / 40 | 61 → 50 | 48 → 39 | −11 · 1/4 (−) | −9 · 1/4 (−) | −2.06 · 1/4 (−) | 0 · 3/4 (1+, 2−) |
| `ar_unbounded` + `kmeans_8` | all | 160 / 160 | 441 → 86 | 432 → 70 | −355 · 11/16 (−) | −362 · 11/16 (−) | −5.42 · 12/16 (−) | +0.05 · 10/16 (8+, 2−) |

</details>

<details><summary>Intervals per rate × churn cell</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI] | Δ \|bias\| [95% CI] | Δ RMSE [95% CI] | Δ Spearman [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `no_ar` | 0.01 | 20% | 10 | 71 → 58 | 59 → 37 | **−13 [−21, −5]** | **−23 [−36, −9]** | **−0.02 [−0.04, −0.01]** | **+0.04 [+0.01, +0.06]** |
| `no_ar` | 0.01 | 40% | 10 | 136 → 96 | 132 → 76 | −40 [−80, +2] | **−55 [−103, −5]** | −0.05 [−0.11, +0.01] | +0.01 [−0.04, +0.05] |
| `no_ar` | 0.01 | 60% | 10 | 291 → 129 | 290 → 102 | **−162 [−206, −121]** | **−188 [−245, −135]** | **−0.20 [−0.25, −0.15]** | **+0.06 [+0.01, +0.11]** |
| `no_ar` | 0.01 | 80% | 10 | 759 → 197 | 755 → 121 | **−562 [−783, −369]** | **−634 [−862, −433]** | **−0.41 [−0.51, −0.31]** | **+0.09 [+0.05, +0.13]** |
| `no_ar` | 0.05 | 20% | 10 | 45 → 75 | 42 → 72 | **+30 [+13, +49]** | **+30 [+9, +51]** | **+0.37 [+0.12, +0.66]** | +0.02 [−0.21, +0.25] |
| `no_ar` | 0.05 | 40% | 10 | 99 → 110 | 98 → 108 | +11 [−15, +35] | +10 [−17, +36] | +0.04 [−0.14, +0.21] | **+0.22 [+0.08, +0.35]** |
| `no_ar` | 0.05 | 60% | 10 | 158 → 149 | 157 → 144 | −9 [−70, +49] | −13 [−79, +49] | −0.09 [−0.31, +0.13] | **+0.34 [+0.31, +0.37]** |
| `no_ar` | 0.05 | 80% | 10 | 497 → 122 | 497 → 111 | **−375 [−435, −311]** | **−386 [−449, −318]** | **−0.76 [−0.90, −0.61]** | **+0.32 [+0.27, +0.37]** |
| `no_ar` | 0.10 | 20% | 10 | 33 → 58 | 31 → 56 | **+25 [+6, +45]** | **+25 [+2, +46]** | **+0.82 [+0.51, +1.10]** | **−0.09 [−0.11, −0.07]** |
| `no_ar` | 0.10 | 40% | 10 | 58 → 94 | 57 → 91 | **+36 [+3, +67]** | +34 [−1, +67] | **+0.58 [+0.09, +0.98]** | −0.06 [−0.11, +0.01] |
| `no_ar` | 0.10 | 60% | 10 | 69 → 103 | 61 → 99 | **+34 [+11, +65]** | **+38 [+12, +70]** | **+0.29 [+0.11, +0.53]** | **−0.07 [−0.11, −0.04]** |
| `no_ar` | 0.10 | 80% | 10 | 233 → 265 | 225 → 265 | +32 [−69, +131] | +40 [−68, +146] | +0.19 [−0.20, +0.56] | **+0.13 [+0.01, +0.26]** |
| `no_ar` | 0.30 | 20% | 10 | 19 → 36 | 17 → 24 | **+17 [+6, +31]** | +7 [−8, +25] | **+1.39 [+0.67, +2.45]** | **−0.14 [−0.15, −0.12]** |
| `no_ar` | 0.30 | 40% | 10 | 14 → 36 | 9 → 29 | **+22 [+14, +30]** | **+20 [+9, +31]** | **+1.60 [+1.17, +2.17]** | **−0.14 [−0.16, −0.12]** |
| `no_ar` | 0.30 | 60% | 10 | 18 → 57 | 9 → 50 | **+39 [+22, +58]** | **+41 [+18, +64]** | **+1.46 [+1.08, +1.89]** | **−0.11 [−0.13, −0.10]** |
| `no_ar` | 0.30 | 80% | 10 | 32 → 96 | 17 → 88 | **+64 [+30, +104]** | **+71 [+30, +116]** | **+0.79 [+0.48, +1.17]** | **−0.07 [−0.08, −0.05]** |
| `no_ar` + `kmeans_8` | 0.01 | 20% | 10 | 86 → 55 | 77 → 35 | **−31 [−45, −16]** | **−42 [−64, −21]** | **−0.05 [−0.08, −0.03]** | −0.01 [−0.04, +0.01] |
| `no_ar` + `kmeans_8` | 0.01 | 40% | 10 | 140 → 148 | 136 → 134 | +9 [−46, +77] | −2 [−64, +73] | +0.11 [−0.05, +0.29] | −0.01 [−0.06, +0.04] |
| `no_ar` + `kmeans_8` | 0.01 | 60% | 10 | 273 → 120 | 271 → 87 | **−153 [−196, −106]** | **−184 [−237, −126]** | **−0.17 [−0.26, −0.09]** | 0 [−0.04, +0.06] |
| `no_ar` + `kmeans_8` | 0.01 | 80% | 10 | 729 → 195 | 725 → 109 | **−534 [−717, −385]** | **−616 [−812, −455]** | **−0.38 [−0.48, −0.27]** | **+0.04 [0, +0.08]** |
| `no_ar` + `kmeans_8` | 0.05 | 20% | 10 | 47 → 67 | 40 → 62 | **+21 [+6, +36]** | **+22 [+3, +40]** | **+0.18 [+0.04, +0.33]** | +0.03 [−0.01, +0.07] |
| `no_ar` + `kmeans_8` | 0.05 | 40% | 10 | 82 → 107 | 81 → 105 | +24 [−5, +58] | +23 [−8, +58] | **+0.28 [+0.10, +0.49]** | +0.02 [0, +0.04] |
| `no_ar` + `kmeans_8` | 0.05 | 60% | 10 | 172 → 134 | 172 → 131 | **−38 [−70, −3]** | **−41 [−75, −4]** | −0.09 [−0.24, +0.10] | **+0.10 [+0.05, +0.17]** |
| `no_ar` + `kmeans_8` | 0.05 | 80% | 10 | 506 → 181 | 506 → 167 | **−325 [−406, −240]** | **−339 [−425, −252]** | **−0.70 [−0.90, −0.50]** | +0.01 [−0.05, +0.06] |
| `no_ar` + `kmeans_8` | 0.10 | 20% | 10 | 39 → 57 | 35 → 54 | **+19 [+7, +31]** | **+19 [+3, +34]** | **+0.42 [+0.08, +0.74]** | +0.03 [−0.03, +0.07] |
| `no_ar` + `kmeans_8` | 0.10 | 40% | 10 | 74 → 95 | 74 → 93 | **+20 [+3, +35]** | +19 [0, +36] | **+0.57 [+0.21, +0.88]** | **+0.06 [+0.02, +0.10]** |
| `no_ar` + `kmeans_8` | 0.10 | 60% | 10 | 132 → 109 | 132 → 106 | −24 [−57, +6] | −26 [−59, +5] | −0.11 [−0.38, +0.18] | **+0.06 [+0.02, +0.09]** |
| `no_ar` + `kmeans_8` | 0.10 | 80% | 10 | 309 → 203 | 309 → 201 | **−105 [−157, −52]** | **−107 [−159, −53]** | −0.03 [−0.32, +0.29] | **+0.05 [0, +0.10]** |
| `no_ar` + `kmeans_8` | 0.30 | 20% | 10 | 28 → 30 | 27 → 21 | +2 [−3, +7] | −6 [−13, +2] | −0.11 [−0.52, +0.38] | **+0.02 [0, +0.04]** |
| `no_ar` + `kmeans_8` | 0.30 | 40% | 10 | 53 → 55 | 53 → 53 | +1 [−10, +11] | 0 [−12, +10] | +0.27 [−0.31, +0.78] | **−0.04 [−0.05, −0.03]** |
| `no_ar` + `kmeans_8` | 0.30 | 60% | 10 | 90 → 65 | 90 → 65 | **−24 [−35, −13]** | **−25 [−36, −14]** | **−0.61 [−1.07, −0.17]** | **−0.03 [−0.04, −0.02]** |
| `no_ar` + `kmeans_8` | 0.30 | 80% | 10 | 142 → 120 | 142 → 119 | −22 [−54, +12] | −23 [−55, +12] | −0.37 [−0.99, +0.26] | **−0.02 [−0.03, −0.02]** |
| `ar_bounded` | 0.01 | 20% | 10 | 65 → 62 | 47 → 44 | −2 [−24, +17] | −2 [−33, +24] | −0.01 [−0.06, +0.03] | **+0.06 [+0.03, +0.10]** |
| `ar_bounded` | 0.01 | 40% | 10 | 121 → 75 | 114 → 58 | **−45 [−61, −28]** | **−56 [−77, −33]** | **−0.07 [−0.10, −0.04]** | **+0.07 [+0.04, +0.10]** |
| `ar_bounded` | 0.01 | 60% | 10 | 196 → 98 | 190 → 60 | **−98 [−135, −69]** | **−129 [−170, −95]** | **−0.11 [−0.16, −0.07]** | **+0.10 [+0.06, +0.15]** |
| `ar_bounded` | 0.01 | 80% | 10 | 591 → 176 | 576 → 90 | **−416 [−658, −204]** | **−486 [−750, −253]** | **−0.28 [−0.40, −0.16]** | **+0.15 [+0.08, +0.21]** |
| `ar_bounded` | 0.05 | 20% | 10 | 48 → 68 | 47 → 63 | **+20 [+3, +35]** | +16 [−7, +36] | **+0.28 [+0.11, +0.43]** | **−0.05 [−0.08, −0.02]** |
| `ar_bounded` | 0.05 | 40% | 10 | 56 → 65 | 52 → 54 | +9 [−17, +40] | +2 [−29, +38] | +0.03 [−0.14, +0.24] | +0.01 [0, +0.03] |
| `ar_bounded` | 0.05 | 60% | 10 | 67 → 77 | 61 → 69 | +11 [−12, +38] | +8 [−22, +44] | +0.04 [−0.05, +0.12] | 0 [−0.02, +0.01] |
| `ar_bounded` | 0.05 | 80% | 10 | 60 → 102 | 27 → 84 | **+42 [+7, +82]** | **+57 [+11, +108]** | **+0.06 [+0.03, +0.11]** | +0.01 [−0.01, +0.02] |
| `ar_bounded` | 0.10 | 20% | 10 | 32 → 52 | 30 → 45 | **+20 [+5, +36]** | +15 [−2, +34] | **+0.42 [+0.17, +0.69]** | **−0.02 [−0.02, −0.01]** |
| `ar_bounded` | 0.10 | 40% | 10 | 34 → 41 | 28 → 33 | +6 [−5, +18] | +5 [−11, +20] | +0.08 [−0.16, +0.34] | −0.05 [−0.16, +0.01] |
| `ar_bounded` | 0.10 | 60% | 10 | 33 → 115 | 21 → 113 | **+82 [+47, +123]** | **+92 [+57, +133]** | **+0.63 [+0.29, +1.03]** | **−0.02 [−0.03, −0.01]** |
| `ar_bounded` | 0.10 | 80% | 10 | 45 → 102 | 15 → 90 | **+57 [+24, +97]** | **+74 [+37, +118]** | **+0.12 [+0.02, +0.22]** | 0 [−0.01, +0.01] |
| `ar_bounded` | 0.30 | 20% | 10 | 12 → 37 | 10 → 27 | **+26 [+15, +38]** | **+17 [+2, +34]** | **+1.56 [+0.57, +2.73]** | **−0.05 [−0.06, −0.03]** |
| `ar_bounded` | 0.30 | 40% | 10 | 13 → 42 | 6 → 35 | **+29 [+19, +42]** | **+29 [+14, +45]** | **+1.22 [+0.58, +1.98]** | **−0.11 [−0.21, −0.04]** |
| `ar_bounded` | 0.30 | 60% | 10 | 19 → 67 | 10 → 62 | **+48 [+27, +72]** | **+52 [+28, +77]** | **+1.20 [+0.65, +1.80]** | **−0.04 [−0.05, −0.02]** |
| `ar_bounded` | 0.30 | 80% | 10 | 35 → 54 | 19 → 46 | **+19 [+8, +30]** | **+27 [+8, +45]** | **+0.17 [+0.01, +0.32]** | **−0.02 [−0.04, −0.01]** |
| `ar_bounded` + `kmeans_8` | 0.01 | 20% | 10 | 72 → 73 | 58 → 59 | +1 [−23, +34] | 0 [−30, +39] | +0.02 [−0.06, +0.11] | +0.03 [0, +0.06] |
| `ar_bounded` + `kmeans_8` | 0.01 | 40% | 10 | 121 → 102 | 116 → 80 | −19 [−62, +35] | −36 [−86, +26] | 0 [−0.10, +0.14] | **+0.05 [0, +0.08]** |
| `ar_bounded` + `kmeans_8` | 0.01 | 60% | 10 | 245 → 103 | 241 → 67 | **−142 [−180, −98]** | **−174 [−224, −119]** | **−0.16 [−0.23, −0.07]** | **+0.10 [+0.05, +0.16]** |
| `ar_bounded` + `kmeans_8` | 0.01 | 80% | 10 | 445 → 196 | 429 → 112 | **−249 [−392, −133]** | **−316 [−474, −181]** | **−0.12 [−0.21, −0.04]** | +0.04 [−0.05, +0.11] |
| `ar_bounded` + `kmeans_8` | 0.05 | 20% | 10 | 39 → 79 | 29 → 77 | **+40 [+29, +49]** | **+48 [+37, +59]** | **+0.45 [+0.32, +0.59]** | **+0.06 [+0.03, +0.08]** |
| `ar_bounded` + `kmeans_8` | 0.05 | 40% | 10 | 64 → 56 | 61 → 48 | −8 [−30, +10] | −13 [−38, +9] | 0 [−0.16, +0.14] | **+0.07 [+0.03, +0.10]** |
| `ar_bounded` + `kmeans_8` | 0.05 | 60% | 10 | 110 → 155 | 108 → 152 | **+44 [+4, +89]** | **+44 [+1, +90]** | **+0.22 [+0.03, +0.38]** | **+0.04 [+0.01, +0.07]** |
| `ar_bounded` + `kmeans_8` | 0.05 | 80% | 10 | 290 → 94 | 287 → 73 | **−195 [−286, −110]** | **−213 [−304, −125]** | **−0.47 [−0.70, −0.26]** | +0.01 [−0.01, +0.04] |
| `ar_bounded` + `kmeans_8` | 0.10 | 20% | 10 | 33 → 45 | 28 → 38 | **+12 [0, +27]** | +10 [−8, +28] | +0.25 [−0.03, +0.54] | +0.04 [−0.02, +0.09] |
| `ar_bounded` + `kmeans_8` | 0.10 | 40% | 10 | 53 → 58 | 52 → 54 | +4 [−11, +20] | +3 [−14, +20] | +0.26 [−0.06, +0.64] | **+0.01 [0, +0.03]** |
| `ar_bounded` + `kmeans_8` | 0.10 | 60% | 10 | 93 → 108 | 92 → 106 | +15 [−14, +49] | +14 [−16, +49] | +0.26 [−0.05, +0.63] | **+0.04 [+0.01, +0.06]** |
| `ar_bounded` + `kmeans_8` | 0.10 | 80% | 10 | 134 → 177 | 131 → 175 | **+43 [+16, +72]** | **+44 [+16, +73]** | **+0.27 [+0.05, +0.51]** | **−0.03 [−0.06, 0]** |
| `ar_bounded` + `kmeans_8` | 0.30 | 20% | 10 | 26 → 32 | 25 → 25 | **+7 [+2, +12]** | 0 [−8, +7] | +0.13 [−0.37, +0.58] | −0.03 [−0.08, 0] |
| `ar_bounded` + `kmeans_8` | 0.30 | 40% | 10 | 45 → 47 | 44 → 45 | +2 [−7, +11] | +1 [−11, +11] | +0.23 [−0.34, +0.73] | −0.01 [−0.02, 0] |
| `ar_bounded` + `kmeans_8` | 0.30 | 60% | 10 | 71 → 85 | 71 → 83 | +13 [−10, +42] | +13 [−12, +42] | +0.60 [−0.30, +1.83] | **−0.01 [−0.03, 0]** |
| `ar_bounded` + `kmeans_8` | 0.30 | 80% | 10 | 84 → 91 | 83 → 87 | +7 [−25, +42] | +4 [−29, +42] | +0.47 [−0.13, +1.12] | **−0.02 [−0.04, 0]** |
| `ar_unbounded` | 0.01 | 20% | 10 | 513 → 49 | 502 → 19 | **−464 [−1018, −55]** | **−484 [−1040, −72]** | **−5.87 [−11.82, −1.12]** | **+0.11 [+0.06, +0.16]** |
| `ar_unbounded` | 0.01 | 40% | 10 | 189 → 78 | 182 → 55 | **−111 [−189, −42]** | **−127 [−207, −52]** | **−1.17 [−2.64, −0.11]** | **+0.23 [+0.16, +0.30]** |
| `ar_unbounded` | 0.01 | 60% | 10 | 514 → 127 | 514 → 79 | **−387 [−556, −253]** | **−434 [−599, −299]** | **−2.64 [−4.84, −0.75]** | **+0.26 [+0.21, +0.31]** |
| `ar_unbounded` | 0.01 | 80% | 10 | 838 → 345 | 821 → 273 | **−493 [−864, −67]** | **−549 [−938, −109]** | −0.83 [−2.02, +0.14] | **+0.12 [+0.03, +0.19]** |
| `ar_unbounded` | 0.05 | 20% | 10 | 240 → 91 | 240 → 90 | **−149 [−221, −89]** | **−150 [−223, −89]** | **−11.54 [−14.49, −8.96]** | **+0.08 [+0.06, +0.09]** |
| `ar_unbounded` | 0.05 | 40% | 10 | 313 → 103 | 313 → 101 | **−210 [−282, −140]** | **−212 [−285, −141]** | **−8.07 [−10.78, −5.55]** | **+0.13 [+0.04, +0.19]** |
| `ar_unbounded` | 0.05 | 60% | 10 | 462 → 124 | 462 → 121 | **−337 [−500, −219]** | **−340 [−503, −222]** | **−9.03 [−11.16, −6.71]** | **+0.13 [+0.08, +0.18]** |
| `ar_unbounded` | 0.05 | 80% | 10 | 1315 → 100 | 1314 → 77 | **−1215 [−1702, −695]** | **−1237 [−1723, −719]** | **−11.05 [−15.65, −6.31]** | −0.07 [−0.23, +0.07] |
| `ar_unbounded` | 0.10 | 20% | 10 | 301 → 102 | 301 → 101 | **−199 [−297, −114]** | **−200 [−297, −115]** | **−16.15 [−21.99, −10.75]** | **+0.05 [+0.02, +0.07]** |
| `ar_unbounded` | 0.10 | 40% | 10 | 375 → 91 | 375 → 89 | **−284 [−383, −209]** | **−285 [−384, −210]** | **−14.13 [−20.01, −8.99]** | **+0.06 [+0.01, +0.10]** |
| `ar_unbounded` | 0.10 | 60% | 10 | 307 → 159 | 301 → 159 | **−148 [−280, −6]** | −142 [−278, +4] | **−3.99 [−7.34, −1.07]** | **+0.05 [+0.01, +0.10]** |
| `ar_unbounded` | 0.10 | 80% | 10 | 307 → 245 | 295 → 245 | −62 [−268, +119] | −50 [−262, +132] | −1.39 [−3.92, +0.51] | +0.03 [−0.01, +0.07] |
| `ar_unbounded` | 0.30 | 20% | 10 | 268 → 71 | 268 → 69 | **−196 [−246, −146]** | **−198 [−249, −146]** | **−23.54 [−29.57, −17.13]** | +0.03 [−0.03, +0.07] |
| `ar_unbounded` | 0.30 | 40% | 10 | 161 → 86 | 158 → 81 | −76 [−167, +10] | −77 [−172, +11] | **−7.81 [−15.64, −0.59]** | −0.05 [−0.13, +0.03] |
| `ar_unbounded` | 0.30 | 60% | 9 | 66 → 97 | 60 → 96 | **+31 [+10, +52]** | **+36 [+13, +59]** | **+0.75 [+0.18, +1.30]** | −0.03 [−0.06, 0] |
| `ar_unbounded` | 0.30 | 80% | 9 | 65 → 118 | 49 → 112 | +53 [−23, +126] | +64 [−21, +143] | +0.43 [−1.80, +2.49] | **−0.04 [−0.06, −0.03]** |
| `ar_unbounded` + `kmeans_8` | 0.01 | 20% | 10 | 250 → 65 | 235 → 46 | **−185 [−321, −63]** | **−189 [−325, −65]** | **−4.22 [−6.84, −1.77]** | **+0.11 [+0.07, +0.16]** |
| `ar_unbounded` + `kmeans_8` | 0.01 | 40% | 10 | 1304 → 83 | 1300 → 60 | **−1221 [−2312, −338]** | **−1240 [−2333, −354]** | **−10.62 [−17.16, −4.63]** | **+0.11 [+0.01, +0.21]** |
| `ar_unbounded` + `kmeans_8` | 0.01 | 60% | 10 | 904 → 120 | 900 → 83 | **−784 [−1355, −315]** | **−817 [−1391, −346]** | **−4.79 [−8.94, −1.20]** | **+0.14 [+0.05, +0.24]** |
| `ar_unbounded` + `kmeans_8` | 0.01 | 80% | 10 | 2015 → 216 | 1993 → 145 | **−1799 [−3304, −575]** | **−1849 [−3342, −620]** | **−5.76 [−11.44, −0.90]** | **+0.11 [+0.02, +0.20]** |
| `ar_unbounded` + `kmeans_8` | 0.05 | 20% | 10 | 189 → 69 | 188 → 64 | **−120 [−156, −77]** | **−124 [−161, −79]** | **−8.70 [−10.83, −6.43]** | +0.01 [−0.06, +0.05] |
| `ar_unbounded` + `kmeans_8` | 0.05 | 40% | 10 | 229 → 78 | 221 → 68 | **−151 [−290, −38]** | **−153 [−296, −35]** | **−7.26 [−12.37, −2.90]** | **+0.09 [+0.01, +0.15]** |
| `ar_unbounded` + `kmeans_8` | 0.05 | 60% | 10 | 359 → 103 | 357 → 98 | **−256 [−483, −66]** | **−259 [−486, −69]** | **−6.60 [−10.25, −3.17]** | +0.07 [−0.05, +0.18] |
| `ar_unbounded` + `kmeans_8` | 0.05 | 80% | 10 | 919 → 97 | 914 → 72 | **−823 [−1152, −471]** | **−842 [−1181, −478]** | **−10.29 [−13.53, −6.84]** | **+0.06 [+0.02, +0.11]** |
| `ar_unbounded` + `kmeans_8` | 0.10 | 20% | 10 | 145 → 47 | 141 → 39 | **−98 [−162, −41]** | **−102 [−167, −43]** | **−9.23 [−14.11, −4.57]** | +0.05 [−0.03, +0.12] |
| `ar_unbounded` + `kmeans_8` | 0.10 | 40% | 10 | 174 → 77 | 167 → 72 | **−97 [−168, −30]** | **−95 [−169, −21]** | **−5.98 [−10.35, −1.89]** | **+0.06 [0, +0.11]** |
| `ar_unbounded` + `kmeans_8` | 0.10 | 60% | 10 | 142 → 76 | 133 → 70 | −65 [−171, +21] | −63 [−175, +30] | **−3.24 [−5.88, −0.74]** | +0.01 [−0.05, +0.07] |
| `ar_unbounded` + `kmeans_8` | 0.10 | 80% | 10 | 189 → 150 | 171 → 142 | −39 [−198, +80] | −29 [−193, +98] | −1.77 [−4.29, +0.35] | −0.01 [−0.04, +0.02] |
| `ar_unbounded` + `kmeans_8` | 0.30 | 20% | 10 | 80 → 37 | 76 → 30 | **−43 [−74, −13]** | **−46 [−82, −10]** | **−6.76 [−12.65, −2.24]** | **+0.04 [+0.02, +0.07]** |
| `ar_unbounded` + `kmeans_8` | 0.30 | 40% | 10 | 62 → 41 | 50 → 28 | −21 [−58, +8] | −22 [−63, +14] | −2.40 [−7.03, +0.80] | −0.01 [−0.05, +0.03] |
| `ar_unbounded` + `kmeans_8` | 0.30 | 60% | 10 | 54 → 65 | 39 → 55 | +12 [−2, +24] | +16 [−6, +37] | +0.56 [−0.03, +1.23] | **−0.02 [−0.04, 0]** |
| `ar_unbounded` + `kmeans_8` | 0.30 | 80% | 10 | 47 → 56 | 26 → 43 | +9 [−1, +19] | +17 [−1, +37] | +0.37 [−0.09, +0.86] | **−0.02 [−0.03, 0]** |

</details>

### 7. Neural models capture seasonality; Pareto/NBD cannot

**Verdict: supported** for this one seasonal pattern.

- **Shape.** The LSTM's predicted weekly totals correlate with the actual ones at 0.11–0.96
  and the Transformer's at 0.09–0.62, against −0.09 to +0.13 for Pareto/NBD. The
  correlation rises with the rate and falls with churn.
- **Counterfactual.** Multiplying Pareto/NBD's forecast by the true seasonal curve (rescaled
  to mean 1) lowers its MAPE by 5–20 points, supported in all 16 cells. |bias| and Spearman
  barely move (at most 1.0 point and 0.001 per cell): several of those tiny changes are supported,
  because the rescaling moves every panel by nearly the same amount, but the season changes
  the weekly shape, not the yearly total or the order of customers.
- **Given the season, Pareto/NBD beats the LSTM** on MAPE in every cell at rates 0.01–0.10
  (+7 to +439 points for the LSTM) and at rate 0.30 with churn 80% (+12). At rate 0.30 with
  churn 20–40% the LSTM keeps a small lead (2–3 points), and churn 60% shows no clear
  difference. So the LSTM's dense-panel MAPE wins in claim 1 come mostly from modelling the
  season.

Shape correlation, mean over panels: Pareto/NBD / LSTM `ar_bounded` / Transformer `ar_bounded`

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled |
| --- | --- | --- | --- | --- | --- |
| 0.01 | −0.02 / +0.26 / +0.32 | 0 / +0.29 / +0.22 | −0.02 / +0.14 / +0.11 | +0.05 / +0.11 / +0.09 | 0 / +0.20 / +0.19 |
| 0.05 | −0.09 / +0.70 / +0.50 | +0.04 / +0.64 / +0.51 | +0.06 / +0.53 / +0.35 | +0.06 / +0.21 / +0.23 | +0.02 / +0.52 / +0.40 |
| 0.10 | −0.07 / +0.84 / +0.62 | +0.01 / +0.73 / +0.50 | +0.10 / +0.67 / +0.41 | +0.11 / +0.36 / +0.30 | +0.04 / +0.65 / +0.46 |
| 0.30 | −0.08 / +0.96 / +0.60 | +0.01 / +0.91 / +0.55 | +0.12 / +0.83 / +0.53 | +0.13 / +0.49 / +0.47 | +0.05 / +0.80 / +0.54 |

**Pareto/NBD → with true season**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **−9** / **0** / 0 | **−8** / **0** / **0** | **−5** / **−1** / 0 | **−6** / **−1** / **0** | *−7 / −1 / 0* |
| 0.05 | **−20** / **0** / 0 | **−14** / **−1** / 0 | **−14** / **−1** / 0 | **−7** / −1 / **0** | *−14 / −1 / 0* |
| 0.10 | **−20** / **0** / 0 | **−18** / 0 / 0 | **−16** / 0 / 0 | **−11** / 0 / **0** | *−16 / 0 / 0* |
| 0.30 | **−15** / **0** / **0** | **−16** / **+1** / **0** | **−16** / **+1** / 0 | **−12** / +1 / 0 | *−15 / +1 / 0* |
| all |  |  |  |  | *−13 / 0 / 0* |

**Pareto/NBD with season → LSTM `ar_bounded`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **+25** / **+35** / **−0.25** | **+63** / **+81** / **−0.34** | **+113** / **+150** / **−0.31** | **+439** / **+522** / **−0.27** | *+160 / +197 / −0.29* |
| 0.05 | **+33** / **+38** / **−0.07** | **+31** / **+37** / **−0.06** | **+38** / **+49** / **−0.04** | **+9** / +7 / **−0.04** | *+28 / +33 / −0.06* |
| 0.10 | **+22** / **+27** / **−0.03** | **+21** / **+23** / **−0.03** | **+12** / **+13** / **−0.03** | **+7** / −5 / **−0.04** | *+15 / +15 / −0.03* |
| 0.30 | **−3** / **−5** / **−0.01** | **−2** / **−8** / **−0.01** | +3 / −3 / **−0.07** | **+12** / +5 / **−0.19** | *+2 / −3 / −0.07* |
| all |  |  |  |  | *+51 / +60 / −0.11* |

<details><summary>By rate, churn pooled (descriptive)</summary>

| Comparison | Rate | n (A / B) | MAPE A → B | \|bias\| A → B | Δ MAPE (pooled mean) · cells supported | Δ \|bias\| (pooled mean) · cells supported | Δ RMSE (pooled mean) · cells supported | Δ Spearman (pooled mean) · cells supported |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pareto/NBD → with true season | 0.01 | 40 / 40 | 90 → 83 | 35 → 35 | −7 · 4/4 (−) | −1 · 4/4 (−) | 0 · 4/4 (−) | 0 · 2/4 (−) |
| Pareto/NBD → with true season | 0.05 | 40 / 40 | 44 → 30 | 14 → 14 | −14 · 4/4 (−) | −1 · 3/4 (−) | 0 · 2/4 (−) | 0 · 1/4 (−) |
| Pareto/NBD → with true season | 0.10 | 40 / 40 | 37 → 21 | 9 → 9 | −16 · 4/4 (−) | 0 · 1/4 (−) | 0 · 2/4 (+) | 0 · 1/4 (−) |
| Pareto/NBD → with true season | 0.30 | 40 / 40 | 32 → 17 | 13 → 14 | −15 · 4/4 (−) | +1 · 3/4 (+) | +0.01 · 4/4 (+) | 0 · 2/4 (−) |
| Pareto/NBD → with true season | all | 160 / 160 | 51 → 38 | 18 → 18 | −13 · 16/16 (−) | 0 · 11/16 (3+, 8−) | 0 · 12/16 (6+, 6−) | 0 · 6/16 (−) |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.01 | 40 / 40 | 83 → 243 | 35 → 232 | +160 · 4/4 (+) | +197 · 4/4 (+) | +0.15 · 4/4 (+) | −0.29 · 4/4 (−) |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.05 | 40 / 40 | 30 → 58 | 14 → 46 | +28 · 4/4 (+) | +33 · 3/4 (+) | +0.20 · 4/4 (+) | −0.06 · 4/4 (−) |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.10 | 40 / 40 | 21 → 36 | 9 → 24 | +15 · 4/4 (+) | +15 · 3/4 (+) | +0.23 · 4/4 (+) | −0.03 · 4/4 (−) |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.30 | 40 / 40 | 17 → 20 | 14 → 11 | +2 · 3/4 (1+, 2−) | −3 · 2/4 (−) | −0.24 · 3/4 (−) | −0.07 · 4/4 (−) |
| Pareto/NBD with season → LSTM `ar_bounded` | all | 160 / 160 | 38 → 89 | 18 → 78 | +51 · 15/16 (13+, 2−) | +60 · 12/16 (10+, 2−) | +0.09 · 15/16 (12+, 3−) | −0.11 · 16/16 (−) |

</details>

<details><summary>Intervals per rate × churn cell</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI] | Δ \|bias\| [95% CI] | Δ RMSE [95% CI] | Δ Spearman [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pareto/NBD → with true season | 0.01 | 20% | 10 | 48 → 39 | 12 → 12 | **−9 [−11, −7]** | **0 [0, 0]** | **0 [0, 0]** | 0 [0, 0] |
| Pareto/NBD → with true season | 0.01 | 40% | 10 | 66 → 58 | 34 → 34 | **−8 [−11, −5]** | **0 [0, 0]** | **0 [0, 0]** | **0 [0, 0]** |
| Pareto/NBD → with true season | 0.01 | 60% | 10 | 88 → 83 | 41 → 40 | **−5 [−8, −2]** | **−1 [−1, −1]** | **0 [0, 0]** | 0 [0, 0] |
| Pareto/NBD → with true season | 0.01 | 80% | 10 | 159 → 153 | 54 → 53 | **−6 [−12, −1]** | **−1 [−1, −1]** | **0 [0, 0]** | **0 [0, 0]** |
| Pareto/NBD → with true season | 0.05 | 20% | 10 | 35 → 16 | 8 → 8 | **−20 [−20, −19]** | **0 [0, 0]** | **0 [0, 0]** | 0 [0, 0] |
| Pareto/NBD → with true season | 0.05 | 40% | 10 | 39 → 24 | 15 → 15 | **−14 [−16, −13]** | **−1 [−1, −1]** | **0 [0, 0]** | 0 [0, 0] |
| Pareto/NBD → with true season | 0.05 | 60% | 10 | 42 → 29 | 12 → 12 | **−14 [−15, −12]** | **−1 [−1, 0]** | 0 [0, 0] | 0 [0, 0] |
| Pareto/NBD → with true season | 0.05 | 80% | 10 | 58 → 51 | 21 → 20 | **−7 [−11, −4]** | −1 [−2, 0] | 0 [0, 0] | **0 [0, 0]** |
| Pareto/NBD → with true season | 0.10 | 20% | 10 | 31 → 11 | 3 → 3 | **−20 [−21, −20]** | **0 [0, 0]** | **0 [0, 0]** | 0 [0, 0] |
| Pareto/NBD → with true season | 0.10 | 40% | 10 | 32 → 13 | 5 → 5 | **−18 [−20, −17]** | 0 [−1, 0] | **0 [0, 0]** | 0 [0, 0] |
| Pareto/NBD → with true season | 0.10 | 60% | 10 | 37 → 21 | 9 → 8 | **−16 [−18, −14]** | 0 [−1, 0] | 0 [0, 0] | 0 [0, 0] |
| Pareto/NBD → with true season | 0.10 | 80% | 10 | 49 → 38 | 21 → 20 | **−11 [−14, −9]** | 0 [−1, +1] | 0 [0, 0] | **0 [0, 0]** |
| Pareto/NBD → with true season | 0.30 | 20% | 10 | 30 → 15 | 14 → 15 | **−15 [−17, −14]** | **0 [0, 0]** | **+0.01 [+0.01, +0.01]** | **0 [0, 0]** |
| Pareto/NBD → with true season | 0.30 | 40% | 10 | 31 → 15 | 14 → 14 | **−16 [−17, −15]** | **+1 [0, +1]** | **+0.02 [+0.01, +0.02]** | **0 [0, 0]** |
| Pareto/NBD → with true season | 0.30 | 60% | 10 | 32 → 16 | 13 → 13 | **−16 [−17, −14]** | **+1 [0, +1]** | **+0.02 [+0.01, +0.02]** | 0 [0, 0] |
| Pareto/NBD → with true season | 0.30 | 80% | 10 | 35 → 23 | 13 → 14 | **−12 [−14, −10]** | +1 [0, +1] | **+0.01 [+0.01, +0.02]** | 0 [0, 0] |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.01 | 20% | 10 | 39 → 65 | 12 → 47 | **+25 [+13, +41]** | **+35 [+16, +55]** | **+0.07 [+0.04, +0.11]** | **−0.25 [−0.33, −0.19]** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.01 | 40% | 10 | 58 → 121 | 34 → 114 | **+63 [+51, +77]** | **+81 [+67, +96]** | **+0.10 [+0.08, +0.13]** | **−0.34 [−0.39, −0.30]** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.01 | 60% | 10 | 83 → 196 | 40 → 190 | **+113 [+76, +157]** | **+150 [+105, +199]** | **+0.14 [+0.09, +0.20]** | **−0.31 [−0.38, −0.23]** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.01 | 80% | 10 | 153 → 591 | 53 → 576 | **+439 [+235, +671]** | **+522 [+303, +769]** | **+0.29 [+0.18, +0.40]** | **−0.27 [−0.31, −0.22]** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.05 | 20% | 10 | 16 → 48 | 8 → 47 | **+33 [+26, +39]** | **+38 [+32, +45]** | **+0.36 [+0.31, +0.42]** | **−0.07 [−0.11, −0.05]** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.05 | 40% | 10 | 24 → 56 | 15 → 52 | **+31 [+23, +38]** | **+37 [+29, +45]** | **+0.25 [+0.21, +0.28]** | **−0.06 [−0.08, −0.05]** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.05 | 60% | 10 | 29 → 67 | 12 → 61 | **+38 [+28, +48]** | **+49 [+32, +62]** | **+0.12 [+0.06, +0.17]** | **−0.04 [−0.05, −0.03]** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.05 | 80% | 10 | 51 → 60 | 20 → 27 | **+9 [+4, +17]** | +7 [−4, +19] | **+0.08 [+0.06, +0.11]** | **−0.04 [−0.05, −0.03]** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.10 | 20% | 10 | 11 → 32 | 3 → 30 | **+22 [+14, +30]** | **+27 [+17, +37]** | **+0.41 [+0.31, +0.54]** | **−0.03 [−0.04, −0.02]** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.10 | 40% | 10 | 13 → 34 | 5 → 28 | **+21 [+14, +29]** | **+23 [+13, +34]** | **+0.32 [+0.22, +0.44]** | **−0.03 [−0.04, −0.03]** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.10 | 60% | 10 | 21 → 33 | 8 → 21 | **+12 [+9, +15]** | **+13 [+7, +18]** | **+0.11 [+0.08, +0.15]** | **−0.03 [−0.04, −0.02]** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.10 | 80% | 10 | 38 → 45 | 20 → 15 | **+7 [+2, +12]** | −5 [−14, +5] | **+0.08 [+0.04, +0.13]** | **−0.04 [−0.05, −0.03]** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.30 | 20% | 10 | 15 → 12 | 15 → 10 | **−3 [−5, −1]** | **−5 [−9, −2]** | **−0.45 [−0.66, −0.22]** | **−0.01 [−0.01, 0]** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.30 | 40% | 10 | 15 → 13 | 14 → 6 | **−2 [−4, 0]** | **−8 [−11, −5]** | **−0.35 [−0.46, −0.23]** | **−0.01 [−0.02, −0.01]** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.30 | 60% | 10 | 16 → 19 | 13 → 10 | +3 [0, +6] | −3 [−9, +2] | **−0.13 [−0.21, −0.05]** | **−0.07 [−0.08, −0.06]** |
| Pareto/NBD with season → LSTM `ar_bounded` | 0.30 | 80% | 10 | 23 → 35 | 14 → 19 | **+12 [+9, +16]** | +5 [−3, +14] | −0.03 [−0.13, +0.07] | **−0.19 [−0.21, −0.18]** |

</details>

### 8. Pareto/NBD's low bias means it is accurate per customer

**Verdict: not supported.** In every cell Pareto/NBD under-serves customers who are still
alive (alive ratio R_A 0.51–0.87; R_A < 1 supported in all 16 cells) and assigns volume to
customers who have already died (dead leakage L_D 0.11–1.04; L_D > 0 supported in all 16).
Each panel gives one R_A and one L_D, tested per cell against 1 and 0 by the one-statistic
rule, paired, 10 panels a cell. The two errors
cancel in the total, which is why its |bias| is small. Both worsen with churn: at rate 0.01
and churn 80% it puts more volume on dead customers (L_D 1.04) than the living ones should
get. R_A and L_D use the generator's hidden truth (each customer's λ and death week).

| Rate | Churn | n | \|bias\| (mean) | R_A: mean, R_A − 1 [95% CI] | L_D: mean [95% CI] | RMSE (mean) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 20% | 10 | 12 | **0.85, −0.15 [−0.18, −0.12]** | **0.28 [0.27, 0.28]** | 0.79 |
| 0.01 | 40% | 10 | 34 | **0.77, −0.23 [−0.26, −0.20]** | **0.55 [0.52, 0.59]** | 0.68 |
| 0.01 | 60% | 10 | 41 | **0.65, −0.35 [−0.39, −0.30]** | **0.76 [0.72, 0.80]** | 0.57 |
| 0.01 | 80% | 10 | 54 | **0.51, −0.49 [−0.53, −0.45]** | **1.04 [0.90, 1.21]** | 0.42 |
| 0.05 | 20% | 10 | 8 | **0.87, −0.13 [−0.15, −0.12]** | **0.22 [0.21, 0.24]** | 1.92 |
| 0.05 | 40% | 10 | 15 | **0.77, −0.23 [−0.25, −0.21]** | **0.39 [0.38, 0.40]** | 1.63 |
| 0.05 | 60% | 10 | 12 | **0.65, −0.35 [−0.37, −0.32]** | **0.44 [0.40, 0.48]** | 1.32 |
| 0.05 | 80% | 10 | 21 | **0.56, −0.44 [−0.49, −0.39]** | **0.53 [0.46, 0.60]** | 0.89 |
| 0.10 | 20% | 10 | 3 | **0.84, −0.16 [−0.17, −0.15]** | **0.18 [0.17, 0.19]** | 2.91 |
| 0.10 | 40% | 10 | 5 | **0.74, −0.26 [−0.28, −0.24]** | **0.27 [0.26, 0.29]** | 2.47 |
| 0.10 | 60% | 10 | 9 | **0.67, −0.33 [−0.35, −0.30]** | **0.35 [0.31, 0.40]** | 2.00 |
| 0.10 | 80% | 10 | 21 | **0.60, −0.40 [−0.44, −0.36]** | **0.42 [0.35, 0.49]** | 1.37 |
| 0.30 | 20% | 10 | 14 | **0.74, −0.26 [−0.26, −0.25]** | **0.11 [0.11, 0.12]** | 6.93 |
| 0.30 | 40% | 10 | 14 | **0.68, −0.32 [−0.33, −0.31]** | **0.19 [0.17, 0.20]** | 5.99 |
| 0.30 | 60% | 10 | 13 | **0.63, −0.37 [−0.38, −0.35]** | **0.25 [0.23, 0.27]** | 4.72 |
| 0.30 | 80% | 10 | 13 | **0.58, −0.42 [−0.44, −0.39]** | **0.29 [0.24, 0.34]** | 3.03 |
| 0.01 | pooled (descriptive) | 40 | 35 | 0.70 | 0.66 | 0.62 |
| 0.05 | pooled (descriptive) | 40 | 14 | 0.71 | 0.40 | 1.44 |
| 0.10 | pooled (descriptive) | 40 | 9 | 0.71 | 0.31 | 2.19 |
| 0.30 | pooled (descriptive) | 40 | 13 | 0.66 | 0.21 | 5.17 |
| all | pooled (descriptive) | 160 | 18 | 0.70 | 0.39 | 2.35 |

### 9. Neural models cannot detect a customer who has stopped

**Verdict: partly.** On dead leakage L_D, lower is better.

- **LSTM `ar_bounded`** leaks more than Pareto/NBD in every cell at rates 0.01–0.10 (+0.10
  to +4.44), most at rate 0.01, churn 80%, and slightly more at rate 0.30 with churn 20–40%
  (+0.05, +0.03). At rate 0.30 with churn 60–80% there is no clear difference (+0.01 and
  −0.02 at n = 10): there the flags come closest to working.
- **Transformer `ar_bounded`** leaks more than Pareto/NBD in every cell (+0.08 to +0.83),
  supported in 15 of 16 (all but rate 0.01, churn 40%).

**LSTM**: Δ L_D / Δ R_A

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **+0.14** / **+0.18** | **+0.54** / **+0.26** | **+1.25** / **+0.29** | **+4.44** / **+0.52** | *+1.59 / +0.31* |
| 0.05 | **+0.20** / **+0.18** | **+0.34** / +0.03 | **+0.48** / **+0.04** | **+0.27** / **−0.21** | *+0.32 / +0.01* |
| 0.10 | **+0.14** / **+0.12** | **+0.23** / +0.04 | **+0.17** / −0.01 | **+0.10** / **−0.13** | *+0.16 / +0.01* |
| 0.30 | **+0.05** / **+0.19** | **+0.03** / **+0.12** | +0.01 / **+0.05** | −0.02 / +0.01 | *+0.02 / +0.09* |
| all |  |  |  |  | *+0.52 / +0.11* |

**Transformer**: Δ L_D / Δ R_A

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **+0.11** / +0.12 | +0.08 / −0.16 | **+0.28** / **−0.14** | **+0.55** / **−0.13** | *+0.25 / −0.08* |
| 0.05 | **+0.29** / **+0.24** | **+0.36** / +0.03 | **+0.58** / 0 | **+0.83** / **−0.19** | *+0.51 / +0.02* |
| 0.10 | **+0.21** / **+0.20** | **+0.26** / 0 | **+0.79** / **+0.20** | **+0.65** / +0.06 | *+0.48 / +0.11* |
| 0.30 | **+0.12** / **+0.27** | **+0.19** / **+0.18** | **+0.37** / **+0.35** | **+0.37** / **+0.21** | *+0.26 / +0.25* |
| all |  |  |  |  | *+0.38 / +0.08* |

<details><summary>By rate, churn pooled (descriptive)</summary>

| Comparison | Rate | n (A / B) | L_D A → B | R_A A → B | Δ L_D (pooled mean) · cells supported | Δ R_A (pooled mean) · cells supported |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM | 0.01 | 40 / 40 | 0.66 → 2.25 | 0.70 → 1.01 | +1.59 · 4/4 (+) | +0.31 · 4/4 (+) |
| LSTM | 0.05 | 40 / 40 | 0.40 → 0.72 | 0.71 → 0.72 | +0.32 · 4/4 (+) | +0.01 · 3/4 (2+, 1−) |
| LSTM | 0.10 | 40 / 40 | 0.31 → 0.47 | 0.71 → 0.72 | +0.16 · 4/4 (+) | +0.01 · 2/4 (1+, 1−) |
| LSTM | 0.30 | 40 / 40 | 0.21 → 0.23 | 0.66 → 0.75 | +0.02 · 2/4 (+) | +0.09 · 3/4 (+) |
| LSTM | all | 160 / 160 | 0.39 → 0.92 | 0.70 → 0.80 | +0.52 · 14/16 (+) | +0.11 · 12/16 (10+, 2−) |
| Transformer | 0.01 | 40 / 40 | 0.66 → 0.91 | 0.70 → 0.62 | +0.25 · 3/4 (+) | −0.08 · 2/4 (−) |
| Transformer | 0.05 | 40 / 40 | 0.40 → 0.91 | 0.71 → 0.73 | +0.51 · 4/4 (+) | +0.02 · 2/4 (1+, 1−) |
| Transformer | 0.10 | 40 / 40 | 0.31 → 0.79 | 0.71 → 0.83 | +0.48 · 4/4 (+) | +0.11 · 2/4 (+) |
| Transformer | 0.30 | 40 / 40 | 0.21 → 0.47 | 0.66 → 0.91 | +0.26 · 4/4 (+) | +0.25 · 4/4 (+) |
| Transformer | all | 160 / 160 | 0.39 → 0.77 | 0.70 → 0.77 | +0.38 · 15/16 (+) | +0.08 · 10/16 (7+, 3−) |

</details>

<details><summary>Intervals per rate × churn cell</summary>

| Comparison | Rate | Churn | n | L_D A → B | R_A A → B | Δ L_D [95% CI] | Δ R_A [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM | 0.01 | 20% | 10 | 0.28 → 0.42 | 0.85 → 1.03 | **+0.14 [+0.08, +0.21]** | **+0.18 [+0.03, +0.34]** |
| LSTM | 0.01 | 40% | 10 | 0.55 → 1.09 | 0.77 → 1.03 | **+0.54 [+0.46, +0.61]** | **+0.26 [+0.19, +0.35]** |
| LSTM | 0.01 | 60% | 10 | 0.76 → 2.01 | 0.65 → 0.95 | **+1.25 [+0.89, +1.63]** | **+0.29 [+0.14, +0.46]** |
| LSTM | 0.01 | 80% | 10 | 1.04 → 5.48 | 0.51 → 1.03 | **+4.44 [+2.77, +6.23]** | **+0.52 [+0.27, +0.77]** |
| LSTM | 0.05 | 20% | 10 | 0.22 → 0.42 | 0.87 → 1.05 | **+0.20 [+0.18, +0.23]** | **+0.18 [+0.13, +0.23]** |
| LSTM | 0.05 | 40% | 10 | 0.39 → 0.73 | 0.77 → 0.80 | **+0.34 [+0.29, +0.38]** | +0.03 [−0.01, +0.07] |
| LSTM | 0.05 | 60% | 10 | 0.44 → 0.92 | 0.65 → 0.69 | **+0.48 [+0.38, +0.57]** | **+0.04 [0, +0.07]** |
| LSTM | 0.05 | 80% | 10 | 0.53 → 0.80 | 0.56 → 0.34 | **+0.27 [+0.17, +0.37]** | **−0.21 [−0.26, −0.17]** |
| LSTM | 0.10 | 20% | 10 | 0.18 → 0.32 | 0.84 → 0.97 | **+0.14 [+0.11, +0.18]** | **+0.12 [+0.05, +0.19]** |
| LSTM | 0.10 | 40% | 10 | 0.27 → 0.51 | 0.74 → 0.78 | **+0.23 [+0.17, +0.30]** | +0.04 [−0.02, +0.10] |
| LSTM | 0.10 | 60% | 10 | 0.35 → 0.53 | 0.67 → 0.67 | **+0.17 [+0.14, +0.20]** | −0.01 [−0.04, +0.02] |
| LSTM | 0.10 | 80% | 10 | 0.42 → 0.52 | 0.60 → 0.47 | **+0.10 [+0.01, +0.20]** | **−0.13 [−0.19, −0.06]** |
| LSTM | 0.30 | 20% | 10 | 0.11 → 0.16 | 0.74 → 0.93 | **+0.05 [+0.04, +0.06]** | **+0.19 [+0.17, +0.20]** |
| LSTM | 0.30 | 40% | 10 | 0.19 → 0.22 | 0.68 → 0.80 | **+0.03 [+0.02, +0.04]** | **+0.12 [+0.09, +0.15]** |
| LSTM | 0.30 | 60% | 10 | 0.25 → 0.26 | 0.63 → 0.68 | +0.01 [−0.02, +0.04] | **+0.05 [0, +0.09]** |
| LSTM | 0.30 | 80% | 10 | 0.29 → 0.27 | 0.58 → 0.59 | −0.02 [−0.06, +0.02] | +0.01 [−0.04, +0.07] |
| Transformer | 0.01 | 20% | 10 | 0.28 → 0.39 | 0.85 → 0.97 | **+0.11 [+0.04, +0.18]** | +0.12 [−0.06, +0.30] |
| Transformer | 0.01 | 40% | 10 | 0.55 → 0.63 | 0.77 → 0.61 | +0.08 [−0.10, +0.25] | −0.16 [−0.32, 0] |
| Transformer | 0.01 | 60% | 10 | 0.76 → 1.04 | 0.65 → 0.51 | **+0.28 [+0.01, +0.55]** | **−0.14 [−0.26, −0.02]** |
| Transformer | 0.01 | 80% | 10 | 1.04 → 1.59 | 0.51 → 0.37 | **+0.55 [+0.10, +1.10]** | **−0.13 [−0.25, −0.02]** |
| Transformer | 0.05 | 20% | 10 | 0.22 → 0.51 | 0.87 → 1.11 | **+0.29 [+0.20, +0.36]** | **+0.24 [+0.09, +0.36]** |
| Transformer | 0.05 | 40% | 10 | 0.39 → 0.75 | 0.77 → 0.79 | **+0.36 [+0.16, +0.59]** | +0.03 [−0.10, +0.17] |
| Transformer | 0.05 | 60% | 10 | 0.44 → 1.02 | 0.65 → 0.65 | **+0.58 [+0.35, +0.85]** | 0 [−0.09, +0.09] |
| Transformer | 0.05 | 80% | 10 | 0.53 → 1.36 | 0.56 → 0.37 | **+0.83 [+0.40, +1.31]** | **−0.19 [−0.30, −0.07]** |
| Transformer | 0.10 | 20% | 10 | 0.18 → 0.39 | 0.84 → 1.04 | **+0.21 [+0.15, +0.27]** | **+0.20 [+0.05, +0.35]** |
| Transformer | 0.10 | 40% | 10 | 0.27 → 0.54 | 0.74 → 0.74 | **+0.26 [+0.16, +0.36]** | 0 [−0.11, +0.09] |
| Transformer | 0.10 | 60% | 10 | 0.35 → 1.14 | 0.67 → 0.87 | **+0.79 [+0.49, +1.12]** | **+0.20 [+0.03, +0.36]** |
| Transformer | 0.10 | 80% | 10 | 0.42 → 1.07 | 0.60 → 0.65 | **+0.65 [+0.38, +0.93]** | +0.06 [−0.08, +0.18] |
| Transformer | 0.30 | 20% | 10 | 0.11 → 0.23 | 0.74 → 1.02 | **+0.12 [+0.09, +0.15]** | **+0.27 [+0.14, +0.42]** |
| Transformer | 0.30 | 40% | 10 | 0.19 → 0.37 | 0.68 → 0.86 | **+0.19 [+0.11, +0.27]** | **+0.18 [+0.03, +0.34]** |
| Transformer | 0.30 | 60% | 10 | 0.25 → 0.62 | 0.63 → 0.98 | **+0.37 [+0.22, +0.55]** | **+0.35 [+0.21, +0.49]** |
| Transformer | 0.30 | 80% | 10 | 0.29 → 0.66 | 0.58 → 0.79 | **+0.37 [+0.28, +0.45]** | **+0.21 [+0.13, +0.28]** |

</details>

### 10. A bigger hyperparameter search helps

**Verdict: partly.** From the archived 10-trial (LSTM) and 20-trial (Transformer) `no_ar`
runs on the same panels to the 100-trial runs, the level improves most consistently at
rate 0.30: in three of four cells for the LSTM (churn 40–80%, −19 to −42 MAPE) and two for
the Transformer (churn 40–60%, −22 and −39). Below rate 0.30 the level changes clearly in
only a few scattered cells: two for the LSTM (both better) and four for the Transformer
(three better, one worse: rate 0.01, churn 60%, +38). The ranking improves in 11 of 16
cells for the Transformer, every cell at rate 0.30, three of four at rates 0.05 and 0.10
and one at rate 0.01 (+0.03 to +0.22), and in 5 for the LSTM (all four at rate 0.30 and one
at rate 0.10, by 0.01–0.03). The archived
run did not record its embedder, so the two runs may also differ in that.

**LSTM, 10 → 100 trials**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | −3 / −4 / 0 | **−49** / **−53** / +0.01 | −32 / −32 / 0 | +47 / +45 / −0.01 | *−9 / −11 / 0* |
| 0.05 | −10 / −9 / −0.13 | +6 / +6 / −0.02 | **−32** / **−33** / −0.06 | +79 / +79 / 0 | *+11 / +11 / −0.05* |
| 0.10 | −7 / −5 / **+0.03** | −11 / −12 / −0.02 | −32 / **−39** / 0 | −46 / −52 / −0.04 | *−24 / −27 / −0.01* |
| 0.30 | −3 / −2 / **+0.03** | **−19** / **−23** / **+0.02** | **−42** / **−46** / **+0.01** | **−30** / **−38** / **+0.01** | *−23 / −27 / +0.02* |
| all |  |  |  |  | *−12 / −14 / −0.01* |

**Transformer, 20 → 100 trials**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **−14** / **−23** / +0.02 | +12 / +9 / −0.02 | **+38** / **+55** / +0.02 | −12 / −24 / **+0.05** | *+6 / +4 / +0.02* |
| 0.05 | −1 / −2 / +0.12 | **−41** / **−42** / **+0.22** | −19 / −23 / **+0.14** | **−143** / **−149** / **+0.22** | *−51 / −54 / +0.17* |
| 0.10 | −19 / −20 / +0.03 | −7 / −8 / **+0.05** | −24 / −25 / **+0.08** | −20 / −19 / **+0.04** | *−17 / −18 / +0.05* |
| 0.30 | +2 / −6 / **+0.04** | **−22** / **−26** / **+0.04** | **−39** / **−45** / **+0.03** | −69 / −72 / **+0.03** | *−32 / −37 / +0.04* |
| all |  |  |  |  | *−24 / −26 / +0.07* |

<details><summary>By rate, churn pooled (descriptive)</summary>

| Comparison | Rate | n (A / B) | MAPE A → B | \|bias\| A → B | Δ MAPE (pooled mean) · cells supported | Δ \|bias\| (pooled mean) · cells supported | Δ RMSE (pooled mean) · cells supported | Δ Spearman (pooled mean) · cells supported |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM, 10 → 100 trials | 0.01 | 40 / 40 | 324 → 314 | 320 → 309 | −9 · 1/4 (−) | −11 · 1/4 (−) | −0.03 · 1/4 (−) | 0 · 0/4 |
| LSTM, 10 → 100 trials | 0.05 | 40 / 40 | 189 → 200 | 188 → 199 | +11 · 1/4 (−) | +11 · 1/4 (−) | −0.02 · 1/4 (−) | −0.05 · 0/4 |
| LSTM, 10 → 100 trials | 0.10 | 40 / 40 | 122 → 98 | 121 → 93 | −24 · 0/4 | −27 · 1/4 (−) | −0.27 · 2/4 (−) | −0.01 · 1/4 (+) |
| LSTM, 10 → 100 trials | 0.30 | 40 / 40 | 44 → 21 | 40 → 13 | −23 · 3/4 (−) | −27 · 3/4 (−) | −0.92 · 4/4 (−) | +0.02 · 4/4 (+) |
| LSTM, 10 → 100 trials | all | 160 / 160 | 170 → 158 | 167 → 154 | −12 · 5/16 (−) | −14 · 6/16 (−) | −0.31 · 8/16 (−) | −0.01 · 5/16 (+) |
| Transformer, 20 → 100 trials | 0.01 | 40 / 40 | 114 → 120 | 80 → 84 | +6 · 2/4 (1+, 1−) | +4 · 2/4 (1+, 1−) | +0.01 · 1/4 (+) | +0.02 · 1/4 (+) |
| Transformer, 20 → 100 trials | 0.05 | 40 / 40 | 165 → 114 | 163 → 109 | −51 · 2/4 (−) | −54 · 2/4 (−) | −0.20 · 2/4 (−) | +0.17 · 3/4 (+) |
| Transformer, 20 → 100 trials | 0.10 | 40 / 40 | 147 → 130 | 146 → 128 | −17 · 0/4 | −18 · 0/4 | −0.26 · 1/4 (−) | +0.05 · 3/4 (+) |
| Transformer, 20 → 100 trials | 0.30 | 40 / 40 | 88 → 56 | 85 → 48 | −32 · 2/4 (−) | −37 · 2/4 (−) | −0.71 · 3/4 (−) | +0.04 · 4/4 (+) |
| Transformer, 20 → 100 trials | all | 160 / 160 | 129 → 105 | 118 → 92 | −24 · 6/16 (1+, 5−) | −26 · 6/16 (1+, 5−) | −0.29 · 7/16 (1+, 6−) | +0.07 · 11/16 (+) |

</details>

<details><summary>Intervals per rate × churn cell</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI] | Δ \|bias\| [95% CI] | Δ RMSE [95% CI] | Δ Spearman [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM, 10 → 100 trials | 0.01 | 20% | 10 | 74 → 71 | 63 → 59 | −3 [−20, +15] | −4 [−27, +20] | −0.01 [−0.05, +0.02] | 0 [−0.01, +0.01] |
| LSTM, 10 → 100 trials | 0.01 | 40% | 10 | 186 → 136 | 185 → 132 | **−49 [−73, −18]** | **−53 [−78, −20]** | **−0.11 [−0.15, −0.05]** | +0.01 [−0.01, +0.03] |
| LSTM, 10 → 100 trials | 0.01 | 60% | 10 | 323 → 291 | 322 → 290 | −32 [−68, +8] | −32 [−69, +8] | −0.06 [−0.11, 0] | 0 [−0.02, +0.03] |
| LSTM, 10 → 100 trials | 0.01 | 80% | 10 | 712 → 759 | 710 → 755 | +47 [−45, +137] | +45 [−49, +139] | +0.04 [−0.05, +0.13] | −0.01 [−0.03, +0.01] |
| LSTM, 10 → 100 trials | 0.05 | 20% | 10 | 55 → 45 | 51 → 42 | −10 [−33, +9] | −9 [−33, +13] | −0.15 [−0.40, +0.05] | −0.13 [−0.29, 0] |
| LSTM, 10 → 100 trials | 0.05 | 40% | 10 | 93 → 99 | 93 → 98 | +6 [−14, +27] | +6 [−15, +28] | −0.02 [−0.16, +0.12] | −0.02 [−0.18, +0.13] |
| LSTM, 10 → 100 trials | 0.05 | 60% | 10 | 190 → 158 | 190 → 157 | **−32 [−54, −12]** | **−33 [−55, −12]** | **−0.10 [−0.19, −0.01]** | −0.06 [−0.16, +0.01] |
| LSTM, 10 → 100 trials | 0.05 | 80% | 10 | 418 → 497 | 418 → 497 | +79 [−21, +182] | +79 [−21, +182] | +0.19 [−0.05, +0.42] | 0 [−0.02, +0.02] |
| LSTM, 10 → 100 trials | 0.10 | 20% | 10 | 40 → 33 | 37 → 31 | −7 [−20, +6] | −5 [−20, +9] | **−0.52 [−0.63, −0.41]** | **+0.03 [+0.01, +0.05]** |
| LSTM, 10 → 100 trials | 0.10 | 40% | 10 | 69 → 58 | 68 → 57 | −11 [−39, +20] | −12 [−40, +20] | −0.19 [−0.63, +0.37] | −0.02 [−0.08, +0.03] |
| LSTM, 10 → 100 trials | 0.10 | 60% | 10 | 101 → 69 | 100 → 61 | −32 [−63, +2] | **−39 [−75, −2]** | **−0.18 [−0.35, −0.01]** | 0 [−0.02, +0.02] |
| LSTM, 10 → 100 trials | 0.10 | 80% | 10 | 279 → 233 | 277 → 225 | −46 [−184, +87] | −52 [−195, +84] | −0.20 [−0.74, +0.29] | −0.04 [−0.26, +0.19] |
| LSTM, 10 → 100 trials | 0.30 | 20% | 10 | 22 → 19 | 20 → 17 | −3 [−12, +5] | −2 [−14, +8] | **−0.60 [−0.82, −0.39]** | **+0.03 [+0.02, +0.04]** |
| LSTM, 10 → 100 trials | 0.30 | 40% | 10 | 33 → 14 | 32 → 9 | **−19 [−30, −10]** | **−23 [−36, −14]** | **−1.58 [−1.96, −1.21]** | **+0.02 [+0.01, +0.03]** |
| LSTM, 10 → 100 trials | 0.30 | 60% | 10 | 60 → 18 | 55 → 9 | **−42 [−77, −12]** | **−46 [−83, −14]** | **−0.95 [−1.45, −0.47]** | **+0.01 [+0.01, +0.01]** |
| LSTM, 10 → 100 trials | 0.30 | 80% | 10 | 62 → 32 | 55 → 17 | **−30 [−78, 0]** | **−38 [−93, −2]** | **−0.55 [−0.90, −0.19]** | **+0.01 [0, +0.01]** |
| Transformer, 20 → 100 trials | 0.01 | 20% | 10 | 72 → 58 | 60 → 37 | **−14 [−27, 0]** | **−23 [−45, −2]** | −0.03 [−0.06, 0] | +0.02 [−0.01, +0.04] |
| Transformer, 20 → 100 trials | 0.01 | 40% | 10 | 84 → 96 | 67 → 76 | +12 [−20, +48] | +9 [−33, +55] | +0.03 [−0.02, +0.09] | −0.02 [−0.05, +0.03] |
| Transformer, 20 → 100 trials | 0.01 | 60% | 10 | 92 → 129 | 47 → 102 | **+38 [+4, +79]** | **+55 [+8, +109]** | **+0.04 [+0.01, +0.08]** | +0.02 [−0.02, +0.06] |
| Transformer, 20 → 100 trials | 0.01 | 80% | 10 | 209 → 197 | 145 → 121 | −12 [−43, +17] | −24 [−79, +28] | −0.01 [−0.03, +0.01] | **+0.05 [0, +0.10]** |
| Transformer, 20 → 100 trials | 0.05 | 20% | 10 | 76 → 75 | 74 → 72 | −1 [−18, +14] | −2 [−24, +16] | −0.04 [−0.23, +0.14] | +0.12 [−0.01, +0.25] |
| Transformer, 20 → 100 trials | 0.05 | 40% | 10 | 151 → 110 | 151 → 108 | **−41 [−72, −7]** | **−42 [−74, −7]** | **−0.37 [−0.61, −0.11]** | **+0.22 [+0.11, +0.32]** |
| Transformer, 20 → 100 trials | 0.05 | 60% | 10 | 168 → 149 | 167 → 144 | −19 [−91, +40] | −23 [−101, +39] | −0.12 [−0.46, +0.14] | **+0.14 [+0.06, +0.23]** |
| Transformer, 20 → 100 trials | 0.05 | 80% | 10 | 265 → 122 | 260 → 111 | **−143 [−276, −20]** | **−149 [−289, −18]** | **−0.29 [−0.51, −0.09]** | **+0.22 [+0.12, +0.31]** |
| Transformer, 20 → 100 trials | 0.10 | 20% | 10 | 77 → 58 | 76 → 56 | −19 [−45, +6] | −20 [−48, +6] | **−0.47 [−1.05, −0.03]** | +0.03 [−0.01, +0.07] |
| Transformer, 20 → 100 trials | 0.10 | 40% | 10 | 100 → 94 | 99 → 91 | −7 [−43, +28] | −8 [−48, +29] | −0.24 [−0.76, +0.21] | **+0.05 [+0.02, +0.07]** |
| Transformer, 20 → 100 trials | 0.10 | 60% | 10 | 127 → 103 | 124 → 99 | −24 [−71, +25] | −25 [−76, +26] | −0.26 [−0.68, +0.10] | **+0.08 [+0.03, +0.16]** |
| Transformer, 20 → 100 trials | 0.10 | 80% | 10 | 285 → 265 | 284 → 265 | −20 [−137, +87] | −19 [−138, +89] | −0.04 [−0.48, +0.33] | **+0.04 [+0.02, +0.06]** |
| Transformer, 20 → 100 trials | 0.30 | 20% | 10 | 34 → 36 | 31 → 24 | +2 [−9, +14] | −6 [−22, +11] | −0.43 [−1.15, +0.39] | **+0.04 [0, +0.10]** |
| Transformer, 20 → 100 trials | 0.30 | 40% | 10 | 58 → 36 | 55 → 29 | **−22 [−37, −7]** | **−26 [−46, −7]** | **−0.60 [−1.16, −0.01]** | **+0.04 [+0.02, +0.06]** |
| Transformer, 20 → 100 trials | 0.30 | 60% | 10 | 96 → 57 | 94 → 50 | **−39 [−69, −10]** | **−45 [−77, −12]** | **−1.00 [−1.81, −0.23]** | **+0.03 [+0.01, +0.05]** |
| Transformer, 20 → 100 trials | 0.30 | 80% | 10 | 166 → 96 | 160 → 88 | −69 [−134, +1] | −72 [−141, +2] | **−0.83 [−1.48, −0.08]** | **+0.03 [+0.01, +0.06]** |

</details>

### 11. Specific hyperparameters drive the error

**Verdict: not supported; described only, no test.** A correlation between a chosen
hyperparameter and the error, over a cell's 10 panels, is not a difference of means, so
the protocol's interval cannot be put on it; and pooling the 16 cells into one test is not
allowed. The table therefore describes the correlations. Pooled over the grid, they mix
cells, and there the rate and churn choose both the hyperparameter and the error: the LSTM
`no_ar` batch size correlates with |bias| at +0.48 over the grid, but within cells it is
positive in 5 of 12 cells and negative in 7. Within cells almost every hyperparameter
splits roughly evenly between positive and negative, with cell values spread from about
−0.7 to +0.7. The one consistent pattern is the Transformer `no_ar` layer count: positive
in 13 of 16 cells against both |bias| and MAPE (more encoder layers with a worse level).
The LSTM `ar_bounded` dense width leans negative (10 of 15 cells). Neither pattern is a
tested effect.

| Tree | Hyperparameter | ρ with \|bias\|, pooled | ρ with \|bias\| within cells: positive / negative of cells, range | ρ with MAPE, pooled | ρ with MAPE within cells: positive / negative of cells, range |
| --- | --- | --- | --- | --- | --- |
| LSTM `no_ar` | `batch_size` | +0.48 | 5 / 7 of 12, −0.63 to +0.62 | +0.49 | 5 / 5 of 12, −0.60 to +0.62 |
| LSTM `no_ar` | `learning_rate` | −0.15 | 9 / 7 of 16, −0.61 to +0.93 | −0.16 | 8 / 8 of 16, −0.61 to +0.87 |
| LSTM `no_ar` | `lstm_hidden_size` | −0.20 | 9 / 6 of 15, −0.66 to +0.64 | −0.21 | 9 / 6 of 15, −0.66 to +0.57 |
| LSTM `no_ar` | `dense_units` | +0.28 | 8 / 7 of 15, −0.73 to +0.57 | +0.28 | 8 / 7 of 15, −0.78 to +0.43 |
| LSTM `no_ar` | `dropout` | −0.05 | 12 / 4 of 16, −0.76 to +0.66 | −0.05 | 11 / 5 of 16, −0.76 to +0.66 |
| LSTM `ar_bounded` | `batch_size` | +0.07 | 7 / 6 of 13, −0.43 to +0.59 | +0.06 | 6 / 6 of 13, −0.32 to +0.71 |
| LSTM `ar_bounded` | `learning_rate` | −0.02 | 10 / 6 of 16, −0.44 to +0.47 | −0.10 | 10 / 6 of 16, −0.39 to +0.32 |
| LSTM `ar_bounded` | `lstm_hidden_size` | 0 | 9 / 7 of 16, −0.69 to +0.54 | −0.04 | 9 / 7 of 16, −0.45 to +0.62 |
| LSTM `ar_bounded` | `dense_units` | −0.28 | 5 / 10 of 15, −0.85 to +0.51 | −0.22 | 5 / 10 of 15, −0.62 to +0.49 |
| LSTM `ar_bounded` | `dropout` | −0.02 | 12 / 4 of 16, −0.61 to +0.59 | +0.03 | 11 / 5 of 16, −0.61 to +0.56 |
| Transformer `no_ar` | `batch_size` | −0.04 | 5 / 8 of 13, −0.50 to +0.70 | −0.03 | 5 / 8 of 13, −0.52 to +0.70 |
| Transformer `no_ar` | `learning_rate` | −0.09 | 8 / 8 of 16, −0.54 to +0.54 | +0.05 | 8 / 8 of 16, −0.54 to +0.57 |
| Transformer `no_ar` | `d_model` | −0.01 | 1 / 1 of 2, −0.17 to +0.17 | −0.03 | 1 / 1 of 2, −0.06 to +0.09 |
| Transformer `no_ar` | `nhead` | +0.08 | 10 / 6 of 16, −0.76 to +0.66 | +0.05 | 10 / 6 of 16, −0.72 to +0.78 |
| Transformer `no_ar` | `num_encoder_layers` | +0.24 | 13 / 3 of 16, −0.29 to +0.57 | +0.29 | 13 / 2 of 16, −0.05 to +0.57 |
| Transformer `no_ar` | `dropout` | +0.17 | 10 / 6 of 16, −0.35 to +0.60 | +0.13 | 10 / 6 of 16, −0.44 to +0.65 |
| Transformer `ar_bounded` | `batch_size` | +0.03 | 11 / 5 of 16, −0.61 to +0.57 | +0.04 | 11 / 5 of 16, −0.61 to +0.53 |
| Transformer `ar_bounded` | `learning_rate` | −0.05 | 4 / 12 of 16, −0.75 to +0.52 | +0.13 | 7 / 9 of 16, −0.70 to +0.58 |
| Transformer `ar_bounded` | `d_model` | −0.13 | 0 / 1 of 1, −0.52 to −0.52 | −0.13 | 0 / 1 of 1, −0.52 to −0.52 |
| Transformer `ar_bounded` | `nhead` | −0.05 | 8 / 7 of 16, −0.76 to +0.44 | −0.01 | 8 / 7 of 16, −0.76 to +0.32 |
| Transformer `ar_bounded` | `num_encoder_layers` | −0.01 | 8 / 8 of 16, −0.54 to +0.80 | −0.10 | 9 / 7 of 16, −0.49 to +0.79 |
| Transformer `ar_bounded` | `dropout` | −0.01 | 8 / 8 of 16, −0.70 to +0.62 | +0.03 | 7 / 9 of 16, −0.66 to +0.68 |

### 12. More customers improve the neural forecast

**Verdict: supported, conditionally.** Tripling the cohort (1,000 → 3,000 customers).
Panel j of the two grids shares its seed, so its first 1,000 customers share their
purchase-rate draw λ; but the generator draws all N λ's before any dropout rate μ, so the
μ's, the death weeks and every count come from a different part of the random stream
(checked: the μ's of the first 1,000 customers correlate at −0.01). The two grids are
therefore different customer populations, and each cell is compared as 10 vs 10
independent replications, not paired.

- **LSTM `ar_bounded`:** MAPE improves in every cell at rates 0.01–0.10 (−13 to −439
  points), and at rate 0.30 at churn 20–40% (−5, −3). Ranking improves in every cell at rate
  0.01 (+0.07 to +0.17) and in one cell at each other rate.
- **LSTM `no_ar`:** MAPE improves in every cell at rates 0.05–0.10 and at rate 0.30 with
  churn 20–60%. At rate 0.01 it is mixed: better at churn 80% (−360), worse at churn 20%
  (+13), no clear difference at churn 40–60%. Without flags, extra customers do not
  reliably help the sparsest panels.
- **Pareto/NBD:** MAPE improves in 14 of 16 cells, modestly (−1 to −81 points, most at high
  churn); there is no clear RMSE difference in any cell.
- The Transformer was not run at 3,000 customers.

**LSTM `no_ar`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **+13** / **+24** / −0.02 | −5 / −1 / −0.02 | −30 / −28 / −0.01 | **−360** / **−356** / +0.03 | *−95 / −90 / 0* |
| 0.05 | **−18** / **−17** / **+0.22** | **−30** / **−29** / **+0.28** | **−52** / **−55** / **+0.31** | **−221** / **−223** / **+0.10** | *−80 / −81 / +0.23* |
| 0.10 | **−15** / **−14** / +0.01 | **−29** / **−29** / +0.05 | **−52** / **−53** / **+0.03** | **−201** / **−203** / **+0.21** | *−74 / −75 / +0.08* |
| 0.30 | **−12** / **−12** / **+0.01** | **−6** / **−5** / +0.01 | **−6** / −2 / −0.01 | −4 / +6 / 0 | *−7 / −3 / 0* |
| all |  |  |  |  | *−64 / −62 / +0.08* |

**LSTM `ar_bounded`**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **−27** / **−22** / **+0.07** | **−69** / **−78** / **+0.17** | **−107** / **−117** / **+0.13** | **−439** / **−449** / **+0.14** | *−160 / −166 / +0.13* |
| 0.05 | **−22** / **−22** / **+0.04** | **−26** / **−27** / +0.01 | **−34** / **−39** / +0.01 | **−23** / −11 / +0.02 | *−26 / −25 / +0.02* |
| 0.10 | **−19** / **−19** / **+0.02** | **−23** / **−23** / +0.01 | **−16** / **−14** / +0.01 | **−13** / −2 / +0.01 | *−18 / −14 / +0.01* |
| 0.30 | **−5** / **−6** / **+0.01** | **−3** / +1 / +0.01 | −2 / +3 / −0.01 | −5 / +5 / +0.01 | *−4 / +1 / 0* |
| all |  |  |  |  | *−52 / −51 / +0.04* |

**Pareto/NBD**: Δ MAPE / Δ \|bias\| / Δ Spearman

| Rate | Churn 20% | Churn 40% | Churn 60% | Churn 80% | Churn pooled (descriptive) |
| --- | --- | --- | --- | --- | --- |
| 0.01 | **−11** / −1 / −0.01 | **−19** / **−13** / **+0.03** | **−26** / −11 / +0.02 | **−81** / **−32** / **+0.09** | *−34 / −14 / +0.03* |
| 0.05 | **−4** / −1 / 0 | **−6** / **−5** / 0 | **−9** / −5 / 0 | **−19** / **−12** / +0.02 | *−9 / −6 / +0.01* |
| 0.10 | **−1** / −1 / 0 | **−2** / **−3** / −0.01 | **−5** / **−6** / 0 | **−14** / **−13** / +0.01 | *−5 / −6 / 0* |
| 0.30 | 0 / +1 / 0 | −1 / 0 / 0 | **−2** / 0 / 0 | **−4** / −1 / 0 | *−2 / 0 / 0* |
| all |  |  |  |  | *−13 / −7 / +0.01* |

<details><summary>By rate, churn pooled (descriptive)</summary>

| Comparison | Rate | n (A / B) | MAPE A → B | \|bias\| A → B | Δ MAPE (pooled mean) · cells supported | Δ \|bias\| (pooled mean) · cells supported | Δ RMSE (pooled mean) · cells supported | Δ Spearman (pooled mean) · cells supported |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM `no_ar` | 0.01 | 40 / 40 | 314 → 219 | 309 → 219 | −95 · 2/4 (1+, 1−) | −90 · 2/4 (1+, 1−) | −0.06 · 1/4 (−) | 0 · 0/4 |
| LSTM `no_ar` | 0.05 | 40 / 40 | 200 → 119 | 199 → 118 | −80 · 4/4 (−) | −81 · 4/4 (−) | −0.32 · 4/4 (−) | +0.23 · 4/4 (+) |
| LSTM `no_ar` | 0.10 | 40 / 40 | 98 → 24 | 93 → 19 | −74 · 4/4 (−) | −75 · 4/4 (−) | −0.36 · 4/4 (−) | +0.08 · 2/4 (+) |
| LSTM `no_ar` | 0.30 | 40 / 40 | 21 → 14 | 13 → 10 | −7 · 3/4 (−) | −3 · 2/4 (−) | −0.34 · 3/4 (−) | 0 · 1/4 (+) |
| LSTM `no_ar` | all | 160 / 160 | 158 → 94 | 154 → 91 | −64 · 13/16 (1+, 12−) | −62 · 12/16 (1+, 11−) | −0.27 · 12/16 (−) | +0.08 · 7/16 (+) |
| LSTM `ar_bounded` | 0.01 | 40 / 40 | 243 → 83 | 232 → 65 | −160 · 4/4 (−) | −166 · 4/4 (−) | −0.12 · 4/4 (−) | +0.13 · 4/4 (+) |
| LSTM `ar_bounded` | 0.05 | 40 / 40 | 58 → 31 | 46 → 22 | −26 · 4/4 (−) | −25 · 3/4 (−) | −0.11 · 2/4 (−) | +0.02 · 1/4 (+) |
| LSTM `ar_bounded` | 0.10 | 40 / 40 | 36 → 18 | 24 → 9 | −18 · 4/4 (−) | −14 · 3/4 (−) | −0.17 · 2/4 (−) | +0.01 · 1/4 (+) |
| LSTM `ar_bounded` | 0.30 | 40 / 40 | 20 → 16 | 11 → 12 | −4 · 2/4 (−) | +1 · 1/4 (−) | −0.17 · 2/4 (−) | 0 · 1/4 (+) |
| LSTM `ar_bounded` | all | 160 / 160 | 89 → 37 | 78 → 27 | −52 · 14/16 (−) | −51 · 11/16 (−) | −0.14 · 10/16 (−) | +0.04 · 7/16 (+) |
| Pareto/NBD | 0.01 | 40 / 40 | 90 → 56 | 35 → 21 | −34 · 4/4 (−) | −14 · 2/4 (−) | −0.01 · 0/4 | +0.03 · 2/4 (+) |
| Pareto/NBD | 0.05 | 40 / 40 | 44 → 34 | 14 → 8 | −9 · 4/4 (−) | −6 · 2/4 (−) | 0 · 0/4 | +0.01 · 0/4 |
| Pareto/NBD | 0.10 | 40 / 40 | 37 → 32 | 9 → 4 | −5 · 4/4 (−) | −6 · 3/4 (−) | −0.02 · 0/4 | 0 · 0/4 |
| Pareto/NBD | 0.30 | 40 / 40 | 32 → 30 | 13 → 13 | −2 · 2/4 (−) | 0 · 0/4 | −0.05 · 0/4 | 0 · 0/4 |
| Pareto/NBD | all | 160 / 160 | 51 → 38 | 18 → 12 | −13 · 14/16 (−) | −7 · 7/16 (−) | −0.02 · 0/16 | +0.01 · 2/16 (+) |

</details>

<details><summary>Intervals per rate × churn cell</summary>

| Comparison | Rate | Churn | n | MAPE A → B | \|bias\| A → B | Δ MAPE [95% CI] | Δ \|bias\| [95% CI] | Δ RMSE [95% CI] | Δ Spearman [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LSTM `no_ar` | 0.01 | 20% | 10 / 10 | 71 → 84 | 59 → 83 | **+13 [+1, +25]** | **+24 [+8, +39]** | +0.02 [−0.03, +0.06] | −0.02 [−0.04, +0.02] |
| LSTM `no_ar` | 0.01 | 40% | 10 / 10 | 136 → 132 | 132 → 131 | −5 [−25, +15] | −1 [−23, +20] | +0.03 [−0.01, +0.07] | −0.02 [−0.05, +0.02] |
| LSTM `no_ar` | 0.01 | 60% | 10 / 10 | 291 → 261 | 290 → 261 | −30 [−70, +12] | −28 [−69, +14] | −0.06 [−0.12, 0] | −0.01 [−0.06, +0.04] |
| LSTM `no_ar` | 0.01 | 80% | 10 / 10 | 759 → 399 | 755 → 399 | **−360 [−612, −132]** | **−356 [−611, −125]** | **−0.22 [−0.29, −0.14]** | +0.03 [−0.02, +0.08] |
| LSTM `no_ar` | 0.05 | 20% | 10 / 10 | 45 → 26 | 42 → 25 | **−18 [−27, −9]** | **−17 [−27, −6]** | **−0.29 [−0.43, −0.15]** | **+0.22 [+0.07, +0.38]** |
| LSTM `no_ar` | 0.05 | 40% | 10 / 10 | 99 → 69 | 98 → 69 | **−30 [−47, −13]** | **−29 [−46, −12]** | **−0.29 [−0.40, −0.18]** | **+0.28 [+0.13, +0.42]** |
| LSTM `no_ar` | 0.05 | 60% | 10 / 10 | 158 → 105 | 157 → 102 | **−52 [−90, −16]** | **−55 [−96, −16]** | **−0.27 [−0.45, −0.10]** | **+0.31 [+0.17, +0.43]** |
| LSTM `no_ar` | 0.05 | 80% | 10 / 10 | 497 → 277 | 497 → 274 | **−221 [−316, −130]** | **−223 [−322, −130]** | **−0.42 [−0.61, −0.24]** | **+0.10 [+0.02, +0.21]** |
| LSTM `no_ar` | 0.10 | 20% | 10 / 10 | 33 → 18 | 31 → 17 | **−15 [−23, −8]** | **−14 [−22, −7]** | **−0.22 [−0.35, −0.09]** | +0.01 [0, +0.03] |
| LSTM `no_ar` | 0.10 | 40% | 10 / 10 | 58 → 29 | 57 → 27 | **−29 [−62, −8]** | **−29 [−62, −7]** | **−0.41 [−0.92, −0.06]** | +0.05 [0, +0.12] |
| LSTM `no_ar` | 0.10 | 60% | 10 / 10 | 69 → 17 | 61 → 8 | **−52 [−76, −30]** | **−53 [−81, −26]** | **−0.28 [−0.47, −0.12]** | **+0.03 [+0.02, +0.05]** |
| LSTM `no_ar` | 0.10 | 80% | 10 / 10 | 233 → 32 | 225 → 22 | **−201 [−333, −96]** | **−203 [−339, −91]** | **−0.55 [−0.85, −0.25]** | **+0.21 [+0.09, +0.33]** |
| LSTM `no_ar` | 0.30 | 20% | 10 / 10 | 19 → 8 | 17 → 6 | **−12 [−15, −8]** | **−12 [−17, −6]** | **−0.72 [−1.03, −0.42]** | **+0.01 [0, +0.02]** |
| LSTM `no_ar` | 0.30 | 40% | 10 / 10 | 14 → 9 | 9 → 4 | **−6 [−8, −4]** | **−5 [−8, −1]** | **−0.28 [−0.51, −0.04]** | +0.01 [0, +0.02] |
| LSTM `no_ar` | 0.30 | 60% | 10 / 10 | 18 → 12 | 9 → 7 | **−6 [−9, −3]** | −2 [−7, +2] | **−0.29 [−0.55, −0.05]** | −0.01 [−0.02, +0.01] |
| LSTM `no_ar` | 0.30 | 80% | 10 / 10 | 32 → 28 | 17 → 23 | −4 [−9, +2] | +6 [−1, +14] | −0.06 [−0.35, +0.22] | 0 [−0.02, +0.02] |
| LSTM `ar_bounded` | 0.01 | 20% | 10 / 10 | 65 → 38 | 47 → 25 | **−27 [−43, −13]** | **−22 [−45, −1]** | **−0.05 [−0.11, −0.01]** | **+0.07 [+0.01, +0.13]** |
| LSTM `ar_bounded` | 0.01 | 40% | 10 / 10 | 121 → 52 | 114 → 37 | **−69 [−84, −55]** | **−78 [−97, −60]** | **−0.07 [−0.11, −0.03]** | **+0.17 [+0.11, +0.23]** |
| LSTM `ar_bounded` | 0.01 | 60% | 10 / 10 | 196 → 89 | 190 → 72 | **−107 [−153, −66]** | **−117 [−168, −72]** | **−0.13 [−0.19, −0.07]** | **+0.13 [+0.04, +0.22]** |
| LSTM `ar_bounded` | 0.01 | 80% | 10 / 10 | 591 → 152 | 576 → 127 | **−439 [−692, −214]** | **−449 [−710, −210]** | **−0.24 [−0.34, −0.14]** | **+0.14 [+0.06, +0.21]** |
| LSTM `ar_bounded` | 0.05 | 20% | 10 / 10 | 48 → 26 | 47 → 24 | **−22 [−31, −13]** | **−22 [−32, −12]** | **−0.29 [−0.34, −0.23]** | **+0.04 [0, +0.07]** |
| LSTM `ar_bounded` | 0.05 | 40% | 10 / 10 | 56 → 29 | 52 → 25 | **−26 [−37, −15]** | **−27 [−40, −13]** | **−0.13 [−0.19, −0.06]** | +0.01 [−0.02, +0.04] |
| LSTM `ar_bounded` | 0.05 | 60% | 10 / 10 | 67 → 33 | 61 → 22 | **−34 [−47, −20]** | **−39 [−57, −18]** | −0.06 [−0.15, +0.02] | +0.01 [−0.01, +0.03] |
| LSTM `ar_bounded` | 0.05 | 80% | 10 / 10 | 60 → 37 | 27 → 16 | **−23 [−32, −15]** | −11 [−27, +4] | +0.02 [−0.06, +0.09] | +0.02 [−0.01, +0.05] |
| LSTM `ar_bounded` | 0.10 | 20% | 10 / 10 | 32 → 14 | 30 → 11 | **−19 [−28, −10]** | **−19 [−30, −7]** | **−0.36 [−0.49, −0.23]** | **+0.02 [0, +0.03]** |
| LSTM `ar_bounded` | 0.10 | 40% | 10 / 10 | 34 → 12 | 28 → 6 | **−23 [−31, −15]** | **−23 [−34, −12]** | **−0.24 [−0.35, −0.12]** | +0.01 [0, +0.03] |
| LSTM `ar_bounded` | 0.10 | 60% | 10 / 10 | 33 → 16 | 21 → 7 | **−16 [−21, −12]** | **−14 [−21, −7]** | −0.04 [−0.13, +0.04] | +0.01 [0, +0.02] |
| LSTM `ar_bounded` | 0.10 | 80% | 10 / 10 | 45 → 32 | 15 → 14 | **−13 [−19, −8]** | −2 [−10, +6] | −0.04 [−0.18, +0.11] | +0.01 [−0.01, +0.04] |
| LSTM `ar_bounded` | 0.30 | 20% | 10 / 10 | 12 → 6 | 10 → 3 | **−5 [−7, −4]** | **−6 [−9, −3]** | **−0.43 [−0.72, −0.14]** | **+0.01 [0, +0.02]** |
| LSTM `ar_bounded` | 0.30 | 40% | 10 / 10 | 13 → 10 | 6 → 7 | **−3 [−5, −1]** | +1 [−2, +4] | **−0.26 [−0.48, −0.04]** | +0.01 [0, +0.02] |
| LSTM `ar_bounded` | 0.30 | 60% | 10 / 10 | 19 → 18 | 10 → 13 | −2 [−5, +2] | +3 [−3, +9] | −0.14 [−0.34, +0.05] | −0.01 [−0.02, +0.01] |
| LSTM `ar_bounded` | 0.30 | 80% | 10 / 10 | 35 → 31 | 19 → 24 | −5 [−10, 0] | +5 [−4, +14] | +0.15 [−0.08, +0.37] | +0.01 [−0.02, +0.03] |
| Pareto/NBD | 0.01 | 20% | 10 / 10 | 48 → 37 | 12 → 11 | **−11 [−15, −7]** | −1 [−6, +3] | −0.01 [−0.05, +0.02] | −0.01 [−0.04, +0.02] |
| Pareto/NBD | 0.01 | 40% | 10 / 10 | 66 → 46 | 34 → 22 | **−19 [−24, −15]** | **−13 [−20, −4]** | 0 [−0.03, +0.03] | **+0.03 [0, +0.06]** |
| Pareto/NBD | 0.01 | 60% | 10 / 10 | 88 → 62 | 41 → 30 | **−26 [−32, −20]** | −11 [−24, +2] | −0.02 [−0.06, +0.01] | +0.02 [−0.02, +0.06] |
| Pareto/NBD | 0.01 | 80% | 10 / 10 | 159 → 78 | 54 → 22 | **−81 [−104, −57]** | **−32 [−55, −11]** | −0.01 [−0.05, +0.04] | **+0.09 [+0.05, +0.13]** |
| Pareto/NBD | 0.05 | 20% | 10 / 10 | 35 → 32 | 8 → 7 | **−4 [−5, −2]** | −1 [−4, +2] | −0.03 [−0.07, 0] | 0 [−0.02, +0.02] |
| Pareto/NBD | 0.05 | 40% | 10 / 10 | 39 → 33 | 15 → 10 | **−6 [−8, −4]** | **−5 [−10, −1]** | +0.01 [−0.03, +0.05] | 0 [−0.01, +0.02] |
| Pareto/NBD | 0.05 | 60% | 10 / 10 | 42 → 34 | 12 → 7 | **−9 [−12, −5]** | −5 [−11, 0] | −0.02 [−0.09, +0.04] | 0 [−0.01, +0.02] |
| Pareto/NBD | 0.05 | 80% | 10 / 10 | 58 → 39 | 21 → 8 | **−19 [−26, −13]** | **−12 [−22, −3]** | +0.04 [−0.02, +0.11] | +0.02 [−0.01, +0.04] |
| Pareto/NBD | 0.10 | 20% | 10 / 10 | 31 → 30 | 3 → 2 | **−1 [−2, 0]** | −1 [−3, +1] | −0.05 [−0.11, 0] | 0 [−0.01, +0.02] |
| Pareto/NBD | 0.10 | 40% | 10 / 10 | 32 → 30 | 5 → 2 | **−2 [−3, 0]** | **−3 [−5, −2]** | +0.02 [−0.08, +0.11] | −0.01 [−0.02, +0.01] |
| Pareto/NBD | 0.10 | 60% | 10 / 10 | 37 → 32 | 9 → 3 | **−5 [−7, −3]** | **−6 [−10, −2]** | 0 [−0.07, +0.07] | 0 [−0.02, +0.01] |
| Pareto/NBD | 0.10 | 80% | 10 / 10 | 49 → 35 | 21 → 7 | **−14 [−22, −8]** | **−13 [−24, −4]** | −0.03 [−0.15, +0.09] | +0.01 [0, +0.03] |
| Pareto/NBD | 0.30 | 20% | 10 / 10 | 30 → 30 | 14 → 15 | 0 [−1, +1] | +1 [−1, +2] | +0.01 [−0.22, +0.23] | 0 [−0.01, +0.01] |
| Pareto/NBD | 0.30 | 40% | 10 / 10 | 31 → 30 | 14 → 14 | −1 [−2, 0] | 0 [−2, +2] | −0.14 [−0.40, +0.10] | 0 [−0.01, +0.01] |
| Pareto/NBD | 0.30 | 60% | 10 / 10 | 32 → 30 | 13 → 13 | **−2 [−3, 0]** | 0 [−3, +3] | −0.11 [−0.29, +0.07] | 0 [−0.01, +0.01] |
| Pareto/NBD | 0.30 | 80% | 10 / 10 | 35 → 31 | 13 → 12 | **−4 [−6, −2]** | −1 [−8, +5] | +0.04 [−0.18, +0.26] | 0 [−0.02, +0.02] |

</details>

### 13. RMSE can rank these models

**Verdict (described, not tested): not supported for per-week RMSE; supported within
one rate for customer-total RMSE.** RMSE is a descriptive metric under the protocol, so this
claim rests on the spread of the 13 tree means, not on an interval. Per customer-week, RMSE puts 10 of 13 trees on the same value to two decimals at
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
  with unseeded training variation. The per-cell intervals therefore measure robustness to
  drawing a new panel and training on it once, together; neither part can be separated.
- **Transformer at 3,000 customers.** Not run, so claim 12 is LSTM-only.
- **Other encodings.** Only none / bounded / unbounded history and K = 8 were tested. The flag
  bins {2, 4, 8, 16, 32} were not tuned, and the `projected` embedder was not run.
- **Changing seasonality.** It is fixed across the grid, so nothing here says how the models
  behave under other seasonal patterns.
- **Transfer to real data.** The generator is a Pareto/NBD, so the benchmark is correct by
  construction here. On real panels there is no known ceiling (see
  `docs/benchmarks-real-panels.md`).
