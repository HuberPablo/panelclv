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
The frozen ValendinLSTM could not run, because it refuses every input except count and
week, and every arm below adds inputs (F11).

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
| Shape correlation | correlation of predicted and actual weekly totals; ignores level | 1 |
| Alive ratio R_A / dead leakage L_D | forecast volume before / after each customer's true death week, over true volume (uses the hidden truth) | 1 / 0 |

Every result table shows three metrics: **RMSE on customer totals / bias % / MAPE**. RMSE is
computed on each customer's holdout-year total, the paper's definition. It grows with the
purchase rate (about 0.6 at rate 0.01, about 5 at rate 0.30), so it is compared within one
rate only. The per-week RMSE in `results.csv` is not used: 10 of 13 trees score 0.18 on it.
Per-customer Spearman was never computed on this grid.

**Tests.** Wilcoxon signed-rank, paired on the same panels (per rate: 40 panels). Mann–Whitney
U where panels differ (1,000 vs 3,000 customers). The effect is the median of per-panel
differences.

## Results

The ranking depends on the purchase rate. At rates 0.01–0.10 Pareto/NBD is best on all three
metrics (MAPE at rate 0.10 is a tie, 37 vs 36); at rate 0.30 the LSTM is best on bias and
MAPE and level on RMSE. Two Transformer `ar_unbounded` studies whose forecasts did not match
their stored results are left out.

**Setting for this section.** Seasonal panels (4 peaks, amplitude 1.5), 1,000 customers, the
4 churn levels pooled. Each cell is RMSE on customer totals / bias % / MAPE, mean over 40
panels. RMSE grows with the purchase rate, so compare it within a column only. The last
column is the share of all 160 panels where the tree's MAPE beats Pareto/NBD's on the same
panel.

| Model | Arm | Rate 0.01 | 0.05 | 0.10 | 0.30 | Beats Pareto/NBD |
| --- | --- | --- | --- | --- | --- | ---: |
| **Pareto/NBD** | — | **0.62 / +35 / 90** | **1.44 / +11 / 44** | **2.19 / +4 / 37** | 5.17 / −13 / 32 | — |
| LSTM | `ar_bounded` | 0.77 / +231 / 243 | 1.64 / +44 / 58 | 2.42 / +20 / 36 | **4.94 / −2 / 20** | **46%** |
| LSTM | `no_ar` | 0.84 / +309 / 314 | 2.10 / +199 / 200 | 2.71 / +93 / 98 | 5.17 / +2 / 21 | 31% |
| LSTM | `ar_bounded` + `kmeans_8` | 0.77 / +210 / 221 | 1.89 / +121 / 126 | 2.86 / +76 / 78 | 6.61 / +56 / 56 | 9% |
| LSTM | `no_ar` + `kmeans_8` | 0.85 / +302 / 307 | 2.08 / +200 / 202 | 3.13 / +138 / 139 | 7.31 / +78 / 78 | 6% |
| LSTM | `ar_unbounded` + `kmeans_8` | 7.05 / +1107 / 1118 | 10.18 / +418 / 424 | 8.21 / +147 / 162 | 9.06 / +33 / 61 | 8% |
| LSTM | `ar_unbounded` | 3.38 / +505 / 513 | 11.88 / +582 / 582 | 12.71 / +315 / 322 | 16.35 / +131 / 141 | 6% |
| Transformer | `ar_bounded` | 0.65 / +50 / 103 | 1.74 / +64 / 78 | 2.73 / +64 / 77 | 5.98 / +38 / 50 | 32% |
| Transformer | `no_ar` | 0.67 / +74 / 120 | 1.99 / +109 / 114 | 3.18 / +126 / 130 | 6.47 / +43 / 56 | 24% |
| Transformer | `ar_bounded` + `kmeans_8` | 0.71 / +60 / 119 | 1.94 / +85 / 96 | 3.12 / +92 / 97 | 6.96 / +60 / 64 | 19% |
| Transformer | `ar_unbounded` + `kmeans_8` | 0.70 / +44 / 121 | 1.97 / +72 / 87 | 3.16 / +81 / 88 | 7.00 / +36 / 50 | 17% |
| Transformer | `no_ar` + `kmeans_8` | 0.73 / +71 / 130 | 2.00 / +115 / 122 | 3.34 / +114 / 116 | 7.11 / +64 / 67 | 18% |
| Transformer | `ar_unbounded` | 0.75 / +89 / 150 | 1.95 / +95 / 105 | 3.79 / +148 / 149 | 8.92 / +89 / 92 | 16% |

**Share of panels where the model beats Pareto/NBD on MAPE, by rate** (same setting, 40
panels each):

| Model | Arm | Rate 0.01 | 0.05 | 0.10 | 0.30 |
| --- | --- | ---: | ---: | ---: | ---: |
| LSTM | `ar_bounded` | 8% | 25% | 65% | **88%** |
| LSTM | `no_ar` | 0% | 5% | 28% | **92%** |
| Transformer | `ar_bounded` | 38% | 22% | 30% | 38% |
| Transformer | `no_ar` | 35% | 10% | 10% | 42% |

Churn-level tables are in the AR-feature section, and the 3,000-customer grid in the
cohort-size section.

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
- **Customer ranking.** Per-customer Spearman was never computed on this grid.
