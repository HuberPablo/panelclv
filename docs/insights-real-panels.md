# Insights from the real panels

What the experiments on the real panels found: CDNOW, electronics, gift, multichannel
and electronic_5y.

Covered elsewhere:
- benchmark rows and their reproduction: `docs/benchmarks.md`
- training length: `docs/insight-training-efficiency.md`
- model selection: `docs/model-selection.md`
- what each input is and why it was tested: `docs/feature-engineering.md`
- the synthetic grid: `docs/insights-synthetic-grid.md`

Conventions:
- Every comparison is Δ = mean(B) − mean(A) with a 95% percentile-bootstrap interval
  from `evaluation.effects.effect` (`docs/statistical-protocol.md`), independent
  replications unless stated. **Bold** marks an interval that excludes 0.
- Spearman (of per-customer holdout totals) carries ranking claims; aggregate MAPE
  carries level claims; |bias| is secondary.
- RMSE is descriptive and separates nothing on panels that are 98–99.7% zeros
  (`docs/loss-functions.md` §4.1).
- Each panel is stated separately. Refit-noise magnitudes are in
  `docs/model-selection.md` §2.

**Caveat on every neural row.** The archived models stopped training early. On
electronics and multichannel that understates them (`docs/insight-training-efficiency.md`).

## 1. Summary

1. **Count-only neural models collapse on long sparse panels.** On electronics and
   multichannel they give every customer nearly the same forecast, so they cannot rank
   customers. CDNOW does not collapse. (§3)
2. **Any persistent per-customer input undoes the collapse, and so does training longer.**
   With both applied, the best cell ranks within about ±0.01 of Pareto/NBD on the two
   collapsed panels. (§3, §5.3)
3. **AR encodings trade level against ranking.** Bounded flags protect the level;
   compressed encodings (log, ratio) protect the ranking. No encoding wins both on
   electronics or multichannel. `ar_ratio` ranks above Pareto/NBD on gift. (§4)
4. **Unbounded counters are catastrophic for the LSTM, and the cause is the fitted
   conditional, not the rollout.** The Transformer largely tolerates them. (§4.2, §4.4)
5. **The cluster label helps ranking only where the model collapsed.** It does nothing
   clear on CDNOW or gift, and lowers ranking on electronic_5y. It is a remedy for
   collapse, not a general input. (§5)
6. **Three years of calibration lifts ranking for Pareto/NBD and the LSTM together**, so the
   ranking gap does not close. The MAPE gain on electronics and gift belongs to the
   model. (§6)
7. **On electronic_5y (the paper's split) the LSTM beats Pareto/NBD on MAPE and ranking,**
   and every neural cell over-forecasts by about 10%. The Transformer is the weakest
   model there. (§7)
8. **The aggregate tables hide shape.** On CDNOW every model sits against a MAPE floor of
   about 18, and a good bias can come from cancellation. (§9)

## 2. The panels

Windows, budgets and the benchmark setup: `docs/benchmarks.md`.

| panel | customers | T_CAL / T_HOLD | zero cells (holdout) | calibration tx | holdout tx | holdout/calibration rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cdnow | 2,357 | 39 / 39 | 98.0% | 4,796 | 1,895 | 0.40 |
| electronics | 829 | 104 / 52 | 98.6% | 4,684 | 1,467 | 0.63 |
| gift | 2,062 | 104 / 52 | 99.0% | 4,207 | 1,146 | 0.54 |
| multichannel | 1,402 | 104 / 52 | 99.7% | 2,016 | 228 | 0.23 |

How much each customer buys. Transactions per customer are over the whole window, not per
week. The top 10% are the customers with the most calibration transactions, so their
holdout column shows how much the heaviest known buyers keep buying.

| panel | tx per customer, calibration | tx per customer, holdout | top 10%, calibration | top 10%, holdout | < 4 tx, calibration | < 4 tx, holdout | < 4 tx, both windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cdnow | 2.03 | 0.80 | 6.97 | 3.53 | 87.6% | 93.6% | 77.6% |
| electronics | 5.65 | 1.77 | 18.77 | 5.20 | 45.0% | 83.5% | 36.8% |
| gift | 2.04 | 0.56 | 5.48 | 1.63 | 86.2% | 97.7% | 76.9% |
| multichannel | 1.44 | 0.16 | 3.70 | 0.57 | 96.0% | 99.6% | 93.9% |

"< 4 tx" is the share of customers with at most 3 transactions in that window. Counts are
the clipped target the models read.

No measured panel characteristic predicts where a model collapses. Electronics is *less*
sparse than gift and has the most holdout transactions per customer, yet it collapses and
gift does not.

## 3. The forecast collapse

**The measure.** Forecast CV is `std / mean` of the per-customer predicted holdout totals.
It is 0 when every customer gets the same number. Spearman says whether that variation is
in the right order.

**Electronics census.** Every stored forecast on the benchmark windows: 1,643 distinct
forecasts over 488 suites, matched by execution (holdout, cohort and order checked). A row
is what the model read, i.e. its per-customer channels beside the count. Regenerate with
`PYTHONPATH=src python scripts/measure_forecast_collapse.py`.

| per-customer input beside the count | model | calendar | forecasts | median CV | mean Spearman |
| --- | --- | --- | ---: | ---: | ---: |
| count only | ValendinLSTM | week | 20 | 0.05 | 0.03 |
| count only | LSTM | week_sin, week_cos | 184 | 0.07 | 0.04 |
| count only | ValendinLSTM | none | 20 | 0.08 | 0.03 |
| count only | LSTM | none | 20 | 0.09 | 0.05 |
| count only | Transformer | week_sin, week_cos | 34 | 0.16 | 0.09 |
| count only | Transformer | none | 20 | 0.19 | 0.15 |
| active_in_last_{2,4,8,16,32}_periods, has_transacted_before | Transformer | week_sin, week_cos | 20 | 0.21 | 0.20 |
| active_in_last_{2,4,8,16,32,52}_periods, has_transacted_before | LSTM | week_sin, week_cos | 100 | 0.30 | 0.20 |
| active_in_last_{2,4,8,16,32}_periods, has_transacted_before | LSTM | week_sin, week_cos | 220 | 0.34 | 0.26 |
| active_in_last_{2,4,8,16,32}_periods, has_transacted_before, recency_over_tenure, transaction_rate, saturating_tenure_26_periods | LSTM | week_sin, week_cos | 100 | 0.36 | 0.30 |
| recency_over_tenure, transaction_rate, saturating_tenure_26_periods, has_transacted_before | LSTM | week_sin, week_cos | 200 | 0.43 | 0.30 |
| saturating_recency_26_periods, transaction_rate, saturating_tenure_26_periods | LSTM | week_sin, week_cos | 100 | 0.49 | 0.30 |
| kmeans_16 | LSTM | week_sin, week_cos | 40 | 0.75 | 0.26 |
| kmeans_8 | LSTM | week_sin, week_cos | 60 | 0.93 | 0.27 |
| recency, frequency, tenure (HB MCMC) | ParetoNBD | — | 2 | 1.12 | 0.30 |
| active_in_last_{2,4,8,16,32}_periods, has_transacted_before, kmeans_8 | Transformer | week_sin, week_cos | 20 | 1.16 | 0.31 |
| kmeans_4 | LSTM | week_sin, week_cos | 40 | 1.24 | 0.26 |
| kmeans_8 | Transformer | week_sin, week_cos | 20 | 1.26 | 0.30 |
| active_in_last_{2,4,8,16,32}_periods, has_transacted_before, kmeans_8 | LSTM | week_sin, week_cos | 20 | 1.32 | 0.29 |
| kmeans_8 | Transformer | none | 20 | 1.37 | 0.31 |
| log_period_since_last_transaction, cumulative_transactions, log_period_since_first_transaction | LSTM | week_sin, week_cos | 200 | 1.42 | 0.29 |
| kmeans_8 | ValendinLSTM | none | 20 | 1.42 | 0.29 |
| kmeans_8 | LSTM | none | 20 | 1.50 | 0.30 |
| period_since_last_transaction, cumulative_transactions, period_since_first_transaction, kmeans_8 | LSTM | week_sin, week_cos | 40 | 1.69 | 0.28 |
| period_since_last_transaction, cumulative_transactions, period_since_first_transaction | LSTM | week_sin, week_cos | 100 | 1.71 | 0.23 |
| recency, frequency, tenure (MLE) | ParetoNBD_MLE | — | 3 | 2.26 | 0.28 |

- **Every neural model collapses when the count is its only per-customer input.** That
  holds across three architectures and three calendar treatments: CV 0.05–0.19, Spearman
  0.03–0.15. The calendar is the same for every customer, so it cannot separate them.
- **One persistent per-customer channel undoes it, and which one barely matters.** Any
  such channel gives CV ≥ 0.21 and Spearman ≥ 0.20.
- **More spread is not better ranking.** The unbounded counters give the widest forecasts
  (CV 1.71) and rank worse (0.23) than the bounded ratio set (CV 0.43, Spearman 0.30).
- **The Transformer collapses less.** Count-only, on the same budget (`real_panel_arms`,
  20 / 20), it ranks higher than the LSTM: Δ Spearman **+0.077** (+0.020, +0.132). MAPE
  shows no clear difference (−3.3, −7.1 to +0.8).

**The trigger is the panel.** The same count-only configuration (measured 13 Sep):
- does not collapse on CDNOW (39 calibration weeks; CV 1.10–1.45, Spearman 0.35–0.43);
- half-collapses on gift (CV 0.65, Spearman 0.37);
- fully collapses on multichannel (CV 0.09, Spearman 0.005).

**Collapsed forecasts ignore history, and what history effect there is fades.**
- Rank correlation of the forecast with calibration frequency: Pareto/NBD 0.87 on
  electronics, ValendinLSTM 0.10. The same model reaches 0.84 on CDNOW.
- On electronics, ValendinLSTM forecasts 1.51× more for recently active customers in
  holdout week 1, but only 1.03× over the whole holdout. Pareto/NBD holds 3.5×.
- Only 10 of 204 count-only LSTM forecasts escape (CV > 0.2). No hyperparameter
  correlates with CV beyond |ρ| = 0.46.

**Mechanism.** `docs/absorbing-death-state.md` §4 measures it. Fed the true history, the
fitted conditional ranks customers at ρ = 0.816. Rolled out on its own samples it drops to
0.240, and 86–93% of the loss runs through the recency encoding. Training longer also lifts
the collapse (`docs/insight-training-efficiency.md` §5.2).

## 4. AR encodings

### 4.1 Electronics and CDNOW: families E and H

LSTM and Transformer with the encodings of `docs/feature-engineering.md` §4.3, against both
benchmarks.
- `ar_encoding` (family E) and `real_panel_arms` (family H) run 50 trials and 200–300
  paths, against the benchmark's 100 / 500.
- **On CDNOW they use the pre-ADR-0009 38-week holdout.** So they are compared only with
  each other and with one Pareto/NBD fit on that window (MAPE 18.7), and their Spearman
  cannot be recomputed (`n/a`).
- Family E arms are not budget-matched to each other (40–100 replications;
  `docs/studies-run.md` §4.1).

| panel | model and features | n | trials / paths | bias % | MAPE | RMSE | Spearman |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| electronics | **Benchmark: ValendinLSTM**, count + week | 20 | 100 / 500 | +46.0 ± 14.8 | 70.8 ± 6.4 | 0.3770 ± 0.0003 | 0.032 ± 0.033 |
| electronics | **Benchmark: Pareto/NBD** | 20 fits | — | −63.1 ± 0.4 | 65.7 ± 0.3 | 0.3758 ± 0.0000 | 0.314 ± 0.010 |
| electronics | LSTM no_ar (`ar_encoding`) | 100 | 50 / 300 | +22.4 ± 16.4 | 56.3 ± 7.5 | 0.3767 ± 0.0003 | 0.041 ± 0.065 |
| electronics | LSTM ar_bounded_32 (`ar_encoding`) | 100 | 50 / 300 | +1.2 ± 23.6 | 46.3 ± 8.4 | 0.3760 ± 0.0003 | 0.257 ± 0.044 |
| electronics | LSTM ar_bounded_52 (`ar_encoding`) | 100 | 50 / 300 | −8.2 ± 16.4 | 45.4 ± 5.0 | 0.3760 ± 0.0004 | 0.202 ± 0.121 |
| electronics | **LSTM ar_saturating (`ar_encoding`)** | 100 | 50 / 300 | **−11.6 ± 18.0** | **43.3 ± 3.6** | 0.3757 ± 0.0002 | **0.299 ± 0.019** |
| electronics | LSTM ar_log (`ar_encoding`) | 100 | 50 / 300 | +41.5 ± 34.5 | 63.0 ± 24.3 | 0.3855 ± 0.0167 | 0.292 ± 0.015 |
| electronics | LSTM ar_ratio (`ar_encoding`) | 100 | 50 / 300 | +47.1 ± 27.7 | 70.4 ± 19.1 | 0.3772 ± 0.0015 | 0.304 ± 0.013 |
| electronics | LSTM ar_unbounded (`ar_encoding`) | 40 | 50 / 300 | +235.5 ± 230.4 | 240.7 ± 226.3 | 0.4325 ± 0.0632 | 0.227 ± 0.065 |
| electronics | LSTM no_ar (`real_panel_arms`) | 20 | 50 / 200 | +23.8 ± 12.8 | 56.3 ± 6.6 | 0.3769 ± 0.0004 | 0.036 ± 0.073 |
| electronics | LSTM ar_bounded (`real_panel_arms`) | 20 | 50 / 200 | +2.0 ± 31.1 | 48.4 ± 11.2 | 0.3762 ± 0.0005 | 0.249 ± 0.038 |
| electronics | Transformer no_ar (`real_panel_arms`) | 20 | 50 / 200 | +15.9 ± 19.2 | 53.0 ± 6.3 | 0.3792 ± 0.0006 | 0.113 ± 0.107 |
| electronics | Transformer ar_bounded (`real_panel_arms`) | 20 | 50 / 200 | +2.5 ± 19.9 | 49.5 ± 4.6 | 0.3763 ± 0.0003 | 0.201 ± 0.135 |
| cdnow | **Benchmark: ValendinLSTM**, count + week | 20 | 100 / 500 | −23.7 ± 20.2 | 36.2 ± 8.2 | 0.1468 ± 0.0008 | 0.404 ± 0.024 |
| cdnow | **Benchmark: Pareto/NBD** | 20 fits | — | −14.1 ± 2.3 | 20.5 ± 0.8 | 0.1455 ± 0.0000 | 0.450 ± 0.006 |
| cdnow | Pareto/NBD on the arms' 38-week window (`real_panel_arms__ParetoNBD__cdnow`) | 1 fit | — | −11.6 | 18.7 | 0.1468 | n/a |
| cdnow | LSTM no_ar (`ar_encoding`) | 100 | 50 / 300 | −0.7 ± 16.2 | 22.4 ± 8.2 | 0.1475 ± 0.0003 | n/a |
| cdnow | LSTM ar_bounded_16 (`ar_encoding`) | 100 | 50 / 300 | −8.5 ± 15.2 | 22.6 ± 5.7 | 0.1473 ± 0.0002 | n/a |
| cdnow | LSTM ar_bounded_32 (`ar_encoding`) | 40 | 50 / 300 | +42.9 ± 43.7 | 55.4 ± 37.2 | 0.1498 ± 0.0030 | n/a |
| cdnow | LSTM ar_saturating (`ar_encoding`) | 100 | 50 / 300 | +12.9 ± 14.6 | 27.4 ± 8.7 | 0.1479 ± 0.0013 | n/a |
| cdnow | LSTM ar_log (`ar_encoding`) | 100 | 50 / 300 | +20.5 ± 15.8 | 33.3 ± 11.7 | 0.1488 ± 0.0011 | n/a |
| cdnow | LSTM ar_ratio (`ar_encoding`) | **80** | 50 / 300 | +24.3 ± 15.4 | 34.6 ± 11.2 | 0.1492 ± 0.0031 | n/a |
| cdnow | LSTM ar_unbounded (`ar_encoding`) | 40 | 50 / 300 | +334.1 ± 525.2 | 345.7 ± 522.3 | 0.2652 ± 0.1899 | n/a |
| cdnow | LSTM no_ar (`real_panel_arms`) | 20 | 50 / 200 | −0.5 ± 23.2 | 26.7 ± 12.4 | 0.1479 ± 0.0007 | n/a |
| cdnow | LSTM ar_bounded (`real_panel_arms`) | 20 | 50 / 200 | −11.8 ± 9.9 | 21.0 ± 3.3 | 0.1475 ± 0.0002 | n/a |
| cdnow | Transformer ar_bounded (`real_panel_arms`) | 20 | 50 / 200 | −13.6 ± 17.3 | 25.6 ± 6.9 | 0.1479 ± 0.0006 | n/a |

- **Electronics: the flags beat both benchmarks on MAPE** (45–48 against 70.8 and 65.7;
  every Δ supported). Pareto/NBD's −63% is a unit mismatch (`docs/benchmarks.md`).
- **Electronics ranking:** the flags gain **+0.160 to +0.216** over their own `no_ar` arm
  (Transformer **+0.088**), but stay below Pareto/NBD's 0.314 (Δ −0.056 to −0.113).
- **`ar_saturating` has the family's best level** (MAPE 43.3 ± 3.6, 22.4 below Pareto/NBD)
  and ranks 0.015 below Pareto/NBD (−0.020, −0.009). It was never run on gift or
  multichannel.
- **CDNOW: the flags only shift the level down.** Bias Δ −7.8 (−12.2, −3.5) and −11.4
  (−22.4, −1.1) against `no_ar`. Every CDNOW arm has a higher MAPE than the one Pareto/NBD
  fit on its window; `real_panel_arms` flags are closest (+2.3, +0.9 to +3.8).

### 4.2 Unbounded counters fail before any rollout happens

On electronics, a teacher-forced pass of the `ar_unbounded` LSTM (true counts *and* true AR
values fed at every step, so no sampling and no feedback) still reproduces +169% bias,
against +235% rolled out.
- Beyond the fitted range the predicted rate stops decaying and settles near 0.072, while
  the true long-silence rate is 0.0153.
- That region holds 49.9% of holdout cells against 7.7% of calibration cells, so 54% of
  the excess comes from it.

The fitted conditional is wrong, which is extrapolation, not exposure bias. A bounded
encoding brings |bias| back to the no-AR baseline on both panels. On CDNOW, a 32-week flag
on a 39-week window fails the same way: 3.5% of calibration cells lie past the bin against
68.9% of holdout cells.

### 4.3 Four panels: family O

LSTM, count (embedded) plus `week_sin`/`week_cos` plus one encoding, on the benchmark's
2-year windows. 100 replications × 100 trials × 500 paths, budget-matched to the
benchmark.

| panel | model | n | bias % | MAPE | RMSE | Spearman |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cdnow | LSTM + bounded32 | 100 | +37.1 ± 47.7 | 65.3 ± 30.7 | 0.1498 ± 0.0026 | 0.085 ± 0.170 |
| cdnow | LSTM + log | 100 | +65.8 ± 29.1 | 68.4 ± 26.9 | 0.1536 ± 0.0055 | 0.442 ± 0.015 |
| cdnow | **LSTM + ratio** | 100 | +17.6 ± 24.0 | 32.2 ± 14.4 | 0.1466 ± 0.0012 | 0.439 ± 0.009 |
| cdnow | ValendinLSTM (benchmark) | 20 | −23.7 ± 20.2 | 36.2 ± 8.2 | 0.1468 ± 0.0008 | 0.404 ± 0.024 |
| cdnow | Pareto/NBD (benchmark) | 20 fits | −14.1 ± 2.3 | 20.5 ± 0.8 | 0.1455 ± 0.0000 | 0.450 ± 0.006 |
| electronics | **LSTM + bounded32** | 100 | −1.7 ± 18.9 | 45.0 ± 4.5 | 0.3757 ± 0.0002 | 0.263 ± 0.042 |
| electronics | LSTM + log | 100 | +32.5 ± 25.7 | 55.7 ± 16.5 | 0.3816 ± 0.0116 | 0.296 ± 0.010 |
| electronics | LSTM + ratio | 100 | +49.1 ± 37.4 | 71.0 ± 29.8 | 0.3772 ± 0.0034 | 0.306 ± 0.013 |
| electronics | ValendinLSTM (benchmark) | 20 | +46.0 ± 14.8 | 70.8 ± 6.4 | 0.3770 ± 0.0003 | 0.032 ± 0.033 |
| electronics | Pareto/NBD (benchmark) | 20 fits | −63.1 ± 0.4 | 65.7 ± 0.3 | 0.3758 ± 0.0000 | 0.314 ± 0.010 |
| gift | LSTM + bounded32 | 100 | −18.5 ± 15.3 | 31.4 ± 5.2 | 0.1066 ± 0.0001 | 0.331 ± 0.035 |
| gift | LSTM + log | 100 | +11.0 ± 17.7 | 32.1 ± 9.9 | 0.1102 ± 0.0106 | 0.381 ± 0.007 |
| gift | **LSTM + ratio** | 100 | −2.9 ± 16.4 | 32.4 ± 10.0 | 0.1066 ± 0.0007 | 0.392 ± 0.008 |
| gift | ValendinLSTM (benchmark) | 20 | −15.7 ± 16.4 | 29.7 ± 4.9 | 0.1064 ± 0.0001 | 0.368 ± 0.017 |
| gift | Pareto/NBD (benchmark) | 20 fits | −11.4 ± 0.7 | 42.7 ± 0.2 | 0.1066 ± 0.0000 | 0.378 ± 0.006 |
| multichannel | **LSTM + bounded32** | 100 | +17.0 ± 20.0 | 53.2 ± 8.2 | 0.0569 ± 0.0000 | 0.096 ± 0.054 |
| multichannel | LSTM + log | 100 | +70.9 ± 81.8 | 90.9 ± 73.8 | 0.0596 ± 0.0077 | 0.123 ± 0.116 |
| multichannel | LSTM + ratio | 100 | +79.7 ± 47.8 | 98.0 ± 43.0 | 0.0571 ± 0.0015 | 0.175 ± 0.015 |
| multichannel | ValendinLSTM (benchmark) | 20 | +66.6 ± 55.5 | 96.6 ± 39.7 | 0.0570 ± 0.0001 | 0.005 ± 0.029 |
| multichannel | Pareto/NBD (benchmark) | 20 fits | +6.5 ± 1.8 | 55.8 ± 0.3 | 0.0567 ± 0.0000 | 0.185 ± 0.012 |
| cdnow | LSTM + **bounded32 + ratio** | 100 | +26.5 ± 32.4 | 40.7 ± 26.3 | 0.1466 ± 0.0017 | 0.425 ± 0.025 |
| electronics | LSTM + **bounded32 + ratio** | 100 | +33.0 ± 23.3 | 59.4 ± 15.5 | 0.3762 ± 0.0007 | 0.302 ± 0.014 |
| gift | LSTM + **bounded32 + ratio** | 100 | +5.3 ± 18.8 | 35.3 ± 10.7 | 0.1068 ± 0.0020 | 0.385 ± 0.013 |
| multichannel | LSTM + **bounded32 + ratio** | 100 | +55.0 ± 25.3 | 73.8 ± 18.3 | 0.0569 ± 0.0002 | 0.176 ± 0.015 |

Every comparison below is n = 100 / 100 (or 100 / 20 against a benchmark).

- **Flags protect the level; compressed encodings protect the ranking.** On every panel
  log and ratio rank above bounded32 (all 8 supported, +0.027 to +0.357). Where the level is
  hard (electronics, multichannel) they over-forecast by +32 to +80%.
- **`ratio` against Pareto/NBD's 20 fits:**
  - gift **+0.014** (+0.011, +0.017): the first neural configuration to rank above it;
  - CDNOW −0.010, electronics −0.008, multichannel −0.010, all supported below by less
    than the refit noise.
- **CDNOW's bounded32 failure is the 32-week flag** (§4.2). Log and ratio read the same
  calendar and rank at 0.44.
- **Against ValendinLSTM, bounded32 cuts MAPE by 25.8 on electronics and 43.4 on
  multichannel** (both supported); on gift it shows no clear MAPE difference and ranks
  lower (**−0.037**).
- **Combining flags and ratio is a compromise.** It keeps most of ratio's ranking (−0.004
  to −0.015) and only halves its over-forecast (electronics +33.0, multichannel +55.0). It
  is the best configuration on no panel. Adding the ratio set does rescue the 32-week
  flag on CDNOW (Spearman +0.340).
- **Best per panel:**

  | panel | best configuration |
  | --- | --- |
  | CDNOW | ratio (MAPE 32.2; Pareto/NBD's is 11.7 lower) |
  | electronics | bounded32 for level (MAPE 45.0), ratio or log for ranking |
  | gift | ratio: \|bias\| level with Pareto/NBD, best ranking |
  | multichannel | bounded32 for MAPE (2.6 below Pareto/NBD), while Pareto/NBD keeps level and ranking |

### 4.4 The unbounded blow-up is LSTM-specific

Family G (CDNOW 38-week window and electronics; 20 studies × 50 trials × 50 paths;
ensemble of 20):

| CDNOW `ar_unbounded` | LSTM | Transformer |
|---|---|---|
| `no_cluster` | +155.5% (sd 343, max +1,480) | **+35.5% (sd 21)** |
| `kmeans_8` | +244.7% (sd 510, max +1,885) | **−29.8% (sd 17)** |

Electronics shows the same pattern (LSTM +196.7%, Transformer +56.4%). On per-study
|bias| (20 / 20), the Transformer is closer to the truth in all four panel × cluster
cells:
- CDNOW Δ **−127.5** (−288.7, −15.0) and **−241.6** (−473.7, −59.7);
- electronics **−138.3** (−243.5, −52.2) and **−110.5** (−229.6, −14.4).

The encoding is dangerous; the recurrence makes it catastrophic. The synthetic grid agrees
(claim 4).

## 5. The cluster label

### 5.1 The K sweep (family F)

LSTM only, `valendin` embedder, 40 replications × 50 trials × 300 paths per arm. Cluster arms
carry no AR channel. CDNOW is on the pre-ADR-0009 window, so it has no Spearman.

| panel | arm | mean \|bias\| | Δ \|bias\| [95% CI] | mean MAPE | Δ MAPE [95% CI] |
|---|---|---:|---:|---:|---:|
| electronics | `no_cluster` | 21.7 | — | 54.9 | — |
| | `cluster_4` | 19.8 | −1.8 [−8.6, +5.8] | 51.8 | −3.1 [−7.6, +2.0] |
| | `cluster_8` | 19.0 | −2.7 [−9.0, +3.9] | 53.1 | −1.9 [−6.3, +3.2] |
| | `cluster_16` | 21.0 | −0.6 [−7.0, +5.9] | 55.2 | +0.3 [−3.2, +3.9] |
| | `ar_unbounded` | 208.6 | **+186.9 [+136.0, +242.9]** | 213.4 | **+158.5 [+108.8, +213.1]** |
| | `ar_plus_cluster_8` | 94.5 | **+72.9 [+37.2, +114.0]** | 115.5 | **+60.6 [+27.9, +99.4]** |
| CDNOW | `no_cluster` | 12.9 | — | 23.1 | — |
| | `cluster_4` | 30.1 | **+17.2 [+8.5, +27.4]** | 39.6 | **+16.5 [+8.9, +25.6]** |
| | `cluster_8` | 14.3 | +1.4 [−3.8, +7.1] | 26.9 | +3.9 [−0.2, +8.2] |
| | `cluster_16` | 15.7 | +2.8 [−3.1, +9.7] | 29.1 | **+6.1 [+1.2, +11.7]** |
| | `ar_unbounded` | 314.6 | **+301.8 [+156.5, +476.1]** | 325.6 | **+302.6 [+158.2, +475.8]** |
| | `ar_plus_cluster_8` | 273.2 | **+260.3 [+160.4, +373.2]** | 296.7 | **+273.7 [+174.1, +384.8]** |

| arm | Spearman | Δ vs `no_cluster` [95% CI] | mean \|bias\| |
|---|---:|---:|---:|
| `no_cluster` | 0.039 ± 0.078 | — | 21.7 |
| `cluster_4` | 0.264 ± 0.015 | **+0.226 [+0.199, +0.248]** | 19.8 |
| `cluster_8` | 0.270 ± 0.055 | **+0.232 [+0.200, +0.259]** | 19.0 |
| `cluster_16` | 0.257 ± 0.043 | **+0.219 [+0.189, +0.244]** | 21.0 |
| Pareto/NBD (benchmark, 20 seeded refits) | 0.314 ± 0.010 | +0.275 [+0.249, +0.297] | 63.1 |

Against Pareto/NBD's 20 fits (40 / 20), every labelled arm still ranks below it:
`cluster_4` **−0.049**, `cluster_8` **−0.043**, `cluster_16` **−0.056**,
`ar_plus_cluster_8` **−0.035**.

- **No K beats `no_cluster` on either level metric on either panel.** The only supported
  level effects are on CDNOW, and they are worse.
- **On electronics the label lifts ranking sevenfold at every K, at no cost in level.**
- **K has no common shape across panels.** Electronics has a shallow optimum at 8;
  CDNOW's median |bias| improves monotonically in K. The label also widens the spread
  across replications (bias sd 14.1 → 19.0–24.8 on electronics).
- **It rescues the unbounded counters on electronics only:** `ar_plus_cluster_8` against
  `ar_unbounded` gives |bias| **−114.0** (−179.0, −49.5). On CDNOW the difference is not
  clear (−41.5, −246.6 to +144.1). The rescued arm is still far worse than `no_cluster`.

### 5.2 Training × label on four panels (family U)

ValendinLSTM and LSTM, `archive` (patience 7, 100-trial search) against `floored` (the paper
recipe plus `min_epochs=90`), crossed with `no_cluster` / `kmeans_8`. 20 replications a
cell, 2-year windows. The floor's own effect is in `docs/insight-training-efficiency.md`
§5.2.

| panel | model | archive / no_cluster | archive / kmeans_8 | floored / no_cluster | **floored / kmeans_8** | Pareto/NBD |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cdnow | ValendinLSTM | 0.364 | 0.403 | 0.383 | **0.406** | 0.450 |
| | LSTM | 0.386 | 0.398 | 0.352 | **0.403** | |
| electronics | ValendinLSTM | 0.021 | 0.305 | 0.178 | **0.305** | 0.314 |
| | LSTM | 0.029 | 0.289 | 0.182 | **0.302** | |
| gift | ValendinLSTM | 0.349 | 0.359 | 0.280 | **0.363** | 0.378 |
| | LSTM | 0.326 | 0.356 | 0.263 | **0.359** | |
| multichannel | ValendinLSTM | −0.004 | 0.178 | 0.119 | **0.195** | 0.185 |
| | LSTM | 0.003 | 0.175 | 0.079 | **0.189** | |

Forecast CV on the collapsed cells (`archive / no_cluster`): electronics 0.08–0.09 and
multichannel 0.13–0.15, against 1.14–2.16 with the label.

| panel | model | label alone | floor alone | +floor (on label) | +label (on floor) |
| --- | --- | ---: | ---: | ---: | ---: |
| cdnow | ValendinLSTM | +0.039 (−0.003, +0.086) | +0.019 (−0.019, +0.065) | +0.003 (−0.023, +0.033) | **+0.023 (+0.001, +0.047)** |
| | LSTM | +0.012 (−0.013, +0.039) | −0.034 (−0.095, +0.019) | +0.004 (−0.014, +0.024) | **+0.051 (+0.001, +0.113)** |
| electronics | ValendinLSTM | **+0.283 (+0.264, +0.302)** | **+0.157 (+0.119, +0.193)** | +0.001 (−0.011, +0.012) | **+0.127 (+0.094, +0.162)** |
| | LSTM | **+0.260 (+0.235, +0.282)** | **+0.153 (+0.114, +0.188)** | +0.013 (−0.005, +0.035) | **+0.120 (+0.088, +0.156)** |
| gift | ValendinLSTM | +0.010 (−0.005, +0.026) | **−0.069 (−0.118, −0.027)** | +0.004 (−0.005, +0.014) | **+0.084 (+0.043, +0.130)** |
| | LSTM | **+0.030 (+0.005, +0.061)** | **−0.063 (−0.118, −0.011)** | +0.003 (−0.005, +0.011) | **+0.096 (+0.055, +0.144)** |
| multichannel | ValendinLSTM | **+0.182 (+0.163, +0.199)** | **+0.123 (+0.099, +0.147)** | **+0.018 (+0.003, +0.035)** | **+0.077 (+0.054, +0.100)** |
| | LSTM | **+0.173 (+0.158, +0.186)** | **+0.076 (+0.051, +0.101)** | +0.014 (−0.004, +0.031) | **+0.111 (+0.083, +0.137)** |

`label alone` and `floor alone` are measured against `archive / no_cluster`; `+label` and
`+floor` against the other lever already applied.

- **The label alone is supported on electronics and multichannel** (+0.17 to +0.28), on
  gift for the LSTM only (+0.030), and not on CDNOW.
- **The label added to an already-floored model helps in all eight cells** (+0.023 to
  +0.127). That is the robust form of the claim.
- **Stacking adds little.** The floor on top of the label is supported in 1 of 8 cells
  (multichannel/ValendinLSTM, +0.018). Every interval rules out a gain above about +0.035,
  which is the order of the refit noise.
- **Against Pareto/NBD** (best cell `floored / kmeans_8`):
  - electronics: below on both models (ValendinLSTM **−0.009**, −0.017 to −0.000; LSTM
    **−0.012**);
  - multichannel: ValendinLSTM above (**+0.010**, +0.000 to +0.020), LSTM no clear
    difference;
  - CDNOW and gift: clearly below (−0.015 to −0.047).

  `kmeans_8` is k-means over Pareto/NBD's own sufficient statistics, so the cell that
  draws level has been handed the benchmark's summary.

### 5.3 What the label is worth: one verdict

| Evidence | Verdict |
| --- | --- |
| Collapsed panels (electronics, multichannel), families F and U | **Lifts ranking decisively**, no clear level cost |
| Non-collapsed panels (CDNOW, gift), family U | No clear ranking gain alone; small gain on top of a floor |
| CDNOW level, family F | Worse for K = 4, and for K = 16 on MAPE |
| electronic_5y, three models (§7) | **Lowers ranking** under every rule (−0.030 to −0.051) |
| Synthetic grid, level (claim 5) | Hurts in most cells where a usable AR encoding is present |

**The label is a remedy for collapse, not a general input.** It helps exactly where the
count-only model gives every customer the same forecast. Elsewhere it is neutral or
harmful, because it is frozen and cannot update when a simulated customer goes quiet.
Experiment E2 is still owed: recompute the label before the validation window and re-run
electronics `archive / kmeans_8` (`docs/feature-engineering.md` §5).

## 6. Three-year calibration

Family P: the four family O encodings on electronics, gift and multichannel, with three
calibration years (two to fit, the third for validation) and the following year as the
holdout. Family Q is Pareto/NBD on the same windows, 20 seeded fits. CDNOW is too short.
**The holdout year moves with the window**, so Pareto/NBD is the control for an easier or
harder year.

| panel | customers | calibration | validation from | holdout | T_CAL / T_HOLD | holdout transactions | zero cells |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| electronics | 829 | 1999-01-01 → 2001-12-31 | 2001-01-01 | 2002-01-01 → 2002-12-31 | 156 / 52 | 1,541 | 98.7% |
| gift | 2,062 | 2001-02-25 → 2004-02-24 | 2003-02-25 | 2004-02-25 → 2005-02-24 | 156 / 52 | 1,040 | 99.1% |
| multichannel | 1,402 | 2005-01-01 → 2007-12-31 | 2007-01-01 | 2008-01-01 → 2008-12-31 | 156 / 52 | 173 | 99.8% |

Each cell is bias % / MAPE / Spearman, mean of 100 replications:

| panel | encoding | 2-year calibration | **3-year calibration** |
| --- | --- | --- | --- |
| electronics | bounded32 | −1.7 / 45.0 / 0.263 | **−12.8 / 38.2 / 0.322** |
| electronics | log | +32.5 / 55.7 / 0.296 | **+10.3 / 40.0 / 0.307** |
| electronics | ratio | +49.1 / 71.0 / 0.306 | **+6.0 / 39.1 / 0.323** |
| electronics | bounded32 + ratio | +33.0 / 59.4 / 0.302 | **+8.8 / 39.9 / 0.323** |
| gift | bounded32 | −18.5 / 31.4 / 0.331 | **−5.4 / 27.9 / 0.423** |
| gift | log | +11.0 / 32.1 / 0.381 | **+33.1 / 38.3 / 0.430** |
| gift | ratio | −2.9 / 32.4 / 0.392 | **+15.1 / 31.3 / 0.441** |
| gift | bounded32 + ratio | +5.3 / 35.3 / 0.385 | **+15.0 / 30.7 / 0.441** |
| multichannel | bounded32 | +17.0 / 53.2 / 0.096 | **+33.3 / 63.8 / 0.236** |
| multichannel | log | +70.9 / 90.9 / 0.123 | **+57.0 / 75.0 / 0.248** |
| multichannel | ratio | +79.7 / 98.0 / 0.175 | **+46.2 / 68.1 / 0.241** |
| multichannel | bounded32 + ratio | +55.0 / 73.8 / 0.176 | **+47.1 / 68.3 / 0.240** |

| panel | model | bias % | MAPE | RMSE | Spearman |
| --- | --- | ---: | ---: | ---: | ---: |
| electronics | **Pareto/NBD, 3-year (20 fits)** | −65.2 ± 0.3 | 66.0 ± 0.3 | 0.4049 | 0.332 ± 0.005 |
| electronics | LSTM + bounded32, 3-year (best MAPE) | −12.8 ± 19.0 | 38.2 ± 4.8 | 0.4041 | 0.322 ± 0.017 |
| gift | **Pareto/NBD, 3-year (20 fits)** | −8.1 ± 0.5 | 42.8 ± 0.2 | 0.1001 | 0.437 ± 0.005 |
| gift | LSTM + bounded32, 3-year (best MAPE) | −5.4 ± 10.9 | 27.9 ± 2.6 | 0.1001 | 0.423 ± 0.013 |
| multichannel | **Pareto/NBD, 3-year (20 fits)** | +22.6 ± 1.6 | 60.4 ± 0.4 | 0.0488 | 0.253 ± 0.009 |
| multichannel | LSTM + bounded32, 3-year (best MAPE) | +33.3 ± 25.5 | 63.8 ± 12.2 | 0.0489 | 0.236 ± 0.032 |

- **Ranking improves in all 12 cells** (+0.011 to +0.140), **but Pareto/NBD gains too.**
  Its gains are electronics **+0.019**, gift **+0.059**, multichannel **+0.068**. Against
  Pareto/NBD on the same windows, the best LSTM arm stands about where it did at two
  years: gift **+0.004**, multichannel −0.005 (not clear), electronics **−0.009**. Longer
  calibration lifts both models together.
- **The MAPE gain on electronics and gift belongs to the model.** Pareto/NBD's MAPE gets
  very slightly worse (+0.25, +0.17), while bounded32 improves by **−6.8** and **−3.5**. At
  three years it is 27.7 and 15.0 MAPE points below Pareto/NBD.
- **On electronics the encodings converge** (bias −13 to +10, MAPE 38–40), so the 2-year
  level/ranking trade-off largely disappears.
- **Multichannel 2008 is a harder year.** Pareto/NBD goes from +6.5% to +22.6%, so about a
  third of the LSTM's +33 to +57% is the year. Every 3-year LSTM arm is worse than the
  3-year Pareto/NBD on MAPE.
- ValendinLSTM has not been run on these windows.

## 7. electronic_5y: inputs and architectures

The paper's electronics cohort at trip level, on Valendin et al.'s split: 3,755 households,
260 calibration weeks (the last 52 for validation), a 52-week holdout. The benchmark rows
are in `docs/benchmarks.md`.

**The input side** is a census, with no interval. The long window shrinks the escape,
and K = 52 is well populated:

| feature | escapes calibration range | z past the ceiling |
| --- | ---: | ---: |
| `period_since_first_transaction` | 60.6% | 0.70 |
| `period_since_last_transaction` | 16.3% | 0.84 |
| `cumulative_transactions` | 0.03% | 3.14 |
| `transaction_rate`, `has_transacted_before`, `active_in_last_{32,52}_periods` | 0% | — |

| share of cells after a customer's first purchase whose silence is ≥ K | calibration | holdout |
| --- | ---: | ---: |
| K = 32 | 65.4% | 79.7% |
| K = 52 (the deepest flag of `ar_bounded_52`) | 51.5% | 71.7% |

**Setting.**
- Every model reads the count and the embedded week (`valendin` embedder); each other row
  adds exactly one input: `ar_bounded_52` or `kmeans_8`.
- Epoch rules:
  - `searched`: a 100-trial search per study, input "none" only;
  - `nofloor`: the least-biased searched study's settings pinned, one trial;
  - `from20` / `from30`: the same, keeping only weights from epoch 20 or 30 on.
- 20 studies per cell, 500 paths.
- The pinned settings:
  - **LSTM:** hidden and dense 128, dropout 0, lr 0.002195, batch 32. This is ValendinLSTM
    with input "none", and `MultinomialLSTMModel` in its shape otherwise.
  - **LSTMAttention:** hidden and dense 64, lr 0.0025.
  - **Transformer:** d_model 64, 4 heads, 3 layers, lr 0.0027.
- Pareto/NBD over 20 fits: RMSE 1.228, bias −16.6%, MAPE 27.7, Spearman 0.395, CV 1.40.

Mean [95% interval] over 20 studies. LSTM:

| rule | input | RMSE (customer total) | bias % | MAPE | Spearman | forecast CV |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `searched` | none | 1.163 [1.159, 1.168] | +13.8 [+9.3, +18.8] | 20.5 [17.9, 23.7] | 0.405 [0.404, 0.406] | 1.27 [1.24, 1.31] |
| `nofloor` | none | 1.178 [1.164, 1.198] | +13.2 [+6.7, +20.8] | 21.5 [17.6, 27.3] | 0.402 [0.399, 0.404] | 1.24 [1.17, 1.29] |
| `nofloor` | `ar_bounded_52` | 1.183 [1.174, 1.193] | +8.7 [+5.1, +12.1] | 17.5 [16.6, 18.6] | 0.399 [0.396, 0.402] | 1.14 [1.10, 1.17] |
| `nofloor` | `kmeans_8` | 1.279 [1.268, 1.289] | +9.0 [+4.9, +13.5] | 18.5 [16.4, 21.0] | 0.358 [0.352, 0.364] | 1.62 [1.56, 1.68] |
| `from20` | none | 1.159 [1.154, 1.164] | +10.2 [+7.1, +13.4] | 17.8 [16.6, 19.3] | 0.403 [0.400, 0.405] | 1.30 [1.27, 1.33] |
| `from20` | `ar_bounded_52` | 1.154 [1.146, 1.161] | +13.5 [+10.5, +16.7] | 19.5 [17.9, 21.3] | 0.401 [0.399, 0.403] | 1.28 [1.24, 1.32] |
| `from20` | `kmeans_8` | 1.270 [1.261, 1.279] | +12.4 [+9.0, +15.9] | 18.7 [17.0, 20.6] | 0.354 [0.349, 0.359] | 1.63 [1.58, 1.68] |
| `from30` | none | 1.157 [1.151, 1.163] | +12.7 [+9.5, +15.7] | 19.4 [17.9, 20.8] | 0.402 [0.399, 0.405] | 1.31 [1.28, 1.34] |
| `from30` | `ar_bounded_52` | 1.157 [1.149, 1.165] | +10.0 [+7.1, +13.1] | 18.4 [17.2, 19.9] | 0.400 [0.397, 0.402] | 1.23 [1.19, 1.26] |
| `from30` | `kmeans_8` | 1.266 [1.257, 1.276] | +11.5 [+7.6, +15.7] | 18.3 [16.1, 20.9] | 0.351 [0.344, 0.357] | 1.66 [1.59, 1.72] |
| Pareto/NBD, 20 fits | — | 1.228 [1.228, 1.229] | −16.6 [−16.9, −16.3] | 27.7 [27.7, 27.7] | 0.395 [0.394, 0.396] | 1.40 [1.40, 1.41] |

LSTMAttention:

| rule | input | RMSE (customer total) | bias % | MAPE | Spearman | forecast CV |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `searched` | none | 1.208 [1.205, 1.213] | +16.0 [+11.2, +21.4] | 21.4 [18.2, 25.3] | 0.403 [0.402, 0.405] | 1.07 [1.05, 1.10] |
| `nofloor` | none | 1.205 [1.201, 1.210] | +14.1 [+10.4, +18.0] | 19.6 [17.5, 22.2] | 0.399 [0.397, 0.401] | 1.06 [1.03, 1.10] |
| `nofloor` | `ar_bounded_52` | 1.221 [1.217, 1.224] | +9.1 [+4.7, +13.9] | 18.3 [16.1, 20.8] | 0.399 [0.397, 0.400] | 1.09 [1.06, 1.11] |
| `nofloor` | `kmeans_8` | 1.294 [1.287, 1.300] | +15.7 [+12.9, +18.6] | 20.2 [18.4, 22.1] | 0.366 [0.362, 0.369] | 1.51 [1.48, 1.55] |
| `from20` | none | 1.199 [1.193, 1.203] | +14.8 [+11.2, +18.3] | 20.1 [18.2, 22.4] | 0.400 [0.398, 0.403] | 1.13 [1.10, 1.16] |
| `from20` | `ar_bounded_52` | 1.208 [1.204, 1.211] | +9.3 [+5.7, +13.1] | 17.6 [16.0, 19.6] | 0.401 [0.399, 0.403] | 1.16 [1.14, 1.18] |
| `from20` | `kmeans_8` | 1.283 [1.276, 1.290] | +13.7 [+10.6, +16.9] | 19.1 [17.3, 20.9] | 0.370 [0.369, 0.372] | 1.52 [1.49, 1.55] |
| `from30` | none | 1.200 [1.195, 1.204] | +10.8 [+7.8, +13.6] | 18.1 [17.1, 19.3] | 0.400 [0.398, 0.403] | 1.17 [1.14, 1.19] |
| `from30` | `ar_bounded_52` | 1.202 [1.196, 1.207] | +9.3 [+7.2, +11.6] | 17.0 [16.1, 18.0] | 0.402 [0.400, 0.404] | 1.18 [1.16, 1.20] |
| `from30` | `kmeans_8` | 1.289 [1.282, 1.296] | +15.5 [+12.5, +18.4] | 19.9 [17.9, 21.9] | 0.369 [0.366, 0.372] | 1.52 [1.48, 1.55] |
| Pareto/NBD, 20 fits | — | 1.228 [1.228, 1.229] | −16.6 [−16.9, −16.3] | 27.7 [27.7, 27.7] | 0.395 [0.394, 0.396] | 1.40 [1.40, 1.41] |

Transformer:

| rule | input | RMSE (customer total) | bias % | MAPE | Spearman | forecast CV |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `searched` | none | 1.325 [1.279, 1.377] | +51.7 [+42.0, +62.1] | 54.2 [45.1, 64.0] | 0.400 [0.398, 0.402] | 1.33 [1.28, 1.37] |
| `nofloor` | none | 1.290 [1.268, 1.314] | +34.1 [+25.3, +42.8] | 43.5 [37.1, 49.9] | 0.396 [0.395, 0.398] | 1.32 [1.28, 1.35] |
| `nofloor` | `ar_bounded_52` | 1.257 [1.238, 1.282] | +12.5 [+4.1, +20.8] | 31.5 [27.4, 35.9] | 0.401 [0.400, 0.402] | 1.28 [1.21, 1.37] |
| `nofloor` | `kmeans_8` | 1.330 [1.316, 1.346] | +16.4 [+11.9, +21.2] | 32.2 [28.4, 36.5] | 0.347 [0.342, 0.351] | 1.62 [1.59, 1.65] |
| `from20` | none | 1.298 [1.274, 1.326] | +39.8 [+32.8, +46.7] | 44.4 [38.8, 50.1] | 0.396 [0.393, 0.398] | 1.32 [1.29, 1.36] |
| `from20` | `ar_bounded_52` | 1.251 [1.239, 1.265] | +13.7 [+8.8, +19.1] | 24.2 [21.2, 27.6] | 0.399 [0.398, 0.400] | 1.27 [1.22, 1.31] |
| `from20` | `kmeans_8` | 1.314 [1.305, 1.324] | +11.6 [+8.0, +15.3] | 26.2 [23.8, 28.8] | 0.349 [0.345, 0.352] | 1.67 [1.63, 1.70] |
| `from30` | none | 1.294 [1.275, 1.315] | +40.1 [+32.2, +48.0] | 44.0 [37.4, 50.9] | 0.396 [0.394, 0.399] | 1.31 [1.28, 1.34] |
| `from30` | `ar_bounded_52` | 1.246 [1.240, 1.251] | +12.5 [+8.9, +16.3] | 21.7 [19.7, 24.0] | 0.397 [0.396, 0.399] | 1.24 [1.20, 1.28] |
| `from30` | `kmeans_8` | 1.306 [1.295, 1.318] | +10.6 [+6.7, +14.7] | 24.4 [22.1, 27.0] | 0.352 [0.349, 0.354] | 1.69 [1.66, 1.73] |
| Pareto/NBD, 20 fits | — | 1.228 [1.228, 1.229] | −16.6 [−16.9, −16.3] | 27.7 [27.7, 27.7] | 0.395 [0.394, 0.396] | 1.40 [1.40, 1.41] |

**Every one of the 30 cells over-forecasts, supported.** The lowest lower bound is +4.1%.
Every cell's bias is above Pareto/NBD's by +25.3 to +68.2.

**LSTM against Pareto/NBD**, n = 20 / 20:

| rule | input | Δ RMSE (customer total) | Δ \|bias\| | Δ MAPE | Δ Spearman | Δ bias % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `searched` | none | **−0.065 [−0.070, −0.060]** | −1.7 [−5.6, +2.7] | **−7.2 [−9.8, −4.0]** | **+0.010 [+0.009, +0.011]** | **+30.4 [+25.8, +35.3]** |
| `nofloor` | none | **−0.050 [−0.065, −0.031]** | −1.1 [−6.3, +5.8] | **−6.2 [−10.1, −0.5]** | **+0.007 [+0.004, +0.009]** | **+29.7 [+23.2, +37.4]** |
| `nofloor` | `ar_bounded_52` | **−0.045 [−0.054, −0.036]** | **−6.3 [−8.8, −3.7]** | **−10.2 [−11.1, −9.1]** | **+0.004 [+0.000, +0.007]** | **+25.3 [+21.6, +28.7]** |
| `nofloor` | `kmeans_8` | **+0.051 [+0.040, +0.061]** | **−7.2 [−11.2, −2.9]** | **−9.2 [−11.3, −6.8]** | **−0.037 [−0.043, −0.031]** | **+25.6 [+21.5, +30.1]** |
| `from20` | none | **−0.069 [−0.074, −0.064]** | **−5.8 [−8.5, −2.9]** | **−9.9 [−11.2, −8.4]** | **+0.008 [+0.005, +0.010]** | **+26.8 [+23.6, +30.0]** |
| `from20` | `ar_bounded_52` | **−0.074 [−0.082, −0.067]** | −3.0 [−6.1, +0.2] | **−8.2 [−9.8, −6.4]** | **+0.006 [+0.004, +0.009]** | **+30.1 [+27.0, +33.3]** |
| `from20` | `kmeans_8` | **+0.042 [+0.032, +0.051]** | **−4.1 [−7.5, −0.6]** | **−9.0 [−10.7, −7.1]** | **−0.041 [−0.046, −0.036]** | **+29.0 [+25.6, +32.5]** |
| `from30` | none | **−0.071 [−0.077, −0.065]** | **−3.8 [−7.0, −0.8]** | **−8.4 [−9.8, −6.9]** | **+0.007 [+0.004, +0.010]** | **+29.3 [+26.0, +32.3]** |
| `from30` | `ar_bounded_52` | **−0.072 [−0.079, −0.063]** | **−6.4 [−9.3, −3.4]** | **−9.3 [−10.5, −7.8]** | **+0.005 [+0.002, +0.007]** | **+26.6 [+23.7, +29.7]** |
| `from30` | `kmeans_8` | **+0.038 [+0.029, +0.047]** | **−4.7 [−8.4, −0.7]** | **−9.4 [−11.7, −6.8]** | **−0.044 [−0.051, −0.038]** | **+28.0 [+24.1, +32.2]** |

**What each input changes**, within one model and one rule, n = 20 / 20:

| model | rule | input added | Δ Spearman | Δ MAPE | Δ \|bias\| | Δ RMSE (customer total) | Δ forecast CV |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| LSTM | `nofloor` | `ar_bounded_52` | −0.003 [−0.007, +0.001] | −4.0 [−10.0, +0.1] | −5.2 [−12.7, +0.8] | +0.005 [−0.016, +0.023] | **−0.10 [−0.17, −0.03]** |
| LSTM | `from20` | `ar_bounded_52` | −0.002 [−0.005, +0.002] | +1.7 [−0.4, +3.9] | +2.8 [−1.4, +7.0] | −0.005 [−0.014, +0.004] | −0.02 [−0.07, +0.03] |
| LSTM | `from30` | `ar_bounded_52` | −0.002 [−0.006, +0.001] | −0.9 [−2.9, +1.1] | −2.6 [−6.8, +1.7] | −0.000 [−0.010, +0.010] | **−0.08 [−0.13, −0.04]** |
| LSTMAttention | `nofloor` | `ar_bounded_52` | −0.000 [−0.002, +0.002] | −1.3 [−4.7, +2.0] | −3.9 [−9.5, +1.8] | **+0.016 [+0.010, +0.021]** | +0.03 [−0.02, +0.07] |
| LSTMAttention | `from20` | `ar_bounded_52` | +0.001 [−0.003, +0.004] | −2.5 [−5.3, +0.2] | −4.8 [−9.6, +0.0] | **+0.009 [+0.003, +0.015]** | +0.03 [−0.01, +0.07] |
| LSTMAttention | `from30` | `ar_bounded_52` | +0.002 [−0.002, +0.005] | −1.1 [−2.6, +0.4] | −1.9 [−5.1, +1.5] | +0.002 [−0.005, +0.009] | +0.01 [−0.02, +0.04] |
| Transformer | `nofloor` | `ar_bounded_52` | **+0.005 [+0.003, +0.007]** | **−12.1 [−19.7, −4.5]** | **−15.2 [−25.3, −5.1]** | −0.033 [−0.064, +0.001] | −0.03 [−0.12, +0.06] |
| Transformer | `from20` | `ar_bounded_52` | **+0.003 [+0.001, +0.006]** | **−20.2 [−26.7, −13.8]** | **−25.2 [−33.4, −16.8]** | **−0.047 [−0.077, −0.019]** | −0.06 [−0.11, +0.00] |
| Transformer | `from30` | `ar_bounded_52` | +0.001 [−0.002, +0.003] | **−22.3 [−29.7, −15.5]** | **−27.3 [−36.2, −18.7]** | **−0.048 [−0.071, −0.028]** | **−0.07 [−0.12, −0.02]** |
| LSTM | `nofloor` | `kmeans_8` | **−0.044 [−0.051, −0.037]** | −3.0 [−9.2, +1.8] | −6.1 [−14.3, +0.8] | **+0.100 [+0.079, +0.119]** | **+0.38 [+0.30, +0.47]** |
| LSTM | `from20` | `kmeans_8` | **−0.049 [−0.055, −0.043]** | +0.9 [−1.4, +3.2] | +1.7 [−2.8, +6.1] | **+0.111 [+0.100, +0.121]** | **+0.33 [+0.27, +0.39]** |
| LSTM | `from30` | `kmeans_8` | **−0.051 [−0.058, −0.044]** | −1.0 [−3.8, +1.9] | −0.9 [−5.7, +4.1] | **+0.109 [+0.098, +0.120]** | **+0.35 [+0.28, +0.42]** |
| LSTMAttention | `nofloor` | `kmeans_8` | **−0.033 [−0.037, −0.029]** | +0.6 [−2.6, +3.5] | +1.6 [−3.2, +6.3] | **+0.088 [+0.081, +0.096]** | **+0.45 [+0.40, +0.49]** |
| LSTMAttention | `from20` | `kmeans_8` | **−0.030 [−0.033, −0.027]** | −1.1 [−3.9, +1.5] | −1.1 [−5.8, +3.5] | **+0.084 [+0.075, +0.092]** | **+0.39 [+0.35, +0.43]** |
| LSTMAttention | `from30` | `kmeans_8` | **−0.031 [−0.035, −0.027]** | +1.8 [−0.6, +4.0] | **+4.3 [+0.4, +8.2]** | **+0.089 [+0.081, +0.098]** | **+0.35 [+0.31, +0.39]** |
| Transformer | `nofloor` | `kmeans_8` | **−0.050 [−0.055, −0.044]** | **−11.4 [−19.0, −3.8]** | **−17.7 [−27.6, −7.9]** | **+0.040 [+0.012, +0.067]** | **+0.30 [+0.26, +0.34]** |
| Transformer | `from20` | `kmeans_8` | **−0.047 [−0.051, −0.043]** | **−18.2 [−24.5, −11.9]** | **−27.3 [−34.8, −19.6]** | +0.016 [−0.012, +0.043] | **+0.34 [+0.29, +0.39]** |
| Transformer | `from30` | `kmeans_8` | **−0.045 [−0.048, −0.042]** | **−19.6 [−27.0, −12.5]** | **−28.3 [−37.1, −19.8]** | +0.012 [−0.012, +0.035] | **+0.38 [+0.34, +0.43]** |

- **Without the label, every LSTM cell beats Pareto/NBD on MAPE** (−6.2 to −10.2) **and
  ranking** (+0.004 to +0.010, small). It misses the total by less in 7 of 10 cells, but in
  the opposite direction.
- **The flags matter only for the Transformer.** They cut its MAPE by −12.1, −20.2 and
  −22.3 under the three rules. For both LSTMs nothing is supported, and the intervals
  reach from about −10 to +4.
- **The label lowers ranking for every model under every rule** (−0.030 to −0.051, three to
  five times the borrowed refit noise), while raising forecast CV by +0.30 to +0.45. It
  separates customers along its eight groups rather than along what they go on to buy.
  For the Transformer it lowers MAPE about as much as the flags do.
- **Attention does not help the LSTM.** There is no clear difference in MAPE, |bias| or
  Spearman without the label.
- **The Transformer is the weakest model here.** Without features its MAPE is 22.0–26.6
  above the LSTM's. With the flags the gap narrows to +3.3 to +13.9, still supported.
- **No model reaches the published LSTM's +2.7% bias.**
- **Not re-tested:** a first draw of the four floored feature cells was overwritten
  (summaries only: bias within 4 points and customer-level RMSE within 0.011 of the
  draw above).

## 8. Architectures on the real panels

| Comparison | Finding | § |
| --- | --- | --- |
| Transformer vs LSTM, count only, electronics | Transformer ranks higher (**+0.077**), collapses less | 3 |
| Transformer vs LSTM, unbounded counters | Transformer far closer to the truth in all 4 cells | 4.4 |
| LSTMAttention vs LSTM, electronic_5y | No clear difference without the label | 7 |
| Transformer vs LSTM, electronic_5y | Transformer worse on MAPE (+22 to +27 without features) | 7 |
| P-sLSTM vs LSTM, electronics | No clear forecast difference at n = 8, at ~14× the cost | `docs/p-slstm.md` |

Families O and P are LSTM-only; `scripts/run_real_panel_ar.py --model transformer` has
never been run. The candidates in `Papers/Lstm_prediction/` (xLSTM, Mamba, TFT, …) all
promise a better next-period density, which P-sLSTM shows does not survive the rollout.
- **NOA-LSTM** is the cheapest of them to run: a one-line change to the cell
  (`y_t = c_t ⊙ o_t`), at 5–20× slower epochs.
- **Deep Renewal Processes** (Türkmen et al.) is the one paper aimed at mostly-zero counts
  rather than at sequence capacity. It is still unread.

## 9. What the aggregate tables hide

Family G, ensembles of 20 forecasts.

**CDNOW has a MAPE floor at about 18.** Detrended, the weekly actuals have a lag-1
autocorrelation of +0.198, effectively white noise, with sd 13.1 where Poisson sampling
would give 7.0. The best smooth curve fitted to the actuals themselves scores:

| CDNOW | MAPE |
|---|---|
| oracle trend line fitted to the actuals | **18.10** |
| ValendinLSTM ensemble | 18.13 |
| LSTM ensemble | 18.29 |
| Pareto/NBD | 18.70 |

Every model sits against the floor, so the CDNOW ranking inside it is not a model ordering.
Doing better would need exogenous calendar or promotion covariates.

**A good bias can be cancellation.** On CDNOW the Transformer `no_ar-kmeans_8` has the best
bias (+1.3%) but flattens after week 25 and ends near 53 against an actual 30. It
under-predicts early and over-predicts late. Its MAPE (22.5) is the worst. Never rank on
`bias_percent` alone.

**Electronics has a structured seasonal shape** (a dip in weeks 8–20, a climb in weeks
45–51). The models that read the calendar track it. Pareto/NBD, flat at about 10, does
not (correlation 0.015). Correlation of each ensemble's weekly curve with the actual:

| arm | model | time features | corr |
|---|---|---|---|
| `ar_bounded-kmeans_8` | LSTM | yes | **0.568** |
| `no_ar-no_cluster` | Transformer | yes | **0.558** |
| `no_ar-kmeans_8` | Transformer | yes | 0.502 |
| `no_ar-kmeans_8` | LSTM | yes | 0.334 |
| `no_ar-kmeans_8-no_tf` | Transformer | no | 0.305 |
| `no_ar-kmeans_8-no_tf` | ValendinLSTM | no | 0.176 |
| `no_ar-no_cluster-no_tf` | LSTM | no | 0.128 |
| `no_ar-no_cluster-no_tf` | Transformer | no | −0.018 |
| `no_ar-no_cluster-no_tf` | ValendinLSTM | no | −0.006 |
| — | Pareto/NBD | n/a | 0.015 |

Removing the calendar removes shape tracking for every model. ValendinLSTM only ever runs
in the `-no_tf` arms, because it refuses non-embedded `week_sin`/`week_cos` (ADR-0004).
So a ValendinLSTM-vs-LSTM comparison must hold the arm fixed:

| electronics, `-no_tf` arms | LSTM | ValendinLSTM |
|---|---|---|
| `no_ar-no_cluster-no_tf` | +19.5% | +34.7% |
| `no_ar-kmeans_8-no_tf` | +9.9% | +18.1% |

The fair sentence is "the frozen architecture cannot consume calendar covariates", not
"the benchmark forecasts worse".

## 10. A few bad runs, or bad throughout?

All 2,880 stored forecasts behind the family N, O and P tables were scored one at a time. A
run is **extreme** if it lies more than 3 scaled MADs from its cell's median.

| cell | bias mean | bias median | extreme runs (bias) | Spearman mean → without extremes |
| --- | ---: | ---: | ---: | --- |
| ValendinLSTM, electronics | +46.0 | +46.8 | 1 / 20 | 0.032 → 0.032 |
| ValendinLSTM, multichannel | +66.6 | +60.6 | 0 / 20 | 0.005 → 0.012 |
| LSTM + bounded32, cdnow | +37.1 | +50.2 | 0 / 100 | 0.085 → 0.085 |
| LSTM + ratio, multichannel | +79.7 | +72.6 | 4 / 100 | 0.175 → 0.175 |
| LSTM + bounded32 + ratio, multichannel | +55.0 | +55.1 | 0 / 100 | 0.176 → 0.176 |
| **LSTM + log, multichannel** | **+70.9** | **+47.4** | **14 / 100** | **0.123 → 0.185** |
| **LSTM + ratio, electronics** | **+49.1** | **+38.7** | **8 / 100** | 0.306 → 0.306 |

- **ValendinLSTM's collapse is systematic.** Its best Spearman is 0.083 on electronics and
  0.049 on multichannel.
- **The CDNOW bounded32 failure has no outliers, just a wide spread.** 38 of its 100 runs
  rank customers below zero.
- **Two cells are skewed by extreme runs; read the median beside the mean.** In
  multichannel log, without the extremes mean bias is +43.9 rather than +70.9. In
  electronics ratio, +40.9 rather than +49.1.
- Elsewhere, mean and median bias lie within about 10 points. No trial crashed; the 40–75%
  that did not complete were pruned by `MedianPruner`.

## 11. Owed and limits

- **`ar_saturating`** is the best level arm on electronics, but was never run on gift or
  multichannel.
- **Transformer** on the four-panel runs (families O and P): never run.
- **`projected` embedder:** never run anywhere. Every number here uses `valendin`.
- **ValendinLSTM on the 3-year windows:** 0 of 20 replications.
- **E2:** the cluster label recomputed before the validation window (§5.3).
- **CDNOW Spearman** for families E–I is unrecoverable: those families used the
  pre-ADR-0009 window.
- **Electronics is not the paper's cohort** (829 customers, line items). electronic_5y is
  (`docs/benchmarks.md`).
