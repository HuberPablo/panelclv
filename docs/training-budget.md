# Family U and other training-budget remainders

Undertraining — how it was found, tested and what fixes it — is in
`docs/insight-training-efficiency.md`; selection is in `docs/model-selection.md`; the
refit noise is in `docs/model-selection.md` §2. What is left here has no home yet and
moves to `docs/insights-real-panels.md`: the training × cluster-label factorial
(family U) on ranking, and the panel descriptions it is read against.

## The panels

Every result is stated for the panel it was measured on, and where panels
disagree that is heterogeneity to be explained. The panels differ in ways that plausibly matter:

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

"< 4 tx" is the share of customers with at most 3 transactions in that window; "both
windows" counts calibration and holdout together. Counts are the target channel the
models read, so a week's count is already capped at the panel's top class.

**No measured panel characteristic yet predicts where a model collapses.** Under the
archive recipe with no cluster label, both LSTMs collapse on electronics and multichannel
(forecast CV 0.08–0.15, §15.1) and not on cdnow or gift (0.54–1.18). Yet electronics is
*less* sparse than gift and has the most holdout transactions per customer. A reader may form hypotheses from this table; this document does not
assert one.


## 6. How much of the ranking collapse this explains

`docs/benchmarks-real-panels.md` reports that on electronics the benchmark gives nearly
every customer the same forecast (Spearman 0.032). Training length is part of that, but it
is the smaller part. Per-customer Spearman recomputed from the stored `Predictions/` of
1,580 archived electronics studies:

| what changes | Spearman | source |
| --- | ---: | --- |
| no per-customer feature, patience 7 (archived benchmark) | 0.03 | archive |
| same inputs, trained to the paper's epoch count | **0.178** | §15.1, n = 20 |
| a `kmeans_8` cluster label added, patience 7 | **0.305** | §15.1, n = 20 |
| Pareto/NBD on the same panel | 0.314 | 20 seeded fits (§15.1) |

*(This table first reported 0.09 for the trained row, from a five-replication pilot.
§15.1 measures 0.178 at n = 20 with an interval of +0.119 to +0.193 on the difference, so
the trained row is roughly twice what the pilot suggested — which narrows the gap between
the two levers without closing it.)*

Within the studies that carry no cluster label (n = 40, the pre-experiment archive only —
family T, U and V suites are excluded, or this would be circular), more training still
helps, though one of the two correlations does not clear the standard: **+0.328 with
updates (95% CI +0.024 to +0.580, supported)** and **+0.280 with best epoch (−0.015 to
+0.532, not supported)**, n = 40 studies. Each is one rank correlation computed *across*
the studies, not a difference of two means, so `effect()` cannot express it; the
interval is still the protocol's — the study is the unit, its (x, y) pair is resampled
intact, 95% percentile bootstrap from 10,000 resamples at the package's fixed seed
(`correlation_interval` in `.scratch/training-budget/all_effects.py`). So the archive's
hint is real but thin, and it does not
approach what one persistent per-customer input buys. That ordering matches
`docs/insights-cluster-ablation.md` §5.1, and the honest summary is: **on electronics both
levers are large, the input is the larger, and §15.1 measures the two crossed — stacking
them adds at most a few hundredths.** An earlier version of this line called the collapse
"mostly an input problem with a training-length component"; at 0.178 against 0.305 that
understates the training half.

On CDNOW, which never collapsed, training length shows no visible relationship with
ranking (Spearman flat at ~0.40 across every quartile of best epoch; descriptive quartile
means with no interval, so "no clear relationship", not "none", and not re-tested under
the statistical protocol) — even though CDNOW is the
panel leaving the most validation loss on the table. Whether its level metrics move is
untested.

*(`.scratch/training-budget/hparams.py`, `spearman_vs_epoch.py`; the two intervals from
`all_effects.py`)*


## 15. Family U: crossing the two levers adds little, and one of them may be dangerous

Family T moved the training lever and the cluster ablation the input lever, each from the same floor, and neither knew what the other was
doing. Family U crosses them: **training** (`archive` = patience 7 and the 100-trial
search, against `floored` = the paper's recipe pinned with `min_epochs=90` and one trial)
× **inputs** (`no_cluster` against `kmeans_8`), on two models and all four panels, 20
replications a cell. 640 suites on vast.ai, 21 September 2026, $1.68.
`scripts/run_factorial.py`; tests by `.scratch/training-budget/factorial_analysis.py`.

### 15.1 Ranking: the label does most of the work, and the floor adds little on top of it

Per-customer Spearman, mean over 20 replications (Pareto/NBD: mean of 20 seeded fits on
the same windows, `real_panel_benchmarks__ParetoNBD__<panel>__r00`–`r19`):

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

Forecast CV of the same forecasts, mean over 20 replications:

| panel | model | archive / no_cluster | archive / kmeans_8 | floored / no_cluster | floored / kmeans_8 |
| --- | --- | ---: | ---: | ---: | ---: |
| cdnow | ValendinLSTM | 1.18 | 2.30 | 1.03 | 2.29 |
| | LSTM | 0.85 | 1.94 | 0.36 | 1.85 |
| electronics | ValendinLSTM | **0.08** | 1.30 | 0.30 | 1.35 |
| | LSTM | **0.09** | 1.14 | 0.25 | 1.38 |
| gift | ValendinLSTM | 0.57 | 1.35 | 0.42 | 1.38 |
| | LSTM | 0.54 | 1.43 | 0.48 | 1.35 |
| multichannel | ValendinLSTM | **0.15** | 1.92 | 0.47 | 2.16 |
| | LSTM | **0.13** | 1.85 | 0.39 | 2.10 |

The collapsed cells (bold) are the ones with near-zero Spearman above, and each lever
lifts them out: the floor by a factor of three, the label by an order of magnitude. The two
measures do not always move together — cdnow's floored LSTM without a label has CV 0.36,
flatter than any other cdnow cell, yet ranks at 0.352.

**Every contrast the 2×2 supports, under the standard** — Δ of condition means with a 95%
percentile-bootstrap CI (`effect()`, independent), 20 replications a cell, beside that
panel's Spearman refit noise; regenerated by `.scratch/training-budget/all_effects.py`
into `results/effects_spearman.csv`. `label alone` and `floor alone` are measured against `archive / no_cluster`;
`+label` and `+floor` are measured against the other lever already applied.

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

Bold = interval excludes zero. Three readings, and the first corrects an earlier claim:

**The label's effect is supported on two panels, not four.** On electronics and
multichannel it is large and supported. On cdnow there is no clear difference for either
model at n = 20 (+0.039 and +0.012, both intervals spanning zero) and on gift it is
supported only for the LSTM, by +0.030. The label does not lift ranking on every panel.

**What is supported on all eight cells is the label added to an already-floored model**
(+0.023 to +0.127). That is the robust form of the claim: whatever the floor does, adding
a per-customer channel on top of it helps everywhere.

**The floor alone is the heterogeneous one** — large and positive on electronics and
multichannel, **negative and supported on gift**, and not supported on cdnow.

**Crossing them adds little.** Δ from adding the floor on top of the label, 95%
percentile-bootstrap CI, 20 replications a cell, beside that panel's Spearman refit noise:

| panel | model | Δ | 95% CI | supported | refit noise |
| --- | --- | ---: | :---: | :---: | ---: |
| cdnow | ValendinLSTM | +0.003 | −0.023 to +0.033 | no | 0.0159 |
| | LSTM | +0.004 | −0.014 to +0.024 | no | |
| electronics | ValendinLSTM | +0.001 | −0.011 to +0.012 | no | 0.0105 |
| | LSTM | +0.013 | −0.005 to +0.035 | no | |
| gift | ValendinLSTM | +0.004 | −0.005 to +0.014 | no | 0.0116 |
| | LSTM | +0.003 | −0.005 to +0.011 | no | |
| multichannel | ValendinLSTM | **+0.018** | **+0.003 to +0.035** | **yes** | 0.0152 |
| | LSTM | +0.014 | −0.004 to +0.031 | no | |

So: **in one of eight cells the increment is supported, and everywhere the interval rules
out a gain larger than about +0.035** — the same order as what an unseeded refit moves on
its own. That is the honest statement, and it is weaker than "they do not add": an
interval containing zero is no clear difference at n = 20, not an established absence.
Only gift/LSTM meets the equivalence margin fixed in "How claims are made" (the panel's
refit noise): no difference larger than ±0.0116 is detectable there at n = 20. On
electronics neither model meets it (ValendinLSTM −0.011 to +0.012 against ±0.0105).
The label reaches most of what is reachable by itself, the floor reaches a little over
half of it by itself, and stacking them buys at most a few hundredths.

**On its own the floor is a large effect on two panels, nothing on one, and negative on
another** — Δ Spearman from adding the floor with no label, ValendinLSTM:

| panel | Δ | 95% CI | supported | refit noise |
| --- | ---: | :---: | :---: | ---: |
| electronics | **+0.157** | +0.119 to +0.193 | yes | 0.0105 |
| multichannel | **+0.123** | +0.099 to +0.147 | yes | 0.0152 |
| cdnow | +0.019 | −0.019 to +0.065 | no | 0.0159 |
| gift | **−0.069** | −0.118 to −0.027 | yes (**worse**) | 0.0116 |

That heterogeneity is the result, not a nuisance to average away: the same intervention
helps decisively on two panels, shows no clear difference on a third and **hurts** on the fourth. It
matters because it needs no extra input — it is the fix available when no cluster label
is allowed — but it is not the better of the two, and it is not safe everywhere.

**Against the statistical benchmark, the collapse is closed and nothing more.** The best
cell (`floored / kmeans_8`) against Pareto/NBD, which is now a replicated condition: 20
seeded MCMC fits per panel on the same 2-year windows (every factorial suite's
`config.json` names the benchmark's windows). Δ = best cell − Pareto/NBD, independent
bootstrap, n = 20 / 20; `all_effects.py`:

| panel | model | best cell | Pareto/NBD (20 fits) | Δ | 95% CI | supported | refit noise |
| --- | --- | ---: | ---: | ---: | :---: | :---: | ---: |
| electronics | ValendinLSTM | 0.305 | 0.314 | −0.009 | −0.017 to −0.000 | yes (**below**, small) | 0.0105 |
| | LSTM | 0.302 | 0.314 | −0.012 | −0.023 to −0.002 | yes (**below**) | |
| multichannel | ValendinLSTM | 0.195 | 0.185 | +0.010 | +0.000 to +0.020 | yes (above, small) | 0.0152 |
| | LSTM | 0.189 | 0.185 | +0.004 | −0.012 to +0.018 | no | |
| cdnow | ValendinLSTM | 0.406 | 0.450 | −0.044 | −0.056 to −0.032 | yes (**below**) | 0.0159 |
| | LSTM | 0.403 | 0.450 | −0.047 | −0.060 to −0.034 | yes (**below**) | |
| gift | ValendinLSTM | 0.363 | 0.378 | −0.015 | −0.021 to −0.009 | yes (**below**) | 0.0116 |
| | LSTM | 0.359 | 0.378 | −0.019 | −0.025 to −0.013 | yes (**below**) | |

Pareto/NBD's electronics mean over 20 fits is 0.314 (range 0.297–0.334; the seed-42 fit
earlier versions of this document quoted alone is the lowest of them), and **both models'
best cell sits below Pareto/NBD on electronics**, by about the refit noise; on
multichannel the frozen benchmark sits above it by a similarly small margin and the LSTM
shows no clear difference; on the two panels that never collapsed both sit clearly below. So the collapse is closed on both collapsed panels in
the sense that the best cell ranks within about ±0.01 of Pareto/NBD there — not in the
sense of an equivalence, which none of these intervals was tested for. One caveat
belongs beside every one of those comparisons: `kmeans_8` is k-means over the Pareto/NBD
sufficient statistics, so the cell that draws level has been handed the benchmark's own
summary.


### 15.4 What follows

1. **Give the model a per-customer channel.** The cluster label is worth more than
   anything else measured here, and it brings the best cell to within about ±0.01 of
   Pareto/NBD's 20 seeded fits on both collapsed panels (§15.1) — while being, by construction, the benchmark's own summary fed back in. The
   honest reading is that the architecture cannot build that signal from counts alone,
   which is the case `docs/absorbing-death-state.md` makes for a learned survival
   variable rather than a borrowed one.
2. **Use a floor only where it pays, and never unconditionally** (`docs/insight-training-efficiency.md` §5.2).
3. **Stop selecting on validation cross-entropy where the panel collapses** — there it is
   wrong-signed — and keep it where the panel does not (`docs/model-selection.md` §3.4).
4. **An apparent plateau, not an established ceiling.** Two levers applied together land
   within a few hundredths of where either reaches alone, under these configurations. That
   is an observation about what has been tried, not a proof that ~0.30 bounds the
   architecture on electronics; nothing here rules out a third lever doing better.


## Owed

| # | what | why | status |
| ---: | --- | --- | --- |
| E2 | Recompute `kmeans_8` from calibration periods **before** the validation window, re-run the electronics `archive/kmeans_8` cell, 20 replications | The label is read at the last calibration period (`cluster_features.py:97`) and broadcast to every calibration row (`panel_dataset.py:961`), so a model training on period 5 sees a summary of periods 1–104. Holdout scoring is clean — the statistic is available at the forecast origin — but validation-based selection and early stopping see a label that has seen the window they score. | to do |
