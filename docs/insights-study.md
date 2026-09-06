# Insights study: what the archived results say about which model to build next

Read `CONTEXT.md` first for the vocabulary (*calibration window*, *holdout window*,
*rollout*, *aggregate bias*, *study*, *grid*, *benchmark*, *contribution*), and
`CLAUDE.md` for the model contract: logits `(B, T, K)`, a count as a class not a
quantity, cross-entropy on a class index, evaluation by sampling-and-averaging.

Unlike `docs/hurdle-models-vs-pareto-nbd.md`, which is a reading exercise, **nothing
here comes from the literature.** Every number below was recomputed from the
`results.csv` files already under `Studies/`, from `.scratch/p-slstm/`, and from the
grid declarations in `grids/`. Where a claim rests on someone else's measurement
rather than an archived one, it says so.

## Contents

1. [The question and the short answer](#1-the-question-and-the-short-answer)
2. [What the seasonal grid measured](#2-what-the-seasonal-grid-measured)
3. [The diagnosis: no absorbing state](#3-the-diagnosis-no-absorbing-state)
4. [Three corroborating results already in the repo](#4-three-corroborating-results-already-in-the-repo)
5. [What to try, in order](#5-what-to-try-in-order)
6. [What this does not establish](#6-what-this-does-not-establish)
7. [Two loose ends found while reading](#7-two-loose-ends-found-while-reading)
8. [The real panels: what the arm axis did on CDNOW and electronics](#8-the-real-panels-what-the-arm-axis-did-on-cdnow-and-electronics)
9. [Score the ensemble, not the mean of the runs](#9-score-the-ensemble-not-the-mean-of-the-runs)

---

## 1. The question and the short answer

The standing question is which architecture to add next — the `Papers/Lstm_prediction`
folder holds NOA-LSTM, xLSTM, xLSTM-Mixer, xLSTMTime, Mamba, TFT, DeepAR, EA-LSTM and
P-LSTM, and each is a candidate.

**The archived results say architecture is not the binding constraint.** Every one of
those papers improves the *conditional distribution over the next period*, and this
repo already contains a direct measurement that such an improvement does not survive
the rollout: P-sLSTM won validation cross-entropy in all eight seeds and lost the
forecast, at roughly fourteen times the training cost (§4.1).

What the results do point at is structural, and it is visible on one axis of the
seasonal grid: **the neural models have no way to represent a customer who has stopped
buying, and their aggregate bias grows monotonically with the churn rate of the panel**
(§2, §3). That is the one mechanism the Pareto/NBD has and the contribution does not,
and it is the axis along which the two separate.

So the recommendation is a negative one about the paper folder and a positive one about
the model: add an absorbing state, not a better sequence encoder.

---

## 2. What the seasonal grid measured

`grids/seasonal_4x4x10.py` generates 160 synthetic panels — 4 mean transaction rates
`{0.01, 0.05, 0.10, 0.30}` × 4 churn rates `{0.20, 0.40, 0.60, 0.80}` × 10 seeds — of
1000 customers over 156 weeks, with four seasonal peaks. Two calibration years, one
holdout year, `clip_target_upper=6`, **no AR features** (`F = 3`: the target plus the
cyclical pair), and a 200-path forecast.

**The two neural arms did not get equal search effort.** The archived `config.json`
files record `n_trials=10` for the LSTM against `20` for the Transformer, so every
LSTM-vs-Transformer comparison below is between a 10-trial search and a 20-trial one
(see §6). `grids/seasonal_4x4x10.py` has since equalised both at 100 (commit
`2d815b3`), so a re-run does not carry the confound — but the numbers in this section
were produced before that, and stand as they are. **That re-run has since happened**, at
100 trials for both models and across six arms: see `docs/insights-arm-sweep.md`, which
reproduces this section's baseline arm and reports what the arm axis did and did not move.

The panels are generated *by* a Pareto/NBD process, so that benchmark is the correct
model by construction and is the ceiling, not a competitor. What the grid asks is how
far below the ceiling the neural models sit, and where.

Aggregate bias %, marginal over the four transaction rates (n = 40 studies per cell,
mean ± sd across them):

| churn rate | LSTM | Transformer | Pareto/NBD |
|---|---|---|---|
| 0.20 | +40.3 ± 35.1 | +50.2 ± 46.0 | **+2.2 ± 10.8** |
| 0.40 | +83.8 ± 76.3 | +91.0 ± 53.1 | **+9.2 ± 19.0** |
| 0.60 | +161.9 ± 120.4 | +107.2 ± 74.9 | **+10.2 ± 22.3** |
| 0.80 | +357.9 ± 334.0 | +212.4 ± 166.7 | **+15.8 ± 34.1** |

Marginal over the four churn rates instead:

| mean transaction rate | LSTM | Transformer | Pareto/NBD |
|---|---|---|---|
| 0.01 | +319.9 | +70.1 | +35.4 |
| 0.05 | +187.8 | +163.1 | +11.4 |
| 0.10 | +119.6 | +145.7 | +3.6 |
| 0.30 | +16.5 | +81.9 | −13.0 |

Worst cell for each model, both at churn 0.80:

| model | cell | aggregate bias % |
|---|---|---|
| LSTM | rate 0.01 | **+710.0 ± 368.4** |
| Transformer | rate 0.10 | +283.7 ± 192.2 |
| Pareto/NBD | rate 0.01 | +54.5 ± 33.5 |

Three things to read off this.

**The churn axis is monotone for all three models, and steep for two.** The LSTM's bias
rises 40 → 84 → 162 → 358 across the four churn levels; the Transformer 50 → 91 → 107 →
212; the Pareto/NBD 2 → 9 → 10 → 16. The benchmark degrades by a factor of 7 across the
axis and stays inside ±16%. The contribution degrades by a factor of 9 and ends an order
of magnitude outside it.

**The sparsity axis is monotone for the LSTM but not for the Transformer.** The LSTM
worsens steadily as the transaction rate falls (16.5 → 320); the Transformer peaks in the
middle of the axis (70 → 163 → 146 → 82). Whatever the Transformer is doing on the
sparsest panels, it is not the same failure the LSTM has there. This is worth a second
look before building anything, because it means the two architectures are not failing
identically and a single fix may not serve both.

**RMSE separates nothing.** Averaged over all 160 panels: LSTM 0.1792, Transformer
0.1828, Pareto/NBD 0.1754 — a spread far below the study-to-study noise of either
neural model, on panels where their aggregate MAPE differs by a factor of three (169.8,
128.6, 50.8). On a target that is mostly zeros, per-customer per-period RMSE is
dominated by getting the zeros right, which is the trap
`docs/hurdle-models-vs-pareto-nbd.md` §2 documents — it quotes Valendin et al.'s own
footnote 19, that forecasting mostly zero for everyone would "outperform" every model
in their study on MAE. **Do not rank models on
RMSE in this thesis.** The aggregate metrics are the ones carrying signal.

---

## 3. The diagnosis: no absorbing state

The Pareto/NBD has two latent per-customer quantities: a purchase rate `λᵢ` while alive
and an exponential lifetime governed by `μᵢ`. A churned customer contributes nothing
further, by construction. The `churn_rates` axis of the grid is literally the parameter
controlling how many customers die.

The models in `models/` have no equivalent. They emit a softmax over count classes at
every period, every class has positive mass, and a rollout samples from it 52 times in
succession. A customer who has gone quiet keeps drawing occasional non-zero counts;
because each sampled count is fed back as the next period's input, a spurious purchase
also re-primes the state that produced it. Over a 52-period horizon that compounds, and
it compounds hardest exactly where there are most dead customers to over-serve.

That is the shape of the table in §2: bias monotone in churn rate, and worst where the
transaction rate is lowest — the regime where "alive but quiet" and "dead" are hardest
to tell apart from the target channel alone.

This is a hypothesis fitted to one grid, not a proven mechanism, and §5.1 is the cheap
test that would confirm or kill it before any model is built.

Note what it is **not**. The head cap (`clip_target_upper=6`) biases forecasts
*downward*, so it cannot explain over-prediction; it is a separate, smaller issue
(`docs/loss-functions.md` §2.3(c) measures it: 25 holdout cells exceed class 6, carrying
80 transactions of 1,467, so a perfectly calibrated model still has a structural −5.45%
ceiling on electronics aggregate bias, and exactly 0 on CDNOW). And the grid carries no AR features at all, so this is not the
out-of-range AR failure of §4.3 — it is a distinct problem that the same rollout
amplifies.

---

## 4. Three corroborating results already in the repo

### 4.1 P-sLSTM: better density model, worse forecast

`.scratch/p-slstm/comparison-p-slstm.md`, run 2026-08-18 on the electronics panel,
8 seeds, both neural models at one hand-picked architecture point:

| model | best validation CE | RMSE | aggregate bias % | aggregate MAPE % |
|---|---|---|---|---|
| LSTM | 0.1004 (0.0997–0.1012) | 0.3807 ± 0.0012 | +2.41 ± 24.38 | 54.14 ± 5.90 |
| P-sLSTM | **0.0967** (0.0963–0.0968) | 0.3815 ± 0.0018 | +11.87 ± 27.81 | 56.93 ± 6.37 |
| Pareto/NBD | — | 0.3758 | −63.70 ± 0.35 | 66.18 ± 0.28 |

P-sLSTM is lower on validation cross-entropy in **every** seed, by a consistent ~0.004,
and converges in fewer epochs. It is worse on all three forecast metrics and costs
~14× as much to train (102–146s per seed against 7–8s).

This is the single most decision-relevant measurement in the repo for the "which
architecture next" question, because every candidate in the paper folder is offering the
same thing P-sLSTM offered: a better next-period density. One has already been bought
and it did not pay.

### 4.2 ADR-0003's retirement left the failure mode unguarded

`docs/adr/0003-rollout-composite-selection.md` records that selection on rollout quality
was removed, and says so in its own words: "selection on rollout quality is gone, so a
model that scores well next-step and drifts over a long horizon is unguarded against
again."

§4.1 is that sentence, measured. Optuna currently selects on teacher-forced validation
cross-entropy, which is precisely the metric P-sLSTM won and the forecast ignored.

The ADR was retired for two good reasons — it was never reachable from the production
path, and `tuning.weekly_aggregate_rollout_metrics` disagreed with
`models.monte_carlo_forecasting.compute_forecast_metrics` by 62× on RMSE, which made the
single-scoring-authority claim false. Neither reason is an argument that the idea is
wrong; both are arguments that the implementation was. Reinstating it now has empirical
support the original decision never had.

### 4.3 Out-of-range AR features, and what they say about the rollout

From the arms of `scripts/run_ar_encoding_ablation.py` under `Studies/`, 20 studies per
shard, aggregate bias %:

| arm | electronics (a / b) | CDNOW (a / b) |
|---|---|---|
| `no_ar` | +21.8 / +22.2 | +3.4 / +0.7 |
| bounded (K=32 / K=16) | +0.4 / +6.7 | −9.5 / −9.8 |
| bounded (K=52 / K=32) | −10.3 / −7.5 | +22.6 / +63.3 |
| **unbounded triple** | **+198.1 / +272.9** | **+460.9 / +207.3** |

The unbounded arm is `(period_since_last_transaction, cumulative_transactions,
period_since_first_transaction)` — the Pareto/NBD sufficient statistics `(t_x, x, T)`.
Two of the three are capped by the calibration window and keep counting through the
holdout, so much of the scored region sits outside the range the weights were fitted on.
Commit `03f4727` records the mechanism precisely: the predicted rate stops decaying and
settles near 0.072 against a true 0.0153.

This belongs here for a reason beyond the encoding lesson, which is already recorded.
**It is the same failure as §3 seen through a different channel.** In both cases the
model has no representation for "this customer has stopped", so the rollout's estimate of
their rate floors out at something well above zero and 52 steps of compounding do the
rest. The bounded encoding fixes the symptom by keeping the input in range; it does not
give the model a way to say the customer is gone.

---

## 5. What to try, in order

### 5.1 Confirm the diagnosis before building anything

Split the holdout aggregate bias by customer group and ask whether the over-prediction is
concentrated in customers with **zero holdout activity**. The segment machinery already
exists (`CONTEXT.md`, "Behavioural cluster", second paragraph — the customer groups of the
segment analysis are derived from calibration *and* holdout activity and exist to break a
results table apart, which is exactly this use).

Cost: no new model, no new training — it reads the forecasts already archived under
`Studies/seasonal_4x4x10__LSTM/`. If the bias is concentrated in the dead, §3 is
confirmed and it is a thesis paragraph whether or not any fix works. If it is spread
evenly, §3 is wrong and §5.2 should not be built.

Do this first. Everything below is conditional on it.

### 5.2 Give the model an absorbing state

The cheapest form that fits the package's contract: an additional outcome meaning
"inactive" which, once sampled during a rollout, **latches** — the customer emits zero
for every remaining period. It is a head change plus a latch in
`models.monte_carlo_forecasting.simulate_recurrent_path`, and it is the neural analogue
of the Pareto/NBD's `μ`.

Two things to decide before writing it, both real:

- **Where the latch lives.** As an extra softmax class it is inside the existing
  contract and `Embedder`'s head-size check still holds, but it competes with class 0
  for probability mass during teacher-forced training, where no "dead" label exists.
  As a separate Bernoulli head it trains against a target that also does not exist.
  Neither is free; the first is smaller and I would try it first.
- **That it is a departure worth an ADR.** `CLAUDE.md` states the categorical-head
  contract, and a latching state changes what a rollout is. It is the kind of decision
  `docs/adr/` exists for.

This is the contribution. It is not a reimplementation of anything in the paper folder.

### 5.3 Read the intermittent-demand paper, not the LSTM-cell papers

`Papers/Lstm_prediction/To_read_Intermittent_Demand_Forecasting_with_Renewal_Process__RNN_in_mainly_0_data.pdf`
(Türkmen et al., Deep Renewal Processes) is the only paper in that folder aimed at this
failure mode rather than at sequence-modelling capacity: mostly-zero counts, with
inter-arrival time modelled separately from size. That factorisation is a different route
to the same thing §5.2 buys, and it is published, which matters for a thesis.
Still unread, by the filename.

### 5.4 Restore rollout-based selection, properly

Reinstate what ADR-0003 retired, scored through `compute_forecast_metrics` so the
single-scoring-authority property holds this time, and wired into `StudySuiteConfig` so
it is reachable from `scripts/run_studies.py` rather than from notebook cells only.

This directly targets §4.1 and §4.2. It is expensive per trial — a rollout inside every
Optuna trial — so it competes with §5.2 for compute. §5.2 first: selection can only pick
the best available model, and if none of them can represent death, better selection picks
the least-bad drifter.

### 5.5 Skip the architecture papers, or spend one on a negative result

NOA-LSTM, xLSTM, xLSTMTime, xLSTM-Mixer, Mamba, TFT, EA-LSTM all make the same offer
P-sLSTM made. Expected movement on the metrics that separate models: approximately none.

If the thesis wants scope coverage, **NOA-LSTM is the cheapest one to run** — it is a
one-line change to the LSTM cell (`y_t = c_t ⊙ o_t` instead of `y_t = tanh(c_t) ⊙ o_t`),
about half a day including a correctness check against `nn.LSTM`, and it earns a
legitimate "we varied the cell and it moved nothing" paragraph. Note that the paper's
§3.7 contradicts its own title on which activation is removed, so the choice has to be
stated. Its cost is not the code but the Python-level timestep loop it forces in place
of the fused `nn.LSTM` kernel — roughly 5–20× slower per epoch at `T_CAL = 104`.

Everything else on that list is the same bet at 10–50× the price.

---

## 6. What this does not establish

- **One grid, one generating process.** `seasonal_4x4x10` panels are generated *by* a
  Pareto/NBD with seasonality, so the benchmark is correct by construction there. The
  §2 table measures how far the neural models fall below a ceiling on that process,
  not that they lose on real data.
- **On real panels the direction reverses for the benchmark.** On electronics,
  Pareto/NBD sits at **−63.7%** aggregate bias (§4.1) — it predicts barely a third of
  the transactions that occur — while the LSTM is at +2.4% on average. The
  §3 diagnosis is about a regime (high churn, sparse), not a verdict on either family.
- **The across-study spread is the other unsolved problem, and §3 does not address
  it.** The LSTM's +2.4% mean bias on electronics averages runs at −26% and +34%,
  sd 24. In the §2 table the sd is of the same order as the mean in every cell. A model
  that is right on average and unreliable per fit is what the study-suite design exists
  to report honestly; making it *reliable* is a separate line of work, and averaging a
  forecast over several independently refit models is the obvious untried thing.
- **The Transformer's non-monotone sparsity profile is unexplained** (§2). It may mean
  the two architectures fail for different reasons, in which case one fix will not serve
  both.
- **The archived neural arms were under-searched, and unequally.** 10 Optuna trials
  for the LSTM and 20 for the Transformer, against the 100 the single-panel suites use.
  So the LSTM's worse bias in §2 is confounded with half the search budget, and the two
  neural columns are not directly comparable to each other. The churn-axis finding does
  not rest on that comparison — it is within-model, and it holds separately for both —
  but any LSTM-vs-Transformer statement here does. The Pareto/NBD needs no tuning
  (its `n_trials=50` is recorded but inert, the entry being declarative), so its column
  is its true one. The grid now searches all three neural models at 100 trials, so this
  limitation is on the archived numbers, not on the grid.
- **Nothing here was re-run.** Every number is recomputed from archived `results.csv`
  files. If an archive is stale relative to the code that wrote it, this document
  inherits that.

---

## 7. Two loose ends found while reading

The second was closed by commit `2d815b3` while this document was being written; it is
kept, with what closed it, because the reasoning is what justifies the arm axis.

**The frozen benchmark is missing from the headline grid.**
`Studies/seasonal_4x4x10__ValendinLSTM/` contains `config.json` for its suites but
**zero `results.csv` files**. `grids/seasonal_4x4x10.py` declares `VALENDIN` in its
`models` tuple with `workers={"valendin_lstm": 0}` — the orchestrator rather than a
rented worker. So the arm was declared and never produced output. The §2 table has
three models in it because the fourth is not there, and a grid whose point is comparing
a contribution against published work is currently missing the published work.

**The unbounded triple needed a bounded arm beside it — CLOSED by `2d815b3`.**
A grid carrying only the unbounded `(t_x, x, T)` confounds the information with its
encoding, so it cannot answer the question that config exists to ask: does a neural model
close the gap when handed the same summary of a customer's history the Pareto/NBD
conditions on, or is the remaining gap about functional form rather than information?
Two arms separate information from representation; one arm measures their product.

`grids/seasonal_4x4x10.py` now crosses `AR_AXIS = {no_ar, ar_unbounded, ar_bounded}`
with `CLUSTER_AXIS = {no_cluster, kmeans_8}` — six arms, with `no_ar × no_cluster ×
valendin` reproducing the archived configuration so the §2 numbers remain the baseline
the rest are read against. That is the paired design, and it supersedes the separate
`grids/seasonal_4x4x10_ar.py` this section originally described.

That file is committed rather than deleted only because `docs/feature_engineering.md`
cites it as where the triple reaches a model as three continuous channels — a reference
that has been dangling, the file having never been tracked. When the file goes, that
citation moves to the `ar_unbounded` arm.

The point the arm axis inherits, from commit `8979946`: keeping the unbounded triple is
deliberate, not an oversight. The triple is "the right information in a form a rolled-out
neural model cannot use", and on these panels it is the *true* model's own sufficient
statistic — so the arm reproduces §4.3's blowup where the truth is known rather than
assumed.

---

## 8. The real panels: what the arm axis did on CDNOW and electronics

Run 2026-09-06 with `scripts/run_real_panel_arms.py` on a vast.ai fleet of eight
workers: the same six arms as the seasonal grid, plus two electronics-only arms with
the engineered time features removed, 20 studies per (model, panel, arm), 50 trials
(25 for the frozen benchmark), 50 Monte Carlo paths. 32 arm-suites, 640 studies,
~$2.50 of GPU. Everything below is the **ensemble** — the mean of the 20 forecasts,
scored once — for the reason §9 gives.

### It reproduces

Three of four shared arms land on the archived `ar_encoding` figures:

| | got | archived | |
|---|---|---|---|
| electronics `no_ar` | +20.0% | +22.0% | ok |
| electronics `ar_unbounded` | +196.7% | +235.0% | ok |
| cdnow `no_ar` | +6.1% | +2.0% | ok |
| cdnow `ar_unbounded` | +155.5% | +334.0% | flagged |

The fourth is a soft failure not worth leaning on: the archived CDNOW `ar_unbounded`
shards were +461% and +207%, sd 640 and 310. An arm whose own two halves differ by 254
points cannot be reproduced to a tolerance narrower than that. The arm is unstable; the
run is not.

### The frozen benchmark, finally

ValendinLSTM has never produced a number in this repo before — F11 refused it in both
grids. On CDNOW, at the arms it can read:

| CDNOW, ensemble | bias % | sd across 20 | MAPE | Spearman |
|---|---|---|---|---|
| **ValendinLSTM** `no_ar-no_cluster` | **−2.4** | 17.3 | **18.13** | 0.385 |
| LSTM `ar_bounded-no_cluster` | −10.0 | 13.7 | 18.29 | 0.418 |
| **Pareto/NBD** | −11.6 | — | 18.70 | 0.448 |
| ValendinLSTM `no_ar-kmeans_8` | +0.1 | 22.4 | 20.48 | 0.448 |
| Transformer `no_ar-kmeans_8` | +1.3 | 12.7 | 22.54 | 0.426 |

The published benchmark is the best model on this panel by MAPE and by bias, narrowly
ahead of our LSTM and the Pareto/NBD. That is the comparison the thesis exists to make
and it had never been run.

On **electronics** the picture inverts: Pareto/NBD sits at **−62.6%** bias and 65.4
MAPE, while every neural arm lands between −2% and +35%. The best is LSTM
`ar_bounded-kmeans_8` at **−1.6% ± 10.1**, the tightest distribution in the run. So the
benchmark is not uniformly strong — it is strong where the panel suits it.

### The unbounded-AR blowup is LSTM-specific

§4.3 reads out-of-range AR features as a property of the *encoding*. On real panels that
is only half true:

| CDNOW `ar_unbounded` | LSTM | Transformer |
|---|---|---|
| `no_cluster` | +155.5% (sd 343, max +1,480) | **+35.5% (sd 21)** |
| `kmeans_8` | +244.7% (sd 510, max +1,885) | **−29.8% (sd 17)** |

Same panel, same channels, same out-of-range hazard — and the attention model largely
absorbs what wrecks the recurrence. Electronics shows the same asymmetry (LSTM +196.7%
against the Transformer's +56.4%). This is a genuine finding and it is **not** what the
AR encoding ablation concluded, because that ablation only ever ran the LSTM
(`run_ar_encoding_ablation.py` declares one `ModelSpec`). The encoding is dangerous; the
recurrence is what makes it catastrophic.

### Clusters help here, unlike on synthetic panels

`kmeans_8` is in the best arm on electronics (`ar_bounded-kmeans_8`, −1.6%) and the best
Transformer arm on CDNOW (`no_ar-kmeans_8`, +1.3%). On the seasonal grid the cluster axis
was inert or harmful (§5 of the grid read). Behavioural clusters appear to buy something
on real heterogeneity that a Pareto/NBD-generated panel does not have.

### The tracking plot, which the tables hide

`figures/real_panel_arms__cdnow__weekly_aggregate.png`, written by
`scripts/run_real_panel_arms.py --plot` — each model's ensemble against the actual weekly
aggregate over the 38 holdout weeks, each model shown at its best arm **by MAPE**. Not by
`bias_percent`: selecting on bias picks whichever arm hits the total by cancellation,
which is the failure this figure exists to show, so it would hide the effect behind its
own symptom. (The first draft of the plotting code made exactly that mistake.)

The actual series decays from ~50 to ~30 transactions/week and is very noisy (29 to 85).
Pareto/NBD and the LSTM track that decay closely. **The Transformer does not: it flattens
after week 25 and ends near 53 against an actual of 30.** Its +1.3% aggregate bias — the
best of any model — is obtained by *cancellation*, under-predicting the first weeks and
over-predicting the last. Bias is a signed total and cannot see this; MAPE is an L1 on
the per-period curve and can, which is why its MAPE (22.5) is the worst in that table
while its bias is the best.

### Why every curve is smooth, and why that is correct

The plotted forecasts are near-straight lines while the actual jumps between 29 and 85.
That is not underfitting; it is three things, and the third is the important one.

**The model has no calendar input.** CDNOW's `PanelConfig` engineers no time features, so
`seq_cols` is `['Transactions']` in the `no_ar` arms and the target plus five activity
flags in the bounded ones. Nothing says what week it is, so during a rollout the only
quantity changing step to step is the customer's own sampled history, which decays
monotonically. A smooth curve is the only shape the input design permits — and every
model in the plot is smooth, not just the LSTM.

**There is nothing week-to-week to learn.** Detrended, the actual weekly totals have a
lag-1 autocorrelation of **+0.198** — effectively white. No history-conditioned model can
anticipate next week's spike, because last week's carries almost no information about it.

**The straight line is within 0.2 points of optimal.** Fit the best smooth curve to the
*actuals themselves* — an oracle no forecast could beat — and score it the same way:

| CDNOW | MAPE |
|---|---|
| oracle trend line fitted to the actuals | **18.10** |
| ValendinLSTM ensemble | 18.13 |
| LSTM ensemble | 18.29 |
| Pareto/NBD | 18.70 |

**MAPE ≈ 18 is a floor on this panel and all four models sit against it.** The ranking in
the table above is inside that ceiling and should not be read as a model ordering. This is
the single most important qualification on the CDNOW result.

What is left over is real but not learnable from these inputs: the weekly totals have
sd 13.1 where Poisson sampling on their mean would give 7.0, so the series is roughly
2x overdispersed. That excess is not autocorrelated, so it is not recoverable from
transaction history — it would need exogenous calendar or promotional covariates the
panel does not carry. That is the concrete answer to "could a better architecture do
better on CDNOW": not without different inputs.

Two things follow. First, this is the concrete reason `CONTEXT.md` defines **tracking**
separately from aggregate bias, and the reason `compute_forecast_metrics` returns three
numbers rather than one. **Never rank models on `bias_percent` alone.** Second, a model
that fails to decay is the §3 diagnosis showing up on a real panel: CDNOW's cohort dies
across the holdout, and the model with no absorbing state is the one that keeps selling
to it.

---

## 9. Score the ensemble, not the mean of the runs

A comparison in §8 looked wrong at first: Pareto/NBD's MAPE of 18.70 beat every neural
arm, whose means started at 22.3. The gap was an artefact of the reporting convention.

`mape_aggregate` is `100 · Σ|actual_t − pred_t| / Σ actual_t` — a positive magnitude, so
it is **convex** in the prediction. By Jensen, the error of the averaged forecast is at
most the average of the errors, and the slack is exactly the run-to-run scatter.
Pareto/NBD is a single deterministic MCMC fit with no scatter; every neural row was the
mean of 20 independent fits. The two are not comparable on a convex metric.

Averaging the 20 forecasts first, then scoring once — which is both what one would
deploy and what Pareto/NBD gets for free:

| CDNOW MAPE | mean of 20 | ensemble |
|---|---|---|
| ValendinLSTM `no_ar-no_cluster` | 22.54 | **18.13** |
| LSTM `ar_bounded-no_cluster` | 22.34 | **18.29** |
| LSTM `no_ar-no_cluster` | 24.39 | 21.64 |
| Transformer `no_ar-no_cluster` | 49.62 | 37.31 |
| Pareto/NBD (n=1) | 18.70 | 18.70 |

The benchmark's apparent lead disappears. **`bias_percent` is unchanged in every row** —
it is linear in the prediction, so averaging forecasts and averaging biases agree
exactly. Only the magnitude metrics move, which is a clean check that the ensembling is
doing what is claimed here and nothing else.

This is the ensembling §6 named as "the obvious untried thing", now measured: averaging
20 fits cuts CDNOW MAPE by 3 to 12 points, most where the scatter is worst. It costs
nothing — the fits already exist.

**Report both.** The distribution across replications is what says a single fit is
unreliable (bias sd of 10 to 40 points), and that is a real property of these models
that a thesis must not hide. The ensemble is what a practitioner would deploy and the
only figure comparable to a deterministic benchmark. Picking whichever flatters is how
this section's mistake happened in the first place.
