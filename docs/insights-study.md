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
cyclical pair). Each model gets 20 Optuna trials per panel and a 200-path forecast.

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
- **The grid's neural arms ran 20 Optuna trials per panel**, against the 100 the
  single-panel suites use. Both neural models are under-searched relative to a real
  study; the Pareto/NBD needs no tuning, so its column is its true one.
- **Nothing here was re-run.** Every number is recomputed from archived `results.csv`
  files. If an archive is stale relative to the code that wrote it, this document
  inherits that.

---

## 7. Two loose ends found while reading

**The frozen benchmark is missing from the headline grid.**
`Studies/seasonal_4x4x10__ValendinLSTM/` contains `config.json` for its suites but
**zero `results.csv` files**. `grids/seasonal_4x4x10.py` declares `VALENDIN` in its
`models` tuple with `workers={"valendin_lstm": 0}` — the orchestrator rather than a
rented worker. So the arm was declared and never produced output. The §2 table has
three models in it because the fourth is not there, and a grid whose point is comparing
a contribution against published work is currently missing the published work.

**The unbounded triple in the AR grid is deliberate — keep it, but pair it.**
`grids/seasonal_4x4x10_ar.py` (untracked at the time of writing) declares the same
unbounded `(t_x, x, T)` that §4.3 shows blowing up. That is not an oversight: commit
`8979946` states the reason explicitly — the triple is "the right information in a form a
rolled-out neural model cannot use", and that *is* the finding the config carries.

The refinement, not a correction: as written the grid confounds the information with its
encoding, so it cannot answer the question its own docstring poses ("does a neural model
close the gap when it is given the same summary of a customer's history that Pareto/NBD
conditions on? Or is the remaining gap about the functional form rather than the
information?"). Answering that needs a **bounded** arm on the same panels — the same
statistics in an encoding that stays in range — run beside the unbounded one. Two arms
separate information from representation; one arm measures their product.
