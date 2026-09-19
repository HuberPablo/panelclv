# An absorbing death state

Whether the count head should carry a **death** outcome `D` that, once drawn, forces
every later period of that customer's path to zero — and, if it should, what the loss
can possibly be, given that `D` is never observed in any panel.

Read `CLAUDE.md` for the model contract and `CONTEXT.md` for the vocabulary first, and
`docs/feature_engineering.md` before touching anything a rollout reads. This document
continues an argument those two started: `loss-functions.md` §5.6 shows that
zero-inflation and hurdle formulations are vacuous over a free softmax, and
`feature_engineering.md` shows that the recency clock the BTYD literature leans on is
capped by the calibration window and drifts out of range during the holdout. An
absorbing state is the first proposal in this package that answers both at once, and
the one place where "add a class to the softmax" is *not* vacuous.

**The verdict, up front.** The idea is sound but the obvious implementation cannot
train, and the motivation usually given for it — that simulated paths *resurrect* dead
customers — is not what the measurements show. §4 retracts it. What the measurements do
show is worse and more interesting: on the archived electronics study the neural
forecasts carry **no per-customer information at all** (Spearman ρ = 0.004 for the LSTM
against actual holdout totals), while the single number "weeks since last purchase"
carries ρ = 0.296 on its own — and on per-customer holdout totals the LSTM scores
*worse* than assigning every customer the same average path. An absorbing state is worth
building not because it stops resurrection but because it is the smallest architectural change that forces a
per-customer survival variable into a model that currently has none, and because it
extrapolates past the calibration window's recency cap **by construction**.

## Contents

1. [The proposal, and why the obvious version cannot train](#1-the-proposal-and-why-the-obvious-version-cannot-train)
2. [The formulation that can train](#2-the-formulation-that-can-train)
3. [What the panels say](#3-what-the-panels-say)
4. [The resurrection argument, retracted](#4-the-resurrection-argument-retracted)
5. [What an absorbing state can and cannot buy here](#5-what-an-absorbing-state-can-and-cannot-buy-here)
6. [What these measurements do not prove](#6-what-these-measurements-do-not-prove)
7. [What it costs to build in this package](#7-what-it-costs-to-build-in-this-package)
8. [What to run before building it](#8-what-to-run-before-building-it)
9. [Sources](#9-sources)

**Measured vs. read.** Everything in §3, §4 and §5 was computed here with the project
venv against the real panels and the real archived forecasts in `FOR_ANALYSIS/`; the
scripts are given inline and are short enough to paste. Nothing in this document is
taken on the authority of a docstring. The model-side numbers come from three archived
replications per model type of one study configuration, named in §3.3 — that
configuration carries **no autoregressive features**, which is the single most important
caveat in this document and is repeated wherever it bites.

---

## 1. The proposal, and why the obvious version cannot train

The proposal: give the head a `K+1`-th class `D`. When a rollout samples `D`, latch the
customer — emit zero for every remaining period instead of sampling again.

The rollout half is fine. The training half cannot work, and the reason is not a
detail:

> **`D` is never in the data.** A panel records counts. A dead customer and an alive
> customer who bought nothing in that week are the same row — a zero. There is no
> period whose target index is `D`.

`training/loop.py:106` computes `criterion(output.reshape(-1, K), targets.reshape(-1))`
against observed class indices. If `D` is class index `K`, that index never appears in
any target, so the `D` logit receives gradient from no positive example in the entire
training set. Cross-entropy drives it toward −∞, `P(D)` goes to zero, the latch never
fires, and the model that comes out is the model that went in.

Any fix has to start from the fact that **death is latent** — a state to be inferred,
never a label to be supervised. That single observation determines the rest of the
design.

## 2. The formulation that can train

### 2.1 Two states, one of them absorbing

Keep the count head at exactly `K` classes — the size `Embedder` derives from the target
column's cardinality (`models/embedders.py:87-96`), so this keeps constraint C1 of
`loss-functions.md` §1 intact — and add a **separate scalar hazard head**:

```
h_t      = sigmoid(w · hidden_t + b)     # P(die at t | alive at t-1), one number
p_t(·)   = softmax(logits_t)             # emission while alive, K classes, unchanged
                                         # emission while dead: point mass on zero
```

A separate head rather than a `K+1`-th softmax column is not cosmetic. A softmax forces
`D` to compete with the count classes on one normalisation, which makes the "die" and
"stay alive and buy nothing" logits differ by a constant the data cannot pin down; a
separate sigmoid leaves the count head exactly as it is today and adds one parameter
vector.

The latent state `z_t ∈ {A, D}` with `P(D → A) = 0` makes the model a two-state hidden
Markov model whose second state is absorbing and emits `δ₀`.

### 2.2 The forward recursion

Do not supervise the death time. **Marginalise over it.** With `α_t(·)` the joint
probability of the observations up to `t` and the state at `t`:

```
α_t(A) = α_{t-1}(A) · (1 − h_t) · p_t(y_t)
α_t(D) = [ α_{t-1}(D) + α_{t-1}(A) · h_t ] · 1[y_t = 0]

loss   = − log( α_T(A) + α_T(D) )
```

Exact, differentiable, `O(T · 2)` per customer, no EM outer loop and no sampling. In
practice it runs in log space with `logsumexp`. The `1[y_t = 0]` factor is what does all
the work: a single purchase anywhere kills the dead branch outright, because a dead
customer cannot buy.

### 2.3 What the loss does at later steps

This is the question the formulation exists to answer: *if the model has decided a
customer is dead, what is cross-entropy still doing in the periods after that?*

Differentiate the marginal likelihood above. By the Fisher identity its gradient equals
the expected gradient of the complete-data loss under the posterior over the latent
state, so with `γ_t = P(z_t = A | y_{1:T})` from a forward–backward pass:

```
∇ loss  =  − Σ_t  γ_t · ∇ log p_t(y_t)          ← the existing cross-entropy, weighted
           − (discrete-time survival term driving h_t against the soft labels γ)
```

**The death state turns cross-entropy into posterior-weighted cross-entropy.** Periods
after a likely death get weight `γ_t ≈ 0` and stop teaching the count head to predict
zeros; their signal is rerouted in full to the hazard head. Nothing is masked by hand —
the mask *is* the posterior, and it is soft, so a customer the data cannot classify
contributes partially to both. The death time never has to be known.

The posterior is nearly free to compute, because any `y_t > 0` proves the customer was
alive at every earlier period. Let `ℓ` be the last period with a purchase. Then
`γ_t = 1` for all `t ≤ ℓ` exactly, and `γ_t` decays only across the **trailing run of
zeros**. All survival learning lives in that trailing run — which is precisely the
recency statistic `t_x` that `benchmarks/pareto_nbd.py` conditions on. Arriving at the
benchmark's sufficient statistic from a different direction is a good sign the model is
written down correctly.

### 2.4 One architectural constraint, easy to violate

Under teacher forcing the network reads observed `y_{1:t-1}` and covariates, so `h_t`
and `p_t` are conditionally independent of the latent path given the data. **That is the
only reason two `α`s suffice instead of `2^T` paths.** Feed a "dead" flag back into the
recurrent input — the obvious thing to try, since it is what the rollout does — and the
emission at `t` depends on which latent path was taken, the forward recursion stops
factorising, and the exact loss is gone. The latch belongs in the rollout, not in the
training-time recurrence.

### 2.5 Two design notes worth taking

- **Link function.** With a complementary log-log hazard,
  `h_t = 1 − exp(−exp(η_t))`, a *constant* `η` per customer reproduces exponential
  dropout exactly. The neural model then **nests** the Pareto/NBD's death process, and
  its contribution can be stated precisely: a hazard that depends on history, covariates
  and season. A zero in a dead December week is weaker evidence of death than a zero in
  a live June week, and Pareto/NBD cannot say that.
- **Initialisation.** Initialise the hazard bias strongly negative (of the order of
  0.5%/period). Start it near 0.5 and the first epochs kill the whole cohort, `γ`
  collapses, and the count head trains on almost nothing.
- **A softer variant worth fitting alongside.** Allow a small learned resurrection
  probability `P(D → A) = r`. Absorbing death is the `r = 0` boundary case, the data
  picks, and "the fitted `r` was indistinguishable from zero" is itself a result.

## 3. What the panels say

### 3.1 The purchase rate keeps falling with silence

Bucket every holdout customer-period by how many periods the customer had been silent
going into it, and take the empirical purchase rate in the bucket. Configured windows;
cohort filter `require_calibration_activity`:

```python
# venv python; run from the repo root
import numpy as np, pandas as pd
df = pd.read_csv(PANEL).sort_values(["Id", "year", "week"])
per = df[["year","week"]].drop_duplicates().sort_values(["year","week"]); per["t"] = np.arange(len(per))
y = df.merge(per, on=["year","week"]).pivot(index="Id", columns="t", values="Transactions").fillna(0).to_numpy()
y = y[(y[:, :T_CAL] > 0).any(1)][:, :T_CAL + T_HOLD]        # cohort filter, configured window
a = (y > 0).astype(int)
since = np.zeros_like(a); run = np.zeros(len(y), int)
for t in range(y.shape[1]):
    since[:, t] = run; run = np.where(a[:, t] == 1, 0, run + 1)
# then bucket since[:, T_CAL:].ravel() and average a[:, T_CAL:].ravel() within buckets
```

| silent periods | CDNOW cells | CDNOW P(buy) | electronics cells | electronics P(buy) |
| --- | ---: | ---: | ---: | ---: |
| 0 (bought last period) | 1,799 | **0.1323** | 612 | **0.0980** |
| 1 | 1,559 | 0.0962 | 555 | 0.0523 |
| 2–3 | 2,670 | 0.0880 | 1,028 | 0.0438 |
| 4–7 | 4,336 | 0.0683 | 1,801 | 0.0294 |
| 8–11 | 3,538 | 0.0486 | 1,597 | 0.0263 |
| 12–15 | 3,011 | 0.0365 | 1,480 | 0.0176 |
| 16–23 | 5,171 | 0.0279 | 2,638 | 0.0171 |
| 24–31 | 6,849 | 0.0149 | 2,310 | 0.0169 |
| 32–47 | 27,776 | 0.0077 | 3,967 | 0.0146 |
| 48–63 | 24,556 | 0.0041 | 2,875 | 0.0122 |
| 64+ | 8,301 | **0.0019** | 24,245 | **0.0071** |

CDNOW falls by a factor of **69** from fresh to long-silent and never flattens.
Electronics falls by a factor of **14** and has a visible plateau between 16 and 48
weeks of silence before resuming its decline. Neither curve is flat, which is the
minimum the proposal needs: a customer's purchase rate is not a constant to be
recovered, it is a decaying function of how long they have been quiet.

That matters for what the current AR encoding can express. `feature_engineering.md`
establishes that the unbounded recency clock is catastrophic out of range (+235% bias on
electronics, +335% on CDNOW) and that **bounded activity flags to `K` fix it entirely**.
But a bounded flag set is non-injective past `K` by construction: every customer silent
for more than `K` periods lands in the same bucket and is assigned the same rate
forever. The measured curve keeps falling past any `K` these panels can fit. A bounded
encoding buys correctness at the price of a **floor**; an absorbing state gets the
continued decay back, because the survival product `Π_s (1 − h_s)` keeps shrinking while
every input the hazard head reads stays inside its fitted range. **That is the strongest
structural argument available for the proposal: unbounded extrapolation out of bounded
inputs.**

### 3.2 Silence is not explained by a quiet alive customer

| | CDNOW | electronics |
| --- | ---: | ---: |
| customers entirely silent through the holdout | **71.1%** (1,675 / 2,357) | **60.7%** (503 / 829) |
| purchase rate in the period after a purchase | 0.1323 / wk | 0.0980 / wk |
| ⇒ P(a customer buying at *that* rate is silent all holdout) | **0.46%** | **0.47%** |
| pooled holdout purchase rate | 0.0198 / wk | 0.0140 / wk |
| ⇒ P(a customer buying at *that* rate is silent all holdout) | 46.7% | 48.0% |

Read the two rate rows together. If customers went on buying at the rate they buy at
just after a purchase, essentially none of them would be silent for a whole holdout —
yet most of them are. Something removes customers from the active population. The
pooled rate reproduces the silent share almost exactly, but that is circular: the pooled
rate is already an average over the customers who left.

### 3.3 What the archived forecasts actually do

Three archived replications each of LSTM, Transformer and Pareto/NBD on the electronics
panel, configuration `cfg_2yTrain_1yPred_NoCov` (`FOR_ANALYSIS/`, refit predictions,
500 simulations, `T_CAL = 104`, `T_HOLD = 52`, `N = 829`). Customers are split by their
**calibration-end recency** — the number of silent weeks at the moment the forecast
starts — and the cell is the mean per-customer holdout total.

| weeks silent at forecast start | n | actual | LSTM | Transformer | Pareto/NBD |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–3 | 91 | **4.374** | 2.372 | 2.131 | 1.723 |
| 4–12 | 41 | 3.854 | 2.271 | 2.009 | 1.538 |
| 13–25 | 69 | 2.333 | 2.287 | 2.023 | 1.456 |
| 26–51 | 108 | 1.694 | 2.349 | 2.077 | 0.918 |
| 52+ | 520 | **1.090** | 2.292 | 2.008 | 0.247 |
| all | 829 | 1.770 | 2.307 | 2.032 | 0.661 |

The actual column falls by 4× across the groups. **The LSTM column is flat to three
decimal places** — 2.372 for a customer who bought last week, 2.292 for a customer
silent for a year or more. The Transformer is equally flat. Pareto/NBD spans 7×: it has
the shape and the wrong level, under-forecasting everywhere.

The rank correlations say the same thing without the bucketing. Spearman ρ between
predicted and actual per-customer holdout totals, each archived replication scored
separately (the three Pareto/NBD archives are byte-identical — one seeded result stored
three times — so it has no spread to report):

| model | ρ per replication | |
| --- | --- | ---: |
| LSTM | −0.0040, −0.0115, +0.0151 | **≈ 0** |
| Transformer | −0.0082, +0.0296, +0.0499 | ≈ 0.02 |
| Pareto/NBD | +0.3098 (×3) | **0.310** |
| calibration-end recency alone, as a predictor | — | **0.296** |
| 5-bucket recency lookup fitted on the holdout | — | 0.321 |

**A single integer per customer — weeks since last purchase — recovers 0.296 of the
0.321 that recency can possibly buy, and the neural models recover none of it.**

**The caveat that governs this whole subsection:** that configuration's `seq_cols` are
`["Transactions", "week_sin", "week_cos"]` — **no autoregressive features**. Recency was
never an input; the models could only infer it from the target's own history through the
hidden state across a 104-week warm-up, and evidently did not. So these numbers do
*not* show that an AR-equipped LSTM is recency-blind. What they show is that the target
history alone does not carry recency through a warm-up of this length, and
`feature_engineering.md` independently measures that bounded AR flags lift electronics ρ
from 0.027 to 0.267 — most of the way to Pareto/NBD, and still with a floor past `K`.

## 4. The resurrection argument, retracted

The usual motivation for an absorbing state, and the one this investigation started
from, is that an autoregressive rollout can **resurrect** a customer: at every holdout
step the sampler draws from a softmax with non-zero mass above class 0, so over a 52-week
horizon those small probabilities compound into purchases the customer never made.

That story does not survive the measurements, for two independent reasons.

**It is not what the forecast is scored on.** `compute_forecast_metrics` scores
`prediction_mean = simulations.mean(axis=0)`, a per-customer per-period *mean*. The
metrics are functionals of the marginal mean `E[y_{i,t}]`, not of path structure. A
model whose individual paths resurrect but whose marginal mean is right scores
perfectly. Resurrection is visible in a path and invisible to the metric except through
the mean it produces.

**The defect it is supposed to explain is present without any rollout.**
`feature_engineering.md` already measured this for the unbounded-AR blow-up: a
teacher-forced pass on electronics — true counts *and* true AR values fed at every step,
so no sampling and no feedback at all — reproduces +169% of the +235% bias. The fitted
conditional is wrong before a single path is sampled. Feedback amplification is real and
second-order; it is not the mechanism.

And §3.3 shows the thing actually wrong with the archived neural forecasts is not that
they over-predict the long-silent group *because paths resurrect*. It is that they
predict the same number for everybody. There is no survival signal to corrupt.

**The corrected argument.** An absorbing state is worth building because it installs a
per-customer latent variable whose sufficient statistic is the trailing zero-run — the
statistic §3.3 shows is worth ρ ≈ 0.30 and is currently unused — and because its
survival product decays without ever pushing an input out of its fitted range. Both of
those are properties of the fitted conditional. Neither requires anything to be said
about sampled paths. The rollout latch then follows for free, and is a consistency
requirement rather than a benefit: having trained a model in which death is absorbing,
it would be incoherent to simulate one in which it is not.

## 5. What an absorbing state can and cannot buy here

Scored with the package's single scoring authority
(`models.monte_carlo_forecasting.compute_forecast_metrics`), averaging the three
archived replications per model:

| model | RMSE (cell) | RMSE (customer total) | bias % | MAPE aggregate |
| --- | ---: | ---: | ---: | ---: |
| LSTM | 0.3767 | 3.5613 | +30.37 | 55.68 |
| Transformer | 0.3764 | 3.5275 | +14.83 | 44.87 |
| Pareto/NBD | 0.3758 | 3.4886 | −62.64 | 65.38 |
| constant path for every customer (fitted on the holdout) | 0.3754 | 3.5271 | 0.00 | 0.00 |
| 5-bucket recency lookup (fitted on the holdout) | **0.3734** | **3.3413** | 0.00 | 0.00 |

The last two rows are fitted on the very holdout they are scored on. They are not
forecasts; they are upper bounds on what their information can buy.

**Do not expect cell RMSE to move.** Every row lies within 0.9% of every other, and the
oracle recency lookup beats the constant path by 0.5%. At 98.6% zeros the per-cell
squared error is dominated by irreducible Bernoulli noise, and no amount of survival
modelling touches it.

`rmse_customer_total` — the same function's second RMSE, squaring the error in each
customer's holdout *total* rather than in each cell — is the one that moves, and it
carries the sharpest result in this document:

> **The LSTM (3.5613) is worse on per-customer totals than assigning every customer the
> same average path (3.5271).** Its per-customer variation is not merely uninformative,
> as ρ ≈ 0 already said; it is noise that actively costs accuracy. Pareto/NBD (3.4886)
> beats the constant baseline, and the oracle recency lookup (3.3413) shows the whole
> span available to recency information is about 6%.

So an absorbing state that improves the forecast will show up in **`rmse_customer_total`,
bias, MAPE aggregate, and per-customer rank correlation** — and a thesis chapter that
promises cell-RMSE gains from it will be embarrassed.

What it can plausibly buy, in descending order of confidence:

1. **A per-customer `P(alive)`** directly comparable to the Pareto/NBD's, which
   `benchmarks/pareto_nbd.py:257` already computes. Even at unchanged metrics this is a
   thesis artifact: the same quantity, from a neural model, on the same customers.
2. **Rank correlation**, the axis on which the archived neural models score zero.
3. **Continued decay past the AR encoding's `K`**, where a bounded flag set is flat by
   construction (§3.1).
4. **Aggregate bias on the long-silent majority** — 520 of 829 electronics customers sit
   in the 52+ group, where the LSTM predicts 2.29 against an actual 1.09.

And one thing it will *not* fix, worth stating because it looks like it should. The
electronics holdout does not decay in aggregate at all: the actual mean count per week
runs 0.0417 → 0.0239 → 0.0281 → 0.0425 across the four quarters of the horizon, ending
*above* where it started, because year-end seasonality dominates. **Death is a
cross-sectional effect here, not a time-path effect.** A survival envelope multiplied
into the forecast will fight the seasonal rise unless the hazard and the emission both
see the calendar. Give the hazard head the time features.

## 6. What these measurements do not prove

**A declining empirical hazard does not prove death.** This is the oldest identification
problem in the BTYD literature and §3.1 walks straight into it. A Gamma-mixed Poisson
population with *no dropout whatsoever* also produces a purchase rate that declines with
silence, because conditioning on a long silence selects the low-λ customers. Pure
heterogeneity predicts a power-law-ish decline toward a positive floor; heterogeneity
plus dropout predicts a decline that steepens toward zero. Eleven buckets on two panels
cannot cleanly separate those, and this document does not claim to. What settles it is
out-of-sample forecast performance — which is the experiment the package already runs.

**Both panels are single-cohort, so silence and calendar time are collinear.** 98.7% of
CDNOW customers and 93.2% of electronics customers make their first purchase within the
first 12 weeks of the panel. Every customer's clock is therefore nearly synchronised,
and "silent for `s` weeks" is very nearly "it is now week `s + 6`". **An empirical
hazard curve on a single-cohort panel cannot distinguish a customer-level survival
decline from a cohort-level calendar decline**, and CDNOW — the panel whose decay looks
most dramatic — is also the panel where the confound is tightest. The practical
consequence for the design is direct: if the hazard head cannot see calendar time, it
will absorb the cohort trend into the per-customer hazard and mispredict any customer
whose clock is out of phase with the cohort's.

**The §3.3 model numbers are one configuration, without AR features.** Stated in §3.3
and repeated here because it is the load-bearing caveat: three replications of one
archived study whose `seq_cols` are `["Transactions", "week_sin", "week_cos"]`. The
comparison an absorbing state actually has to win is against a **bounded-AR** LSTM, not
against this one.

## 7. What it costs to build in this package

Against the four constraints in `loss-functions.md` §1:

- **C1 (head is a softmax over `K` classes, logits `(B, T, K)`)** — kept, if the hazard
  is a separate scalar head. The model's `forward` then returns two tensors instead of
  one, which is a change of shape contract even though the count head is untouched.
- **C2 (target is a class index `(B, T)` long)** — kept unchanged.
  `training.loop._validate_targets` is satisfied.
- **C3 (rollout samples and feeds back)** — kept, plus the latch. This is per-model
  already: the registry declares the rollout function each model forecasts through
  (ADR-0006), so an absorbing rollout is a new entry, not a change to an existing one.
- **C4 (scored by `compute_forecast_metrics`)** — untouched.

The real friction is the training loop. `training/loop.py:106` and `:170` both compute

```python
loss = criterion(output.reshape(-1, num_target_classes), targets.reshape(-1))
```

which **flattens the time axis away before the loss sees it**. A sequence likelihood
cannot be computed from `(B*T, K)` rows; the forward recursion of §2.2 needs `(B, T)`
with the periods in order, and it needs the hazard tensor the criterion signature has
nowhere to receive. So this is not a sixth `loss_type` in `models/losses.py` — it is a
new model type with its own loss and its own training path. That is the honest cost, and
it is the largest single reason to run §8 before writing any of it.

## 8. What to run before building it

In order, cheapest first. The first two need no training at all.

1. **Re-run §3.3 against a bounded-AR study.** Every model number in this document comes
   from a no-AR configuration. If an LSTM carrying bounded activity flags already reaches
   ρ ≈ 0.27 and spreads properly across the recency groups, the absorbing state is
   competing for a much smaller margin than §3.3 suggests, and the case narrows to the
   floor-past-`K` argument of §3.1 alone.
2. **Split the residual by silence length.** Take the best bounded-AR archived forecast
   and plot per-customer residual against calibration-end recency. If the residual is
   flat in recency, the survival information is already in the model and an explicit
   state is redundant. If it grows with recency, the floor is real and measurable, and
   its size is the ceiling on what §2 can buy.
3. **Fit the hazard alone, without a network.** A two-state HMM whose hazard is a single
   population constant and whose emission is the empirical count distribution, fitted by
   the §2.2 recursion on the calibration window. It takes minutes, it is a strict special
   case of the proposal, and it separates "an absorbing state helps" from "a neural
   hazard helps". If the constant-hazard version captures most of the gain, build that
   and say so — it is a better thesis result than a network that cannot be interpreted.
4. **Only then** the neural hazard head, with the soft-resurrection variant (§2.5) fitted
   alongside as the null it has to beat.

## 9. Sources

Everything above is either measured here or cited from a document in this repo.

| Claim | Where |
| --- | --- |
| Head size is the target column's cardinality; `Embedder` refuses a mismatch | `src/panelclv/models/embedders.py:87-96` |
| The loss is computed on a time-flattened tensor | `src/panelclv/training/loop.py:106`, `:170` |
| Scoring authority and Monte Carlo mean | `src/panelclv/models/monte_carlo_forecasting.py:401`, `:542` |
| Pareto/NBD `p_alive` and the latent churn time τ | `src/panelclv/benchmarks/pareto_nbd.py:249-258` |
| Hurdle and zero-inflation are vacuous over a free softmax | `docs/loss-functions.md` §5.6 |
| The four constraints a loss proposal must satisfy | `docs/loss-functions.md` §1 |
| Unbounded recency blows the forecast up; bounded flags fix it; the failure is not exposure bias | `docs/feature_engineering.md`, "What this costs, and what fixes it" |
| Bounded AR flags lift electronics ρ from 0.027 to 0.267 | `docs/feature_engineering.md`, same table |
| Recency escapes its calibration range on 37.7% of electronics holdout cells | `docs/feature_engineering.md`, "Which ones to prefer" |
| Per-model rollout functions are declared in the registry | `docs/adr/` ADR-0006 |

The forward recursion of §2.2 is the standard HMM forward algorithm specialised to two
states with one absorbing; the posterior-weighted cross-entropy of §2.3 is the Fisher
identity applied to it, equivalently the M-step of EM with the E-step responsibilities
computed by forward–backward. Neither is novel and neither is cited to a paper here
because neither needs to be.
