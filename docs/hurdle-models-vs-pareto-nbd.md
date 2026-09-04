# Hurdle models against the Pareto/NBD

Whether the customer-base literature contains a **hurdle model** — a two-part model with
an explicit "did they transact" component and a separate positive-count component — that
has been measured against the Pareto/NBD on holdout transaction counts and won; and if
so, whether it could be implemented here.

Read `CONTEXT.md` first for the vocabulary (*calibration*, *holdout*, *rollout*,
*benchmark*, *reference implementation*), and `CLAUDE.md` for the model contract:
logits `(B, T, K)`, a count as a class not a quantity, cross-entropy on a class index,
evaluation by sampling-and-averaging. This document also assumes
[`docs/loss-functions.md` §5.6](loss-functions.md), which already settled what a hurdle
*likelihood* does over a free softmax head; §6 below is where that result becomes the
whole answer rather than a footnote.

Everything below was read from the primary source — the paper's own PDF, the package's
own source, or the publisher's own abstract — and every citation carries a URL. The
handful of claims that could only be reached through a secondary retelling or an
abstract are marked **ABSTRACT-ONLY** or **SECONDARY** inline and listed again in §8.
Nothing here was measured; this is a reading exercise, not a run.

## Contents

1. [The question and the short answer](#1-the-question-and-the-short-answer)
2. [What the Pareto/NBD is, and why it is hard to beat](#2-what-the-paretonbd-is-and-why-it-is-hard-to-beat)
3. [The models that do beat it — and none of them is a hurdle](#3-the-models-that-do-beat-it--and-none-of-them-is-a-hurdle)
4. [The models that are hurdles — and none of them beats it on counts](#4-the-models-that-are-hurdles--and-none-of-them-beats-it-on-counts)
5. [The one hurdle head worth copying, from outside marketing](#5-the-one-hurdle-head-worth-copying-from-outside-marketing)
6. [Why a hurdle head is already in this package](#6-why-a-hurdle-head-is-already-in-this-package)
7. [Ranked recommendation](#7-ranked-recommendation)
8. [What was not verified](#8-what-was-not-verified)
9. [Sources](#9-sources)

---

## 1. The question and the short answer

**There is no published hurdle model that has been shown to beat the Pareto/NBD at
forecasting per-customer transaction counts over a holdout window.** The head-to-head
does not exist in the literature, and the reason is structural rather than accidental:
the two families are asked different questions in different venues, so nobody has ever
run the comparison the question presupposes.

The evidence splits cleanly into three piles, and the whole document is the case for
that split.

| Pile | Members | Beats Pareto/NBD on holdout counts? | Hurdle-shaped? |
|---|---|---|---|
| Latent-attrition generalisations | Pareto/GGG, MBG/CNBD-k, PDO, GGompertz/NBD, Abe's HB | Yes, by 2–18% MAE where regularity exists | **No** |
| Neural sequence models | Valendin et al. LSTM | Yes — 6% RMSE, 18%→2% aggregate bias, 44% MAPE | Only in the weak sense of §6 |
| Two-part / hurdle models | GPPM, Vanderveld, Martínez, ZILN, two-stage GBM | **Never measured on counts against Pareto/NBD** | Yes (mostly) |

The single closest thing to an affirmative answer is Dew & Ansari's **GPPM**, which is a
per-period Bernoulli purchase-incidence model — the first half of a hurdle with the
second half omitted — and which beats the Pareto/NBD on holdout MAPE and RMSE on its own
two datasets (§4.1). But that result does not replicate: Valendin et al. re-ran the
GPPM on eight customer-base panels and found it *worse* than the Pareto/NBD on aggregate
bias, 26% against 18%.

And the recommendation that falls out of all of this is a negative one, for a reason that
is specific to this package rather than to the literature: **a hurdle factorisation over
a free `K`-way softmax is cross-entropy, term for term** (`loss-functions.md` §5.6,
verified numerically there on both panels). The model this package already trains *is* a
hurdle model — an explicit `q₀` and a nonparametric distribution over the positive
classes — with the two parts tied by a single softmax instead of estimated separately.
There is no hurdle to add. What a published hurdle model would contribute is not the
zero-mass structure but a *parametric, unbounded* positive part, and §5.7 of
`loss-functions.md` already scoped what that is worth: 5.45% of holdout mass on
electronics, nothing on CDNOW.

---

## 2. What the Pareto/NBD is, and why it is hard to beat

The model is Schmittlein, Morrison & Colombo (1987), and the version this package
reproduces is the hierarchical-Bayes MCMC estimator from R's BTYDplus — the one Valendin
et al. fit (ADR-0004). Its docstring in `src/panelclv/benchmarks/pareto_nbd.py` states
the structure; the short form is two latent per-customer rates with gamma population
priors, `λᵢ` for purchasing while alive and `μᵢ` for an exponential lifetime, with
`(x, t_x, T)` — repeat transactions, recency, age — as sufficient statistics.

Three properties make it a hard benchmark, and they matter for judging every candidate
below.

**It is fit to the aggregate.** The likelihood is an integral over unobserved churn
times, and what survives estimation is a population-level description that reproduces
cohort totals extremely well. Jerath, Fader & Hardie report a cumulative aggregate MAPE
of **1.35%** for the Pareto/NBD over a 39-week CDNOW holdout (Table 3,
[customer_death.pdf](https://business.columbia.edu/sites/default/files-efs/pubfiles/6057/customer_death.pdf)).
There is very little room above that.

**Its individual-level error is dominated by the zeros, and the zeros are easy.** In a
panel that is 96–98% zeros, a forecaster that predicts near-zero everywhere already
scores well on RMSE and MAE. Valendin et al. make this point sharply enough to be worth
quoting: taking the median implies "forecasting mostly zero events for all individuals, a
very poor prediction, which would nevertheless 'outperform' all models in this study
(with the exception of the LSTM) in terms of MAE" (accepted manuscript, fn. 19). This is
the same trap `loss-functions.md` §5.2 documents from the other direction, and it is why
this package scores `rmse`, `bias_percent` and `mape_aggregate` together rather than any
one of them.

**The negative results are real.** Wübben & von Wangenheim (2008, *Journal of Marketing*
72(3), 82–93) found that simple managerial heuristics — a fixed purchase-hiatus rule with
no free parameters — perform at least as well as the Pareto/NBD and BG/NBD on every
managerially relevant task they examined. Their per-dataset numbers (77% vs 74% correct
for an airline, 83% vs 75% for an apparel retailer, a tie on CDNOW) come from a
**SECONDARY** retelling; the *Journal of Marketing* article itself was unreachable.
The direction of the finding, however, has an independent primary cross-check: Platzer &
Reutterer report a heuristic column in their own Table 4, and there the probabilistic
models win in all six datasets — CDNOW MAE 0.77 for Pareto/NBD against 1.00 for the
heuristic, donations 0.35 against 0.3 (the heuristic edging it), groceries 1.52 against
2.5. Read together the two say the same thing: **the margin over a no-parameter rule is
small enough that it is contested**, which is the calibration to keep in mind whenever a
paper claims a few percent of lift.

Two further cautions belong here because they bear on how to read any published
comparison.

- **P(alive) is not a forecast.** "Dead Reckoning" (arXiv 2607.18623, July 2026) shows
  that BTYD `P(alive)` is the infinite-horizon limit of an observable family of
  finite-horizon repeat-purchase probabilities, and that specifications with *nearly
  identical observable forecasts* put the alive count anywhere from 3,654 to 27,734 on a
  seven-year 31,683-customer panel — a factor of 7.6, with a single default software
  parameter accounting for a 42% swing. On CDNOW the spread is 2.4×. A candidate model
  that wins on a churn-classification metric has not thereby won on forecasting.
- **Machine learning does not automatically win.** Chou et al. (2021, *EJOR*) integrate
  BG/BB estimates into a high-dimensional Lasso over ~100 predictors on a large online
  retailing dataset, and report that the integrated Lasso-BG/BB not only improves on
  BG/BB and on Lasso without it, but **outperforms two recurrent neural networks**, with
  the BG/BB prediction the single most influential feature (**ABSTRACT-ONLY**).

---

## 3. The models that do beat it — and none of them is a hurdle

This section exists because the brief asked for candidates to be cast widely and then
discarded with a reason. Every model here has a genuine, primary-source, holdout win over
the Pareto/NBD or its close relatives. None of them is a hurdle model, and saying why
sharpens what a hurdle would actually have to contribute.

### 3.1 Pareto/GGG — regularity, not zero-inflation

Platzer & Reutterer (2016), *Marketing Science* 35(5), 779–799
([author-hosted PDF](http://www.reutterer.com/papers/platzer&reutterer_pareto-ggg_2016.pdf)).

**Structure.** Identical to the Pareto/NBD except that the exponential inter-transaction
time is replaced by a **gamma** with a customer-level shape `k`, and `{k, λ, μ}` are given
gamma population priors. `k > 1` is regular timing, `k = 1` recovers the Pareto/NBD
exactly, `k < 1` is clumpy. The dropout process is unchanged — exponential lifetime. There
is no zero component anywhere: the model produces counts through a renewal process, and
`P(0)` is whatever the timing process implies. **Not a hurdle.**

**Head-to-head.** Table 4 reports customer-level MAE of the predicted number of holdout
transactions, together with the relative lift `1 − MAE_PGGG/MAE_PNBD`:

| Data set | Relative lift | MAE Pareto/GGG | MAE Pareto/NBD |
|---|---|---|---|
| CDNOW | +2% | 0.76 | 0.77 |
| Apparel and accessories | −2% | 0.44 | 0.43 |
| Donations | +16% | 0.29 | 0.35 |
| Groceries | +8% | 1.39 | 1.52 |
| Dietary supplements | +5% | 0.15 | 0.16 |
| Office supply | +4% | 0.28 | 0.30 |

(The INFORMS PDF renders decimal points as the glyph `0`; the table above decodes them,
and every row is internally consistent — donations 0.35 − 0.29 = 0.06 matches the paper's
absolute lift of +0.06.)

The paper's own summary of the pattern is that the stronger the timing regularity, the
greater the lift, and that on synthetic cohorts the relative lift reaches +20%. **On
CDNOW — a random-timing dataset, `k̂ = 1.0` — the win is 2%.** That is the honest headline
for anyone hoping a BTYD generalisation transfers to an arbitrary panel.

**Implementation.** Authoritative and available: `pggg.mcmc.DrawParameters` in
[BTYDplus](https://github.com/mplatzer/BTYDplus), by the paper's own first author.

**Fit here.** The likelihood is over inter-transaction *times*, continuous, and does not
factor into a per-period categorical distribution over counts without a further
discretisation step the paper never takes. It could be added as a second frozen benchmark
under ADR-0004 exactly the way Pareto/NBD was — a NumPy port of the BTYDplus sampler with
a closed-form or simulated period-count expectation — but it is not a model this
package's registry could carry a *rollout* for.

### 3.2 MBG/CNBD-k — the cheap version of the same idea

Reutterer, Platzer & Schröder (2021), *IJRM* 38(1), 194–215
([author-hosted PDF](http://www.reutterer.com/papers/reutterer&platzer&schroeder_2021.pdf)).

**Structure.** BG/NBD with the exponential inter-transaction time replaced by Erlang-k
(integer `k`, estimated at the cohort level rather than per customer) and the "modified"
beta-geometric dropout that lets a customer churn before any repeat purchase. Maximum
likelihood, no MCMC. **Not a hurdle** — the modification at zero is a churn opportunity,
not a separate count process.

**Head-to-head.** Table 4 gives MAE and MAE lift against a BG/NBD baseline across six
datasets:

| Model | CDNOW | Apparel | Donations | Groceries | Dietary | Office |
|---|---|---|---|---|---|---|
| MBG/NBD | +2.6% | +1.5% | +0.4% | +4.1% | +0.7% | −0.0% |
| BG/CNBD-k | +0.0% | +0.0% | +7.3% | +6.6% | +10.4% | +4.7% |
| MBG/CNBD-k | +2.6% | +1.5% | +12.8% | +9.2% | +14.2% | +6.5% |
| Pareto/GGG | +3.9% | −1.0% | +18.4% | +12.0% | +9.5% | +8.4% |

Two things are worth carrying away. First, on the two Poisson (`k = 1`) datasets — CDNOW
and apparel — the regularity models are *identical* to their base models by construction,
so the entire benefit is contingent on the panel having regular timing. Second, on
aggregate bias (Table 4c) the cheap MBG/CNBD-k beats the expensive Pareto/GGG on five of
six datasets, at a computational speedup the paper reports separately in Table 5.

**Implementation.** `mbgcnbd.EstimateParameters` in BTYDplus, again by the authors.

**Fit here.** Same verdict as §3.1: a candidate frozen benchmark, not a registry model
with a rollout.

### 3.3 The PDO model — a discrete death opportunity, which is *not* a hurdle

Jerath, Fader & Hardie (2011), *Marketing Science* 30(5), 866–880
([Columbia-hosted PDF](https://business.columbia.edu/sites/default/files-efs/pubfiles/6057/customer_death.pdf)).

This is the candidate closest to being hurdle-shaped by a superficial reading, and it is
worth stating precisely why it is not. The PDO model replaces the Pareto/NBD's
continuous-time exponential dropout with a **periodic** one: every `τ` time units the
customer flips a coin and either continues or drops out. As `τ → 0` it recovers the
Pareto/NBD; as `τ → ∞` it degenerates to the NBD. So the discrete per-period binary
event in the PDO is a *death* opportunity, not a *purchase* opportunity — it sits in the
attrition process, where the Pareto/NBD's exponential already sits, and the transaction
process remains a plain Poisson. There is no separate positive-count component. **Not a
hurdle.**

**Head-to-head**, Table 3, MAPE of repeat sales over CDNOW weeks 40–78:

| Model | Cumulative MAPE | Weekly MAPE |
|---|---|---|
| Pareto/NBD (`τ → 0`) | 1.35% | 20.89% |
| PDO, `τ = 3.001` | 0.85% | 19.06% |
| PDO, `τ = 10.001` | 0.70% | 19.18% |
| BG/NBD | 2.6% | 19.4% |
| NBD (`τ → ∞`) | 10.37% | 36.22% |

These are **cohort-level** figures, not per-customer. The authors are unusually candid
about what they add up to: "the improvements in predictions for the holdout period are
not especially dramatic", and "we continue to encourage using the Pareto/NBD model when
the manager's primary goal is forecasting purchases." A model that improves the log
likelihood substantially for one extra parameter and moves the holdout needle by half a
percentage point of aggregate MAPE is the single best illustration in this document of
how tight the ceiling is.

### 3.4 Gamma/Gompertz/NBD — flexible lifetime, marginal forecast

Bemmaor & Glady (2012), *Management Science* 58(5), 1012–1021,
[doi:10.1287/mnsc.1110.1461](https://doi.org/10.1287/mnsc.1110.1461). The INFORMS page
403'd; what follows is **ABSTRACT-ONLY / SECONDARY**.

The model replaces the Pareto lifetime with a **Gompertz** one, which is not memoryless
and whose density can have a mode at zero or an interior mode. Reported: over six
datasets the G/G/NBD gives a notable log-likelihood improvement over the Pareto/NBD in
four, and "on the average" slightly better forecasts of the mean number of transactions.
**Not a hurdle** — again a lifetime distribution swap. Implemented authoritatively as
`ggomnbd` in [CLVTools](https://cran.r-project.org/web/packages/CLVTools/index.html),
whose current model list is Pareto/NBD, Extended Pareto/NBD with time-varying covariates,
BG/NBD, GGom/NBD and Gamma/Gamma.

### 3.5 The Valendin et al. LSTM — the largest measured margin, and it is already here

Valendin, Reutterer, Platzer & Kalcher (2022), *IJRM* 39(4), 988–1018;
[accepted manuscript](https://ars.els-cdn.com/content/image/1-s2.0-S0167811622000180-am.pdf),
[code](https://github.com/valendin/rfm2lstm). This is the model
`benchmarks/valendin_lstm.py` reproduces.

Its benchmarks are exactly the three relevant ones: the Pareto/NBD (BTYDplus MCMC), the
Pareto/GGG (BTYDplus MCMC), and Dew & Ansari's GPPM (the authors' own Stan code). Its
metrics are exactly this package's three. Table 4 of the accepted manuscript is a bitmap
and its per-dataset cells could not be read; the paper's own prose summary of it,
verbatim from §4.2, is:

> Overall, individual-level RMSE improves by an average 6% across all 8 settings,
> compared to the Pareto/NBD (4% and 9% compared to Pareto/GGG and GPPM, respectively).
> As for the two cohort-level accuracy metrics, the Base LSTM model is particularly
> strong, with an average aggregate bias of just 2% across all scenarios (18%, 15% and
> 26% for Pareto/NBD, Pareto/GGG and GPPM, respectively). The MAPE is improved most by
> the Base LSTM in the more seasonal settings […] and is reduced by 44% on average
> overall, compared to the next best benchmark, the Pareto/GGG.

The LSTM performs best in all eight empirical settings on all three metrics. That is a
much larger margin than any BTYD generalisation in §3.1–§3.4 achieves, and the
explanation the authors give is not about the zeros: it is that the Pareto/NBD family
cannot represent "non-stationary behavior other than attrition" — seasonality, habit
formation, a customer whose rate rises. Their opportunity-customer analysis makes it
concrete: five times out of eight the Pareto/NBD and Pareto/GGG fail to identify *any*
customer who transacts more in the holdout than in calibration.

**Is it a hurdle?** In the weak sense of §6 below, yes: the head is a softmax over
transaction-count classes, so `P(0)` is an explicit free parameter and the positive
classes carry their own free mass. It is not a hurdle in the sense the question means —
there is no separate binary sub-model, no separate estimation, no truncated count
distribution.

---

## 4. The models that are hurdles — and none of them beats it on counts

### 4.1 GPPM — a hurdle with the second half missing

Dew & Ansari (2018), *Marketing Science* 37(2), 216–235; the version read here is the
[Wharton-hosted job-market paper](https://marketing.wharton.upenn.edu/wp-content/uploads/2017/08/11-07-2017-Dew-Ryan-PAPER-Ansari_BNP_CBA-JMP.pdf),
not the final journal version, so section and table numbering may differ.

**Structure**, from the paper's Equation 1:

```
Pr(y_ij = 1) = logit⁻¹[ α(t_ij, r_ij, ℓ_ij, q_ij) + z'_i γ + β_i ]
```

`y_ij` is a **binary** purchase indicator for customer `i` at observation `j`; `α(·)` is a
latent propensity function over calendar time, recency, customer lifetime and purchase
number, given additive Gaussian-process priors; `β_i` is an unobserved-heterogeneity
random effect. The authors state plainly that they "use the words purchasing and spending
interchangeably to refer specifically to purchase incidence."

So the GPPM is a **per-period discrete hazard** — precisely the first stage of a hurdle,
with no second stage. It has no positive-count component at all, because in its
application (single-product game purchases) there is "minimal variability in spend
amount". Under this package's contract that is a `K = 2` head.

**Head-to-head**, Table 2 of the working paper, MAPE (first row) and RMSE (second row) in
the 30-day holdout of two mobile games:

| Model | Life Simulator MAPE / RMSE | City Builder MAPE / RMSE |
|---|---|---|
| GPPM | **0.24 / 22.54** | **0.32 / 20.97** |
| Pareto-NBD | 0.33 / 32.10 | 0.45 / 27.80 |
| BGNBD | 0.31 / 30.04 | 0.61 / 37.41 |
| Log-Logistic hazard | 0.67 / 59.35 | 0.77 / 46.55 |
| LPM (linear propensity) | 0.26 / 30.02 | 0.58 / 49.53 |
| SSPM (state-space propensity) | 0.17 / 20.59 | 0.38 / 27.16 |

**This is the affirmative answer, and it is weaker than it looks, for three reasons.**
First, the metrics are computed on the *daily aggregate* spending series — the paper's
Figure 10 plots exactly that — so this is a cohort-tracking result, not a per-customer
one; it is closer to `mape_aggregate` than to `rmse`. Second, the datasets are two mobile
games with dense daily activity and strong calendar-time events, which is the regime the
GPPM's calendar-time GP was built for and the regime a recency/frequency model is worst
in; the authors say as much. Third and most damaging: **it does not replicate.** Valendin
et al. fit the GPPM with the authors' own Stan code on eight customer-base panels and
found an average aggregate bias of **26%, against the Pareto/NBD's 18%**, and an RMSE the
LSTM beat by 9% versus 6% for the Pareto/NBD. Valendin also report the cost: 800 HMC
iterations took approximately eight days on a 5 GHz CPU for a 159-step calibration
window, because the sampler is `O(n³)` in calibration length.

**Fit here.** Structurally excellent — a per-period Bernoulli is a two-class categorical
head, and it reads recency, lifetime and purchase number, all of which
`data_preparation`'s AR features already compute (`docs/feature_engineering.md`). What
kills it is the estimator, not the shape: additive GPs under HMC over a 104-week
calibration window, with a cost curve that makes a study suite impossible, for a model
that lost to the Pareto/NBD in the one independent replication that exists.

### 4.2 Two-stage machine-learning CLV — hurdles, but on money

Three representatives, all genuinely hurdle-shaped, none of them scored on transaction
counts against a Pareto/NBD.

**Lin et al. (2026), "A Two-Stage Hurdle Gradient-Boosting Framework"**, *Applied
Sciences* 16(13), 6550, [doi:10.3390/app16136550](https://doi.org/10.3390/app16136550).
The cleanest published hurdle-versus-BTYD head-to-head that exists, and it is on the
wrong target. Stage 1 is an XGBoost classifier for `P(y > 0 | X)`; Stage 2 is a gradient
boosting regressor for `E[y | y > 0, X]` fit on buyers only; the prediction is the product
`P × E`. UCI Online Retail II, 4,026 customers, 62.5% zero spending, a temporal split at
9 November 2010 with a 44-day prediction window. Table 5:

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Two-stage CatBoost | **0.522** | 359.54 | **1054.72** |
| Two-stage LightGBM | 0.465 | 375.78 | 1115.32 |
| Two-stage XGBoost | 0.435 | 370.50 | 1146.21 |
| BG/NBD + Gamma–Gamma | 0.395 | 391.08 | 1455.88 |
| Single-MSE XGBoost | 0.385 | **354.47** | 1196.14 |
| XGB–Tweedie | 0.309 | 397.34 | 1267.07 |

The hurdle beats the BTYD baseline, 0.522 against 0.395 R², and holds up under
out-of-time validation on 2011 Q4 (0.459 against 0.391 for single-stage CatBoost). But
the target is **net monetary spend**, the metric is R² on an arcsinh-transformed amount,
the benchmark is BG/NBD + Gamma–Gamma rather than the Pareto/NBD, and the Stage-1
classifier's ROC-AUC is 0.739 on a 63.9% non-buyer base. Note also that the single-stage
MSE model has the *lowest* MAE of everything in the table, and that XGB–Tweedie has the
highest top-20% F1 — the paper's own conclusion is that "relative strengths depend on the
evaluation criterion", which is the same warning `loss-functions.md` §4 gives about this
package's own metrics.

**Wang, Liu & Miao (2019), the ZILN loss**, [arXiv:1912.07753](https://arxiv.org/abs/1912.07753),
implemented as [google/lifetime_value](https://github.com/google/lifetime_value). The
loss is the negative log-likelihood of a zero-inflated lognormal — a point mass at zero
mixed with a lognormal — so the network learns purchase propensity and monetary value
simultaneously from one head. Genuinely hurdle-shaped, genuinely well-implemented, and
genuinely about **money**: the target is LTV, the recommended metric is the normalised
Gini coefficient for ranking high-value customers, and the BTYD family appears only in
related work. No Pareto/NBD comparison. A lognormal positive part is meaningless for
integer counts.

**Vanderveld, Pandey, Han & Parekh (2016), KDD**,
[doi:10.1145/2939672.2939693](https://doi.org/10.1145/2939672.2939693). A production
Groupon system: a binary churn classifier followed by regression models for average order
value and order frequency on the non-churned. Textbook hurdle. Its stated novelty is
engagement features, not the two-stage structure, and it reports no BTYD comparison
(**ABSTRACT-ONLY**).

### 4.3 Martínez et al. — a classifier, not a hurdle

Martínez, Schmuck, Pereverzyev, Pirker & Haltmeier (2020), *EJOR* 281(3), 588–596,
[doi:10.1016/j.ejor.2018.04.034](https://doi.org/10.1016/j.ejor.2018.04.034)
(**ABSTRACT-ONLY** — the publisher abstract via the Semantic Scholar Graph API).

Worth including because it is frequently cited as the machine-learning answer to
Pareto/NBD, and it is not. The framework predicts **whether** a customer will purchase
within a near-future time frame, from features derived from times and values of previous
purchases, refreshed monthly; gradient tree boosting wins; the headline is 89% accuracy
and 0.95 AUC on >10,000 customers. There is no count component and therefore no hurdle,
and the abstract names no probabilistic benchmark. A binary classifier evaluated by AUC
cannot be compared to a Pareto/NBD evaluated by holdout RMSE, and this paper does not
attempt it.

### 4.4 The classical device the BTYD literature actually uses: the spike at zero

For completeness, because it is the nearest thing the buy-till-you-die tradition has to
zero-inflation. Fader, Hardie & Lee note in the BG/NBD paper itself that the model could
be extended to a segment of "hard core" never-buyers with one additional parameter `π`,
and explicitly decline: they did not consider the problem severe enough to warrant it as
part of the basic model (**SECONDARY** — the passage was found through a course-notes
retelling; the working-paper PDF at
[brucehardie.com](https://www.brucehardie.com/papers/bgnbd_2004-04-20.pdf) discusses the
"zero class" at length but the `π` extension sentence was not located in that version).

That the field's own authors looked at a zero-inflation extension and judged it not worth
a parameter is, on its own, a fair summary of why §1's answer is what it is.

---

## 5. The one hurdle head worth copying, from outside marketing

The best-specified hurdle *decoder* found anywhere in this search is not a CLV paper at
all. Muşat & Căbuz (eMAG), **"Switch-Hurdle: A MoE Encoder with AR Hurdle Decoder for
Intermittent Demand Forecasting"**, [arXiv:2602.22685](https://arxiv.org/abs/2602.22685),
February 2026.

The encoder is a Top-1-routed mixture-of-experts Transformer and is irrelevant here. The
head is not. From the decoder state `h_t` it emits a positive-demand probability and a
negative-binomial mean/dispersion pair:

```
p⁺_t = σ(w_p ᵀ h_t)
μ_t  = softplus(w_μ ᵀ h_t + b_μ)
α_t  = softplus(w_α ᵀ h_t + b_α)
p_0,t = (1 + α_t μ_t)^(−1/α_t)               # the NB's own zero mass

P(Y_t = 0)     = 1 − p⁺_t
P(Y_t = y > 0) = p⁺_t / (1 − p_0,t) · P_NB(y; μ_t, α_t)
```

That is Mullahy's hurdle exactly: a Bernoulli gate and a **zero-truncated** negative
binomial for the positive part, renormalised by `1 − p_0,t`. The training loss is binary
cross-entropy for the gate plus the truncated-NB negative log-likelihood for the positive
counts, and the decoder is autoregressive — it attends to its own previous prediction plus
future covariates — so the sampling contract matches this package's rollout almost
exactly.

Results, on M5 (30,490 daily SKU series, horizon 28, context 56) and a ~40,000-series
proprietary retail panel:

| Model | M5 WRMSSE | M5 RMSE | M5 MASE | Internal WAPE | Internal MASE |
|---|---|---|---|---|---|
| PatchTST | 1.0393 | 2.4562 | 0.9471 | 81.22% | 0.8478 |
| DeepAR | 0.7895 | 2.9534 | 0.9087 | 64.86% | 0.6770 |
| TFT | 0.6932 | 2.4686 | 0.8983 | 55.60% | 0.5803 |
| TSMixer | 0.6403 | — | — | — | — |
| Switch-Hurdle | **0.6307** | 2.4744 | 0.8992 | **53.99%** | 0.5865 |

The margins are thin — the paper's own phrasing is "coming slightly ahead of the TFT
models in some cases" — and **no BTYD model appears anywhere in it**. It is not a
candidate benchmark. It is the reference for how a hurdle head is wired to an
autoregressive decoder, if one were ever wanted here.

---

## 6. Why a hurdle head is already in this package

This is the section the whole document exists to reach, and it does not depend on any of
the literature above.

`loss-functions.md` §5.6 proves — and verifies numerically to ten decimals on both panels
— that writing Mullahy's hurdle over the *same* `K`-way softmax and training by log
likelihood **is cross-entropy, term for term**:

```
−[ 1{y=0} log q₀ + 1{y>0} ( log(1 − q₀) + log( q_y / (1 − q₀) ) ) ] = −log q_y
```

The chain rule is not an approximation there. A hurdle factorisation over a free
categorical head is a re-parameterisation of the loss this package already minimises,
with no new degrees of freedom. It differs only if the two terms are weighted unequally —
and unequal weighting on a strictly proper loss is precisely what §5.2 rules out and what
ADR-adjacent commit `0d61deb` ("Refuse class weights on a strictly proper loss") already
refused in code.

So the multinomial LSTM in `benchmarks/valendin_lstm.py`, and the Transformer in
`models/`, are already hurdle models in every sense that affects the forecast:

| Hurdle component | What this package already has |
|---|---|
| Binary zero/non-zero gate | `q₀` — one free softmax coordinate, per customer per period |
| Positive-count distribution | `q₁ … q_{K−1}`, nonparametric, no functional form assumed |
| Joint estimation | One cross-entropy, one backward pass |
| Sampling from the mixture | `Categorical(softmax(logits)).sample()` in the rollout (C3) |

What the categorical head gives up relative to a *parametric* hurdle is exactly one thing:
**support above `K − 1`**. And `loss-functions.md` §5.7 has already priced that: on CDNOW
the holdout maximum is 3 against `K = 5`, so there is nothing above the head at all; on
electronics 5.45% of holdout mass lies above the cap, with `var/mean = 4.15` in the
holdout and a 26-count week that the head cannot reach. A truncated-NB positive part in
the Switch-Hurdle style is the *only* variant of "add a hurdle" that would change a number
here, and it would do so by relaxing the class cap, not by modelling the zeros better.

That change breaks C1 and C2 — a parametric count likelihood is not a softmax over `K`
classes and its target is not a class index — and `CLAUDE.md` states that any new model
keeps the categorical shape. It is therefore an ADR, not an experiment.

---

## 7. Ranked recommendation

**1 — Implement nothing, and record the negative result.** The question "which hurdle
model beats the Pareto/NBD" has no answer in the literature, and the reason is worth a
paragraph in the thesis rather than a model in the registry: the hurdle structure is a
remedy for a *parametric* count distribution's inability to place enough mass at zero,
and this package's head has no such constraint. §6 is the finding. It composes with
`loss-functions.md` §5.6 into a single defensible claim — the categorical-head contract
subsumes the hurdle family — which is stronger than any of the marginal wins in §3.

**2 — If a second frozen benchmark is wanted, take the Pareto/GGG.** It is the only model
in this document that beats the Pareto/NBD on per-customer holdout counts with a primary
source, an authoritative implementation by the paper's own author (BTYDplus
`pggg.mcmc.DrawParameters`), and a precedent in the very paper this thesis reproduces —
Valendin et al. benchmark against it, so adopting it makes this package's comparison a
superset of theirs rather than a different one. **What would need building:** a NumPy port
of the BTYDplus Pareto/GGG Gibbs sampler alongside the existing Pareto/NBD port, a
period-count expectation for the gamma renewal process to replace the closed form in
`compute_pareto_predictions`, a registry entry with no builder and no rollout (ADR-0006
already permits this — Pareto/NBD's entry is declarative), and a
`scripts/validate_pareto_ggg_benchmark.py` cross-checking against the installed R package,
per ADR-0004's rule that the validation script *is* the ADR. Expect roughly the effort of
the existing Pareto/NBD port. **Expect a small win:** +2% MAE on CDNOW, and CDNOW is
`k̂ = 1.0`, meaning the Pareto/GGG collapses to the Pareto/NBD there by construction. The
electronics panel's regularity has never been measured; Wheat & Morrison's estimator is
two lines and would settle in advance whether this is worth doing at all.

**3 — If cheapness matters more than fidelity, take MBG/CNBD-k instead.** Maximum
likelihood rather than MCMC, in BTYDplus, beats its base model by 5–14% MAE where
regularity exists and beats the Pareto/GGG on aggregate bias in five of six datasets. It
is not, however, a model Valendin et al. benchmarked, so it makes this package's
comparison different from theirs rather than a superset.

**4 — Do not port the GPPM.** It is the only per-period two-part model in the CLV
literature with a holdout win over the Pareto/NBD, and every part of that sentence has a
caveat: aggregate metrics, two mobile games, and a failed replication on eight
customer-base panels at eight days per fit.

**5 — If a hurdle *contribution* is ever wanted, it is the Switch-Hurdle head, and it is
an ADR.** Bernoulli gate plus zero-truncated negative binomial on the existing LSTM or
Transformer trunk, sampled step by step through the existing rollout. The only thing it
buys over the current head is unbounded support, which matters on electronics and not on
CDNOW. Write the ADR first, because it breaks C1 and C2.

---

## 8. What was not verified

- **Bemmaor & Glady (2012).** The INFORMS article page returned 403. The "four of six
  datasets improve the log-likelihood; on average slightly better forecasts of the mean
  number of transactions" claim in §3.4 is **SECONDARY**, from a search-engine summary of
  the abstract. Not load-bearing: the model is not a hurdle either way.
- **Wübben & von Wangenheim (2008).** The *Journal of Marketing* article was unreachable.
  The 77/74 and 83/75 figures in §2 are **SECONDARY**. The direction of the finding has an
  independent primary cross-check in Platzer & Reutterer's Table 4 heuristic column, and
  that cross-check partly *disagrees* — the probabilistic models beat the heuristic in all
  six of their datasets. Both are reported.
- **Valendin et al. Table 4.** The per-dataset RMSE/bias/MAPE cells are a bitmap in the
  accepted manuscript and could not be extracted. Only the authors' own prose averages,
  quoted verbatim in §3.5, are used.
- **Dew & Ansari.** Read from the 2017 Wharton job-market-paper PDF, not the 2018
  *Marketing Science* version. Equation and table numbers may have changed. The
  Valendin replication figures are from a different source and do not depend on this.
- **Martínez et al. (2020), Chou et al. (2021), Vanderveld et al. (2016).**
  **ABSTRACT-ONLY.** No results table was read for any of them. None is load-bearing:
  the first two are cited as *negative* evidence (not a hurdle; ML does not automatically
  win), the third only for its structure.
- **Fader, Hardie & Lee's "hard core never-buyer" remark** (§4.4) is **SECONDARY**; the
  sentence was not located in the working-paper PDF actually read.
- **Batislam, Denizel & Filiztekin (2007)** was not read directly. The MBG/NBD figures in
  §3.2 come from Reutterer et al. (2021) Table 4, which benchmarks it against BG/NBD
  rather than Pareto/NBD. Note also that an **erratum** to the MBG/NBD conditional
  expectation exists (*IJRM* 2008,
  [doi:10.1016/j.ijresmar.2008.02.001](https://doi.org/10.1016/j.ijresmar.2008.02.001)),
  which any implementation would have to honour.
- **The search itself.** Crossref and the Semantic Scholar Graph API were queried directly
  for `hurdle`/`zero-inflated` crossed with `Pareto/NBD`, `customer base analysis` and
  `customer lifetime value`; arXiv's API was swept for `Pareto/NBD`, `customer base
  analysis`, `buy till you die`, and `zero-inflated`/`hurdle` crossed with forecasting.
  The absence claimed in §1 is an absence **in those indices**, not a proof.

---

## 9. Sources

| Claim | Source |
|---|---|
| Pareto/NBD: two latent rates, gamma priors, `(x, t_x, T)` sufficient | Schmittlein, Morrison & Colombo (1987); derivation in Hardie note 009 — http://www.brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf |
| Pareto/GGG structure (gamma ITT with per-customer shape `k`, exponential lifetime unchanged); Table 4 MAE and relative lift on six datasets; heuristic comparison column | Platzer & Reutterer, *Marketing Science* 35(5):779–799 (2016) — http://www.reutterer.com/papers/platzer&reutterer_pareto-ggg_2016.pdf (doi:10.1287/mksc.2015.0963) |
| MBG/CNBD-k structure; Table 2 descriptives; Table 4a/b/c MAE, MAE lift and bias across six datasets; Table 5 computational speedup | Reutterer, Platzer & Schröder, *IJRM* 38(1):194–215 (2021) — http://www.reutterer.com/papers/reutterer&platzer&schroeder_2021.pdf (doi:10.1016/j.ijresmar.2020.09.002) |
| PDO model: periodic death opportunity, `τ→0` recovers Pareto/NBD; Table 3 CDNOW cumulative and weekly MAPE; "improvements … not especially dramatic"; "continue to encourage using the Pareto/NBD … when the manager's primary goal is forecasting purchases" | Jerath, Fader & Hardie, *Marketing Science* 30(5):866–880 (2011) — https://business.columbia.edu/sites/default/files-efs/pubfiles/6057/customer_death.pdf (doi:10.1287/mksc.1110.0654) |
| Gamma/Gompertz/NBD flexibility and forecast comparison (**SECONDARY**) | Bemmaor & Glady, *Management Science* 58(5):1012–1021 (2012) — https://doi.org/10.1287/mnsc.1110.1461 |
| LSTM benchmarked against Pareto/NBD, Pareto/GGG and GPPM; RMSE 6% / bias 2% vs 18% / MAPE −44%; best in all eight settings; opportunity-customer F1; fn. 19 against MAE; GPPM `O(n³)` HMC cost | Valendin, Reutterer, Platzer & Kalcher, *IJRM* 39(4):988–1018 (2022) — accepted manuscript https://ars.els-cdn.com/content/image/1-s2.0-S0167811622000180-am.pdf (doi:10.1016/j.ijresmar.2022.02.007); code https://github.com/valendin/rfm2lstm |
| GPPM: binary purchase indicator, inverse-logit discrete hazard, additive GP priors over calendar time / recency / lifetime / purchase number (Eq. 1); "purchase incidence"; Table 2 holdout MAPE and RMSE for GPPM, Pareto-NBD, BGNBD, log-logistic, LPM, SSPM | Dew & Ansari, *Marketing Science* 37(2):216–235 (2018); working-paper version read — https://marketing.wharton.upenn.edu/wp-content/uploads/2017/08/11-07-2017-Dew-Ryan-PAPER-Ansari_BNP_CBA-JMP.pdf (doi:10.1287/mksc.2017.1050) |
| Two-stage hurdle GBM: `P(y>0|X)` × `E[y|y>0,X]`; UCI Online Retail II, 4,026 customers, 62.5% zeros, temporal split 2010-11-09; Table 5 R²/MAE/RMSE incl. BG/NBD + Gamma–Gamma at R²=0.395; Table 6 top-20% F1; Table 7 OOT | Lin, Chen, Kuo, Yen & Lo, *Applied Sciences* 16(13):6550 (2026) — https://doi.org/10.3390/app16136550 |
| ZILN loss = NLL of a zero-inflated lognormal; target is monetary LTV; normalised Gini as the recommended metric; BTYD in related work only | Wang, Liu & Miao (2019) — https://arxiv.org/abs/1912.07753; implementation https://github.com/google/lifetime_value |
| Switch-Hurdle head: `p⁺_t = σ(w_pᵀh_t)`, NB `(μ_t, α_t)`, `p_0,t=(1+α_tμ_t)^{−1/α_t}`, truncated-NB positive part (Eq. 8–14); BCE + truncated-NB NLL; cross-attention AR decoder; Tables 4 and 5 on M5 and the internal panel | Muşat & Căbuz (eMAG), arXiv:2602.22685 (2026) — https://arxiv.org/abs/2602.22685 |
| Two-stage churn classifier + AOV/frequency regressors at Groupon (**ABSTRACT-ONLY**) | Vanderveld, Pandey, Han & Parekh, KDD 2016 — https://doi.org/10.1145/2939672.2939693 |
| Binary next-month purchase prediction, gradient tree boosting, 89% accuracy / 0.95 AUC, no count component (**ABSTRACT-ONLY**) | Martínez, Schmuck, Pereverzyev, Pirker & Haltmeier, *EJOR* 281(3):588–596 (2020) — https://doi.org/10.1016/j.ejor.2018.04.034 |
| Lasso-BG/BB beats BG/BB, plain Lasso, and two recurrent neural networks; BG/BB estimate the most influential of ~100 features (**ABSTRACT-ONLY**) | Chou, Chuang, Chou & Liang, *EJOR* (2021) — https://doi.org/10.1016/j.ejor.2021.04.021 |
| `P(alive)` confounds two quantities; 3,654–27,734 spread at near-identical observable forecasts; 42% swing from one default parameter; 2.4× spread on CDNOW | "Dead Reckoning: Counting Your Customers Who Never Say Goodbye", arXiv:2607.18623 (2026) — https://arxiv.org/abs/2607.18623 |
| Eleven-model comparison on ten CEE online stores, 2.3M customers; Pareto/GGG most stable at 90% average forecast-vs-actual with 9% s.d.; BG/NBD best FA at 100.44%; customer-level MAE ~92–117% of mean profit for *every* model | Jasek, Vrana, Sperkova, Smutny & Kobulsky, *JBEM* 20(3):398–423 (2019) — https://doi.org/10.3846/jbem.2019.9597 (PDF: https://journals.vilniustech.lt/index.php/JBEM/article/download/9597/8454) |
| Simple managerial heuristics match or beat Pareto/NBD and BG/NBD on all managerially relevant tasks (**SECONDARY** for the per-dataset figures) | Wübben & von Wangenheim, *Journal of Marketing* 72(3):82–93 (2008) — https://doi.org/10.1509/jmkg.72.3.082 |
| Hurdle over a free `K`-way softmax = cross-entropy term for term; zero-inflation is a fix for a constraint this head does not have; NB alone recovers 608/683 zeros | `docs/loss-functions.md` §5.6, citing Zeileis, Kleiber & Jackman, JSS 27(8) — https://www.jstatsoft.org/index.php/jss/article/view/v027i08/v27i08.pdf, and Warton (2005) — https://doi.org/10.1002/env.702 |
| What a parametric count head would actually buy: 5.45% of electronics holdout mass above the cap, nothing on CDNOW | `docs/loss-functions.md` §2.3(c) and §5.7 |
| Available reference implementations: NBD, MBG/NBD, BG/CNBD-k, MBG/CNBD-k, Pareto/NBD (HB), Pareto/NBD (Abe), Pareto/GGG | BTYDplus (Platzer) — https://github.com/mplatzer/BTYDplus |
| Available reference implementations: Pareto/NBD, Extended Pareto/NBD (time-varying covariates), BG/NBD, GGom/NBD, Gamma/Gamma | CLVTools 0.12.1 — https://cran.r-project.org/web/packages/CLVTools/index.html |
| MBG/NBD conditional-expectation erratum | *IJRM* 25(2) (2008) — https://doi.org/10.1016/j.ijresmar.2008.02.001 |
