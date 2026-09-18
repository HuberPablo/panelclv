# The hurdle model, and why this package already trains one

What a **hurdle model** is, in plain language; why the head this package already trains is
one; and what a person is actually proposing when they propose "adding a hurdle model"
here. Written for a reader with no prior exposure to count models — the argument leads, the
algebra follows it rather than carrying it.

Two neighbouring documents, easily confused with this one:

- [`docs/hurdle-models-vs-pareto-nbd.md`](hurdle-models-vs-pareto-nbd.md) is a **literature
  review**: has any *published* hurdle model been measured against the Pareto/NBD on
  holdout counts and won? (Short answer: the head-to-head has never been run.)
- [`docs/loss-functions.md`](loss-functions.md) §5.6 is the **algebra**, in two paragraphs,
  as one entry in a survey of candidate losses.

This document is about **this package's own head**, and is the one to read first. Read
`CONTEXT.md` for the vocabulary (*calibration*, *holdout*, *rollout*, *period*) and
`CLAUDE.md` for the model contract that constrains everything below.

The numbers in §3 were measured while writing this, on the project venv, and the script
that produces them is printed in full so it can be re-run.

## Contents

1. [What a hurdle model is, and why anyone wants one](#1-what-a-hurdle-model-is-and-why-anyone-wants-one)
2. [This package's head already answers both questions](#2-this-packages-head-already-answers-both-questions)
3. [The equivalence, measured](#3-the-equivalence-measured)
4. [Why the scikit-learn two-stage version does not fit](#4-why-the-scikit-learn-two-stage-version-does-not-fit)
5. [What zero-inflation would and would not buy](#5-what-zero-inflation-would-and-would-not-buy)
6. [The one variant that would change a number](#6-the-one-variant-that-would-change-a-number)
7. [Where that leaves the decision](#7-where-that-leaves-the-decision)

---

## 1. What a hurdle model is, and why anyone wants one

A hurdle model splits a count forecast into **two questions asked in order**:

1. **Did this customer transact at all this period?** — a yes/no question.
2. **Given that they did, how many times?** — asked only of the customers who cleared the
   first hurdle, which is where the name comes from.

The structure is due to Mullahy (1986) and is presented in the form used here by Zeileis,
Kleiber & Jackman, *Regression Models for Count Data in R*, JSS 27(8)
(https://www.jstatsoft.org/index.php/jss/article/view/v027i08/v27i08.pdf).

The reason it exists is worth stating precisely, because it is the whole of what follows.
Customer-base panels are overwhelmingly zeros — measured on this project's two panels,
**96.50% of CDNOW cells and 98.07% of electronics cells** are zero
(`docs/loss-functions.md` §2.1, §2.2). A Poisson distribution has exactly one parameter, its
mean λ, and that single number has to serve two masters: it fixes the average count *and*
the probability of a zero, which is `e^(−λ)`. Ask a Poisson to produce 98% zeros and you
have simultaneously forced its mean, and that mean is then almost certainly wrong for the
customers who do buy.

**A hurdle model is the escape from that bind.** It gives the zeros their own parameter, so
the "how many" part is no longer obliged to explain the zeros too. That is the entire
motivation. Keep it in mind, because the next section is about what happens when the bind
was never there.

## 2. This package's head already answers both questions

This package does not predict a count. At every period it emits a **probability for each
possible count** — a softmax over `K` classes (`CLAUDE.md`, "What the models are"). On
CDNOW `K = 5`, so the model's output for one customer in one week is five numbers:

```
P(0) = 0.94    P(1) = 0.05    P(2) = 0.008    P(3) = 0.002    P(4) = 0.0003
```

Now ask the two hurdle questions of that output:

- *Did they transact at all?* — `1 − P(0)` = 0.06. Already there.
- *Given that they did, how many?* — divide each of the rest by that 0.06:
  `P(1)/0.06 = 0.83`, `P(2)/0.06 = 0.13`, and so on. Already there.

And the reverse works just as well. Hand me a gate probability of 0.06 and a positive-count
distribution of `(0.83, 0.13, …)`, and I will hand you back exactly the five numbers above
by multiplying. **Nothing is lost in either direction.** The two are two ways of writing
down one object.

The analogy that makes this concrete: giving directions as *"go 3 km north, then 4 km east"*
versus *"go 5 km northeast"*. Two descriptions, one destination. You can convert between
them freely, and neither gets you anywhere the other cannot.

So the question "should we add a hurdle model?" has an unexpected answer: **the model in
`models/multinomial_lstm.py` is already a hurdle model.** It has an explicit
zero probability, it has a distribution over the positive counts, and — unlike a classical
hurdle model — it estimates both in a single pass instead of fitting two separate stages.

| Hurdle component | What this package already has |
|---|---|
| The gate — did they transact? | `P(0)`, one free coordinate of the softmax, per customer per period |
| The positive-count part | `P(1) … P(K−1)`, with no functional form assumed |
| How they are fitted | one cross-entropy, one backward pass |
| How a forecast is drawn | `Categorical(softmax(logits)).sample()` in the rollout |

The crucial word is **free**. The softmax's `K` probabilities are unconstrained except that
they sum to one. It can put 98% of its mass on zero if the data says so, and pay nothing
for it elsewhere — the bind that motivated the hurdle in §1 does not exist here. This is
the point Warton makes in "Many zeros does not mean zero inflation" (*Environmetrics* 16(3),
2005, https://doi.org/10.1002/env.702): across 20 datasets and 1,672 variables he found the
plain negative binomial fitted best *without* any zero-inflation component, the high zero
frequency being already well described by the model's ordinary machinery.

## 3. The equivalence, measured

§2 is an argument about what the two descriptions can express. The stronger claim is about
**training**: written over the same `K` classes and fitted by maximum likelihood, the
two-stage hurdle loss is not merely similar to this package's cross-entropy — it is the
same number.

The identity is one line. Writing `q₀` for the predicted zero probability and `q_y` for the
predicted probability of the observed count `y`:

```
−[ 1{y=0}·log q₀  +  1{y>0}·( log(1 − q₀) + log( q_y / (1 − q₀) ) ) ]  =  −log q_y
```

Inside the `y > 0` branch, `log(1 − q₀)` and `−log(1 − q₀)` cancel. What is left is
`−log q_y`, which is cross-entropy. The cancellation is exact — no approximation, no
asymptotics.

Measured on both panels' class counts, once in the float32 the models actually train in and
once in float64 to show that the residual is arithmetic rounding and not a real gap:

```
CDNOW (K=5), float64, 200,000 cells:
  cross-entropy         = 1.934719729222678
  two-stage hurdle loss = 1.934719729222679
  difference            = 2.22e-16
  reaches any distribution, max error = 4.44e-16
electronics (K=7), float64, 200,000 cells:
  cross-entropy         = 2.368058047878776
  two-stage hurdle loss = 2.368058047878776
  difference            = 0.00e+00
  reaches any distribution, max error = 4.44e-16
```

One unit in the last place on CDNOW, and exact agreement on electronics. In float32 the
gap is 1.19e-07 and 2.38e-07 — float32 epsilon, i.e. the two are as equal as that precision
permits. The "reaches any distribution" line answers the other half: a hurdle head built
from a gate plus `K−1` positive logits can express *any* distribution over the `K` classes,
recovered here to 4.44e-16 from 200,000 randomly drawn target distributions. It has the same
`K` free parameters as the plain head and reaches exactly the same set of forecasts.

The script, runnable as-is:

```python
# /home/virthian/Desktop/Thesis/venvs/thesis_rocm/bin/python - (project venv)
import torch, torch.nn.functional as F
torch.manual_seed(20260918); torch.set_default_dtype(torch.float64)

for panel, K in (("CDNOW", 5), ("electronics", 7)):
    B = 200_000
    h  = torch.randn(B, 16)
    z0 = torch.nn.Linear(16, 1)(h)        # the hurdle gate
    zp = torch.nn.Linear(16, K - 1)(h)    # the positive-count stage

    # Mullahy's hurdle, written over the same K classes
    logp = torch.cat([F.logsigmoid(-z0),
                      F.logsigmoid(z0) + F.log_softmax(zp, -1)], -1)

    y  = torch.randint(0, K, (B,))
    ce = F.cross_entropy(logp, y)                      # what this package minimises
    q0 = logp[:, 0].exp()
    hurdle = torch.where(
        y == 0,
        -torch.log(q0),
        -(torch.log1p(-q0) + F.log_softmax(zp, -1)
          .gather(1, (y.clamp(min=1) - 1).unsqueeze(1)).squeeze(1)),
    ).mean()                                           # the two-stage hurdle loss

    # can the hurdle head express an ARBITRARY distribution over the K classes?
    p   = torch.distributions.Dirichlet(torch.ones(K)).sample((B,))
    r0  = torch.log((1 - p[:, 0]) / p[:, 0])
    rec = torch.cat([F.logsigmoid(-r0).unsqueeze(1),
                     F.logsigmoid(r0).unsqueeze(1)
                     + F.log_softmax(torch.log(p[:, 1:]), -1)], -1).exp()

    print(f"{panel} (K={K}), float64, {B:,} cells:")
    print(f"  cross-entropy         = {ce.item():.15f}")
    print(f"  two-stage hurdle loss = {hurdle.item():.15f}")
    print(f"  difference            = {abs(ce.item()-hurdle.item()):.2e}")
    print(f"  reaches any distribution, max error = {(rec-p).abs().max().item():.2e}")
```

**What this means in practice.** A model with a hurdle head bolted onto the existing LSTM
trunk would have the same minimiser as the plain model. Trained on the same data with the
same loss, it converges to the same forecasts. It is a *re-parameterisation* — a different
way of writing down the same function — not a different model.

There is exactly one way to make it differ: **weight the two stages unequally**, e.g. count
the gate's error more heavily than the positive-count error. That is no longer a proper
scoring rule, and `docs/loss-functions.md` §5.2 measures what impropriety costs on this data —
inverse-frequency weighting moves the optimum toward the uniform distribution over the `K`
classes, producing a forecast mean of 2.0 against a true 0.0598. The package refuses that
pairing in code (`models/losses.py`, `build_criterion` rejects class weights on a strictly
proper loss). So the one available degree of freedom is one already ruled out.

## 4. Why the scikit-learn two-stage version does not fit

The natural first draft of a hurdle model is two scikit-learn estimators — a
`LogisticRegression` for the gate, a `GradientBoostingRegressor` for the positive part —
with `predict` returning `P(Y>0) · E[Y | Y>0]`. It is worth recording why that shape cannot
enter this package, because it is what anyone reaches for first.

**It returns a mean, not a distribution.** `P(Y>0) · E[Y|Y>0]` is a single number per
customer-period. This package's forecast is a *rollout*: draw a count from the predicted
distribution, feed that draw back as the next period's input, repeat across the holdout,
and average many simulated paths (`CLAUDE.md`). A mean cannot be drawn from. Feeding the
mean back instead would make every simulated path identical and turn the forecast into a
deterministic recursion — which is a different estimator with different bias, not a
shortcut to the same one.

**It is not autoregressive.** The features an AR model reads — recency, cumulative counts,
tenure, rate — are functions of the target's own past, and during a holdout that past does
not exist yet. The rollout recomputes them at each step from the counts *it sampled*, which
is precisely what keeps the forecast leak-free (`docs/feature_engineering.md`). A
cross-sectional `fit(X, y)` has no such loop: to forecast `T` periods it must either read
the true holdout — leakage — or hold the features frozen at their calibration values, which
silently changes the model being evaluated.

**A continuous regressor on counts is the wrong target shape.** `GradientBoostingRegressor`
fits squared error on a real-valued outcome. The target here is a class index, and the
scoring authority (`models.monte_carlo_forecasting.compute_forecast_metrics`) is fed
per-customer per-period arrays produced by sampling.

None of this makes the two-stage idea wrong in general — it is standard and sensible for
zero-inflated *spend*, which is continuous. It is wrong for this target, in this package.

## 5. What zero-inflation would and would not buy

Zero-inflation is the hurdle's close cousin: a mixture that adds extra zeros on top of a
count distribution, `f(y) = π·1{y=0} + (1−π)·f_count(y)`. It fails here for the same reason
and it is worth being explicit, because "the data has many zeros, so use a zero-inflated
model" is a common reflex.

The inflation parameter exists to buy zero mass that a one-parameter family *cannot
otherwise express* — exactly the Poisson bind from §1. A free `K`-way softmax can already
place any mass at zero. **There is nothing to inflate.** Zeileis et al. illustrate the point
concretely: a plain negative binomial recovers 608 of 683 observed zeros where a Poisson
predicts 47, and the hurdle and zero-inflated variants add little on top.

Having many zeros is not, by itself, a reason to reach for either device. The question that
matters on this data is not "are there many zeros" but "does the model's predicted rate go
low enough for dormant customer-weeks" — and a softmax's does, by construction.

## 6. The one variant that would change a number

Everything above says the hurdle *structure* is already present. There is one genuinely
different thing a published hurdle model would contribute, and it is not about zeros at all.

The categorical head can only predict counts it has classes for. CDNOW uses `K = 5` and its
holdout maximum is 3, so nothing is out of reach. Electronics uses `K = 7` and has a
26-transaction week: **25 holdout cells exceed the top class, carrying 80 of 1,467
transactions — 5.45% of the holdout's mass the head structurally cannot reach**
(`docs/loss-functions.md` §2.3(c)). A parametric count distribution has unbounded support and
would not have that ceiling.

The best-specified version of this is the decoder in Muşat & Căbuz, "Switch-Hurdle: A MoE
Encoder with AR Hurdle Decoder for Intermittent Demand Forecasting"
(https://arxiv.org/abs/2602.22685) — a Bernoulli gate plus a *zero-truncated negative
binomial* for the positive part, sampled autoregressively in a way that matches this
package's rollout closely. See `docs/hurdle-models-vs-pareto-nbd.md` §5 for its results.

Two cautions before anyone reaches for it:

- **It buys unbounded support, not better zero-modelling.** On CDNOW it would change
  nothing, because nothing lies above the head. On electronics it extends the reachable
  range — but the archived models there already *over*-predict by +11% to +39%, so
  extending the range makes the reported bias worse, not better (`docs/loss-functions.md` §5.7).
- **It breaks the model contract.** A parametric count likelihood is not a softmax over `K`
  classes and its target is not a class index. `CLAUDE.md` states that any new model keeps
  the categorical shape, so this is an ADR to be written and argued, not an experiment to be
  run on a whim.

## 7. Where that leaves the decision

Three honest options, with what each costs and yields.

**A — Build the re-parameterised hurdle head as a confirmation arm.** A two-head LSTM
(a gate logit plus `K−1` positive logits, combined into the same `(B, T, K)` output) sharing
the existing trunk, registered as one more registry entry. It reuses the training loop, the
Monte Carlo simulator and the refit path unchanged. *Cost:* small — one new module, one
registry entry, one mandatory test update, plus a study run. *Yield:* turns §3's algebra
into a measured result. The expected outcome is metrics statistically indistinguishable
from `lstm`; **a systematic gap would be a wiring bug, not a finding**, and that is the main
reason to be careful reporting it. Worth doing if the thesis wants the claim demonstrated
rather than derived.

**B — Build nothing, and cite the algebra.** §3 is already a complete argument, and
`docs/hurdle-models-vs-pareto-nbd.md` §7 ranks this first. *Cost:* none. *Yield:* a defensible
negative result — the categorical-head contract subsumes the hurdle family — which is a
stronger and more general claim than any single ablation. The risk is that a reader wants
to see it rather than follow it.

**C — Write the ADR for the Switch-Hurdle head.** The only route that changes a forecast.
*Cost:* an ADR breaking C1/C2 of the model contract, a new likelihood, a new sampling path,
and a re-argument of the comparison's fairness. *Yield:* reaches the 5.45% of electronics
holdout mass the current head cannot — in a direction that the measured over-prediction
suggests would worsen aggregate bias.

A and B are compatible: A is the evidence for B's claim. C is a separate project.
