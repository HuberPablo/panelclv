# Hurdle models

What a hurdle model is, why this package's head already is one, and which published
hurdle-type architectures were considered and why. **No hurdle model has been trained
here.** The question was settled by the algebra in §2 and by reading the literature.

## 1. What a hurdle model is, and why it was considered

A hurdle model splits a count forecast into two questions asked in order (Mullahy 1986,
in the form of Zeileis, Kleiber & Jackman,
[JSS 27(8)](https://www.jstatsoft.org/index.php/jss/article/view/v027i08/v27i08.pdf)):

1. **Gate.** Did the customer transact this period? (Bernoulli, `P(y > 0)`.)
2. **Count.** Given that they did, how many times? (A zero-truncated count distribution.)

```
P(y = 0)     = 1 − p⁺
P(y = k > 0) = p⁺ · f_count(k) / (1 − f_count(0))
```

**Why it is tempting here.** Our panels are 96.5% (CDNOW) to 98.1% (electronics) zeros. A
Poisson has one parameter, so it cannot fit both the zero mass and the mean of the
buyers. The hurdle gives the zeros their own parameter.

## 2. This package's head is already a hurdle model

The model emits a free softmax over `K` count classes (`CLAUDE.md`). Both hurdle parts
are read straight off it:

| Hurdle part | In the categorical head |
| --- | --- |
| gate `P(y > 0)` | `1 − q₀` |
| positive-count distribution | `q_k / (1 − q₀)` for `k = 1 … K−1`, nonparametric |
| joint estimation | one cross-entropy |
| sampling | `Categorical(softmax(logits)).sample()` in the rollout |

**The losses are identical.** Written over the same `K` classes, Mullahy's hurdle
likelihood is cross-entropy term for term:

```
−[ 1{y=0} log q₀ + 1{y>0} ( log(1 − q₀) + log(q_y / (1 − q₀)) ) ] = −log q_y
```

The `log(1 − q₀)` terms cancel. A gate-plus-`K−1`-logit head can express exactly the same
set of distributions as the plain head (checked numerically: they agree to 4e-16 in
float64). So a "hurdle head" on the existing trunk is a re-parameterisation, not a new
model.

It would only differ if the two stages were **weighted unequally**. That makes the loss
improper, and `models/losses.py` (`build_criterion`) refuses class weights on a proper
loss (`docs/loss-functions.md` §5.2).

**Zero-inflation fails for the same reason.** `f(y) = π·1{y=0} + (1−π)·f_count(y)` exists to
buy zero mass a parametric family cannot otherwise place. A free softmax already can.
"Many zeros does not mean zero inflation" (Warton 2005,
[doi:10.1002/env.702](https://doi.org/10.1002/env.702)).

**The off-the-shelf two-stage version does not fit the contract.** A `LogisticRegression`
gate times a `GradientBoostingRegressor` mean:
- returns a mean, which the rollout cannot sample from;
- is not autoregressive, so it would either read the holdout or freeze the features;
- regresses a real-valued target where the target is a class index.

## 3. Published hurdle-type architectures considered

| Model | Architecture | Source | Why not built |
| --- | --- | --- | --- |
| **GPPM** (Dew & Ansari 2018) | Per-period Bernoulli purchase incidence, `logit⁻¹[α(t, recency, lifetime, purchase no.) + z'γ + βᵢ]`, with additive Gaussian-process priors on `α`, fitted by HMC. A gate with no count stage, i.e. a `K = 2` head. | *Marketing Science* 37(2); [working paper](https://marketing.wharton.upenn.edu/wp-content/uploads/2017/08/11-07-2017-Dew-Ryan-PAPER-Ansari_BNP_CBA-JMP.pdf) | The shape fits, but the estimator does not: HMC is `O(n³)` in calibration length (about eight days per fit in Valendin et al.'s replication), so a study suite is impossible. |
| **Two-stage gradient boosting** (Lin et al. 2026) | XGBoost classifier for `P(y > 0 \| X)` times a GBM regressor for `E[y \| y > 0, X]`, fitted on buyers only. | *Applied Sciences* 16(13), [doi:10.3390/app16136550](https://doi.org/10.3390/app16136550) | Models monetary spend and returns a mean, not a sampled distribution (§2). |
| **ZILN loss** (Wang, Liu & Miao 2019) | One network head; loss is the NLL of a point mass at zero mixed with a lognormal. | [arXiv:1912.07753](https://arxiv.org/abs/1912.07753), [google/lifetime_value](https://github.com/google/lifetime_value) | A lognormal positive part is continuous, meaningless for integer counts. |
| **Groupon two-stage CLV** (Vanderveld et al. 2016) | Churn classifier, then regressions for order value and frequency on non-churners. | KDD, [doi:10.1145/2939672.2939693](https://doi.org/10.1145/2939672.2939693) | Money-valued and not autoregressive. |
| **BTYD "spike at zero"** (Fader, Hardie & Lee) | One extra parameter `π`: a segment of never-buyers added to BG/NBD. | [BG/NBD paper](https://www.brucehardie.com/papers/bgnbd_2004-04-20.pdf) (the extension sentence was found only in a secondary source) | The authors themselves judged it not worth a parameter. Our head has no zero constraint to relax. |
| **Switch-Hurdle decoder** (Muşat & Căbuz 2026) | From decoder state `h_t`: `p⁺ = σ(w_pᵀh)`, NB mean `μ = softplus(·)`, dispersion `α = softplus(·)`; `P(0) = 1 − p⁺`, `P(y>0) = p⁺·P_NB(y)/(1 − (1+αμ)^(−1/α))`. Loss is gate BCE plus zero-truncated-NB NLL. Autoregressive decoder, matching our rollout. | [arXiv:2602.22685](https://arxiv.org/abs/2602.22685) | The only variant that would change a forecast, and it does so through **unbounded support**, not better zeros. Our head caps counts at `K−1`: nothing on CDNOW, 5.45% of electronics holdout mass (`docs/loss-functions.md` §2.3(c)). It replaces the softmax with a parametric likelihood, which breaks the categorical-head contract, so it needs an ADR first. |

**Related, but not hurdles.** These models were found in the same review and beat the
Pareto/NBD in their own papers. None has a zero component; each changes the timing or
the lifetime process instead.
- **Pareto/GGG** (Platzer & Reutterer 2016, [PDF](http://www.reutterer.com/papers/platzer&reutterer_pareto-ggg_2016.pdf)): gamma inter-transaction times with customer shape `k`, so that `k = 1` is the Pareto/NBD. Implemented as BTYDplus `pggg.mcmc.DrawParameters`. Benchmarked by Valendin et al., so it is the natural second frozen benchmark.
- **MBG/CNBD-k** (Reutterer, Platzer & Schröder 2021, [PDF](http://www.reutterer.com/papers/reutterer&platzer&schroeder_2021.pdf)): Erlang-k timing with dropout that can occur before any repeat purchase; maximum likelihood. Implemented as BTYDplus `mbgcnbd`.
- **PDO** (Jerath, Fader & Hardie 2011, [PDF](https://business.columbia.edu/sites/default/files-efs/pubfiles/6057/customer_death.pdf)): dropout only at periodic opportunities every `τ`.
- **Gamma/Gompertz/NBD** (Bemmaor & Glady 2012, [doi:10.1287/mnsc.1110.1461](https://doi.org/10.1287/mnsc.1110.1461)): Gompertz lifetime. Implemented as CLVTools `ggomnbd`.

## 4. Options, if the question is reopened

- **A. A re-parameterised hurdle head as a confirmation arm.** Gate plus `K−1` logits
  combined into the same `(B, T, K)` output, as one registry entry. It is expected to
  match `lstm`, so any gap would be a wiring bug.
- **B. Build nothing.** §2 is a complete argument. This is the current choice.
- **C. The Switch-Hurdle head.** Gate plus zero-truncated NB on the existing trunk. Needs
  an ADR, because it breaks the categorical-head contract.
