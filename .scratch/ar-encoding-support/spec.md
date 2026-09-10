# Spec: encode the AR clocks so the holdout lands near the fitted range

Status: ready-for-agent

Source: `docs/feature_engineering.md` §4, which measures that the unbounded AR set
forecasts +235% aggregate bias on electronics and +335% on CDNOW against ~22% and ~15%
for the same model with no AR features, and that a bounded flag encoding removes it
entirely. This spec asks the next question: **the flags fix the level by discarding
resolution past the deepest bin — is there an encoding that keeps the resolution?**

Every number below is measured by `scripts/measure_ar_support.py` on the two real
panels. Nothing here is modelled.

## 1. The correction that motivates the whole spec

§4 ranks AR features by their **support-escape fraction**: the share of holdout cells
outside the `[min, max]` the channel took in calibration. That is the right diagnostic
for deciding *whether* to carry a counter and the wrong one for deciding *how to encode*
it, because **it is invariant to every order-preserving transform**. `log1p(recency)`
escapes on exactly the cells `recency` does, to the last cell:

| encoding | escape % elec | escape % cdnow |
| --- | ---: | ---: |
| `period_since_last_transaction` | 37.726 | 56.861 |
| `log_period_since_last_transaction` | 37.726 | 56.861 |
| `saturating_recency_8_periods` | 37.726 | 56.861 |

So the existing `active_in_last_<K>_periods` fix does not work by "bounding" in the loose
sense. It works because the map is **non-injective**: every gap at or beyond the deepest
bin collapses onto the all-zero vector, a value calibration is full of. That is one of
exactly two ways to help, and the other is to shorten the distance travelled outside.

**Distance is the quantity that was never measured.** Theorem 1 of Xu et al. (2021,
ICLR, [arXiv:2009.11848](https://arxiv.org/abs/2009.11848)) proves that "ReLU MLPs
quickly converge to linear functions along any direction from the origin". A network
does not go quiet past its fitted range; it continues along a fitted slope, so the error
grows with how far out the input is. `standardize_covariates` fits on calibration, so
the units that matter are calibration z-scores:

    z_beyond = (holdout value - calibration max) / calibration sd

## 2. What that measures

`z worst` is the single most extreme holdout cell; `z avg` is the mean excess over the
cells that actually escape.

| encoding | escape % | z worst | z avg | escape % | z worst | z avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| | **electronics** | | | **cdnow** | | |
| `period_since_last_transaction` | 37.7 | 1.895 | 0.786 | 56.9 | 3.924 | 1.628 |
| `log_period_since_last_transaction` | 37.7 | **0.357** | 0.161 | 56.9 | **0.711** | 0.339 |
| `saturating_recency_8_periods` | 37.7 | **0.093** | 0.045 | 56.9 | **0.332** | 0.174 |
| `saturating_recency_26_periods` | 37.7 | 0.246 | 0.117 | 56.9 | 0.890 | 0.444 |
| `recency_over_tenure` | **0.0** | 0.000 | 0.000 | **0.0** | 0.000 | 0.000 |
| `active_in_last_8_periods` | **0.0** | 0.000 | 0.000 | **0.0** | 0.000 | 0.000 |
| `period_since_first_transaction` | 88.8 | 1.743 | 0.796 | 85.0 | 3.516 | 1.557 |
| `log_period_since_first_transaction` | 88.8 | **0.331** | 0.163 | 85.0 | **0.564** | 0.285 |
| `saturating_tenure_8_periods` | 88.8 | **0.090** | 0.047 | 85.0 | **0.276** | 0.153 |
| `cumulative_transactions` | 0.04 | 0.639 | 0.639 | 0.18 | **9.193** | 3.865 |
| `transaction_rate` | 0.0 | -10.633 | 0.000 | 0.0 | -5.742 | 0.000 |

Three readings, and the third was not anticipated:

- **Compression buys an order of magnitude.** Raw recency sits 1.90 z past its ceiling
  on electronics and 3.92 z on CDNOW. Under `log1p` that becomes 0.36 and 0.71; under a
  saturating form with C = 8, 0.09 and 0.33. Same cells outside, a tenth of the distance.
- **The ratio is the only recency encoding with nothing outside at all.** In
  `(T - t_x)/T` both terms advance through the holdout, so the channel stays in the
  region calibration covered, with no bin depth to choose and no resolution discarded.
- **`cumulative_transactions` is riskier on CDNOW than the escape fraction suggests.**
  It escapes on 0.18% of cells but by 9.2 z at the extreme and 3.9 z on average — far
  further than either clock. §4 currently files it under "stays in range in practice",
  which is true by count and false by distance. The two metrics disagree here and both
  matter; a channel can escape rarely and catastrophically.

## 3. Which coordinate is the right one, and why

The second half of the same paper is the design brief. Hypothesis 1, *linear algorithmic
alignment*: "if we encode appropriate non-linearities into the model architecture and
input representations so that the MLP modules only need to learn nearly linear steps,
then the resulting neural network can extrapolate well." On the feature side they spell
out the route as decomposing the target into a feature embedding and a simpler function,
obtained "via specialized features or feature transforms using domain knowledge".

Pareto/NBD supplies the domain knowledge in closed form. The lifetime is exponential with
a Gamma(s, β) rate, so unconditional survival is the Laplace transform of a gamma —
Lomax, hence the model's name — and

    log S(t) = -s · log(1 + t/β)

is **linear in `log(1 + gap)`** and in nothing simpler. Under Theorem 1 the network will
extrapolate linearly in whatever coordinate it is handed; log1p is the coordinate where
that is the correct thing to do.

Measured by `scripts/measure_ar_support.py --linearity`, fitting the empirical
log-hazard on calibration cells only and then extrapolating past the calibration ceiling:

| panel | R² vs gap | R² vs log(1+gap) | extrapolated hazard: gap / log(1+gap) | holdout truth |
| --- | ---: | ---: | ---: | ---: |
| electronics | 0.591 | 0.720 | 0.0020 / 0.0050 | 0.0055 |
| cdnow | 0.871 | 0.909 | 0.0008 / 0.0050 | 0.0046 |

The log coordinate straightens the response and, extrapolated blind past the ceiling,
lands on the truth. The raw coordinate undershoots by 2.8x and 5.8x. The fitted slopes,
-0.56 and -0.92, are estimates of the Pareto/NBD lifetime shape parameter s, which is a
sanity check that the mechanism assumed is the mechanism present.

**This is necessary, not sufficient.** Theorem 2 of the same paper says an MLP recovers
a linear target only when the training support "contains a connected subset S, where for
any non-zero w there exists k > 0 so that kw ∈ S" — the support must cover all
directions. §4 already measured that this drift is strictly one-sided: no holdout cell
falls below the calibration minimum. That is the geometry Theorem 2 excludes, and it is
why the arms below have to be trained and scored rather than argued.

## 4. What was built

Five names, all pure functions of the five running states `_base_states` already
maintains, so `ARFeatureState` needed no new state and the precompute/rollout equality
holds unchanged:

| name | value | role |
| --- | --- | --- |
| `log_period_since_last_transaction` | `log(1 + since)` | the aligned recency coordinate |
| `log_period_since_first_transaction` | `log(1 + tenure)` | the same for observation age T |
| `saturating_recency_<C>_periods` | `since / (since + C)` | bounded in [0,1), half at C |
| `saturating_tenure_<C>_periods` | `tenure / (tenure + C)` | the same for T |
| `recency_over_tenure` | `(T - t_x) / T`, 0 before the first purchase | zero escape |

`recency_over_tenure` collides "never transacted" with "transacted this period" at 0, so
it belongs alongside `has_transacted_before`. Its value is exactly 1 for a customer who
bought once and vanished, which is the Pareto/NBD dead signal.

**A complete bounded re-encoding of the sufficient statistics** `(x, t_x, T)` is then
`recency_over_tenure` + `transaction_rate` + `saturating_tenure_<C>_periods`. The first
two escape on nothing; the third carries the absolute age, which is what weights the
precision of the rate estimate, at 0.09 z on electronics and 0.28 z on CDNOW.

## 5. Rejected, with the measurement

- **Shrunk posterior rate `(x + r)/(T + α)`.** The obvious Bayesian encoding of the
  Pareto/NBD frequency, and it is *worse* than the plain ratio on support: with r = 1,
  α = 13 it escapes on 27.4% of electronics and 49.8% of CDNOW holdout cells, falling
  **below** the calibration floor as the denominator grows while a lapsed customer's
  numerator stops. `transaction_rate` escapes on nothing. Not implemented.
- **An exponentially weighted activity trace** `a_t = ρ a_{t-1} + (1-ρ)·1[y_t > 0]`.
  Measured at ρ ∈ {0.7, 0.9, 0.97}: escapes on 0% of electronics cells and ≤0.1% of
  CDNOW cells, so it works. Deferred anyway, because it is the only candidate needing a
  **float** entry in a running state whose integer-valued exactness is what guarantees
  the precompute and the rollout agree to the bit. Revisit if the arms below show the
  flags' lost resolution actually costs something. Open question, issue 04.

## 6. What still has to be tested

Support is an input-side precondition, not a result. §4's own warning applies with full
force: CDNOW's `ar_bounded_32` arm was nominally bounded and still lost, because only
3.5% of calibration cells sat beyond its deepest bin while 68.9% of holdout cells did.
None of the encodings here can be adopted on the strength of the tables above.

Issue 03 defines the arms. The bar is set by the existing ablation: `|bias|` back at or
below the no-AR baseline (22.4% electronics, 14.6% CDNOW) **and** per-customer Spearman
at or above the bounded flag arms (0.267 electronics, 0.438 CDNOW), because an arm that
fixes the level by going blind is the failure mode this family is meant to avoid.
