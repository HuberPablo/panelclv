# Loss functions

What the training loss has to satisfy in this package, what the five selectable
`loss_type` values actually compute, and which changes to the loss could plausibly
improve a *forecast* — as opposed to improving a *classification* — on the two panels
this project runs on. Two of the changes it proposed have since been run and are
falsified on CDNOW; the Measured blocks in §6 record that, and the recommendations are
kept as written so the prediction and the outcome sit side by side.

Read `CLAUDE.md` for the model contract and `CONTEXT.md` for the vocabulary first.
The short version, because everything below hangs off it: these models are
**classifiers driving a simulator**, so the loss is not scored on the thing it trains.

## Contents

1. [The contract the loss has to satisfy](#1-the-contract-the-loss-has-to-satisfy)
2. [The measured data](#2-the-measured-data)
3. [What is already implemented](#3-what-is-already-implemented)
   - [3.1 What the benchmark this package reproduces actually chose, and why](#31-what-the-benchmark-this-package-reproduces-actually-chose-and-why)
4. [The train/eval mismatch, measured](#4-the-traineval-mismatch-measured)
5. [Candidate losses, one by one](#5-candidate-losses-one-by-one)
6. [Recommendations, ranked](#6-recommendations-ranked)
7. [What was not verified](#7-what-was-not-verified)
8. [Sources](#8-sources)

**Measured vs. read.** Everything in §2 and §4, and the minimiser tables in §5, was
computed here with the project venv against the real panels and the real archived study
results; the commands are given inline. Claims attributed to a paper were taken from the
primary source and carry a URL; the handful that could not be reached behind a paywall or
a bot-block are named individually in §7 and none of them is load-bearing. Where a
measurement and a source disagree, the measurement wins and the disagreement is stated —
§2.3(a) is one such case.

---

## 1. The contract the loss has to satisfy

Four constraints, all from `CLAUDE.md` and the ADRs. A loss proposal that breaks any of
them is a different proposal.

| # | Constraint | Where it lives |
|---|---|---|
| C1 | The head is a **softmax over `K` transaction-count classes**; logits are `(B, T, K)` and a count is a category, never a quantity. `Embedder` refuses to build a model whose head size disagrees with the target column's cardinality. | `CLAUDE.md`, `models/embedders.py` |
| C2 | The target is a **class index**, `(B, T)` long. `training.loop._validate_targets` (`training/loop.py:52`) rejects anything else. | `training/loop.py:52-66` |
| C3 | A forecast is a **rollout that samples**: `Categorical(softmax(logits)).sample()` at every step, fed back as the next period's input, `n_simulations` paths averaged. | `models/multinomial_lstm.py:225-226`, `models/multinomial_transformer.py:304-305`, `models/monte_carlo_forecasting.py:401` |
| C4 | Scoring is on **Monte-Carlo-averaged expected counts**: `compute_forecast_metrics` takes two `(N, T_HOLD)` arrays and returns `rmse`, `bias_percent`, `mape_aggregate`. It is the single scoring authority. | `models/monte_carlo_forecasting.py:542-576` |

C3 and C4 together are the whole difficulty. The sampled class index **is** the
predicted count: `sampled_path[:, t] = sample` in
`monte_carlo_forecasting.py:160/178/266`, and `prediction_mean = simulations.mean(axis=0)`
at line 401. So for a customer-week with predicted class distribution `q`, the forecast
converges as `n_simulations → ∞` to

```
    E_q[y] = Σ_k k · q_k
```

The forecast is a **linear functional of the whole probability vector**. Every
distortion of `q_k` passes straight into the number that is scored, undamped, and no
argmax ever intervenes to hide it. That single sentence decides most of §5.

One consequence worth stating separately, because it is the trap: a loss that improves
*classification* on this data — accuracy, weighted F1, both of which `training/loop.py`
logs — can make the forecast strictly worse, and the standard imbalance remedies do
exactly that. See §5.2 and §5.3.

---

## 2. The measured data

Both panels were run through the package's own `prepare_dataset`, so the cohort filter,
`clip_target_upper` and the window dates are exactly the ones training sees. The script
that produced everything in this section:

```python
# PYTHONPATH=src python - (venv interpreter)
import pandas as pd, numpy as np
from panelclv.configs.panel_config import PanelConfig
from panelclv.data_preparation import panel_dataset
from panelclv.data_preparation.target_channel import calibration_counts, holdout_actuals

data = panel_dataset.prepare_dataset(pd.read_csv(PANEL), CONFIG)
tgt  = np.asarray(data["targets"]).squeeze(-1)          # (N, T_CAL-1) — what the loss sees
s    = int(data["val_start_idx"])
np.bincount(tgt[:, :s-1].astype(int).ravel())           # training-prefix class counts
np.bincount(holdout_actuals(data).astype(int).ravel())  # holdout actuals (never clipped)
```

`CONFIG` is the CDNOW config from `scripts/run_cdnow_embedding_ablation.py` and the
electronics config from `scripts/run_studies.py`, copied verbatim.

### 2.1 CDNOW — `cdnow_customer_week_panel.csv`

N = 2357 customers, T_CAL = 39, T_HOLD = 38, `clip_target_upper=4` → **K = 5**,
`val_start_idx = 31` (training periods 0–30, validation periods 31–38).

| class | training prefix (n=70,710) | validation window (n=18,856) | holdout actuals (n=89,566) |
|---|---|---|---|
| 0 | 66,671 — 94.288% | 18,464 — 97.921% | 87,807 — 98.036% |
| 1 | 3,869 — 5.472% | 361 — 1.915% | 1,668 — 1.863% |
| 2 | 157 — 0.222% | 29 — 0.154% | 83 — 0.093% |
| 3 | 9 — 0.013% | 2 — 0.011% | 8 — 0.009% |
| 4 | 4 — 0.006% | 0 | 0 |
| **mean** | **0.05977** | 0.02254 | **0.02074** |
| var / mean | 1.039 | 1.142 | 1.094 |

Whole-panel figures (calibration + holdout, 181,489 cells): 175,142 / 6,055 / 269 / 19 / 1,
plus three cells at 5, 6 and 7 that the cap clips — **96.503% zeros**, matching the figure
recorded in the ablation script's docstring. Holdout maximum is 3, so **no holdout mass
lies above the head's top class**.

### 2.2 Electronics — `electronics_customer_week_panel.csv`

N = 829 customers, T_CAL = 104, T_HOLD = 52, `clip_target_upper=6` → **K = 7**,
`val_start_idx = 78`.

| class | training prefix (n=63,833) | validation window (n=21,554) | holdout actuals (n=43,108) |
|---|---|---|---|
| 0 | 62,069 — 97.237% | 21,231 — 98.501% | 42,504 — 98.599% |
| 1 | 780 — 1.222% | 125 — 0.580% | 250 — 0.580% |
| 2 | 490 — 0.768% | 89 — 0.413% | 152 — 0.353% |
| 3 | 218 — 0.342% | 47 — 0.218% | 83 — 0.193% |
| 4 | 112 — 0.176% | 21 — 0.097% | 54 — 0.125% |
| 5 | 60 — 0.094% | 15 — 0.070% | 22 — 0.051% |
| 6 | 104 — 0.163% | 26 — 0.121% | 18 — 0.042% |
| 7+ | — (clipped) | — (clipped) | 25 cells, up to **26** |
| **mean** | **0.05931** | 0.03521 | **0.03403** |
| var / mean | 3.041 | 3.326 | 4.153 |

Whole-panel: 172,432 cells, **98.066% zeros**, mean 0.04611, maximum 26.

### 2.3 Three facts that drive everything else

**(a) The brief's "~89% zeros on electronics" is wrong.** Measured, electronics is
**98.07%** zeros on the raw panel and **97.24%** in the training prefix — *more*
zero-inflated than CDNOW, not less. What genuinely separates the two panels is not the
zero rate but the **tail**: CDNOW's variance-to-mean ratio is 1.04 (indistinguishable
from Poisson at this mean) while electronics' is 3.04 in training and 4.15 in the
holdout (strongly overdispersed, with a 26-transaction week in it). Any argument that
turns on "CDNOW is more imbalanced" is backwards.

**(b) Both panels' holdout rate is far below their calibration rate.** CDNOW:
0.0598 → 0.0207, a 65% drop. Electronics: 0.0593 → 0.0340, a 43% drop. This is cohort
decay, and it dominates aggregate bias. Carrying the training rate forward unchanged
gives `bias_percent = +188.1` on CDNOW and `+74.3` on electronics. **The level problem
these models face is a distribution shift between calibration and holdout, and no
per-period loss reweighting can see it.**

**(c) On electronics the head cannot reach 5.45% of the holdout's transactions.**
25 holdout cells exceed class 6, carrying 80 transactions of 1,467 total. `clip_target_upper`
caps the *inputs and targets*; `holdout_actuals` is explicitly never clipped
(`data_preparation/target_channel.py:51-58`). So a perfectly calibrated model still has
a structural `-5.45%` ceiling on electronics aggregate bias. On CDNOW the figure is
exactly 0.

---

## 3. What is already implemented

`src/panelclv/models/losses.py`, 277 lines, five selectable strings, one factory. It is
reached from `training/loop.py:260` (`fit_model`) and `:450` (`refit_full_calibration`),
configured from `tuning/optuna_tuning.py:314-336`, and surfaced as the `training` dict
keys `loss_type` / `class_weights` / `focal_gamma` / `emd_weight` (`studies/config.py:68`).

| `loss_type` | Lines | What it computes | Intended use |
|---|---|---|---|
| `cross_entropy` | `259-260` | `nn.CrossEntropyLoss()` — mean over cells of `−log q_y`. | Default; the value all but four archived study records use. |
| `weighted_ce` | `261-267` | `nn.CrossEntropyLoss(weight=class_weights)` — mean over cells of `w_y · (−log q_y)`, normalised by `Σ w_y` over the batch (PyTorch's `reduction='mean'` with weights divides by the summed weight, not by N). Raises if `class_weights` is `None`. | Class imbalance; weights from `compute_class_weights`. |
| `focal` | `268-269`, class at `41-68` | `mean over cells of α_y · (1 − q_y)^γ · (−log q_y)`. `α` is the optional `class_weights` tensor, `γ` is `focal_gamma` (default 2.0); `γ=0` reduces to weighted CE. | Class imbalance, "focus on hard examples". |
| `emd` | `270-271`, class at `76-91` | `mean over cells of Σ_k (F_q(k) − 1{y ≤ k})²`, where `F_q = cumsum(softmax(logits))` and the true CDF is `cumsum(one_hot(y))`. | Ordinal structure — "predict 0 when actual is 10 is worse than predict 0 when actual is 1". |
| `ce_emd` | `272-273`, class at `99-130` | `mean over cells of (−log q_y) + λ · Σ_k (F_q(k) − 1{y ≤ k})²`. λ is `emd_weight`, an ordinary searched float; `λ=0` recovers `cross_entropy` bit for bit, and a negative λ raises. | Both at once — the log score's grip on the rare classes plus EMD's control of the CDF residuals (R2). |

**Three of the five refuse class weights.** `build_criterion` raises if `class_weights`
is supplied alongside `cross_entropy`, `emd` or `ce_emd` — the members of
`_PROPER_LOSS_TYPES` (`losses.py:228`), checked at `losses.py:249-257` before any
dispatch. The three are strictly proper, and weighting them forfeits exactly the property
§5.5 relies on; the guard exists because accepting and silently dropping the weights left
a study comparing `weighted_ce` against `ce_emd` confounded on two axes at once. This is
the enforcement half of R3.

`compute_class_weights` (`losses.py:138-212`) is inverse-frequency, normalised so the
weights sum to `K`, with absent classes clamped to a count of 1. It respects the
temporal split (ADR-0001): `training_only=True` slices `targets[:, :val_start_idx-1]`
so the validation window's class mix never leaks into the loss. Measured on the real
panels it produces:

```
CDNOW       w = [0.000204, 0.003514, 0.086599, 1.510672, 3.399012]   ratio w4/w0 ≈ 16,700 : 1
electronics w = [0.002614, 0.208044, 0.331173, 0.744379, 1.448880, 2.704577, 1.560333]  ≈ 1,000 : 1
```

**Two of the four alternatives have now been used; two never have.** The census as of
2026-09-01:

```console
$ grep -rho '"loss_type":[^,}]*' Studies/ | sort | uniq -c
      2 "loss_type": "ce_emd"
   1871 "loss_type": "cross_entropy"
      2 "loss_type": "emd"
```

The `emd` and `ce_emd` records are the loss ablation of R1 and R2
(`Studies/loss_ablation_cdnow`, run 2026-08-31, three arms × 10 studies on CDNOW); the
counts are 2 apiece because `loss_type` is recorded once in the suite `config.json` and
once per arm, not once per study. Their measured outcome is in §6 under R1 and R2, and it
is negative on this panel.

`weighted_ce` and `focal` remain implemented, wired — and entirely unevidenced. Nothing
has ever been trained with either, which is a large part of why R3 proposes closing them
out on the argument in §5.2 rather than on a run.

### 3.1 What the benchmark this package reproduces actually chose, and why

Worth establishing before proposing anything, because the frozen Valendin LSTM (ADR-0004)
is the reason the categorical head exists at all. Their §2.2 states that "during training
the model learns to output a multinomial probability distribution so that it maximizes the
likelihood of reproducing the true class labels", with the loss "given by the negative
log-likelihood" — plain cross-entropy on a class index
(accepted manuscript: https://ars.els-cdn.com/content/image/1-s2.0-S0167811622000180-am.pdf;
DOI https://doi.org/10.1016/j.ijresmar.2022.02.007. There is no arXiv preprint.) Their own
code, vendored in this repo at `Original_paper_model/banking_transactions_demo.ipynb`,
confirms it directly and is the stronger primary source for implementation detail:

```python
softmax_layer = Dense(max_trans, activation='softmax', name='softmax')
from tensorflow.keras.losses import sparse_categorical_crossentropy
model_train.compile(loss=sparse_categorical_crossentropy, optimizer=optimizer)
tfp.distributions.Categorical(probs=probs).sample()
```

Three findings from reading them, all of which bear on this document:

**(1) Their stated reason for the softmax is narrower than usually assumed.** Footnote 9:
"Softmax is the recommended choice if the goal is to approximate a probability
distribution, because of the favourable properties of the error gradient, helping the model
adjust incorrect outputs faster." That is a claim about *gradients*, and it is preserved
unchanged by any loss that keeps the softmax head. There is **no** statement anywhere in the
paper rejecting Poisson/NBD as an output distribution, and no mention of multimodality. Do
not over-attribute a rationale they did not give.

**(2) They explicitly leave the loss open.** Footnote 30: "Note that other loss functions
can be useful, in particular when the outputs of the network are not categorical." A loss
change that keeps the categorical head is squarely inside what the source contemplates.

**(3) Their `K` is a data-driven rule, not a constant — and this package departs from it.**
§2.1: "We set the number of neurons k in the softmax layer to reflect the transaction counts
observed across all individuals in the training data… if in the calibration period
individuals only make between zero and three transactions during any of the discrete time
periods, then a softmax layer with four neurons is sufficient." The rule is *observed
maximum + 1*. This package's `clip_target_upper` of 6 and 4 has **no citation in the paper**;
the clip appears in their code only as a commented-out convenience
("`# to make the job easier for the model, we can clip the value at 6:`"). On electronics the
observed calibration maximum is 6 so `K=7` happens to match the rule, but §2.3(c) shows the
holdout reaches 26 — so the departure is real and should be justified from the panels'
statistics, not attributed to Valendin.

One more of their choices corroborates a measurement in §4.1 rather neatly. Their footnote
19 rejects MAE because it "is only appropriate for forecasts of the median (Hanley et al.,
2001)", the argument being that "taking the median value implies forecasting mostly zero
events for all individuals, a very poor prediction, which would nevertheless 'outperform'
all models in this study… in terms of MAE." §4.1 measures the same pathology one metric
over: on electronics, predicting zero everywhere scores **better RMSE than the LSTM**.

---

## 4. The train/eval mismatch, measured

Training minimises per-period cross-entropy over `K` classes on *teacher-forced*
histories. Scoring is RMSE / bias / MAPE of Monte-Carlo-averaged expected counts over a
38–52 week rollout on *self-generated* histories. Three separate gaps hide in that
sentence, and they are not equally important.

### 4.1 RMSE is almost entirely irreducible — a loss cannot move it

Per-cell MSE decomposes as `E[Var(y|x)] + E[(pred − E[y|x])²] + Var/S`. On a panel
that is 98% zeros the first term swamps the second. To bound the second, oracle
forecasters that *read the holdout* were scored with `compute_forecast_metrics`:

| forecaster | CDNOW rmse | electronics rmse |
|---|---|---|
| zeros everywhere | 0.15210 | 0.37750 |
| flat at the calibration mean | 0.15395 | 0.37651 |
| ORACLE flat at the holdout mean | 0.15068 | 0.37596 |
| ORACLE per-customer holdout mean | 0.14087 | 0.36979 |
| ORACLE per-period holdout mean | 0.15058 | 0.37536 |
| ORACLE customer × period (rank-1) | **0.14026** | **0.36686** |

The entire span between predicting zeros and a rank-1 oracle that cheats is **7.8% of
RMSE on CDNOW and 2.8% on electronics.** For comparison, the archived 10-study
electronics suite
(`Studies/cross_entropy_cfg_2y_Train_1yPred_NoCov_V1_10Studies_100_simulations/results.csv`):

| model | rmse (mean ± sd over 10 studies) | bias_percent | mape_aggregate |
|---|---|---|---|
| LSTM | 0.379 ± 0.001 | **+39.1 ± 27.5** | 71.3 ± 18.1 |
| Transformer | 0.377 ± 0.000 | **+11.1 ± 16.2** | 47.3 ± 5.7 |
| Pareto/NBD | 0.375 | **−53.4** | 59.5 |

Three architectures, one classical and two neural, span 0.375–0.379 — a 1% band, well
inside the oracle headroom, and the LSTM is *worse at RMSE than predicting zero
everywhere*. **RMSE on these panels does not discriminate between models and will not
discriminate between losses.** Any loss ablation must be argued and reported on
`bias_percent` and `mape_aggregate`.

(Related, and cheaper than any loss change: the Monte Carlo average itself adds
`Var(y)/S` to MSE — 3.33% at `n_simulations=30`, 1.00% at 100, 0.33% at 300. At S=30
that noise alone is larger than the entire electronics oracle headroom. Use S ≥ 300.)

### 4.2 The bias is a level problem, and it is only partly fixable

Rescaling each archived electronics forecast by the single scalar `c` that zeroes its
bias, then rescoring:

| model | as reported | c (mean, range) | after rescaling | level share of MAPE |
|---|---|---|---|---|
| LSTM | rmse 0.37869, bias +39.07, mape 71.31 | 0.744 (0.541–0.988) | rmse 0.37705, bias 0, mape **53.58** | 24.9% |
| Transformer | rmse 0.37743, bias +11.11, mape 47.29 | 0.917 (0.709–1.157) | rmse 0.37711, bias 0, mape **44.51** | 5.9% |

So on electronics a *perfect* level correction removes all of `bias_percent`, a quarter
of the LSTM's `mape_aggregate`, 6% of the Transformer's, and nothing at all of RMSE
(−0.43% / −0.09%). The residual 44–54 points of MAPE are **shape** error — the model
tracking the wrong weekly profile — and a per-period loss reweighting does not address
shape.

Two things follow. First, the honest upper bound on what any loss change can buy here
is "bias to zero, plus up to a quarter of MAPE". Second, both neural models are biased
**high** (+39%, +11%) while Pareto/NBD is biased **low** (−53%). That direction matters
enormously for §5: every popular imbalance remedy pushes probability mass off class 0
and onto the higher classes, which raises `E_q[y]`, which makes a positive bias worse.

### 4.3 Exposure bias: trained teacher-forced, rolled out on its own samples

`train_one_epoch` feeds the true previous count (and true AR features); the rollout
feeds a sampled count and AR features recomputed from the sampled history
(`monte_carlo_forecasting.py:165-176`). The Monte Carlo average is an unbiased estimate
of the model's *own* joint distribution, so this is not estimator bias — it is that the
conditionals are only ever fit on histories the data produced, and are queried during
the rollout on histories the model produced. On this data the mechanism has a clear
sign: a spuriously sampled transaction resets `period_since_last_transaction` to 0, and
the model has learned that recent buyers buy again, so a false positive raises the next
step's probability. Errors compound upward over 38–52 steps.

This is the mismatch that best explains the observed **positive** neural bias against
Pareto/NBD's negative one — Pareto/NBD integrates its expectation analytically and never
samples. Bengio et al. name the general phenomenon exactly: "At inference, the unknown
previous token is then replaced by a token generated by the model itself. This discrepancy
between training and inference can yield errors that can accumulate quickly along the
generated sequence" (https://arxiv.org/abs/1506.03099). Wen et al. restate it for
forecasting and attribute the fix to a *Direct Multi-Horizon* strategy that "avoids error
accumulation" (https://arxiv.org/abs/1711.11053), explicitly casting DeepAR — the closest
published analogue of this package's rollout — as the recursive teacher-forced baseline
they beat.

**But the obvious remedy is not free, and the reason is the subject of this document.**
Huszár shows that "despite this impressive empirical performance, the objective function
underlying scheduled sampling is improper and leads to an inconsistent learning algorithm"
(https://arxiv.org/abs/1511.05101). Scheduled sampling buys rollout robustness by
surrendering exactly the property §5.1 identifies as what makes the simulator correct. And
the Direct Multi-Horizon alternative is structurally unavailable here: the target column is
one of the model's own inputs and the simulator *is* the model's semantics (`CLAUDE.md`,
ADR-0002), so a recursive rollout is not a design choice that could be swapped out.

The actionable reading is therefore narrow and worth stating plainly: expect a
per-step-optimal model to be sub-optimal at horizon 38–52, measure that gap, and do not
reach for scheduled sampling as if it were a free correction. **UNVERIFIED**: the
compounding-*upward* mechanism specific to this data is inferred from the code path and the
sign of the archived bias, not demonstrated by an experiment. §6 says how to test it.

Note also that ADR-0003 already considered and **retired** the closest available lever
(selecting trials on rollout quality instead of validation cross-entropy). Its "Why it
goes" section is explicit that the idea is sound and someone will propose it again. Any
proposal in that direction must reckon with that ADR rather than rediscover it.

---

## 5. Candidate losses, one by one

### 5.0 Method

For a fixed context the population risk of a loss `L` is `E_{y∼p}[L(q, y)]`. Minimising
it over the simplex by Adam on logits recovers the loss's **population minimiser** `q*` —
the distribution the model is being asked to converge to. If `q* = p`, the loss is proper
and the rollout samples from the right thing. If `q* ≠ p`, the gap is the forecast
distortion, and `E_{q*}[y]` versus `E_p[y]` is exactly the bias it injects.

`p` below is the measured training-prefix class mix from §2. Script:
`E_{q*}[y]` in the tables was computed by 6,000 Adam steps at lr 0.08 on free logits.

| loss (population minimiser on the measured mix) | CDNOW `E_q*[y]` | vs truth | electronics `E_q*[y]` | vs truth |
|---|---|---|---|---|
| truth `p` | 0.05977 | — | 0.05931 | — |
| `cross_entropy` | 0.05976 | **−0.0%** | 0.05931 | **+0.0%** |
| `weighted_ce` (repo weights) | 2.00000 | **+3246%** | 3.00000 | **+4958%** |
| `focal`, γ=2, no α | 0.27315 | **+357%** | 0.44192 | **+645%** |
| `focal`, γ=2, α = repo weights | 2.00000 | **+3246%** | 3.00000 | **+4958%** |
| `emd` (squared EMD) | 0.05978 | **+0.0%** | 0.05931 | **−0.0%** |
| label smoothing ε=0.01 | 0.07917 | **+32%** | 0.08872 | **+50%** |
| `cross_entropy + λ·emd`, λ ≤ 100 | 0.05977–0.06081 | ≤ +1.7% | 0.05931–0.05933 | ≤ +0.04% |

Two of the four implemented losses recover the truth. Two do not, by three orders of
magnitude.

### 5.1 `cross_entropy` — proper, and that is the whole argument

The log score is **strictly proper**: Gneiting & Raftery define a rule as strictly proper
when `S(Q,Q) ≥ S(P,Q)` with equality iff `P = Q`, and state that "the logarithmic score
is strictly proper relative to the class L₁ of the probability measures dominated by µ"
(JASA 2007, §4.2 — https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf).
Guo et al. put the same fact operationally: "in expectation, NLL is minimized if and only
if π̂(Y|X) recovers the ground truth conditional distribution π(Y|X)"
(https://arxiv.org/abs/1706.04599).

Under C3 the rollout samples from `q` and averages. If `q` is the true conditional, the
average converges to the true conditional mean and the forecast is unbiased *by
construction*. Properness is not a nicety here; it is the property that makes the
simulator correct. The measurement above confirms it numerically on both panels'
actual class mixes.

**Failure mode being fixed, stated precisely.** There is no evidence that the class
probabilities are the problem. §4.1 shows RMSE is irreducible; §4.2 shows the residual
error after a perfect level fix is *shape*; §2.3(b) shows the level error is a
calibration→holdout distribution shift of 43–65%. None of those is a properness defect
in the loss. Whatever the four `loss_type` strings can do, they cannot make a
per-period-proper estimator extrapolate a level shift it was never shown.

**What the field expects.** Fader & Hardie list four things a customer-base model is
asked to do: "(i) estimate the model parameters, (ii) create the expected frequency
distribution of transactions given these parameter estimates, (iii) generate the
aggregate sales forecast, and (iv) predict a particular customer's future purchasing"
(note 005 — http://www.brucehardie.com/notes/005/). This package's metrics cover (iii)
through `bias_percent` / `mape_aggregate` and (iv) through per-customer `rmse`. Item (ii)
— the predicted distribution of counts, the display Pareto/NBD papers always show — is
exactly what a strictly proper loss is the estimator for, and the pipeline does not
currently produce it. That is an argument for keeping the loss proper regardless of which
of the two proper options wins §5.5, and a cheap diagnostic worth adding to
`evaluation/` independently of any loss change.

The field is candid that (iv) is its weak point. Fader, Hardie & Lee report a correlation
of 0.626 between actual holdout transactions and BG/NBD conditional expectations and add:
"Is this high or low? To the best of our knowledge, no other researchers have reported such
measures of individual-level predictive performance"
(http://www.brucehardie.com/papers/018/fader_et_al_mksc_05.pdf) — while in aggregate the
same predictions are "indistinguishable from each other and from the actual transaction
numbers". Fader & Hardie's 2009 review is firmer: "the ability to make such individual-level
predictions lies at the heart of any serious attempt to compute the (residual) customer
lifetime value"
(https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_09.pdf).

**And nothing in this field optimises a distance-aware loss.** A targeted search of the
customer-base-analysis literature found no paper using an ordinal, EMD, CRPS or RPS
objective; the neural CLV work is uniformly cross-entropy or parametric likelihood. That
gap appears real, and it makes R1/R2 a defensible novelty claim for the thesis rather than
merely a tuning tweak.

### 5.2 `weighted_ce` — provably wrong here, by a factor of ~33×

Minimising `−Σ_k w_k p_k log q_k` over the simplex gives `q*_k ∝ w_k p_k`. With
inverse-frequency weights `w_k ∝ 1/n_k` and `p_k = n_k/N`, the product `w_k p_k` is
**constant in k**, so the minimiser is the **uniform distribution over the K classes**.
That is not an approximation; the closed form and the numerical minimiser agree to six
decimals in both panels (§5.0 table). Uniform over `{0..4}` has mean 2.0 against a true
0.0598; uniform over `{0..6}` has mean 3.0 against 0.0593.

Primary sources say the same thing in three notations. Elkan's Theorem 2 gives the
binary posterior under a changed base rate, `p′ = b′(p − pb)/(b − pb + b′p − bb′)`, and
is explicit that weighting *is* rebalancing: "If a learning algorithm can use weights on
training examples, then the weight of each negative example can be set to the factor
given by the theorem" (https://cseweb.ucsd.edu/~elkan/rescale.pdf). Menon et al. state the
multiclass version — inverse-frequency weighting targets `P_bal(y|x) ∝ P(y|x)/P(y)`, the
prior-removed posterior, correctable by adding `ln P(y)` back to the logits
(https://arxiv.org/abs/2007.07314, Eq. 8). Ren et al.'s Theorem 1 gives the softmax form
`φ̂_j ∝ n_j e^{η_j}` (https://arxiv.org/abs/2007.10740). And Gneiting & Raftery's
transformation rule (Eq. 2) shows why properness is lost: `S* = cS + h(ω)` preserves
properness only when `c` is a **constant**; `w_y · log q_y` multiplies by a factor that
depends on the outcome, so it falls outside the guarantee.

The one argument for harmlessness does not apply. Byrd & Lipton found that "for most
weight ratios considered (between 256:1 and 1:256) the effect of importance weighting is
indistinguishable from unweighted risk minimization after sufficient training epochs"
(https://arxiv.org/abs/1812.03372) — but that result is for **separable** data trained to
zero training error, and they say so ("on (linearly) separable data, deep linear networks
optimized by SGD learn weight-agnostic solutions"). A 98%-zero weekly count panel is
irreducibly stochastic; the network cannot drive training loss to zero, so the tilt
persists. Their weight ratios also top out at 256:1, and CDNOW's `compute_class_weights`
output spans **16,700:1**.

**Verdict: do not use, and do not search over.** It moves the forecast in the direction
the archived results say is already wrong (§4.2: neural bias is +39% / +11%). If it must
be used for some other reason, divide the weights back out of the softmax and renormalise
before the rollout samples.

### 5.3 `focal` — the intuitive fix for imbalance, and the wrong one here

Focal loss is `FL(p_t) = −α_t (1 − p_t)^γ log p_t` (Lin et al., Eq. 4–5,
https://arxiv.org/abs/1708.02002). Its purpose in the source paper is detection average
precision — the paper makes **no** claim about probability quality (a case-insensitive
search of the full text for "calibrat" returns zero hits).

The two papers that do analyse its probabilities disagree about ECE but agree completely
about the mechanism, and the mechanism is the problem. Mukhoti et al. show focal loss is
an upper bound on an **entropy-regularised** KL, `L_f ≥ KL(q‖p̂) − γH[p̂]`, and that
"focal loss favours a more entropic solution p̂ that is closer to 0.5… solutions to focal
loss will always have higher entropy than those of cross-entropy"
(https://arxiv.org/abs/2002.09437, §4 and Appendix B). They present that as the *feature* —
it counteracts the over-confidence Guo et al. documented. Here it is the bug: "more
entropic" on a 97%-zero head means **moving probability mass off class 0 and onto classes
1…K−1**, and every unit of mass moved to class `k` adds `k` to `E_q[y]`.

Charoenphakdee et al. settle the theory, and correct a plausible prior in both
directions: focal loss **is** classification-calibrated (Theorem 3) but is **not strictly
proper** — "For any γ > 0, the focal loss ℓ^γ_FL is not strictly proper" (Theorem 5), and
"it is strictly proper if and only if γ = 0, i.e., when it coincides with the
cross-entropy loss" (https://arxiv.org/abs/2011.09172). Their abstract states the
consequence directly: "the confidence score of the classifier obtained by focal loss
minimization does not match the true class-posterior probability and thus it is not
reliable as a class-posterior probability estimator." Their Corollary 9 gives the sign:
when `max_y q* > 0.5` — every normal week on these panels — the focal minimiser is
η-*under*confident, shaving the dominant class. Proposition 12 states that the
correcting map never changes the argmax, which is precisely why the damage is invisible
to accuracy and F1 and visible only to `E_q[y]`.

So focal loss is engineered for the one operation this package never performs (an argmax)
and broken for the one it always performs (sampling from `q`). Measured, γ=2 with no `α`
inflates the forecast mean by **+357%** (CDNOW) and **+645%** (electronics); with the
repo's `α` weights it collapses to the same uniform minimiser as `weighted_ce`.

Note also that Mukhoti's footnote 2 — "when q is a one-hot encoding… minimising focal
loss does lead to p̂ being equal to q" — is about the *empirical per-sample* target, not
the population risk minimiser, and does not rescue this case. On genuinely stochastic
labels the population minimiser is what the network converges toward, and
Charoenphakdee's Theorem 5 governs it.

**Verdict: do not use, and do not search over `focal_gamma`.** If it must be used,
Charoenphakdee's Theorem 11 gives the exact inverse map `Ψ^γ` to apply to the softmax
before sampling; they warn that temperature scaling is not an adequate substitute
("using such heuristics may fail to recover the true class-posterior probability", §6.3).

### 5.4 Label smoothing — same failure, dismissed for the same reason

Not currently implemented, and it should stay that way. Smoothing the target to
`(1−ε)·onehot + ε/K` makes the minimiser the smoothed mix, so `E_q*[y]` gains
`ε · (K−1)/2` regardless of how small `ε` is relative to the true probabilities. Measured
at ε=0.01 — the smallest value anyone would bother with — the forecast mean is already
inflated by **+32%** (CDNOW) and **+50%** (electronics). On a head whose true mean is
0.06, any additive floor on the non-zero classes is enormous.

### 5.5 `emd` — the interesting one: it is the discrete CRPS, and it is proper

The implementation (`losses.py:85-91`) is `Σ_k (F_q(k) − 1{y ≤ k})²`. That is not merely
"ordinal-aware" — it is term for term the **Ranked Probability Score** (Epstein 1969), the
discrete-ordinal case of the CRPS. Gneiting & Raftery construct the RPS from the Brier
score (Eq. 49) and it sits in their catalogue of strictly proper rules
(https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf). The proof is one
line: the risk `Σ_k [F_q(k)² − 2F_q(k)F(k) + F(k)]` is minimised term-by-term at
`F_q(k) = F(k)`, and the true CDF satisfies the monotonicity constraint, so the minimiser
is `q = p` uniquely. Measured, it recovers the training mix on both panels to
`max|q* − p| < 1e−7`.

The identity is exact, not an analogy, and it is verified from primary sources at both
ends. Hou, Yu & Samaras drop the normalisation from the closed-form Mallows distance to
get their final loss `E_E(p,t) = Σ_{i=1..C} (CDF_i(p) − CDF_i(t))²`
(https://arxiv.org/abs/1611.05916), choosing the square because "squaring usually leads to
faster convergence with gradient descent". Epstein's RPS is
`Σ_{i=1..r−1} (Σ_{j≤i} (p_j − o_j))²` — the same sum, with Hou's `i=C` term identically
zero because both CDFs equal 1 there. Gneiting & Raftery supply the bridge in Eq. (49):
`S(F,x) = ∫ S(F(y), 1{x ≤ y}) ν(dy)` "is proper", and — verbatim — "The CRPS (20)
corresponds to the special case in (49) in which S is the quadratic or Brier score and ν
is the Lebesgue measure. If S is the Brier score and ν is a sum of point measures, then
the ranked probability score (Epstein 1969) emerges." For a distribution on `{0,…,K−1}`
the CDF is constant on each `[k, k+1)`, so the Lebesgue integral collapses term by term
onto the discrete sum — which is `losses.py:91`.

**A property that matters specifically for training, not just for scoring.** Bellemare et
al. call the same object the Cramér distance and prove (Theorem 2) that it has *unbiased
sample gradients*: `E[∇_θ l²₂(P̂_m, Q_θ)] = ∇_θ l²₂(P, Q_θ)`, a property the Wasserstein
distance lacks — "if a divergence d does not possess (U) then minimizing it with
stochastic gradient descent may not converge, or it may converge to the wrong minimum"
(https://arxiv.org/abs/1705.10743). Minibatch SGD on this loss is therefore sound, which
is not automatic for a distance between distributions.

**The overclaim to avoid.** "Squared EMD is an ad-hoc geometric loss, unlike the
principled cross-entropy" is false — but so is the converse. **Properness alone does not
favour EMD over CE, because CE is strictly proper too.** The argument has to rest entirely
on *distance sensitivity*: Gneiting & Raftery note that scores like the log score "are not
sensitive to distance, meaning that no credit is given for assigning high probabilities to
values near but not identical to the one materializing", and CE's blindness is structural —
they cite Bernardo (1979) that "every proper local scoring rule is equivalent to the
logarithmic score", so locality *is* distance-blindness.

And distance sensitivity is genuinely contested. Wheatcroft, "Evaluating probabilistic
forecasts of football matches: The case against the Ranked Probability Score"
(https://arxiv.org/abs/1908.08980), argues that "the probability placed on potential
outcomes that didn't happen are irrelevant" and reports that "the ignorance score has been
found to outperform both the RPS and the Brier scores". **Scope limit, and it is
important:** his experiments measure *discrimination power as an evaluation metric* — how
quickly a score identifies the better forecaster from finite samples — not training
dynamics. He does not show RPS is a worse training loss. But it is the honest counter-
citation and belongs in the thesis beside the recommendation.

There is a second, sharper reason to care, and it is specific to this project. For a
non-negative integer variable clipped at `K−1`,

```
    E[y] = Σ_{k=0}^{K-2} (1 − F(k))
```

(verified numerically). So the **forecast mean is a sum of exactly the K−1 quantities
whose squared errors the EMD loss penalises**, and the top-class term, where
`F(K−1) ≡ 1`, contributes nothing to either. Cross-entropy penalises `−log q_y`, which is
dominated by whichever class happened to occur; EMD penalises the CDF residuals that
*are* the forecast's error budget. Concretely, per-cell aggregate bias is
`Σ_k (1−F_q(k)) − Σ_k (1−F_p(k)) = −Σ_k (F_q(k) − F_p(k))`, so by Cauchy–Schwarz the
mean bias is bounded by `√(K−1) · ‖F_q − F_p‖₂`, i.e. by the square root of the per-cell
EMD. **Cross-entropy admits no such bound.** For a project whose headline metric is
`bias_percent`, an objective that directly controls the aggregate-bias residuals is the
theoretically best-matched of the four already in the box.

**Empirical precedent, in exactly this setup.** Hou et al.'s AADB experiment discretises a
continuous target into 10 bins and — verbatim — "During testing, we compute the expected
aesthetic scores according to the predicted distributions": train on bins, score the
expectation, which is what this package does. Squared EMD beats cross-entropy on Spearman
ρ in all three of their backbones (0.6682 vs 0.6283; 0.5448 vs 0.5003; 0.6768 vs 0.6693).
It is not uniform — on another dataset CE wins exact-match (60.0 vs 59.3) while EMD wins
within-one-category (92.5 vs 91.5), which is the expected signature of a distance-aware
loss and a fair description of the trade. More directly, Marchesoni-Acland et al., "A CRPS
Loss for Deep Probabilistic Regression" (IEEE URUCON 2024,
https://doi.org/10.1109/URUCON63440.2024.10850406), take a binned softmax head, derive the
CRPS loss, and report: "we derive and implement the CRPS loss and showcase its performance
against cross-entropy… Surprisingly, using the CRPS loss provides superior results even
when training a deterministic regressor."

Worth recording as a gap: Hou et al. never mention proper scoring rules, CRPS or RPS
anywhere in their paper. They rediscovered a 1969 forecasting score and justified it
geometrically.

The cost: RPS is a quadratic rule, far less sensitive to small probabilities than the log
score. On a head that is 97% class 0 the gradient is dominated by the `F(0)` residual, so
EMD alone risks under-fitting classes 2+ relative to CE. That argues for the mixture
rather than the swap — and because a non-negative combination of proper rules is proper,
`cross_entropy + λ·emd` is strictly proper for every `λ ≥ 0`. Measured, its minimiser
stays on the truth up to λ=100 (`E_q*[y]` within +1.7% on CDNOW, +0.04% on electronics;
the residual is optimiser tolerance, not tilt).

**Verdict: the only one of the four alternatives worth an ablation, and the mixture is
worth more than the swap.** The swap is a one-string change today; the mixture needs one
new `loss_type` and one new coefficient.

### 5.6 Zero-inflated and hurdle formulations — vacuous over a free softmax

A hurdle model factorises `f_hurdle(y) = f_zero(0)` if `y=0`, else
`(1 − f_zero(0)) · f_count(y)/(1 − f_count(0))` — the formulation Zeileis, Kleiber &
Jackman give and attribute to Mullahy (1986)
(JSS 27(8), https://www.jstatsoft.org/index.php/jss/article/view/v027i08/v27i08.pdf).
Written over the **same** `K`-way softmax and trained by log-likelihood, that
factorisation is cross-entropy, term for term:

```
−[1{y=0} log q₀ + 1{y>0}(log(1−q₀) + log(q_y/(1−q₀)))] = −log q_y
```

Verified numerically to ten decimals on both panels' class mixes. The chain rule is not
an approximation here — a hurdle "loss" over a free categorical head is a
re-parameterisation of CE with no new degrees of freedom. It differs only if the two
terms are weighted unequally, which reintroduces exactly the impropriety of §5.2.

The same argument kills zero-inflation. The ZIP mixture
`f_zi(y) = f_zero(0)·I_0(y) + (1 − f_zero(0))·f_count(y)` (same source) exists because a
Poisson with mean `λ` cannot place enough mass at zero; the inflation parameter buys the
extra zero mass a one-parameter family cannot express. A free `K`-way softmax can already
place any mass at zero, so there is nothing to inflate. **Zero-inflation is a fix for a
constraint this head does not have.**

The count-modelling literature reaches the same conclusion even for parametric heads.
Warton, "Many zeros does not mean zero inflation" (Environmetrics 16(3), 2005,
https://doi.org/10.1002/env.702), fitted 20 datasets and 1,672 variables and found the
negative binomial best-fitting *without* zero-inflation, the high zero frequency being
already well described by the systematic component (abstract reconstructed from OpenAlex,
near-verbatim — see §7). Zeileis et al. illustrate it concretely: an NB alone recovers 608
of 683 observed zeros where a Poisson predicts 47, and hurdle/ZINB add little. With 98%
zeros the question is never "are there many zeros" but "does the model's mean go low
enough for dormant customer-weeks" — and a softmax's does, by construction.

This is a genuinely useful negative result for the thesis: the categorical head that
`CLAUDE.md` mandates already dominates ZIP/hurdle *on the classes it covers*. The only
thing it gives up is support above `K−1`.

### 5.7 Poisson / negative binomial / Tweedie heads — a different proposal, flagged

These require replacing the softmax with a parametric count likelihood, which breaks C1
and C2 and contradicts `CLAUDE.md`'s "any new model keeps this shape: categorical head,
class-index target". Flagged as such, not proposed. For completeness, what the
measurements say about whether it would even be worth the ADR:

- **CDNOW**: `var/mean = 1.04`, maximum count 4, zero holdout mass above the head. A
  negative binomial's whole advantage is overdispersion, and there is none. A Poisson
  head is a one-parameter restriction of a five-way softmax and can only lose. **Nothing
  to gain.**
- **Electronics**: `var/mean = 3.04` in training, 4.15 in the holdout, with a 26-count
  week. Here NB has a real argument — but the specific benefit is *unbounded support*
  (the 5.45% of holdout mass the head cannot reach, §2.3(c)), not the likelihood shape,
  since a 7-way softmax already fits any distribution on `{0..6}` exactly. And the
  archived models over-predict by +11% to +39%, so extending the reachable range makes
  the reported bias **worse**, not better.

**Tweedie is disqualified outright, not merely deprioritised.** scikit-learn's GLM guide
gives the power parameter's meaning (0 Normal, 1 Poisson, (1,2) compound Poisson-Gamma, 2
Gamma, 3 inverse Gaussian) and states "For 0 < power < 1, no distribution exists"
(https://scikit-learn.org/stable/modules/linear_model.html §1.1.12). The zero-inflation-
friendly range is `1 < p < 2`, and Dunn's `tweedie` manual — he wrote the density
algorithm — states that there "the distribution are continuous for Y greater than zero,
with a positive mass at Y = 0" (https://cran.r-project.org/web/packages/tweedie/tweedie.pdf).
A continuous law with a zero atom would assign density to 0.37 transactions. Only `p = 1`
is a genuine count law, and that is just Poisson with no overdispersion. One sentence in
the thesis; no experiment.

DeepAR is the canonical precedent for an NB likelihood plus ancestral sampling in a
recurrent forecaster (Salinas et al., https://arxiv.org/abs/1704.04110): "the negative
binomial distribution is a commonly used choice" for positive count data, parameterised by
mean and shape through softplus so that `Var[z] = μ + μ²α`, with the rollout drawing
`z̃ ~ l(·|θ)` and feeding it back exactly as this package does. Two details matter if it is
ever pursued. First, their scale handling exists precisely because a count head cannot be
standardised away: "while for real-valued data one could alternatively scale the input in a
preprocessing step, this is not possible for count distributions" — so the `ν_i` machinery
would have to be reimplemented, though their own ablation suggests it buys little off
power-law data ("On the parts data set, which does not exhibit the power-law behavior,
rnn-negbin performs similar to DeepAR"). Second, PyTorch's `NegativeBinomial` uses the
*inverted* convention (`probs` is the success probability,
https://docs.pytorch.org/docs/stable/distributions.html#negativebinomial), so DeepAR's
`(μ, α)` maps as `total_count = 1/α`, `logits = log(αμ)`. `PoissonNLLLoss`
(https://docs.pytorch.org/docs/stable/generated/torch.nn.PoissonNLLLoss.html) defaults to
`full=False`, which drops the `log(target!)` term — its values are therefore **not**
comparable in absolute terms to a cross-entropy, which would silently corrupt any Optuna
study storage shared across the two.

None of this is the right change to make first, on this data, under this contract.

### 5.8 Directly targeting the aggregate

`bias_percent` and `mape_aggregate` are aggregate quantities, and `E_q[y] = Σ_k k q_k`
is differentiable in the logits, so a batch-level moment-matching penalty is expressible:

```
    L = CE + λ · ( mean_cells E_q[y] − mean_cells y )²
```

Its appeal is that it is **proper-preserving**: at the true conditional the penalty's
expectation is zero, so the truth remains a global minimiser and the term acts as a
shrinkage regulariser toward first-moment consistency rather than as a tilt. Its limit is
that it constrains the *teacher-forced one-step* aggregate, whereas the reported bias is
the *rolled-out multi-step* aggregate over a window whose rate is 43–65% below the
training window's (§2.3(b)). It can stop a model from over-predicting the periods it was
trained on; it cannot make it extrapolate a level shift.

Ranked below the EMD mixture for that reason, and because it needs real new code.

### 5.9 What is not a loss change but dominates all of them

Stated once, clearly, because a document that recommends a loss tweak while ignoring
these would be misleading:

- **The archived electronics runs feed the model almost nothing.** The suite config
  records `seq_cols = ["Transactions", "week_sin", "week_cos"]` — no covariates and, more
  importantly, **no AR features**. The current `scripts/run_studies.py` electronics config
  is worse still: `prepare_dataset` reports `seq_cols = ['Transactions']`, F=1. CDNOW's
  config carries `period_since_last_transaction` and `has_transacted_before`; electronics
  carries neither, despite `Gender`, `Income` and `high.season` sitting unused in the CSV.
  Recency is the single feature that lets a model represent the cohort decay that §2.3(b)
  identifies as the dominant source of bias. Adding it is a `PanelConfig` edit.
- **`n_simulations`.** At 30 the Monte Carlo noise alone adds 3.33% to MSE, larger than
  the entire electronics oracle headroom. The archived suite used 100; the CDNOW ablation
  uses 300. Use 300.
- **Selection.** Optuna selects on validation cross-entropy (`results.csv` `objective`
  column). ADR-0003 retired the rollout-based alternative and explained why. Any loss
  change alters the objective's scale, so cross-loss study storage must not be shared.

---

## 6. Recommendations, ranked

Each is stated as (a) the concrete change, (b) the expected effect and its sign, (c) how
to test it. The testing pattern throughout is the one in
`scripts/run_cdnow_embedding_ablation.py`: arms differing in exactly one pinned thing,
one shared `prepare_dataset` object, identical remaining search space and trial budget,
paired seeds (`base_seed + j` for study `j`), read back with
`study_metrics(root, panel_path, standard_deviation=True)`.

**A sizing constraint that binds every arm below.** The across-studies SD of
`bias_percent` in the archived electronics suite is **27.5 (LSTM)** and **16.2
(Transformer)** on 10 studies. With `N_STUDIES = 3` — the CDNOW ablation's setting — the
standard error of an arm's mean bias is ~16 and ~9 points. **A loss effect smaller than
about 20 points of bias is not detectable at N_STUDIES=3.** Budget 10 studies per arm, or
report the paired per-seed difference between arms rather than the difference of means.
Do not report RMSE as the discriminating metric (§4.1).

---

### R1 — Ablate `emd` against `cross_entropy`, and read it on bias, not RMSE

**Change.** `LOSS_TYPE` in the arm's `training` dict: `"cross_entropy"` → `"emd"`. One
string. No code.

**Why.** `emd` is the discrete CRPS: strictly proper (so the rollout still samples from
an unbiased estimate of the truth, §5.5), with unbiased minibatch gradients (Bellemare et
al., Thm. 2), *and* it penalises exactly the `K−1` CDF residuals that sum to the forecast's
aggregate bias, bounding that bias by `√(K−1)·√(per-cell EMD)` — a guarantee cross-entropy
does not offer. It is the only one of the three then-unused `loss_type` values that is not
provably harmful here, and at the time this was written no archived `loss_type` record
had ever read anything but `cross_entropy`. **It has since been tried — see Measured
below, where the argument of this paragraph does not survive contact with the panel.**

It is also the change with the least contractual friction available. The softmax head, the
sampler, the architecture and the metrics are all untouched (C1–C4 hold verbatim); Valendin
et al.'s own footnote 30 says "other loss functions can be useful"; their stated reason for
the softmax (footnote 9, gradient behaviour) is preserved; and there is direct published
precedent for train-on-bins/score-the-expectation, which is this package's exact setup —
Hou et al.'s AADB result and Marchesoni-Acland et al.'s CRPS-beats-CE finding on a binned
head (§5.5). Against that, Wheatcroft (2019) is the standing counter-argument on distance
sensitivity and should be cited whichever way the ablation lands.

**Expected effect.** A reduction in `|bias_percent|` and in `mape_aggregate`; no
meaningful movement in RMSE (§4.1). Risk in the other direction: RPS is a quadratic rule
and under-weights small probabilities, so it may under-fit classes 2+ on the 97%-zero
head and *lower* the forecast — which on electronics, where bias is already positive,
would help, and on a panel with negative bias would hurt.

**Test.** Copy `scripts/run_cdnow_embedding_ablation.py` to a loss ablation: two arms,
`LSTM_ce` and `LSTM_emd`, identical `SHARED_SEARCH_SPACE`, `N_STUDIES = 10`,
`N_SIMULATIONS = 300`. Run on **both** panels — CDNOW (no tail, no clipping loss) and
electronics (heavy tail, 5.45% unreachable mass) exercise the ordinal argument
differently and could disagree. Note in the write-up that the arms' Optuna objectives are
on different scales, so only the forecast metrics are comparable across arms, never the
`objective` column.

**Measured — CDNOW, 2026-08-31.** Run by `scripts/run_loss_ablation.py --panel cdnow`:
three arms, 10 studies each, paired seeds 43–52, 20 Optuna trials per study, 300
simulations, archived at `Studies/loss_ablation_cdnow`. Across studies, mean ± SD:

| arm | `bias_percent` | `mape_aggregate` | `rmse` |
|---|---|---|---|
| `LSTM_ce` | −17.0 ± 39.0 | **41.2 ± 25.6** | 0.149 ± 0.003 |
| `LSTM_emd` | +3.5 ± 134.4 | 123.5 ± 33.5 | 0.154 ± 0.003 |
| `LSTM_ce_emd` | +6.5 ± 65.0 | 58.9 ± 48.0 | 0.154 ± 0.010 |

**The `bias_percent` means are a trap and should not be read as a ranking.** `emd`'s +3.5
looks like the best of the three. It is an artefact of averaging a bimodal distribution
whose two modes are −100% and +150%: six of its ten studies collapse to a forecast of
essentially zero (`bias_percent ≤ −99.9`) and the other four overshoot by +126% to +184%.
Nothing lands in between, and the signed mean cancels the halves against each other.
`mape_aggregate`, which cannot cancel, is the honest column — 123.5 against
cross-entropy's 41.2.

Paired per-seed, which is what the SD above makes necessary, `emd` is worse than
cross-entropy on `mape_aggregate` in **9 of 10 seeds**, by 82.3 points on average. `rmse`
separates nothing (0.149 vs 0.154), exactly as §4.1 predicts it cannot.

**Verdict: R1 is falsified on CDNOW.** The prediction was a reduction in `|bias_percent|`
and in `mape_aggregate`; the measurement is a large increase in both, from a training run
that collapses to the zero forecast more often than not. The risk this section itself
flagged — "RPS is a quadratic rule and under-weights small probabilities, so it may
under-fit classes 2+ on the 97%-zero head and *lower* the forecast" — is precisely the
failure observed, in six of ten studies. Wheatcroft (2019) should be cited accordingly.
**Electronics has not been run**, and per this section's own instruction a result on one
panel is not a result on the other.

---

### R2 — Add `cross_entropy + λ·emd` as a fifth `loss_type` and search λ

**Change.** One new branch in `build_criterion` (implemented: the dispatch arm is
`models/losses.py:272-273`, the module `CrossEntropyPlusEMDLoss` at `99-130`) returning a
module that sums `nn.CrossEntropyLoss()` and `SquaredEMDLoss()` with a coefficient, plus
one new training key (`emd_weight`, mirroring how `focal_gamma` is plumbed through
`tuning/optuna_tuning.py:314-326` and `trials/refit.py:47-49`). λ becomes an ordinary
searched float.

**Why.** It keeps the log score's sensitivity to the rare classes — which is what fits
the tail EMD alone would neglect — while adding EMD's direct control of the
aggregate-bias residuals. A non-negative combination of two strictly proper rules is
strictly proper, so the contract in §1 is untouched; measured, the minimiser stays on the
truth to λ=100 on both panels. λ=0 recovers today's default exactly, so the arm cannot
lose to the baseline except through search noise.

**Expected effect.** Strictly dominates R1 in principle — R1 is the λ→∞ end — at the cost
of one searched dimension and a small code change. Same metrics, same caveats.

**Test.** The same two-arm ablation as R1, with a third arm pinning `emd_weight` to a
searched range. Because λ=0 is inside the range, also report the distribution of the
selected λ across studies: if the search consistently picks λ≈0 that is itself the
result, and it retires the whole line of enquiry cleanly.

**Measured — CDNOW, 2026-08-31.** The third arm of the same ablation. `mape_aggregate`
58.9 ± 48.0 against cross-entropy's 41.2 ± 25.6, and worse on the paired per-seed
comparison in **8 of 10 seeds**, by 17.7 points on average. It lands where the argument
said it would — between plain `emd` and plain CE — but on the wrong side of the baseline.

**The λ distribution is the cleaner result, and it is the one this test named as
decisive.** Searched over `[0, 10]`, the selected λ across the ten studies runs
0.016 / 0.052 / 1.335 (min / median / max), with nine of ten below 0.33. The search
pushes λ toward zero — that is, toward plain cross-entropy — which is exactly the outcome
called out above: "if the search consistently picks λ≈0 that is itself the result, and it
retires the whole line of enquiry cleanly." On CDNOW it does.

One honest qualification. λ=0 is inside the range and recovers the baseline exactly, so
in principle the arm cannot lose except through search noise — yet it did lose, by 17.7
points. With 20 trials over a six-dimensional space the search does not reliably find
λ≈0, so the gap measures the cost of spending a search dimension on λ, not a defect in
the loss. That is still a cost, and it is the argument for retiring the line rather than
re-running it wider.

---

### R3 — Formally close out `weighted_ce` and `focal` rather than leaving them selectable

**Change.** Documentation, not deletion: record in `models/losses.py`'s module docstring
that both are improper for this head, with the measured minimisers from §5.0, and remove
them from any search space. If they are ever used, the correction map must be applied to
the softmax before the rollout samples (`Ψ^γ` for focal, dividing out `w` for weighted
CE).

**Why.** They are the intuitive fix for 98% zeros and they are wrong here, by a factor of
33× and 6× respectively on the forecast mean. Both push the forecast up, and the archived
neural models are already biased up by +39% / +11%. Someone will reach for them; the
measurement should be on record when they do.

**Test.** This one is *worth* an ablation precisely because the prediction is so specific
and so large: run a two-arm `cross_entropy` vs `weighted_ce` ablation at
`N_STUDIES = 3` (three studies is plenty to detect a predicted +1000%-scale bias) and put
the resulting `bias_percent` in the thesis as a negative result. It is cheap, it is
decisive, and "the standard remedy for class imbalance is catastrophic for a
sampling-based forecast" is a more interesting sentence than a marginal RMSE improvement.

---

### R4 — Add a batch-level aggregate-matching term (`CE + λ(E_q[y] − ȳ)²`)

**Change.** A new loss module in `models/losses.py` plus a `loss_type` string. It needs
the class-index axis to compute `Σ_k k q_k`, which the existing `(B*T, K)` call signature
supports unchanged.

**Why.** It is the only candidate here that targets `bias_percent` directly, and it is
proper-preserving (the truth remains a global minimiser, so it shrinks rather than
tilts).

**Expected effect.** Smaller than R1/R2 and bounded by §4.2's finding that a perfect level
correction buys all of bias but only 6–25% of MAPE and none of RMSE. It also only sees
the teacher-forced one-step aggregate, not the rolled-out one.

**Test.** Same ablation shape. Worth doing only after R1/R2 have been read.

---

### R5 — Things that are not loss changes, ordered by expected value

Listed because on the measured evidence each of these is worth more than any loss change,
and a reader deciding where to spend a GPU week deserves to know.

1. **Give the electronics config AR features.** `ar_features=("period_since_last_transaction",
   "has_transacted_before")` in `scripts/run_studies.py`'s `build_panel_config`, matching
   CDNOW. Recency is the channel through which a model can represent the 43% cohort decay
   that §2.3(b) identifies as the dominant bias driver, and the archived runs did not have
   it. Testable as a two-arm suite, and the expected effect on `bias_percent` is far
   larger than 20 points.
2. **Declare the electronics covariates.** `Gender`, `Income` and `high.season` are in the
   CSV and used by nothing. The covariate-subset search exists to decide whether they
   help; it cannot decide about columns `PanelConfig` never names.
3. **`n_simulations = 300` minimum** (§4.1).
4. **Diagnose the exposure-bias gap of §4.3 before trying to fix it.** This is the
   mismatch that best explains the *sign* of the neural bias. The cheap diagnostic is to
   score a trained model two ways over the validation window — teacher-forced, and through
   a leak-free rollout — and report the gap; that is most of what retired ADR-0003 built,
   so read its "Why it goes" section first. Do **not** reach for scheduled sampling as the
   fix: Huszár proves its objective "is improper and leads to an inconsistent learning
   algorithm" (https://arxiv.org/abs/1511.05101), which surrenders the exact property §5.1
   identifies as making the simulator correct, and the Direct Multi-Horizon alternative is
   structurally unavailable under ADR-0002. **Cannot be tested by a pinned-arm study suite
   without new code**, which is why it is last despite being the most interesting hypothesis
   in this document.

---

## 7. What was not verified

Measurement gaps first, then reading gaps.

**Not measured**

- **That any of R1–R4 actually improves a forecast.** Every claim in §5 is about a loss's
  *population minimiser* on the measured class mix, computed on the simplex — not about
  what a finite LSTM/Transformer trained by AdamW on 63k–70k cells converges to. The
  direction of each effect is established; the magnitude in a real study is not. No
  training run was performed for this document as first written. **R1 and R2 have since
  been run on CDNOW** (2026-08-31) and are falsified there — see the Measured blocks in
  §6. R3's argument remains unrun by design and R4 is untested, and no recommendation has
  been run on electronics at all.
- **The exposure-bias mechanism of §4.3.** The *literature* is verified (Bengio et al.,
  Huszár, Wen et al.); the claim that the compounding runs *upward on this data* is
  inferred from the code path and the sign of the archived bias (neural +39%/+11% vs
  Pareto/NBD −53%), not demonstrated.
- **Any CDNOW baseline beyond the loss ablation's own control arm.** `ls Studies/` now
  returns 19 entries, of which exactly one is CDNOW: `Studies/loss_ablation_cdnow`. Its
  `LSTM_ce` arm is the first archived CDNOW forecast this project has (10 studies,
  `mape_aggregate` 41.2 ± 25.6, `bias_percent` −17.0 ± 39.0), and it exists only as that
  ablation's control. `scripts/run_cdnow_embedding_ablation.py` still has never produced
  an archive. Every archived number in §4 is electronics, and the CDNOW arguments in §5
  rest on the panel's measured statistics (§2.1) and the loss minimisers (§5.0), not on
  that one arm.
- **Whether the archived electronics suite is comparable to today's config.** Its
  `config.json` records `validation_start = 2000-01-01` and `seq_cols` including
  `week_sin`/`week_cos`; the current `scripts/run_studies.py` uses `2000-07-01` and no time
  features. The §4.2 decomposition is valid *for that suite*; its numbers are not today's
  baseline.
- **Pareto/NBD's −53% bias** is quoted from the archived `results.csv` and not
  investigated. Used only to establish that the neural bias has the opposite sign.

**Not read — sources behind paywalls or bot-blocks**

- **Lambert (1992)** ZIP and **Mullahy (1986)** hurdle. Publisher hosts returned 403; for
  Mullahy no abstract text is retrievable at all. §5.6 now cites Zeileis, Kleiber & Jackman
  (JSS 27(8), open access) for both formulations instead, and its own argument is derived
  and numerically verified regardless.
- **Warton (2005)**, "Many zeros does not mean zero inflation". Wiley 403'd; the §5.6 quote
  is an OpenAlex inverted-index reconstruction — **near-verbatim, not exact**. Treat the
  wording as paraphrase.
- **Schmittlein, Morrison & Colombo (1987)** full text. INFORMS blocks automated access;
  only Crossref/OpenAlex metadata was available. Note that this matters less than it looks:
  Hardie's derivation note 009 states the Pareto/NBD likelihood is "something not presented
  in SMC", so citing SMC for the likelihood is second-hand in the first place.
- **Ben Taieb & Atiya (2016)** on recursive-vs-direct multi-step bias. IEEE paywall; the
  abstract confirms only that they derive a bias/variance analysis, not the specific verdict.
- **Frank & Hall (2001)** — abstract verified, the class-probability differencing formula
  itself not. **Cheng, Wang & Pollastri (2007)** — abstract verified; the thermometer/
  cumulative encoding and the co-author list are not confirmed from the arXiv record. Neither
  is load-bearing: both predict a *rank* rather than a normalised PMF, so neither is
  compatible with C3's sampler (§5.5 note).
- **Wheatcroft (2019)** was read via its arXiv PDF and is quoted accurately, but its
  experiments concern discrimination power *as an evaluation metric*, not training
  dynamics. The extrapolation to "RPS may be a worse training loss" is **not** something he
  demonstrates, and this document does not assert it.

**Verified since first draft** (recorded so the caveat is not re-added): Valendin et al.'s
loss is cross-entropy on a class index, confirmed from both the accepted manuscript and the
vendored code in `Original_paper_model/`; squared EMD is the RPS / discrete CRPS and is
strictly proper (G&R Eq. 49, plus an elementary collapse of the Lebesgue integral); Hou et
al.'s formulation and AADB result; the PyTorch `PoissonNLLLoss` and `NegativeBinomial`
parameterisations; Tweedie's disqualification for integer counts.

**Deliberately not investigated**

- **Per-dataset values of `k` in Valendin.** The paper gives only the data-driven rule
  (observed maximum + 1), so there is no published constant to compare `clip_target_upper`
  against.
- **CRPS for parametric count heads.** `scoringRules` implements exact CRPS for Poisson and
  negative binomial, so the R1/R2 argument would survive a head change — but §5.7 rules a
  head change out of scope under `CLAUDE.md`, so this was not pursued.

---

## 8. Sources

| Claim | Source |
|---|---|
| Strictly proper scoring rules; log score is strictly proper; `S* = cS + h(ω)` preserves properness only for constant `c` | Gneiting & Raftery, JASA 2007 — https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf |
| "NLL is minimized if and only if π̂(Y\|X) recovers the ground truth conditional"; modern nets are over-confident; temperature scaling | Guo, Pleiss, Sun, Weinberger, ICML 2017 — https://arxiv.org/abs/1706.04599 |
| Focal loss `−α_t(1−p_t)^γ log p_t`; α by inverse class frequency; no calibration claim | Lin, Goyal, Girshick, He, Dollár, ICCV 2017 — https://arxiv.org/abs/1708.02002 |
| Focal loss as entropy-regularised KL, `L_f ≥ KL(q‖p̂) − γH[p̂]`; "favours a more entropic solution" | Mukhoti et al., NeurIPS 2020 — https://arxiv.org/abs/2002.09437 |
| Focal loss is classification-calibrated (Thm 3) but **not strictly proper** (Thm 5); η-underconfidence (Cor. 9); correction map Ψ^γ (Thm 11); temperature scaling insufficient (§6.3) | Charoenphakdee, Vongkulbhisal, Chairatanakul, Sugiyama, CVPR 2021 — https://arxiv.org/abs/2011.09172 |
| Weighting is rebalancing; posterior under a changed base rate (Thm 2) | Elkan, IJCAI 2001 — https://cseweb.ucsd.edu/~elkan/rescale.pdf |
| Inverse-frequency weighting targets `P(y\|x)/P(y)`; correct by adding `ln P(y)` to logits | Menon et al., ICLR 2021 — https://arxiv.org/abs/2007.07314 |
| Softmax under class-imbalanced training: `φ̂_j ∝ n_j e^{η_j}` (Thm 1) | Ren et al., NeurIPS 2020 — https://arxiv.org/abs/2007.10740 |
| Importance weighting washes out — **on separable data trained to zero loss** | Byrd & Lipton, ICML 2019 — https://arxiv.org/abs/1812.03372 |
| Exposure bias in autoregressive rollouts; scheduled sampling | Bengio, Vinyals, Jaitly, Shazeer, NeurIPS 2015 — https://arxiv.org/abs/1506.03099 |
| The Pareto/NBD likelihood is **not** in Schmittlein, Morrison & Colombo (1987) — that derivation is Hardie's, and recency/frequency `(x, t_x, T)` are sufficient statistics | Hardie, derivation note 009 — http://www.brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf |
| What the customer-base field asks a count model to deliver: "(i) estimate the model parameters, (ii) create the expected frequency distribution of transactions…, (iii) generate the aggregate sales forecast, and (iv) predict a particular customer's future purchasing" | Fader & Hardie, note 005 — http://www.brucehardie.com/notes/005/ |
| Negative-binomial likelihood + ancestral sampling in a recurrent forecaster | Salinas, Flunkert, Gasthaus, Januschowski — https://arxiv.org/abs/1704.04110 |
| Squared EMD loss `E_E = Σ_i (CDF_i(p) − CDF_i(t))²`; Mallows closed form; "squaring usually leads to faster convergence"; the AADB train-on-bins / score-the-expectation result | Hou, Yu, Samaras (2016) — https://arxiv.org/abs/1611.05916 |
| RPS as the discrete-ordinal score | Epstein (1969), J. Appl. Meteor. 8:985-987 — https://journals.ametsoc.org/view/journals/apme/8/6/1520-0450_1969_008_0985_assfpf_2_0_co_2.xml |
| CRPS Eq. (20) strictly proper; Eq. (49) Brier + point measures ⇒ RPS; distance sensitivity; Bernardo (1979) on locality | Gneiting & Raftery, JASA 2007 — https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf |
| "The CRPS is a strictly proper scoring rule for the class of probability distributions with finite first moment"; RPS as "the precursory, discrete version of the CRPS" | Jordan, Krüger & Lerch, JSS 90(12) — https://cran.r-project.org/web/packages/scoringRules/vignettes/article.pdf |
| Cramér distance = CRPS; **unbiased sample gradients** (Thm. 2), which Wasserstein lacks | Bellemare et al. — https://arxiv.org/abs/1705.10743 |
| The case against RPS — distance sensitivity contested; scope is evaluation discrimination, not training | Wheatcroft (2019) — https://arxiv.org/abs/1908.08980 |
| CRPS loss on a binned softmax head beats cross-entropy | Marchesoni-Acland et al., IEEE URUCON 2024 — https://doi.org/10.1109/URUCON63440.2024.10850406 |
| Valendin et al.'s loss is NLL over a multinomial (§2.2); fn. 9 softmax rationale; fn. 30 "other loss functions can be useful"; fn. 19 against MAE; `k` = observed maximum + 1 | Accepted manuscript — https://ars.els-cdn.com/content/image/1-s2.0-S0167811622000180-am.pdf (DOI https://doi.org/10.1016/j.ijresmar.2022.02.007); code https://github.com/valendin/rfm2lstm; vendored copy `Original_paper_model/banking_transactions_demo.ipynb` |
| Negative-binomial likelihood, softplus parameterisation, `Var[z] = μ + μ²α`, ancestral sampling, and why count heads cannot be scaled in preprocessing | Salinas, Flunkert, Gasthaus, Januschowski — https://arxiv.org/abs/1704.04110 |
| ZIP and hurdle formulations (attributing the hurdle to Mullahy 1986); NB alone recovers 608 of 683 zeros | Zeileis, Kleiber & Jackman, JSS 27(8) — https://www.jstatsoft.org/index.php/jss/article/view/v027i08/v27i08.pdf |
| "Many zeros does not mean zero inflation" — NB without zero-inflation best-fitting across 20 datasets | Warton (2005), Environmetrics 16(3) — https://doi.org/10.1002/env.702 (**abstract reconstructed, near-verbatim**) |
| Tweedie power semantics; "For 0 < power < 1, no distribution exists"; counts → Poisson with a log link | scikit-learn GLM guide §1.1.12 — https://scikit-learn.org/stable/modules/linear_model.html |
| For `1 < p < 2` Tweedie is "continuous for Y greater than zero, with a positive mass at Y = 0" | Dunn, CRAN `tweedie` manual — https://cran.r-project.org/web/packages/tweedie/tweedie.pdf |
| `PoissonNLLLoss` drops `log(target!)` when `full=False` | https://docs.pytorch.org/docs/stable/generated/torch.nn.PoissonNLLLoss.html |
| `NegativeBinomial` uses the inverted convention (`probs` = success probability) | https://docs.pytorch.org/docs/stable/distributions.html#negativebinomial |
| Scheduled sampling's objective "is improper and leads to an inconsistent learning algorithm" | Huszár (2015) — https://arxiv.org/abs/1511.05101 |
| Recursive vs Direct Multi-Horizon; "avoids error accumulation"; DeepAR as the recursive baseline | Wen, Torkkola, Narayanaswamy, Madeka (MQ-RNN) — https://arxiv.org/abs/1711.11053 |
| BG/NBD likelihood and estimation; aggregate tracking excellent while individual correlation is 0.626 and "no other researchers have reported such measures" | Fader, Hardie & Lee, Marketing Science 2005 — http://www.brucehardie.com/papers/018/fader_et_al_mksc_05.pdf |
| "the ability to make such individual-level predictions lies at the heart of any serious attempt to compute the (residual) customer lifetime value" | Fader & Hardie, J. Interactive Marketing 2009 — https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Fader_hardie_jim_09.pdf |
