# The training budget: how long a model actually trains, and what that costs

Every neural result in this package comes from a model that stopped training when
`fit_model` saw `patience` epochs in a row without an improvement in validation
cross-entropy. That rule has never been examined. This document examines it.

**The verdict, up front.** The rule ends every run long before the validation loss stops
falling — our archived winners kept the weights from a median epoch of 8–16, while the
same model trained without early stopping keeps improving to epoch 88–247. The reason is
not that 7 is a badly chosen constant. It is that our training recipe is not the one the
paper we reproduce uses: the notebook trains at batch 32 with plain Adam for about 90
epochs, roughly 2,300 gradient updates, and our benchmark winners receive about 32. §4 is
that comparison, and the rest of the document is the evidence that it matters.

**Tested, §9.** On electronics, 20 independent replications per arm: training the frozen
benchmark for the paper's own 90 epochs moves MAPE by −22.3 (95% CI −28.3 to −16.2) and
per-customer Spearman by +0.150 (95% CI +0.107 to +0.191). Copying the paper's *settings*
without the epochs shows no clear difference in either at n = 20. **The electronics collapse reported in
`docs/benchmarks-real-panels.md` is substantially a training artefact** — that row is
ours, not a published result of Valendin et al.

**And §10 finds why, which is not what §4 assumed.** The paper's stopping rule keeps the
checkpoint from epoch 1 here, terminating after 7 epochs; under the paper's own
customer-wise split on this same panel it keeps epoch 28–56 and runs 34–62. The defect is
in the combination — our split with their stopping rule — not in either alone, and §11
states what that does and does not let the thesis claim.

Everything below was measured on 2026-09-20: the archive numbers by reading `Studies/`,
the rest by re-running the frozen benchmark with the project venv on the ROCm
workstation. The scripts are in `.scratch/training-budget/` and are named at each step.

---

## Revised under the statistical protocol (2026-09-28)

Every section was re-scored under `docs/statistical-protocol.md`. What changed, by the
sections each revision covers:

### Selection

Moved to `docs/model-selection.md`.

### Training length and inputs: §9, §15.1, §15.2 (register rows 5, 6, 14–19)

Every Δ is recomputed by `effect()` — §9 by `.scratch/training-budget/family_t_stats.py`
from `results/family_t_scores.csv`, §15 by `all_effects.py` and `factorial_analysis.py`
from `results/factorial.csv` — and Pareto/NBD is now 20 seeded fits per panel on the same
2-year windows, compared with the independent bootstrap. Interval endpoints moved in the
last digit; no verdict on a contrast between two arms flipped. What did:

- **Row 19, best cell against Pareto/NBD on electronics:** "interval above it"
  (ValendinLSTM) / "contains it" (LSTM) → **supported below** on both (Δ −0.009, −0.017 to
  −0.000; −0.012, −0.023 to −0.002). Pareto/NBD's mean over 20 fits is 0.314, not the
  single fit's 0.297.
- **Row 19, multichannel:** ValendinLSTM "contains it" → supported above (+0.010, +0.000 to
  +0.020, smaller than the refit noise); LSTM "contains it" → no clear difference.
- **Row 19b, cdnow and gift:** below → below, now as supported Δ (−0.015 to −0.047).
- **Row 6, the paper's settings without the floor:** `bounded` → **no clear difference** in
  MAPE and Spearman at n = 20 (no equivalence margin is met), and ValendinLSTM's |bias| is
  supported lower (Δ −11.8, −21.2 to −2.3).
- **Row 17, floor on top of the label on electronics:** `bounded` within ±0.012 → **no
  clear difference** (ValendinLSTM −0.011 to +0.012, LSTM −0.005 to +0.035); the
  refit-noise margin of ±0.0105 is not met.
- **Row 16, floor on cdnow Spearman:** "no detectable effect" → no clear difference at
  n = 20 (not an absence).
- **Row 18b, context added:** with the cluster label the same floor shows no clear MAPE
  difference on CDNOW's LSTM (+4.5, −27.1 to +34.4).

## Claims register

What this document currently asserts, how strongly, and where the evidence is. **Status**
is `established` (a 95% bootstrap CI on the difference excludes zero, or the observation
is a direct measurement), `bounded` (an effect that is real only within a stated
interval, or absent to within a stated margin), `retracted`, or `not identified` (the
experiment cannot separate the cause claimed). Every row names the panel it was measured
on; none generalises.

| # | claim | type | panel | metric | status | § |
| ---: | --- | --- | --- | --- | --- | ---: |
| 1 | Patience, not the epoch budget, ends every archived run | descriptive | all four | — | established | 1 |
| 2 | With early stopping off, validation CE keeps falling to epoch 88–247 | mechanism | cdnow, electronics, multichannel (n=3) | val CE | established | 2 |
| 3 | The reference notebook's training recipe differs from ours in optimizer, weight decay, learning-rate selection, batch size, patience and epoch budget | descriptive | — | — | established | 4 |
| 4 | Under patience 7 the search prefers the batch size that trains least | descriptive | electronics | val CE | established | 5 |
| 5 | Training to the paper's epoch count improves level and discrimination | superiority | electronics (n = 20 / 20) | MAPE, Spearman | established (ValendinLSTM MAPE −22.3, −28.3 to −16.2; Spearman +0.150, +0.107 to +0.191) | 9 |
| 6 | The paper's settings without the floor show no clear difference | descriptive | electronics (n = 20 / 20) | MAPE, Spearman | no clear difference on both models (**was `bounded`**; not an equivalence); ValendinLSTM's \|bias\| is supported lower (−11.8) | 9 |
| 7 | The paper's rule stops early under our split and not under theirs | mechanism | electronics (n=3) | epochs | established | 10 |
| 8 | The temporal curve gains 5.4×10⁻⁵/epoch against a 10⁻⁴ threshold | mechanism | electronics (n=1/split) | val CE | established | 13.2 |
| 9 | Validation CE is wrong-signed against the holdout | superiority | electronics (LSTM, ValendinLSTM; n = 40 studies each) | MAPE, \|bias\| | established, both models | 14.1 |
| 9b | …and right-signed on cdnow | superiority | cdnow (LSTM, ValendinLSTM; n = 5 studies each) | MAPE, \|bias\|, Spearman | LSTM: established on all three; ValendinLSTM: established on Spearman only (MAPE, \|bias\| **not supported**) | 15.3 |
| 10 | A validation rollout ranks trials better than CE | superiority | electronics (n = 40 per model) | MAPE | established, both models (LSTM Δ +0.140, +0.068 to +0.213; ValendinLSTM +0.208, +0.138 to +0.281) | 14.3 |
| 10b | …but its own correlation with the holdout is not supported | descriptive | electronics (n = 40 per model) | MAPE | LSTM: holds (−0.041, −0.112 to +0.032); ValendinLSTM: **overturned** — supported +0.107 (+0.046 to +0.168) | 14.3 |
| 11 | A composite of three criteria is worse than whichever matches the target | superiority (negative) | electronics (n = 40 per model) | MAPE, Spearman | established for MAPE on both models and Spearman on LSTM; ValendinLSTM Spearman not supported | 14.3 |
| 12 | *The wrong sign comes from a calibration/holdout rate shift* | mechanism | — | — | **retracted** | 14.2, 15.3 |
| 13 | CE selects well where a study's trials differ and badly where they do not | mechanism | cdnow, electronics | — | described on two panels, not a tested mechanism | 15.3 |
| 14 | A cluster label improves discrimination | superiority | electronics, multichannel (n = 20 / 20) | Spearman | established | 15.1 |
| 14b | …on cdnow, and on gift's benchmark | superiority | cdnow, gift | Spearman | **not supported** | 15.1 |
| 14c | A cluster label added to an already-floored model improves discrimination | superiority | all four, separately (8/8 cells) | Spearman | established | 15.1 |
| 15 | A training floor improves discrimination | superiority | electronics, multichannel | Spearman | established | 15.1 |
| 16 | A training floor shows no clear difference in discrimination | descriptive | cdnow (n = 20 / 20) | Spearman | no clear difference at n = 20 (not an absence) | 15.1 |
| 16b | A training floor **worsens** discrimination | superiority (negative) | gift | Spearman | established | 15.1 |
| 17 | Adding the floor on top of the label shows no clear difference | descriptive | electronics (n = 20 / 20) | Spearman | no clear difference (ValendinLSTM −0.011 to +0.012, LSTM −0.005 to +0.035); **was `bounded` within ±0.012 — the refit-noise margin ±0.0105 is not met** | 15.1 |
| 18 | The floored paper-recipe arm triples CDNOW's LSTM error | superiority (negative) | cdnow (n = 20 / 20) | MAPE | established (+126.2, +77.2 to +173.2) | 15.2 |
| 18b | *…and the floor is what causes it* | mechanism | cdnow | MAPE | **not identified** (E1) | 15.2 |
| 19 | The best cell ranks within about ±0.01 of Pareto/NBD (20 seeded fits) | superiority, both directions | electronics, multichannel (n = 20 / 20) | Spearman | electronics: **below**, supported, on both models (−0.009, −0.012; was "contains or exceeds"); multichannel: ValendinLSTM above (+0.010), LSTM no clear difference | 15.1 |
| 19b | …and remains below it | superiority (negative) | cdnow, gift (n = 20 / 20) | Spearman | established (−0.015 to −0.047, every interval below 0) | 15.1 |
| 20 | The replication RNG coupling does not reach the archive | mechanism | all four | val CE | established | 7 |

## How claims are made

The standard the whole document is held to. It was written after §15 and then applied
back to every section, including those written before it; the rows above reading "no
clear difference" are claims it downgraded from the stronger form they were first stated
in.

**The rules are `docs/statistical-protocol.md`.** Every effect below is Δ = M̄_B − M̄_A
with a 95% percentile-bootstrap interval, supported when it excludes zero, computed by
`panelclv.evaluation.effects.effect`. Replications of a real-panel condition are
independent searches, so they are resampled separately; criteria scored on the same
studies are paired. The metric roles, the reading of refit noise as magnitude only, the
equivalence rule and the one-panel-at-a-time rule are all stated there. What this section
keeps is what is specific to this document: the refit-noise values, the panels, and the
seeding protocol that makes the replications independent.

Forecast CV (`std / mean` of per-customer predicted holdout totals) is reported beside
Spearman wherever forecasts are compared. Spearman is rank-based: a forecast varying by a
few percent can still rank well, and CV near 0 is what this document calls a collapse.
Collapse belongs to a model under a configuration, not to a panel, so CV lives in the
comparison tables.

**The refit noise is a magnitude reference, not a second threshold.** Refit the same
checkpoint a second time, change nothing else, and the forecast still moves. Measured over
the 80 archived family-N winners paired with their re-scored refits
(`.scratch/training-budget/refit_noise.py`):

| panel | MAPE | bias % | Spearman | RMSE |
| --- | ---: | ---: | ---: | ---: |
| cdnow | 5.83 | 12.58 | 0.0159 | 0.00036 |
| electronics | 3.63 | 5.94 | 0.0105 | 0.00014 |
| gift | 5.89 | 14.99 | 0.0116 | 0.00009 |
| multichannel | 7.71 | 12.56 | 0.0152 | 0.00003 |

**The panels.** Every result is stated for the panel it was measured on, and where panels
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

**Seeding protocol**, because "independent replications" is the load-bearing assumption
above. Within a worker, replication *r*'s forecast seeds the RNG entering replication
*r+1*'s training (§7); work lists are arm-major and strided across workers, so two arms at
the same replication index land at different positions on different machines and are **not
paired**. Conditions are therefore compared as independent samples, which is what the
bootstrap resamples. The archive-wide check found 0 exact repeats across 3,020 suites.

## Contents

1. [Early stopping, not the epoch budget, ends every run](#1-early-stopping-not-the-epoch-budget-ends-every-run)
2. [The validation loss keeps falling long after that](#2-the-validation-loss-keeps-falling-long-after-that)
3. [What the early stop costs the forecast](#3-what-the-early-stop-costs-the-forecast)
4. [The recipe the paper actually uses](#4-the-recipe-the-paper-actually-uses)
5. [Why the search selects the setting that trains least](#5-why-the-search-selects-the-setting-that-trains-least)
6. [How much of the ranking collapse this explains](#6-how-much-of-the-ranking-collapse-this-explains)
7. [An unrelated finding: the forecast seeds the next replication's training](#7-an-unrelated-finding-the-forecast-seeds-the-next-replications-training)
8. [What this does not establish](#8-what-this-does-not-establish)
9. [The experiment, and what it showed](#9-the-experiment-and-what-it-showed)
10. [Why the paper's rule keeps epoch 1 here: the split, not the panel](#10-why-the-papers-rule-keeps-epoch-1-here-the-split-not-the-panel)
11. [What this is, relative to Valendin et al.](#11-what-this-is-relative-to-valendin-et-al)
12. [What this changes](#12-what-this-changes)
13. [What to try next: keep the temporal split, fix what it feeds](#13-what-to-try-next-keep-the-temporal-split-fix-what-it-feeds)
14. [The selection test: cross-entropy is worse than useless, and nothing else is good](#14-the-selection-test-cross-entropy-is-worse-than-useless-and-nothing-else-is-good)
15. [Family U: crossing the two levers adds little, and one of them may be dangerous](#15-family-u-crossing-the-two-levers-adds-little-and-one-of-them-may-be-dangerous)

---

## 1. Early stopping, not the epoch budget, ends every run

`training/loop.py` breaks after `patience` consecutive non-improving epochs and restores
the best epoch's weights. So the number of epochs a trial ran is

```
epochs_run = best_epoch + 1 + patience      (capped at n_epochs)
```

and `best_epoch`, the value stored in every trial's `user_attrs`, is the epoch whose
weights were kept — not the epoch training ended at.

The formula is ours, derived from `fit_model`'s stopping rule; it is not in Valendin et
al.'s code or paper. Trials store only `best_epoch`, so "epochs run" below is computed from
it, not recorded. It holds only when two conditions are met:

- **`min_epochs = 0`.** A training floor holds the break until the floor, so a floored run
  trains at least `min_epochs` epochs whatever its best epoch was. Every trial in the table
  below has no floor: it is the archive before the `training_budget`, `factorial` and
  `selection_rescore` families, which are the only ones that set one.
- **The trial completed.** A trial Optuna pruned stopped for another reason. Only
  `COMPLETE` trials are counted.

Where these do not hold, or wherever a comparison can be made on `best_epoch` directly,
this document uses `best_epoch`.

Across every archived study on the four real panels (407,950 trials, of which 163,363
completed; `patience = 7`, `n_epochs = 100` in all but 308 of them):

| panel | model | median best epoch | mean epochs run | share reaching the 100-epoch budget |
| --- | --- | ---: | ---: | ---: |
| cdnow | LSTM | 10 | 21.3 | 0.8% |
| | ValendinLSTM | 14 | 24.0 | 0.1% |
| | Transformer | 16 | 30.6 | 3.8% |
| electronics | LSTM | 8 | 21.2 | 0.8% |
| | ValendinLSTM | 7 | 16.1 | 0.3% |
| | Transformer | 15 | 29.6 | 5.2% |
| gift | LSTM | 13 | 23.9 | 0.4% |
| multichannel | LSTM | 8 | 20.0 | 0.5% |
| | ValendinLSTM | 4 | 12.5 | 0.0% |

Almost nothing ever reaches the epoch budget. The 100 in `n_epochs` is decoration; the
number that decided how long every model trained is the 7 in `patience`.

This extends two things `docs/benchmarks-real-panels.md` already reports. Its section
"Optuna's validation loss cannot see which runs forecast badly" notes that very early
stopping is common on multichannel — the best trial stopped at epoch 3 or earlier in 55%
of ValendinLSTM runs there — and its section "The refit noise, and the stopping
epoch" finds that within a study, the stopping epoch's rank correlation with holdout
Spearman averages +0.48 on CDNOW and gift: trials that trained longer rank better. Both
treat the stopping epoch as something observed. This document asks what sets it.

Because `patience` is 7 everywhere, it has no variance to study. The variable that does
vary is the **number of gradient updates** the selected weights received:

```
updates = ceil(N_customers / batch_size) × (best_epoch + 1)
```

That is the quantity to condition on when comparing runs, and §5 is about it.

*(`.scratch/training-budget/epochs.py`)*

## 2. The validation loss keeps falling long after that

Re-run the frozen benchmark with early stopping switched off — 200 epochs, no break, the
hyperparameters an archived electronics study actually selected (lr 1.15e-3, weight decay
5.67e-4, batch 256), three replications per panel:

| panel | patience 7 stops at | its best val CE | best over 200 epochs | at epoch | improvement forgone |
| --- | ---: | ---: | ---: | ---: | ---: |
| electronics | 19 | 0.0888–0.0896 | 0.0848–0.0858 | 137–197 | 3.4–5.3% |
| multichannel | 12–16 | 0.0229–0.0230 | 0.0220–0.0226 | 169–181 | 1.4–4.4% |
| cdnow | 11 | 0.1078–0.1098 | 0.0946–0.0964 | 88–120 | 10.7–13.0% |

The curve does not climb steadily. It sits flat and then drops, and the flat stretches are
long: before reaching its own optimum, each run went 37–62 epochs (electronics), 29–62
(cdnow) and 56–131 (multichannel) without a single improvement. A patience of 7 cannot
survive any of those, and a larger constant is an unreliable fix — with `patience = 40`,
one of five electronics replications found the later optimum; the other four trained about
50 epochs and still kept the weights from epoch 6–11.

*(`.scratch/training-budget/probe_patience.py`)*

## 3. What the early stop costs the forecast

A better validation loss is only interesting if the forecast follows. Electronics, frozen
benchmark (transaction count and calendar week, nothing else), the ADR-0008 refit and a
200-path Monte Carlo forecast — exactly what a study does — five replications per arm,
differing only in the stopping rule:

| arm | best epoch | val CE | bias % | MAPE | Spearman | sd of predicted totals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| patience 7 (the archived setting) | 8 | 0.0894 | +31.0 ± 12.9 | 64.3 | 0.013 ± 0.028 | 0.19 |
| no early stop, 300 epochs | 183 | 0.0859 | +9.9 ± 27.3 | **49.9** | **0.091 ± 0.060** | 0.23 |

Four of the five unstopped runs reached a deep optimum at epoch 215–247 and scored MAPE
35.9–56.0 with Spearman 0.08–0.17; the fifth never left epoch 11 and looks like a
patience-7 run.

> **Superseded by §9**, which runs the same comparison at n = 20 with the standard's Δ and
> intervals. Read this section as the pilot that motivated it, not as evidence: five
> replications, no interval, and the arms here are RNG-coupled (§7), so they are paired
> where §9's are independent. Its direction held; its magnitudes moved. Its last column is the
> raw sd of per-customer predicted totals rather than forecast CV, because the pilot did
> not store its forecasts.

**Read MAPE and Spearman here, not bias.** Bias is the secondary metric, and
`docs/benchmarks-real-panels.md` measures a refit noise on electronics of 8.9 points of sd
— refitting one checkpoint twice moves aggregate bias by that much with everything else
held fixed — as the magnitude to hold the bias difference above (+31.0 → +9.9, five
replications) against. None of this pilot's differences was tested; §9 tests them.

The patience-7 rows reproduce the archived electronics benchmark
(`docs/benchmarks-real-panels.md`: bias +46.0 ± 14.8, MAPE 70.8, Spearman 0.032), which is
the check that this comparison measures the same thing the archive did.

*(`.scratch/training-budget/end_to_end.py`)*

## 4. The recipe the paper actually uses

ADR-0004 freezes `benchmarks/valendin_lstm.py` against a reference notebook,
`Original_paper_model/banking_transactions_demo.ipynb`. The architecture was transcribed
from it layer for layer. The *training recipe* in the same notebook was not.

| | notebook | our studies |
| --- | --- | --- |
| optimizer | `Adam()`, all defaults — lr 1e-3, **no weight decay** | AdamW, lr searched 1e-4–3e-3, weight decay searched 1e-6–1e-2 |
| batch size | **32** | searched over **{64, 128, 256}** (`registry/model_registry.py:373`) |
| patience | 5 | 7 |
| epoch budget | 150 | 100 |
| epochs actually trained | ~90–100 | median best epoch 7–14 (§1) |

The first four rows are settings; the last is what they produce, not a setting of its own
(Keras's Adam also uses epsilon 1e-7 against torch's 1e-8, left out as immaterial). The
notebook's figure there is its own claim: *"This example takes about 100 epochs in total"*
and *"the final validation loss should end up around 0.44 after ±90 epochs with the
default parameters"* (cells 15 and 16).

Put that in updates. On electronics, 829 customers at batch 32 is 26 batches per epoch, so
90 epochs is about **2,300 gradient updates**. Our archived benchmark winners stopped at a
median epoch of 7 at batch 256 — about **32 updates**. Two orders of magnitude.

Nothing hides this: `scripts/validate_valendin_lstm.py`, the gate that proves the frozen
benchmark still reproduces the published numbers, runs the notebook's recipe exactly —
batch 32, Adam at defaults, patience 5, 150 epochs — and reaches the published validation
loss. The recipe works in this codebase. It has simply never been used for a study, because
studies reach `fit_model` through the Optuna search, and the search space cannot express
it: batch 32 is not in the set, and the weight decay range has no zero.

## 5. Why the search selects the setting that trains least

The 20 archived electronics benchmark studies ran 2,000 trials between them (713
completed, 1,287 pruned) over learning rate, weight decay and batch size. Grouped by the
batch size each trial sampled:

| batch size | completed trials | mean val CE | best val CE | median best epoch | median updates |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 136 | 0.0935 | 0.0899 | 3 | 52 |
| 128 | 178 | 0.0920 | 0.0911 | 4 | 35 |
| 256 | 399 | 0.0898 | 0.0886 | 7 | 32 |

Batch 256 won all 20 studies, and on this evidence it deserved to: under patience 7 it
gives the lowest validation loss. But it wins partly *because* of the stopping rule — a
small-batch trial needs more epochs to show its advantage, and patience 7 usually stops it
first. The two settings reinforce each other, and the search never sees the region worth
searching: the best validation loss across all 2,000 trials is 0.0886, while training one
model for longer reaches 0.0846.

*(`.scratch/training-budget/trials_by_bs.py`)*

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

*(This table first reported 0.09 for the trained row, from §3's five-replication pilot.
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

## 7. An unrelated finding: the forecast seeds the next replication's training

`forecast_recurrent` calls `torch.manual_seed(42 + replication)` before sampling
(`models/monte_carlo_forecasting.py:374`), and in a study suite each replication's
forecast runs immediately before the next replication's training. Mechanically, then, the
global RNG state entering replication *r+1* is fixed by replication *r*'s forecast seed,
and two runs that reach that point having drawn the same number of random numbers train
an identical model.

It is easy to hit. In the §3 experiment, replications 1–4 produced identical validation
losses across both arms and across a re-run (0.08891, 0.09127, 0.08893, 0.08907), because
each arm did the same work in the same order. That made §3 a paired comparison, which is
a benefit there.

**It does not reach the archive.** Checked across all 3,020 archived suites: among suites
of the same family and model, the winning validation loss repeats exactly at **no**
replication index — 0 of them, at replication 1 and at every later one (the only exact
repeats anywhere are Pareto/NBD, which is deterministic by design). An Optuna search
consumes a variable number of draws — different trial counts, different pruning — so by
the time a suite reaches its next replication the states have diverged. The replication
spreads reported in `docs/benchmarks-real-panels.md` and the insights documents are
genuine draws.

So this is a hazard for tightly controlled reruns, not a defect in the archive. Any future
experiment whose arms perform identical work in identical order — which includes the
`paper` arm of §9, at one trial per study — should either seed deliberately or vary the
order, and say which. Measurement:
`.scratch/training-budget/issues/05-measure-seed-coupling.md`.

## 8. What this does not establish

**Scope: §1–§7.** Written before the experiments, and three of these limits were later
lifted — §9 replaces the five replications with twenty and an interval, and §15 takes the
forecast comparison to all four panels. The last two still stand. The document's current
limits are in the closing **To do** section, §D.

- **One hyperparameter setting.** §2 and §3 fix learning rate, weight decay and batch size
  at one archived winner's values. A search allowed to train longer might select
  differently, and might close part of the gap by itself.
- **One panel end to end.** The forecast comparison in §3 is electronics only. The
  validation curves in §2 cover three panels, but a validation gain is not a holdout
  result.
- **Five replications, paired.** Enough to see a 14-point MAPE difference; not enough to
  put an interval on the Spearman difference, and not enough for bias at all (§3).
- **Nothing here says `patience` is the knob to turn.** A larger constant caught the later
  optimum once in five tries (§2). The paper's recipe (§4) and a minimum-epoch floor are
  the candidates worth testing, and §9 tests both.
- **The benchmark's schedule is ours, not the paper's.** ADR-0004 freezes the published
  architecture; the stopping rule and the search space around it are this package's
  choices. Changing them moves the benchmark rows, so it is a decision to record rather
  than a quiet edit.

## 9. The experiment, and what it showed

Family T, run on vast.ai on 20 September 2026: two models x four training recipes x 20
replications on electronics, 100 trials for the searched arms, 200 Monte Carlo paths,
160 suites, $0.44 of rented GPU. Declared in `.scratch/training-budget/spec.md`, run by
`scripts/run_training_budget.py`.

| arm | recipe | trials |
| --- | --- | ---: |
| `archive` | lr / weight decay / batch searched, `patience=7`, `n_epochs=100` | 100 |
| `paper` | pinned: `lr=1e-3`, `weight_decay=0.0`, `batch_size=32`, `patience=5`, `n_epochs=150` | 1 |
| `paper90` | the same, plus `min_epochs=90` — the notebook's own epoch count | 1 |
| `floor50` | `archive`'s search plus `min_epochs=50`, `n_epochs=300` | 100 |

### ValendinLSTM (the frozen benchmark)

| arm | bias % | MAPE | Spearman | forecast CV |
| --- | ---: | ---: | ---: | ---: |
| `archive` | +39.8 ± 20.4 | 69.1 | 0.027 ± 0.043 | 0.09 ± 0.03 |
| `paper` | +28.7 ± 11.1 | 68.5 | 0.016 ± 0.029 | 0.08 ± 0.00 |
| **`paper90`** | **−5.7 ± 27.0** | **46.8** | **0.177 ± 0.089** | **0.31 ± 0.18** |
| `floor50` | +4.7 ± 29.3 | 47.9 | 0.168 ± 0.084 | 0.24 ± 0.14 |

### LSTM, `no_ar-no_cluster-valendin`

| arm | bias % | MAPE | Spearman | forecast CV |
| --- | ---: | ---: | ---: | ---: |
| `archive` | +30.4 ± 16.7 | 59.3 | 0.030 ± 0.044 | 0.08 ± 0.01 |
| `paper` | +34.5 ± 15.7 | 63.5 | 0.020 ± 0.029 | 0.08 ± 0.01 |
| `paper90` | +31.1 ± 31.7 | 54.7 | 0.174 ± 0.075 | 0.24 ± 0.14 |
| **`floor50`** | **+0.3 ± 18.2** | **50.1** | 0.074 ± 0.049 | 0.11 ± 0.04 |

Each arm against `archive`, as Δ of condition means with a 95% percentile-bootstrap CI
(`effect()`, independent bootstrap), 20 replications each. Electronics' refit noise is
3.63 MAPE and 0.0105 Spearman. Regenerated by `.scratch/training-budget/family_t_stats.py`
from `results/family_t_scores.csv`.

| model | arm | Δ MAPE | 95% CI | Δ Spearman | 95% CI |
| --- | --- | ---: | :---: | ---: | :---: |
| ValendinLSTM | `paper` | −0.6 | −6.3 to +5.0 | −0.011 | −0.034 to +0.011 |
| | `paper90` | **−22.3** | **−28.3 to −16.2** | **+0.150** | **+0.107 to +0.191** |
| | `floor50` | **−21.2** | **−26.8 to −15.5** | **+0.141** | **+0.100 to +0.180** |
| LSTM | `paper` | +4.3 | −0.2 to +9.0 | −0.011 | −0.034 to +0.011 |
| | `paper90` | −4.5 | −12.9 to +4.6 | **+0.144** | **+0.106 to +0.180** |
| | `floor50` | **−9.2** | **−13.3 to −4.5** | **+0.043** | **+0.015 to +0.070** |

Bold rows are the supported ones — interval excludes zero. Two readings stand out:
**the LSTM's level gain under `paper90` is not
supported** (−4.5, interval spanning zero), and **the LSTM's discrimination gain under
`floor50`, though supported, is +0.043 against a refit noise of 0.0105** — about four
times the refit noise, where the benchmark's is about fourteen times it.

The secondary metric, |bias|, says one thing the primary ones do not: **`paper` lowers the
frozen benchmark's |bias|** (Δ −11.8, 95% CI −21.2 to −2.3; `paper90` −16.4, −26.0 to
−6.6; `floor50` −14.5, −24.3 to −4.5). For the LSTM only `floor50` does (−17.6, −25.9 to
−8.8); `paper` (+4.0, −5.2 to +13.6) and `paper90` (+3.7, −9.8 to +17.8) show no clear
difference.

**Three things follow, and the first is the one to remember.**

**The settings were never the point; the epochs were.** `paper` — the notebook's optimizer,
batch size and patience, pinned exactly — shows no clear difference from `archive` in MAPE
or Spearman at n = 20 (it moves the benchmark's |bias|, above, and nothing else), because
it keeps the checkpoint from **epoch 1** and terminates after 7:
copying the recipe copies its stopping rule, and that rule quits immediately here (§10
shows why). Best epoch and stop epoch are not the same number — §1 — and the one that
matters for how much training happened is the 7. Add the floor and
the same recipe cuts MAPE by 22 points and multiplies the ranking correlation by seven.
**Batch 32 is not the fix. Training past the early plateau is** — and *what* should be
held fixed is open: `paper90` gets there with ~2,300 updates at batch 32 while `floor50`
gets there with a few hundred at batch 256, so neither epochs nor gradient updates is a
clean invariant. §5's argument points at an optimizer-step budget; it is untested.

**The forecast collapse on electronics is substantially a training artefact.** The frozen
benchmark's published row (MAPE 70.8, Spearman 0.032) reproduces as `archive`; trained to
the paper's own epoch count the same architecture on the same inputs scores MAPE 46.8 and
Spearman 0.177, and the spread of its per-customer predictions more than doubles (0.21 to
0.49). It was not that the model could not tell customers apart. It was not trained long
enough to try.

**A floor works with the search intact**, which matters because only the benchmark has a
published recipe to copy. `floor50` reaches `paper90`'s ground on both models and is the
better of the two on level (LSTM bias +0.3 ± 18.2 against `archive`'s +30.4).

### What it does not overturn

Inputs still dominate ranking. A `kmeans_8` cluster label reaches Spearman 0.27
(`docs/insights-cluster-ablation.md` §5.1) and Pareto/NBD 0.314 (mean of 20 seeded fits),
against 0.177 here; §6's
ordering holds, with training length worth more than it looked at five replications
(0.178 at n = 20 against the pilot's 0.09) and still less than one persistent
per-customer channel.

RMSE separates nothing, as everywhere on these panels: every arm sits between 0.3763 and
0.3774 against the all-zero forecast's 0.3775.

Bias is the secondary metric: `paper90`'s −5.7 ± 27.0 is a better centre than
`archive`'s +39.8 ± 20.4, and its |bias| is supported lower (above), but the
across-replication spread grows. The claims of this section rest on MAPE and Spearman.

## 10. Why the paper's rule keeps epoch 1 here: the split, not the panel

`paper` keeping the checkpoint from epoch 1 — it runs 7 and rolls back — has two
possible causes, and they point in opposite
directions. Either the panel's validation curve is flat from the start — in which case the
published protocol does not transfer to sparse retail data — or **our** validation split
makes it flat, in which case the fault is ours.

It is ours. Running the notebook's recipe under the notebook's **own** split on electronics
— a random 10% of customers held out, every period scored, nothing else changed:

| replication | epochs run | best epoch | best val CE |
| --- | ---: | ---: | ---: |
| 0 | 62 | 56 | 0.1423 |
| 1 | 39 | 33 | 0.1421 |
| 2 | 34 | 28 | 0.1460 |

Under its own split the paper's rule trains **28-56 epochs** on this panel, in the same
range as the ~90 it reports on its banking data. Under our temporal split (ADR-0001) the
identical recipe quits at epoch 1.

(The two validation losses are not comparable — 0.142 customer-wise against 0.089 temporal
— because they score different quantities on different rows. Only the epoch counts are.)

So the mechanism is an **interaction**, not a defect in the published method: a temporal
validation window over all customers gives a much flatter early curve than a held-out
slice of customers does, and a patience rule reads that flatness as convergence. §13.2
measures exactly how: the temporal curve gains 5.4×10⁻⁵ per epoch on average while
`fit_model` requires **10⁻⁴** to count an epoch as an improvement
(`loop.py`, `improved = (val_loss + 1e-4) < best_val_loss`), so the average epoch improves
by half the threshold it must clear. (`min_delta=0` is the *notebook's* Keras setting, not
ours; an earlier draft of this section attributed it to our loop.)
This package departs from the paper's split deliberately and documents why: a
customer-wise split scores the same calendar periods the model trained on, so it measures
generalisation across the cross-section rather than across time, and the thing being
forecast is the future (ADR-0001). **It is not a leak** — both splits stay inside the
calibration window and neither touches the holdout; §13.1 states this correctly and an
earlier draft of this paragraph said "leaks time", which was wrong. What nobody checked is
what the departure did to the stopping rule bolted on beside it. It quietly cost most of
the training.

*(`.scratch/training-budget/paper_split_check.py`)*

## 11. What this is, relative to Valendin et al.

Worth stating exactly, because the temptation is to claim too much and the honest version
is still worth having.

**Not a correction of the paper.** Their protocol, run as published — their model, their
optimizer, their batch size, their patience, their customer-wise split — trains for a
sensible number of epochs on our panels too (§10). Nothing here says their reported
results are wrong, and every archived number in this repository stands: `archive`
reproduces the published electronics row to within its spread (§9).

**A correction of this package's reproduction of it.** ADR-0001 replaces the paper's
customer-wise validation split with a temporal one, which is the right call for a
forecasting evaluation and is the one deliberate departure the benchmark documents. What
went unnoticed is that the paper's early-stopping rule does not survive that swap: on a
temporal window over a 98.6%-zero panel the curve is flat from epoch 1 and patience fires
immediately. Every neural result in this repository was trained under that combination.

**And a contribution, stated narrowly.** Combining this architecture with a temporally
honest validation split **can** require a training floor — or a different stopping
criterion — or the model stops after the first large adjustment and never reaches the long,
slow refinement that the forecast depends on — §13.2 measures that first epoch moving
validation CE from 0.1303 to 0.0935, so it is emphatically not stuck at its
initialisation. On electronics the floor is worth 22 MAPE points and a sevenfold increase in per-customer rank correlation, at 20 replications per
arm, on a benchmark whose architecture, inputs and windows are otherwise untouched. That
is a result about *applying* Valendin et al.'s model under a stricter evaluation protocol,
not about their model.

The thesis claim this supports is therefore, stated as narrowly as the evidence allows:
**on our electronics and multichannel panels, the weak per-customer discrimination
observed when the published LSTM architecture is applied under this package's temporal
split is substantially an artefact of how the model was trained here, and the training
protocol has to be re-derived when the validation split changes.**

Three limits on that sentence, all measured. It names *our* panels and *our* protocol, not
Valendin et al.'s results. It names two panels, because §15.1 found the floor doing nothing
on cdnow and actively worsening discrimination on gift — so "sparse retail panels" as a
class is not a thing this document may say. And the re-derivation claim is about the
*need* for one, not about the floor being the right answer: §13.3's principled
alternatives are all untested.

## 12. What this changes

1. **`min_epochs` earns its place on electronics, and must not become an unconditional
   default** — a floor of 50 with the existing search improves both models *on this
   panel*, and every model in the package except the benchmark has no published recipe to
   fall back on. **§15.1 and §15.2 then measured it on the other three and it does not
   transfer**: no detectable effect on cdnow, a *supported worsening* on gift (Δ Spearman
   −0.069, −0.118 to −0.027), and a threefold MAPE blow-up in the floored arm on cdnow's
   LSTM whose cause is not yet identified (E1). This item said "should become the default,
   not an opt-in" before those panels were run. The floor is also a patch: the real problem (§13.2) is an absolute 10⁻⁴ improvement threshold
   applied to a curve that gains 5.4×10⁻⁵ an epoch. A stopping criterion that suits
   that curve — a relative improvement threshold, or selection on the rollout
   (`docs/model-selection.md` §5) — would be the principled fix, and is untested.
2. **The published electronics rows are undertrained**, and by more than a footnote: MAPE
   70.8 against 46.8 on the same architecture and inputs. Whether family N is re-run under
   a floor is an ADR-level decision, taken in
   `.scratch/training-budget/issues/06-report-and-decide.md`.
3. ~~**The other three panels are untested.**~~ **Tested in §15**, and the answer is
   heterogeneity rather than generalisation: the floor helps decisively on electronics and
   multichannel, shows no clear difference on cdnow and hurts on gift. It is an electronics-and-
   multichannel story, and this document does not claim it is a property of sparse panels
   as a class.
4. **Adding batch 32 to the registry's search space is not the follow-up.** `paper` settles
   that: at patience 7 the search would still stop it early, and at a floor the batch size
   is not what is doing the work.

## 13. What to try next: keep the temporal split, fix what it feeds

§10 leaves an obvious but wrong move on the table — go back to the paper's customer-wise
split, since the stopping rule works there. This section argues against it, says what is
actually broken, and lists what to try instead. Nothing here has been run.

### 13.1 Is the temporal split the right one? Mostly yes, and the reason is narrower than "time series"

The blanket claim "time series must be split temporally" is too strong. For a purely
autoregressive model with uncorrelated errors, [Bergmeir, Hyndman and Koo
(2018)](https://robjhyndman.com/publications/cv-time-series/) show standard k-fold
cross-validation is valid and can beat out-of-sample evaluation, because what makes
random splitting unsafe is dependence between the held-out points and the training
points, not the calendar as such.

The argument for a temporal split here is more specific, and it survives:

- **The test task is temporal.** The holdout is a future year for customers the model has
  already seen. A validation split should resemble the test it stands in for, and out-of-time
  backtesting (rolling origin) is the standard for exactly this reason in forecasting practice.
- **A customer-wise split measures a different thing.** It scores the same calendar periods
  the model trained on, for customers it did not. That is generalisation across the cross-section,
  not across time — which is what ADR-0001 says, and §13.2 now shows numerically.
- **It is not a leak.** Both splits stay inside the calibration window, so neither touches the
  holdout. The customer-wise split is not *invalid*; it answers a question we are not asking.

On the second half of the question — **do other papers split by customer?** Valendin et al.'s
own reference notebook does (a random 10% of customers). Deep-learning forecasting over many
related series generally does not: global models are backtested on held-out time windows, which
is the convention this package follows. So the departure is from that paper, not from the field.

**Keep the temporal split.** What needs to change is what is computed on it.

### 13.2 Why the temporal curve is flat, measured

One training run per split on electronics, early stopping off, 120 epochs, the notebook's
recipe (`.scratch/training-budget/why_flat.py`):

| split | val CE at epoch 0 | at epoch 1 | best | at epoch | mean gain per epoch after epoch 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| temporal | 0.1303 | 0.0935 | 0.0893 | 79 | **5.4×10⁻⁵** |
| customer-wise | 0.2433 | 0.1623 | 0.1427 | 39 | **5.2×10⁻⁴** |

`fit_model` counts an epoch as an improvement only if it beats the best by **10⁻⁴**
(`loop.py`, `improved = (val_loss + 1e-4) < best_val_loss`). So under the temporal split the
average epoch improves by **half the threshold it must clear**, and under the customer-wise
split by five times it. That single comparison is the whole of §10's mechanism: nothing is
wrong with the curve, the rule simply cannot see it.

Decomposing the same runs by cell type — the temporal validation window is 43,108 cells of
which 589 (1.37%) carry a positive count:

| | epoch 0 | epoch 1 | epoch 79 |
| --- | ---: | ---: | ---: |
| CE on zero cells | 0.00039 | 0.0135 | 0.0122 |
| CE on positive cells | 9.51 | 5.87 | 5.66 |

The untrained model is almost perfect on silence and useless on purchases; the first epoch
trades one for the other, and everything after is slow refinement on the 1.4% of cells that
carry 86% of the loss. **The flatness is real, not an artefact — one-step-ahead cross-entropy
on future periods genuinely has little left to give.** Yet family T shows the forecast keeps
improving for another 200 epochs. So the quantity being watched is not the quantity that
matters, which is the same conclusion `docs/benchmarks-real-panels.md` reaches from the other
direction ("Optuna's validation loss cannot see which runs forecast badly").

### 13.3 The menu

**A. Change what is measured on the validation window** (rollout criterion, composite,
two-stage, multi-objective): tested and reported in `docs/model-selection.md` §3 and §5.

**B. Change the stopping rule so a flat curve does not read as convergence.**

5. **A relative improvement threshold.** 10⁻⁴ absolute is 0.11% of this panel's loss and
   0.4% of multichannel's — a threshold that means different things per panel. Express it as
   a fraction of the current best.
6. **Count patience in optimizer steps, not epochs.** An epoch is 4 batches at batch 256 on
   electronics and 26 at batch 32; patience 7 therefore means two very different amounts of
   training, which is what §5 shows the search exploiting.
7. **Stop on a smoothed curve.** Compare a k-epoch moving average rather than single epochs,
   so noise of the same size as the trend stops resetting the counter.
8. **Drop early stopping entirely.** A fixed step budget with a cosine-decayed learning rate,
   selecting the final weights — standard practice for deep models, and it removes the
   interaction rather than patching it. `min_epochs` (§12) is the timid version of this.
9. **Early-stop on the rollout metric** (A1 evaluated every k epochs). The most expensive
   option and the most aligned.

**C. Rolling-origin validation:** `docs/model-selection.md` §5 (S8).

**D. Use both splits for different jobs.** Hold out a block of customers *and* a time window:
early-stop on the customer-wise part, where the curve has slope, and select trials on the
temporal part, which matches the test. It is pragmatic and it is a hybrid — two criteria that
can disagree, and a model stopped on one may not be the one the other would pick. Listed for
completeness, below A and B in priority.

### 13.4 What to run first

The selection test it proposed was run: `docs/model-selection.md` §3.3.

### 13.5 What not to do

- **Do not switch back to the customer-wise split** to make the curve cooperate. It scores
  a different question (§13.1), and family T shows the temporal criterion is fixable.
- **Do not put MAPE, bias or Spearman in the training loss.** They are not proper scoring
  rules for a sampled categorical distribution; `models/losses.py` makes the argument for
  class weights and it applies identically here. Selection, yes; gradients, no.
- **Do not re-implement rollout metrics beside `compute_forecast_metrics`.** That is what
  cost ADR-0003 its credibility: its RMSE was 62x off the authority's, and only bias agreed.

## 14. The selection test

Moved to `docs/model-selection.md` §3.3–3.5.

## 15. Family U: crossing the two levers adds little, and one of them may be dangerous

§13 and §14 each moved one lever from the same floor and neither knew what the other was
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

### 15.2 Level: the floor is the lever, and on one panel it is a disaster

Aggregate MAPE, same cells:

Δ MAPE from the floored arm against `archive`, no cluster label, 95% percentile-bootstrap
CI (`effect()`, independent, n = 20 / 20; `results/effects_mape.csv`):

| panel | model | archive | floored | Δ | 95% CI | supported | refit noise |
| --- | --- | ---: | ---: | ---: | :---: | :---: | ---: |
| multichannel | LSTM | 140.9 | **52.8** | **−88.1** | −111.4 to −66.1 | yes | 7.71 |
| multichannel | ValendinLSTM | 84.9 | **56.0** | **−28.9** | −37.8 to −20.2 | yes | 7.71 |
| electronics | ValendinLSTM | 69.1 | **46.3** | **−22.8** | −28.1 to −17.4 | yes | 3.63 |
| electronics | LSTM | 56.9 | 51.7 | −5.2 | −12.3 to +2.5 | no | 3.63 |
| gift | ValendinLSTM | 32.2 | 28.9 | −3.3 | −6.9 to −0.1 | yes (below refit noise) | 5.89 |
| gift | LSTM | 30.3 | 29.9 | −0.5 | −4.0 to +3.3 | no | 5.89 |
| cdnow | ValendinLSTM | 51.7 | 56.5 | +4.8 | −19.8 to +30.8 | no | 5.83 |
| **cdnow** | **LSTM** | **57.7** | **183.9** | **+126.2** | +77.2 to +173.2 | yes (**worse**) | 5.83 |

Gift/ValendinLSTM is the case the refit-noise column exists for: the interval excludes zero, so
the effect is supported, but Δ = −3.3 sits under that panel's 5.89 refit noise. Supported
and small.

**Something in the floored arm triples CDNOW's LSTM error** (Δ +126.2, +77.2 to +173.2).
That is the single most important line in this document for anyone about to act on §12.

> **The cause is not identified, and an earlier draft of this paragraph claimed it was.**
> It read: "a 90-epoch floor on a 39-week calibration window overfits a model that then
> extrapolates catastrophically". That does not follow from this experiment. Family U's
> `floored` arm changed the floor **and** the batch size, the weight decay, the learning
> rate, the search and the epoch budget, so the blow-up could belong to any of them. What
> is established is that **the floored paper-recipe configuration catastrophically worsens
> CDNOW's LSTM**; which ingredient does it is to-do E1, which runs `floor50` — the archive
> recipe and search with `min_epochs` as the only change — on this panel.

The frozen benchmark on the same panel shows no clear difference (Δ +4.8, −19.8 to
+30.8, n = 20 — a wide interval, not a demonstration that it is unharmed), so whatever it
is appears with the developed model rather than with the panel alone. **With the cluster
label the same floor shows no clear MAPE difference on CDNOW's LSTM** (Δ +4.5, −27.1 to
+34.4; `factorial_analysis.py`), nor on any other cell with the label — so the blow-up
also depends on the input set, which E1 does not vary.

**So `min_epochs` must not become an unconditional default.** §12's first item is hereby
qualified: a floor pays where the panel is long and sparse and the model collapses
(electronics, multichannel), shows no clear MAPE difference or a supported-but-small one
where the model already works (gift: LSTM no clear difference, ValendinLSTM −3.3), and
does real damage on a short window (cdnow + LSTM). It has to be chosen per panel, or scaled to the
calibration length rather than fixed in epochs.

### 15.3 Why cross-entropy selects well on CDNOW and badly on electronics

Moved to `docs/model-selection.md` §3.4.

### 15.4 What follows

1. **Give the model a per-customer channel.** The cluster label is worth more than
   anything else measured here, and it brings the best cell to within about ±0.01 of
   Pareto/NBD's 20 seeded fits on both collapsed panels (§15.1) — while being, by construction, the benchmark's own summary fed back in. The
   honest reading is that the architecture cannot build that signal from counts alone,
   which is the case `docs/absorbing-death-state.md` makes for a learned survival
   variable rather than a borrowed one.
2. **Use a floor only where it pays, and never unconditionally** (§15.2).
3. **Stop selecting on validation cross-entropy where the panel collapses** — there it is
   wrong-signed — and keep it where the panel does not (§15.3).
4. **An apparent plateau, not an established ceiling.** Two levers applied together land
   within a few hundredths of where either reaches alone, under these configurations. That
   is an observation about what has been tried, not a proof that ~0.30 bounds the
   architecture on electronics; nothing here rules out a third lever doing better.

---

## To do

Raised by an external review of this document on 21 September 2026, plus what the review
prompted me to check in the code. Grouped by whether it needs an experiment, a
measurement, or only a correction. Corrections marked ✔ are applied; the register at the
top carries the resulting status of each claim.

### A. Experiments owed

| # | what | why it is owed | status |
| ---: | --- | --- | --- |
| E1 | CDNOW `archive` / `paper` / `floor50` — the floor as the **only** change against the control | §15.2 attributes CDNOW's MAPE blow-up to the epoch floor, but family U's `floored` arm also changed the batch size, the weight decay, the learning rate, the search and the epoch budget. The claim is **not identified** by that design. | running |
| E2 | Recompute `kmeans_8` from calibration periods **before** the validation window, re-run the electronics `archive/kmeans_8` cell, 20 replications | The label is read at the last calibration period (`cluster_features.py:97`) and broadcast to every calibration row (`panel_dataset.py:961`), so a model training on period 5 sees a summary of periods 1–104. Holdout scoring is clean — the statistic is available at the forecast origin — but validation-based selection and early stopping see a label that has seen the window they score. | to do |
| E3 | The Spearman refit noise | ✔ done — 0.0105 to 0.0159 depending on panel, now in "How claims are made" | done |
| E4 | A stopping criterion that suits a flat curve — relative threshold, smoothed curve, step budget, or no early stopping at all (§13.3 B5–B8) | The floor is a patch. §13.2 says the mechanism is an absolute 10⁻⁴ threshold against a curve gaining 5.4×10⁻⁵ an epoch; none of the principled alternatives has been tried. | to do |
| E5 | Whether the epoch floor should scale with calibration length rather than be fixed | `paper90` fixes 90 epochs whether the window is 39 weeks (cdnow) or 104 (electronics, gift, multichannel). If E1 confirms harm on cdnow, a fixed epoch count is the likely culprit. | blocked on E1 |

### B. Corrections applied

| # | correction | ✔ |
| ---: | --- | :---: |
| T1 | `min_delta=0` described our loop; ours uses an **absolute 10⁻⁴** (`loop.py`). `min_delta=0` is the notebook's Keras setting. §13.2's mechanism is the correct one and is now used throughout | ✔ |
| T2 | "stops at epoch 1" conflated best epoch with stop epoch, a distinction §1 sets up. The `paper` arm **keeps the checkpoint from epoch 1 and terminates after 7** | ✔ |
| T3 | "a customer-wise split leaks time" contradicted §13.1 and was wrong: both splits stay inside calibration and neither touches the holdout. It measures cross-sectional rather than temporal generalisation | ✔ |
| T4 | "they are substitutes" and "the ceiling is real" read an interval containing zero as equivalence. Now stated as no clear difference with its interval, and as an apparent plateau | ✔ |
| T5 | "matches Pareto/NBD" — the benchmark was a single fit with no interval. It is now 20 seeded fits per panel and compared with the independent bootstrap (§15.1); no equivalence is claimed | ✔ |
| T6 | "Batch 32 is not the fix; 90 epochs is" claimed an invariant the evidence does not identify — `floor50` reaches the same place with a few hundred updates where `paper90` needs ~2,300 | ✔ |
| T7 | "never leaves its initialisation" is contradicted by §13.2's own first epoch (CE 0.1303 → 0.0935) | ✔ |
| T8 | "the published electronics collapse" reads as a Valendin et al. result. It is **our** benchmark row | ✔ |
| T9 | The seeding protocol was inferable but unstated, and "independent replications" is what the bootstrap assumes | ✔ |
| T10 | Every effect is Δ with a 95% percentile-bootstrap CI, computed by one implementation (`panelclv.evaluation.effects`, via `.scratch/training-budget/effects.py`) | ✔ |

### C. Decisions deferred, and what unblocks them

| decision | blocked on |
| --- | --- |
| Whether `min_epochs` becomes a package default rather than an opt-in | E1, and E5 if E1 confirms harm |
| Whether family N's published rows are regenerated under a floor — an ADR-level change to `docs/benchmarks-real-panels.md` | E1 |
| Whether the selection criterion is replaced rather than floored | E4, and §14's finding that a validation rollout beats cross-entropy but lands on zero |
| Whether the cluster result can be central to the thesis | E2 |

### D. What this document still cannot say

- **Anything about panels as a class.** Four panels, each stated separately, and no
  measured characteristic yet predicts which of them collapse.
- **That better selection would help much.** The largest correlation in §14 is +0.202
  (the LSTM's rollout Spearman).
- **That ~0.30 bounds the architecture on electronics.** Two levers stopped there; that is
  an observation about two levers.
- **That the cluster label's lift is free of within-calibration hindsight.** E2 decides it.
