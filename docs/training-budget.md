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

**Tested, §9.** On electronics, 20 replications per arm: training the frozen benchmark for
the paper's own 90 epochs takes MAPE from 69.1 to 46.8 and per-customer Spearman from
0.027 to 0.177 (p ≈ 2×10⁻⁶ on both). Copying the paper's *settings* without the epochs
changes nothing. The published electronics collapse is substantially a training artefact.

**And §10 finds why, which is not what §4 assumed.** The paper's stopping rule works fine
under the paper's own customer-wise split on this same panel (28-56 epochs). It quits at
epoch 1 under the temporal split this package substitutes for it (ADR-0001). The defect is
in the combination — our split with their stopping rule — not in either alone, and §11
states what that does and does not let the thesis claim.

Everything below was measured on 2026-09-20: the archive numbers by reading `Studies/`,
the rest by re-running the frozen benchmark with the project venv on the ROCm
workstation. The scripts are in `.scratch/training-budget/` and are named at each step.

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
10. [Why the paper's rule stops at epoch 1 here: the split, not the panel](#10-why-the-papers-rule-stops-at-epoch-1-here-the-split-not-the-panel)
11. [What this is, relative to Valendin et al.](#11-what-this-is-relative-to-valendin-et-al)
12. [What this changes](#12-what-this-changes)

---

## 1. Early stopping, not the epoch budget, ends every run

`training/loop.py` breaks after `patience` consecutive non-improving epochs and restores
the best epoch's weights. So the number of epochs a trial ran is

```
epochs_run = best_epoch + 1 + patience      (capped at n_epochs)
```

and `best_epoch`, the value stored in every trial's `user_attrs`, is the epoch whose
weights were kept — not the epoch training ended at.

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
of ValendinLSTM runs there — and its section "The refit noise floor, and the stopping
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

**Read MAPE and Spearman here, not bias.** `docs/benchmarks-real-panels.md` measures a
refit noise floor on electronics of 8.9 points of sd — refitting one checkpoint twice
moves aggregate bias by that much with everything else held fixed. The bias difference
above (+31.0 → +9.9, five replications) sits inside that floor and is not a result. The
MAPE difference of 14 points and the sevenfold Spearman difference are outside it.

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

The last row is the notebook's own claim: *"This example takes about 100 epochs in total"*
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

| what changes | Spearman |
| --- | ---: |
| no per-customer feature, patience 7 (archived benchmark) | 0.03 |
| same inputs, trained to the validation optimum (§3) | 0.09 |
| a `kmeans_8` cluster label added, patience 7 | **0.29** |
| Pareto/NBD on the same panel | 0.30 |

Within the studies that carry no cluster label (n = 40), more training still helps —
Spearman correlates +0.33 with updates and +0.28 with best epoch — but it does not
approach what one persistent per-customer input buys. That ordering matches
`docs/insights-cluster-ablation.md` §5.1, and it is the honest summary: **the collapse is
mostly an input problem with a training-length component, not the other way round.**

On CDNOW, which never collapsed, training length has no relationship with ranking at all
(Spearman flat at ~0.40 across every quartile of best epoch) — even though CDNOW is the
panel leaving the most validation loss on the table. Whether its level metrics move is
untested.

*(`.scratch/training-budget/hparams.py`, `spearman_vs_epoch.py`)*

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

| arm | bias % | MAPE | Spearman | sd of predicted totals |
| --- | ---: | ---: | ---: | ---: |
| `archive` | +39.8 ± 20.4 | 69.1 | 0.027 ± 0.043 | 0.21 |
| `paper` | +28.7 ± 11.1 | 68.5 | 0.016 ± 0.029 | 0.19 |
| **`paper90`** | **−5.7 ± 27.0** | **46.8** | **0.177 ± 0.089** | **0.49** |
| `floor50` | +4.7 ± 29.3 | 47.9 | 0.168 ± 0.084 | 0.44 |

### LSTM, `no_ar-no_cluster-valendin`

| arm | bias % | MAPE | Spearman | sd of predicted totals |
| --- | ---: | ---: | ---: | ---: |
| `archive` | +30.4 ± 16.7 | 59.3 | 0.030 ± 0.044 | 0.19 |
| `paper` | +34.5 ± 15.7 | 63.5 | 0.020 ± 0.029 | 0.19 |
| `paper90` | +31.1 ± 31.7 | 54.7 | 0.174 ± 0.075 | 0.56 |
| **`floor50`** | **+0.3 ± 18.2** | **50.1** | 0.074 ± 0.049 | 0.19 |

Each arm against `archive`, Mann-Whitney, 20 vs 20:

| model | arm | MAPE | p | Spearman | p |
| --- | --- | ---: | ---: | ---: | ---: |
| ValendinLSTM | `paper` | 68.5 | 0.76 | 0.016 | 0.70 |
| | `paper90` | 46.8 | **2×10⁻⁶** | 0.177 | **5×10⁻⁶** |
| | `floor50` | 47.9 | **1×10⁻⁶** | 0.168 | **1×10⁻⁵** |
| LSTM | `paper` | 63.5 | 0.15 | 0.020 | 0.20 |
| | `paper90` | 54.7 | 0.10 | 0.174 | **2×10⁻⁶** |
| | `floor50` | 50.1 | **1×10⁻⁴** | 0.074 | **2×10⁻³** |

**Three things follow, and the first is the one to remember.**

**The settings were never the point; the epochs were.** `paper` — the notebook's optimizer,
batch size and patience, pinned exactly — is indistinguishable from `archive` on every
metric (p = 0.15 to 0.76), because it stops at epoch 1: copying the recipe copies its
stopping rule, and that rule quits immediately here (§10 shows why). Add the floor and
the same recipe cuts MAPE by 22 points and multiplies the ranking correlation by seven.
**Batch 32 is not the fix; 90 epochs is.**

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
(`docs/insights-cluster-ablation.md` §5.1) and Pareto/NBD 0.297, against 0.177 here; §6's
ordering holds, with training length worth more than it looked at five replications (0.09)
and still less than one persistent per-customer channel.

RMSE separates nothing, as everywhere on these panels: every arm sits between 0.3763 and
0.3774 against the all-zero forecast's 0.3775.

Bias moves a lot and means less: `paper90`'s −5.7 ± 27.0 is a better centre than
`archive`'s +39.8 ± 20.4, but the across-replication spread grows and the refit noise floor
on this panel is 8.9 points of sd (§3). Read MAPE and Spearman.

## 10. Why the paper's rule stops at epoch 1 here: the split, not the panel

`paper` stopping at epoch 1 has two possible causes, and they point in opposite
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
slice of customers does, and `min_delta=0` patience reads that flatness as convergence.
This package departs from the paper's split deliberately and documents why — a customer-wise
split leaks time, and the thing being forecast is the future — but nothing checked what
that departure did to the stopping rule bolted on beside it. It quietly cost most of the
training.

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
honest validation split requires a training floor — or a different stopping criterion —
or the model never leaves its initialisation. On electronics that floor is worth 22 MAPE
points and a sevenfold increase in per-customer rank correlation, at 20 replications per
arm, on a benchmark whose architecture, inputs and windows are otherwise untouched. That
is a result about *applying* Valendin et al.'s model under a stricter evaluation protocol,
not about their model.

The thesis claim this supports is therefore: **the published LSTM's weak per-customer
discrimination on sparse retail panels is substantially an artefact of how it was trained
here, and the training protocol has to be re-derived when the validation split changes.**
It is one panel; §12 says what is still owed.

## 12. What this changes

1. **`min_epochs` should become the default, not an opt-in** — a floor of 50 with the
   existing search is a strict improvement on both models here, and every model in the
   package except the benchmark has no published recipe to fall back on. The floor is a
   patch, though: §10 says the real problem is that a patience rule with `min_delta=0`
   reads a flat temporal-validation curve as convergence. A stopping criterion that suits
   that curve — a relative `min_delta`, or selection on the rollout
   (`docs/insights-study.md` §5.4) — would be the principled fix, and is untested.
2. **The published electronics rows are undertrained**, and by more than a footnote: MAPE
   70.8 against 46.8 on the same architecture and inputs. Whether family N is re-run under
   a floor is an ADR-level decision, taken in
   `.scratch/training-budget/issues/06-report-and-decide.md`.
3. **The other three panels are untested.** CDNOW leaves the most validation loss on the
   table (10.7-13.0%, §2) and is the panel whose ranking never collapsed, so it is the one
   that says whether this generalises or is an electronics story.
4. **Adding batch 32 to the registry's search space is not the follow-up.** `paper` settles
   that: at patience 7 the search would still stop it early, and at a floor the batch size
   is not what is doing the work.

