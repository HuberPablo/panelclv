# Training efficiency: the models stop training too early

Every neural result in this package comes from a model that stopped training while its
validation loss was still falling. This doc covers:
- how that was established;
- why it happens;
- what was tested to fix it, and what it changed;
- what is left to try;
- which runs failed worst and best, and what they share (§8).

How the stopped models are then *selected* is in `docs/model-selection.md`.

Every Δ is a difference of condition means with a 95% percentile-bootstrap interval from
`evaluation.effects.effect` (`docs/statistical-protocol.md`). The replications are
independent studies. Bold marks an interval that excludes 0. Refit-noise magnitudes are
in `docs/model-selection.md` §2.

## Claims

| # | Claim | Panel | Status | § |
| --- | --- | --- | --- | --- |
| 1 | Patience, not the epoch budget, ends every archived run | all four | established | 2 |
| 2 | With early stopping off, validation CE keeps falling to epoch 88–247 | cdnow, electronics, multichannel | established (3 runs each) | 3.2 |
| 3 | The stop is caused by our temporal split combined with an absolute 1e-4 threshold, not by the panel | electronics | established | 3.3–3.4 |
| 4 | Under patience 7 the search prefers the batch size that takes the fewest updates | electronics | established, descriptive | 3.5 |
| 5 | The notebook's recipe with a 90-epoch floor (`paper90`) improves level and ranking | electronics | **established** (ValendinLSTM MAPE −22.3, Spearman +0.150) | 5.1 |
| 6 | The notebook's settings without a floor (`paper`) change nothing | electronics | no clear difference | 5.1 |
| 7 | A training floor improves ranking | electronics, multichannel | **established** | 5.2 |
| 8 | …and worsens it | gift | **established** (−0.069) | 5.2 |
| 9 | The floored paper recipe triples CDNOW's LSTM error | cdnow | **established** (+126.2) | 5.2 |
| 10 | …and the floor alone is not the cause | cdnow | floor alone: no clear difference (E1) | 5.3 |
| 11 | On a 260-week window a floor tightens the spread but does not move the mean | electronic_5y | no clear difference in means | 5.4 |
| 12 | The worst level errors are over-forecasts by collapsed forecasts (low CV) | electronics, multichannel, cdnow | descriptive (43 of 46 cells) | 8.2 |
| 13 | Collapse is undertraining on electronics and multichannel; on cdnow and gift it comes with the floored no-label recipe, at many updates | all four | descriptive | 8.2 |
| 14 | Validation CE does not separate the worst runs from the best within a cell | all four | descriptive | 8.2 |

## 1. How training stops

`fit_model` (`training/loop.py`) counts an epoch as an improvement only if
`val_loss + 1e-4 < best_val_loss`. It stops after `patience` (7) epochs without one and
restores the best epoch's weights.
- `best_epoch` is stored per trial: it is the epoch whose weights were kept.
- The number of epochs actually run is not stored. With no floor and a completed trial
  it is `best_epoch + 1 + patience`.
- How much a kept model trained is reported in two currencies, always with its batch
  size: **epochs**, `best_epoch + 1`, the passes over the N training customers, and
  **gradient updates**, `ceil(N / batch_size) × (best_epoch + 1)`, the optimiser steps.
  Neither alone is comparable across batch sizes (§1.1).

### 1.1 Why updates alone do not measure training

One update at batch 256 is not worth one update at batch 32. The larger batch averages
the gradient over eight times as many customers, so each step points more reliably
downhill, but it costs eight times the data. The literature describes the trade-off
with a **critical batch size**, set by how noisy the gradient is (McCandlish et al.,
2018). Well below it, doubling the batch roughly halves the updates needed and leaves
the data needed unchanged, so epochs are the fair currency. Well above it, a larger batch
barely cuts the updates needed, so updates are. In between, both count, and where the
switch happens depends on the model, the data and the optimiser (Shallue et al., 2019).
The learning rate is part of the same quantity: SGD's gradient noise scales roughly as
lr × N / batch (Smith et al., 2018), large-batch SGD needs its learning rate scaled up
linearly with the batch (Goyal et al., 2017), and Adam by about its square root (Malladi
et al., 2022). The large-batch "generalisation gap" largely closes once large batches
get as many updates as small ones (Hoffer et al., 2017). That is the case for counting
updates, under those scaling rules, not a licence to compare raw counts across settings.

Our searches vary batch, learning rate and weight decay jointly, under AdamW, so an
update count compared across trials mixes all three. Batch 256 on electronics' 829
customers takes four steps per epoch and is close to full-batch descent, where updates
are likely the binding currency. But a trial with few updates at batch 256 can still have
seen *more* data than one with more updates at batch 64 (§3.5). The doc therefore reports
batch, epochs and updates side by side, and does not rest an undertraining claim on an
update count. The direct evidence is §3.2: with early stopping off, validation loss keeps
falling long after the kept epoch.

References: Goyal et al. (2017), *Accurate, Large Minibatch SGD*, arXiv:1706.02677.
Hoffer, Hubara & Soudry (2017), *Train Longer, Generalize Better*, NeurIPS. McCandlish,
Kaplan, Amodei et al. (2018), *An Empirical Model of Large-Batch Training*,
arXiv:1812.06162. Smith, Kindermans, Ying & Le (2018), *Don't Decay the Learning Rate,
Increase the Batch Size*, ICLR. Shallue et al. (2019), *Measuring the Effects of Data
Parallelism on Neural Network Training*, JMLR. Malladi, Lyu, Panigrahi & Arora (2022), *On
the SDEs and Scaling Rules for Adaptive Gradient Algorithms*, NeurIPS.

## 2. Patience ends every run

Across every archived study on the four real panels (407,950 trials, 163,363 completed;
patience 7, `n_epochs` 100):

| panel | model | median best epoch | share reaching 100 epochs |
| --- | --- | ---: | ---: |
| cdnow | LSTM / ValendinLSTM / Transformer | 10 / 14 / 16 | 0.8% / 0.1% / 3.8% |
| electronics | LSTM / ValendinLSTM / Transformer | 8 / 7 / 15 | 0.8% / 0.3% / 5.2% |
| gift | LSTM | 13 | 0.4% |
| multichannel | LSTM / ValendinLSTM | 8 / 4 | 0.5% / 0.0% |

The budget of 100 epochs never binds; the 7 does. On multichannel the best trial stopped at
epoch ≤ 3 in 55% of ValendinLSTM runs, and earlier stopping goes with worse ranking
(rank correlation 0.59 between best epoch and Spearman in the log arm; descriptive).
*(`.scratch/training-budget/epochs.py`)*

## 3. How we know it undertrains, and why

### 3.1 The reference notebook: its recipe, and an epoch count that does not transfer

The "Valendin et al. recipe" this doc refers to is the training setup of the demo notebook
published with the paper (`Original_paper_model/banking_transactions_demo.ipynb`). It is
not something the paper itself specifies:

| | Valendin et al. demo notebook | Our studies |
| --- | --- | --- |
| data | Czech banking transactions (data.world): 2,239 accounts, calibration 1993–1995, a random 10% of customers held out for validation | our four panels, temporal validation split (ADR-0001) |
| optimizer | Adam defaults: lr 1e-3, no weight decay | AdamW; lr searched 1e-4–3e-3, weight decay 1e-6–1e-2 |
| batch size | 32 | searched over {64, 128, 256} |
| patience / budget | 5 / 150 epochs | 7 / 100 epochs |
| epochs trained | one logged run: stopped at epoch 90, kept epoch 85 | median best epoch 7–14 (§2) |

**Where "~90 epochs" comes from.** It comes from the notebook only. Its saved log (one run,
16 March 2022) ends `Epoch 90: early stopping … best epoch: 85`. Its markdown says the
validation loss "should end up around 0.44 after ±90 epochs", and a code comment says
"about 100 epochs in total". **The paper gives no epoch count.** §2.2 trains "until we
reach a minimum" of the validation error. Appendix B.2 early-stops, restores the best
epoch, then runs "several fine-tuning epochs" on the full calibration set at a large batch
and reduced learning rate. Appendix B.1 says learning rate and batch size were *searched*
(a random walk over architectures and settings), so the paper does not fix those either.

**What the figure cannot say.** 90 is how long one run of the recipe trained on the
banking data. How many epochs a model needs depends on the panel: its size, its sparsity,
its calibration length and the validation split. So 90 is not an expected training length
for any of our panels, and comparing it with our epoch counts says nothing about whether
our models undertrain.

**What does compare.** §3.4 runs the same recipe on electronics under the notebook's own
split. It keeps epoch 28–56 (3 replications), i.e. 29–57 epochs and about 750–1,500 updates
at batch 32. Our electronics benchmark winners kept epoch 7 at batch 256, i.e. 8 epochs
and about 32 updates. That is 4–7× fewer epochs on the same panel. The update gap is
wider because our batch is eight times the notebook's, and updates alone do not compare
across batch sizes (§1.1). The direct evidence that our models stop too early does not
rest on this comparison at all. It is §3.2: with early stopping off, the validation loss
keeps falling long after the kept epoch.

`scripts/validate_valendin_lstm.py` runs the notebook's recipe on the notebook's banking
data and reproduces its validation loss of about 0.44, which checks our implementation of
the recipe. Our search space cannot express that recipe: batch 32 is absent and weight
decay has no zero.

### 3.2 The validation loss keeps falling after patience 7 stops

**What was measured.** The validation cross-entropy of the frozen ValendinLSTM, epoch by
epoch, with early stopping switched off for 200 epochs. This is the teacher-forced loss
on the validation weeks of our temporal split, the same number the stopping rule watches.
No forecast was made.

**How.** 3 runs on each of electronics, multichannel and CDNOW, all at one fixed setting:
the hyperparameters one archived electronics study selected (lr 1.15e-3, weight decay
5.7e-4, batch 256). Multichannel and CDNOW run that electronics setting too, not one of
their own. Afterwards, the patience-7 rule is replayed on each recorded curve: "patience 7
stops at" is where it would have stopped, and "improvement forgone" is how far the
200-epoch minimum lies below the best loss patience 7 had seen.

| panel | patience 7 stops at | its best val CE | best over 200 epochs | at epoch | improvement forgone |
| --- | ---: | ---: | ---: | ---: | ---: |
| electronics | 19 | 0.0888–0.0896 | 0.0848–0.0858 | 137–197 | 3.4–5.3% |
| multichannel | 12–16 | 0.0229–0.0230 | 0.0220–0.0226 | 169–181 | 1.4–4.4% |
| cdnow | 11 | 0.1078–0.1098 | 0.0946–0.0964 | 88–120 | 10.7–13.0% |

**What it shows.** The curve is not a smooth descent. It stays flat for a long stretch,
then drops below where patience 7 gave up. Before reaching its minimum, each run went
29–131 epochs without one counted improvement, so no fixed patience can be trusted to
wait it out. §3.3 explains why the curve looks flat.

**What it does not show.**
- **That the forecast improves.** A lower validation loss is not a better holdout
  forecast. Across trials, validation loss does not predict the forecast
  (`docs/model-selection.md`; §8.2, pattern 8). The forecast evidence for longer training
  is §5, at 20 replications per arm.
- **That it holds generally.** One model, one hyperparameter setting borrowed from
  electronics, 3 runs per panel.

**Provenance.** The numbers are from the run recorded in commit `d6cea22` (20 Sep).
`probe_patience.py` writes `results/patience_probe.csv`, which is no longer on disk, so
they cannot be re-checked from the repo. A companion script, `end_to_end.py`, forecast the
holdout on electronics with patience 7 against patience 40 (5 runs per arm). Under patience
40, one of five runs got past the plateau to the later minimum; the other four still kept
weights from epochs 6–11. Its output is lost too, and its old write-up describes its long
arm in two inconsistent ways, so its forecast numbers are not used here.
*(`.scratch/training-budget/probe_patience.py`, `end_to_end.py`)*

### 3.3 Why the curve looks flat: each epoch gains less than the rule requires

The stopping rule counts an epoch as progress only if the validation loss drops by more
than 10⁻⁴ (§1). Under our temporal split, the loss drops by less than that in a typical
epoch. The rule therefore sees no progress and stops, even though the loss is still going
down.

The measurement: electronics, the notebook's recipe, early stopping off, 120 epochs, one
run under each split.

| split | val loss after epoch 1 | best val loss (epoch 61) | average drop per epoch | vs the 10⁻⁴ threshold |
| --- | ---: | ---: | ---: | ---: |
| temporal (ours, ADR-0001) | 0.0935 | 0.0887 | 8.1×10⁻⁵ | 0.8× |
| customer-wise (the notebook's) | 0.1623 | 0.1419 | 3.4×10⁻⁴ | 3.4× |

Under the notebook's split a typical epoch clears the threshold easily, so training
continues (§3.4). Under ours it falls short.

**Why the drop is so small under our split.** The validation loss is an average over
every customer-week in the validation window. For each week, the model assigns a
probability to every possible purchase count. The loss for that week is −log of the
probability given to the count that actually happened, so a confident wrong prediction
costs a lot. Under our split, 98.6% of the customer-weeks have no purchase.

The first two epochs show what this does. (Epochs are numbered from 0, so epoch 0 is the
end of the first pass over the training data.)

| after | weeks with no purchase | weeks with a purchase | average over all weeks |
| --- | --- | --- | ---: |
| epoch 0 | loss 0.0004 (P(no purchase) ≈ 99.96%) | loss 9.51 (P(true count) ≈ 0.007%) | 0.1303 |
| epoch 1 | loss 0.0135 (P(no purchase) ≈ 98.7%) | loss 5.87 (P(true count) ≈ 0.28%) | 0.0935 |

- **After epoch 0** the model predicts that almost nobody buys. That is nearly right for
  the silent weeks and badly wrong for the weeks with a purchase.
- **After epoch 1** it moves a little probability toward purchases. The silent weeks get
  slightly worse, the purchase weeks get about 40 times better, and the average loss falls
  by 28%. This is the easy gain, and it is real learning, not a model stuck where it
  started.
- **After that,** most of the remaining loss sits in the purchase weeks: 1.4% of the weeks
  carry 74–86% of the loss for the rest of the run. Lowering it means predicting which
  customer buys in which week, which is hard. The average improves slowly and unevenly,
  from 0.0935 to 0.0887 over the next 60 epochs, which is the 8.1×10⁻⁵ per epoch above.

This run only tracks the validation loss. It makes no forecast, so it cannot say whether
the forecast keeps improving over those epochs; §5 tests that.

*(`.scratch/training-budget/why_flat.py` → `results/why_flat.csv`. An earlier run
reported slopes of 5.4×10⁻⁵ and 5.2×10⁻⁴, but its output was overwritten.)*

### 3.4 It is the split, not the panel

The notebook's recipe, under the notebook's own split (a random 10% of customers) on
electronics:

| replication | epochs run | best epoch |
| --- | ---: | ---: |
| 0 | 62 | 56 |
| 1 | 39 | 33 |
| 2 | 34 | 28 |

So it trains 28–56 epochs, but under our temporal split it keeps epoch 1. The two
validation losses score different rows, so only the epoch counts compare. The temporal
split is still right for a forecasting evaluation: it is not a leak, and it matches the
test task (ADR-0001). What went unchecked is what it does to the stopping rule.
*(`.scratch/training-budget/paper_split_check.py`, `why_flat.py`)*

### 3.5 The search rewards the setting that trains least

In 20 electronics benchmark studies (2,000 trials, 713 completed), grouped by batch size:

| batch | completed | best val CE | median best epoch | median epochs | median updates |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 136 | 0.0899 | 3 | 4 | 52 |
| 128 | 178 | 0.0911 | 4 | 5 | 35 |
| 256 | 399 | 0.0886 | 7 | 8 | 32 |

The two currencies rank the batches in opposite orders. Batch 256 takes the fewest steps
but, by epochs, sees the most data. "Trains least" therefore holds in updates only, and
under patience 7 every batch size trains little in both: at most 8 epochs, against 29–57
for the notebook's recipe on the same panel under its own split (§3.4).

Within the 40 pre-experiment electronics studies with no label, more training goes with
better ranking: Spearman's rank correlation with updates received is **+0.328** (+0.024,
+0.580; best epoch, i.e. epochs, +0.280, not supported). It is one across-study
correlation, resampled by study, and batch, learning rate and weight decay vary with it,
so it does not say which currency matters (§1.1).

Batch 256 won all 20 studies. Small-batch trials need more epochs to show their
advantage, and patience 7 stops them first. Inside a study, the longest-trained trial is
the best pick of any criterion on electronics (`docs/model-selection.md` §3.5).
*(`.scratch/training-budget/trials_by_bs.py`)*

## 4. What was tested

Four experiments, run in this order, each to answer a question the previous one left
open. The family letters are the ones in `docs/studies-run.md` §4, which holds each
family's full specification. All four keep everything else fixed: our temporal split
(ADR-0001), the ADR-0008 refit, cross-entropy, and the two models' architectures and
inputs. Only the training recipe changes.

### 4.1 The arms, and what "paper" means

"Paper" always means **the training recipe of Valendin et al.'s reference notebook**
(`Original_paper_model/banking_transactions_demo.ipynb`, the code published with the
paper), copied setting by setting. It does **not** mean the paper's data, its split or its
published numbers. `scripts/validate_valendin_lstm.py` checks that this recipe, run under
the notebook's own split, reproduces the notebook's validation loss (§3.1).

| arm | what it copies | settings | trials |
| --- | --- | --- | ---: |
| `archive` | our own recipe, the one behind every archived result (family N, `docs/benchmarks.md`). It is the control. | lr, weight decay and batch searched over the registry's ranges; patience 7; 100 epochs | 100 |
| `paper` | the notebook's recipe, literally | Adam defaults (lr 1e-3, no weight decay); batch 32; patience 5; 150 epochs; LSTM 128/128, dropout 0 | 1 |
| `paper90` | the notebook's recipe plus a 90-epoch floor | `paper` + `min_epochs=90` | 1 |
| `floored` | family U's name for `paper90` | identical to `paper90` | 1 |
| `floor50` | nothing from the paper: the floor alone, added to our recipe | `archive` + `min_epochs=50`, 300 epochs | 100 |

**Why 90.** It is the length of the notebook's one logged run, on its banking data (§3.1).
It is not a training length the paper reports, which it never does, and it is not what
the recipe needs on our panels: on electronics, under the notebook's own split, the same
recipe keeps epoch 28–56 (§3.4). Read `paper90` as "the notebook's recipe with a 90-epoch
floor", a long floor that rules out an early stop, not as "training the way the paper
trained".

A **floor** (`min_epochs`) stops early stopping from firing before that epoch. The best
epoch is still chosen by validation loss, so the kept weights can come from before the
floor (§8.2, pattern 10). The pinned arms run one trial because pinning removes
hyperparameter selection, so a difference against `archive` comes from training and not from
which trial won. Selection does not predict the forecast anyway (`docs/model-selection.md`).

### 4.2 The four experiments

**Family T** (`scripts/run_training_budget.py`, 20 Sep; electronics; ValendinLSTM and LSTM ×
`archive` / `paper` / `paper90` / `floor50` × 20 replications, 200 paths).
*Why:* electronics' benchmark forecast is collapsed (MAPE 70.8, Spearman 0.032), and its
winners kept epoch 7 at batch 256, while the notebook's recipe on the same panel, under
its own split, keeps epoch 28–56 (§3.4). The question is whether the notebook's recipe,
trained long enough, removes the collapse. The arms separate three
things. `paper` copies the settings. `paper90` adds a 90-epoch floor to
them. `floor50` adds a 50-epoch floor to our own search, which is the only
option for a model with no published recipe, such as our Transformer. Pre-launch probes
had shown that `paper` alone keeps epoch 1 under our split, which is why `paper90` exists.
Reported in §5.1.

**Family U** (`scripts/run_factorial.py`, 21 Sep; all four panels; ValendinLSTM and LSTM ×
{`archive`, `floored`} × {`no_cluster`, `kmeans_8`} × 20 replications).
*Why:* two levers had each lifted electronics' Spearman separately. Training (family T)
took it from 0.027 to 0.177, and a cluster label (family F) took it from 0.039 to 0.27.
Neither experiment knew about the other. Family U crosses them to see whether they add, and
runs on all four panels to see whether family T's electronics result holds elsewhere. Only
the training half is reported here (§5.2). The label half is in
`docs/insights-real-panels.md` §5.2. Family U's LSTM carries no year index, unlike family
T's, so its electronics LSTM cells are not the same configuration as family T's.

**Family T′ = experiment E1** (`scripts/run_training_budget.py` on CDNOW, 21 Sep;
ValendinLSTM and LSTM × `archive` / `floor50` / `paper` × 20 replications;
analysed by `.scratch/model-selection/e1_cdnow.py`).
*Why:* family U's `floored` arm tripled CDNOW's LSTM error (§5.2, claim 9). But `floored`
changes the floor, batch, learning rate, weight decay, search and epoch budget all at once.
E1 runs the same family T runner on CDNOW so that each arm changes one ingredient against
`archive`. `floor50` changes only the floor, and `paper` only the settings. "T′" (the
family name in `docs/studies-run.md` and §8) and "E1" (its to-do name in
`docs/model-selection.md`) are the same 120 suites. Reported in §5.3.

**Family X, the epoch floor on electronic_5y** (`scripts/run_epoch_floor_5y.py`, 26–28 Sep;
ValendinLSTM, LSTM + `ar_bounded_52` and LSTM + `kmeans_8` × `nofloor` / `from20` /
`from30` × 20 replications, 500 paths).
*Why:* this is the only test on the paper's own electronics split (260 calibration weeks,
`docs/datasets.md`). There the ValendinLSTM benchmark (family W) spread its bias from
−7.9% to +37.4% at identical validation loss, and a later kept epoch went with lower bias.
X pins the least-biased study's settings and varies only which epochs may be kept.
`nofloor` allows any epoch, while `from20` and `from30` allow only epoch 20 or 30 onwards
(`select_from_epoch`). Unlike `min_epochs`, this forces late weights. Reported in §5.4.

**E4 and E5** are not experiments that ran. They are the to-do names (`docs/studies-run.md`
§6) for a stopping rule suited to the flat temporal curve and for a floor scaled to the
calibration length (§7).

## 5. Results

### 5.1 Electronics: it is the epochs, not the settings (family T)

Δ against `archive`, n = 20 / 20:

| model | arm | Δ MAPE | Δ Spearman | Δ \|bias\| |
| --- | --- | ---: | ---: | ---: |
| ValendinLSTM | `paper` | −0.6 (−6.3, +5.0) | −0.011 (−0.034, +0.011) | **−11.8** (−21.2, −2.3) |
| | `paper90` | **−22.3** (−28.3, −16.2) | **+0.150** (+0.107, +0.191) | **−16.4** (−26.0, −6.6) |
| | `floor50` | **−21.2** (−26.8, −15.5) | **+0.141** (+0.100, +0.180) | **−14.5** (−24.3, −4.5) |
| LSTM | `paper` | +4.3 (−0.2, +9.0) | −0.011 (−0.034, +0.011) | +4.0 (−5.2, +13.6) |
| | `paper90` | −4.5 (−12.9, +4.6) | **+0.144** (+0.106, +0.180) | +3.7 (−9.8, +17.8) |
| | `floor50` | **−9.2** (−13.3, −4.5) | **+0.043** (+0.015, +0.070) | **−17.6** (−25.9, −8.8) |

Means: ValendinLSTM `archive` MAPE 69.1, Spearman 0.027; `paper90` 46.8, 0.177.

- **Copying the paper's settings changes nothing**, because the recipe's own stopping rule
  keeps epoch 1 and stops after 7 (§3.3). Adding the 90-epoch floor cuts MAPE by 22 points
  and multiplies ranking by about seven.
- **The invariant is not identified.** `paper90`'s kept weights come from a median epoch of
  56 (ValendinLSTM) and 71.5 (LSTM) at batch 32, about 1,500–1,900 updates. `floor50`'s come
  from epoch 67 and 26 at batch 256, about 330 and 110 updates. The two also differ in learning rate
  and weight decay, so neither updates nor epochs is shown to be what matters (§1.1).
- **A floor works with the search intact.** Under a floor, the searched `floor50` and the
  pinned `paper90` show no clear MAPE difference on either model. On Spearman the
  searched LSTM ranks worse (−0.100, −0.138 to −0.063).

### 5.2 Four panels: the floor helps two, hurts one, and blows one up (family U)

ValendinLSTM Spearman, `floored` against `archive`, no label (n = 20 / 20):

| panel | Δ Spearman |
| --- | ---: |
| electronics | **+0.157** (+0.119, +0.193) |
| multichannel | **+0.123** (+0.099, +0.147) |
| cdnow | +0.019 (−0.019, +0.065) |
| gift | **−0.069** (−0.118, −0.027) |

The LSTM agrees in direction on the three supported panels (electronics **+0.153**,
multichannel **+0.076**, gift **−0.063**); on cdnow it shows no clear difference (−0.034).

Aggregate MAPE, `floored` against `archive`, no label:

| panel | model | archive | floored | Δ |
| --- | --- | ---: | ---: | ---: |
| multichannel | LSTM | 140.9 | 52.8 | **−88.1** (−111.4, −66.1) |
| multichannel | ValendinLSTM | 84.9 | 56.0 | **−28.9** (−37.8, −20.2) |
| electronics | ValendinLSTM | 69.1 | 46.3 | **−22.8** (−28.1, −17.4) |
| electronics | LSTM | 56.9 | 51.7 | −5.2 (−12.3, +2.5) |
| gift | ValendinLSTM | 32.2 | 28.9 | **−3.3** (−6.9, −0.1) |
| gift | LSTM | 30.3 | 29.9 | −0.5 (−4.0, +3.3) |
| cdnow | ValendinLSTM | 51.7 | 56.5 | +4.8 (−19.8, +30.8) |
| cdnow | LSTM | 57.7 | 183.9 | **+126.2** (+77.2, +173.2) |

The gains sit on the two panels where the count-only model collapses, i.e. gives every
customer nearly the same forecast (electronics, multichannel; forecast CV 0.08–0.15 under
`archive`). With the cluster label present, the floor shows no clear MAPE difference on
CDNOW's LSTM (+4.5, −27.1 to +34.4), so the blow-up also depends on the inputs.

### 5.3 CDNOW: the floor alone is not what blows up (E1)

`floored` changed the floor, batch size, weight decay, learning rate, search and epoch
budget all at once. E1 changes one ingredient at a time (MAPE, n = 20 / 20):

| model | `floor50` vs `archive` | `paper` vs `archive` |
| --- | ---: | ---: |
| LSTM | +3.5 (−0.4, +8.2) | **+3.6** (+0.2, +7.0) |
| ValendinLSTM | −3.7 (−20.8, +9.7) | +2.5 (−15.0, +16.4) |

Neither comes near a tripling. But E1's LSTM `archive` scores MAPE 21.7 against family U's
57.7. Family U's LSTM carries no year index, so the two are not the same setup, and E1
has no 90-epoch arm. **The tripling is still unexplained.** The likeliest remaining
suspect is a 90-epoch floor on a 39-week window, untested.

### 5.4 Electronic_5y: tighter, not better on average

On the paper's own split (260 calibration weeks), with pinned settings, 20 studies each:

| model | rule | bias % (mean ± sd) | MAPE (mean ± sd) |
| --- | --- | ---: | ---: |
| LSTM (ValendinLSTM shape) | `nofloor` | +13.2 ± 16.7 | 21.5 ± 11.7 |
| | `from20` | +10.2 ± 7.4 | 17.8 ± 3.2 |
| | `from30` | +12.7 ± 7.3 | 19.4 ± 3.4 |

- `from20` against `nofloor`: MAPE −3.7 (−9.6, +0.5), no clear difference. `from30`
  against `from20`: MAPE +1.5 (−0.5, +3.5).
- What the floor visibly changes is the spread: the bias sd halves and the worst study
  moves from +66.7% to +23.9%.
- Every cell still over-forecasts by about 10% (`from20` mean bias +10.2, +7.1 to +13.2),
  against the published +2.7%.

The flags/label and attention cells of the same experiment are
in `docs/insights-real-panels.md` §7.

## 6. What this means relative to Valendin et al.

- **Not a correction of the paper.** Their protocol run as published (their split, their
  recipe) trains a sensible number of epochs on our panels too (§3.4).
- **A correction of our reproduction.** ADR-0001's temporal split is the one deliberate
  departure, and the paper's stopping rule does not survive it. Every archived neural
  result trained under that combination. The electronics and multichannel benchmark rows
  in `docs/benchmarks.md` are lower bounds on the architecture, not
  measurements of it. CDNOW and gift are not shown to be affected.
- **The narrow thesis claim.** On our electronics and multichannel panels, the weak
  per-customer discrimination of the published LSTM under a temporal split is
  substantially an artefact of training, and the training protocol has to be
  re-derived when the validation split changes.

## 7. What can be done

| Option | Status |
| --- | --- |
| Epoch floor (`min_epochs`) | Tested (§5). Pays on electronics and multichannel; not on cdnow; harmful on gift. **Never an unconditional default.** |
| The notebook's recipe | Tested (§5.1). Only works with a floor added. |
| Adding batch 32 to the search space | Not the fix: at patience 7 it would still stop early. |
| Relative improvement threshold (a fraction of the current best) | Untested (E4). Addresses §3.3 directly: 10⁻⁴ is 0.11% of electronics' loss and 0.4% of multichannel's. |
| Patience counted in optimiser steps, not epochs | Untested (E4). 4 batches per epoch at batch 256 against 26 at batch 32. |
| Stop on a smoothed curve | Untested (E4). |
| Fixed step budget with cosine decay, no early stopping | Untested (E4). Removes the interaction instead of patching it. |
| Floor scaled to calibration length | Untested (E5). The prime suspect for CDNOW (§5.3). |
| Early-stop on a validation rollout | Untested, the most expensive; see `docs/model-selection.md` S1 for its CDNOW risk. |
| Go back to the customer-wise split | **Rejected.** It measures generalisation across customers, not across time. |
| MAPE, bias or Spearman in the training loss | **Rejected.** Not proper scoring rules (`docs/loss-functions.md`). |

**Recommendation.** Run E4 on electronics first, 20 replications against family T's
`archive`, then check CDNOW and gift before adopting anything. Whether family N's
published rows are regenerated under the winner is an ADR-level decision.

## 8. The worst and best runs

Every study of families T, T′ (= E1) and U (§4.2), ranked inside its panel: 920 studies, each one
search's winner (or the pinned arm's single trial), refit and forecast over 200 paths.
Panels are never ranked against each other (`docs/statistical-protocol.md` §7). Each panel
gets four tables: the five worst and five best runs by aggregate MAPE (level), and by
Spearman (ranking), the two primary metrics. All four metrics are shown in every table.
*(`.scratch/training-budget/worst_best_runs.py` → `results/worst_best_runs.csv`; the
recomputed metrics match the archived `factorial.csv` exactly)*

How to read the columns:
- **run**: family · model · arm (U: `training/label`) · replication.
- **CV**: std / mean of the per-customer predicted holdout totals. Near 0 means every
  customer gets nearly the same forecast, i.e. a collapse (§5.2).
- **best epoch**: the epoch whose weights were kept; the model saw the data `best epoch + 1`
  times. **updates**: `ceil(N / batch) × (best epoch + 1)` (§1). Read updates together
  with **batch**: the same count means 8× more data at batch 256 than at batch 32 (§1.1).
- **val CE**: the winner's validation cross-entropy, the quantity the search selected on.
- `paper`, `paper90` and `floored` pin lr 1e-3, wd 0, batch 32, LSTM 128/128 with dropout 0.
  ValendinLSTM's architecture is frozen (ADR-0004).

### 8.1 The tables

#### electronics (320 studies)

*Worst 5 by MAPE*

| run | bias % | RMSE | MAPE | Spearman | CV | best epoch | updates | batch | lr | wd | hidden/dense/dropout | val CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| U · ValendinLSTM · `archive/kmeans_8` · r18 | +140.8 | 0.3896 | 144.1 | +0.274 | 1.23 | 89 | 1,170 | 64 | 1.6e-04 | 9e-05 | frozen | 0.0817 |
| U · ValendinLSTM · `archive/kmeans_8` · r15 | +134.1 | 0.3909 | 137.7 | +0.270 | 1.19 | 35 | 468 | 64 | 5.8e-04 | 9e-06 | frozen | 0.0807 |
| T · LSTM · `paper90` · r11 | +98.7 | 0.3786 | 103.4 | +0.235 | 0.18 | 74 | 1,950 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0891 |
| U · ValendinLSTM · `archive/kmeans_8` · r14 | +85.2 | 0.3814 | 97.3 | +0.325 | 1.03 | 24 | 325 | 64 | 5.1e-04 | 3e-04 | frozen | 0.0806 |
| U · LSTM · `floored/kmeans_8` · r16 | +83.0 | 0.3827 | 93.5 | +0.299 | 1.37 | 14 | 390 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0832 |

*Best 5 by MAPE*

| run | bias % | RMSE | MAPE | Spearman | CV | best epoch | updates | batch | lr | wd | hidden/dense/dropout | val CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| T · LSTM · `paper90` · r08 | -10.4 | 0.3761 | 33.8 | +0.188 | 0.32 | 82 | 2,158 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0888 |
| U · LSTM · `floored/no_cluster` · r11 | -10.7 | 0.3761 | 33.9 | +0.258 | 0.31 | 82 | 2,158 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0888 |
| U · LSTM · `floored/no_cluster` · r06 | -8.0 | 0.3759 | 36.2 | +0.194 | 0.24 | 66 | 1,742 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0886 |
| T · LSTM · `paper90` · r18 | +3.2 | 0.3764 | 36.3 | +0.117 | 0.15 | 76 | 2,002 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0901 |
| T · LSTM · `paper90` · r03 | -8.2 | 0.3763 | 36.4 | +0.204 | 0.25 | 66 | 1,742 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0886 |

*Worst 5 by Spearman*

| run | bias % | RMSE | MAPE | Spearman | CV | best epoch | updates | batch | lr | wd | hidden/dense/dropout | val CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| U · LSTM · `archive/no_cluster` · r18 | +10.6 | 0.3768 | 50.1 | -0.058 | 0.09 | 8 | 36 | 256 | 2.7e-03 | 1e-05 | 128/32/0.18 | 0.0891 |
| U · ValendinLSTM · `archive/no_cluster` · r04 | +48.9 | 0.3774 | 73.5 | -0.049 | 0.08 | 7 | 32 | 256 | 2.5e-03 | 1e-06 | frozen | 0.0889 |
| U · LSTM · `archive/no_cluster` · r15 | +3.7 | 0.3766 | 53.2 | -0.044 | 0.09 | 35 | 144 | 256 | 2.1e-04 | 2e-03 | 64/32/0.26 | 0.0897 |
| T · LSTM · `floor50` · r04 | +16.8 | 0.3769 | 52.4 | -0.043 | 0.09 | 38 | 156 | 256 | 2.5e-03 | 2e-04 | 128/128/0.16 | 0.0884 |
| T · LSTM · `archive` · r15 | +39.8 | 0.3770 | 62.4 | -0.043 | 0.08 | 8 | 36 | 256 | 2.2e-03 | 2e-04 | 128/64/0.06 | 0.0891 |

*Best 5 by Spearman*

| run | bias % | RMSE | MAPE | Spearman | CV | best epoch | updates | batch | lr | wd | hidden/dense/dropout | val CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| U · LSTM · `floored/kmeans_8` · r07 | -13.3 | 0.3759 | 40.2 | +0.337 | 1.51 | 12 | 338 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0825 |
| U · ValendinLSTM · `archive/kmeans_8` · r07 | +23.6 | 0.3785 | 52.8 | +0.335 | 1.56 | 33 | 442 | 64 | 2.4e-03 | 8e-05 | frozen | 0.0802 |
| U · ValendinLSTM · `archive/kmeans_8` · r09 | +33.9 | 0.3785 | 53.4 | +0.333 | 1.29 | 34 | 455 | 64 | 1.1e-03 | 2e-03 | frozen | 0.0810 |
| U · ValendinLSTM · `floored/kmeans_8` · r07 | +44.3 | 0.3780 | 61.8 | +0.333 | 0.94 | 16 | 442 | 32 | 1.0e-03 | 0 | frozen | 0.0868 |
| U · ValendinLSTM · `archive/kmeans_8` · r16 | +70.6 | 0.3817 | 78.5 | +0.331 | 1.33 | 25 | 338 | 64 | 8.8e-04 | 4e-03 | frozen | 0.0809 |


#### multichannel (160 studies)

*Worst 5 by MAPE*

| run | bias % | RMSE | MAPE | Spearman | CV | best epoch | updates | batch | lr | wd | hidden/dense/dropout | val CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| U · LSTM · `archive/kmeans_8` · r17 | +297.9 | 0.0582 | 297.9 | +0.159 | 0.42 | 3 | 24 | 256 | 1.8e-03 | 2e-05 | 128/32/0.15 | 0.0225 |
| U · LSTM · `archive/no_cluster` · r12 | +251.9 | 0.0579 | 254.8 | +0.010 | 0.09 | 3 | 24 | 256 | 1.8e-03 | 2e-05 | 128/64/0.34 | 0.0228 |
| U · LSTM · `archive/no_cluster` · r10 | +242.2 | 0.0578 | 244.7 | -0.029 | 0.09 | 3 | 24 | 256 | 1.9e-03 | 1e-06 | 128/32/0.09 | 0.0228 |
| U · LSTM · `archive/no_cluster` · r18 | +192.1 | 0.0576 | 200.4 | +0.023 | 0.10 | 2 | 18 | 256 | 2.7e-03 | 4e-05 | 128/64/0.38 | 0.0228 |
| U · LSTM · `archive/no_cluster` · r08 | +176.9 | 0.0575 | 184.1 | -0.006 | 0.11 | 6 | 42 | 256 | 8.7e-04 | 4e-06 | 128/32/0.24 | 0.0228 |

*Best 5 by MAPE*

| run | bias % | RMSE | MAPE | Spearman | CV | best epoch | updates | batch | lr | wd | hidden/dense/dropout | val CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| U · LSTM · `floored/no_cluster` · r02 | -13.6 | 0.0569 | 43.4 | +0.023 | 0.22 | 89 | 3,960 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0236 |
| U · ValendinLSTM · `archive/kmeans_8` · r19 | -12.2 | 0.0568 | 43.7 | +0.185 | 2.04 | 12 | 286 | 64 | 2.2e-03 | 4e-04 | frozen | 0.0188 |
| U · ValendinLSTM · `archive/kmeans_8` · r16 | -7.7 | 0.0569 | 45.5 | +0.214 | 2.15 | 24 | 550 | 64 | 1.4e-03 | 1e-06 | frozen | 0.0192 |
| U · LSTM · `floored/no_cluster` · r10 | -0.1 | 0.0569 | 46.0 | +0.047 | 0.23 | 89 | 3,960 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0234 |
| U · LSTM · `floored/no_cluster` · r04 | +0.6 | 0.0569 | 46.3 | +0.103 | 0.31 | 81 | 3,608 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0234 |

*Worst 5 by Spearman*

| run | bias % | RMSE | MAPE | Spearman | CV | best epoch | updates | batch | lr | wd | hidden/dense/dropout | val CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| U · ValendinLSTM · `archive/no_cluster` · r10 | +0.2 | 0.0569 | 63.1 | -0.067 | 0.17 | 5 | 36 | 256 | 2.1e-03 | 6e-05 | frozen | 0.0228 |
| U · ValendinLSTM · `archive/no_cluster` · r19 | +37.9 | 0.0571 | 80.5 | -0.057 | 0.15 | 4 | 30 | 256 | 1.2e-03 | 1e-04 | frozen | 0.0227 |
| U · LSTM · `archive/no_cluster` · r19 | +121.1 | 0.0573 | 134.5 | -0.041 | 0.12 | 3 | 24 | 256 | 1.9e-03 | 3e-05 | 128/64/0.23 | 0.0228 |
| U · LSTM · `archive/no_cluster` · r06 | +55.2 | 0.0571 | 93.7 | -0.038 | 0.14 | 3 | 24 | 256 | 2.0e-03 | 2e-04 | 128/64/0.11 | 0.0228 |
| U · LSTM · `floored/no_cluster` · r09 | +24.9 | 0.0570 | 53.6 | -0.036 | 0.16 | 79 | 3,520 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0239 |

*Best 5 by Spearman*

| run | bias % | RMSE | MAPE | Spearman | CV | best epoch | updates | batch | lr | wd | hidden/dense/dropout | val CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| U · ValendinLSTM · `floored/kmeans_8` · r06 | +1.1 | 0.0570 | 49.0 | +0.232 | 2.10 | 33 | 1,496 | 32 | 1.0e-03 | 0 | frozen | 0.0192 |
| U · LSTM · `floored/kmeans_8` · r06 | +28.6 | 0.0572 | 53.0 | +0.229 | 2.18 | 24 | 1,100 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0193 |
| U · LSTM · `floored/kmeans_8` · r10 | +9.3 | 0.0571 | 52.7 | +0.225 | 2.45 | 71 | 3,168 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0193 |
| U · ValendinLSTM · `floored/kmeans_8` · r13 | +10.0 | 0.0574 | 51.3 | +0.224 | 2.96 | 48 | 2,156 | 32 | 1.0e-03 | 0 | frozen | 0.0191 |
| U · ValendinLSTM · `floored/kmeans_8` · r02 | +14.9 | 0.0571 | 53.4 | +0.222 | 2.19 | 39 | 1,760 | 32 | 1.0e-03 | 0 | frozen | 0.0188 |


#### cdnow (280 studies)

*Worst 5 by MAPE*

| run | bias % | RMSE | MAPE | Spearman | CV | best epoch | updates | batch | lr | wd | hidden/dense/dropout | val CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| U · LSTM · `floored/no_cluster` · r05 | +347.2 | 0.1760 | 348.2 | +0.150 | 0.10 | 27 | 2,072 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0927 |
| U · LSTM · `floored/no_cluster` · r16 | +336.0 | 0.1797 | 341.6 | +0.415 | 0.13 | 9 | 740 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0952 |
| U · LSTM · `floored/no_cluster` · r03 | +333.4 | 0.1750 | 334.1 | +0.228 | 0.06 | 8 | 666 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0975 |
| U · LSTM · `floored/no_cluster` · r17 | +305.6 | 0.1773 | 307.4 | +0.414 | 0.17 | 21 | 1,628 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0947 |
| U · LSTM · `floored/no_cluster` · r15 | +269.4 | 0.1659 | 273.1 | +0.393 | 0.20 | 39 | 2,960 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0928 |

*Best 5 by MAPE*

| run | bias % | RMSE | MAPE | Spearman | CV | best epoch | updates | batch | lr | wd | hidden/dense/dropout | val CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| T' · LSTM · `archive` · r09 | -1.5 | 0.1462 | 18.4 | +0.451 | 1.38 | 22 | 851 | 64 | 2.9e-03 | 9e-06 | 128/64/0.22 | 0.0910 |
| T' · LSTM · `paper` · r15 | -6.2 | 0.1465 | 18.5 | +0.434 | 0.95 | 11 | 888 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0918 |
| T' · LSTM · `archive` · r14 | -6.4 | 0.1465 | 18.5 | +0.428 | 0.94 | 31 | 1,184 | 64 | 2.0e-03 | 3e-04 | 64/128/0.20 | 0.0911 |
| T' · LSTM · `archive` · r15 | -1.0 | 0.1461 | 18.7 | +0.426 | 1.00 | 35 | 1,332 | 64 | 9.0e-04 | 4e-04 | 64/64/0.09 | 0.0910 |
| T' · LSTM · `paper` · r05 | -4.0 | 0.1463 | 18.7 | +0.417 | 1.03 | 19 | 1,480 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0916 |

*Worst 5 by Spearman*

| run | bias % | RMSE | MAPE | Spearman | CV | best epoch | updates | batch | lr | wd | hidden/dense/dropout | val CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| T' · ValendinLSTM · `floor50` · r12 | +48.9 | 0.1591 | 91.7 | -0.150 | 0.60 | 24 | 925 | 64 | 2.6e-03 | 1e-04 | frozen | 0.0910 |
| T' · ValendinLSTM · `floor50` · r19 | +45.4 | 0.1549 | 77.0 | -0.147 | 0.68 | 32 | 1,221 | 64 | 1.2e-03 | 7e-04 | frozen | 0.0913 |
| U · LSTM · `floored/no_cluster` · r06 | +198.7 | 0.1589 | 203.7 | -0.045 | 0.17 | 46 | 3,478 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0915 |
| U · ValendinLSTM · `archive/no_cluster` · r11 | +17.3 | 0.1513 | 75.8 | +0.029 | 0.41 | 13 | 518 | 64 | 2.4e-03 | 8e-03 | frozen | 0.0916 |
| U · LSTM · `floored/no_cluster` · r05 | +347.2 | 0.1760 | 348.2 | +0.150 | 0.10 | 27 | 2,072 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0927 |

*Best 5 by Spearman*

| run | bias % | RMSE | MAPE | Spearman | CV | best epoch | updates | batch | lr | wd | hidden/dense/dropout | val CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| U · ValendinLSTM · `archive/kmeans_8` · r08 | -3.6 | 0.1492 | 29.4 | +0.470 | 2.62 | 51 | 1,924 | 64 | 1.0e-03 | 5e-04 | frozen | 0.0682 |
| U · ValendinLSTM · `archive/kmeans_8` · r01 | +8.5 | 0.1502 | 29.7 | +0.469 | 2.68 | 21 | 814 | 64 | 1.5e-03 | 2e-04 | frozen | 0.0691 |
| U · LSTM · `floored/kmeans_8` · r15 | +17.9 | 0.1510 | 37.6 | +0.453 | 2.38 | 26 | 1,998 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0686 |
| U · LSTM · `floored/kmeans_8` · r16 | +23.1 | 0.1504 | 42.9 | +0.452 | 2.20 | 41 | 3,108 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0680 |
| T' · LSTM · `archive` · r09 | -1.5 | 0.1462 | 18.4 | +0.451 | 1.38 | 22 | 851 | 64 | 2.9e-03 | 9e-06 | 128/64/0.22 | 0.0910 |


#### gift (160 studies)

*Worst 5 by MAPE*

| run | bias % | RMSE | MAPE | Spearman | CV | best epoch | updates | batch | lr | wd | hidden/dense/dropout | val CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| U · ValendinLSTM · `archive/no_cluster` · r12 | -50.8 | 0.1067 | 51.3 | +0.350 | 0.76 | 22 | 759 | 64 | 2.3e-03 | 5e-04 | frozen | 0.0512 |
| U · ValendinLSTM · `archive/kmeans_8` · r04 | +37.8 | 0.1075 | 50.2 | +0.372 | 1.29 | 7 | 264 | 64 | 6.2e-04 | 5e-03 | frozen | 0.0450 |
| U · LSTM · `floored/no_cluster` · r15 | -47.2 | 0.1068 | 48.0 | +0.159 | 0.35 | 39 | 2,600 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0513 |
| U · LSTM · `floored/kmeans_8` · r16 | +12.6 | 0.1072 | 47.1 | +0.345 | 1.32 | 8 | 585 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0466 |
| U · LSTM · `floored/kmeans_8` · r02 | +12.8 | 0.1071 | 46.8 | +0.344 | 1.32 | 8 | 585 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0466 |

*Best 5 by MAPE*

| run | bias % | RMSE | MAPE | Spearman | CV | best epoch | updates | batch | lr | wd | hidden/dense/dropout | val CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| U · LSTM · `archive/kmeans_8` · r02 | -6.6 | 0.1068 | 20.6 | +0.343 | 1.37 | 31 | 1,056 | 64 | 2.0e-03 | 2e-05 | 128/64/0.02 | 0.0458 |
| U · LSTM · `archive/no_cluster` · r15 | -11.2 | 0.1065 | 21.3 | +0.354 | 0.79 | 39 | 1,320 | 64 | 2.7e-03 | 1e-05 | 128/128/0.19 | 0.0515 |
| U · ValendinLSTM · `archive/kmeans_8` · r15 | -9.3 | 0.1068 | 21.6 | +0.327 | 1.43 | 33 | 1,122 | 64 | 9.5e-04 | 2e-06 | frozen | 0.0448 |
| U · ValendinLSTM · `floored/kmeans_8` · r16 | -9.5 | 0.1067 | 22.5 | +0.380 | 1.47 | 20 | 1,365 | 32 | 1.0e-03 | 0 | frozen | 0.0451 |
| U · LSTM · `floored/kmeans_8` · r07 | -1.4 | 0.1067 | 22.6 | +0.366 | 1.34 | 39 | 2,600 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0451 |

*Worst 5 by Spearman*

| run | bias % | RMSE | MAPE | Spearman | CV | best epoch | updates | batch | lr | wd | hidden/dense/dropout | val CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| U · LSTM · `floored/no_cluster` · r19 | -9.2 | 0.1070 | 27.5 | +0.014 | 0.11 | 23 | 1,560 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0518 |
| U · LSTM · `floored/no_cluster` · r14 | -12.9 | 0.1070 | 23.3 | +0.021 | 0.11 | 22 | 1,495 | 32 | 1.0e-03 | 0 | 128/128/0.00 | 0.0520 |
| U · ValendinLSTM · `floored/no_cluster` · r11 | -23.9 | 0.1070 | 29.4 | +0.023 | 0.11 | 32 | 2,145 | 32 | 1.0e-03 | 0 | frozen | 0.0508 |
| U · ValendinLSTM · `floored/no_cluster` · r03 | +7.9 | 0.1071 | 29.6 | +0.042 | 0.11 | 29 | 1,950 | 32 | 1.0e-03 | 0 | frozen | 0.0505 |
| U · LSTM · `archive/no_cluster` · r13 | -6.3 | 0.1070 | 22.7 | +0.104 | 0.11 | 39 | 1,320 | 64 | 3.0e-03 | 4e-03 | 32/64/0.11 | 0.0525 |

*Best 5 by Spearman*

| run | bias % | RMSE | MAPE | Spearman | CV | best epoch | updates | batch | lr | wd | hidden/dense/dropout | val CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| U · ValendinLSTM · `archive/no_cluster` · r06 | +2.3 | 0.1064 | 24.3 | +0.384 | 0.62 | 18 | 627 | 64 | 2.6e-03 | 4e-03 | frozen | 0.0515 |
| U · ValendinLSTM · `archive/no_cluster` · r05 | +24.5 | 0.1067 | 32.0 | +0.381 | 0.52 | 58 | 1,947 | 64 | 9.3e-04 | 2e-04 | frozen | 0.0513 |
| U · ValendinLSTM · `archive/kmeans_8` · r11 | +5.8 | 0.1071 | 28.4 | +0.381 | 1.44 | 24 | 825 | 64 | 8.1e-04 | 3e-03 | frozen | 0.0451 |
| U · ValendinLSTM · `archive/kmeans_8` · r13 | +20.8 | 0.1071 | 38.9 | +0.381 | 1.18 | 13 | 462 | 64 | 1.3e-03 | 4e-06 | frozen | 0.0454 |
| U · ValendinLSTM · `floored/kmeans_8` · r16 | -9.5 | 0.1067 | 22.5 | +0.380 | 1.47 | 20 | 1,365 | 32 | 1.0e-03 | 0 | frozen | 0.0451 |

### 8.2 Patterns

These are descriptive. The runs were picked on the outcome, so the worst-vs-best contrasts
below are not protocol claims. Within-cell correlations are Spearman rank correlations
over a cell's 20 studies, summarised by their median over the 46 cells. "Decile" means a
panel's 10% worst or best runs (16–32 studies).

**1. Level failures and ranking failures are different runs.** The worst-MAPE and
worst-Spearman deciles share 5 of 32 runs on electronics, 4 of 16 on multichannel, 5 of 28
on CDNOW and 1 of 16 on gift. A run can rank customers well and still miss the total by
140% (electronics r18 above), or get the total within 10% and rank at chance.

**2. A level failure is an over-forecast.** On electronics, multichannel and CDNOW every
run in the worst-MAPE decile over-forecasts, with median bias +68%, +130% and +187%. What
is left after bias, `MAPE − |bias|`, is only 4–13 points there, against 15–39 points in the
best decile. So the worst runs have the right weekly shape and the wrong level. Gift is the
exception: its worst runs miss in both directions (under-forecasts of −45 to −51% without
the label, mostly over-forecasts of up to +38% with it), and its whole range is narrow (MAPE 21–51).
RMSE follows bias but only in the third decimal, so it separates nothing (§4 of the
protocol).

**3. A collapsed forecast over-forecasts.** In 43 of the 46 cells, a lower forecast CV goes
with a higher bias (median within-cell correlation −0.47). The clearest case is CDNOW's
`U · LSTM · floored/no_cluster` cell, the one behind the +126 MAPE blow-up of §5.2:

| runs in the cell | CV | bias % |
| --- | ---: | ---: |
| 12 of 20 | < 0.25 | +99 to +347 |
| 4 of 20 | > 0.8 | +5 to +74 |

Spearman stays at 0.39–0.43 in 8 of the 12 collapsed runs, so the model still orders
customers. What it loses is the spread of the forecast. This is consistent with
a forecast that gives every customer something close to the average rate, including the
many who have stopped buying. `docs/absorbing-death-state.md` covers that mechanism; it is
not tested here.

**4. A ranking failure is a collapse, and on electronics and multichannel a collapse is
undertraining.** Across all runs, CV and Spearman correlate at 0.91 on electronics and 0.89
on multichannel. The worst-Spearman decile there kept epoch 8 and 3.5 (median), received 48
and 27 updates, ran at batch 256 in 69% and 94% of runs, and had no label in all of them.
That is short in both currencies: 9 and 4.5 epochs, not just few large-batch steps.
Every one of the 40 `archive/no_cluster` runs on multichannel, and 79 of 80 on electronics,
collapsed (CV < 0.2).

**5. On CDNOW and gift, collapse is not caused by too little training.** The collapsed
runs there received a median of 1,443 (CDNOW) and 1,560 (gift) updates, mostly at batch
32, which is 25 and 30 epochs: long in both currencies. 11 of CDNOW's 12
and 5 of gift's 7 collapsed runs come from `floored/no_cluster`: batch 32, no dropout, no
weight decay, no label. This is the floor's harm on gift (§5.2, claim 8) and its blow-up on
CDNOW (claim 9), seen run by run. **The recipe alone does not explain it.** On CDNOW,
`T′ · LSTM · paper` runs the same pinned recipe and supplies two of the five best runs by
MAPE, with CV ≈ 1. The two cells differ in their inputs: the T′ LSTM reads `Transactions`
alone, while the U LSTM reads `Transactions, week_sin, week_cos` (each winner's
`selected_features`). So the CDNOW collapse comes from the recipe combined with that input
set. Which part of the difference matters is not identified.

**6. The cluster label prevents collapse, carries the best ranking, and widens the level
tail.** Every run in electronics' best-Spearman decile has the label, and 15 of 16 on
multichannel do. But four of electronics' five worst MAPE runs also carry it: the
`ValendinLSTM · archive` MAPE sd is 29.8 with the label against 10.0 without. Those label
runs reached a lower validation CE (0.081 against 0.089) and still produced the worst
level, so a better selection objective did not protect the total.

**7. Long training lifts the level on electronics and multichannel, but not reliably on
the LSTM.** The best-MAPE decile kept epoch 67.5 and 64 (median), with 1,742 and 2,860
updates. Multichannel's worst-MAPE decile kept epoch 3.5 with 27 updates, and 13 of its 16
runs are `LSTM · archive/no_cluster`. But one recipe produced both extremes on electronics:
`T · LSTM · paper90` holds the best run (r08, MAPE 33.8, epoch 82) and the third worst
(r11, MAPE 103.4, epoch 74). That cell's MAPE sd is 19.0, against 7.8 for `archive`.

**8. Validation CE does not flag a bad run.** Within a cell, the correlation between the
winner's validation CE and its holdout MAPE has a median of +0.04 (CDNOW +0.06, electronics
+0.03, gift −0.01, multichannel +0.08). In the CDNOW cell of pattern 3, validation CE spans
0.0915–0.0975 while MAPE spans 22.6–348.2. This is `docs/model-selection.md`'s finding
again, at the level of the individual run.

**9. Within a cell, the kept epoch matters little.** The within-cell correlation between
best epoch and MAPE has a median of −0.15 (electronics −0.22, multichannel −0.24, CDNOW 0.00,
gift −0.10). The epoch effect of §5 is between arms: what a recipe lets training reach, not
which epoch one search happened to keep.

**10. The floor works by getting past the flat stretch, not by keeping late weights.** 92%
of the 440 floored runs kept weights from *before* their floor epoch. The floor lets training
continue past the plateau of §3.2 until it finds a lower loss, which often sits before the
floor anyway. CDNOW's blown-up runs kept epochs 8–61, so §5.3's suspect, a 90-epoch floor on
a 39-week window, cannot work by handing the forecast overtrained weights. It would have to
act through which early epoch gets selected, or through the refit.

## 9. Limits

- Four panels plus electronic_5y, each stated separately. No measured panel
  characteristic predicts where a floor helps.
- §3.2–3.4 rest on 3 runs per panel at one hyperparameter setting.
- Training is unseeded, but a replication's forecast seed fixes the RNG entering the next
  replication's training. Arms doing identical work in identical order are therefore
  coupled; the archive shows 0 exact repeats across 3,020 suites, so it is unaffected.
