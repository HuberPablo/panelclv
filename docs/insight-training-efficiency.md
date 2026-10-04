# Training efficiency: the models stop training too early

Every neural result in this package comes from a model that stopped training while its
validation loss was still falling. This doc covers:
- how we know, and why it happens (§1–3);
- what we tried, and what it changed (§4–5);
- what it means for the comparison with Valendin et al. (§6);
- what is left to try (§7);
- which individual runs did worst and best, and what they share (§8).

How the stopped models are then *selected* is a separate question, covered in
`docs/model-selection.md`.

**How to read the numbers.** A comparison is reported as Δ, the difference between two
conditions' means, followed by its 95% bootstrap interval in brackets
(`docs/statistical-protocol.md`, computed by `evaluation.effects.effect`). Each
replication is an independent study. **Bold** marks an interval that excludes 0, i.e. a
supported difference. How much a forecast moves when the same model is simply refitted
(refit noise) is in `docs/model-selection.md` §2.

## Claims

| # | Claim | Panel | Status | § |
| --- | --- | --- | --- | --- |
| 1 | Patience, not the epoch budget, ends every archived run | all four | established | 2 |
| 2 | With early stopping off, validation loss keeps falling until epoch 88–247 | cdnow, electronics, multichannel | established (3 runs each) | 3.2 |
| 3 | The early stop comes from our temporal split combined with the fixed 10⁻⁴ improvement threshold, not from the panel | electronics | established | 3.3–3.4 |
| 4 | Under patience 7 the search prefers the batch size that takes the fewest updates | electronics | established, descriptive | 3.5 |
| 5 | The notebook's recipe with a 90-epoch floor (`paper90`) improves level and ranking | electronics | **established** (ValendinLSTM MAPE −22.3, Spearman +0.150) | 5.1 |
| 6 | The notebook's settings without a floor (`paper`) change nothing | electronics | no clear difference | 5.1 |
| 7 | A training floor improves ranking | electronics, multichannel | **established** | 5.2 |
| 8 | …and worsens it | gift | **established** (−0.069) | 5.2 |
| 9 | The floored notebook recipe triples CDNOW's LSTM error | cdnow | **established** (+126.2) | 5.2 |
| 10 | …and the floor alone is not the cause | cdnow | floor alone: no clear difference (E1) | 5.3 |
| 11 | On a 260-week window a floor narrows the spread across studies but does not move the mean | electronic_5y | no clear difference in means | 5.4 |
| 12 | The worst level errors are over-forecasts from collapsed forecasts (low CV) | electronics, multichannel, cdnow | descriptive (43 of 46 cells) | 8.2 |
| 13 | Collapse is undertraining on electronics and multichannel; on cdnow and gift it comes with the floored no-label recipe, after many updates | all four | descriptive | 8.2 |
| 14 | Validation loss does not separate the worst runs from the best within a cell | all four | descriptive | 8.2 |

## 1. How training stops

The training loop (`fit_model`, `training/loop.py`) works like this:
- After every epoch it computes the validation loss.
- The epoch counts as an improvement only if that loss beats the best so far by more than
  10⁻⁴.
- After `patience` (7) epochs in a row without an improvement, training stops and the
  weights from the best epoch are restored.

What each trial stores:
- `best_epoch`, the epoch whose weights were kept.
- Not the number of epochs actually run. With no floor and a completed trial, that is
  `best_epoch + 1 + patience`.

How much a kept model trained is reported in two ways, always together with its batch
size:
- **epochs**, `best_epoch + 1`: how many times the model saw the N training customers;
- **updates**, `ceil(N / batch_size) × (best_epoch + 1)`: how many optimiser steps it took.

Neither one alone compares runs with different batch sizes (§1.1).

### 1.1 Why updates alone do not measure training

One update at batch 256 is not worth one update at batch 32, so a raw update count does
not compare runs that used different batch sizes.

**Why.** A larger batch averages the gradient over more customers, so each step points
more reliably downhill, but it uses eight times the data. Which of the two counts matters
depends on the **critical batch size**, which is set by how noisy the gradient is
(McCandlish et al., 2018):
- Well below it, doubling the batch roughly halves the updates needed and leaves the data
  needed unchanged. Epochs are the fair measure.
- Well above it, a larger batch barely reduces the updates needed. Updates are the fair
  measure.
- In between, both matter, and where the switch happens depends on the model, the data and
  the optimiser (Shallue et al., 2019).

The learning rate is part of the same quantity. SGD's gradient noise scales roughly as
lr × N / batch (Smith et al., 2018). Large-batch SGD needs its learning rate scaled up in
proportion to the batch (Goyal et al., 2017), and Adam by about its square root (Malladi
et al., 2022). The large-batch "generalisation gap" largely closes once large batches get
as many updates as small ones (Hoffer et al., 2017). So counting updates is justified under
those scaling rules, not as a raw comparison across settings.

**What this means here.** Our searches vary batch, learning rate and weight decay
together, under AdamW, so an update count compared across trials mixes all three. At batch
256, electronics' 829 customers take four steps per epoch, close to full-batch descent,
where updates are probably the measure that binds. But a trial with few updates at batch
256 can still have seen *more* data than one with more updates at batch 64 (§3.5).

This doc therefore reports batch, epochs and updates side by side, and does not rest the
undertraining claim on an update count. The direct evidence is §3.2: with early stopping
switched off, the validation loss keeps falling long after the kept epoch.

References: Goyal et al. (2017), *Accurate, Large Minibatch SGD*, arXiv:1706.02677.
Hoffer, Hubara & Soudry (2017), *Train Longer, Generalize Better*, NeurIPS. McCandlish,
Kaplan, Amodei et al. (2018), *An Empirical Model of Large-Batch Training*,
arXiv:1812.06162. Smith, Kindermans, Ying & Le (2018), *Don't Decay the Learning Rate,
Increase the Batch Size*, ICLR. Shallue et al. (2019), *Measuring the Effects of Data
Parallelism on Neural Network Training*, JMLR. Malladi, Lyu, Panigrahi & Arora (2022), *On
the SDEs and Scaling Rules for Adaptive Gradient Algorithms*, NeurIPS.

## 2. Patience ends every run

The 100-epoch budget never decides when training ends. The patience of 7 does.

The measurement covers every archived study on the four real panels: 407,950 trials, of
which 163,363 completed, all with patience 7 and a 100-epoch budget.

| panel | model | median best epoch | share reaching 100 epochs |
| --- | --- | ---: | ---: |
| cdnow | LSTM / ValendinLSTM / Transformer | 10 / 14 / 16 | 0.8% / 0.1% / 3.8% |
| electronics | LSTM / ValendinLSTM / Transformer | 8 / 7 / 15 | 0.8% / 0.3% / 5.2% |
| gift | LSTM | 13 | 0.4% |
| multichannel | LSTM / ValendinLSTM | 8 / 4 | 0.5% / 0.0% |

- At most 5.2% of runs in any row reach the 100-epoch budget. The median run keeps its
  weights from epoch 4–16.
- On multichannel, the winning trial kept epoch 3 or earlier in 55% of ValendinLSTM runs.
- Runs that stopped earlier ranked customers worse: in the `log` arm, the rank correlation
  between best epoch and holdout Spearman is 0.59 (descriptive).

*(`.scratch/training-budget/epochs.py`)*

## 3. How we know it undertrains, and why

### 3.1 The reference notebook: its recipe, and an epoch count that does not transfer

The "Valendin et al. recipe" in this doc is the training setup of the demo notebook
published with the paper (`Original_paper_model/banking_transactions_demo.ipynb`). The
paper itself does not specify it. The notebook's often-quoted "~90 epochs" is one run on a
banking dataset, and it does not tell us how long our panels need to train.

| | Valendin et al. demo notebook | Our studies |
| --- | --- | --- |
| data | Czech banking transactions (data.world): 2,239 accounts, calibration 1993–1995, a random 10% of customers held out for validation | our four panels, temporal validation split (ADR-0001) |
| optimizer | Adam defaults: lr 1e-3, no weight decay | AdamW; lr searched 1e-4–3e-3, weight decay 1e-6–1e-2 |
| batch size | 32 | searched over {64, 128, 256} |
| patience / budget | 5 / 150 epochs | 7 / 100 epochs |
| epochs trained | one logged run: stopped at epoch 90, kept epoch 85 | median best epoch 7–14 (§2) |

**Where "~90 epochs" comes from.** Only from the notebook:
- its saved log (one run, 16 March 2022) ends `Epoch 90: early stopping … best epoch: 85`;
- its markdown says the validation loss "should end up around 0.44 after ±90 epochs";
- a code comment says "about 100 epochs in total".

**The paper gives no epoch count.**
- §2.2 trains "until we reach a minimum" of the validation error.
- Appendix B.2 early-stops, restores the best epoch, then runs "several fine-tuning epochs"
  on the full calibration set at a large batch and a reduced learning rate.
- Appendix B.1 says the learning rate and batch size were *searched*, by a random walk over
  architectures and settings. So the paper does not fix those either.

**Why 90 does not transfer.** How many epochs a model needs depends on the panel: its
size, its sparsity, its calibration length and the validation split. 90 is how long one
run trained on the banking data. It is not an expected training length for any of our
panels, and comparing it with our epoch counts says nothing about whether our models
undertrain.

**What does compare.** §3.4 runs the same recipe on electronics under the notebook's own
split:

| | kept epoch | epochs | updates | batch |
| --- | ---: | ---: | ---: | ---: |
| notebook's recipe, notebook's split (§3.4) | 28–56 | 29–57 | ≈ 750–1,500 | 32 |
| our electronics benchmark winners | 7 | 8 | ≈ 32 | 256 |

That is 4–7 times fewer epochs on the same panel. The update gap is wider only because our
batch is eight times larger, and updates alone do not compare across batch sizes (§1.1).
The undertraining claim does not rest on this comparison. It rests on §3.2.

`scripts/validate_valendin_lstm.py` runs the notebook's recipe on the notebook's banking
data and reproduces its validation loss of about 0.44, which checks our implementation of
that recipe. Our own search space cannot express the recipe: batch 32 is absent and weight
decay has no zero.

### 3.2 The validation loss keeps falling after patience 7 stops

With early stopping switched off, the validation loss keeps falling long after patience 7
would have stopped. The curve stays flat for a long stretch, then drops.

**The measurement.**
- The frozen ValendinLSTM, trained for 200 epochs with early stopping off, 3 runs on each
  of electronics, multichannel and CDNOW.
- One fixed setting for every run: the hyperparameters one archived electronics study
  selected (lr 1.15e-3, weight decay 5.7e-4, batch 256). Multichannel and CDNOW use this
  electronics setting too, not one of their own.
- The loss tracked is the one the stopping rule watches: the validation loss on the
  validation weeks of our temporal split. No forecast is made.
- Afterwards, the patience-7 rule is replayed on each recorded curve. "Patience 7 stops
  at" is where it would have stopped. "Improvement forgone" is how far the 200-epoch
  minimum lies below the best loss patience 7 had seen.

| panel | patience 7 stops at | its best val loss | best over 200 epochs | at epoch | improvement forgone |
| --- | ---: | ---: | ---: | ---: | ---: |
| electronics | 19 | 0.0888–0.0896 | 0.0848–0.0858 | 137–197 | 3.4–5.3% |
| multichannel | 12–16 | 0.0229–0.0230 | 0.0220–0.0226 | 169–181 | 1.4–4.4% |
| cdnow | 11 | 0.1078–0.1098 | 0.0946–0.0964 | 88–120 | 10.7–13.0% |

Before reaching its minimum, each run went 29–131 epochs without a single counted
improvement. No fixed patience can be trusted to wait that out. §3.3 explains why the
curve looks flat.

**What this does not show.**
- **That the forecast improves.** A lower validation loss is not a better holdout
  forecast. Across trials, validation loss does not predict the forecast
  (`docs/model-selection.md`; §8.2, Finding 3). The forecast evidence for longer training
  is §5, at 20 replications per arm.
- **That it holds generally.** It is one model, one hyperparameter setting borrowed from
  electronics, and 3 runs per panel.

*(`.scratch/training-budget/probe_patience.py`. The numbers are from the run recorded in
commit `d6cea22`, 20 Sep. Its output, `results/patience_probe.csv`, is no longer on disk,
so they cannot be re-checked from the repo. A companion script, `end_to_end.py`, forecast
the holdout on electronics under patience 7 and patience 40, 5 runs each. Under patience
40, one of the five runs got past the plateau to the later minimum, and the other four
still kept weights from epochs 6–11. Its output is lost too, and its old write-up describes
its long arm in two inconsistent ways, so its forecast numbers are not used here.)*

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

The early stop comes from our temporal split, not from electronics. Under the notebook's
own split, the same recipe on the same panel trains for 28–56 epochs before it stops.

The measurement: the notebook's recipe on electronics, validated the notebook's way (a
random 10% of customers held out), 3 replications.

| replication | epochs run | best epoch |
| ---: | ---: | ---: |
| 0 | 62 | 56 |
| 1 | 39 | 33 |
| 2 | 34 | 28 |

Under our temporal split, the same recipe keeps epoch 1 (§4.2).

- Only the epoch counts compare. The two validation losses are averaged over different
  customer-weeks, so their values do not.
- The temporal split is still the right one for a forecasting evaluation: it does not
  leak, and it matches the test task (ADR-0001). What went unchecked was what it does to
  the stopping rule.

*(`.scratch/training-budget/paper_split_check.py`, `why_flat.py`)*

### 3.5 The search rewards the setting that takes the fewest steps

Under patience 7, the search picks batch 256, the batch size that takes the fewest
optimiser steps, although by epochs it is the one that sees the most data. Every batch
size trains little.

The measurement: 20 electronics benchmark studies (2,000 trials, 713 completed), grouped
by batch size.

| batch | completed | best val loss | median best epoch | median epochs | median updates |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 136 | 0.0899 | 3 | 4 | 52 |
| 128 | 178 | 0.0911 | 4 | 5 | 35 |
| 256 | 399 | 0.0886 | 7 | 8 | 32 |

- **Batch 256 won all 20 studies.** A likely reason: small-batch trials need more epochs to
  show their advantage, and patience 7 stops them first.
- **The two measures rank the batches in opposite orders.** Batch 256 takes the fewest
  steps but sees the most data, so it "trains least" only in updates.
- **Every batch size trains little in both measures:** at most 8 epochs, against 29–57 for
  the notebook's recipe on the same panel under its own split (§3.4).

**More training goes with better ranking.** Across the 40 pre-experiment electronics
studies without a label, the rank correlation between a study's holdout Spearman and the
updates its winner received is **+0.328** (+0.024, +0.580). With epochs instead (best
epoch), it is +0.280, not supported. This is one correlation across studies, resampled by
study. Batch, learning rate and weight decay vary along with it, so it does not say which
measure matters (§1.1). Inside a single study, the longest-trained trial is the best pick
of any selection criterion on electronics (`docs/model-selection.md` §3.5).

*(`.scratch/training-budget/trials_by_bs.py`)*

## 4. What was tested

Four experiments, run in this order, each answering a question the previous one left
open. Only the training recipe changes. Everything else stays fixed: our temporal split
(ADR-0001), the ADR-0008 refit, the cross-entropy loss, and the two models' architectures
and inputs. The family letters follow `docs/studies-run.md` §4, which holds each family's
full specification.

### 4.1 The arms, and what "paper" means

**"Paper" in an arm name always means the training recipe of Valendin et al.'s demo
notebook** (`Original_paper_model/banking_transactions_demo.ipynb`), copied setting by
setting. It never means the paper's data, its split or its published numbers.
`scripts/validate_valendin_lstm.py` checks that this recipe, run under the notebook's own
split, reproduces the notebook's validation loss (§3.1).

**A floor** (`min_epochs`) stops early stopping from firing before that epoch. The kept
epoch is still the one with the lowest validation loss, so the kept weights can come from
before the floor (§8.2, Finding 6).

| arm | what it is | settings | trials |
| --- | --- | --- | ---: |
| `archive` | our own recipe, the one behind every archived result (family N, `docs/benchmarks.md`); the control | lr, weight decay and batch searched over the registry's ranges; patience 7; 100 epochs | 100 |
| `paper` | the notebook's recipe, literally | Adam defaults (lr 1e-3, no weight decay); batch 32; patience 5; 150 epochs; LSTM 128/128, dropout 0 | 1 |
| `paper90` | the notebook's recipe plus a 90-epoch floor | `paper` + `min_epochs=90` | 1 |
| `floored` | family U's name for `paper90` | identical to `paper90` | 1 |
| `floor50` | nothing from the paper: a floor added to our own recipe | `archive` + `min_epochs=50`, 300 epochs | 100 |

**Why a floor of 90.** It is the length of the notebook's one logged run, on its banking
data (§3.1). The paper reports no training length, and 90 is not what the recipe needs on
our panels: on electronics, under the notebook's own split, the same recipe keeps epoch
28–56 (§3.4). Read `paper90` as "the notebook's recipe with a long floor that rules out an
early stop", not as "training the way the paper trained".

**Why the pinned arms run one trial.** Pinning every setting removes hyperparameter
selection, so a difference against `archive` comes from training, not from which trial
happened to win. Selection does not predict the forecast anyway
(`docs/model-selection.md`).

### 4.2 The four experiments

**Family T: electronics** (`scripts/run_training_budget.py`, 20 Sep). ValendinLSTM and
LSTM × `archive` / `paper` / `paper90` / `floor50` × 20 replications, 200 paths.
- *Question:* does the notebook's recipe, trained long enough, remove electronics'
  collapse?
- *Why:* electronics' benchmark forecast is collapsed (MAPE 70.8, Spearman 0.032). Its
  winners kept epoch 7 at batch 256, while the notebook's recipe on the same panel, under
  its own split, keeps epoch 28–56 (§3.4).
- *Design:* `paper` copies the settings. `paper90` adds a 90-epoch floor to them. `floor50`
  adds a 50-epoch floor to our own search, the only option for a model with no published
  recipe, such as our Transformer. `paper90` exists because pre-launch probes had shown
  that `paper` alone keeps epoch 1 under our split.
- *Results:* §5.1.

**Family U: all four panels** (`scripts/run_factorial.py`, 21 Sep). ValendinLSTM and LSTM
× {`archive`, `floored`} × {`no_cluster`, `kmeans_8`} × 20 replications.
- *Question:* do the training fix and a cluster label add up, and does family T's
  electronics result hold on the other panels?
- *Why:* each lever had lifted electronics' Spearman on its own. Training (family T) took
  it from 0.027 to 0.177, and a cluster label (family F) from 0.039 to 0.27. Neither
  experiment included the other.
- *Inputs differ from family T and T′.* On CDNOW, family U's LSTM reads `Transactions,
  week_sin, week_cos`, while family T′'s reads `Transactions` alone. On electronics both
  read the same three columns; family T's panel also builds a year index, but neither LSTM
  reads it.
- *Results:* the training half is in §5.2. The label half is in
  `docs/insights-real-panels.md` §5.2.

**Family T′ = experiment E1: CDNOW** (`scripts/run_training_budget.py` on CDNOW, 21 Sep;
analysed by `.scratch/model-selection/e1_cdnow.py`). ValendinLSTM and LSTM × `archive` /
`floor50` / `paper` × 20 replications.
- *Question:* which ingredient of `floored` tripled CDNOW's LSTM error (§5.2, claim 9)?
- *Why:* `floored` changes the floor, batch, learning rate, weight decay, search and epoch
  budget all at once. Here each arm changes one thing against `archive`: `floor50` only the
  floor, `paper` only the settings.
- *Naming:* "T′" (the family name in `docs/studies-run.md` and §8) and "E1" (its to-do
  name in `docs/model-selection.md`) are the same 120 suites.
- *Results:* §5.3.

**Family X: the epoch floor on electronic_5y** (`scripts/run_epoch_floor_5y.py`,
26–28 Sep). ValendinLSTM, LSTM + `ar_bounded_52` and LSTM + `kmeans_8` × `nofloor` /
`from20` / `from30` × 20 replications, 500 paths.
- *Question:* on the paper's own electronics split, does keeping only later weights reduce
  the bias?
- *Why:* this is the only test on the paper's own electronics split (260 calibration weeks,
  `docs/datasets.md`). There the ValendinLSTM benchmark (family W) spread its bias from
  −7.9% to +37.4% at identical validation loss, and a later kept epoch went with a lower
  bias.
- *Design:* the least-biased study's settings are pinned, and only the eligible epochs
  vary. `nofloor` may keep any epoch; `from20` and `from30` only epoch 20 or 30 onwards
  (`select_from_epoch`). Unlike `min_epochs`, this forces late weights.
- *Results:* §5.4.

**E4 and E5** did not run. They are the to-do names (`docs/studies-run.md` §6) for a
stopping rule suited to the flat temporal curve, and for a floor scaled to the calibration
length (§7).

## 5. Results

### 5.1 Electronics: it is the epochs, not the settings (family T)

On electronics, copying the notebook's settings does not help. Adding a floor does.

Δ against `archive`, n = 20 / 20:

| model | arm | Δ MAPE | Δ Spearman | Δ \|bias\| |
| --- | --- | ---: | ---: | ---: |
| ValendinLSTM | `paper` | −0.6 (−6.3, +5.0) | −0.011 (−0.034, +0.011) | **−11.8** (−21.2, −2.3) |
| | `paper90` | **−22.3** (−28.3, −16.2) | **+0.150** (+0.107, +0.191) | **−16.4** (−26.0, −6.6) |
| | `floor50` | **−21.2** (−26.8, −15.5) | **+0.141** (+0.100, +0.180) | **−14.5** (−24.3, −4.5) |
| LSTM | `paper` | +4.3 (−0.2, +9.0) | −0.011 (−0.034, +0.011) | +4.0 (−5.2, +13.6) |
| | `paper90` | −4.5 (−12.9, +4.6) | **+0.144** (+0.106, +0.180) | +3.7 (−9.8, +17.8) |
| | `floor50` | **−9.2** (−13.3, −4.5) | **+0.043** (+0.015, +0.070) | **−17.6** (−25.9, −8.8) |

For scale: ValendinLSTM `archive` scores MAPE 69.1 and Spearman 0.027; `paper90` scores
46.8 and 0.177.

- **The settings alone do not help.** Under our split, `paper`'s own stopping rule keeps
  epoch 1 and stops after 7 epochs (§3.3), so neither MAPE nor Spearman moves. Only
  ValendinLSTM's |bias| falls.
- **The floor does.** For ValendinLSTM, `paper90` cuts MAPE by 22 points and raises
  Spearman about sevenfold. For the LSTM it lifts Spearman (+0.144) but shows no clear MAPE
  difference.
- **We cannot tell whether epochs or updates are what matters.** `paper90`'s kept weights
  come from a median epoch of 56 (ValendinLSTM) and 71.5 (LSTM) at batch 32, about
  1,500–1,900 updates. `floor50`'s come from epoch 67 and 26 at batch 256, about 330 and 110
  updates. The two arms also differ in learning rate and weight decay (§1.1).
- **A floor works with our search left in place.** The searched `floor50` and the pinned
  `paper90` show no clear MAPE difference on either model. On Spearman, the searched LSTM
  ranks worse (−0.100, −0.138 to −0.063).

### 5.2 Four panels: the floor helps two, hurts one, and blows one up (family U)

The floor improves ranking on electronics and multichannel, worsens it on gift, and
triples CDNOW's LSTM error.

ValendinLSTM Spearman, `floored` against `archive`, no label (n = 20 / 20):

| panel | Δ Spearman |
| --- | ---: |
| electronics | **+0.157** (+0.119, +0.193) |
| multichannel | **+0.123** (+0.099, +0.147) |
| cdnow | +0.019 (−0.019, +0.065) |
| gift | **−0.069** (−0.118, −0.027) |

The LSTM moves the same way on the three panels with a supported difference (electronics
**+0.153**, multichannel **+0.076**, gift **−0.063**). On CDNOW it shows no clear
difference (−0.034).

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

- **The gains are on the two panels where the model collapses**, i.e. gives every customer
  nearly the same forecast: electronics and multichannel, whose forecast CV under `archive`
  is 0.08–0.15.
- **CDNOW's blow-up also depends on the inputs.** With the cluster label added, the floor
  shows no clear MAPE difference on CDNOW's LSTM (+4.5, −27.1 to +34.4).

### 5.3 CDNOW: the floor alone is not what blows up (E1)

Neither the floor alone nor the notebook's settings alone come close to tripling CDNOW's
LSTM error. The tripling is still unexplained.

`floored` changed the floor, batch size, weight decay, learning rate, search and epoch
budget all at once. E1 changes one at a time (MAPE, n = 20 / 20):

| model | `floor50` vs `archive` | `paper` vs `archive` |
| --- | ---: | ---: |
| LSTM | +3.5 (−0.4, +8.2) | **+3.6** (+0.2, +7.0) |
| ValendinLSTM | −3.7 (−20.8, +9.7) | +2.5 (−15.0, +16.4) |

- **E1 is not the same setup as family U.** E1's LSTM `archive` scores MAPE 21.7, against
  57.7 in family U. Their LSTMs read different inputs: `Transactions` alone in E1, plus
  `week_sin`/`week_cos` in family U (§4.2).
- **E1 has no 90-epoch arm.** The remaining suspect is a 90-epoch floor on a 39-week
  window, untested. §8.2 (Finding 6) narrows how it could act.

### 5.4 Electronic_5y: tighter, not better on average

On the paper's own electronics split, keeping only later weights narrows the spread across
studies but does not clearly lower the average error.

Pinned settings, 260 calibration weeks, 20 studies per rule:

| model | rule | bias % (mean ± sd) | MAPE (mean ± sd) |
| --- | --- | ---: | ---: |
| LSTM (ValendinLSTM shape) | `nofloor` | +13.2 ± 16.7 | 21.5 ± 11.7 |
| | `from20` | +10.2 ± 7.4 | 17.8 ± 3.2 |
| | `from30` | +12.7 ± 7.3 | 19.4 ± 3.4 |

- **No clear difference in the mean.** `from20` against `nofloor`: MAPE −3.7 (−9.6, +0.5).
  `from30` against `from20`: MAPE +1.5 (−0.5, +3.5).
- **A much narrower spread.** The bias sd halves, and the worst study moves from +66.7%
  to +23.9%.
- **Still biased.** Every cell over-forecasts by about 10% (`from20` mean bias +10.2, +7.1
  to +13.2), against the published +2.7%.

The flags, label and attention cells of the same experiment are in
`docs/insights-real-panels.md` §7.

## 6. What this means for the comparison with Valendin et al.

- **It does not correct the paper.** Run the notebook's way, with its split and its
  recipe, the model trains a sensible number of epochs on electronics too (§3.4).
- **It corrects our reproduction.** Our one deliberate departure, the temporal split
  (ADR-0001), breaks the notebook's stopping rule, and every archived neural result
  trained under that combination. The electronics and multichannel benchmark rows in
  `docs/benchmarks.md` are therefore lower bounds on what the architecture can do, not
  measurements of it. CDNOW and gift are not shown to be affected.
- **The narrow thesis claim.** On our electronics and multichannel panels, much of the
  published LSTM's weak per-customer ranking under a temporal split is an artefact of
  training. When the validation split changes, the training protocol has to be
  re-derived.

## 7. What can be done

| Option | Status |
| --- | --- |
| Epoch floor (`min_epochs`) | Tested (§5). Helps on electronics and multichannel, not on CDNOW, and hurts on gift. **Never an unconditional default.** |
| The notebook's recipe | Tested (§5.1). Only works with a floor added. |
| Batch 32 in the search space | Not the fix: under patience 7 it would still stop early. |
| A relative improvement threshold (a fraction of the current best loss) | Untested (E4). Targets §3.3 directly: 10⁻⁴ is 0.11% of electronics' loss but 0.4% of multichannel's. |
| Patience counted in optimiser steps, not epochs | Untested (E4). An epoch is 4 steps at batch 256 and 26 at batch 32. |
| Stopping on a smoothed curve | Untested (E4). |
| A fixed step budget with cosine decay, no early stopping | Untested (E4). Removes the problem instead of patching it. |
| A floor scaled to the calibration length | Untested (E5). The prime suspect for CDNOW (§5.3). |
| Early stopping on a validation rollout | Untested, and the most expensive; see `docs/model-selection.md` S1 for its CDNOW risk. |
| Going back to the customer-wise split | **Rejected.** It measures generalisation across customers, not across time. |
| MAPE, bias or Spearman in the training loss | **Rejected.** They are not proper scoring rules (`docs/loss-functions.md`). |

**Recommendation.** Run E4 on electronics first, 20 replications against family T's
`archive`, then check CDNOW and gift before adopting anything. Whether family N's
published rows are regenerated under the winner is a decision for an ADR.

## 8. The worst and best runs

This section shows, configuration by configuration, the runs that forecast worst and best
across families T, T′ (= E1) and U (§4.2), then what the bad runs share.

**What a run is.** One study: one search's winner (or a pinned arm's single trial), refitted
and forecast over 200 paths. There are 920 of them. Runs are compared only within their own
panel, never across panels (`docs/statistical-protocol.md` §7).

**Why the results are split by arm and input set.** A configuration (a *cell*) is panel ×
arm × model × input set, and holds 20 runs. Results are never pooled across arms or input
sets: the arm decides which hyperparameters are searched at all, and the inputs change
which ones the search picks (§8.2, Finding 2). Comparing a `floored` run's batch with an
`archive` run's, or a label run's learning rate with a no-label run's, would compare two
different searches.

**What the tables show.** Per panel, then per arm, one table per cell with four rows: the
cell's median, its best and worst run by aggregate MAPE (how well the total is forecast),
and its worst run by Spearman (how well customers are ranked). The line above each table
says how many of the cell's 20 runs fall in the panel's worst 10% by MAPE, by |bias| and by
Spearman — the three measures that define a bad run (§8.2).

How to read a table:
- **heading**: family · model · the columns the model reads. *count* is `Transactions`;
  *week sin/cos* is `week_sin` and `week_cos` (the LSTM's week encoding); *week* is the raw
  week index ValendinLSTM embeds; *label* is the `kmeans_8` customer-cluster label (family
  U only). Every winner in a cell reads the same columns (each winner's
  `selected_features`): no feature subset was searched.
- **epoch floor** (next to the arm): `min_epochs`, the epoch before which early stopping
  cannot end training (0 = no floor). It does not force the kept weights to come from
  after it (§4.1).
- **batch, lr, wd**: the batch size, learning rate and weight decay the run trained with.
  In `archive` and `floor50` Optuna chose them; `paper`, `paper90` and `floored` pin batch
  32, lr 1e-3 and wd 0, and the LSTM to 128/128 with dropout 0.
- **hidden, dense, dropout**: the LSTM's memory width, the dense layer's width and the
  dropout rate. Values marked `*` are ValendinLSTM's frozen published architecture
  (128/128, no dropout, ADR-0004); only its batch, lr and wd were ever searched.
- **median of 20**: each metric's median over the cell's runs. For the settings, the most
  common batch (with how many of the 20 runs used it, when not all), the most common
  widths and the median of the rest.
- **best epoch**: the epoch whose weights were kept; the model saw the data `best epoch + 1`
  times.
- **updates**: `ceil(N / batch) × (best epoch + 1)` (§1). Read it with **batch**: the same
  number of updates means eight times more data at batch 256 than at batch 32 (§1.1).
- **val CE**: the run's validation loss, the number the search selected on.
- **CV**: the standard deviation of the per-customer predicted holdout totals divided by
  their mean. Near 0 means every customer gets nearly the same forecast: a collapse (§5.2).

*(`.scratch/training-budget/worst_best_runs.py` → `results/worst_best_runs.csv`, recomputed
metrics matching the archived `factorial.csv` exactly; `per_arm_tables.py` prints the
tables.)*

### 8.1 The tables

#### electronics (320 runs; worst 10% = 32)

**`archive`** (epoch floor 0)

*U · LSTM · inputs: count, week sin/cos, label* — in the panel's worst 10%: 1 by MAPE, 1 by |bias|, 0 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 64 (9) | 1.4e-3 | 1e-4 | 128 | 128 | 0.29 | 13 | 69 | 0.0844 | +15.8 | 0.3769 | 53.9 | +0.297 | 1.16 |
| best MAPE | 64 | 1.4e-3 | 6e-6 | 64 | 32 | 0.39 | 24 | 325 | 0.0832 | -0.9 | 0.3768 | 42.6 | +0.312 | 1.38 |
| worst MAPE | 64 | 5.1e-4 | 3e-3 | 128 | 128 | 0.10 | 38 | 507 | 0.0824 | +69.8 | 0.3793 | 77.1 | +0.289 | 1.15 |
| worst Spearman | 128 | 2.7e-3 | 5e-6 | 128 | 128 | 0.27 | 4 | 35 | 0.0874 | +27.2 | 0.3775 | 56.9 | +0.138 | 1.16 |

*U · LSTM · inputs: count, week sin/cos* — in the panel's worst 10%: 0 by MAPE, 0 by |bias|, 4 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 256 (18) | 2.1e-3 | 1e-4 | 128 | 128 | 0.21 | 8 | 36 | 0.0890 | +20.3 | 0.3768 | 53.4 | +0.028 | 0.08 |
| best MAPE | 256 | 2.6e-4 | 2e-3 | 64 | 128 | 0.05 | 24 | 100 | 0.0892 | -9.4 | 0.3764 | 47.0 | +0.068 | 0.09 |
| worst MAPE | 256 | 2.1e-3 | 3e-6 | 128 | 128 | 0.08 | 8 | 36 | 0.0887 | +55.4 | 0.3779 | 73.0 | -0.003 | 0.08 |
| worst Spearman | 256 | 2.7e-3 | 1e-5 | 128 | 32 | 0.18 | 8 | 36 | 0.0891 | +10.6 | 0.3768 | 50.1 | -0.058 | 0.09 |

*U · ValendinLSTM · inputs: count, week, label* — in the panel's worst 10%: 7 by MAPE, 8 by |bias|, 0 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 64 (17) | 1.1e-3 | 5e-5 | 128* | 128* | 0* | 29 | 383 | 0.0808 | +33.4 | 0.3786 | 59.2 | +0.303 | 1.31 |
| best MAPE | 64 | 2.0e-3 | 2e-4 | 128* | 128* | 0* | 26 | 351 | 0.0808 | -10.2 | 0.3773 | 38.9 | +0.316 | 1.64 |
| worst MAPE | 64 | 1.6e-4 | 9e-5 | 128* | 128* | 0* | 89 | 1,170 | 0.0817 | +140.8 | 0.3896 | 144.1 | +0.274 | 1.23 |
| worst Spearman | 64 | 5.8e-4 | 9e-6 | 128* | 128* | 0* | 35 | 468 | 0.0807 | +134.1 | 0.3909 | 137.7 | +0.270 | 1.19 |

*U · ValendinLSTM · inputs: count, week* — in the panel's worst 10%: 4 by MAPE, 1 by |bias|, 5 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 256 (19) | 1.9e-3 | 7e-5 | 128* | 128* | 0* | 8 | 36 | 0.0888 | +46.7 | 0.3774 | 68.8 | +0.024 | 0.08 |
| best MAPE | 256 | 2.6e-3 | 5e-3 | 128* | 128* | 0* | 7 | 32 | 0.0883 | +27.8 | 0.3769 | 50.3 | +0.003 | 0.08 |
| worst MAPE | 256 | 2.5e-3 | 4e-5 | 128* | 128* | 0* | 6 | 28 | 0.0891 | +55.1 | 0.3782 | 89.5 | +0.049 | 0.08 |
| worst Spearman | 256 | 2.5e-3 | 1e-6 | 128* | 128* | 0* | 7 | 32 | 0.0889 | +48.9 | 0.3774 | 73.5 | -0.049 | 0.08 |

*T · LSTM · inputs: count, week sin/cos* — in the panel's worst 10%: 0 by MAPE, 0 by |bias|, 3 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 256 (17) | 2.1e-3 | 2e-4 | 128 | 128 | 0.19 | 8 | 36 | 0.0890 | +32.6 | 0.3770 | 58.1 | +0.028 | 0.08 |
| best MAPE | 256 | 1.7e-3 | 1e-4 | 32 | 64 | 0.05 | 10 | 44 | 0.0885 | -2.0 | 0.3767 | 47.6 | +0.058 | 0.09 |
| worst MAPE | 256 | 2.5e-3 | 6e-5 | 128 | 128 | 0.03 | 6 | 28 | 0.0889 | +53.3 | 0.3778 | 71.8 | -0.011 | 0.08 |
| worst Spearman | 256 | 2.2e-3 | 2e-4 | 128 | 64 | 0.06 | 8 | 36 | 0.0891 | +39.8 | 0.3770 | 62.4 | -0.043 | 0.08 |

*T · ValendinLSTM · inputs: count, week* — in the panel's worst 10%: 6 by MAPE, 4 by |bias|, 7 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 256 (17) | 1.9e-3 | 4e-5 | 128* | 128* | 0* | 8 | 36 | 0.0889 | +42.6 | 0.3775 | 70.4 | +0.018 | 0.08 |
| best MAPE | 64 | 2.6e-3 | 7e-5 | 128* | 128* | 0* | 35 | 468 | 0.0895 | +16.3 | 0.3765 | 50.8 | +0.138 | 0.22 |
| worst MAPE | 256 | 8.3e-4 | 5e-3 | 128* | 128* | 0* | 13 | 56 | 0.0891 | +64.6 | 0.3784 | 90.9 | -0.002 | 0.07 |
| worst Spearman | 256 | 1.7e-3 | 3e-6 | 128* | 128* | 0* | 9 | 40 | 0.0886 | +48.9 | 0.3778 | 78.3 | -0.028 | 0.08 |

**`floor50`** (epoch floor 50)

*T · LSTM · inputs: count, week sin/cos* — in the panel's worst 10%: 0 by MAPE, 0 by |bias|, 1 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 256 (18) | 7.5e-4 | 1e-4 | 64 | 128 | 0.22 | 26 | 108 | 0.0881 | -2.0 | 0.3765 | 47.9 | +0.083 | 0.10 |
| best MAPE | 128 | 2.5e-3 | 4e-6 | 128 | 64 | 0.01 | 65 | 462 | 0.0881 | -21.9 | 0.3762 | 43.0 | +0.123 | 0.27 |
| worst MAPE | 256 | 1.4e-3 | 2e-6 | 128 | 128 | 0.37 | 48 | 196 | 0.0880 | +46.1 | 0.3775 | 73.1 | +0.000 | 0.08 |
| worst Spearman | 256 | 2.5e-3 | 2e-4 | 128 | 128 | 0.16 | 38 | 156 | 0.0884 | +16.8 | 0.3769 | 52.4 | -0.043 | 0.09 |

*T · ValendinLSTM · inputs: count, week* — in the panel's worst 10%: 0 by MAPE, 0 by |bias|, 2 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 256 (14) | 2.6e-3 | 2e-4 | 128* | 128* | 0* | 67 | 328 | 0.0861 | +7.8 | 0.3764 | 48.6 | +0.189 | 0.17 |
| best MAPE | 128 | 2.4e-3 | 4e-5 | 128* | 128* | 0* | 76 | 539 | 0.0870 | -18.3 | 0.3761 | 37.7 | +0.202 | 0.20 |
| worst MAPE | 256 | 2.4e-3 | 2e-5 | 128* | 128* | 0* | 67 | 272 | 0.0856 | +48.2 | 0.3772 | 59.1 | +0.169 | 0.21 |
| worst Spearman | 256 | 2.1e-3 | 9e-3 | 128* | 128* | 0* | 57 | 232 | 0.0876 | +21.1 | 0.3768 | 49.6 | -0.017 | 0.09 |

**`paper`** (epoch floor 0)

*T · LSTM · inputs: count, week sin/cos* — in the panel's worst 10%: 1 by MAPE, 2 by |bias|, 5 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 1 | 52 | 0.0930 | +36.1 | 0.3772 | 62.0 | +0.017 | 0.08 |
| best MAPE | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 1 | 52 | 0.0933 | +6.7 | 0.3766 | 51.7 | -0.009 | 0.09 |
| worst MAPE | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 1 | 52 | 0.0930 | +68.1 | 0.3780 | 83.2 | +0.013 | 0.07 |
| worst Spearman | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 1 | 52 | 0.0928 | +32.0 | 0.3769 | 60.7 | -0.031 | 0.08 |

*T · ValendinLSTM · inputs: count, week* — in the panel's worst 10%: 2 by MAPE, 0 by |bias|, 5 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 1 | 52 | 0.0931 | +30.6 | 0.3773 | 68.9 | +0.014 | 0.08 |
| best MAPE | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 1 | 52 | 0.0930 | +7.3 | 0.3767 | 55.4 | -0.016 | 0.09 |
| worst MAPE | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 1 | 52 | 0.0931 | +38.2 | 0.3772 | 77.1 | +0.001 | 0.08 |
| worst Spearman | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 1 | 52 | 0.0935 | +30.5 | 0.3777 | 74.9 | -0.030 | 0.08 |

**`paper90`** (epoch floor 90)

*T · LSTM · inputs: count, week sin/cos* — in the panel's worst 10%: 2 by MAPE, 4 by |bias|, 0 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 71.5 | 1,885 | 0.0891 | +30.4 | 0.3766 | 49.6 | +0.185 | 0.22 |
| best MAPE | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 82 | 2,158 | 0.0888 | -10.4 | 0.3761 | 33.8 | +0.188 | 0.32 |
| worst MAPE | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 74 | 1,950 | 0.0891 | +98.7 | 0.3786 | 103.4 | +0.235 | 0.18 |
| worst Spearman | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 1 | 52 | 0.0925 | +40.6 | 0.3774 | 63.5 | +0.021 | 0.08 |

*T · ValendinLSTM · inputs: count, week* — in the panel's worst 10%: 0 by MAPE, 0 by |bias|, 0 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 56 | 1,482 | 0.0887 | -10.6 | 0.3763 | 45.3 | +0.198 | 0.26 |
| best MAPE | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 71 | 1,872 | 0.0885 | +7.3 | 0.3761 | 37.3 | +0.241 | 0.43 |
| worst MAPE | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 80 | 2,106 | 0.0884 | +38.7 | 0.3773 | 62.8 | +0.197 | 0.20 |
| worst Spearman | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 1 | 52 | 0.0935 | +27.9 | 0.3769 | 61.7 | +0.008 | 0.08 |

**`floored`** (epoch floor 90)

*U · LSTM · inputs: count, week sin/cos, label* — in the panel's worst 10%: 4 by MAPE, 5 by |bias|, 0 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 26.5 | 715 | 0.0824 | +2.9 | 0.3770 | 43.9 | +0.306 | 1.38 |
| best MAPE | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 32 | 858 | 0.0823 | +11.9 | 0.3781 | 36.4 | +0.320 | 1.56 |
| worst MAPE | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 14 | 390 | 0.0832 | +83.0 | 0.3827 | 93.5 | +0.299 | 1.37 |
| worst Spearman | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 9 | 260 | 0.0912 | -21.2 | 0.3764 | 45.0 | +0.234 | 0.75 |

*U · LSTM · inputs: count, week sin/cos* — in the panel's worst 10%: 1 by MAPE, 2 by |bias|, 0 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 71.5 | 1,885 | 0.0891 | +26.2 | 0.3765 | 45.9 | +0.197 | 0.22 |
| best MAPE | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 82 | 2,158 | 0.0888 | -10.7 | 0.3761 | 33.9 | +0.258 | 0.31 |
| worst MAPE | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 82 | 2,158 | 0.0890 | +89.1 | 0.3778 | 91.8 | +0.233 | 0.58 |
| worst Spearman | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 16 | 442 | 0.0927 | +35.1 | 0.3769 | 59.1 | +0.013 | 0.08 |

*U · ValendinLSTM · inputs: count, week, label* — in the panel's worst 10%: 4 by MAPE, 5 by |bias|, 0 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 33 | 884 | 0.0814 | +41.9 | 0.3795 | 59.2 | +0.309 | 1.35 |
| best MAPE | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 17 | 468 | 0.0839 | -8.3 | 0.3767 | 38.1 | +0.292 | 1.37 |
| worst MAPE | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 43 | 1,144 | 0.0809 | +73.4 | 0.3872 | 87.3 | +0.329 | 1.80 |
| worst Spearman | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 42 | 1,118 | 0.0816 | +3.1 | 0.3772 | 41.7 | +0.282 | 1.42 |

*U · ValendinLSTM · inputs: count, week* — in the panel's worst 10%: 0 by MAPE, 0 by |bias|, 0 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 58.5 | 1,547 | 0.0886 | -3.4 | 0.3762 | 45.3 | +0.176 | 0.25 |
| best MAPE | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 71 | 1,872 | 0.0885 | +6.2 | 0.3761 | 36.4 | +0.250 | 0.44 |
| worst MAPE | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 80 | 2,106 | 0.0884 | +39.8 | 0.3771 | 63.0 | +0.162 | 0.21 |
| worst Spearman | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 1 | 52 | 0.0935 | +27.5 | 0.3774 | 61.1 | +0.014 | 0.08 |

#### multichannel (160 runs; worst 10% = 16)

**`archive`** (epoch floor 0)

*U · LSTM · inputs: count, week sin/cos, label* — in the panel's worst 10%: 1 by MAPE, 1 by |bias|, 0 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 64 (17) | 1.9e-3 | 5e-5 | 128 | 32 | 0.24 | 19 | 407 | 0.0189 | +27.0 | 0.0571 | 54.9 | +0.174 | 1.94 |
| best MAPE | 64 | 2.3e-3 | 3e-6 | 128 | 128 | 0.24 | 8 | 198 | 0.0189 | -10.7 | 0.0569 | 48.7 | +0.173 | 2.13 |
| worst MAPE | 256 | 1.8e-3 | 2e-5 | 128 | 32 | 0.15 | 3 | 24 | 0.0225 | +297.9 | 0.0582 | 297.9 | +0.159 | 0.42 |
| worst Spearman | 64 | 2.5e-4 | 8e-5 | 32 | 64 | 0.33 | 74 | 1,650 | 0.0223 | +29.9 | 0.0570 | 60.2 | +0.101 | 0.82 |

*U · LSTM · inputs: count, week sin/cos* — in the panel's worst 10%: 13 by MAPE, 11 by |bias|, 7 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 256 (19) | 1.9e-3 | 1e-5 | 128 | 64 | 0.17 | 4 | 30 | 0.0228 | +109.4 | 0.0573 | 134.8 | +0.002 | 0.12 |
| best MAPE | 64 | 2.8e-3 | 2e-6 | 32 | 64 | 0.06 | 11 | 264 | 0.0241 | +9.0 | 0.0570 | 50.7 | +0.012 | 0.17 |
| worst MAPE | 256 | 1.8e-3 | 2e-5 | 128 | 64 | 0.34 | 3 | 24 | 0.0228 | +251.9 | 0.0579 | 254.8 | +0.010 | 0.09 |
| worst Spearman | 256 | 1.9e-3 | 3e-5 | 128 | 64 | 0.23 | 3 | 24 | 0.0228 | +121.1 | 0.0573 | 134.5 | -0.041 | 0.12 |

*U · ValendinLSTM · inputs: count, week, label* — in the panel's worst 10%: 1 by MAPE, 1 by |bias|, 0 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 64 | 1.5e-3 | 2e-5 | 128* | 128* | 0* | 13 | 308 | 0.0188 | +32.4 | 0.0572 | 61.6 | +0.180 | 1.96 |
| best MAPE | 64 | 2.2e-3 | 4e-4 | 128* | 128* | 0* | 12 | 286 | 0.0188 | -12.2 | 0.0568 | 43.7 | +0.185 | 2.04 |
| worst MAPE | 64 | 8.8e-4 | 1e-6 | 128* | 128* | 0* | 13 | 308 | 0.0191 | +130.0 | 0.0599 | 141.2 | +0.061 | 1.41 |
| worst Spearman | 64 | 8.8e-4 | 1e-6 | 128* | 128* | 0* | 13 | 308 | 0.0191 | +130.0 | 0.0599 | 141.2 | +0.061 | 1.41 |

*U · ValendinLSTM · inputs: count, week* — in the panel's worst 10%: 0 by MAPE, 2 by |bias|, 8 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 256 | 1.7e-3 | 1e-4 | 128* | 128* | 0* | 3 | 24 | 0.0228 | +45.2 | 0.0571 | 80.8 | -0.004 | 0.15 |
| best MAPE | 256 | 1.1e-3 | 4e-4 | 128* | 128* | 0* | 4 | 30 | 0.0228 | +21.5 | 0.0570 | 59.7 | +0.027 | 0.16 |
| worst MAPE | 256 | 7.8e-4 | 1e-6 | 128* | 128* | 0* | 6 | 42 | 0.0228 | +108.0 | 0.0572 | 123.5 | +0.036 | 0.12 |
| worst Spearman | 256 | 2.1e-3 | 6e-5 | 128* | 128* | 0* | 5 | 36 | 0.0228 | +0.2 | 0.0569 | 63.1 | -0.067 | 0.17 |

**`floored`** (epoch floor 90)

*U · LSTM · inputs: count, week sin/cos, label* — in the panel's worst 10%: 0 by MAPE, 0 by |bias|, 0 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 25 | 1,144 | 0.0193 | +20.1 | 0.0571 | 52.3 | +0.197 | 2.13 |
| best MAPE | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 31 | 1,408 | 0.0192 | -2.9 | 0.0569 | 46.4 | +0.176 | 2.56 |
| worst MAPE | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 4 | 220 | 0.0198 | +71.8 | 0.0576 | 90.7 | +0.081 | 1.75 |
| worst Spearman | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 4 | 220 | 0.0198 | +71.8 | 0.0576 | 90.7 | +0.081 | 1.75 |

*U · LSTM · inputs: count, week sin/cos* — in the panel's worst 10%: 0 by MAPE, 0 by |bias|, 1 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 80.5 | 3,586 | 0.0234 | +17.3 | 0.0569 | 51.9 | +0.084 | 0.38 |
| best MAPE | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 89 | 3,960 | 0.0236 | -13.6 | 0.0569 | 43.4 | +0.023 | 0.22 |
| worst MAPE | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 71 | 3,168 | 0.0233 | +52.4 | 0.0569 | 67.1 | +0.083 | 0.40 |
| worst Spearman | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 79 | 3,520 | 0.0239 | +24.9 | 0.0570 | 53.6 | -0.036 | 0.16 |

*U · ValendinLSTM · inputs: count, week, label* — in the panel's worst 10%: 1 by MAPE, 1 by |bias|, 0 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 28 | 1,276 | 0.0194 | +12.4 | 0.0572 | 55.2 | +0.193 | 2.19 |
| best MAPE | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 33 | 1,496 | 0.0192 | +1.1 | 0.0570 | 49.0 | +0.232 | 2.10 |
| worst MAPE | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 17 | 792 | 0.0195 | +127.8 | 0.0585 | 135.2 | +0.177 | 1.77 |
| worst Spearman | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 6 | 308 | 0.0196 | +1.6 | 0.0571 | 55.8 | +0.160 | 1.94 |

*U · ValendinLSTM · inputs: count, week* — in the panel's worst 10%: 0 by MAPE, 0 by |bias|, 0 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 72.5 | 3,234 | 0.0234 | +20.8 | 0.0569 | 53.0 | +0.129 | 0.51 |
| best MAPE | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 81 | 3,608 | 0.0235 | +19.3 | 0.0568 | 48.5 | +0.138 | 0.67 |
| worst MAPE | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 50 | 2,244 | 0.0232 | +55.3 | 0.0569 | 70.2 | +0.123 | 0.67 |
| worst Spearman | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 75 | 3,344 | 0.0234 | -2.6 | 0.0569 | 49.1 | +0.043 | 0.27 |

#### cdnow (280 runs; worst 10% = 28)

**`archive`** (epoch floor 0)

*U · LSTM · inputs: count, week sin/cos, label* — in the panel's worst 10%: 4 by MAPE, 4 by |bias|, 1 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 64 (19) | 1.9e-3 | 6e-5 | 128 | 32 | 0.17 | 26 | 999 | 0.0681 | +54.5 | 0.1538 | 59.7 | +0.398 | 2.05 |
| best MAPE | 64 | 1.8e-3 | 5e-5 | 128 | 64 | 0.05 | 27 | 1,036 | 0.0678 | +2.3 | 0.1495 | 21.8 | +0.394 | 2.64 |
| worst MAPE | 64 | 1.8e-3 | 2e-5 | 64 | 64 | 0.28 | 26 | 999 | 0.0690 | +190.0 | 0.1669 | 190.0 | +0.364 | 0.97 |
| worst Spearman | 64 | 2.9e-3 | 5e-4 | 128 | 32 | 0.40 | 31 | 1,184 | 0.0662 | +129.4 | 0.1627 | 131.5 | +0.305 | 1.38 |

*U · LSTM · inputs: count, week sin/cos* — in the panel's worst 10%: 1 by MAPE, 1 by |bias|, 3 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 64 (19) | 2.2e-3 | 1e-5 | 128 | 64 | 0.11 | 24.5 | 925 | 0.0912 | -3.2 | 0.1477 | 39.9 | +0.399 | 0.97 |
| best MAPE | 64 | 8.8e-4 | 1e-5 | 128 | 64 | 0.12 | 35 | 1,332 | 0.0913 | +4.0 | 0.1464 | 21.8 | +0.348 | 1.21 |
| worst MAPE | 128 | 2.3e-3 | 4e-4 | 128 | 128 | 0.09 | 29 | 570 | 0.0920 | +212.0 | 0.1621 | 222.4 | +0.397 | 0.12 |
| worst Spearman | 64 | 2.7e-3 | 2e-3 | 64 | 128 | 0.31 | 52 | 1,961 | 0.0911 | +113.4 | 0.1559 | 123.6 | +0.222 | 0.26 |

*U · ValendinLSTM · inputs: count, week, label* — in the panel's worst 10%: 1 by MAPE, 1 by |bias|, 3 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 64 | 2.2e-3 | 3e-5 | 128* | 128* | 0* | 21 | 814 | 0.0687 | +29.2 | 0.1528 | 42.5 | +0.425 | 2.35 |
| best MAPE | 64 | 1.0e-3 | 5e-4 | 128* | 128* | 0* | 51 | 1,924 | 0.0682 | -3.6 | 0.1492 | 29.4 | +0.470 | 2.62 |
| worst MAPE | 64 | 2.8e-3 | 2e-5 | 128* | 128* | 0* | 20 | 777 | 0.0693 | +260.0 | 0.2157 | 269.3 | +0.209 | 1.25 |
| worst Spearman | 64 | 2.8e-3 | 2e-5 | 128* | 128* | 0* | 20 | 777 | 0.0693 | +260.0 | 0.2157 | 269.3 | +0.209 | 1.25 |

*U · ValendinLSTM · inputs: count, week* — in the panel's worst 10%: 1 by MAPE, 1 by |bias|, 3 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 64 | 1.8e-3 | 2e-5 | 128* | 128* | 0* | 19.5 | 758 | 0.0912 | -24.4 | 0.1473 | 45.1 | +0.383 | 1.18 |
| best MAPE | 64 | 2.0e-3 | 6e-6 | 128* | 128* | 0* | 20 | 777 | 0.0910 | -7.8 | 0.1461 | 22.8 | +0.418 | 1.21 |
| worst MAPE | 64 | 1.8e-3 | 2e-5 | 128* | 128* | 0* | 14 | 555 | 0.0913 | +204.8 | 0.1640 | 209.7 | +0.416 | 0.56 |
| worst Spearman | 64 | 2.4e-3 | 8e-3 | 128* | 128* | 0* | 13 | 518 | 0.0916 | +17.3 | 0.1513 | 75.8 | +0.029 | 0.41 |

*T′ · LSTM · inputs: count* — in the panel's worst 10%: 0 by MAPE, 0 by |bias|, 0 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 64 | 2.1e-3 | 7e-5 | 128 | 128 | 0.20 | 25 | 962 | 0.0909 | -1.9 | 0.1463 | 20.2 | +0.428 | 1.10 |
| best MAPE | 64 | 2.9e-3 | 9e-6 | 128 | 64 | 0.22 | 22 | 851 | 0.0910 | -1.5 | 0.1462 | 18.4 | +0.451 | 1.38 |
| worst MAPE | 64 | 1.7e-3 | 2e-6 | 128 | 128 | 0.28 | 17 | 666 | 0.0910 | -31.1 | 0.1465 | 31.8 | +0.416 | 1.29 |
| worst Spearman | 64 | 8.1e-4 | 5e-3 | 128 | 128 | 0.40 | 27 | 1,036 | 0.0909 | -26.4 | 0.1459 | 27.7 | +0.402 | 1.62 |

*T′ · ValendinLSTM · inputs: count, week* — in the panel's worst 10%: 1 by MAPE, 1 by |bias|, 1 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 64 | 1.9e-3 | 3e-5 | 128* | 128* | 0* | 19.5 | 758 | 0.0913 | -20.4 | 0.1471 | 35.3 | +0.391 | 1.08 |
| best MAPE | 64 | 2.2e-3 | 1e-3 | 128* | 128* | 0* | 11 | 444 | 0.0912 | -5.0 | 0.1464 | 24.1 | +0.387 | 1.04 |
| worst MAPE | 64 | 2.8e-3 | 8e-6 | 128* | 128* | 0* | 23 | 888 | 0.0916 | +166.8 | 0.1568 | 169.7 | +0.441 | 0.75 |
| worst Spearman | 64 | 1.8e-3 | 2e-3 | 128* | 128* | 0* | 12 | 481 | 0.0911 | -55.6 | 0.1482 | 56.9 | +0.330 | 1.03 |

**`floor50`** (epoch floor 50)

*T′ · LSTM · inputs: count* — in the panel's worst 10%: 0 by MAPE, 0 by |bias|, 0 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 64 | 2.3e-3 | 2e-5 | 128 | 128 | 0.13 | 45 | 1,702 | 0.0906 | +3.5 | 0.1459 | 21.7 | +0.432 | 1.32 |
| best MAPE | 64 | 2.2e-3 | 7e-6 | 64 | 128 | 0.00 | 31 | 1,184 | 0.0906 | -7.5 | 0.1463 | 18.7 | +0.448 | 1.05 |
| worst MAPE | 64 | 2.6e-3 | 1e-5 | 64 | 32 | 0.11 | 36 | 1,369 | 0.0906 | +54.3 | 0.1475 | 59.0 | +0.423 | 0.84 |
| worst Spearman | 64 | 1.4e-3 | 8e-4 | 64 | 128 | 0.33 | 46 | 1,739 | 0.0907 | -21.9 | 0.1461 | 24.9 | +0.393 | 1.36 |

*T′ · ValendinLSTM · inputs: count, week* — in the panel's worst 10%: 0 by MAPE, 0 by |bias|, 5 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 64 | 1.5e-3 | 2e-4 | 128* | 128* | 0* | 30.5 | 1,165 | 0.0910 | -18.3 | 0.1472 | 35.2 | +0.400 | 1.22 |
| best MAPE | 64 | 1.3e-3 | 7e-4 | 128* | 128* | 0* | 33 | 1,258 | 0.0911 | -13.6 | 0.1480 | 24.7 | +0.380 | 2.35 |
| worst MAPE | 64 | 2.6e-3 | 1e-4 | 128* | 128* | 0* | 24 | 925 | 0.0910 | +48.9 | 0.1591 | 91.7 | -0.150 | 0.60 |
| worst Spearman | 64 | 2.6e-3 | 1e-4 | 128* | 128* | 0* | 24 | 925 | 0.0910 | +48.9 | 0.1591 | 91.7 | -0.150 | 0.60 |

**`paper`** (epoch floor 0)

*T′ · LSTM · inputs: count* — in the panel's worst 10%: 0 by MAPE, 0 by |bias|, 1 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 12 | 962 | 0.0916 | +1.6 | 0.1465 | 22.3 | +0.411 | 1.04 |
| best MAPE | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 11 | 888 | 0.0918 | -6.2 | 0.1465 | 18.5 | +0.434 | 0.95 |
| worst MAPE | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 18 | 1,406 | 0.0916 | +38.0 | 0.1468 | 41.3 | +0.429 | 0.89 |
| worst Spearman | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 24 | 1,850 | 0.0913 | -0.7 | 0.1469 | 19.5 | +0.323 | 0.73 |

*T′ · ValendinLSTM · inputs: count, week* — in the panel's worst 10%: 0 by MAPE, 0 by |bias|, 5 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 13 | 1,036 | 0.0931 | -10.5 | 0.1474 | 43.0 | +0.378 | 0.96 |
| best MAPE | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 9 | 740 | 0.0923 | -7.2 | 0.1467 | 24.2 | +0.449 | 0.98 |
| worst MAPE | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 13 | 1,036 | 0.0933 | +63.4 | 0.1563 | 108.4 | +0.375 | 0.33 |
| worst Spearman | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 13 | 1,036 | 0.0934 | -56.5 | 0.1486 | 57.4 | +0.198 | 0.78 |

**`floored`** (epoch floor 90)

*U · LSTM · inputs: count, week sin/cos, label* — in the panel's worst 10%: 3 by MAPE, 4 by |bias|, 0 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 30 | 2,294 | 0.0693 | +73.4 | 0.1586 | 77.8 | +0.405 | 1.70 |
| best MAPE | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 13 | 1,036 | 0.0689 | -5.7 | 0.1493 | 21.9 | +0.397 | 2.82 |
| worst MAPE | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 31 | 2,368 | 0.0692 | +174.2 | 0.1680 | 174.2 | +0.359 | 1.15 |
| worst Spearman | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 27 | 2,072 | 0.0717 | +116.4 | 0.1609 | 119.5 | +0.348 | 1.45 |

*U · LSTM · inputs: count, week sin/cos* — in the panel's worst 10%: 15 by MAPE, 14 by |bias|, 4 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 18.5 | 1,443 | 0.0926 | +155.4 | 0.1568 | 160.7 | +0.405 | 0.19 |
| best MAPE | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 16 | 1,258 | 0.0919 | +5.0 | 0.1470 | 22.6 | +0.385 | 1.08 |
| worst MAPE | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 27 | 2,072 | 0.0927 | +347.2 | 0.1760 | 348.2 | +0.150 | 0.10 |
| worst Spearman | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 46 | 3,478 | 0.0915 | +198.7 | 0.1589 | 203.7 | -0.045 | 0.17 |

*U · ValendinLSTM · inputs: count, week, label* — in the panel's worst 10%: 0 by MAPE, 0 by |bias|, 0 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 24.5 | 1,887 | 0.0715 | +33.6 | 0.1535 | 43.9 | +0.404 | 2.42 |
| best MAPE | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 20 | 1,554 | 0.0730 | +0.8 | 0.1493 | 23.8 | +0.447 | 2.52 |
| worst MAPE | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 25 | 1,924 | 0.0712 | +105.0 | 0.1635 | 115.4 | +0.395 | 1.53 |
| worst Spearman | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 8 | 666 | 0.0707 | +65.1 | 0.1573 | 88.6 | +0.351 | 1.21 |

*U · ValendinLSTM · inputs: count, week* — in the panel's worst 10%: 2 by MAPE, 2 by |bias|, 2 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 23 | 1,776 | 0.0927 | -7.9 | 0.1474 | 43.4 | +0.387 | 0.99 |
| best MAPE | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 27 | 2,072 | 0.0943 | +1.0 | 0.1463 | 25.7 | +0.431 | 1.01 |
| worst MAPE | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 44 | 3,330 | 0.0926 | +199.8 | 0.1634 | 212.3 | +0.407 | 0.28 |
| worst Spearman | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 13 | 1,036 | 0.0934 | -50.2 | 0.1485 | 52.3 | +0.232 | 0.70 |

#### gift (160 runs; worst 10% = 16)

**`archive`** (epoch floor 0)

*U · LSTM · inputs: count, week sin/cos, label* — in the panel's worst 10%: 2 by MAPE, 1 by |bias|, 0 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 64 (19) | 2.5e-3 | 3e-4 | 128 | 64 | 0.19 | 22.5 | 775 | 0.0457 | -7.3 | 0.1068 | 28.6 | +0.350 | 1.41 |
| best MAPE | 64 | 2.0e-3 | 2e-5 | 128 | 64 | 0.02 | 31 | 1,056 | 0.0458 | -6.6 | 0.1068 | 20.6 | +0.343 | 1.37 |
| worst MAPE | 128 | 1.9e-3 | 2e-4 | 128 | 128 | 0.25 | 17 | 306 | 0.0482 | -31.6 | 0.1067 | 44.0 | +0.369 | 1.58 |
| worst Spearman | 64 | 1.1e-3 | 2e-6 | 128 | 32 | 0.03 | 12 | 429 | 0.0460 | +1.4 | 0.1068 | 29.3 | +0.335 | 1.33 |

*U · LSTM · inputs: count, week sin/cos* — in the panel's worst 10%: 2 by MAPE, 4 by |bias|, 3 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 64 | 2.7e-3 | 6e-5 | 128 | 128 | 0.18 | 34 | 1,155 | 0.0516 | -23.4 | 0.1066 | 28.6 | +0.349 | 0.58 |
| best MAPE | 64 | 2.7e-3 | 1e-5 | 128 | 128 | 0.19 | 39 | 1,320 | 0.0515 | -11.2 | 0.1065 | 21.3 | +0.354 | 0.79 |
| worst MAPE | 64 | 2.7e-3 | 1e-5 | 64 | 128 | 0.08 | 33 | 1,122 | 0.0514 | -45.2 | 0.1067 | 45.9 | +0.332 | 0.68 |
| worst Spearman | 64 | 3.0e-3 | 4e-3 | 32 | 64 | 0.11 | 39 | 1,320 | 0.0525 | -6.3 | 0.1070 | 22.7 | +0.104 | 0.11 |

*U · ValendinLSTM · inputs: count, week, label* — in the panel's worst 10%: 4 by MAPE, 2 by |bias|, 0 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 64 | 1.4e-3 | 3e-5 | 128* | 128* | 0* | 19 | 660 | 0.0451 | -0.7 | 0.1069 | 28.8 | +0.364 | 1.31 |
| best MAPE | 64 | 9.5e-4 | 2e-6 | 128* | 128* | 0* | 33 | 1,122 | 0.0448 | -9.3 | 0.1068 | 21.6 | +0.327 | 1.43 |
| worst MAPE | 64 | 6.2e-4 | 5e-3 | 128* | 128* | 0* | 7 | 264 | 0.0450 | +37.8 | 0.1075 | 50.2 | +0.372 | 1.29 |
| worst Spearman | 64 | 9.5e-4 | 2e-6 | 128* | 128* | 0* | 33 | 1,122 | 0.0448 | -9.3 | 0.1068 | 21.6 | +0.327 | 1.43 |

*U · ValendinLSTM · inputs: count, week* — in the panel's worst 10%: 3 by MAPE, 5 by |bias|, 1 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 64 | 2.2e-3 | 1e-4 | 128* | 128* | 0* | 26.5 | 907 | 0.0514 | -12.7 | 0.1067 | 30.5 | +0.353 | 0.57 |
| best MAPE | 64 | 2.6e-3 | 4e-3 | 128* | 128* | 0* | 18 | 627 | 0.0515 | +2.3 | 0.1064 | 24.3 | +0.384 | 0.62 |
| worst MAPE | 64 | 2.3e-3 | 5e-4 | 128* | 128* | 0* | 22 | 759 | 0.0512 | -50.8 | 0.1067 | 51.3 | +0.350 | 0.76 |
| worst Spearman | 64 | 2.2e-3 | 4e-3 | 128* | 128* | 0* | 41 | 1,386 | 0.0515 | -11.2 | 0.1068 | 28.1 | +0.252 | 0.28 |

**`floored`** (epoch floor 90)

*U · LSTM · inputs: count, week sin/cos, label* — in the panel's worst 10%: 4 by MAPE, 0 by |bias|, 0 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 20 | 1,365 | 0.0453 | -4.1 | 0.1068 | 28.1 | +0.365 | 1.34 |
| best MAPE | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 39 | 2,600 | 0.0451 | -1.4 | 0.1067 | 22.6 | +0.366 | 1.34 |
| worst MAPE | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 8 | 585 | 0.0466 | +12.6 | 0.1072 | 47.1 | +0.345 | 1.32 |
| worst Spearman | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 11 | 780 | 0.0461 | -16.4 | 0.1067 | 28.7 | +0.331 | 1.39 |

*U · LSTM · inputs: count, week sin/cos* — in the panel's worst 10%: 1 by MAPE, 2 by |bias|, 6 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 46.5 | 3,087 | 0.0510 | -17.6 | 0.1066 | 28.8 | +0.294 | 0.49 |
| best MAPE | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 22 | 1,495 | 0.0520 | -12.9 | 0.1070 | 23.3 | +0.021 | 0.11 |
| worst MAPE | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 39 | 2,600 | 0.0513 | -47.2 | 0.1068 | 48.0 | +0.159 | 0.35 |
| worst Spearman | 32 | 1.0e-3 | 0 | 128 | 128 | 0.00 | 23 | 1,560 | 0.0518 | -9.2 | 0.1070 | 27.5 | +0.014 | 0.11 |

*U · ValendinLSTM · inputs: count, week, label* — in the panel's worst 10%: 0 by MAPE, 0 by |bias|, 0 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 16 | 1,105 | 0.0451 | -8.8 | 0.1069 | 27.2 | +0.365 | 1.42 |
| best MAPE | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 20 | 1,365 | 0.0451 | -9.5 | 0.1067 | 22.5 | +0.380 | 1.47 |
| worst MAPE | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 11 | 780 | 0.0448 | +17.6 | 0.1072 | 38.7 | +0.367 | 1.22 |
| worst Spearman | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 23 | 1,560 | 0.0449 | -8.0 | 0.1069 | 29.6 | +0.337 | 1.29 |

*U · ValendinLSTM · inputs: count, week* — in the panel's worst 10%: 0 by MAPE, 2 by |bias|, 6 by Spearman

| run | batch | lr | wd | hidden | dense | dropout | best epoch | updates | val CE | bias % | RMSE | MAPE | Spearman | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **median of 20** | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 36 | 2,405 | 0.0508 | -14.3 | 0.1066 | 29.7 | +0.322 | 0.44 |
| best MAPE | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 45 | 2,990 | 0.0507 | -4.8 | 0.1069 | 23.3 | +0.189 | 0.22 |
| worst MAPE | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 28 | 1,885 | 0.0507 | -31.7 | 0.1065 | 33.7 | +0.371 | 0.70 |
| worst Spearman | 32 | 1.0e-3 | 0 | 128* | 128* | 0* | 32 | 2,145 | 0.0508 | -23.9 | 0.1070 | 29.4 | +0.023 | 0.11 |



### 8.2 What the bad runs have in common

The question here: when a run forecasts badly, is it because of its model, its
architecture or its training, and which? The answer is built in five steps: define a bad
run, measure how much of the badness the configuration explains at all, show how the
inputs change what the search picks, look inside a configuration for settings that predict
a bad run, and then describe the failure types the bad runs fall into. Every step keeps
arms and input sets apart.

All of this is descriptive, not a protocol claim: the bad runs are picked by their
outcome. Two terms:
- a **cell** is one configuration: family, panel, model and arm (and label for family U).
  It holds 20 runs that differ only in their search and training randomness;
- the **worst 10%** of a panel is its 32 (electronics), 28 (CDNOW) or 16 (multichannel,
  gift) worst runs on one metric.

*(`.scratch/training-budget/bad_run_patterns.py` prints every number below.)*

#### What counts as a bad run

A run is bad on the **total** if it is in its panel's worst 10% by MAPE or by |bias|, and
bad on the **ranking** if it is in the worst 10% by Spearman (the secondary measure).

| panel | worst 10% | shared by MAPE and \|bias\| | median bias of the worst-MAPE runs | over-forecasts among them | shared by MAPE and Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| electronics | 32 | 25 | +68% | 32 of 32 | 5 |
| multichannel | 16 | 14 | +130% | 16 of 16 | 4 |
| CDNOW | 28 | 27 | +187% | 28 of 28 | 5 |
| gift | 16 | 7 | −1% | 8 of 16 | 1 |

- **On three panels, a bad total is a total that is too high.** MAPE and |bias| pick almost
  the same runs, and every one of them over-forecasts. What is left of MAPE once the bias
  is removed, `MAPE − |bias|`, is only 4–13 points in those runs, against 15–39 in the best
  10%: the weekly shape is roughly right and the level is wrong.
- **Gift is the exception.** Its worst runs by |bias| under-forecast (15 of 16, median
  −34%), while its worst by MAPE miss in both directions. Its whole MAPE range is narrow
  (21–51).
- **A bad total and a bad ranking are mostly different runs** (last column). A run can rank
  customers well and still miss the total by 140% (the worst MAPE row of electronics' `U · ValendinLSTM` with the label in §8.1), or
  get the total within 10% and rank customers at chance. So the two are analysed
  separately below.
- RMSE follows the bias, but only in the third decimal, so it separates nothing
  (`docs/statistical-protocol.md` §4).

#### Finding 1. The configuration explains at most half of a bad total

If bad runs came from bad configurations, the 20 runs of a cell would score alike and
cells would differ. The table splits each metric's variance into the part between cells
(configuration) and the part between runs of the same cell (what one search and one
training happened to produce).

| panel | MAPE | \|bias\| | Spearman |
| --- | ---: | ---: | ---: |
| electronics | 28% | 16% | 82% |
| multichannel | 45% | 33% | 84% |
| CDNOW | 46% | 45% | 12% |
| gift | 4% | 20% | 29% |

*Share of the variance that lies between cells.*

- **For the total, half or more of the spread is between runs of the same configuration.**
  On gift, almost all of it. A configuration sets a run's typical error, but where one run
  lands around that typical error is largely chance.
- The clearest case is a fully pinned recipe. `T · LSTM · paper90` on electronics fixes
  every setting, and its 20 runs include the panel's best run (MAPE 33.8, kept epoch 82)
  and its third worst (MAPE 103.4, kept epoch 74).
- **For the ranking on electronics and multichannel, the configuration decides** (82–84%).
  That is the collapse of failure type A below, which whole cells fall into.

#### Finding 2. The inputs change the hyperparameters the search picks

Family U runs every search twice, once without and once with the cluster label, on the
same panel, model and arm. Any difference between the two searches' winners comes from the
one extra input. The table shows, for every searched cell, what the search chose across its
20 runs and how the cell forecast (medians).

| panel | model | arm | inputs | batch (runs of 20) | lr, median [IQR] | dropout | best epoch | updates | MAPE | bias % | Spearman |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cdnow | T′ · LSTM | `archive` | count | 64: 20 | 2.1e-3 [1.3e-3, 2.6e-3] | 0.20 | 25 | 962 | 20.2 | -1.9 | +0.428 |
| cdnow | T′ · LSTM | `floor50` | count | 64: 20 | 2.3e-3 [2.1e-3, 2.8e-3] | 0.13 | 45 | 1,702 | 21.7 | +3.5 | +0.432 |
| cdnow | U · LSTM | `archive` | count, week sin/cos | 64: 19, 128: 1 | 2.2e-3 [1.7e-3, 2.5e-3] | 0.11 | 24.5 | 925 | 39.9 | -3.2 | +0.399 |
| cdnow | U · LSTM | `archive` | count, week sin/cos, label | 64: 19, 128: 1 | 1.9e-3 [1.4e-3, 2.4e-3] | 0.17 | 26 | 999 | 59.7 | +54.5 | +0.398 |
| cdnow | T′ · ValendinLSTM | `archive` | count, week | 64: 20 | 1.9e-3 [1.6e-3, 2.5e-3] | 0* | 19.5 | 758 | 35.3 | -20.4 | +0.391 |
| cdnow | T′ · ValendinLSTM | `floor50` | count, week | 64: 20 | 1.5e-3 [1.1e-3, 2.6e-3] | 0* | 30.5 | 1,165 | 35.2 | -18.3 | +0.400 |
| cdnow | U · ValendinLSTM | `archive` | count, week | 64: 20 | 1.8e-3 [1.5e-3, 2.3e-3] | 0* | 19.5 | 758 | 45.1 | -24.4 | +0.383 |
| cdnow | U · ValendinLSTM | `archive` | count, week, label | 64: 20 | 2.2e-3 [1.4e-3, 2.6e-3] | 0* | 21 | 814 | 42.5 | +29.2 | +0.425 |
| electronics | T · LSTM | `archive` | count, week sin/cos | 256: 17, 128: 3 | 2.1e-3 [1.7e-3, 2.5e-3] | 0.19 | 8 | 36 | 58.1 | +32.6 | +0.028 |
| electronics | T · LSTM | `floor50` | count, week sin/cos | 256: 18, 128: 2 | 7.5e-4 [4.7e-4, 2.0e-3] | 0.22 | 26 | 108 | 47.9 | -2.0 | +0.083 |
| electronics | U · LSTM | `archive` | count, week sin/cos | 256: 18, 128: 2 | 2.1e-3 [1.5e-3, 2.7e-3] | 0.21 | 8 | 36 | 53.4 | +20.3 | +0.028 |
| electronics | U · LSTM | `archive` | count, week sin/cos, label | 64: 9, 256: 9, 128: 2 | 1.4e-3 [5.4e-4, 2.6e-3] | 0.29 | 13 | 69 | 53.9 | +15.8 | +0.297 |
| electronics | T · ValendinLSTM | `archive` | count, week | 256: 17, 128: 2, 64: 1 | 1.9e-3 [1.1e-3, 2.4e-3] | 0* | 8 | 36 | 70.4 | +42.6 | +0.018 |
| electronics | T · ValendinLSTM | `floor50` | count, week | 256: 14, 128: 5, 64: 1 | 2.6e-3 [2.3e-3, 2.8e-3] | 0* | 67 | 328 | 48.6 | +7.8 | +0.189 |
| electronics | U · ValendinLSTM | `archive` | count, week | 256: 19, 128: 1 | 1.9e-3 [1.5e-3, 2.4e-3] | 0* | 8 | 36 | 68.8 | +46.7 | +0.024 |
| electronics | U · ValendinLSTM | `archive` | count, week, label | 64: 17, 256: 2, 128: 1 | 1.1e-3 [7.2e-4, 2.1e-3] | 0* | 29 | 383 | 59.2 | +33.4 | +0.303 |
| gift | U · LSTM | `archive` | count, week sin/cos | 64: 20 | 2.7e-3 [2.4e-3, 2.8e-3] | 0.18 | 34 | 1,155 | 28.6 | -23.4 | +0.349 |
| gift | U · LSTM | `archive` | count, week sin/cos, label | 64: 19, 128: 1 | 2.5e-3 [1.8e-3, 2.9e-3] | 0.19 | 22.5 | 775 | 28.6 | -7.3 | +0.350 |
| gift | U · ValendinLSTM | `archive` | count, week | 64: 20 | 2.2e-3 [1.9e-3, 2.6e-3] | 0* | 26.5 | 907 | 30.5 | -12.7 | +0.353 |
| gift | U · ValendinLSTM | `archive` | count, week, label | 64: 20 | 1.4e-3 [1.1e-3, 2.2e-3] | 0* | 19 | 660 | 28.8 | -0.7 | +0.364 |
| multichannel | U · LSTM | `archive` | count, week sin/cos | 256: 19, 64: 1 | 1.9e-3 [1.2e-3, 2.4e-3] | 0.17 | 4 | 30 | 134.8 | +109.4 | +0.002 |
| multichannel | U · LSTM | `archive` | count, week sin/cos, label | 64: 17, 128: 2, 256: 1 | 1.9e-3 [1.2e-3, 2.3e-3] | 0.24 | 19 | 407 | 54.9 | +27.0 | +0.174 |
| multichannel | U · ValendinLSTM | `archive` | count, week | 256: 20 | 1.7e-3 [1.4e-3, 2.3e-3] | 0* | 3 | 24 | 80.8 | +45.2 | -0.004 |
| multichannel | U · ValendinLSTM | `archive` | count, week, label | 64: 20 | 1.5e-3 [8.8e-4, 2.3e-3] | 0* | 13 | 308 | 61.6 | +32.4 | +0.180 |

*`0*`: ValendinLSTM has no dropout (frozen architecture).*

- **On electronics and multichannel, the label flips the batch size.** Without it, the
  search picks batch 256 in 17–20 of 20 runs, and training stops after 3–8 epochs and
  24–36 updates: the too-little-training collapse of type A below (Spearman ≈ 0). With the
  label it picks batch 64 (9 of 20 for electronics' LSTM, 17–20 otherwise), trains 13–29
  epochs and ranks customers (Spearman 0.17–0.30). On CDNOW and gift the search picks batch
  64 with or without the label.
- **With the label, the search picks a lower learning rate** in 6 of the 8 pairs
  (electronics 1.4e-3 against 2.1e-3 for the LSTM, 1.1e-3 against 1.9e-3 for ValendinLSTM;
  equal on multichannel's LSTM; higher on CDNOW's ValendinLSTM), **and more dropout** in all
  four LSTM pairs (0.29 against 0.21 on electronics, 0.24 against 0.17 on multichannel, 0.17
  against 0.11 on CDNOW, 0.19 against 0.18 on gift). The widths show no shift.
- **The label also moves the bias, in a panel-dependent direction.** On CDNOW it turns a
  near-zero or negative median bias into an over-forecast (LSTM −3.2% → +54.5%,
  ValendinLSTM −24.4% → +29.2%); on gift it pulls an under-forecast towards zero; on
  electronics and multichannel, where it prevents the collapse, it lowers the
  over-forecast.
- **The arm changes the picks too.** `floor50` keeps the batch of `archive` but, on
  electronics, its LSTM winners take a lower learning rate (7.5e-4 against 2.1e-3) and its
  ValendinLSTM winners a higher one (2.6e-3 against 1.9e-3), and both keep a much later
  epoch (26 and 67 against 8).

So a setting's effect can only be read inside one arm and one input set. That is how the
next finding and the §8.1 tables are built.

#### Finding 3. Inside a configuration, no searched setting predicts a bad run, except a low learning rate with the label

In the searched arms (`archive`, `floor50`), Optuna picks a different learning rate,
dropout and so on for each of a cell's 20 runs. If one of those settings caused bad totals,
runs that drew it would have higher MAPE. The table gives the rank correlation between each
setting and MAPE across a cell's 20 runs: the median over cells, then in brackets how many
cells pass +0.3 / −0.3, out of the cells where the setting varies. Arms and input sets are
kept apart.

| setting | `archive`, no label (12 cells) | `archive`, label (8 cells) | `floor50`, no label (4 cells) |
| --- | --- | --- | --- |
| learning rate | -0.01 (1 / 1 of 12) | -0.23 (1 / 3 of 8) | +0.05 (0 / 1 of 4) |
| batch size | +0.07 (3 / 0 of 6) | +0.38 (3 / 0 of 5) | +0.33 (1 / 0 of 2) |
| weight decay | -0.05 (0 / 1 of 12) | +0.05 (0 / 0 of 8) | +0.02 (1 / 0 of 4) |
| dropout | +0.06 (1 / 1 of 6) | +0.05 (1 / 0 of 4) | +0.22 (1 / 0 of 2) |
| hidden width | +0.13 (2 / 0 of 6) | -0.10 (0 / 0 of 4) | +0.03 (0 / 0 of 2) |
| dense width | +0.25 (2 / 1 of 6) | -0.02 (1 / 1 of 4) | -0.05 (0 / 1 of 2) |
| best epoch | -0.08 (0 / 1 of 12) | -0.21 (0 / 1 of 8) | -0.16 (1 / 1 of 4) |
| validation loss | +0.04 (2 / 1 of 12) | +0.14 (2 / 0 of 8) | +0.03 (1 / 0 of 4) |
| forecast CV (an outcome) | -0.65 (0 / 7 of 12) | -0.54 (0 / 6 of 8) | -0.42 (0 / 3 of 4) |

- **By chance alone**, with 20 runs, a correlation passes +0.3 (or −0.3) about 11% of the
  time: about 1 of 12 cells, 1 of 8, 0–1 of 4 in each direction. Without the label, every
  setting stays at that level, with medians near zero and signs that flip between cells.
- **With the label, a low learning rate goes with a worse total**: median −0.23, 3 of 8
  cells below −0.3. In 5 of the 8 label cells, the quarter of runs with the lowest learning
  rate has a higher median MAPE than the rest. The clearest is electronics'
  `U · ValendinLSTM · archive` with the label (r −0.68): its lowest-learning-rate quarter
  (≤ 7.2e-4) has median MAPE 97.3 against 53.3, and holds its two worst runs (lr 1.6e-4 and
  5.8e-4, bias +141% and +134%). Multichannel's ValendinLSTM with the label shows the same
  (75.6 against 58.4). This is descriptive, from 8 cells.
- **Batch size has a lean, but it barely varies.** The search picks the same batch for a
  median of 19 of a cell's 20 runs. The odd run on a larger batch tended to be worse:
  multichannel's `U · LSTM · archive` with the label ran batch 256 once, and that run is
  the panel's worst total (MAPE 297.9). So batch acts mostly *between* input sets, as
  Finding 2 shows, not as a knob that varies within one.
- **The validation loss does not flag a bad run.** Over all 46 cells, its within-cell
  correlation with MAPE has a median of +0.04 (CDNOW +0.06, electronics +0.03, gift −0.01,
  multichannel +0.08). In CDNOW's `U · LSTM · floored/no_cluster` (type B below), it spans
  0.0915–0.0975 while MAPE spans 22.6–348.2. This is `docs/model-selection.md`'s finding
  again, at the level of single runs.
- **Only forecast CV tracks MAPE**: the flatter the forecast across customers, the worse
  the total, in 16 of 24 searched cells, with and without the label. But CV is a symptom of
  the forecast, not a setting anyone chooses. In 43 of the 46 cells, a lower CV goes with a
  higher bias (median −0.47). The types below say which configurations produce flat
  forecasts.

#### Finding 4. The bad runs fall into four types

Counting which cells the worst 10% come from gives four recognisable types, one or two
per panel.

| type | where | cells holding the bad runs | training | forecast |
| --- | --- | --- | --- | --- |
| **A. Too little training, collapse** | multichannel (total and ranking), electronics (ranking, and 13 of its 32 worst totals) | `archive/no_cluster` and T's `archive`/`paper` | batch 256 or stopped at epoch 1, 4–9 epochs, 27–48 updates | flat (CV ≈ 0.1), over-forecast, ranking at chance |
| **B. Long training, collapse** | CDNOW (total) | `U · LSTM · floored/no_cluster` | batch 32, no dropout, ~25 epochs, ~1,300 updates | flat, over-forecast by +100 to +350%, ranking intact |
| **C. Label, wide spread** | electronics (16 of its 32 worst totals) | label runs, mostly ValendinLSTM | ordinary: batch 32–64, ~27 epochs | spread out (CV ≈ 1.3), over-forecast by ~+72%, best ranking |
| **D. No type** | gift | spread over all seven cells | — | under-forecast without the label |

**A. Too little training leads to collapse** (electronics ranking, multichannel both).
- 13 of multichannel's 16 worst totals and 7 of its 16 worst rankings come from one cell,
  `U · LSTM · archive/no_cluster`; ValendinLSTM's `archive/no_cluster` adds 8 more of the
  worst rankings. On electronics, 22 of the 32 worst rankings ran at batch 256, and the
  other 10 are `paper` runs that kept epoch 1. The same collapsed runs also give 13 of
  electronics' 32 worst totals (10 ValendinLSTM `archive` runs at batch 256, 3 `paper` runs
  at epoch 1).
- These runs kept epoch 3.5 (multichannel) and 8 (electronics) at the median: 4.5 and 9
  epochs, 27 and 48 updates. That is short by both measures, not just a few large-batch
  steps. None had the label.
- The result is a collapse: every one of the 40 `archive/no_cluster` runs on multichannel,
  and 79 of 80 on electronics, has CV < 0.2. Across all runs, CV and Spearman correlate at
  0.91 on electronics and 0.89 on multichannel, so on these panels a ranking failure *is*
  a collapse.
- The other side confirms it: the best 10% by MAPE kept epoch 67.5 (electronics) and 64
  (multichannel) at the median, with 1,742 and 2,860 updates.

**B. Long training can collapse too** (CDNOW total).
- 15 of CDNOW's 28 worst totals come from `U · LSTM · floored/no_cluster`, the cell behind
  the +126 MAPE blow-up of §5.2. Its runs split in two:

| runs in the cell | CV | bias % |
| --- | ---: | ---: |
| 12 of 20 | < 0.25 | +99 to +347 |
| 4 of 20 | > 0.8 | +5 to +74 |

- These runs are not short: the collapsed runs on CDNOW and gift received a median of
  1,443 and 1,560 updates, mostly at batch 32, i.e. 25 and 30 epochs. 11 of CDNOW's 12 and
  5 of gift's 7 collapsed runs come from `floored/no_cluster`: batch 32, no dropout, no
  weight decay, no label. This is the floor's harm on gift (claim 8) and its blow-up on
  CDNOW (claim 9), seen run by run.
- In 8 of the 12 collapsed runs, Spearman stays at 0.39–0.43: the model still orders
  customers, but gives every one something close to the average rate, including the many
  who have stopped buying. `docs/absorbing-death-state.md` discusses that mechanism; it is
  not tested here.
- **The recipe alone does not explain it.** `T′ · LSTM · paper` runs the same pinned recipe
  on CDNOW, supplies two of its five best totals, and does not collapse (CV ≈ 1). The two
  cells read different inputs: `Transactions` alone in T′, plus `week_sin` and `week_cos`
  in U (each winner's `selected_features`). So the CDNOW collapse comes from the recipe
  combined with that input set. Which part of the difference matters is not identified.

**C. The label widens the spread of the total** (electronics total).
- 16 of electronics' 32 worst totals carry the cluster label (11 of them ValendinLSTM).
  They trained ordinarily (batch 32 or 64, kept epoch 26 at the median) and their
  forecasts are not flat (CV 1.31) but over-forecast by +72% at the median. Label runs also
  rank customers best: every run in electronics' best 10% by Spearman has the label, and
  15 of 16 on multichannel do.
- The label does not raise the typical error. On electronics, family U's median MAPE is
  54.2 with the label and 53.8 without. It widens the spread around it: MAPE sd 20.4
  against 13.5, and 29.8 against 10.0 for `ValendinLSTM · archive`. The bad label runs are
  that wider tail.
- They reached a lower validation loss (0.081 against 0.089) and still produced the worst
  totals: a better selection score did not protect the total.

**D. Gift has no type.**
- Only 4% of gift's MAPE variance lies between cells. Its worst totals come from six of
  its eight cells, with no cell holding more than four.
- What it does show is a direction: without the label, runs under-forecast (−45 to −51% at
  the extreme). Its worst rankings are the floored no-label cells (12 of 16), the
  long-training collapse of type B in milder form (CV 0.22).

#### Finding 5. How far a recipe trains matters; which epoch one run kept does not

Type A shows that recipes stopping after a handful of epochs give bad runs. But within a
cell, the kept epoch barely moves MAPE: the median within-cell correlation is −0.15 over all
46 cells (electronics −0.22, multichannel −0.24, CDNOW 0.00, gift −0.10), and −0.10 over the
24 searched cells of Finding 3. The epoch effect of §5 is between arms: it is about how far
a recipe lets training go, not about which epoch one search happened to keep.

#### Finding 6. A floor works by carrying training past the flat stretch, not by keeping late weights

- 92% of the 440 floored runs kept weights from *before* their floor epoch. The floor lets
  training continue past the plateau of §3.2 until it finds a lower loss, and that lower
  loss often lies before the floor anyway.
- CDNOW's blown-up runs (type B) kept epochs 8–61. So §5.3's remaining suspect, a 90-epoch
  floor on a 39-week window, cannot act by handing the forecast overtrained weights. It
  would have to act through which early epoch gets selected, or through the refit.

#### In short

**Terms.**
- **Cluster label (`kmeans_8`)**: an extra input column. Each customer is assigned to one of
  8 groups by k-means on three summaries of their purchase history: number of repeat
  purchases `x`, time of last purchase `t_x`, and time observed `T`. These are the
  statistics Pareto/NBD is fitted on. Family U runs every configuration twice, without this
  column (`no_cluster`) and with it (`kmeans_8`).
- **Configuration**: one panel, one arm, one model and one set of input columns, run 20
  times.
- **Collapse**: the model gives every customer almost the same forecast (CV near 0), so it
  cannot rank customers (Spearman ≈ 0).
- **Bad run**: in its panel's worst 10% by MAPE or |bias| (total), or by Spearman
  (ranking).

**What a bad run looks like.** On electronics, multichannel and CDNOW, a bad total is
always an over-forecast; on gift, an under-forecast. A bad ranking is a collapsed
forecast. The two are mostly different runs (1–5 shared per panel).

**1. The cluster column decides the batch size the search picks, and that decides the
collapse.**
- On electronics and multichannel *without* the cluster column, Optuna picks batch 256 in
  17–20 of 20 runs. Training stops after 3–8 epochs (24–36 updates) and the forecast
  collapses. This produces 13 of multichannel's 16 worst totals and most of electronics'
  worst rankings.
- *With* the cluster column, Optuna picks batch 64, training runs 13–29 epochs, nothing
  collapses, and Spearman is 0.17–0.30. It also picks a lower learning rate (6 of 8 pairs)
  and more dropout (4 of 4 LSTM pairs).
- On CDNOW and gift, Optuna picks batch 64 either way.

**2. The cluster column prevents collapse but makes the total less stable.**
- On electronics, median MAPE is the same with and without it (54 against 54), but the
  spread between runs is wider (sd 20 against 13). Runs with the column supply 16 of the
  32 worst totals.
- On CDNOW, adding it turns a near-zero bias into an over-forecast (LSTM median −3% →
  +55%).

**3. Within one configuration, no hyperparameter explains which runs are bad.**
- Learning rate, weight decay, dropout, widths and kept epoch correlate with MAPE no more
  often than chance. The validation loss does not flag bad runs either (median correlation
  +0.04).
- One exception: with the cluster column, a low learning rate goes with worse totals. On
  electronics' ValendinLSTM, the quarter of runs with the lowest learning rate has median
  MAPE 97 against 53, and contains the two worst runs. This rests on 8 configurations, so
  it is a lead, not a finding.

**4. About half the spread between runs is luck.**
- Configurations with every hyperparameter fixed spread as much as searched ones (median
  MAPE sd 33 against 36 on CDNOW, 11 against 9 on electronics, 5 against 7 on gift).
- The variation comes from random weight initialisation and data shuffling, not from
  Optuna's choices. One fixed recipe on electronics (`T · LSTM · paper90`) produced both
  the panel's best run (MAPE 34) and its third worst (MAPE 103).
- Multichannel is the exception (sd 38 searched against 10 fixed), because of its
  collapsing no-cluster `archive` configurations.

**5. One collapse comes after long training.** On CDNOW, the LSTM with the fixed `floored`
recipe and no cluster column collapses after about 25 epochs at batch 32, over-forecasting
by +100 to +350%. The same recipe reading only `Transactions` (family T′, without
`week_sin`/`week_cos`) does not collapse. The cause is the recipe combined with those extra
inputs; which part matters is unknown.

**Implications.**
- On electronics and multichannel without the cluster column, remove batch 256 from the
  search or impose a minimum number of epochs.
- With the cluster column, test a lower bound on the learning rate.
- Always report a configuration as the distribution over its 20 runs.

All of this is descriptive: each point needs its own experiment before it becomes a claim.

## 9. Limits

- Four panels plus electronic_5y, each reported separately. No measured panel
  characteristic predicts where a floor helps.
- §3.2–3.4 rest on 3 runs per panel at one hyperparameter setting.
- Training is unseeded, but a replication's forecast seed fixes the random state that the
  next replication's training starts from. Arms that do identical work in identical order
  are therefore coupled. The archive shows 0 exact repeats across 3,020 suites, so it is
  unaffected.
