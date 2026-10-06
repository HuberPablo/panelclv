# Hyperparameter search

How the Optuna search that picks every neural model is set up, how each piece compares
with standard practice, what was measured, and what is still to test. Model selection
(whether the criterion picks the right trial) is in `docs/model-selection.md`; training
length is in `docs/insight-training-efficiency.md`.

It is not a grid search. Optuna's TPE sampler draws each trial's settings from what
earlier trials scored, so it concentrates on whatever the score favours. That makes the
score itself the first thing to get right (§3.1).

Conventions: every interval is a 95% percentile bootstrap from `evaluation.effects.effect`
(`docs/statistical-protocol.md`), and bold marks one that excludes 0. Family letters are
those of `docs/studies-run.md` §4.

## 1. How a study runs

One **study** is one search, its refit and its forecast. In order:

1. **Sample.** TPE, seeded `base_seed + i`, draws a trial's settings from the model's
   registry entry (`registry/model_registry.py`). Searched archives used 100 trials.
2. **Train.** `fit_model` (`training/loop.py`) trains with AdamW, gradient clipping at
   1.0 and cross-entropy against the count class, for at most `n_epochs` (100 in the
   archives).
3. **Stop early.** After each epoch it scores the validation window (ADR-0001). An epoch
   counts as an improvement only if it beats the best by an absolute 1e-4
   (`loop.py:377`). After `patience` epochs (7) without one, training stops, unless
   `min_epochs` has not been reached yet. The best epoch's weights are kept.
4. **Prune.** From the 4th epoch on, Optuna's `MedianPruner` stops a trial whose best loss
   so far is worse than the median of finished trials at the same epoch
   (`tuning/optuna_tuning.py:504`).
5. **Select.** The trial with the lowest validation loss wins.
6. **Refit.** The winner is fine-tuned for 5 epochs on the full calibration window at
   learning rate 1e-3 and batch 512 (`trials/refit.py`, ADR-0008).
7. **Forecast.** The refit model is rolled out over the holdout, averaged over 200–500
   paths.

### What is searched

| model | searched | fixed |
| --- | --- | --- |
| LSTM | hidden width {32, 64, 128}, dense width {32, 64, 128}, dropout 0–0.4, learning rate 1e-4–3e-3 (log), batch {32, 64, 128, 256} | embedder `valendin`, weight decay 0 |
| ValendinLSTM | learning rate 1e-4–3e-3 (log), batch {32, 64, 128, 256} | architecture 128/128, no dropout (ADR-0004); weight decay 0 |
| Transformer | `d_model` {32, 64, 128}, heads {2, 4, 8}, layers 1–3, dropout, learning rate, batch | embedder `valendin`, weight decay 0 |

**The archived searches used a different space.** Before 23 September 2026, weight decay
was searched over 1e-6–1e-2 (log) and batch over {64, 128, 256}; commits d7e902d and
57b8c18 pinned weight decay to 0 and added batch 32. Families T, T′, U and V
ran on the old space (and earlier families on older ones), so a rerun meant to reproduce
one must restore it (family U′ does, §4).

## 2. Against standard practice

| # | Piece | What this package does | Standard practice | Status |
| --- | --- | --- | --- | --- |
| 1 | Epoch loss | **was** a mean of per-batch means; now the per-cell mean | the per-sample mean (Keras metrics, Lightning's epoch logging) | **fixed**, ADR-0010 (§3.1) |
| 2 | Pruner | `MedianPruner`, warm-up 3 epochs, 5 startup trials | Optuna's defaults are warm-up 0, 5 startup trials; Optuna's own benchmark pairs **Hyperband** with TPE and the median rule with random search | open (§3.2) |
| 3 | Pruning unit | epochs | epochs: standard | noted (§3.2) |
| 4 | Improvement threshold | absolute 1e-4 | a threshold relative to the loss, or patience sized to the curve | open (§3.3) |
| 5 | Training floor | `min_epochs` under early stopping, pruner warm-up raised to match | not standard; a remedy for flat curves | open (§3.4) |
| 6 | Refit | lr 1e-3, batch 512, 5 epochs, fresh AdamW, whatever the search chose | fine-tune at the tuned learning rate or below | open (§3.5) |
| 7 | Sampler | TPE, seeded per study | standard | fine after #1 (§3.6) |

## 3. What was found

### 3.1 The validation score depended on the batch size (fixed)

Until commit af2b14b, `validate_one_epoch` returned the mean of its per-batch means. The
validation loader runs at the trial's batch size, in fixed customer order, so the short
last batch counted as much as a full one. At batch 256 on electronics, the 61 highest-id
customers carried 25% of the score instead of 7%.

Family U's stored winners, scored at every batch size with their weights fixed, against
the true per-cell mean:

| panel | the short batch's loss vs average | batch 64 | batch 256 |
| --- | ---: | ---: | ---: |
| electronics | 0.79× | −0.1% | **−4.1%** |
| multichannel | 0.29× | −0.3% | **−6.2%** |
| cdnow | 1.35× | +0.2% | **+2.8%** |
| gift | 2.28× | +3.0% | **+13.4%** |

- **The bias outweighed the real differences.** The winning loss varies by only 0.08–3.1%
  across replications (`docs/model-selection.md` §2).
- **TPE followed it.** Batch 256 was drawn in 61% and 51% of all electronics and
  multichannel trials, against 8–9% on CDNOW and gift.
- **Its direction is an accident.** On electronics and multichannel the highest ids buy
  less than average; on CDNOW and gift, more.

**The fix** weights each batch by the cells it scored (`_batch_weight`, `loop.py:67`), so
the score is the exact per-cell mean at any batch size. Shuffling the validation set was
rejected: at batch 256 it leaves a 95% range of ±3–9% around the true mean.

**The rerun** (family U′, electronics · `archive` · no cluster label, 20 replications per
model, `docs/model-selection.md` §3.9):

| | LSTM | ValendinLSTM |
| --- | --- | --- |
| winners at batch 256, archive → rerun | 18 → 2 of 20 | 19 → 1 of 20 |
| Δ MAPE | −3.0 [−7.5, +1.5] | **−9.4 [−16.0, −3.0]** |
| Δ bias % | −8.0 [−20.1, +4.0] | **−11.4 [−21.8, −1.6]** |
| Δ Spearman | +0.017 [−0.009, +0.044] | **+0.055 [+0.013, +0.099]** |

The fix removed the preference for batch 256. ValendinLSTM then trains longer and 5 of 20
runs no longer collapse. The LSTM moves to batch 64 but still stops after about 4 epochs
and collapses: the score was one cause of stopping too early, not the only one.

### 3.2 The pruner judges trials before their curves separate (open)

`MedianPruner` here starts at the 4th epoch, after 5 finished trials, and stops a trial
whose best loss so far is worse than the median of finished trials at that epoch. It
stopped 56–73% of trials per panel in the archive (48–50% in the U′ rerun).

- **The curves are flat at first.** With early stopping off, validation loss stays flat
  for 29–131 epochs before a run's own best (`docs/insight-training-efficiency.md` §3.2).
  At epoch 4 most trials look alike, so the rule mainly rewards those that drop fast
  early, not those that end best.
- **It compares trials at equal epochs.** That is the standard unit, but at the same
  epoch a batch-32 trial has made 8× more updates than a batch-256 trial.
- **It compared trials on the biased score of §3.1** in every archived search.
- **Which epoch trials were pruned at is not recoverable.** The per-epoch values lived in
  Optuna's in-memory storage and were never written out.

Proposed replacements, to be tested against the current rule and against no pruning
(`pruner=False`, already supported):

```python
optuna.pruners.HyperbandPruner(min_resource=10, max_resource=n_epochs, reduction_factor=3)
```

No trial is judged before epoch 10; survivors are re-ranked at about 30 and 90 epochs,
within groups of trials of similar age. On a floored arm, `min_resource` should equal
`min_epochs`.

```python
optuna.pruners.PatientPruner(
    optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10), patience=7)
```

Keeps the median rule but lets it fire only after 7 epochs without improvement, the same
patience early stopping uses.

Pruning may not be worth it at all: trials already stop after 8–25 epochs, so turning it
off may cost little GPU time.

### 3.3 The improvement threshold is absolute (open)

An epoch counts as an improvement only if it beats the best by 1e-4. That is 0.4% of
multichannel's validation loss (about 0.024) but 0.1% of CDNOW's (about 0.091), so the rule
is about four times stricter on multichannel. With patience 7 on flat curves, runs stop at
epochs 3–8 (`docs/insight-training-efficiency.md` §3.3). It is the leading suspect for the
LSTM's remaining collapse after §3.1's fix.

### 3.4 A training floor buys looking time, not late weights (open)

`min_epochs` stops early stopping from ending a run before the floor; the pruner's warm-up
is raised to match (`optuna_tuning.py:501–505`). It does not force the kept weights to come
from after it: 92% of the 440 floored runs kept weights from before their floor
(`docs/insight-training-efficiency.md` §8.2, Finding 6). `select_from_epoch` is the control
that does force late weights.

A floor has not been tested on top of §3.1's fix. It is the natural next test for the
LSTM's collapse on electronics.

### 3.5 The refit ignores the tuned learning rate and batch (open)

`refit_best_trial` runs at its defaults, and no experiment script passes `refit_kwargs`:
5 epochs at learning rate 1e-3 and batch 512, with a fresh AdamW. A fresh Adam's first
steps move each weight by about the learning rate, so a winner tuned at 1.6e-4 is
fine-tuned at six times its rate, on weights the validation set never scored.

This matches a lead in `docs/insight-training-efficiency.md` §8.2: with the cluster label,
runs that drew a low learning rate over-forecast more, and the two worst electronics
ValendinLSTM runs used 1.6e-4 and 5.8e-4. Not measured. The test: forecast each winner from
its checkpoint and from its refit, and relate the difference to its tuned learning rate.

### 3.6 The sampler converges (fine)

The median winner is trial 46–56 of 100, so TPE is still finding better trials late. In the
archive it converged on the biased score of §3.1; with the fix it converges on the true
one.

## 4. Runs

| family | what | status |
| --- | --- | --- |
| U′ | electronics · `archive` · no cluster label, rerun with the fixed score; archive search space restored | complete, §3.1 (`.scratch/score-fix/`) |
| EP | the epoch probe of §5.1: ValendinLSTM, 20 trials per panel, 150 epochs, no early stopping, no pruning, 2y windows | complete, §5.1 (76 items on vast.ai, 4 locally; $0.84) |
| EP-3y/5y | the same probe on the 3y windows (electronics, gift, multichannel) and the 5y electronics split | complete, §5.2 (78 items on vast.ai, 2 locally) |

## 5. Tests owed

Every test changes one thing against a fixed-score baseline and is compared on MAPE, bias
and Spearman with independent bootstrap intervals, plus which batch, learning rate and
kept epoch the search picks, the number of collapsed runs, and GPU time per study.

| # | Test | Changes | Question |
| --- | --- | --- | --- |
| 0 | **Epoch probe** (§5.1) | nothing; every trial trains 150 epochs and its curve is recorded | when do trials become distinguishable, and when does a single run leave its flat start? Sets the pruner's warm-up and `min_epochs` from data |
| 1 | **Current vs new settings** | the fixed score, Prechelt's PQ1 stopping rule (§5.2) and no pruning, ValendinLSTM, all four panels | do the changes together improve the holdout forecast? Baseline: family U `archive` · no cluster label (no rerun needed) |
| 2 | **Pruner** | current `MedianPruner` vs `HyperbandPruner` (`min_resource` from test 0, η = 3) vs none | does pruning at epoch 4 discard trials that would have won, and does pruning pay at all? |
| 2b | **Hyperband η** | η = 3 vs η = 2, only if test 2 favours Hyperband | does gentler pruning change the winners? |
| 3 | **Improvement threshold** | absolute 1e-4 vs a relative one, on multichannel | does the 4× stricter rule cut multichannel's training short? |
| 4 | **Refit learning rate** | checkpoint vs refit forecasts, related to each winner's tuned learning rate | does refitting at 1e-3 damage winners tuned at a low rate? |

### 5.1 Test 0: the epoch probe

**Why.** The pruner's warm-up (3 epochs) and the `min_epochs` floor (0, 50 or 90 in the
archive) were set by convention or precedent. Both should be set where the curves say:
- the **warm-up** (or Hyperband's first rung) where *trials* become distinguishable from
  one another, since pruning before that point discards trials at random;
- the **floor** past the point where a *single run* leaves its flat start, since early
  stopping before that point ends runs that had not begun to learn.

**Design.** `scripts/run_epoch_probe.py`, run on vast.ai.
- ValendinLSTM only, with count and week as inputs and no cluster label: family U's
  `archive` · no cluster label configuration, on all four panels.
- **20 trials per panel**, each a set of training settings drawn at random from the
  archive search space (learning rate 1e-4–3e-3 log, weight decay 1e-6–1e-2 log, batch
  {64, 128, 256}). Random rather than TPE, so the trials represent the space rather than
  one search's preferences. The same 20 draws on every panel, seeded from one config
  value.
- Each trial trains **150 epochs with no early stopping and no pruning**, under the fixed
  score (ADR-0010). The validation loss is recorded at every epoch.
- At epochs 5, 10, 20, 30, 50, 75, 100 and 150 the weights are **rolled out over the
  validation window** (warm-up on the periods before it, 100 paths) and scored with
  `compute_forecast_metrics` plus Spearman, the family V procedure. The holdout is never
  touched, so nothing chosen from this probe has seen the data it will be tested on.
- 80 work items (4 panels × 20 trials). Output per item in
  `Studies/epoch_probe__ValendinLSTM__<panel>__tNN/`: `history.csv` (per epoch) and
  `rollouts.csv` (per checkpoint), with `results.csv` written last as the completion
  marker the fleet tooling checks.

**Analysis** (`--report`), per panel:
- **Distinguishable:** the first epoch at which the trials' ranking agrees with their
  final ranking (rank correlation ≥ 0.8), by validation loss and by validation-rollout
  MAPE. This gives the warm-up.
- **Flat start:** for each trial, the first epoch whose validation loss beats epoch 1's by
  more than the stopping rule's 1e-4, and the epoch where it reaches half of its total
  improvement; the median over trials gives the floor. Alongside: the epoch at which
  patience 7 would have stopped each trial, and how far that is from its best.
- **Does early CE predict the forecast?** The final objective is forecasting, not
  teacher-forced CE, and §3.1 and `docs/model-selection.md` show the two can disagree. So
  for each epoch t the report also gives the rank correlation, across trials, between the
  CE at t (best so far, what the pruner and early stopping see) and the trial's final
  validation-rollout MAPE, |bias| and Spearman, signed so positive means low CE goes with
  a good forecast. The rollout MAPE at t is the comparison row. If CE tracks the final
  rollout from some epoch on, pruning on CE is safe from there and the warm-up goes
  there; if it never does, the warm-up and floor come from the rollout curve instead, or
  pruning is dropped. With 20 trials a single correlation is noisy (about ±0.4), so
  agreement across the four panels is what counts.
- Plots of validation loss and rollout MAPE against epoch, one line per trial.

It also gives test 2's warm-up; the stopping rule for test 1 is chosen in §5.2.

#### Results (6 October 2026)

`python scripts/run_epoch_probe.py --report`; plots in `.scratch/epoch-probe/`. 20 trials
per panel, medians over trials unless stated.

**The validation loss falls fast, then slowly, for a long time.**

| | cdnow | electronics | gift | multichannel |
| --- | ---: | ---: | ---: | ---: |
| epoch where the loss first beats epoch 1's | 2 | 2 | 2 | 2 |
| epoch with half of the run's total drop | 3 | 3 | 3 | 4 |
| the run's own best epoch | 68 | 138 | 121 | 135 |
| where patience 7 stops it | 19 | 16 | 13 | 15 |
| loss given up by stopping there | 7.3% | 3.4% | 5.4% | 4.5% |
| trials rank as they will at the end (CE, ρ ≥ 0.8 from then on) | 24 | 69 | 76 | 54 |

There is no flat start in the loss: half of each run's drop happens by epoch 3–4. What
is slow is the rest. The loss keeps improving until epoch 68–138, while patience 7 ends
runs at epoch 13–19, before the trials can even be told apart (epoch 24–76).

**The forecast keeps improving long after patience 7 stops.** Validation-rollout medians
over the 20 trials, by epoch; *collapsed* counts trials with forecast CV < 0.2:

| panel | metric | 10 | 20 | 30 | 50 | 75 | 100 | 150 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cdnow | MAPE | 59.5 | 56.8 | **46.1** | 46.3 | 55.5 | 65.7 | 75.3 |
| | bias % | +59.5 | +37.0 | +23.7 | **+20.9** | +43.3 | +53.6 | +43.1 |
| | collapsed | 7 | 1 | 1 | 1 | 0 | 0 | 0 |
| electronics | MAPE | 83.6 | 64.9 | 50.2 | 50.7 | 51.2 | **45.5** | 47.9 |
| | bias % | +44.9 | +42.7 | +31.9 | +20.2 | +19.6 | +18.3 | **+15.5** |
| | collapsed | 16 | 18 | 20 | 18 | 17 | 14 | **9** |
| gift | MAPE | 92.0 | 76.1 | 47.2 | 49.8 | 46.8 | 43.3 | **38.8** |
| | bias % | +61.2 | +53.0 | +34.2 | +27.1 | +25.6 | +20.4 | **+12.1** |
| | collapsed | 19 | 20 | 20 | 16 | 13 | 11 | **9** |
| multichannel | MAPE | 100.6 | 80.8 | 70.5 | 65.6 | 61.0 | 57.8 | **56.4** |
| | bias % | +76.1 | +50.6 | +44.1 | +34.9 | +23.7 | +25.6 | **+18.1** |
| | collapsed | 11 | 13 | 14 | 12 | 6 | 5 | 6 |

- **On electronics, gift and multichannel the forecast improves to epoch 100–150.** From
  epoch 20, where patience 7 stops, to epoch 150, median MAPE falls by 17–37 points, the
  bias by 27–41 points, and the number of collapsed trials halves. The collapse of §8's
  failure type A is in large part training that stopped too early.
- **CDNOW is the exception.** Its forecast is best at epoch 30–50 and worsens after,
  while its loss keeps improving to epoch 68. Its calibration window is the shortest (39
  weeks against 104), and it over-fits first.

**Early CE predicts the final forecast on one panel only.** Rank correlation across
trials between the CE at epoch t and the final validation-rollout result, positive when
low CE goes with a good forecast:

| panel | CE at t → final MAPE | CE at t → final Spearman |
| --- | --- | --- |
| cdnow | about 0 at every t (+0.0 to +0.3) | **negative**, −0.26 to −0.56: lower CE, worse ranking |
| electronics | **negative** at epochs 1–10 (−0.25 to −0.38), −0.2 to +0.3 after | −0.15 to +0.46, unstable |
| gift | −0.1 to +0.3 | **+0.3 to +0.8**, mostly 0.5–0.7 |
| multichannel | **+0.2 to +0.8**, mostly above 0.5 | +0.3 to +0.8, at least 0.68 from epoch 20 |

Pruning compares trials on CE. On CDNOW and electronics, early CE is unrelated or opposed
to the forecast a trial ends with, so pruning there discards trials for a reason that
has nothing to do with forecasting. Even the rollout MAPE at an early epoch predicts the
final rollout poorly (ρ −0.2 to +0.6): forecasts reorder as training goes on.

**What this sets.**
- **Pruner (test 2): turn it off, or start it no earlier than epoch 50–75.** The current
  warm-up of 3 judges trials 20–70 epochs before their CE ranking settles, on a number
  that does not predict the forecast on two of four panels. Test 2 becomes current rule
  against no pruning; Hyperband only with `min_resource` ≥ 50.
- **Stopping rule (test 1): set in §5.2,** where Prechelt's (1998) criteria are compared
  and PQ1 is chosen.

Limits: ValendinLSTM only; 20 trials, so a single correlation carries about ±0.4; the
rollouts are of the trained weights without the ADR-0008 refit; the archive search
space, sampled at random rather than by TPE.

### 5.2 Choosing the settings from calibration data only

**The aim.** Set every training and search setting using only data inside the
calibration window — the validation loss and the validation-window forecast — then freeze
them and touch the holdout once, for the final evaluation. If that works, the settings are
justified without any risk of having been tuned on the data they are judged on.

#### What the literature does

| Practice | References | Assumption | Holds here? |
| --- | --- | --- | --- |
| Early stopping on a validation split, the stopping rule chosen from validation curves | Prechelt (1998), *Early Stopping — But When?*; Bengio (2012), *Practical recommendations for gradient-based training*; Goodfellow et al. (2016) §7.8 | the rule should let a run reach its validation minimum | **yes** (below) |
| Validate on the forecasting task over the horizon, not on a one-step loss | Tashman (2000); M-competition practice | the validation score measures what is reported | **partly**: one-step CE does not track the forecast (§5.1, `docs/model-selection.md` §3) |
| Multi-fidelity search: pruning, Hyperband, learning-curve extrapolation | Li et al. (2018), *Hyperband*; Falkner et al. (2018), *BOHB*; Domhan et al. (2015) | trials rank early as they rank at the end | **no**: the CE ranking settles only at epoch 24–76 and early CE predicts the final forecast on one panel of four (§5.1) |
| Rolling-origin validation: several windows, not one | Tashman (2000); Bergmeir & Benítez (2012); Bergmeir, Hyndman & Koo (2018) | one validation window may not represent the next period | not tested yet |

The references are the standard attributions; their exact sections should be checked
before they are cited in the thesis.

#### The procedure, applied

Both decisions use calibration data only.

1. **Pruner.** Multi-fidelity methods assume trials rank early as they will at the end.
   On the probe curves the CE ranking settles only at epoch 24–76, and early CE predicts
   the final validation forecast on multichannel only (§5.1). **Decision: pruning off.**
2. **Stopping rule: the criteria of Prechelt (1998).** That paper defines three families
   of early-stopping criteria and compares them; all read only the validation and training
   loss. Each was replayed on the 80 recorded curves, keeping as `fit_model` does the
   weights of the lowest validation loss so far, then scored on the validation-window
   forecast at the epoch it keeps:
   - **GLα, generalisation loss:** stop once the validation loss is more than α% above
     its best so far.
   - **PQα, progress quotient:** stop once that generalisation loss divided by the
     training progress over the last 5 epochs exceeds α. It holds off while the training
     loss is still falling.
   - **UPs:** stop once the validation loss has risen at the end of s successive 5-epoch
     strips.

#### The curves

Each figure: left, the validation loss of the 20 trials (grey) and their median (black);
middle, the median validation-window MAPE and |bias| (trials in grey); right, how many of
the 20 trials have collapsed (forecast CV < 0.2). Dashed lines mark the median epoch kept
by the current rule (orange) and by Prechelt's PQ1 (green); the dotted line is the median
epoch of the loss minimum. Regenerate with
`python scripts/run_epoch_probe.py --report --out docs/figures/epoch-probe`.

![CDNOW](figures/epoch-probe/cdnow.png)

![electronics](figures/epoch-probe/electronics.png)

![gift](figures/epoch-probe/gift.png)

![multichannel](figures/epoch-probe/multichannel.png)

#### Did it work?

Validation-window MAPE at the epoch each criterion keeps, medians over the 20 trials per
panel, with the median epoch kept in brackets. *Epochs trained* is the mean over all 80
runs, the cost. The *oracle* is the single checkpoint with the best median validation
forecast: the best a fixed epoch for all trials could do on this window, chosen by looking
at the forecast, which the criteria do not.

| criterion | cdnow | electronics | gift | multichannel | epochs trained |
| --- | --- | --- | --- | --- | ---: |
| current: patience 7 | 65.1 (12) | 79.3 (9) | 78.0 (6) | 99.3 (8) | 19 |
| GL1 | 65.1 (3) | 113.7 (5) | 93.0 (3) | 143.2 (5) | 7 |
| GL2 | 65.1 (5) | 113.7 (5) | 88.9 (3) | 143.2 (5) | 8 |
| GL3 | 65.1 (5) | 113.7 (5) | 83.5 (3) | 143.2 (5) | 10 |
| GL5 | 55.7 (5) | 113.7 (5) | 74.9 (7) | 101.4 (5) | 25 |
| PQ0.5 | 43.1 (24) | 50.2 (12) | 49.8 (45) | 64.6 (64) | 56 |
| **PQ1** | **40.8** (33) | **41.9** (35) | **41.7** (68) | **62.6** (96) | **80** |
| PQ2 | 37.2 (34) | 43.7 (93) | 41.1 (84) | 61.2 (107) | 103 |
| PQ3 | 42.7 (38) | 43.7 (93) | 41.1 (84) | 58.4 (123) | 111 |
| UP2 | 42.5 (39) | 42.7 (19) | 41.7 (47) | 65.6 (55) | 58 |
| UP3 | 42.7 (58) | 50.3 (27) | 41.7 (97) | 63.5 (98) | 102 |
| UP4 | 44.4 (62) | 49.9 (114) | 41.0 (121) | 56.7 (111) | 130 |
| oracle | 46.1 (30) | 45.5 (100) | 38.8 (150) | 56.4 (150) | — |

How far above its own loss minimum each run ends, median: patience 7 3–7%, GL 6–16%,
PQ0.5 1–3%, PQ1 0–0.9%, PQ2 and PQ3 0–0.1%, UP 0–1.5%.

PQ1 in full, against the current rule and the oracle (MAPE / |bias| % / Spearman /
collapsed trials of 20):

| panel | current: patience 7 | PQ1 | oracle |
| --- | --- | --- | --- |
| cdnow | 65.1 / 65.1 / 0.09 / 4 | **40.8 / 30.8 / 0.22 / 1** | 46.1 / 46.0 / 0.24 / 1 |
| electronics | 79.3 / 64.6 / 0.02 / 17 | **41.9 / 15.8 / 0.04 / 17** | 45.5 / 26.1 / 0.07 / 14 |
| gift | 78.0 / 35.1 / 0.00 / 19 | 41.7 / 21.7 / 0.01 / 18 | 38.8 / 14.6 / 0.09 / 9 |
| multichannel | 99.3 / 80.5 / −0.00 / 9 | 62.6 / 24.8 / 0.00 / 7 | 56.4 / 28.0 / 0.03 / 6 |

The forecast is scored at the latest checkpoint at or before each run's kept epoch (the
rollout was recorded at 8 epochs only), so a kept epoch between checkpoints is scored a
little early.

- **GL fails.** The validation loss is noisy (the spikes in the figures), so it rises 1–5%
  above its best within the first 3–5 epochs and GL stops the run there, worse than the
  current rule.
- **PQ works best.** PQ1 and PQ2 lower validation MAPE on every panel, from 65–99 under
  the current rule to 37–63. They match or beat the oracle on CDNOW and electronics, and
  close 86–94% of the gap to it on gift and multichannel. They can beat the oracle because
  they stop each trial at its own moment, where the oracle uses one epoch for all. PQ suits
  these curves: a slowly falling loss with noise on top, which PQ does not stop on while
  training still makes progress.
- **UP comes close but is less consistent:** UP2 is good on CDNOW and electronics but
  weaker on multichannel; UP4 is good on multichannel but trains 130 epochs.
- **Chosen: PQ1,** which is within 4 MAPE points of PQ2 on every panel at 80 epochs
  against 103. The choice among α uses the validation forecast, which is calibration data.
- **Ranking is not fixed.** Spearman stays near 0 on electronics, gift and multichannel
  under every criterion, and 7–18 of 20 trials remain collapsed there. Training longer
  repairs the level of the forecast; ValendinLSTM without a customer-level input still
  cannot tell customers apart there (`docs/insight-training-efficiency.md` §8: the label
  is what lets it rank).

#### Robustness: three-year windows and the 260-week electronics split

The same probe was rerun on the longer calibrations `run_real_panel_benchmarks` declares
(`python scripts/run_epoch_probe.py --calibration 3y,5y`; 80 items on vast.ai and the
local GPU, 6 October 2026):
- **3y:** electronics, gift and multichannel with 156 calibration weeks; the third year
  is the validation window. For the same panels this validates on the year *after* the
  2y probe's validation year, so it is also the second validation window that
  rolling-origin practice asks for.
- **5y:** the paper's electronics cohort (3,755 customers) on its 260-week split, the
  last 52 weeks validating.

The same 20 trials, 150 epochs, no early stopping and no pruning; the holdout is not read.
Validation-window MAPE at the epoch each criterion keeps (median epoch in brackets):

| criterion | 3y electronics | 3y gift | 3y multichannel | epochs trained |
| --- | --- | --- | --- | ---: |
| current: patience 7 | 71.0 (12) | 47.8 (10) | 74.6 (9) | 21 |
| GL1 | 98.4 (4) | 49.9 (6) | 100.2 (4) | 12 |
| GL2 | 98.4 (4) | 34.8 (42) | 91.0 (4) | 26 |
| GL3 | 98.4 (8) | 32.9 (52) | 81.6 (4) | 44 |
| GL5 | 98.4 (8) | 28.3 (102) | 74.6 (6) | 63 |
| PQ0.5 | 49.7 (70) | 28.6 (100) | 56.8 (72) | 91 |
| **PQ1** | **45.0** (105) | **28.3** (105) | **55.0** (88) | **116** |
| PQ2 | 45.0 (118) | 28.6 (142) | 54.5 (146) | 141 |
| PQ3 | 45.0 (124) | 28.6 (144) | 54.5 (146) | 148 |
| UP2 | 53.3 (27) | 32.7 (32) | 62.6 (62) | 64 |
| UP3 | 49.8 (38) | 32.8 (96) | 55.9 (126) | 103 |
| UP4 | 46.4 (105) | 29.5 (143) | 55.9 (140) | 133 |
| oracle | 43.6 (150) | 29.0 (150) | 53.4 (150) | — |

| criterion | 5y electronics | epochs trained |
| --- | --- | ---: |
| current: patience 7 | 43.5 (32) | 41 |
| GL1 | 55.4 (9) | 15 |
| GL2 | 42.8 (23) | 28 |
| GL3 | 33.5 (24) | 62 |
| GL5 | 29.0 (116) | 102 |
| PQ0.5 | 29.8 (37) | 48 |
| **PQ1** | **24.6** (67) | **75** |
| PQ2 | 23.1 (93) | 115 |
| PQ3 | 23.1 (104) | 137 |
| UP2 | 23.1 (81) | 96 |
| UP3 | 23.0 (113) | 134 |
| UP4 | 23.0 (132) | 144 |
| oracle | 23.5 (100) | — |

PQ1 in full (MAPE / |bias| % / Spearman / collapsed trials of 20):

| panel | current: patience 7 | PQ1 | oracle |
| --- | --- | --- | --- |
| 3y electronics | 71.0 / 60.9 / 0.02 / 15 | **45.0 / 14.5 / 0.11 / 11** | 43.6 / 10.8 / 0.18 / 8 |
| 3y gift | 47.8 / 18.8 / 0.01 / 19 | **28.3 / 10.2 / 0.06 / 13** | 29.0 / 10.3 / 0.35 / 7 |
| 3y multichannel | 74.6 / 53.2 / −0.01 / 4 | **55.0 / 26.2 / −0.01 / 2** | 53.4 / 21.9 / 0.02 / 0 |
| 5y electronics | 43.5 / 30.9 / 0.20 / 7 | **24.6 / 15.0 / 0.30 / 3** | 23.5 / 12.3 / 0.34 / 0 |

- **PQ is the best family again, on every new panel.** PQ1 closes 95% of the gap between
  the current rule and the oracle on 3y electronics, 93% on 3y multichannel and 95% on
  5y electronics, and beats the oracle on 3y gift. PQ2 is within 1.5 MAPE points of PQ1
  on every new panel, at 22–53% more epochs.
- **GL fails again** on electronics and multichannel, for the same reason: the noisy
  early loss trips it within 4–8 epochs. It does well only on 3y gift.
- **UP is close on 5y** (UP2–UP4 at 23.0–23.1) but weak on 3y electronics and gift with
  small s, so it is again less consistent than PQ.
- **Patience 7 costs the most on the long windows' forecast, not their loss.** On 5y it
  already runs to epoch 39 and ends only 1.9% above the loss minimum, yet its forecast is
  43.5 against 24.6 under PQ1: the last 2% of the loss carries half of the forecast error.
- **CE becomes a better guide with longer calibration.** Early CE predicts the final
  customer ranking with ρ mostly 0.7–0.9 on 3y electronics (dipping to 0.2–0.3 at epochs
  20–30), 0.5–0.8 on 3y multichannel, 0.6–0.8 on 5y and 0.3–0.7 on 3y gift, against
  negative or unstable values on 2y CDNOW and electronics. It still
  does not predict the level everywhere (CE → final MAPE is negative on 3y gift), and the
  CE ranking of trials settles only at epoch 44–65 (3y) and 47 (5y). Pruning before about
  epoch 50 stays unjustified.
- **On the 5y panel the collapse goes away with training.** With 3,755 customers, no
  trial is collapsed at any checkpoint from epoch 75 on (3 of 20 are where PQ1 stops
  earlier), and PQ1 reaches Spearman 0.30: the panel size, not only training
  length, decides whether ValendinLSTM learns to tell customers apart.

![3y electronics](figures/epoch-probe/cal3y/electronics.png)

![3y gift](figures/epoch-probe/cal3y/gift.png)

![3y multichannel](figures/epoch-probe/cal3y/multichannel.png)

![5y electronics](figures/epoch-probe/cal5y/electronics.png)

**Conclusion.** Calibration data alone settles both: the multi-fidelity assumption fails
on the validation curves, so pruning is off; and of Prechelt's published stopping criteria,
PQ performs best on the validation forecast, with PQ1 the cheapest of the best. That holds
on all eight panel-calibrations tested (2y × 4, 3y × 3, 5y × 1), including a second,
later validation window for electronics, gift and multichannel. PQ1 closes at least 85% of
the gap between the current rule and the best fixed epoch on every one, and beats the best
fixed epoch on three. What remains is **one
holdout test** with the settings frozen: test 1.

