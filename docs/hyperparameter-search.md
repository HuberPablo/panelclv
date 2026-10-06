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
57b8c18 pinned weight decay to 0 and added batch 32. Every family run before then (A to V)
ran on the old
space, so a rerun meant to reproduce one must restore it (family U′ does, §4).

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

## 5. Next tests, in order

1. **Multichannel rerun** with the fixed score: `archive` · no cluster label, both models,
   20 replications (about 4 hours locally). Its bias was the largest (−6.2%).
2. **Pruner comparison** on electronics · `archive` · no cluster label: current rule,
   Hyperband, off; 20 replications each.
3. **Training floor** for the LSTM on the same cell, on top of the fix.
4. **Refit learning rate:** checkpoint against refit forecasts, by tuned learning rate.

Each test changes one thing against the fixed-score baseline, and each is compared on
MAPE, bias and Spearman with independent bootstrap intervals.
