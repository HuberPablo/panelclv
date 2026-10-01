# Training efficiency: the models stop training too early

Every neural result in this package comes from a model that stopped training while its
validation loss was still falling. This doc covers:
- how that was established;
- why it happens;
- what was tested to fix it, and what it changed;
- what is left to try.

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
| 4 | Under patience 7 the search prefers the batch size that trains least | electronics | established, descriptive | 3.5 |
| 5 | Training to the paper's epoch count improves level and ranking | electronics | **established** (ValendinLSTM MAPE −22.3, Spearman +0.150) | 5.1 |
| 6 | The paper's settings without its epoch count change nothing | electronics | no clear difference | 5.1 |
| 7 | A training floor improves ranking | electronics, multichannel | **established** | 5.2 |
| 8 | …and worsens it | gift | **established** (−0.069) | 5.2 |
| 9 | The floored paper recipe triples CDNOW's LSTM error | cdnow | **established** (+126.2) | 5.2 |
| 10 | …and the floor alone is not the cause | cdnow | floor alone: no clear difference (E1) | 5.3 |
| 11 | On a 260-week window a floor tightens the spread but does not move the mean | electronic_5y | no clear difference in means | 5.4 |

## 1. How training stops

`fit_model` (`training/loop.py`) counts an epoch as an improvement only if
`val_loss + 1e-4 < best_val_loss`. It stops after `patience` (7) epochs without one and
restores the best epoch's weights.
- `best_epoch` is stored per trial: it is the epoch whose weights were kept.
- The number of epochs actually run is not stored. With no floor and a completed trial
  it is `best_epoch + 1 + patience`.
- When comparing runs, condition on **gradient updates** received,
  `ceil(N / batch_size) × (best_epoch + 1)`, not on epochs.

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

### 3.1 The reference notebook trains about 70× longer

| | Valendin et al. notebook | Our studies |
| --- | --- | --- |
| optimizer | Adam defaults: lr 1e-3, no weight decay | AdamW; lr searched 1e-4–3e-3, weight decay 1e-6–1e-2 |
| batch size | 32 | searched over {64, 128, 256} |
| patience / budget | 5 / 150 epochs | 7 / 100 epochs |
| epochs trained | ~90 ("final validation loss … after ±90 epochs") | median best epoch 7–14 |

On electronics (829 customers), 90 epochs at batch 32 is about **2,300 gradient updates**.
Our benchmark winners stopped at a median epoch of 7 at batch 256, about **32 updates**.
`scripts/validate_valendin_lstm.py` runs the notebook's recipe exactly and reaches the
published loss. The search space simply cannot express that recipe: batch 32 is absent
and weight decay has no zero.

### 3.2 The validation loss keeps falling

The frozen benchmark was rerun with early stopping off for 200 epochs, at an archived
electronics winner's settings, 3 replications per panel:

| panel | patience 7 stops at | its best val CE | best over 200 epochs | at epoch | improvement forgone |
| --- | ---: | ---: | ---: | ---: | ---: |
| electronics | 19 | 0.0888–0.0896 | 0.0848–0.0858 | 137–197 | 3.4–5.3% |
| multichannel | 12–16 | 0.0229–0.0230 | 0.0220–0.0226 | 169–181 | 1.4–4.4% |
| cdnow | 11 | 0.1078–0.1098 | 0.0946–0.0964 | 88–120 | 10.7–13.0% |

The curve sits flat and then drops. Before reaching its optimum each run went 29–131
epochs without a single counted improvement, so no patience constant is reliable: at
patience 40, one of five electronics runs found the later optimum.
*(`.scratch/training-budget/probe_patience.py`)*

### 3.3 Why the curve looks flat: the threshold is larger than the slope

Electronics, early stopping off, the notebook's recipe:

| split | val CE at epoch 0 → 1 | mean gain per epoch after epoch 1 |
| --- | --- | ---: |
| temporal (ours, ADR-0001) | 0.1303 → 0.0935 | **8.1×10⁻⁵** |
| customer-wise (the notebook's) | 0.2433 → 0.1623 | 5.2×10⁻⁴ |

Under the temporal split the average epoch improves by **less than the 10⁻⁴ it must
clear**; under the customer-wise split, by five times that threshold. The temporal slope
is the run on disk (`.scratch/training-budget/results/why_flat.csv`, best 0.0887 at epoch
61). An earlier run reported 5.4×10⁻⁵, but its output was overwritten.

Epoch 1 is not stuck at initialisation. It trades zero-cell accuracy for purchase-cell
accuracy (CE on positive cells 9.51 → 5.87). Everything after that is slow refinement on
the 1.4% of cells that carry 84–87% of the loss, while the forecast keeps improving.

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

| batch | completed | best val CE | median best epoch | median updates |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 136 | 0.0899 | 3 | 52 |
| 128 | 178 | 0.0911 | 4 | 35 |
| 256 | 399 | 0.0886 | 7 | 32 |

Within the 40 pre-experiment electronics studies with no label, more training goes with
better ranking: Spearman's rank correlation with updates received is **+0.328** (+0.024,
+0.580; best epoch +0.280, not supported). It is one across-study correlation, resampled by
study.

Batch 256 won all 20 studies. Small-batch trials need more epochs to show their
advantage, and patience 7 stops them first. Inside a study, the longest-trained trial is
the best pick of any criterion on electronics (`docs/model-selection.md` §3.5).
*(`.scratch/training-budget/trials_by_bs.py`)*

## 4. What was tested

| Experiment | Panels | Arms | Size |
| --- | --- | --- | --- |
| **Family T** (`scripts/run_training_budget.py`, 20 Sep) | electronics | `archive`: lr/wd/batch searched, patience 7, 100 epochs, 100 trials · `paper`: lr 1e-3, wd 0, batch 32, patience 5, 150 epochs, 1 trial · `paper90`: `paper` + `min_epochs=90` · `floor50`: `archive` + `min_epochs=50`, 300 epochs | 2 models × 4 arms × 20 replications, 200 paths |
| **Family U** (`scripts/run_factorial.py`, 21 Sep) | all four | `archive` vs `floored` (= `paper90`), crossed with `no_cluster` / `kmeans_8` | 2 models × 4 cells × 20 replications. Only the floor half is reported here; the label half is in `docs/insights-real-panels.md` |
| **E1** (`scripts/run_training_budget.py`; `.scratch/model-selection/e1_cdnow.py`) | cdnow | `archive`, `floor50`, `paper`: each training ingredient as the only change | 2 models × 3 arms × 20 replications |
| **Epoch floor** (`scripts/run_epoch_floor_5y.py`, 26–28 Sep) | electronic_5y | the least-biased searched study's settings pinned (`nofloor`), then weights kept only from epoch 20 (`from20`) or 30 (`from30`) via `select_from_epoch` | 3 models × 3 rules × 20 replications, 500 paths |

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
  keeps epoch 1 and stops after 7 (§3.3). Adding the epoch count cuts MAPE by 22 points
  and multiplies ranking by about seven.
- **The invariant is not identified.** `paper90` gets there with ~2,300 updates at batch
  32, and `floor50` with a few hundred at batch 256.
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
| The paper's recipe | Tested (§5.1). Only works with its epoch count added. |
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

## 8. Limits

- Four panels plus electronic_5y, each stated separately. No measured panel
  characteristic predicts where a floor helps.
- §3.2–3.4 rest on 3 runs per panel at one hyperparameter setting.
- Training is unseeded, but a replication's forecast seed fixes the RNG entering the next
  replication's training. Arms doing identical work in identical order are therefore
  coupled; the archive shows 0 exact repeats across 3,020 suites, so it is unaffected.
