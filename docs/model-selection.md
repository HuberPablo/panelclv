# Model selection: what the search picks, and what could pick better

Every neural result in this thesis is the output of one choice made per replication.
Optuna trains up to 100 trials, keeps the one with the lowest **teacher-forced validation
cross-entropy** (CE) on the temporal validation window (ADR-0001), refits it for 5 epochs
over the full calibration window (ADR-0008), and rolls it out over the 52-week holdout.
This document puts in one place what has been **measured** about that choice, and then
the remedies, each with whatever evidence exists for it.

It collects results spread over `docs/training-budget.md` (§5, §13, §14, §15.3),
`docs/benchmarks-real-panels.md` ("Optuna's validation loss cannot see which runs forecast
badly", "Does the search select the best trial?") and `docs/insights-study.md` (§5.4, §9).
Three analyses are new here, all run on 24 September 2026 on the existing selection
rescore (no new training): a per-panel re-check with bootstrap intervals, the holdout
score of each criterion's pick against a random trial, and an offline test of two-stage
selection. Scripts in `.scratch/model-selection/`.

**Standard.** The one `docs/training-budget.md` defines under "How claims are made": an
effect is a mean or a paired difference with a 95% bootstrap interval, supported when the
interval excludes zero, stated per panel and never generalised across panels. The refit
noise (next table) is printed as a magnitude reference, not a second threshold.

| panel | refit noise: MAPE | \|bias\| % | Spearman |
| --- | ---: | ---: | ---: |
| cdnow | 5.83 | 12.58 | 0.0159 |
| electronics | 3.63 | 5.94 | 0.0105 |
| gift | 5.89 | 14.99 | 0.0116 |
| multichannel | 7.71 | 12.56 | 0.0152 |

**Correlation sign convention.** Every criterion and target is oriented so lower is
better, so a **positive** rank correlation means the criterion orders trials the way the
holdout does.

## Contents

- [Part I — The problems, as measured](#part-i--the-problems-as-measured)
  1. [The criterion scores a different task, on a different model](#1-the-criterion-scores-a-different-task-on-a-different-model)
  2. [It is flat exactly where the choice is made](#2-it-is-flat-exactly-where-the-choice-is-made)
  3. [It works on some panels and is wrong-signed on others](#3-it-works-on-some-panels-and-is-wrong-signed-on-others)
  4. [What decides which: whether the trials differ at all](#4-what-decides-which-whether-the-trials-differ-at-all)
  5. [Bias is never selected](#5-bias-is-never-selected)
  6. [The search rewards models that trained least](#6-the-search-rewards-models-that-trained-least)
  7. [The refit sets the resolution](#7-the-refit-sets-the-resolution)
  8. [What is at stake at the pick](#8-what-is-at-stake-at-the-pick)
  9. [A possible leak into selection](#9-a-possible-leak-into-selection)
- [Part II — Possible solutions](#part-ii--possible-solutions)
- [Part III — Recommendation](#part-iii--recommendation)
- [What this does not establish](#what-this-does-not-establish)

---

# Part I — The problems, as measured

## 1. The criterion scores a different task, on a different model

Two mismatches sit between the number Optuna minimises and the number the thesis
reports. Both are properties of the pipeline, read from the code:

| | what selection scores | what is reported |
| --- | --- | --- |
| **task** | one-step-ahead CE, true previous counts fed in (teacher forcing) | 52-week rollout, the model's own sampled counts fed back in |
| **model** | the trial's own checkpoint | a 5-epoch unseeded refit of that checkpoint (`DEFAULT_REFIT_EPOCHS`, `trials/refit.py:33`) |
| **metric** | cross-entropy over all cells | aggregate MAPE, bias, per-customer Spearman |

A model can be good next-step and drift over a long horizon, because its errors become
its inputs. ADR-0003 once guarded against that by selecting on a validation rollout; it
was retired on 12 August 2026 for implementation defects, not for the idea, and nothing
has replaced it.

## 2. It is flat exactly where the choice is made

- **The winning validation loss barely varies.** Its coefficient of variation across the
  replications of one cell is 0.08–3.1%, and the median study has 1–22 other trials within
  0.5% of its winner (`docs/benchmarks-real-panels.md`). Holdout bias across the same
  winners spans tens of points.
- **The curve itself is nearly flat.** One electronics run, early stopping off: after the
  first epoch the temporal validation CE improves by 5.4×10⁻⁵ per epoch on average,
  against the 10⁻⁴ that `fit_model` requires to count an epoch as an improvement
  (`docs/training-budget.md` §13.2).
- **The loss lives in 1.4% of the cells.** Of the 43,108 electronics validation cells,
  589 carry a purchase, and they carry 86% of the CE. After epoch 1 the model is refining
  those few cells slowly while the forecast keeps improving for another ~200 epochs (§13.2
  there, family T).

So a large share of the ± sd the tables report is which near-tied trial happened to win,
not what the architecture can do. More trials will not narrow it: they search a flat
objective more finely.

## 3. It works on some panels and is wrong-signed on others

Two independent rescores took non-winning trials through the production path (refit and
holdout rollout) and asked whether validation CE orders them the way the holdout does.

**The 698-trial rescore** (ValendinLSTM, 36 studies with ≥5 scored trials, 17 September;
`docs/benchmarks-real-panels.md`). Mean per-study rank correlation, and where Optuna's
actual pick lands among its own study's trials (0 = best forecast, 1 = worst, 0.5 = coin
toss). No intervals were computed for this table.

| panel | studies | rho RMSE | rho Spearman | rho \|bias\| | winner's place, Spearman | winner's place, \|bias\| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gift | 14 | +0.83 | +0.56 | +0.09 | 0.12 | 0.49 |
| cdnow | 13 | +0.61 | +0.67 | +0.38 | 0.38 | 0.46 |
| electronics | 2 | −0.31 | +0.44 | −0.25 | 0.13 | 1.00 |
| multichannel | 7 | −0.29 | +0.01 | −0.21 | 0.65 | 0.55 |

**The 3,042-trial selection rescore** (LSTM and ValendinLSTM, 40 random trials per study,
21 September; re-checked here per panel, `.scratch/model-selection/per_panel.py`):

| target | electronics, 80 studies | cdnow, 10 studies |
| --- | ---: | ---: |
| holdout MAPE | **−0.141** (−0.187, −0.093) | **+0.244** (+0.073, +0.417) |
| holdout \|bias\| | **−0.264** (−0.302, −0.225) | +0.176 (−0.029, +0.385), not supported |
| holdout Spearman | +0.045 (−0.005, +0.095), not supported | **+0.472** (+0.342, +0.598) |

Read together:

- **On gift and CDNOW, CE is a real proxy for ranking.** Every gift study agrees in sign
  on RMSE, the pick lands in the best eighth for Spearman, and the median gift trial ranks
  customers at 0.019 where the pick ranks them at 0.374.
- **On electronics and multichannel it is wrong-signed for level.** Across a study's
  trials, lower CE goes with *higher* holdout MAPE and |bias|. Multichannel's pick sits
  behind a coin toss for ranking (0.65).
- **Bias is not controlled anywhere** — see §5.

The two panels where selection works are the two where the models do not collapse
(forecast CV 0.54–1.18 against 0.08–0.15, `docs/training-budget.md` §15.1).

## 4. What decides which: whether the trials differ at all

`docs/training-budget.md` §14.2 first explained the wrong sign as a calibration-to-holdout
rate shift. §15.3 tested that and **retracted** it: every panel's holdout rate is below
its calibration rate (ratios 0.23–0.63), and CDNOW, where CE works, has the steeper
decline. What separates the panels is how different a study's trials are from each other.

Holdout outcome by quartile of a trial's own validation CE, within its study:

| quartile | electronics bias / MAPE / Spearman | cdnow bias / MAPE / Spearman |
| --- | ---: | ---: |
| best CE | +35.1 / 65.3 / 0.02 | +14.3 / 65.1 / 0.34 |
| 2nd | +33.4 / 66.5 / 0.02 | +10.2 / 55.4 / 0.34 |
| 3rd | +29.4 / 64.5 / 0.02 | +50.4 / 97.9 / 0.29 |
| worst CE | +22.0 / 62.7 / 0.02 | +98.7 / 114.2 / 0.16 |

A CDNOW study contains broken trials that miss by +99%, and CE finds them. An electronics
study does not: the new measurement below puts the within-study spread of holdout
outcomes next to the refit noise.

| panel | within-study IQR: MAPE | bias % | Spearman |
| --- | ---: | ---: | ---: |
| cdnow | 62.2 (refit noise 5.83) | 110.5 (12.58) | 0.202 (0.0159) |
| electronics | 10.2 (3.63) | 19.9 (5.94) | **0.021** (0.0105) |

On electronics the trials' Spearman spread is twice what refitting one checkpoint twice
moves. The collapse *is* this: every trial gives every customer nearly the same forecast,
so there is almost nothing to select between, and the small systematic tilt that remains
points the wrong way. **CE selects well where trials differ, and has nothing to work with
where they do not** — which is precisely where it is used to justify a winner.

## 5. Bias is never selected

Over the 698-trial rescore, pooled across panels, the winner's place on |bias| is 0.52 and
rho is +0.12: a coin toss. Yet every study holds a trial with small bias
(`docs/benchmarks-real-panels.md`):

| panel | \|bias %\| selected / best trial / median trial |
| --- | --- |
| gift | 16.9 / 4.3 / 19.2 |
| cdnow | 34.6 / 7.7 / 40.4 |
| electronics | 64.7 / 12.2 / 37.8 |
| multichannel | 48.9 / 1.6 / 42.7 |

The best-trial column is optimistic — it is the minimum of many draws each carrying 6–15
points of refit noise — but the gap to the pick is far larger than that. Bias is the
metric the arm tables lead with, and it is the one the objective does not point at.

## 6. The search rewards models that trained least

- **Batch size.** In the 20 archived electronics benchmark studies (2,000 trials), batch
  256 won all 20. It trains least per epoch, and under patience 7 it reaches the lowest
  loss before small-batch trials get the epochs they need. Best loss over all 2,000
  trials: 0.0886. One model trained longer: 0.0846 (`docs/training-budget.md` §5).
- **Very early winners.** On multichannel the selected trial had stopped at epoch ≤ 3 in
  55% of ValendinLSTM runs and 22–41% of LSTM runs, against 0–2% on the other panels. In
  multichannel log, earlier stopping goes with worse ranking (rank correlation 0.59
  between best epoch and Spearman).
- **The stopping epoch carries signal the loss does not.** Within a study, best epoch
  correlates with holdout Spearman at +0.48 on both CDNOW and gift, and with |bias| at
  −0.30 on CDNOW (`docs/benchmarks-real-panels.md`). It is recorded
  (`user_attrs_best_epoch`) but plays no part in the choice.

Selection and stopping are therefore coupled: the criterion prefers what the stopping rule
cut short.

## 7. The refit sets the resolution

Selection scores the checkpoint, but the forecast comes from a 5-epoch refit with no
seed. Refitting the same winner twice, with panel, weights, features and simulation seed
fixed, moves aggregate bias by 6–14 points sd and up to 51 points at worst
(`docs/benchmarks-real-panels.md`, 80 studies). On gift the bias spread across a study's
trials (19.4 sd) barely clears one checkpoint's spread against itself (14.4). Two trials
closer than the refit noise cannot be told apart by any criterion computed before the
refit, however good.

## 8. What is at stake at the pick

The correlations in §3 describe the whole ranking. What the thesis reports is the argmin.
New here: the holdout score of each criterion's pick, minus a random trial's expected
score (the study's trial mean), paired over studies
(`.scratch/model-selection/pick_vs_random.py`). Negative is better for MAPE and |bias|,
positive for Spearman.

**Electronics**, 80 studies. Random trial: MAPE 64.9, |bias| 31.8, Spearman 0.021.

| pick by | Δ MAPE | Δ \|bias\| | Δ Spearman |
| --- | ---: | ---: | ---: |
| val CE (status quo) | −1.50 (−3.22, +0.10) | **+3.24 (+0.39, +5.91)** | +0.01 (+0.01, +0.02) |
| val rollout MAPE | **−3.57 (−5.23, −2.04)** | −1.41 (−4.20, +1.27) | +0.00 |
| val rollout Spearman | −2.19 (−4.66, +0.35) | −2.76 (−6.28, +0.65) | +0.01 (+0.00, +0.02) |
| longest-trained (best epoch) | **−6.04 (−8.09, −3.93)** | **−12.41 (−15.76, −9.08)** | +0.01 (+0.00, +0.01) |

**CDNOW**, 10 studies. Random trial: MAPE 80.2, |bias| 70.9, Spearman 0.282.

| pick by | Δ MAPE | Δ \|bias\| | Δ Spearman |
| --- | ---: | ---: | ---: |
| val CE (status quo) | −20.6 (−50.5, +13.3) | −22.1 (−54.2, +12.7) | **+0.10 (+0.08, +0.12)** |
| val rollout MAPE | +32.5 (−9.6, +73.0) | +34.4 (−9.0, +76.1) | +0.02 (−0.07, +0.10) |
| val rollout \|bias\| | **−41.1 (−60.1, −22.2)** | **−41.1 (−61.3, −21.2)** | +0.06 (−0.01, +0.11) |
| longest-trained (best epoch) | −14.8 (−45.4, +17.4) | −15.2 (−46.8, +17.4) | +0.04 (−0.03, +0.11) |

Three readings:

- **On electronics the status quo costs little at the pick.** Its MAPE is no worse than a
  random trial's; its |bias| is 3.2 points worse, supported, and under the 5.9-point refit
  noise. The wrong sign in §3 is real across the ranking, but its price at the argmin is
  small, because the trials barely differ (§4).
- **The one criterion worth more than the refit noise on electronics is not a validation
  score.** Picking the longest-trained trial beats a random one by 6.0 MAPE and 12.4
  |bias|, both above the refit noise. That is §6 again: training length is the lever.
- **No criterion is safe on both panels.** On CDNOW, validation rollout MAPE picks a trial
  with holdout MAPE 126–221 in 4 of 10 studies. Those trials fit the validation year's
  level well (validation MAPE 11–12) and then explode on the holdout. The criterion that
  is best on electronics is the most dangerous on CDNOW. Ten studies is a small sample;
  the four blow-ups are not.

## 9. A possible leak into selection

The `kmeans_8` cluster label is computed at the last calibration period and broadcast to
every calibration row, validation weeks included (`cluster_features.py:97`,
`panel_dataset.py:961`). Holdout scoring is clean, but early stopping and selection on any
cluster arm see a label that has already seen the window they score. Experiment E2 in
`docs/training-budget.md` (recompute the label before the validation window) is owed and
not run. Until it is, selection results on cluster arms carry this caveat.

---

# Part II — Possible solutions

Each remedy with what it targets from Part I, the evidence for it so far, and its cost.
"Measured" means a number exists on this pipeline; "untested" means none does.

### S1. Select on a validation-window rollout, scored on the reported metric

Simulate the validation window the way the holdout is simulated, and score it with
`compute_forecast_metrics`. Targets §1 (task mismatch).

- **Measured, electronics:** beats CE on its own target — Δ rho +0.174 (+0.123, +0.226)
  for MAPE, +0.083 (+0.011, +0.158) for Spearman. But its own correlation with holdout
  MAPE is +0.033 (−0.017, +0.083): it removes a harmful signal without supplying a useful
  one. At the pick it gains 3.6 MAPE on a random trial (§8).
- **Measured, CDNOW:** every rollout criterion's point estimate is below CE's on every
  target, supported for rollout Spearman against holdout MAPE (Δ −0.112, −0.183 to
  −0.039) and for rollout MAPE against holdout Spearman (Δ −0.187, −0.325 to −0.038).
  Rollout MAPE's pick blows up in 4 of 10 studies (§8).
- **Cost:** one validation rollout per trial, ~29 s on rented GPU. ADR-0003's two defects
  (a second scoring implementation, no wiring into `StudySuiteConfig`) must not return.
- **Verdict:** not a safe replacement. Panel-dependent sign.

### S2. A composite of rollout MAPE, |bias| and Spearman

- **Measured, electronics:** worse than whichever single criterion matches the target
  (+0.039 vs +0.128 for Spearman; −0.069 vs +0.033 for MAPE,
  `docs/training-budget.md` §14.3).
- **Verdict:** rejected. Match the criterion to the metric the claim is about.

### S3. Two-stage: shortlist on CE, re-rank the shortlist by rollout

Keep the trials within *m*% of the best CE and pick among them by a rollout metric.
Cheap — one rollout per shortlisted trial. Tested offline here on the rescore data
(`.scratch/model-selection/two_stage.py`); Δ is the pick's holdout score minus the plain
CE pick's, lower is better.

| panel | m | shortlist (median) | re-rank by | Δ MAPE | Δ \|bias\| |
| --- | ---: | ---: | --- | ---: | ---: |
| electronics | 1% | 14 | rollout MAPE | **−2.05 (−3.77, −0.27)** | **−3.74 (−6.66, −0.80)** |
| electronics | 5% | 34 | rollout MAPE | **−2.06 (−3.94, −0.13)** | **−4.58 (−8.18, −0.95)** |
| electronics | 1% | 14 | rollout Spearman | +1.33 (−1.03, +3.77) | −0.72 (−4.79, +3.37) |
| cdnow | 0.5% | 4.5 | rollout MAPE | +12.9 (−6.0, +43.0) | +13.4 (−5.4, +42.7) |
| cdnow | 1% | 9 | rollout MAPE | **+36.5 (+4.0, +72.7)** | **+39.7 (+6.7, +76.0)** |
| cdnow | 1% | 9 | rollout Spearman | +3.0 (−15.5, +22.9) | +1.4 (−17.4, +21.6) |

- **Verdict:** on electronics a supported gain of ~2 MAPE, under the refit noise; on
  CDNOW a supported loss of ~36 MAPE at a 1% shortlist. Shortlisting does not contain the
  CDNOW blow-ups; it only delays them to a wider margin. Rejected in this form. No
  Spearman effect on either panel.

### S4. Use training length as a criterion, or as a tie-breaker

Targets §6. The best epoch is already recorded.

- **Measured, electronics:** the longest-trained pick beats a random trial by 6.0 MAPE and
  12.4 |bias| (§8), both above the refit noise — the largest supported gain of any
  criterion here.
- **Measured, CDNOW:** no supported effect at the pick; as a ranking it is worse than CE
  for Spearman, Δ −0.153 (−0.282, −0.034).
- **Verdict:** a symptom, not a remedy. It works on electronics because the trials that
  trained longest are the ones the stopping rule let through, which is what S5 fixes at
  the source. Useful as evidence for S5, not as a selection rule.

### S5. Fix the stopping rule so the search stops preferring undertrained models

Targets §2 and §6: a relative improvement threshold, patience counted in optimiser steps,
a smoothed curve, or a fixed step budget with no early stopping
(`docs/training-budget.md` §13.3, B5–B8; experiment E4).

- **Measured indirectly:** a training floor (family T) moved electronics MAPE by 22
  points, an order of magnitude more than any selection rule. But the floored arm triples
  CDNOW's LSTM error (§15.2 there), and whether the floor itself causes that is not
  identified (E1, running).
- **Untested:** every principled alternative to the floor.
- **Verdict:** the highest-value change on the evidence. Run E4, one panel first, then
  CDNOW before adopting anything.

### S6. Average replications instead of trusting one pick

Score the ensemble of a cell's replication forecasts rather than the mean of their
scores. Targets §2 and §7: it averages away both near-tie luck and refit noise instead of
trying to beat them.

- **Measured, CDNOW (family G, 20 studies):** cuts MAPE by 3–12 points, and puts
  ValendinLSTM at 18.13 against Pareto/NBD's 18.70. Bias is unchanged by construction
  (`docs/insights-study.md` §9).
- **Cost:** none; the forecasts exist.
- **Verdict:** adopt for reporting, beside the replication distribution, never instead of
  it. It does not choose better models; it stops one choice from mattering so much.

### S7. Average or seed the refit

Targets §7. Either refit the winner several times and average the forecasts, or seed the
refit so a result is reproducible.

- **Untested.** Averaging *k* refits should shrink the refit noise by about √k at *k*×
  the refit cost; seeding makes a number reproducible without making it more accurate.
- **Verdict:** worth it only if S6 is not used, since S6 already averages across refits.

### S8. Rolling-origin validation

Average the selection criterion over two or three validation cut points instead of one.
Targets §2 (one flat window) and plausibly the CDNOW blow-ups in §8, which fit one
validation year and fail the next.

- **Untested.** Cost: two to three times the validation compute per trial.
- **Verdict:** the one untried selection change aimed at the failure S1 and S3 exposed.

### S9. Make the trials worth distinguishing first

Targets §4. Where the model collapses, every trial gives every customer the same forecast
and no rule can select between them. A persistent per-customer input (the cluster label)
lifts electronics Spearman from 0.03 to 0.305, against 0.178 from training longer
(`docs/training-budget.md` §6, §15.1); `docs/absorbing-death-state.md` argues for a
learned alternative.

- **Verdict:** a precondition, not a selection rule. On a collapsed panel it is the only
  change here that gives selection something to choose from.

---

# Part III — Recommendation

1. **Do not replace validation CE with any single rollout criterion yet.** The best one on
   electronics (rollout MAPE, directly or as a two-stage re-rank) is the most harmful on
   CDNOW. Selection is panel-dependent, and the thesis cannot pick a panel.
2. **Put the effort into training length first (S5).** It is worth an order of magnitude
   more than selection on electronics, and training length is what the one strong
   "criterion" in §8 was really measuring.
3. **Report the ensemble beside the distribution (S6).** It costs nothing and neutralises
   the near-tie and refit noise that §2 and §7 measure.
4. **If selection is revisited, test rolling-origin validation (S8)** on the existing
   rescore data first, on both panels, before building anything.
5. **Close E2** before any cluster-arm result is used to argue about selection.
6. **In the thesis text, state plainly:** the reported winner is the argmin of a
   criterion that ranks trials correctly on gift and CDNOW, wrong-signed for level on
   electronics and multichannel, and blind to bias everywhere; on electronics that costs
   about 3 points of |bias| at the pick.

## What this does not establish

- **Anything about panels as a class.** The selection rescore covers two panels, one of
  them with 10 studies. Gift and multichannel appear only in the 698-trial ValendinLSTM
  rescore, which has no intervals.
- **How the new pick-level results would change on a floored or cluster-labelled model.**
  Every trial here trained under patience 7 and no floor; S5 would change the trial
  population, and with it every number in §8.
- **That the CDNOW blow-ups are a property of rollout selection rather than of these ten
  studies.** Four of ten is not a rate.
- **That the oracle columns are attainable.** They are the minimum over many noisy
  refits, and part of their advantage is luck.
