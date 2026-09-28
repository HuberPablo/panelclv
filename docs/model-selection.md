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
selection. Scripts in `.scratch/model-selection/`; the per-study correlations come from
`.scratch/training-budget/selection_analysis.py`.

> **Revised under the statistical protocol (2026-09-28).** Every interval is now
> `panelclv.evaluation.effects.effect` — 95% percentile bootstrap, 10,000 resamples,
> paired over studies. The rescore's electronics studies pooled two models (40 LSTM + 40
> ValendinLSTM) and CDNOW's pooled 5 + 5; each model is now its own condition, so every
> claim is stated per panel *and* model, with n = 40 or n = 5. Verdicts that flipped:
> - §3, electronics, CE against holdout Spearman: not supported → supported and weakly
>   right-signed for the LSTM (+0.092); still not supported for ValendinLSTM.
> - §3/§5, CDNOW, CE against holdout MAPE: supported → supported for the LSTM only; against
>   |bias|: not supported → **supported for the LSTM** (+0.356), so "nowhere is CE shown to
>   select for bias" no longer holds.
> - §8, electronics, CE pick vs a random trial: MAPE not distinguishable → **better** for
>   ValendinLSTM (−3.38), not distinguishable for the LSTM; |bias| worse (supported) →
>   worse for ValendinLSTM only (+4.39), not distinguishable for the LSTM.
> - §8, CDNOW: rollout |bias| pick's |bias| gain holds for the LSTM, not for ValendinLSTM;
>   rollout Spearman and longest-trained picks now beat a random trial on Spearman for
>   ValendinLSTM (+0.103, +0.108).
> - S1: rollout MAPE's own correlation with holdout MAPE, not supported → not supported
>   for the LSTM, supported for ValendinLSTM (+0.107); the four supported CDNOW deficits
>   split two per model.
> - S3: the electronics gains of a rollout-MAPE re-rank are the LSTM's (supported at every
>   margin on MAPE); for ValendinLSTM none is supported. The CDNOW losses are the LSTM's
>   (+61 to +92 MAPE from 1%); for ValendinLSTM they are not supported. The
>   "a few intervals will exclude zero by chance" multiplicity note is removed: each row
>   is its own claim and the pattern is described in words.
> - S4, CDNOW: longest-trained worse than CE as a ranking for Spearman holds for the LSTM
>   only.
> - S5: E1 now has a script (`.scratch/model-selection/e1_cdnow.py`); its verdicts are
>   unchanged.
>
> Not re-tested: the 698-trial ValendinLSTM rescore table in §3 and the figures quoted
> from `docs/benchmarks-real-panels.md` in §5–§7 (descriptive, no per-study data here).

**Standard.** `docs/statistical-protocol.md`: an effect is a mean or a paired difference
with a 95% percentile-bootstrap interval, supported when the interval excludes zero,
stated per panel and model and never pooled across them. The study — one complete Optuna
search — is the replication; its trials are not. A per-study rank correlation is one
statistic per replication, tested against 0; two criteria scored on the same studies are
compared paired. The refit noise (next table) is printed as a magnitude reference, not a
second threshold. It is the mean |difference| between two unseeded refits of the same
winner, over 20 archived ValendinLSTM winners per panel
(`.scratch/training-budget/refit_noise.py`), so where it is set beside LSTM trials below
it is a borrowed yardstick, not a measurement of them. An interval that includes zero is
reported as "no clear difference at n = …", never as "no worse" or "no effect": it is
not evidence of equivalence. A table without intervals is descriptive and supports no
claim on its own.

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
  4. [What separates the two rescored panels: whether the trials differ at all](#4-what-separates-the-two-rescored-panels-whether-the-trials-differ-at-all)
  5. [Bias is not what the criterion selects for](#5-bias-is-not-what-the-criterion-selects-for)
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
  replications of one cell is 0.08–3.1%, and the median study has 0–21 other trials within
  0.5% of its winner, across the 32 archived cells of `docs/benchmarks-real-panels.md`
  (4 ValendinLSTM and 28 LSTM cells, 2,880 winners). Holdout bias across the same winners
  spans tens of points.
- **The curve itself is nearly flat.** Electronics, early stopping off: between epoch 1
  and the best epoch the temporal validation CE improves by 8.1×10⁻⁵ per epoch in the one
  run whose output is on disk (`.scratch/training-budget/results/why_flat.csv`, best
  0.0887 at epoch 61), under the 10⁻⁴ that `fit_model` requires to count an epoch as an
  improvement (`training/loop.py:353`). `docs/training-budget.md` §13.2 reports 5.4×10⁻⁵
  from an earlier run whose output was overwritten.
- **The loss lives in 1.4% of the cells.** Of the 43,108 electronics validation cells,
  589 carry a purchase, and in that same run they carry 84–87% of the CE (85.7% at epoch
  1, 84.5% at the best epoch). After epoch 1 the model is refining
  those few cells slowly while the forecast keeps improving for another ~200 epochs (§13.2
  there, family T).

So a large share of the ± sd the tables report is which near-tied trial happened to win,
not what the architecture can do. More trials will not narrow it: they search a flat
objective more finely.

## 3. It works on some panels and is wrong-signed on others

Two independent rescores took non-winning trials through the production path (refit and
holdout rollout) and asked whether validation CE orders them the way the holdout does.

**The ValendinLSTM rescore** (17 September, `docs/benchmarks-real-panels.md`): 698 refits
over the 80 benchmark studies, of which 653 trials in the 36 studies with ≥5 scored
trials make this table. Mean per-study rank correlation, and where Optuna's actual pick
lands among its own study's trials, as (rank − 1)/(n − 1): 0 = best forecast, 1 = worst,
0.5 = coin toss. *(`docs/benchmarks-real-panels.md` prints rank/n under the same
description, which reads worse than this for small studies.)* No intervals were computed
for this table, so it is descriptive and supports no claim on its own; electronics rests
on 2 studies, and its RMSE column supports no ranking claim under the standard.

| panel | studies | rho RMSE | rho Spearman | rho \|bias\| | winner's place, Spearman | winner's place, \|bias\| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gift | 14 | +0.83 | +0.56 | +0.09 | 0.04 | 0.45 |
| cdnow | 13 | +0.61 | +0.67 | +0.38 | 0.33 | 0.41 |
| electronics | 2 | −0.31 | +0.44 | −0.25 | 0.02 | 1.00 |
| multichannel | 7 | −0.29 | +0.01 | −0.21 | 0.64 | 0.53 |

**The 3,042-trial selection rescore** (LSTM and ValendinLSTM, 21 September; re-checked
here per panel and model, `.scratch/training-budget/selection_analysis.py`). Each study
scores a random sample of up to 40 of the trials whose checkpoints were kept: median 39
(16–40) on electronics, 13–34 on CDNOW. Optuna's own winner is outside the sample in 6 of
the 80 electronics studies, so there "val CE" means the sample's best CE, not the reported
pick. Mean per-study rank correlation of val CE with each target, with its 95% interval;
CDNOW has 5 studies per model, and a percentile interval over five values runs narrow.

| target | electronics LSTM (n = 40) | electronics ValendinLSTM (n = 40) | cdnow LSTM (n = 5) | cdnow ValendinLSTM (n = 5) |
| --- | ---: | ---: | ---: | ---: |
| holdout MAPE | **−0.181** (−0.247, −0.112) | **−0.101** (−0.162, −0.038) | **+0.338** (+0.112, +0.553) | +0.149 (−0.050, +0.392) |
| holdout \|bias\| | **−0.266** (−0.323, −0.207) | **−0.261** (−0.310, −0.213) | **+0.356** (+0.128, +0.594) | −0.004 (−0.257, +0.255) |
| holdout Spearman | **+0.092** (+0.013, +0.168) | −0.001 (−0.060, +0.058) | **+0.367** (+0.190, +0.549) | **+0.577** (+0.479, +0.718) |

Bold: interval excludes zero.

Read together:

- **On CDNOW, CE is a real proxy for ranking**, supported on both models: +0.367 (LSTM)
  and +0.577 (ValendinLSTM) against holdout Spearman. For level it is supported only for
  the LSTM (+0.338 against MAPE, +0.356 against |bias|); ValendinLSTM shows no clear
  relationship at n = 5. Gift points the same way for ranking (the pick is nearly its
  study's best trial for Spearman, and the median gift trial ranks customers at 0.019
  where the pick ranks them at 0.374), but only in the table without intervals.
- **On electronics it is wrong-signed for level**, supported on both models: across a
  study's trials, lower CE goes with *higher* holdout MAPE (−0.181, −0.101) and |bias|
  (−0.266, −0.261). For ranking it is weakly right-signed for the LSTM (+0.092) and shows
  no clear signal for ValendinLSTM. Multichannel shows the same signs on RMSE and |bias|,
  and a pick behind a coin toss for ranking (0.64), but only over 7 studies without
  intervals; it is not established there.
- **Bias** — see §5.

The two panels where selection works best are the two where the models do not collapse:
forecast CV of the family U archive winners without a cluster label is 0.53–1.18 on cdnow
and gift against 0.08–0.15 on electronics and multichannel (`docs/training-budget.md`
§15.1, recomputed from its 640 suites' predictions). That is a third population of models
beside the two rescores, so the link is by panel, not by trial.

## 4. What separates the two rescored panels: whether the trials differ at all

`docs/training-budget.md` §14.2 first explained the wrong sign as a calibration-to-holdout
rate shift. §15.3 tested that and **retracted** it: every panel's holdout mean count per
customer-week is below its calibration one (ratios 0.23–0.63; on electronics a count is
line items, and the share of weeks with any purchase falls 0.56-fold rather than
0.63-fold), and CDNOW, where CE works, has the steeper decline (0.40 against 0.63). What
separates electronics from CDNOW is how different a study's trials are from each other.
With two panels this is a description of those two, not an established mechanism.

Holdout outcome by quartile of a trial's own validation CE, within its study
(descriptive means, both models, `.scratch/model-selection/per_panel.py`):

| quartile | electronics bias / MAPE / Spearman | cdnow bias / MAPE / Spearman |
| --- | ---: | ---: |
| best CE | +35.1 / 65.3 / 0.02 | +14.3 / 65.1 / 0.34 |
| 2nd | +33.4 / 66.5 / 0.02 | +10.2 / 55.4 / 0.34 |
| 3rd | +29.4 / 64.5 / 0.02 | +50.4 / 97.9 / 0.29 |
| worst CE | +22.0 / 62.7 / 0.02 | +98.7 / 114.2 / 0.16 |

A CDNOW study contains broken trials that miss by +99%, and CE finds them. An electronics
study does not: the new measurement below puts the within-study spread of holdout
outcomes next to the refit noise (descriptive, both models, `per_panel.py`).

| panel | within-study IQR: MAPE | bias % | Spearman |
| --- | ---: | ---: | ---: |
| cdnow | 62.2 (refit noise 5.83) | 110.5 (12.58) | 0.202 (0.0159) |
| electronics | 10.2 (3.63) | 19.9 (5.94) | **0.021** (0.0105) |

The two columns are different statistics: the IQR spans trials and already contains
refit noise, while the refit noise is a mean |difference| between two refits. On a normal
approximation, refit noise alone would give an electronics Spearman IQR of about 0.013,
so the trials' spread there is about 1.7 times what refitting alone would produce. The
collapse *is* this: every trial gives every customer nearly the same forecast, so there
is little to select between, and the small systematic tilt that remains points the wrong
way. **On these two panels, CE selects well where trials differ and has little to work
with where they do not** — which is where it is used to justify a winner.

## 5. Bias is not what the criterion selects for

Per panel and model, CE's ordering of trials by |bias| is supported wrong-signed on
electronics for both models (rho −0.266 LSTM, −0.261 ValendinLSTM, §3). On CDNOW it is
supported right-signed for the LSTM (+0.356, +0.128 to +0.594, n = 5) and shows no clear
relationship for ValendinLSTM (−0.004, −0.257 to +0.255, n = 5). Gift (+0.09) and
multichannel (−0.21) have no intervals. So the one place CE is shown to select for bias
is CDNOW's LSTM, on five studies; on electronics it selects against it on both models.
Yet every study holds a trial with small bias (`docs/benchmarks-real-panels.md`, no
intervals):

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

*(§6's figures are quoted from `docs/benchmarks-real-panels.md` and the archive; they are
descriptive, carry no intervals, and were not re-tested under the statistical protocol.)*

- **Batch size.** In the 20 archived electronics benchmark studies (2,000 trials), batch
  256 won all 20. It trains least per epoch, and under patience 7 it reaches the lowest
  loss before small-batch trials get the epochs they need. Best loss over all 2,000
  trials: 0.08859; the one run with early stopping off whose output is on disk reaches
  0.0887 (§2). *(`training-budget.md` §5's "0.0846 trained longer" has no stored output
  behind it and is not repeated here.)* The selection rescore shows the same thing inside
  studies: on electronics the longest-trained trial is the best pick of any criterion (§8).
- **Very early winners.** On multichannel the selected trial's best epoch was ≤ 3 in 55%
  of ValendinLSTM runs, 41% of LSTM + ratio and 22% of LSTM + log runs — but 2% for both
  bounded LSTM variants, and 0–2% on the other panels. In
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
fixed, moves aggregate bias by 6–14 points (one refit's sd, estimated as the sd of the
paired difference over √2), and the two refits differ by up to 51 points at worst
(`docs/benchmarks-real-panels.md`, 80 ValendinLSTM studies). On gift the bias spread across a study's
trials (19.4 sd) barely clears one checkpoint's spread against itself (14.4). Two trials
closer than the refit noise cannot be told apart by any criterion computed before the
refit, however good. *(Descriptive figures from `docs/benchmarks-real-panels.md`; not
re-tested under the statistical protocol.)*

## 8. What is at stake at the pick

The correlations in §3 describe the whole ranking. What the thesis reports is the argmin.
New here: the holdout score of each criterion's pick, minus a random trial's expected
score (the study's trial mean), paired over studies
(`.scratch/model-selection/pick_vs_random.py`). Negative is better for MAPE and |bias|,
positive for Spearman. Bold: the interval excludes zero.

**Electronics, LSTM**, n = 40 studies. Random trial: MAPE 58.3, |bias| 26.2, Spearman 0.021.

| pick by | Δ MAPE | Δ \|bias\| | Δ Spearman |
| --- | ---: | ---: | ---: |
| val CE (status quo) | +0.37 (−1.21, +1.96) | +2.10 (−1.64, +5.65) | **+0.012 (+0.008, +0.017)** |
| val rollout MAPE | **−1.73 (−2.92, −0.51)** | −1.68 (−4.93, +1.60) | −0.003 (−0.007, +0.002) |
| val rollout \|bias\| | +0.94 (−1.15, +2.96) | +2.01 (−2.94, +6.71) | −0.002 (−0.008, +0.003) |
| val rollout Spearman | −1.44 (−3.33, +0.50) | −3.40 (−7.68, +0.81) | **+0.012 (+0.005, +0.020)** |
| longest-trained (best epoch) | **−4.66 (−6.15, −3.03)** | **−12.74 (−16.46, −8.88)** | +0.006 (−0.001, +0.013) |

**Electronics, ValendinLSTM**, n = 40 studies. Random trial: MAPE 71.4, |bias| 37.4,
Spearman 0.021.

| pick by | Δ MAPE | Δ \|bias\| | Δ Spearman |
| --- | ---: | ---: | ---: |
| val CE (status quo) | **−3.38 (−6.37, −0.68)** | **+4.39 (+0.12, +8.29)** | **+0.012 (+0.003, +0.026)** |
| val rollout MAPE | **−5.41 (−8.19, −2.64)** | −1.14 (−5.82, +3.14) | **+0.008 (+0.001, +0.017)** |
| val rollout \|bias\| | +0.44 (−2.30, +3.33) | +0.59 (−3.40, +4.63) | +0.002 (−0.004, +0.008) |
| val rollout Spearman | −2.95 (−7.30, +1.72) | −2.11 (−7.60, +3.55) | +0.012 (−0.000, +0.027) |
| longest-trained (best epoch) | **−7.41 (−11.21, −3.42)** | **−12.08 (−17.52, −6.39)** | **+0.011 (+0.002, +0.021)** |

**CDNOW, LSTM**, n = 5 studies. Random trial: MAPE 101.1, |bias| 93.4, Spearman 0.276.

| pick by | Δ MAPE | Δ \|bias\| | Δ Spearman |
| --- | ---: | ---: | ---: |
| val CE (status quo) | −39.7 (−74.3, +9.4) | −41.0 (−79.8, +11.4) | **+0.102 (+0.074, +0.129)** |
| val rollout MAPE | +31.8 (−43.5, +97.8) | +34.8 (−45.8, +103.9) | +0.036 (−0.103, +0.126) |
| val rollout \|bias\| | **−64.0 (−80.8, −44.6)** | **−64.4 (−83.4, −45.5)** | **+0.094 (+0.062, +0.132)** |
| val rollout Spearman | −24.2 (−58.8, +13.9) | −19.8 (−52.1, +16.6) | +0.036 (−0.150, +0.159) |
| longest-trained (best epoch) | −16.4 (−74.4, +41.7) | −16.3 (−77.5, +44.8) | −0.021 (−0.120, +0.078) |

**CDNOW, ValendinLSTM**, n = 5 studies. Random trial: MAPE 59.2, |bias| 48.3, Spearman 0.288.

| pick by | Δ MAPE | Δ \|bias\| | Δ Spearman |
| --- | ---: | ---: | ---: |
| val CE (status quo) | −1.4 (−39.1, +43.5) | −3.3 (−46.2, +40.7) | **+0.097 (+0.073, +0.118)** |
| val rollout MAPE | +33.3 (−8.7, +75.3) | +34.1 (−5.8, +73.9) | −0.002 (−0.142, +0.100) |
| val rollout \|bias\| | **−18.3 (−38.3, −2.6)** | −17.7 (−38.6, +0.6) | +0.027 (−0.099, +0.114) |
| val rollout Spearman | −8.6 (−31.1, +11.4) | −12.8 (−28.8, +3.7) | **+0.103 (+0.069, +0.151)** |
| longest-trained (best epoch) | −13.2 (−36.8, +9.6) | −14.1 (−38.0, +6.0) | **+0.108 (+0.066, +0.161)** |

Three readings:

- **On electronics the status quo costs little at the pick, and the two models differ in
  what.** For the LSTM the CE pick shows no clear difference from a random trial on MAPE
  or |bias|. For ValendinLSTM it is 3.4 MAPE points *better* (supported, under the
  3.6-point refit noise) and 4.4 |bias| points worse (supported, under the 5.9-point refit
  noise). On both it ranks customers 0.012 better, supported and about the refit noise.
  The wrong sign in §3 is real across the ranking, but its price at the argmin is small,
  because the trials barely differ (§4).
- **The one criterion worth more than the refit noise on electronics is not a validation
  score.** Picking the longest-trained trial beats a random one by 4.7 MAPE and 12.7
  |bias| (LSTM) and 7.4 MAPE and 12.1 |bias| (ValendinLSTM), all supported and above the
  refit noise. That is §6 again: training length is the lever. Rollout MAPE also lowers
  MAPE on both models (1.7 and 5.4), supported; the LSTM's gain is under the refit noise.
- **No criterion is shown safe on both panels.** On CDNOW, validation rollout MAPE picks a
  trial with holdout MAPE 126–221 in 6 of 10 studies. Those trials fit the validation
  year's level well (validation MAPE 11–15) and then explode on the holdout. In 2 of the 6
  the CE pick explodes too (165.2 against 165.2, 142.3 against 140.8), so the blow-ups
  are not unique to rollout selection; in the other 4 the CE pick scores 25–41. (These
  counts are description.) At the pick, rollout MAPE's +31.8 (LSTM) and +33.3
  (ValendinLSTM) against a random trial show no clear difference at n = 5; its harm on
  CDNOW is supported only as a re-rank of a CE shortlist, for the LSTM (S3). The
  validation score that is best on electronics is therefore the riskiest on CDNOW, on ten
  studies. Rollout |bias| is the CDNOW pick with supported gains — on MAPE for both models
  and on |bias| for the LSTM — and it has none on electronics.

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

- **Measured, electronics** (n = 40 per model, `selection_analysis.py`): rollout MAPE
  beats CE on its own target for both models — Δ rho +0.140 (+0.068, +0.213) LSTM, +0.208
  (+0.138, +0.281) ValendinLSTM. Its own correlation with holdout MAPE differs by model:
  −0.041 (−0.112, +0.032) for the LSTM, where it removes a harmful signal without
  supplying a useful one, and +0.107 (+0.046, +0.168) for ValendinLSTM, where it supplies
  a weak one. Rollout Spearman does not clearly beat CE on holdout Spearman for either
  model (+0.110, −0.003 to +0.224; +0.056, −0.039 to +0.150). At the pick rollout MAPE
  gains 1.7 (LSTM) and 5.4 (ValendinLSTM) MAPE on a random trial (§8).
- **Measured, CDNOW** (n = 5 per model): four supported deficits against CE, two per
  model. LSTM: rollout Spearman against holdout MAPE (Δ −0.130, −0.219 to −0.058) and
  against holdout |bias| (Δ −0.101, −0.166 to −0.044). ValendinLSTM: rollout MAPE against
  holdout Spearman (Δ −0.298, −0.430 to −0.148) and rollout |bias| against holdout
  Spearman (Δ −0.288, −0.450 to −0.108). No rollout criterion beats CE on any CDNOW
  target. Rollout MAPE's pick blows up in 6 of 10 studies (3 per model), 4 of them where
  CE's does not (§8).
- **Cost:** one validation rollout per trial. The rescore's 29 s a trial on electronics
  (10 s on CDNOW) timed a validation rollout, a refit and a holdout rollout together, so
  the rollout alone is a fraction of that. ADR-0003's two defects
  (a second scoring implementation, no wiring into `StudySuiteConfig`) must not return.
- **Verdict:** not a safe replacement. Panel-dependent sign.

### S2. A composite of rollout MAPE, |bias| and Spearman

- **Measured, electronics** (composite minus the matching single criterion, paired over
  n = 40 studies per model, `docs/training-budget.md` §14.3): worse for holdout MAPE on
  both models (LSTM −0.139, −0.194 to −0.083; ValendinLSTM −0.066, −0.115 to −0.018) and
  for holdout Spearman on the LSTM (−0.166, −0.245 to −0.083); no clear difference for
  ValendinLSTM's Spearman (−0.012, −0.078 to +0.054).
- **Verdict:** rejected. Match the criterion to the metric the claim is about.

### S3. Two-stage: shortlist on CE, re-rank the shortlist by rollout

Keep the trials within *m*% of the best CE and pick among them by a rollout metric.
Cheap — one rollout per shortlisted trial. Tested offline here on the rescore data
(`.scratch/model-selection/two_stage.py`); Δ is the pick's holdout score minus the plain
CE pick's in the same study, paired over studies (n = 40 per model on electronics, 5 on
CDNOW); lower is better for MAPE and |bias|, higher for Spearman. Bold: the interval
excludes zero. Every margin the script tests is shown.

| panel | model | m | shortlist (median) | re-rank by | Δ MAPE | Δ \|bias\| | Δ Spearman |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| electronics | LSTM | 0.5% | 7.5 | rollout MAPE | **−2.81 (−4.61, −1.00)** | **−5.52 (−9.62, −1.47)** | **−0.007 (−0.014, −0.000)** |
| electronics | LSTM | 0.5% | 7.5 | rollout Spearman | +0.80 (−0.68, +2.27) | +1.46 (−1.89, +4.85) | +0.000 (−0.004, +0.004) |
| electronics | LSTM | 1% | 20 | rollout MAPE | **−2.61 (−4.35, −0.86)** | **−4.39 (−8.70, −0.05)** | **−0.011 (−0.019, −0.004)** |
| electronics | LSTM | 1% | 20 | rollout Spearman | −0.38 (−2.16, +1.38) | −1.29 (−5.68, +3.16) | −0.001 (−0.006, +0.004) |
| electronics | LSTM | 2% | 29 | rollout MAPE | **−2.97 (−4.86, −1.05)** | **−5.14 (−9.73, −0.38)** | **−0.012 (−0.019, −0.005)** |
| electronics | LSTM | 2% | 29 | rollout Spearman | −0.80 (−2.94, +1.29) | −2.31 (−7.27, +2.59) | −0.001 (−0.007, +0.005) |
| electronics | LSTM | 5% | 37 | rollout MAPE | **−2.30 (−4.49, −0.07)** | −4.07 (−9.15, +1.23) | **−0.014 (−0.021, −0.007)** |
| electronics | LSTM | 5% | 37 | rollout Spearman | −1.91 (−4.48, +0.53) | **−5.67 (−10.94, −0.56)** | −0.000 (−0.007, +0.007) |
| electronics | ValendinLSTM | 0.5% | 5 | rollout MAPE | +0.53 (−2.40, +3.60) | +0.29 (−3.43, +4.24) | **−0.010 (−0.023, −0.000)** |
| electronics | ValendinLSTM | 0.5% | 5 | rollout Spearman | **+3.80 (+0.63, +7.28)** | +3.84 (−1.01, +9.08) | +0.001 (−0.007, +0.008) |
| electronics | ValendinLSTM | 1% | 12 | rollout MAPE | −1.48 (−4.52, +1.56) | −3.08 (−7.12, +0.87) | −0.005 (−0.015, +0.004) |
| electronics | ValendinLSTM | 1% | 12 | rollout Spearman | +3.03 (−1.25, +7.52) | −0.16 (−6.93, +6.77) | −0.000 (−0.007, +0.008) |
| electronics | ValendinLSTM | 2% | 19.5 | rollout MAPE | −0.69 (−3.66, +2.30) | −3.45 (−7.70, +0.77) | −0.008 (−0.019, +0.002) |
| electronics | ValendinLSTM | 2% | 19.5 | rollout Spearman | +3.12 (−1.63, +8.17) | −1.74 (−8.67, +5.43) | −0.004 (−0.012, +0.005) |
| electronics | ValendinLSTM | 5% | 30 | rollout MAPE | −1.82 (−4.89, +1.36) | −5.10 (−10.23, +0.13) | −0.000 (−0.011, +0.010) |
| electronics | ValendinLSTM | 5% | 30 | rollout Spearman | +0.14 (−5.01, +5.70) | −6.69 (−14.23, +1.29) | −0.000 (−0.010, +0.010) |
| cdnow | LSTM | 0.5% | 5 | rollout MAPE | +28.5 (−1.1, +83.5) | **+33.1 (+1.7, +85.5)** | −0.021 (−0.066, +0.020) |
| cdnow | LSTM | 0.5% | 5 | rollout Spearman | +15.6 (−6.4, +46.8) | +17.9 (−3.8, +43.2) | −0.073 (−0.262, +0.070) |
| cdnow | LSTM | 1% | 15 | rollout MAPE | **+60.7 (+7.5, +113.9)** | **+63.7 (+12.0, +115.4)** | +0.007 (−0.009, +0.029) |
| cdnow | LSTM | 1% | 15 | rollout Spearman | +14.5 (−7.5, +45.9) | +11.6 (−13.4, +41.3) | −0.062 (−0.253, +0.077) |
| cdnow | LSTM | 2% | 20 | rollout MAPE | **+91.5 (+25.7, +157.3)** | **+95.0 (+24.6, +165.3)** | +0.005 (−0.010, +0.028) |
| cdnow | LSTM | 2% | 20 | rollout Spearman | +15.6 (−6.9, +45.9) | +21.1 (−5.1, +45.1) | −0.066 (−0.257, +0.075) |
| cdnow | LSTM | 5% | 23 | rollout MAPE | **+91.5 (+25.7, +157.3)** | **+95.0 (+24.6, +165.3)** | +0.005 (−0.010, +0.028) |
| cdnow | LSTM | 5% | 23 | rollout Spearman | +15.6 (−6.9, +45.9) | +21.1 (−5.1, +45.1) | −0.066 (−0.257, +0.075) |
| cdnow | ValendinLSTM | 0.5% | 4 | rollout MAPE | −2.8 (−16.1, +7.2) | **−6.3 (−14.8, −0.6)** | −0.005 (−0.014, +0.002) |
| cdnow | ValendinLSTM | 0.5% | 4 | rollout Spearman | −11.5 (−34.5, +4.7) | **−14.6 (−35.0, −1.8)** | +0.003 (−0.043, +0.056) |
| cdnow | ValendinLSTM | 1% | 7 | rollout MAPE | +12.2 (−15.4, +52.6) | +15.8 (−15.0, +56.9) | **−0.025 (−0.054, −0.001)** |
| cdnow | ValendinLSTM | 1% | 7 | rollout Spearman | −8.5 (−33.3, +10.2) | −8.8 (−35.3, +16.1) | +0.003 (−0.047, +0.057) |
| cdnow | ValendinLSTM | 2% | 11 | rollout MAPE | +12.2 (−15.4, +52.6) | +15.8 (−15.0, +56.9) | **−0.025 (−0.054, −0.001)** |
| cdnow | ValendinLSTM | 2% | 11 | rollout Spearman | −7.2 (−32.1, +10.2) | −9.4 (−35.9, +16.1) | +0.006 (−0.042, +0.059) |
| cdnow | ValendinLSTM | 5% | 15 | rollout MAPE | +12.2 (−15.4, +52.6) | +15.8 (−15.0, +56.9) | **−0.025 (−0.054, −0.001)** |
| cdnow | ValendinLSTM | 5% | 15 | rollout Spearman | −7.2 (−32.1, +10.2) | −9.4 (−35.9, +16.1) | +0.006 (−0.042, +0.059) |

Each row is its own claim, read by its own interval; the pattern across margins and
models is described in words below.

- **Verdict:**
  - *Electronics, LSTM, re-rank by rollout MAPE:* MAPE improves by 2.3–3.0 at every
    margin (supported; under the 3.6-point refit noise), |bias| by 4.4–5.5 at 0.5–2%
    (supported; under the 5.9-point refit noise) and not clearly at 5%. Spearman falls by
    0.007–0.014 at every margin, supported, about the 0.0105 refit noise.
  - *Electronics, ValendinLSTM, re-rank by rollout MAPE:* no clear MAPE or |bias|
    difference at any margin; Spearman falls by 0.010 at 0.5% only (supported).
  - *Electronics, re-rank by rollout Spearman:* no consistent pattern — ValendinLSTM's MAPE
    worsens by 3.8 at 0.5%, the LSTM's |bias| improves by 5.7 (about the refit noise) at
    5% only.
  - *CDNOW, LSTM, re-rank by rollout MAPE:* a supported loss of 61–92 MAPE and 64–95
    |bias| from 1% up (and 33 |bias| at 0.5%). Shortlisting does not contain the CDNOW
    blow-ups; it only delays them to a wider margin.
  - *CDNOW, ValendinLSTM:* rollout MAPE's MAPE loss (+12.2 from 1% up) shows no clear
    difference at n = 5, while Spearman falls by 0.025 (supported); at 0.5% both re-ranks
    lower |bias| (−6.3, −14.6, supported).
  - Rejected in this form: its one consistent gain is the electronics LSTM's, and its one
    consistent loss is the CDNOW LSTM's, several times larger.

### S4. Use training length as a criterion, or as a tie-breaker

Targets §6. The best epoch is already recorded.

- **Measured, electronics:** the longest-trained pick beats a random trial by 4.7 MAPE and
  12.7 |bias| (LSTM) and by 7.4 MAPE and 12.1 |bias| (ValendinLSTM) (§8), all supported
  and above the refit noise — the largest supported gains of any criterion here.
- **Measured, CDNOW** (n = 5 per model): at the pick, no clear MAPE or |bias| effect on
  either model; for ValendinLSTM it ranks customers 0.108 better than a random trial
  (supported). As a ranking of trials it is worse than CE for holdout Spearman on the LSTM,
  Δ −0.193 (−0.297, −0.084), and shows no clear difference on ValendinLSTM (−0.113,
  −0.340 to +0.080).
- **Verdict:** a symptom, not a remedy. It works on electronics because the trials that
  trained longest are the ones the stopping rule let through, which is what S5 fixes at
  the source. Useful as evidence for S5, not as a selection rule.

### S5. Fix the stopping rule so the search stops preferring undertrained models

Targets §2 and §6: a relative improvement threshold, patience counted in optimiser steps,
a smoothed curve, or a fixed step budget with no early stopping
(`docs/training-budget.md` §13.3, B5–B8; experiment E4).

- **Measured indirectly:** a 90-epoch floor (family T, `paper90`) moved electronics
  ValendinLSTM MAPE by −22.3 (−28.2, −16.2), an order of magnitude more than any selection
  rule — but LSTM by only −4.5 (−12.9, +4.5), not supported. Family U's floored arm
  (paper recipe plus a 90-epoch floor) triples CDNOW's LSTM error, 57.7 → 183.9. E1 has
  since finished (20 independent searches per arm, so the arms are compared as independent
  samples, `.scratch/model-selection/e1_cdnow.py`): a 50-epoch floor alone moves CDNOW
  LSTM MAPE by +3.5 (−0.4, +8.2), no clear difference at n = 20, and the paper recipe
  alone by **+3.6 (+0.2, +7.0)**, supported and small — under the 5.83-point refit noise.
  For ValendinLSTM neither shows a clear difference (floor −3.7, −20.8 to +9.7; paper
  +2.5, −15.0 to +16.4). Neither comes near a tripling, but E1 does not settle it: its
  LSTM baseline MAPE is 21.7 against family U's 57.7, so it is not the same setup, and it
  ran no 90-epoch arm. The tripling is still unexplained. The script reads each study's
  `metrics.csv`; `.scratch/training-budget/results/e1_cdnow.csv` holds the same MAPE
  values. E1 is cited in no other doc.
- **Untested:** every principled alternative to the floor.
- **Verdict:** the highest-value change on the evidence. Run E4, one panel first, then
  CDNOW before adopting anything.

### S6. Average replications instead of trusting one pick

Score the ensemble of a cell's replication forecasts rather than the mean of their
scores. Targets §2 and §7: it averages away both near-tie luck and refit noise instead of
trying to beat them.

- **Measured, CDNOW (family G, 20 studies):** cuts MAPE by 2.8–12.3 points, and puts
  ValendinLSTM at 18.13 against Pareto/NBD's 18.70. Bias is unchanged by construction
  (`docs/insights-study.md` §9). Old recipe: 50 trials, 50 paths, the pre-ADR-0009
  38-week holdout; it reproduces only with the dataset code before `fe37ee5`. Not yet
  measured on the current recipe or on any other panel.
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
and no rule can select between them. In family U on electronics (ValendinLSTM, 20
replications each), a persistent per-customer input (the `kmeans_8` cluster label) lifts
Spearman from 0.021 to 0.305, against 0.178 for the floored arm without it; Pareto/NBD
scores 0.314 over 20 seeded fits (`docs/training-budget.md` §15.1). `docs/absorbing-death-state.md` argues
for a learned alternative to the borrowed label.

- **Verdict:** a precondition, not a selection rule. On a collapsed panel it is the only
  change here that gives selection something to choose from.

---

# Part III — Recommendation

1. **Do not replace validation CE with any single rollout criterion yet.** The best one on
   electronics (rollout MAPE, directly or as a two-stage re-rank) is harmful on CDNOW, with
   support for the LSTM as a re-rank at 1% and wider (S3). Selection is panel-dependent, and the thesis
   cannot pick a panel.
2. **Put the effort into training length first (S5).** It is worth an order of magnitude
   more than selection on electronics, and training length is what the one strong
   "criterion" in §8 was really measuring.
3. **Report the ensemble beside the distribution (S6).** It costs nothing and neutralises
   the near-tie and refit noise that §2 and §7 measure.
4. **If selection is revisited, test rolling-origin validation (S8)** on the existing
   rescore data first, on both panels, before building anything.
5. **Close E2** before any cluster-arm result is used to argue about selection.
6. **In the thesis text, state plainly:** the reported winner is the argmin of a
   criterion that ranks trials correctly for customer ranking on CDNOW, on both models
   (and, without intervals, on gift), is wrong-signed for level and |bias| on
   electronics, on both models (and, without intervals, on multichannel), and is shown to
   select for bias only for CDNOW's LSTM, on five studies. At the pick on electronics
   that costs ValendinLSTM about 4 points of |bias| while gaining about 3 of MAPE, and
   makes no clear difference to the LSTM.

## What this does not establish

- **Anything about panels as a class.** The selection rescore covers two panels, one of
  them with 5 studies per model, whose percentile intervals run narrow. Gift and multichannel
  appear only in the 698-trial ValendinLSTM rescore, which has no intervals.
- **How the new pick-level results would change on a floored or cluster-labelled model.**
  Every trial here trained under patience 7 and no floor; S5 would change the trial
  population, and with it every number in §8.
- **That the CDNOW blow-ups are a property of rollout selection rather than of these ten
  studies.** Six of ten, two shared with the CE pick, is not a rate.
- **The no-early-stopping curve in §2 beyond one run.** Only one run's output is on
  disk; the other slopes cited in `docs/training-budget.md` cannot be recomputed.
- **The S6 ensemble gain on current code.** It was not recomputed here; it needs the
  dataset code before `fe37ee5`.
- **That the oracle columns are attainable.** They are the minimum over many noisy
  refits, and part of their advantage is luck.
