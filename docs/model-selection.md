# Model selection

How a neural model is chosen in this package, what was measured to check whether that
choice is a good one, and what could replace it. Training length, which interacts with
selection, is in `docs/insight-training-efficiency.md`.

Conventions in this doc:
- Every interval is a 95% percentile bootstrap from `evaluation.effects.effect`
  (`docs/statistical-protocol.md`).
- The **study** (one complete Optuna search) is the replication, not the trial.
- Criteria and targets are oriented so lower is better, so a **positive** rank
  correlation means the criterion orders trials the way the holdout does.
- Bold marks an interval that excludes 0.
- Family letters (E, F, G, …) name the experiment families of `docs/studies-run.md` §4,
  which holds each one's budget, dates and design. Each is described in a clause where
  this doc first uses it.

## 1. How a model is chosen

1. Optuna trains up to 100 trials (`tuning/optuna_tuning.py`).
2. The trial with the lowest **teacher-forced validation cross-entropy (CE)** on the
   temporal validation window wins (ADR-0001). Since 4 October 2026 that CE is the exact
   per-cell mean whatever the trial's batch size (ADR-0010); every archived search
   before it compared trials on a CE that moved with the batch size (§3.9).
3. The winner is refit for 5 epochs over the full calibration window, unseeded
   (`DEFAULT_REFIT_EPOCHS`, `trials/refit.py`; ADR-0008).
4. The refit is rolled out over the 52-week holdout and scored.

ADR-0003 once selected on a validation rollout instead. It was retired on 12 August 2026
for implementation defects: it was not wired into `StudySuiteConfig`, and its RMSE
disagreed with `compute_forecast_metrics` by 62×. The idea itself was not refuted, and
nothing has replaced it.

## 2. Why the choice might be wrong

**Three mismatches** between what is minimised and what is reported:

| | What selection scores | What is reported |
| --- | --- | --- |
| task | one-step-ahead, true previous counts fed in | 52-week rollout on the model's own samples |
| model | the trial's checkpoint | a 5-epoch unseeded refit of it |
| metric | CE over all cells | aggregate MAPE, bias, per-customer Spearman |

**The objective is flat where the choice is made.**
- The winning CE varies by 0.08–3.1% across the replications of a cell (32 archived
  cells, 2,880 winners).
- The median study has 0–21 other trials within 0.5% of its winner, while holdout bias
  across those winners spans tens of points.
- On electronics, 1.4% of validation cells carry a purchase and they carry 84–87% of
  the CE.

**The refit sets the resolution.** This is the single home of this table. It gives the
mean |difference| between two unseeded refits of the same winner, 20 ValendinLSTM winners
per panel (`.scratch/training-budget/refit_noise.py`):

| panel | MAPE | \|bias\| | Spearman | RMSE |
| --- | ---: | ---: | ---: | ---: |
| cdnow | 5.83 | 12.58 | 0.0159 | 0.00036 |
| electronics | 3.63 | 5.94 | 0.0105 | 0.00014 |
| gift | 5.89 | 14.99 | 0.0116 | 0.00009 |
| multichannel | 7.71 | 12.56 | 0.0152 | 0.00003 |

Refitting one checkpoint twice moves aggregate bias by up to 51 points. Two trials closer
than this cannot be told apart by any criterion computed before the refit. The table is
a magnitude reference, never a threshold.

## 3. What was tested

| § | Test | Data | Question |
| --- | --- | --- | --- |
| 3.1 | Winners across replications | 2,880 archived winners, 32 cells | Does the winning CE track the forecast? |
| 3.2 | Family R rescore | 698 refits of non-winning ValendinLSTM trials, 4 panels | Over a whole study, does CE order trials correctly? |
| 3.3 | Family V rescore | 3,042 trials (LSTM and ValendinLSTM; electronics 40 studies per model, CDNOW 5), each scored by a validation rollout and by the production holdout path | Is CE right-signed? Does a validation rollout do better? |
| 3.4 | Trial spread | Family V | Why does CE work on one panel and not another? |
| 3.5 | Pick vs a random trial | Family V | What does the argmin cost? |
| 3.6 | Two-stage re-rank | Family V | Shortlist on CE, re-rank by rollout? |
| 3.7 | CE across arms | Cluster ablation (family F) | Can calibration choose K? |
| 3.8 | Ensemble scoring | Family G (the CDNOW/electronics arm grid at 50 paths), CDNOW | Does averaging replications neutralise a bad pick? |
| 3.9 | Batch-dependent score, and its fix | Family U's stored winners, 4 panels; family U′ (electronics · `archive` · no cluster label rerun with the fix, 40 studies) | Did the score itself favour one batch size, and did that change what was selected? |

### 3.1 The winning CE does not track the forecast across replications

Within a cell, the rank correlation between a winner's CE and its holdout |bias| lies
within ±0.25 in 31 of 32 cells, and at most 0.49 against holdout Spearman. These are
descriptive correlations with no intervals. They concern near-ties, because every run
compared is the argmin of its own study.

### 3.2 Over a whole study: right on two panels, wrong on two (family R)

`keep_only_best_checkpoint` failed to run on 30 of the 80 family N studies (the ValendinLSTM benchmark runs,
`docs/benchmarks.md`), so their
losing checkpoints survived. `scripts/run_rescore_trials.py` refit and rolled out each
one at the study's own forecast seed (17 Sep). Studies with ≥ 5 scored trials (653 trials,
36 studies) give one rank correlation each. "Winner's place" is where Optuna's pick lands
in its study (0 = best forecast, 1 = worst). **Descriptive, no intervals.**

| panel | studies | ρ Spearman | ρ \|bias\| | winner's place, Spearman | winner's place, \|bias\| |
| --- | ---: | ---: | ---: | ---: | ---: |
| gift | 14 | +0.56 | +0.09 | 0.04 | 0.45 |
| cdnow | 13 | +0.67 | +0.38 | 0.33 | 0.41 |
| electronics | 2 | +0.44 | −0.25 | 0.02 | 1.00 |
| multichannel | 7 | +0.01 | −0.21 | 0.64 | 0.53 |

Every study contains a trial with small bias that CE does not find (selected / best / median
trial, |bias| %):

| panel | selected | best | median |
| --- | ---: | ---: | ---: |
| gift | 16.9 | 4.3 | 19.2 |
| cdnow | 34.6 | 7.7 | 40.4 |
| electronics | 64.7 | 12.2 | 37.8 |
| multichannel | 48.9 | 1.6 | 42.7 |

The "best" column is optimistic, being the minimum of noisy refits. On gift the median
trial ranks customers at 0.019 and the pick at 0.374, so there the search earns its cost.

### 3.3 CE is wrong-signed for level on electronics (family V)

`scripts/run_selection_rescore.py` (21 Sep) trained studies with every checkpoint kept,
then scored up to 40 random trials per study twice:
- by a leak-free Monte Carlo rollout over the **validation** window (the candidate
  criteria);
- by the production refit plus holdout rollout (the target).

Mean per-study ρ of CE with each target:

| target | electronics LSTM (n = 40) | electronics ValendinLSTM (n = 40) | cdnow LSTM (n = 5) | cdnow ValendinLSTM (n = 5) |
| --- | ---: | ---: | ---: | ---: |
| holdout MAPE | **−0.181** (−0.247, −0.112) | **−0.101** (−0.162, −0.038) | **+0.338** (+0.112, +0.553) | +0.149 (−0.050, +0.392) |
| holdout \|bias\| | **−0.266** (−0.323, −0.207) | **−0.261** (−0.310, −0.213) | **+0.356** (+0.128, +0.594) | −0.004 (−0.257, +0.255) |
| holdout Spearman | **+0.092** (+0.013, +0.168) | −0.001 (−0.060, +0.058) | **+0.367** (+0.190, +0.549) | **+0.577** (+0.479, +0.718) |

On electronics, the trials CE likes best over-forecast the holdout most (bias +35.1 in
the best-CE quartile against +22.0 in the worst). On CDNOW, CE is right-signed for
ranking on both models, and for level on the LSTM only. CDNOW rests on 5 studies, so its
magnitudes are loose.

**Validation-rollout criteria on electronics.** Mean ρ with the target, and Δ against CE
paired over the 40 studies per model:

| target | criterion | LSTM ρ | LSTM Δ vs CE | ValendinLSTM ρ | ValendinLSTM Δ vs CE |
| --- | --- | ---: | ---: | ---: | ---: |
| holdout MAPE | rollout MAPE | −0.041 (−0.112, +0.032) | **+0.140** (+0.068, +0.213) | **+0.107** (+0.046, +0.168) | **+0.208** (+0.138, +0.281) |
| holdout MAPE | composite of three | **−0.180** | +0.001 | +0.041 | **+0.142** |
| holdout Spearman | rollout Spearman | **+0.202** (+0.124, +0.279) | +0.110 (−0.003, +0.224) | +0.055 | +0.056 |

- **Rollout MAPE beats CE for level on both models.** For the LSTM it only removes a
  harmful signal (its own ρ is not distinguishable from 0). For ValendinLSTM it adds a
  weak useful one.
- **The best number anywhere is +0.202**, about 4% of the rank variance.
- **The composite** (rank-normalised rollout MAPE, |bias| and 1 − Spearman) is worse than
  the single matching criterion: holdout MAPE LSTM −0.139, ValendinLSTM −0.066; holdout
  Spearman LSTM −0.166. Match the criterion to the metric being reported.
- **On CDNOW no rollout criterion beats CE on any target.** Four deficits are
  supported, e.g. ValendinLSTM rollout MAPE against holdout Spearman, Δ −0.298
  (−0.430, −0.148).

### 3.4 CE works where trials differ (CDNOW) and has nothing to work with where they do not (electronics)

The first explanation for the wrong sign on electronics was a fall in purchase rate
between calibration and holdout. It is **retracted**: every panel's holdout rate is below
its calibration rate (ratios 0.23–0.63), and CDNOW, where CE works, falls more steeply
(0.40 against 0.63). What separates the two panels is how much a study's trials differ:

| CE quartile within a study | electronics bias / MAPE / Spearman | cdnow bias / MAPE / Spearman |
| --- | ---: | ---: |
| best | +35.1 / 65.3 / 0.02 | +14.3 / 65.1 / 0.34 |
| 2nd | +33.4 / 66.5 / 0.02 | +10.2 / 55.4 / 0.34 |
| 3rd | +29.4 / 64.5 / 0.02 | +50.4 / 97.9 / 0.29 |
| worst | +22.0 / 62.7 / 0.02 | +98.7 / 114.2 / 0.16 |

The within-study IQR of holdout Spearman is 0.202 on CDNOW and **0.021** on electronics,
against refit noise of 0.0159 and 0.0105. A CDNOW study contains broken trials, and CE
finds them. An electronics study is collapsed: every trial gives every customer nearly
the same forecast, so only a small, wrong-signed tilt remains. This describes two
panels; it is not a tested mechanism.

### 3.5 What the argmin costs

The holdout score of each criterion's pick, minus the study's trial mean (a random trial),
paired over studies (`.scratch/model-selection/pick_vs_random.py`). Negative is better for
MAPE and |bias|.

| pick by | elec LSTM Δ MAPE / \|bias\| | elec ValendinLSTM Δ MAPE / \|bias\| | cdnow LSTM Δ MAPE / \|bias\| | cdnow ValendinLSTM Δ MAPE / \|bias\| |
| --- | ---: | ---: | ---: | ---: |
| val CE (status quo) | +0.37 / +2.10 | **−3.38** / **+4.39** | −39.7 / −41.0 | −1.4 / −3.3 |
| val rollout MAPE | **−1.73** / −1.68 | **−5.41** / −1.14 | +31.8 / +34.8 | +33.3 / +34.1 |
| val rollout \|bias\| | +0.94 / +2.01 | +0.44 / +0.59 | **−64.0** / **−64.4** | **−18.3** / −17.7 |
| val rollout Spearman | −1.44 / −3.40 | −2.95 / −2.11 | −24.2 / −19.8 | −8.6 / −12.8 |
| longest-trained (best epoch) | **−4.66** / **−12.74** | **−7.41** / **−12.08** | −16.4 / −16.3 | −13.2 / −14.1 |

On Spearman, the CE pick beats a random trial by +0.012 on electronics (both models) and
by about +0.10 on CDNOW (both models), all supported.

- **On electronics the status quo costs little at the pick**, because the trials barely
  differ (§3.4).
- **The only criterion worth more than the refit noise on electronics is not a
  validation score.** It is the longest-trained trial, i.e. training length
  (`docs/insight-training-efficiency.md`).
- **No criterion is safe on both panels.** On CDNOW, validation rollout MAPE picks a trial
  with holdout MAPE 126–221 in 6 of 10 studies. Those trials fit the validation year well
  (validation MAPE 11–15) and explode on the holdout. In 2 of those 6 the CE pick
  explodes too.

### 3.6 Two-stage: shortlist on CE, re-rank by rollout

Keep the trials within m% of the best CE and pick among them by a rollout criterion. Δ is
against the plain CE pick, paired (`.scratch/model-selection/two_stage.py`; margins
0.5–5%):

| panel / model | re-rank by rollout MAPE: Δ MAPE across margins |
| --- | --- |
| electronics LSTM | **−2.3 to −3.0** at every margin; Spearman −0.007 to −0.014 |
| electronics ValendinLSTM | no clear difference at any margin |
| cdnow LSTM | **+61 to +92** from 1% up |
| cdnow ValendinLSTM | +12.2, no clear difference; Spearman **−0.025** |

Shortlisting does not contain the CDNOW blow-ups; it only moves them to a wider margin.
Rejected in this form.

### 3.7 CE across arms points at discrimination, not level

On the family F cluster arms, ranking the six arms by their mean best CE puts `no_cluster`
last on both panels. Its agreement with the holdout ranking by median |bias| is 0.66
(electronics) and 0.09 (CDNOW), against 0.77 for the holdout ranking by Spearman
(electronics). CE scores a per-customer conditional density, so it rewards what separates
customers, while the level is a property of the 52-step rollout. A calibration-only rule
for K would have to share the rollout's structure.

### 3.8 Score the ensemble beside the distribution

`mape_aggregate` is convex in the prediction, so the error of the averaged forecast is at
most the average error (Jensen); bias is linear and does not move. Family G (family H's
arm grid at 50 paths), CDNOW, 20
studies × 50 trials × 50 paths, pre-ADR-0009 38-week window:

| CDNOW MAPE | mean of 20 runs | ensemble of 20 |
| --- | ---: | ---: |
| ValendinLSTM `no_ar-no_cluster` | 22.54 | 18.13 |
| LSTM `ar_bounded-no_cluster` | 22.34 | 18.29 |
| LSTM `no_ar-no_cluster` | 24.39 | 21.64 |
| Transformer `no_ar-no_cluster` | 49.62 | 37.31 |
| Pareto/NBD (one fit) | 18.70 | 18.70 |

The ensemble is what one would deploy, and the only figure comparable with a single
deterministic fit. The distribution is what shows a single fit is unreliable. Report
both. This has not been recomputed on the current recipe or on another panel; it needs
the dataset code from before `fe37ee5`.

### 3.9 The validation score depended on the batch size, and steered the search

**The score read differently at each batch size for the same weights.** Up to commit
af2b14b, the validation CE was the mean of per-batch means. The validation loader runs at
the trial's batch size in fixed customer order, so the short last batch counted as much as
a full one. At batch 256 on electronics, the 61 highest-id customers carried 25% of the
score instead of 7%. Scoring family U's stored winners at every batch size, weights fixed,
against the true per-cell mean (3 winners per model per panel; `ValendinLSTM` and `LSTM`
agree to 0.4 points):

| panel | the short batch's loss vs the average | batch 32 | batch 64 | batch 128 | batch 256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| electronics | 0.79× | −0.2% | −0.1% | −1.6% | **−4.1%** |
| multichannel | 0.29× | −0.4% | −0.3% | −0.3% | **−6.2%** |
| cdnow | 1.35× | +0.4% | +0.2% | +1.1% | **+2.8%** |
| gift | 2.28× | +1.1% | +3.0% | +6.7% | **+13.4%** |

- **The bias is larger than what the search decides on.** The winning CE varies by
  0.08–3.1% across replications (§2), so a 4–6% discount, or a 3–13% penalty, outweighs
  any real difference between trials of different batch sizes.
- **It matches what the archived searches did.** Batch 256 was drawn in 61% and 51% of all
  electronics and multichannel trials (TPE samples more of what scores well), against 8–9%
  on CDNOW and gift. Without the cluster label, 17–20 of 20 winners per cell on electronics
  and multichannel ran batch 256, then stopped after 3–8 epochs and collapsed
  (`docs/insight-training-efficiency.md` §8, failure type A).
- **The direction is an accident of customer order.** On electronics and multichannel the
  highest ids buy less than average, so the short batch lowers the score; on CDNOW and gift
  they buy more, so it raises it.

**The fix** (ADR-0010) weights each batch by the cells it scored, so the CE is the exact
per-cell mean. Shuffling validation was rejected: at batch 256 it leaves a 95% range of
±3–9% around the true mean on every panel.

**The rerun.** Family U′ reruns electronics · `archive` · no cluster label with the fix as
the only change: the archive's search space restored (weight decay searched, batch over
{64, 128, 256}), the same seeds, 100 trials, patience 7, pruner, refit and 200 paths;
20 replications per model (`.scratch/score-fix/`). Δ is rerun minus archive, resampled
independently: training is unseeded and the two searches diverge after their first trial,
so a shared seed does not make replications pairs.

| | LSTM archive | LSTM rerun | ValendinLSTM archive | ValendinLSTM rerun |
| --- | --- | --- | --- | --- |
| trials at batch 256 | 68% | 16% | 70% | 18% |
| winners' batch | 256: 18, 128: 2 | 64: 12, 128: 6, 256: 2 | 256: 19, 128: 1 | 64: 11, 128: 8, 256: 1 |
| best epoch, mean | 11.1 | 7.8 | 7.8 | 16.7 |

| Δ rerun − archive | LSTM | ValendinLSTM |
| --- | --- | --- |
| MAPE | −3.0 [−7.5, +1.5] | **−9.4 [−16.0, −3.0]** |
| bias % | −8.0 [−20.1, +4.0] | **−11.4 [−21.8, −1.6]** |
| Spearman | +0.017 [−0.009, +0.044] | **+0.055 [+0.013, +0.099]** |
| forecast CV | **+0.008 [+0.004, +0.013]** | **+0.071 [+0.022, +0.129]** |
| best epoch | −3.4 [−7.7, +0.6] | **+8.9 [+1.6, +16.9]** |

- **The fix removed the preference for batch 256.** That establishes the score as the cause
  of the archived searches' batch choice on this cell.
- **ValendinLSTM forecasts better.** Its batch-64 winners kept epoch 27 at the median, and
  5 of those 11 runs no longer collapse (CV 0.2–0.5, Spearman about 0.2); the best reaches
  MAPE 36 at a bias near 0. The other 15 runs still collapse, so the cell's median CV
  barely moves (0.081 to 0.083): the mean gain comes from the runs that escaped.
- **The LSTM does not.** At batch 64 its winners keep epoch 3.5 at the median: about 58
  updates, the same few dozen as at batch 256. It still collapses (CV 0.09, Spearman
  0.05). Batch 256 was one route to stopping too early, not the only one: patience 7 on a
  flat validation curve stops it at any batch size (`docs/insight-training-efficiency.md`
  §3).

**What it means for the archive.** Every searched result before af2b14b was selected under
this score: every family in `docs/studies-run.md` that ran more than one trial per study,
including the `archive` and `floor50` arms of families T, T′ and U, the benchmark searches
of family N and the LSTM arms of families O and P. On electronics and
multichannel those searches leaned to batch 256; on CDNOW and gift they leaned away from
it. Only this one cell has been rerun, so how much each archived number moved is known
here alone. Arms that run one pinned trial select nothing between trials.

## 4. Does this arise with Valendin et al.'s code?

Read from their notebook (`Original_paper_model/banking_transactions_demo.ipynb`):

| | Valendin et al. | This package |
| --- | --- | --- |
| hyperparameter search | none: one fixed model (LSTM 128, dense 128, Adam defaults, batch 32) | Optuna, up to 100 trials |
| what is selected | the **epoch**: Keras `EarlyStopping(patience=5, min_delta=0, restore_best_weights=True)` on `val_loss` | the trial (argmin CE) and, inside it, the epoch (patience 7) |
| validation set | a random 10% of customers, all periods | a temporal window, all customers (ADR-0001) |
| criterion | teacher-forced CE | teacher-forced CE |
| refit | none; the prediction model copies the trained weights | 5-epoch unseeded refit on full calibration |

- **The task mismatch is in their code too.** The epoch is chosen on one-step
  teacher-forced CE, not on the rollout.
- **The trial-selection problems of §3.1–3.6 are ours.** With no search there is no argmin
  over near-tied trials. Our ValendinLSTM benchmark puts a 100-trial search over learning
  rate, weight decay and batch size around their frozen architecture. That was this
  package's choice.
- **The flat curve is ours.** Under their customer-wise split the same recipe trains
  28–56 epochs on electronics; under our temporal split it stops at once
  (`docs/insight-training-efficiency.md`).
- The paper PDF is not in the repo, so these rows come from the notebook alone, not from
  the paper's text.

## 5. Remedies and their status

| | Remedy | Targets | Status |
| --- | --- | --- | --- |
| S1 | Select on a validation-window rollout, scored on the reported metric | task mismatch | Tested (§3.3, §3.5): helps on electronics, blows up on CDNOW. Not a safe replacement. |
| S2 | Composite of rollout MAPE, \|bias\|, Spearman | all three metrics | Tested (§3.3): worse than its parts. **Rejected.** |
| S3 | Two-stage CE shortlist plus rollout re-rank | cost of S1 | Tested (§3.6). **Rejected in this form.** |
| S4 | Training length as a criterion | undertraining | Tested (§3.5): best on electronics, nothing on CDNOW. A symptom, not a remedy. |
| S5 | Fix the stopping rule | undertraining | `docs/insight-training-efficiency.md`. The highest-value change. |
| S6 | Score the ensemble of replications | near-ties, refit noise | Measured once (§3.8). **Adopt for reporting**, beside the distribution. |
| S7 | Average or seed the refit | refit noise | Untested. Redundant if S6 is used. |
| S8 | Rolling-origin validation (2–3 cut points) | one flat window; CDNOW blow-ups | Untested. The one untried selection change aimed at S1's failure. |
| S9 | Make trials worth distinguishing (a per-customer input) | collapsed panels | A precondition, not a rule (`docs/insights-real-panels.md`). |
| S10 | Score validation as a per-cell mean, independent of batch size | a score that moved with a searched hyperparameter | **Adopted** (ADR-0010, §3.9). Archived searches predate it. |
| E2 | Recompute `kmeans_8` before the validation window | possible label leak into selection | Owed. Until it runs, selection results on cluster arms carry this caveat (`docs/feature-engineering.md` §5). |

**Recommendation.**
1. Keep CE for now. No single rollout criterion is safe on both panels.
2. Put the effort into training length (S5).
3. Report the ensemble beside the distribution (S6).
4. If selection is revisited, test S8 offline on the family V data first.
5. In the thesis, state plainly that the reported winner is the argmin of a criterion
   that ranks trials correctly for customer ranking on CDNOW, and is wrong-signed for
   level on electronics.

## 6. Limits

- **Two panels in the tested rescore**, one of them with 5 studies per model. Gift and
  multichannel appear only in family R, which has no intervals.
- **Every rescored trial trained under patience 7 with no floor.** A training floor
  changes the trial population, and with it every number in §3.3–3.6.
- **Six blow-ups in ten CDNOW studies is not a rate.**
- **The "best trial" columns are optimistic**: minima over noisy refits.
