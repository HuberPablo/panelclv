# 03 — Ablation arms: do the compressed encodings beat the flags?

Status: ready-for-agent

Blocked by: 01

## Problem

Support is an input-side precondition, not a result. §4's own counter-example stands:
CDNOW's `ar_bounded_32` arm was nominally bounded and still lost, because only 3.5% of
calibration cells sat beyond its deepest bin against 68.9% of holdout cells. A short
distance outside the fitted range is not evidence that a trained model forecasts well
from it.

## The arms

Added to `arms_for` in `scripts/run_ar_encoding_ablation.py`:

    ar_log        log_period_since_last_transaction, cumulative_transactions,
                  log_period_since_first_transaction
    ar_saturating saturating_recency_<C>, transaction_rate, saturating_tenure_<C>
    ar_ratio      recency_over_tenure, transaction_rate, saturating_tenure_<C>,
                  has_transacted_before          # the bounded Pareto/NBD triple

`C` comes from `PANEL_SATURATION` (26 electronics, 10 CDNOW), roughly a quarter of the
calibration window. It is a half-saturation constant, not a bin edge, so
`check_arm_depth`'s degeneracy case does not apply, but the same "does calibration have
mass where the holdout will sit" judgement does.

`ar_log` deliberately keeps `cumulative_transactions` rather than swapping in
`transaction_rate`: it is the arm that isolates the coordinate change, against
`ar_unbounded`, with the feature *set* held fixed.

## Design, and the measurements it rests on

Every number below comes from the 320 archived replications (4 arms x 2 panels x 40),
rescored per replication rather than read off the pooled table.

### Replications: 100 per arm, five shards of 20

40 is enough to detect the unbounded arm's blow-up — that needs about 3 — and not enough
for the comparison that actually decides this, which is against the *winning flag arm*.
Smallest difference detectable at 80% power, alpha = 0.05 two-sided:

| comparison | SD | n=40 | n=100 | n for 5 pts |
| --- | ---: | ---: | ---: | ---: |
| bias vs electronics `ar_bounded_52` | 14.5 | 9.1 pts | 5.7 pts | 132 |
| bias vs cdnow `ar_bounded_16` | 13.1 | 8.2 pts | 5.2 pts | 108 |
| bias, new arm as noisy as `ar_bounded_32` | 30.5 | 15.0 pts | 9.4 pts | 358 |
| SD-of-bias ratio | — | 1.57x | 1.33x | — |

Required n grows with the square of the resolution, so beyond 100 the return collapses.

**Extend `no_ar` and the winning flag arm to 100 as well.** The comparator carries half
the variance of the difference; leaving it at 40 costs about 25% of the resolution
(MDE 7.6 instead of 5.7 on electronics) for 240 extra studies out of ~840 total.

### Everything else frozen at the archived values

`N_TRIALS = 50`, `N_SIMULATIONS = 300`, same seed grid. Three reasons in order of force:

- **Comparability is binding.** Changing either would force a re-run of all four existing
  arms before any of them could be compared to a new one.
- **Monte Carlo noise is not the limit.** `Var(total) = sum_cells Var(y)/S` puts the MC
  standard error on `bias_percent` at 0.12-0.18% against an across-replication SD of
  15-18%, i.e. under 1.2% of the variance. More paths buy nothing.
- **More Optuna trials optimise the wrong objective.** The search scores teacher-forced
  validation loss, which this ablation's own docstring says is blind to the rollout
  drift. Raising the trial count could as easily hurt as help.

**Keep the seed grid identical across arms** so the paired comparison is available.
Measured: pairing against `no_ar` correlates at +0.24 to +0.36 and cuts the difference SD
by ~20%; between two AR arms it correlates at about -0.1 and buys nothing. Report both,
and do not claim pairing helps where it does not.

### Endpoints

Report **mean and SD** for `bias_percent`, `mape_aggregate` and `rmse`, plus mean
Spearman, and median with interquartile range alongside every mean. The distributions are
right-skewed (skew +0.4 to +1.3 for the arms of interest, +2.5 for `ar_unbounded`), so
mean and median separate and both belong in the table. At n=100 a mean comparison is
safe; anything landing near a threshold gets a rank-based test to confirm.

**Mean and SD are separate claims and both are pre-registered.** Electronics
`ar_bounded_32` scores `|bias|` 22.5 with a mean of +3.5 and an SD of 30.5; `no_ar` scores
22.4 with a mean of +22.0 and an SD of 14.9. The first is unbiased on average and
unreliable per run, the second reliably biased. One column cannot carry both.

### RMSE is reported, and must not be a decision endpoint

Measured, on the real panels, against a forecast of zero in every cell:

| panel | all-zero forecast | best real arm | span |
| --- | ---: | ---: | ---: |
| electronics | 0.3775 | 0.3760 | **0.0015** |
| cdnow | 0.1521 | 0.1474 | **0.0047** |

The panels are 98.0-98.6% zeros, so predicting nothing is within 0.4% of the best model
this package produces. At n=100 the detectable RMSE difference is 0.0002, well inside
that span, so an RMSE test would return highly significant p-values across a range in
which the degenerate solution is nearly optimal. Report it with the all-zero reference
printed beside it, and let the decision rest elsewhere.

**What actually guards against a degenerate forecast is bias with Spearman.** An all-zero
forecast scores `bias_percent` -100 and `mape_aggregate` 100. A constant non-zero
forecast can reach bias 0 but has Spearman 0. The two together exclude both degenerate
forms; MAPE adds a scale-free check on the level.

### Acceptance

Per arm, all of the following, on both panels, with the two panels reported separately
rather than pooled:

1. **Level.** Mean `bias_percent` not further from zero than the winning flag arm, and
   mean `mape_aggregate` no worse than it.
2. **Reliability.** SD of `bias_percent` no more than 1.33x the winning flag arm's, which
   is the smallest ratio n=100 can resolve.
3. **Discrimination.** Mean Spearman non-inferior to the winning flag arm at a margin of
   **0.02 absolute**, one-sided. That is ~7% relative on electronics (0.267) and ~5% on
   CDNOW (0.438), and needs 24-40 replications against the observed SDs, so 100 leaves
   headroom for a noisier arm.
4. **Not degenerate.** RMSE at least 0.001 below the all-zero figure above, as a
   guardrail rather than a comparison.

Criteria 1-3 within one arm form an intersection-union test — every one must pass, so no
correction applies inside an arm. Across the **three new arms within a panel**, apply
Holm. The two panels are independent replications of the same question, not a pooled
family.

## Note on cost

~840 studies at the measured 150-260 s each, so roughly 35-60 single-worker hours, a few
hours across the usual eight. `VastAI/Rules.md` sharding and machine selection apply
unchanged. Run a `--n-studies 1 --n-trials 5` probe on a fresh box before committing a
shard to it.

## Comments

**2026-09-10 — plumbing verified, not yet run.**

All three arms build on both panels and complete a full study end to end: dataset,
Optuna search, ADR-0008 refit, Monte Carlo rollout with the new AR channels rebuilt each
step, metrics written. Probes at 2 trials / 10 simulations / 1 replication, so the
numbers they produced are meaningless and were discarded with the suites.
