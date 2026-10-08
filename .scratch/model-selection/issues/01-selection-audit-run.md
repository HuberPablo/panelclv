# 01 — Selection audit: does the search pick well, and does one setting retrain alike?

Status: ready-for-agent

## Questions

From the model-selection plan (8 October 2026), §2.1.1 and §2.1.2:

1. **Does the Optuna search select the best trial?** Under the corrected validation loss
   (ADR-0010), on every panel-calibration, for a model that searches only its training
   settings (ValendinLSTM) and one that also searches its architecture (LSTM + AR_52).
2. **Does the same setting, retrained from scratch, forecast alike?** The winner's
   settings retrained 4 times per study, everything else unchanged.

Both are asked **with and without the refit** (ADR-0011): every scored model is forecast
from its checkpoint as it stands and from a refit of it, paired on the same Monte Carlo
seed.

## Why no archive answers it

- Families R and V scored non-winning trials, but under the biased validation loss, on
  2y only (R) or electronics and CDNOW only (V), and never LSTM + AR_52.
- The stopping-rule run (SR, 7 Oct) used the corrected loss on all 8 panel-calibrations,
  but deleted every losing checkpoint.
- No archived forecast exists without a refit, apart from the 4-study pilot
  (`docs/hyperparameter-search.md` §6.4).
- Retrains of a fixed setting exist only as the pinned recipe arms (T, U, X), which are
  not search winners, and as refit-only noise (R).

## Runner

`scripts/run_selection_audit.py`. Per work item (calibration, panel, model, replication):

- one 100-trial search, SR `patience7` settings, every checkpoint kept;
- the winner and 15 random other completed trials, each forecast without and with the
  refit (500 paths, the study's seed);
- 4 from-scratch retrains of the winner, each forecast both ways;
- `selection_audit.csv` in the suite, one row per scored model (`kind` = winner / trial /
  retrain); checkpoints deleted afterwards.

20 replications × 2 models × 8 panel-calibrations = 320 work items. The global torch
RNG is reseeded from OS entropy before every refit and retrain, so they are genuine
draws despite the forecast's `torch.manual_seed` (`.scratch/training-budget/issues/05`);
checked locally: 4 retrains gave 4 different validation losses and forecasts.

## Analysis owed (not in the runner yet)

Per study, then bootstrapped across the 20 studies per cell (`evaluation.effects`):

- the winner's place among its scored trials on holdout MAPE, |bias| and Spearman;
- Spearman of validation loss against each holdout metric, across the study's trials;
- the winner's holdout score minus the mean of its scored trials (what the search buys
  over a random pick);
- the spread across the 4 retrains, beside the spread across the 20 studies' winners;
- no-refit minus refit, paired over checkpoints (test 4 / A10 of
  `docs/hyperparameter-search.md`).

Only completed trials are scored, so "the winner's place" is among trials the pruner let
finish. Whether the pruner drops would-be winners is test 1b / 2, not this run.

## Comments

**8 Oct, budget.** $35.14 of vast credit. Measured locally: a 500-path holdout rollout
takes ~45 s on the 2y/3y panels, ~150 s on 5y; each scored model needs two. At 20 trials
and 5 retrains the run was ~510 box-hours (~$31-37 with overhead); cut to 15 and 4,
~400 box-hours (~$22-27). Budget watchdog set to $28. Two fleets as for SR: 2y+3y
(280 items) and 5y (40 items).
