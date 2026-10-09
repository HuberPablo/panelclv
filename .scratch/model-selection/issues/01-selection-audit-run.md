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

**9 Oct, run complete: 320/320 studies, $43.30 in all.**

- Night 1 (8–9 Oct) stopped at the $28 watchdog with 209/320. The estimate above was
  wrong by ~1.6×: vast boxes cost ~$0.15/h, not the ~$0.06/h assumed, and the measured
  cost came out at ~$0.137 per study (~$44 for 320).
- Day 2 (9 Oct) filled the 111 gaps for $14.66: workers stride over a frozen list of the
  missing suites (`run_worker`), so each box gets an equal share.
- **5y LSTM + AR_52 runs out of memory on 12 GB GPUs** (`torch.OutOfMemoryError`, a 9.8 GiB
  LSTM allocation, for the larger sampled architectures). This is why that cell stalled
  at 9/20 on night 1. Eight such studies were rerun on 24 GB RTX 3090s, where a 5y study
  takes ~40 min. Use `gpu_ram >= 24` for 5y LSTM + AR_52.
- The frozen list first lived under `Studies/`, and `pull_results` copied every box's
  `Studies/` tree back over it, so replacement boxes were seeded with a stale list and
  either idled or redid finished studies (4 boxes destroyed, their results pulled first).
  The list now lives at the repo root (`4cd2b90`).
- Leftover checkpoints that were pulled mid-run were deleted locally; only
  `selection_audit.csv`, `results.csv` and the suite configs remain (566 MB).
