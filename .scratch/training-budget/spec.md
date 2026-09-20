# Training budget: is the collapse a training recipe problem?

Status: `ready-for-agent`

## The question

Every neural study in this package trains until `fit_model` sees 7 consecutive epochs
without an improvement in validation cross-entropy. `docs/training-budget.md` shows that
this ends every run far earlier than the paper we reproduce ends its own: our archived
winners received about 32 gradient updates, the notebook's model about 2,300.

So: **if the model is trained the way the paper trains it, does the electronics forecast
stop collapsing?**

Collapse here means the two things `docs/benchmarks-real-panels.md` reports for
electronics — MAPE 70.8 and a per-customer Spearman of 0.032, i.e. nearly the same
forecast for every customer.

## The arms

Two models × four arms, on electronics only, 20 replications each.

Held fixed: the windows from `run_real_panel_benchmarks.WINDOWS`, the temporal validation
split (ADR-0001), the ADR-0008 refit, 200 Monte Carlo paths, seeds 43–62, cross-entropy.

| arm | what it is | why |
| --- | --- | --- |
| `archive` | today's settings: lr / weight decay / batch searched, `patience=7`, `n_epochs=100`, 100 trials | the control — reproduces family N, which also gave this panel 100 trials |
| `paper` | the notebook's recipe, pinned, 1 trial: `lr=1e-3`, `weight_decay=0.0`, `batch_size=32`, `patience=5`, `n_epochs=150` | the literal reading of the paper, with no hyperparameter selection at all |
| `paper90` | the same, plus `min_epochs=90` | the paper's recipe trained for the paper's ~90 epochs. Measured before launch: `paper` alone stops at epoch 1 here (see `issues/01`), so it copies the settings without copying the training |
| `floor50` | `archive`'s search plus `min_epochs=50`, `n_epochs=300`, 100 trials | whether a warm-up floor recovers the same ground while keeping the search — the only option open to models with no published recipe |

Four arms x two models x 20 replications = **160 suites**.

Exact parameter blocks are in `issues/01-paper-recipe-arm.md` and
`issues/03-runner.md`; the runner declares them once, in `scripts/run_training_budget.py`.

## Why `paper` is pinned rather than searched

`docs/benchmarks-real-panels.md` shows that a study's winning validation loss does not
predict its holdout bias (|ρ| ≤ 0.36 in 31 of 32 cells) and says nothing about ranking.
A searched arm therefore confounds "trained longer" with "selected differently". Pinning
every hyperparameter to the notebook's values removes the search entirely, so the arm
measures one thing.

## What would answer the question

- **`paper` clearly better than `archive` on MAPE and Spearman** → the collapse is our
  training recipe. Follow-up: add batch 32 to the registry's search space and re-run
  family N.
- **`paper` ≈ `archive`** → training volume was not the problem. The inputs are: a
  cluster label moves electronics Spearman from 0.036 to 0.29
  (`docs/insights-cluster-ablation.md` §5.1). This line of work stops.
- **`floor50` ≈ `paper90`** → the floor is the general fix, and the developed models get
  it without needing a published recipe.
- **`paper90` ≫ `paper`** → the settings were never the point; the epochs were. That is
  already the direction the pre-launch probes point (`issues/01`).

Bias is **not** a deciding metric here: `docs/benchmarks-real-panels.md` measures a refit
noise floor of 8.9 points of sd on electronics, and the differences at stake are smaller
than that.

## Evidence this rests on

Measured 2026-09-20, scripts beside this file, outputs in `results/`:

| script | what it measures |
| --- | --- |
| `epochs.py` | best epoch and epochs run across every archived real-panel trial |
| `probe_patience.py` | validation curves with early stopping disabled, three panels |
| `end_to_end.py` | patience 7 vs no early stop, through the real refit and forecast |
| `trials_by_bs.py` | why the search selects batch 256 |
| `spearman_vs_epoch.py` | per-customer Spearman recomputed from archived forecasts |
| `hparams.py` | training volume against selected hyperparameters and arms |
| `seed_coupling.py` | whether the replication RNG coupling reaches the archive (it does not — `issues/05`) |
| `paper_split_check.py` | the paper's recipe under the paper's OWN customer-wise split: 28-56 epochs, against epoch 1 under ours |
| `family_t_stats.py` | each arm against the control, Mann-Whitney over 20 replications |

Run them with the project venv and `PYTHONPATH=src` from the repo root.

## Tickets

| # | file | what |
| --- | --- | --- |
| 01 | `issues/01-paper-recipe-arm.md` | the `paper` arm's exact parameters |
| 02 | `issues/02-min-epochs-floor.md` | `min_epochs` in `fit_model` and the pruner |
| 03 | `issues/03-runner.md` | `scripts/run_training_budget.py` and the vast.ai split |
| 04 | `issues/04-register-family.md` | family T in `docs/studies-run.md` |
| 05 | `issues/05-measure-seed-coupling.md` | how far the replication RNG coupling reaches |
| 06 | `issues/06-report-and-decide.md` | read the result, decide what changes |
