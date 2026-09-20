# 03 — `scripts/run_training_budget.py` and the vast.ai split

Status: `ready-for-agent`
Blocked by: 02 (for the `floor50` arm only; `paper` and `archive` need no code change)

## The runner

Modelled on `scripts/run_real_panel_benchmarks.py`, which it imports `WINDOWS` and
`panel_config` from so the windows cannot drift apart.

A **work item is (model, arm, replication)** — one suite each, so a lost box costs one
replication rather than an arm. Suites are named
`training_budget__<Model>__electronics__<arm>__r<NN>`; 2 models × 3 arms × 20
replications = **120 items**.

The work list is **arm-major**, and worker `i` of `N` takes `i::N` (VastAI/Rules §5), so
every worker gets a slice of all six suites and a lost box thins them evenly.

Flags: `--preflight`, `--worker i/N`, `--check-complete`, `--report`.

## Two panel configs

ValendinLSTM refuses any non-embedded channel (known_failures F11), so the two models
cannot share one config:

- **ValendinLSTM** — `run_real_panel_benchmarks.panel_config("electronics", "2y")`
  verbatim: `Transactions` and `week` embedded, nothing else.
- **LSTM** — the archived `no_ar-no_cluster-valendin` arm: `Transactions` embedded,
  `add_week_sin_cos=True`, no AR channel, no cluster label. This is the arm whose ranking
  collapses to 0.036.

## The six suites

Fixed everywhere: electronics; `WINDOWS["electronics"]` (calibration 1999-01-01 →
2000-12-31, validation from 2000-01-01, holdout 2001, `clip_target_upper=6`); temporal
split (ADR-0001); default refit (5 epochs, batch 512, lr 1e-3); 200 Monte Carlo paths;
20 replications, seeds 43–62; `loss_type="cross_entropy"`.

| suite | model | search_space | training | trials |
| --- | --- | --- | --- | ---: |
| `VL-archive` | valendin_lstm | `learning_rate (1e-4, 3e-3, "log")`, `weight_decay (1e-6, 1e-2, "log")`, `batch_size {64,128,256}` | `n_epochs=100, patience=7` | 25 |
| `VL-paper` | valendin_lstm | `learning_rate=1e-3`, `weight_decay=0.0`, `batch_size=32` | `n_epochs=150, patience=5` | 1 |
| `VL-floor50` | valendin_lstm | as `VL-archive` | `n_epochs=300, patience=7, min_epochs=50` | 25 |
| `L-archive` | lstm | `embedder="valendin"`, `lstm_hidden_size {32,64,128}`, `dense_units {32,64,128}`, `dropout (0.0, 0.4)`, `learning_rate (1e-4, 3e-3, "log")`, `weight_decay (1e-6, 1e-2, "log")`, `batch_size {64,128,256}` | `n_epochs=100, patience=7` | 25 |
| `L-paper` | lstm | `embedder="valendin"`, `lstm_hidden_size=128`, `dense_units=128`, `dropout=0.0`, `learning_rate=1e-3`, `weight_decay=0.0`, `batch_size=32` | `n_epochs=150, patience=5` | 1 |
| `L-floor50` | lstm | as `L-archive` | `n_epochs=300, patience=7, min_epochs=50` | 25 |

`--report` prints each arm's bias / MAPE / RMSE from `results.csv` and recomputes
per-customer Spearman from the stored `Predictions/`, with the family N electronics rows
printed beside them as the reference.

## Running it on vast.ai

- **Data by rsync** (Rules §3, never regenerate):
  `Datasets/Dataset_clean/electronics_customer_week_panel.csv` → each worker's
  `/root/panelclv/Datasets/Dataset_clean/`. It is a couple of MB.
- **Launch `paper` and `archive` first.** They answer the question on their own and cost
  about 9.5 h of GPU time between them; `floor50` is another ~15 h and only matters if
  the paper arm wins.
- Estimated from workstation timings: `*-paper` ~3 min per study, `*-archive` ~11 min,
  `*-floor50` ~22 min. Six workers puts the first two arms at roughly 1.5 h wall clock.
  Choose machines on $/study, not $/hr (Rules §7).
- **Return:** rsync `Studies/training_budget__*` back to the orchestrator, then
  `--check-complete` and `--report` there.
