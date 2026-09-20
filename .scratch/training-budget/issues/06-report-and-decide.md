# 06 — Read the result and decide what changes

Status: `needs-info` (waiting on the run)
Blocked by: 03

## Read

`python scripts/run_training_budget.py --report`, against the family N electronics rows in
`docs/benchmarks-real-panels.md`: ValendinLSTM bias +46.0 ± 14.8, MAPE 70.8, RMSE 0.3770,
Spearman 0.032; Pareto/NBD bias −63.0, MAPE 65.7, Spearman 0.297.

Judge on **MAPE and per-customer Spearman**. Bias is not a deciding metric: the refit
noise floor on electronics is 8.9 points of sd (`docs/benchmarks-real-panels.md`, "The
refit noise floor, and the stopping epoch").

## Then decide, in order

1. **If `paper` beats `archive` clearly** — the training recipe is the problem. Decide
   whether to add `32` to the registry's `batch_size` set
   (`registry/model_registry.py:373`) and whether family N is re-run. Both change
   published numbers, so both go through an ADR entry.
2. **If `floor50` matches `paper`** — the floor is the general fix, and it is the one the
   developed models can use, since they have no published recipe. Decide whether
   `min_epochs` becomes a default in the study runners rather than an opt-in.
3. **If neither beats `archive`** — record that in `docs/training-budget.md`, close this
   effort, and leave the inputs (`docs/insights-cluster-ablation.md` §5.1) as the standing
   explanation for the collapse.

Whatever the outcome, update `docs/training-budget.md` §8 with what was run and what it
showed, and register the family in `docs/studies-run.md` (ticket 04).
