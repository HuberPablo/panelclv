# Reconcile the ADRs and CONTEXT.md with the redesign

Type: grilling
Status: open
Blocked by: 06

## Question

The ADRs document why the code is shaped as it is. After an open redesign, three of the
five may describe a shape that no longer exists — ADR-0002 (the simulator lives with the
model), ADR-0004 (frozen reference implementations) and ADR-0005 (the embedder seam).
Docs that lie are worse than no docs.

**Audit 04 adds a fourth, and it is a stronger case than the other three:** ADR-0003
(rollout-composite selection) describes a decision that is *unreachable from the production
path*. `studies/runner._run_neural_model` passes neither `selection_metric` nor
`removable_features`, and `StudySuiteConfig` has no field for either, so the rollout selection
the ADR records — and the entire covariate-subset search — can only be reached through three
`run_optuna_study` cells in the two `Data_integration` notebooks, never from `run_studies.py`
or any suite cell. Worse, audit 04 measured that the selection metric ADR-0003 aligns on does
not agree with the reported one: `tuning.weekly_aggregate_rollout_metrics` recomputes RMSE over
customer-summed totals where `compute_forecast_metrics` uses per-cell values (62× apart) and
MAPE under a different estimator (masked at 5.0, clipped at 300); only bias matches. So
ADR-0003's "aligning selection with the metric actually reported" holds for one of three
metrics. Decide whether the ADR is amended to say notebook-only, or the wiring is a bug to fix.

Decide, per ADR: still true / amended / superseded. Then decide what ADR-0006 records —
the target architecture is exactly the kind of decision the format exists for.

**`CLAUDE.md` needs a line corrected too**, and it is the highest-traffic doc in the repo —
every agent session reads it. It states: "`compute_forecast_metrics` is the single scoring
authority — Plots, tables and study results all delegate to it so they agree to the last
decimal." Audit 04 verified that is true *within* `evaluation/` and `studies/` (it reproduced
all nine stored `results.csv` values of an archived suite to full float precision) but false
package-wide: `tuning.weekly_aggregate_rollout_metrics` recomputes all three metrics, and
`evaluation/__init__.py:9-13` makes the same claim scoped narrowly enough to be true. Either
the `tuning` carve-out gets spelled out or that function is renamed off the shared vocabulary —
`rollout_mape` currently sits one word away from `mape_aggregate_style` while computing
something else.

`CONTEXT.md` needs the same pass. It defines *Trial*, *Study* and *Study suite* but
never *experiment*, while `experiments/` is a subpackage — a name outside the project's
own vocabulary is why nobody can say what separates it from `studies/`. Whatever 06 and
09 decide, the vocabulary follows.

Per the charting session: selective, not exhaustive. The synthesis outcome and any
invariant-collapsing decision become ADRs; routine kill/keep calls stay in tickets.
