# 03 — Execute the ledger's kill list

**What to build:** every symbol the ledger ruled dead is gone, and a test asserts it stays
gone. Nothing in the package has zero callers while reading as live surface.

**Blocked by:** 02

**Status:** done

Source: `.scratch/package-simplification/issues/05-reachability-ledger.md`,
`06-target-architecture.md` (decision 2)

## Scope

**11 rows, ~187 lines** — not the ledger's original 13 rows / 282 lines. The loss-variant
cluster is **kept whole**, including its tests: that reverses two of the ledger's kills and
is a settled decision, not an oversight.

The largest single unit is the forecast-run module — no callers at all, and it carries a
fifth on-disk prediction layout with no writer. The dead rollout aliases and the
suite-distribution function go with it.

**Every deletion in this issue cites its ledger row** in the commit, so the reasoning does
not have to be re-derived. The ledger is at
`.scratch/package-simplification/ledger.csv` / `.md` — 163 rows covering all 10,139 lines.

## Two rules that bind this issue

- **A notebook import is not a caller.** Only a call keeps a symbol alive. Deleting an
  import-only symbol **requires stripping the import line from the notebook in the same
  commit** — the notebook API test resolves every import and will go red otherwise.
- **The thesis carve-out overrides.** Anything that produced a figure or a number in the
  thesis is alive regardless of callers. Check before deleting.

## Test

Retired-name assertions append to the existing retired-symbols pattern in the import test.
This is the mechanism every later kill issue reuses.

- [x] All 11 ledger rows deleted, each citing its row
- [x] The loss-variant cluster and its tests untouched
- [x] Any notebook import lines for deleted symbols removed in the same commit — none existed
- [x] Retired names asserted gone in the import test
- [x] Golden test green; notebook API test green

## Comments

### The 11 rows, as deleted

| # | ledger row | site | lines |
|---|---|---|---|
| 1 | `models/__init__` :: `mc_simulate_one_path (alias)` | `models/__init__.py` | 1 |
| 2 | `models/__init__` :: `mc_simulate_transformer_path (alias)` | `models/__init__.py` | 1 |
| 3 | `evaluation/forecast_run` :: `(module)` | file deleted | 111 |
| 4-10 | `ForecastRun` + `.new` / `.open` / `.path` / `.save_config` / `.save_predictions` / `.predictions` | in that file | — |
| 11 | `studies/analysis` :: `group_metrics_suite_distribution` | `analysis.py:701-778` | 78 |

189 lines of body plus 4 `__init__` export lines — against the issue's ~187 estimate.
Rows 4-10 are the seven members of the `ForecastRun` unit and have no separate line
count; the ledger scores them under the module row, which is why the arithmetic lands
on the module's 111.

### No notebook edit was required

The issue's first binding rule (a notebook import is not a caller, and killing an
import-only symbol means stripping the import in the same commit) did not fire: a grep
for all five names across `notebooks/`, `notebooks/archive/`, `scripts/`, `tests/`,
`docs/`, `CONTEXT.md` and `CLAUDE.md` returned only the definition sites and one doc
paragraph. The three rows that *would* have triggered the rule — `compute_class_weights`,
`alignment_check`, `describe_dataset` — are all kill-*candidates*, none of them in this
issue's 11.

### One doc reference the issue's inventory did not carry

`docs/running-a-model.md` listed `group_metrics_suite_distribution` in its
`studies/analysis.py` function table and then closed section 10 with a paragraph naming
it as *the* way to honour the project's report-a-distribution convention. Deleting the
function without that edit would have left the doc prescribing a call that raises.

The paragraph was rewritten rather than dropped, because the convention outlives the
function: `study_metrics` / `compare_study_metrics` re-score each study separately and
take `standard_deviation=True` / `confidence_interval=True` (verified against their
signatures — both default to `False`, so the doc says "report the spread" as an option,
not as the default). What genuinely dies with row 11 is the per-*segment* spread:
`group_metrics_suite_table` scores the across-studies mean forecast, so it yields one
value per (group, metric) and no variability. The paragraph now says that explicitly,
so the gap is recorded where someone would look for the function.

### Nothing was orphaned by the `analysis.py` cut

`group_metrics_suite_distribution` was the module's only user of nothing: its five
private helpers — `_other_ids`, `_actuals_from_panel`, `_discover_models`,
`_prediction_index`, `_is_deterministic_model` — all retain other callers in the file,
checked after the deletion. Ticket 12's decision 7 splits this module three ways later;
this cut removes 78 of the 1195 lines it plans around.

### The retired-name guard

Appended to `tests/test_imports.py` as `test_retired_dead_surface_is_gone`, next to the
existing `test_retired_metric_helpers_are_gone` it copies. It asserts `hasattr` is false
on the three parent packages and that `panelclv.evaluation.forecast_run` raises
`ImportError` — the module-level check being the one that catches a file restored from
git without its export. This is the mechanism issues 04-15 extend.

Not added to `test_notebooks_current_api.RETIRED`: that blacklist covers names a
notebook might still *mention* in prose, a comment or a stored output, and none of these
five has ever appeared in a notebook in any form.

### `models/__init__.py`'s comment was wrong twice over

The `__all__` preamble justified keeping the two aliases importable-but-unadvertised as
"the per-path simulator entry points", and the import block above it called the `mc_*`
set "the short aliases used throughout the notebooks" — a claim audit 03 verified false
for exactly these two. Both comments now say what is true: the steppers are internals of
the two forecast functions and are not re-exported, and the aliases that are left
(`mc_forecast`, `mc_forecast_transformer`, `mc_compute_metrics`) are the ones the
notebooks actually call.

### Verification

Full suite: **192 passed** in 20.8 s under `venvs/thesis_rocm` — including the golden
end-to-end test across all four model families (issue 01) and the notebook API test. The
nine warnings are the pre-existing `RuntimeWarning`s from the Pareto sampler's log-space
arithmetic, unchanged by this issue.
