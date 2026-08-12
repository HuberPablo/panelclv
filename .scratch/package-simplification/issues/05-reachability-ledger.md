# Merge the audits into one kill/keep/refactor ledger

Type: task
Status: resolved
Blocked by: 01, 02, 03, 04

## Question

Three lane audits plus the golden run's reachability trace, merged into a single
ledger: one row per module and public symbol, with a proposed verdict (kill / keep /
refactor), the evidence behind it, and every caller.

Mechanical work, deliberately separated from ticket 06 so the synthesis session spends
its context on judgement rather than on collating.

Two things the merge must surface, because no single lane audit can see them:

- **Cross-lane duplication** — a concept implemented in two lanes at once.
- **Conflicts** — where two audits propose incompatible fates for code they share.

Resolve by recording where the ledger lives and summarising its headline counts.

## Answer

**Where the ledger lives.** Two files under `.scratch/package-simplification/`, generated
from one table so they cannot disagree:

- **`ledger.csv`** — the rows, machine-sortable: `lane, module, symbol, kind, lines,
  verdict, evidence, callers_src, callers_live, callers_dead_scripts, callers_oneshot,
  callers_tests, traced, audit`.
- **`ledger.md`** — the narrative (method, verdict vocabulary, headline counts, the
  cross-lane duplication list, the conflicts, the dead-code-rule inconsistencies)
  followed by the same rows rendered as a table per subpackage.

Both formats because the two jobs differ: ticket 06 needs to *count and sort* verdicts
(CSV) and to *read* the conflicts and duplication in prose (MD). The row set is
identical.

**Method.** Every module-level public symbol was enumerated by AST (plus the public
methods of `PanelConfig`, `ModelSpec`, `StudySuiteConfig`, `ForecastRun`, `FitResult`,
`Embedder`, and the five `mc_*` aliases in `models/__init__.py`), and every reference to
each was classified by AST as import / call / attribute read / bare name, across five
populations kept separate: `src/panelclv/`, the live entry points, the two dead plot
scripts, the one-shot scripts, `tests/`. Verdicts and evidence come from the lane audits,
with the audit named per row; eleven symbols no audit mentioned carry `audit=none` and a
mechanically-derived verdict rather than a guess.

**Headline counts — 163 rows: 32 modules, 131 public symbols, all 10 139 lines.**

| verdict | all | modules | symbols |
| --- | --- | --- | --- |
| `keep` | 70 | 7 | 63 |
| `refactor` | 74 | 24 | 50 |
| `kill` | 13 | 1 | 12 |
| `kill-candidate` | 3 | 0 | 3 |
| `conditional-10` | 3 | 0 | 3 |

By lane: data 11 keep / 20 refactor; model 22 / 14 + 4 kill + 1 kill-candidate;
experiment 37 / 39 + 9 kill + 2 kill-candidate + 3 conditional-10.

The 13 kills are three clusters — `evaluation/forecast_run.py` (module + `ForecastRun` +
its six methods), `losses.FocalLoss` / `SquaredEMDLoss` / the two dead `mc_simulate_*`
aliases, and `analysis.group_metrics_suite_distribution` — about **282 lines, 2.8 % of the
package**. Nothing else in `src/panelclv/` is dead by the map's rule; the rot is 74
`refactor` rows. Of the 63 `keep` symbols the golden trace reached 8; unreached is not
death.

**Cross-lane duplication: 26 entries (D1–D26)** in `ledger.md`, deduped and reconciled.
Highlights: **D1 is new** — `DataBuilder` is defined twice, `experiments/experiment_utils.py:41`
(DataLoader-typed) and `tuning/optuna_tuning.py:116` (`Any`-typed), neither importing the
other; both audits found only the prose triplication of the same protocol. **D6 reconciles
the writer counts** — 5 prediction layouts, 4 with a live writer, 1 (`ForecastRun`) with
neither writer nor reader, all five funnelling through `save_predictions_to_csv` (verified
for `aggregate_suite_predictions` at `analysis.py:212`); audit 03's "3 writers" omitted the
suite-level aggregate. **D7 merges the calendar findings** into 4 week-index conventions +
3 period-length tables (30.0 vs 30.4368 vs 7.0), two of which both feed
`compute_pareto_predictions`. **D11 reconciles the registry counts** to 7 code enumerations
across 3 subpackages. **D4** is the ADR-0002 breach with prediction I/O in a plotting module
as its cause. Also merged: D16 target-channel derivation (1 producer, 6 re-derivations),
D17 `id_col` (2 fallback strings, ~9 sites, already visible on disk), D18 `"Transactions"`
24× in 9 modules, D19 six `datetime.now()` folder-name sites, D20 four cross-boundary
private imports each with a stated legitimate need.

**Conflicts found (C1–C8 in `ledger.md`).**

1. **C1 — audit 03's stepper ruling rests on a witness that cannot execute.** Its evidence
   #3 is "a live entry point already takes the silent branch" (`main_plot.py`), and it calls
   option (B) the one that "closes the demonstrated bug". Audit 04 showed the script raises
   `ValueError`; I verified further by running it that it dies even earlier, at `:277` in
   the `_load_dataset` stub (`NotImplementedError`), so the miscrossing is unreachable in
   practice. (B) keeps its structural argument and loses its bug-fix argument, and
   `forecast_from_checkpoint` now has "delete" on the table as well as "fix".
2. **C2 — the two audits disagree on whether `compute_forecast_metrics` is the single
   authority.** 03 assumes yes (so only the doc's key name is wrong); 04 proves no. CLAUDE.md
   needs two corrections, not one.
3. **C3 — `save_predictions_to_csv`'s "6 modules and 5 entry points" overstates by audit
   04's own rule**: verified, its live entry-point *call* count is 0 (notebooks import only,
   both scripts dead); it is alive through three live `src/` callers.
4. **C4 — audit 03 is split against itself on `compute_class_weights`** (§1 "live callers,
   no live consumer" vs a conclusion extending the kill verdict), and killing it edits two
   live notebooks. It is one decision with the whole loss-variant cluster.
5. **C5 — `validate_valendin_lstm.py` is both duplication to remove (02) and a floor gate
   that must not move (03/map)**; its private copies may be deliberate insulation.

**Dead-code rule applied inconsistently — four places.** (a) *Granularity*: audit 02's
over-exposure table calls `normalize_embedded_cols`, `PanelConfig.data_config` and
`.schema` "no external callers", but all three are used from
`data_preparation/dynamic_panel_dataset.py` (`:76`, `:605`, `:617`) — it applied the prong
at lane level where 03/04 applied it per module (3 rows affected). (b) *One-shots*: counted
as callers in audit 02's `observed_past` argument, explicitly excluded in audit 03's
`FitResult.history` argument (no verdict affected). (c) **Import vs call in notebooks — the
consequential one**: systematic in audit 04, ad hoc in 03, untested in 02. I re-verified
audit 04's list by AST and it is correct; ticket 06 must pick a reading, because under
"import ≠ caller" `alignment_check` and `describe_dataset` are killable and under
"import = caller" they survive. One concrete stale row it produces: audit 03's
`InferenceMultinomialTransformerModel(seq_len=)` "two callers" is one, the other being the
dead `main_plot_covar.py:96`. (d) *Entry-point drift*: audit 03 predates the correction and
one of its rulings leans on `main_plot.py` running (C1). The **thesis carve-out** was
applied consistently by all three.

No code, test, notebook or doc was changed; `map.md` untouched (the orchestrator owns it).
