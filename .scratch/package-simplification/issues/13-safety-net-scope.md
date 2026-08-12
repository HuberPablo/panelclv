# How wide does the safety net need to be before the refactor lands?

Type: grilling
Status: open
Blocked by: 06

## Question

Ticket 11 requires that "the golden test from ticket 01 stays green at every step" — it
assumes the ticket-01 net is sufficient to catch a regression. Audit 02 shows it is
narrower than that assumption:

- `tests/test_golden_end_to_end.py` pins exactly ONE weekly happy path
  (`clip_target_upper=4`, two AR features, `add_year_idx` + `add_week_sin_cos`,
  all-`"auto"` embeddings).
- 2068 of the data lane's 2293 lines have no dedicated test. Nothing tests any
  `PanelConfig` validation error, the live monthly path, or `pareto_simulation.py`
  at all — the 544-line generator behind the thesis's synthetic grid panels.
- **Audit 03: the entire Transformer rollout has no test.** `simulate_transformer_path`,
  `run_monte_carlo_forecast_transformer` and `InferenceMultinomialTransformerModel` appear in
  zero test files; `test_golden_end_to_end.py` pins the LSTM path only. 2240 of the model
  lane's 2746 lines have no dedicated test. This is the sharpest case in the map, because
  audit 03 also showed the Transformer + recurrent-stepper crossing fails *silently* rather
  than raising — so any issue reshaping that seam (its options B/C/D) has no net at all
  today. Audit 03 argues this specific test is a **prerequisite** rather than a follow-up.
- **Audit 04: 3389 of the experiment lane's 4880 lines are untested**, with
  `pnbd_grid.py`, `studies/runner.py`, `segment_analysis.py` and `forecast_run.py` (1229 lines)
  having no test at all. `runner.py` is the production entry point for every study suite.
  Note ticket 14 now carves out the archive-format *reader* half of this gap, so this ticket
  decides the rest. Two specific candidates it handed over:
  - **A CPU-scale `run_study_suite` smoke test** — the *writer* is the other half of the on-disk
    floor and is still ungated, so a dropped `_suite_record` key is caught only where an archive
    happens to exist. Ticket 14 reports this is cheap now that its format constants exist: a
    1-trial / 1-epoch / 24-customer suite would pin `_suite_record`, `_model_record`,
    `results.csv` and `metrics.csv` against those same constants.
  - **`tests/test_notebooks_current_api.py` binds calls, not imports**, so a notebook importing a
    name it never calls is checked by nothing. That is a coverage gap *and* the mechanism behind
    the "is an import a caller?" ambiguity ticket 06 must settle.

The redesign is open, so it may reshape precisely the surfaces the net does not cover
(the calendar group, `PanelConfig`'s validation, the schema/embedding resolution). A
behaviour change there surfaces as a slightly different forecast, not a crash — the
failure mode ticket 01 was built to catch.

Decide: is the net widened before the refactor lands, and if so around which specific
surfaces? Or is the golden path plus the two benchmark validation scripts
(`validate_pareto_benchmark.py`, `validate_valendin_lstm.py`) accepted as sufficient,
with the rest carried by review?

Blocked by 06 because "widen the net around X" is only answerable once the target shape
says what X is. Whatever is decided becomes issues in the set ticket 11 cuts, and counts
against its ~15-issue tripwire.
