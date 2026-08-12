# Where does thesis-figure code live?

Type: grilling
Status: resolved
Blocked by: 04

## Question

Roughly 730 lines across two modules exist to produce specific thesis figures and
findings rather than to serve a general package:

- **`studies/pnbd_grid.py`** — `seasonality_grid`, `alive_volume_ratio_grid` and
  `dead_volume_leakage_grid` (lines 185–485, half the file) are the engine behind
  `scripts/make_grid_figures.py`'s two figures. They read the *generator's* ground truth
  — true death week `tau`, the true seasonal multiplier — so by construction they can
  never run on a real panel. Only callers: `notebooks/Pareto_Datasets.ipynb` and
  `scripts/make_grid_figures.py`. It also reaches into another module's private
  `_seasonal_weekly_multiplier`, and hardcodes weekly (`_DETREND_WINDOW = 13`,
  week-of-year arithmetic).
- **`studies/analysis.py`** — the 204-line `plot_suite_forecast` plus its helpers and
  the two `group_metrics_suite_*` functions, carrying a hardcoded
  `_CANONICAL_GROUPS = ("At Risk", "Opportunity")` — a thesis segmentation vocabulary
  baked into package code. This is plotting living in `studies/` while
  `evaluation/plot_utils.py` (609 lines) exists for it.

The Q7 carve-out protects this code from deletion — it produced published results. But
*protected* is not the same as *must be package surface*. The likely answer is
**relocation, not removal**: move it to `scripts/` (or a `thesis/` directory) where the
carve-out still guards it and it stops counting as `panelclv`'s API.

Decide: relocate, generalise, or keep. If relocate, decide what the boundary is — which
of the ~350 genuinely-library lines in `analysis.py` stay behind.

## Answer

Settled with Pablo over one round of `/grilling`, 2026-08-12. **Eight decisions, plus one
ruling left conditional on ticket 10.** This ticket also discharges the scope tickets 06
and 09 handed it: `evaluation/`'s internal shape (06's Q14) and `plot_utils.py`'s name plus
the `_pareto_from_data` private import (09's decision 1).

### Correction: this ticket's central premise was wrong

**`scripts/make_grid_figures.py` does not use the three grid functions.** It imports exactly
one name from `pnbd_grid` — `collect_grid_results` — and re-implements its own
`dead_customer_mass()` and `shape_correlation()` for its two figures. So `seasonality_grid`,
`alive_volume_ratio_grid` and `dead_volume_leakage_grid` (lines 185-485) are **not** "the
engine behind `make_grid_figures.py`'s two figures". Their only caller is
`notebooks/Pareto_Datasets.ipynb`, where they are genuinely *called* — `tbl = seasonality_grid(...)`,
`tbl_alive = alive_volume_ratio_grid(...)`, `L = dead_volume_leakage_grid(...)` — so the map's
import-is-not-a-caller rule leaves all three alive.

This matters because the ticket's own suggested answer ("relocate to `scripts/`") would have
moved them next to a script that ignores them, and its implied alternative (delete, since the
figures come from the script) would have removed the notebook's only path to those tables.

**Measured call counts, which drove most of what follows:**

| symbol | notebook **calls** |
|---|---|
| `plot_suite_forecast` | **16** — the most-called function in the package |
| `study_metrics`, `load_model_predictions`, `aggregate_suite_predictions` | 7 each |
| `compare_study_metrics` | 6 |
| `metrics_table` | 5 |
| `plot_weekly_aggregated`, `group_metrics_suite_table`, `describe_suite_dataset` | 3 each |
| `_pareto_from_data` | 2 — **private**, imported across a subpackage boundary |
| `group_metrics_suite_distribution`, `describe_dataset` | **0** — already on the kill list |

### Decisions

1. **The grid finding is a duplication to collapse, not a misplacement.** Two implementations
   measure the same two things on the synthetic grid — the death-state failure and the
   seasonal-tracking strength: the package trio (notebook-facing) and `make_grid_figures.py`'s
   own `dead_customer_mass()` / `shape_correlation()` (which produce the published figures).
   **The script's arithmetic wins**, because its output is what is in the thesis; the package
   functions become that arithmetic exposed for the notebook, and the script becomes a thin
   caller. Folded in: D24's finding that the three package functions share **51 of 55
   byte-identical lines** with each other plus a third copy of the same traversal scaffold —
   factored once as part of the same work.

2. **The test for leaving the package is "can it run on a real panel?", not "did it produce a
   thesis figure?".** The ticket proposed the second; it fails on the evidence, because
   `plot_suite_forecast` produced thesis figures and is called 16 times from notebooks, and
   `scripts/` is not an importable package — relocating there makes code unreachable from the
   notebooks that *are* the thesis's analysis surface. The adopted test is a property of the
   code, readable from it: the three grids read the **generator's ground truth** (true death
   week `tau`, the true seasonal multiplier via `pareto_simulation._seasonal_weekly_multiplier`),
   so by construction they can never run on a real panel. Nothing else in the two modules fails
   that test.

   **Consequence: the three grids are the only code this ticket separates.** No code moves to
   `scripts/` or a `thesis/` directory; the relocation the ticket anticipated does not happen.

3. **`pnbd_grid.py` splits in two, putting the real-panel boundary between two files rather
   than inside one.**
   - `studies/pareto_nbd_grid.py` (ticket 09's rename) keeps `collect_grid_results`,
     `group_summary`, `compare_models_table`, `plot_pattern`, `plot_diff_grid` — all read
     stored results and run on any grid suite.
   - **`studies/synthetic_grid.py`** takes `seasonality_grid`, `alive_volume_ratio_grid`,
     `dead_volume_leakage_grid` and their ground-truth helpers (`_holdout_season`, `_detrend`,
     `_DETREND_WINDOW`).

   A boundary between two files is enforceable by reading an import line; a docstring marker
   inside one file is not. **Amends ticket 09 decision 3**, which renamed the file without
   knowing it would split. An eleventh subpackage was rejected — not worth a folder for 300 lines,
   and ticket 06's "consolidate in place" still governs.

4. **`plot_suite_forecast` stays; `_CANONICAL_GROUPS` is deleted and the group set collapses to
   one encoding.** The function passes decision 2's test and has 16 live call sites. Its constant
   is a different problem than the ticket described: it is not "thesis segmentation vocabulary
   baked into package code" but **the third of five encodings of one set** (D13) —
   `_GROUP_PREDICATES`, `assign_customer_groups`' default, `analysis._CANONICAL_GROUPS`, and the
   `groups=` defaults of both `group_metrics_suite_*`, with the `"Other"` catch-all re-derived at
   three call sites.
   **Survivor: `evaluation/segment_analysis._GROUP_PREDICATES`** — the predicates that *define*
   the groups live there, so its keys are the set by construction and cannot drift from it.
   `"Other"` is derived once, in the same place. **This closes D13's second half**; ticket 06
   decision 10 had listed only the time-flag half.

5. **`evaluation/plot_utils.py` ceases to exist. Three destinations.** After every prior ruling
   only four functions retain live callers, and two of them are not evaluation at all:

   | destination | contents |
   |---|---|
   | `evaluation/predictions.py` | `save_predictions_to_csv`, `load_predictions_from_csv`, `_reduce_to_customer_week` — this **is** ticket 06 decision 6's new prediction-I/O module, and creating it closes the `evaluation` / `models` cycle |
   | `evaluation/plots.py` | `plot_weekly_aggregated`, `metrics_table` |
   | `benchmarks/pareto_nbd.py` | `pareto_forecast`, `_pareto_from_data` — **out of `evaluation/` entirely**, beside `compute_pareto_predictions`, because they build a Pareto/NBD forecast, which is what that file is for |
   | *deleted* | `weekly_aggregate_predictions`, `alignment_check`, `weekly_actuals` (import-only in notebooks; already killed by the map's rule) |

   `evaluation/segment_analysis.py` is unchanged apart from decision 4.
   **This answers ticket 09's deferred naming question:** the surviving plot file is `plots.py`,
   and it earns the name because by then it only plots. The `_utils` suffix is gone from the
   package.

6. **Two private cross-boundary imports are promoted to public, not relocated.** D20 catalogued
   four such imports and concluded each "has a stated legitimate need — which makes this one
   finding about a missing public surface, not four accidents." Two of the four are this ticket's:
   - `_pareto_from_data` → **public** in `benchmarks/pareto_nbd.py`. Two live notebook call sites
     are a public surface whether or not the underscore admits it; a private name two notebooks
     depend on is the worst of both — no stability promise and no discoverability.
   - `pareto_simulation._seasonal_weekly_multiplier` → **public**, since decision 3's
     `synthetic_grid.py` exists precisely to reconstruct the pattern a stored study was
     generated with.

   The other two D20 instances are settled elsewhere: `experiment_utils`' imports of
   `_build_model_for` / `_build_inference_model_for` die with ticket 07, and
   `tests/test_embedders._emb_size` is a test reaching into its own subject.

7. **`studies/analysis.py` (1195 lines) splits into three.** After losing
   `group_metrics_suite_distribution` (0 callers) and `describe_dataset` (0 calls / 3 imports,
   killed by the map's rule):

   | file | contents |
   |---|---|
   | `studies/suite_reader.py` | `load_model_predictions`, `aggregate_suite_predictions`, `_discover_models`, `_read_suite_config`, `_id_col`, `_prediction_index`, `_is_deterministic_model` (~250 lines) |
   | `studies/suite_plots.py` | `plot_suite_forecast` and its helpers (~330 lines) |
   | `studies/suite_metrics.py` | `study_metrics`, `compare_study_metrics`, `_study_metrics_from_data`, `group_metrics_suite_table` (~340 lines) |

   **The split is free at the call sites** — every one of these has live notebook callers and all
   of them import from `panelclv.studies`, so zero notebook edits. The justification is not size:
   a single 1195-line file is what let `_CANONICAL_GROUPS` and a second copy of the Student-t
   interval hide inside it, both found only by audit.

8. **The Student-t interval is the third copy, not a plot helper — D22 closes here.** It is
   implemented three times: `analysis._across_study_band:403-404`,
   `analysis._study_metrics_from_data:918-932`, and `pnbd_grid._mean_ci:495-496` (whose
   `1-(1-ci)/2` is algebraically `0.5+ci/2`). **One implementation, living in
   `studies/suite_metrics.py`**; `_across_study_band` and `_mean_ci` both call it. The band a
   plot draws and the CI a table prints are the same number, and that they were computed twice
   *inside one file* is the strongest single argument for decision 7's split.
   D22 named this ticket as its blocker; it is now unblocked and closed.

### Left conditional: ticket 10

`forecast_from_checkpoint` and `holdout_actuals_NT` live in `plot_utils.py`, and **ticket 10 —
still open — decides their fate** by ruling on `main_plot.py` / `main_plot_covar.py`. Decision 5's
three-file shape holds either way:

- if ticket 10 revives a plot script, both symbols join `evaluation/plots.py`;
- if it drops both scripts, both die with the file.

Ticket 12 deliberately does **not** absorb that call — it is ticket 10's, and nothing here
depends on which way it goes.

### Budget

**+2 execution issues, ~15 total — at ticket 06's tripwire, not over.** Recorded plainly so
ticket 11 can see the headroom is gone:

| work | issue |
|---|---|
| decisions 5, 6 (first half) — the `evaluation/` reshape | **folds into** ticket 06 decision 6's existing prediction-I/O issue |
| decisions 4, 7, 8 — split `analysis.py`, collapse the group set, one Student-t | **+1** |
| decisions 1, 3, 6 (second half) — split `pnbd_grid.py`, collapse the grid duplication and its 51 triplicated lines, promote the multiplier | **+1** |

If ticket 11 needs headroom, decision 7's three-way split of `analysis.py` is the cheapest thing
to defer — it is the one decision here justified by future legibility rather than by a wrong
number or an unreachable path.
