# Where does thesis-figure code live?

Type: grilling
Status: open
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
