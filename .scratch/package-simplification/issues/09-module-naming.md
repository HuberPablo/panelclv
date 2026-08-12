# Settle module and subpackage naming

Type: grilling
Status: open
Blocked by: 06

## Question

Supersedes `.scratch/benchmark-refactor/issues/10-rename-utils-modules.md`, which was
left `ready-for-agent`. Renaming is the last step, not the first: if the redesign merges
or deletes a module, its new name is a moot question — so that ticket is closed as
superseded and its content decided here, with the target shape known.

Carried over from it:

- `_utils` is a null suffix — every module is utilities. Proposed there:
  `evaluation_utils` → `metrics`, `plot_utils` → `plots`, `experiment_utils` →
  `orchestration`, `training_utils` → `training_loop`.
- `pnbd_grid.py` and `pareto_simulation.py` use two abbreviations for one concept.
- `dynamic_panel_dataset.py` — "dynamic" does no work.

Added by this map:

- **`experiments` vs `studies`** — near-synonyms in English, and only one of them is in
  `CONTEXT.md`. If 06 keeps both subpackages, they need names that say which is which.

Every rename updates the four live notebooks in the same commit (map Notes), so include
the notebook cost in the decision.
