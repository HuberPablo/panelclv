# Rename the `*_utils` modules

Status: wontfix

`_utils` is a null suffix — every module is utilities, so the name says nothing about
what is inside. Proposed: `evaluation_utils` to `metrics`, `plot_utils` to `plots`,
`experiment_utils` to `orchestration`, `training_utils` to `training_loop`.

Related naming to settle in the same pass:

- `pnbd_grid.py` and `pareto_simulation.py` use two abbreviations for one concept
- `dynamic_panel_dataset.py` — "dynamic" does no work
- `mape_aggregate_style` — "style" is vague
- `data_info` names a dict of training knobs
- `mc_forecast` and `run_monte_carlo_forecast` are two exported names for one function

Held back from the benchmark work deliberately: this is pure renaming with a wide blast
radius into seven notebooks, which are JSON and not covered by tests. It deserves its
own verification rather than riding on a change that needs to work today.

Done when: `pytest -q` passes and every notebook still runs.

## Comments

**Superseded by the package-simplification map.** Renaming is the last step of a
refactor, not the first: the map licences open redesign, so a module renamed now is a
module that gets moved again once the target shape exists. This ticket's content —
the `_utils` renames, the `pnbd_grid`/`pareto_simulation` abbreviation clash and
`dynamic_panel_dataset`'s meaningless "dynamic" — is carried verbatim into
`.scratch/package-simplification/issues/09-module-naming.md`, which decides it with
the target architecture known, and adds the `experiments` vs `studies` naming problem
this ticket did not cover.
