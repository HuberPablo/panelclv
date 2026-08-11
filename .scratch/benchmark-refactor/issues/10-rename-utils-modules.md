# Rename the `*_utils` modules

Status: ready-for-agent

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
