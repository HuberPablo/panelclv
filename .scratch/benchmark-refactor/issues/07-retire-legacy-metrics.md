# Retire `evaluation_utils.compute_metrics`

Status: ready-for-agent

`compute_forecast_metrics` is the single scoring authority. `evaluation_utils.compute_metrics`
and its keys (`mae`, `mape_positive`, `cumulative_mape`, `aggregate_bias_fraction`) are
back-compat left over from before that consolidation, and `plot_utils.py:34` imports
`compute_metrics` without ever calling it.

Four files, no production caller: `evaluation_utils.py`, `evaluation/__init__.py`,
`plot_utils.py` (dead import), `tests/test_imports.py`.

Keep `rmse` and `mape_positive` only if something still wants them after the removal.

Done when: one scoring path remains and `pytest -q` passes.
