# Retire `evaluation_utils.compute_metrics`

Status: done

`compute_forecast_metrics` is the single scoring authority. `evaluation_utils.compute_metrics`
and its keys (`mae`, `mape_positive`, `cumulative_mape`, `aggregate_bias_fraction`) are
back-compat left over from before that consolidation, and `plot_utils.py:34` imports
`compute_metrics` without ever calling it.

Four files, no production caller: `evaluation_utils.py`, `evaluation/__init__.py`,
`plot_utils.py` (dead import), `tests/test_imports.py`.

Keep `rmse` and `mape_positive` only if something still wants them after the removal.

Done when: one scoring path remains and `pytest -q` passes.

## Comments
Done in `fa1eccb`. `compute_metrics`, `mae`, `mape_positive`, `cumulative_mape` and
`aggregate_bias_fraction` had no caller anywhere, and `plot_utils` imported
`compute_metrics` without using it. All removed; `evaluation_utils.py` is deleted.

"Keep `rmse` and `mape_positive` only if something still wants them": nothing did.
`rmse` was wanted only by the test that existed to test it, and `compute_forecast_metrics`
computes its own. `aggregate_bias` was the one function with a live caller
(`segment_analysis`), so it moved next to that caller — it is the number
`compute_forecast_metrics` does not return, raw-count bias, which the group table needs
because percentage bias is uninformative for a group whose actual total is near zero.

The two deleted tests are replaced by tests on the surviving authority, which had no
direct coverage: its three definitions are pinned against a hand-computed (N, T_HOLD)
example, plus a guard that the retired names stay retired.
