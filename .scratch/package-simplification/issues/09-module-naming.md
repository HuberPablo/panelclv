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

## Comments

**Three items from the superseded ticket were NOT carried over — recorded 2026-08-12.**
`.scratch/benchmark-refactor/issues/10-rename-utils-modules.md`'s closing comment says its
content is carried "verbatim" into this ticket. It is not: three of its six naming items are
missing from the list above and should be added to scope.

- **`mape_aggregate_style`** — "style" is vague. (Ticket 06 established the metric is
  algebraically identical to the paper's equation 5, so a name can now be chosen against a
  known definition rather than a guess.)
- **`data_info`** — names a dict that carries two different things: the Optuna search space
  *and* training controls (`n_epochs`, `checkpoint_dir`, `loss_type`, `seed`). The name
  describes neither.
- **`mc_forecast` / `run_monte_carlo_forecast`** — two exported names for one function, both
  public (`models/__init__.py:44` and `:47`, the second an alias). Same for
  `mc_compute_metrics` / `compute_forecast_metrics`.

**`make_loaders` — placement and name. Raised 2026-08-12, undecided; needs a round.**

`experiments/experiment_utils.py` → `make_loaders(data, batch_size)`. Two complaints:

1. **The name understates what it does.** It is the *sole enforcement point of ADR-0001* —
   nothing else in the package decides that training truncates at `val_start_idx - 1` while
   validation keeps the full sequence and scores from `val_score_start`. That off-by-one
   appears twice and is a modelling decision, sitting in a subpackage whose docstring says it
   "holds no modeling logic".
2. **It returns three things, and the name admits two.** `(train_loader, val_loader,
   metadata)`, where `metadata` is load-bearing — it is what the embedder is constructed from
   (`seq_cols`, `embedded_cols`, `target_col`) and it carries `val_score_start`. Renaming
   without deciding the return shape just relabels the problem; a named return type is the
   cheap fix. Its sibling is `make_refit_loader` (singular), so the pair should be settled
   together.

*Placement.* `experiments/__init__.py` names "DataLoader shaping" as its first
responsibility, so the current home is deliberate, and audit 04 ruled the layering sound.
`data_preparation/` is disqualified — it is deliberately torch-free and `make_loaders` is
where numpy becomes tensors. `training/` is the wrong direction: `tuning` imports
`experiments`, which imports `training`. Recommendation on the evidence is **stay put and fix
the name**, but it was never put to Pablo.

**Correction to this ticket's cost assumption, for this symbol only.** The header says every
rename updates the four live notebooks in the same commit. That does **not** hold for
`make_loaders`: **no notebook calls or imports it.** Two mention it in code *comments* only
(`Data_integration_LSTM_v2` cell 8, `Data_integration_TRANSFORMER_v2` cell 6). Real sites are
`experiment_utils.py` itself, `tests/test_golden_end_to_end.py`,
`scripts/trace_golden_reachability.py`, and the `experiments/__init__.py` export. The
notebook cost must be measured per symbol, not assumed.

**`Inference*` uses a word `CONTEXT.md` rejects — recorded 2026-08-12.** `CONTEXT.md`'s
**Rollout** entry lists *inference* under `_Avoid_`. The package uses it throughout:
`InferenceMultinomialLSTMModel`, `InferenceMultinomialTransformerModel`,
`InferenceValendinLSTMModel`, plus `_build_inference_model_for` and
`build_inference_from_trial`. Renaming to `Rollout*` aligns the code with the ubiquitous
language.

*Cost, measured — unlike `make_loaders`, this one is real.* **All four live notebooks import
the `Inference*` names** (`Data_integration_LSTM_v2` cell 2, `TRANSFORMER_v2` cell 2,
`Pareto_Datasets` cell 0, `Study.ipynb` cell 2), and
`tests/test_notebooks_current_api.py::test_panelclv_imports_resolve` resolves every
`from panelclv... import X`, so a missed import line turns the suite red. Plus ~8 files in
`src/`, `tests/` and `scripts/`. Coordinate with ticket 08, which owns the vocabulary side.

**The registry subpackage needs a `CONTEXT.md` term.** Ticket 06 settled the new tenth
subpackage as `registry/model_registry.py`. *Registry* is not in the ubiquitous language —
input to ticket 08.
