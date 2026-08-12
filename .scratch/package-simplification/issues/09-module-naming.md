# Settle module and subpackage naming

Type: grilling
Status: resolved
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

## Answer

Settled with Pablo over two rounds of `/grilling`, 2026-08-12. **Thirteen decisions, nothing
left open.** Every cost figure below was measured against the four live notebooks, `src/`,
`tests/` and `scripts/` — not assumed from the ticket header, which was wrong about this
(see *Corrections* at the foot).

### The rule this ticket applies

A name is fixed by renaming **unless the name alone cannot tell the truth** — where one name
covers two concepts, or one concept has two exported names, the shape changes too. Three items
fall on that side (decisions 7, 8, 5); everything else is a pure rename.

### Decisions

1. **Scope.** Reshape for `data_info`, `make_loaders` and the `mc_*` pairs; pure rename
   everywhere else. **`evaluation/plot_utils.py`'s name is deferred to ticket 12**, which owns
   `evaluation/`'s internal split and cannot name a file whose contents are still moving
   (decision 6 of ticket 06 removes `save_predictions_to_csv` from it first). Ticket 12
   inherits two constraints: the no-`_utils` rule, and the finding that **two notebooks import
   a private symbol from it** — `from panelclv.evaluation.plot_utils import _pareto_from_data`
   (`Pareto_Datasets`, `Data_integration_TRANSFORMER_v2`) — which no rename legitimises.

2. **`experiments/` → `trials/`.** `CONTEXT.md` states there is deliberately no term for
   "experiment", so the subpackage name had no referent in the ubiquitous language. `trials/`
   is the one candidate the vocabulary already defines, and it makes the altitude ladder
   legible: `trials/` assembles and refits **one** trial, `tuning/` searches over trials,
   `studies/` runs many studies. Ticket 06's ruling that the subpackage survives is untouched —
   this is a rename, not a re-partition. Cost: 4 notebook import lines.
   Rejected: `orchestration/` (the superseded ticket's proposal — says nothing); re-adding
   "experiment" to `CONTEXT.md` (re-opens a decision ticket 08 took two commits ago).

3. **Five module file renames.** Only two are visible to notebooks:

   | current | new | cost |
   |---|---|---|
   | `training/training_utils.py` | `training/loop.py` | internal |
   | `data_preparation/dynamic_panel_dataset.py` | `data_preparation/panel_dataset.py` | **6 notebook sites** |
   | `data_preparation/pareto_simulation.py` | `data_preparation/pareto_nbd_simulation.py` | **5 notebook sites** (the `as ps` alias absorbs most) |
   | `studies/pnbd_grid.py` | `studies/pareto_nbd_grid.py` | internal |
   | `benchmarks/pareto_benchmark.py` | `benchmarks/pareto_nbd.py` | internal |

   The Pareto trio is the one that can mislead: the model-type string is `pareto_nbd` and
   `CONTEXT.md` says Pareto/NBD, so three files spelling one concept three ways is exactly the
   cross-lane collision D25 recorded. `training.loop` reads correctly at the import site;
   `training_loop.py` would stutter. "Dynamic" in `dynamic_panel_dataset` does no work.

4. **`Inference*` → `Rollout*`, prefix position.** `RolloutMultinomialLSTMModel`,
   `RolloutMultinomialTransformerModel`, `RolloutValendinLSTMModel`. `CONTEXT.md` lists
   *inference* under `_Avoid_` for **Rollout** and now defines **Rollout model**; the prefix form
   is a one-word substitution at every site and keeps the pairing with `MultinomialLSTMModel`
   visually adjacent. Suffix (`MultinomialLSTMRollout`) rejected — it drops "Model" and stops
   matching the `CONTEXT.md` noun.
   **Cost is far below the ticket's assumption:** `Inference*` occurs **4 times total** across the
   notebooks, all bare names in import lists, **zero call sites** — and ticket 07 already deletes
   the cells that constructed them. Land this in ticket 07's commit so the notebooks are touched
   once. Also renames `_build_inference_model_for`-adjacent wording in docstrings; the function
   itself dies with ticket 07.

5. **Every `mc_*` alias is deleted, and the two rollout functions are named by mechanism.**
   `models/__init__.py` currently exports one function under two public names, twice over.
   Survivors:

   | current | new |
   |---|---|
   | `run_monte_carlo_forecast` / `mc_forecast` | `forecast_recurrent` |
   | `run_monte_carlo_forecast_transformer` / `mc_forecast_transformer` | `forecast_attention` |
   | `simulate_one_path` / `mc_simulate_one_path` | `simulate_recurrent_path` |
   | `simulate_transformer_path` / `mc_simulate_transformer_path` | `simulate_attention_path` |
   | `compute_forecast_metrics` / `mc_compute_metrics` | `compute_forecast_metrics` |

   **Why mechanism, not model family** (`forecast_lstm` / `forecast_transformer`, which Pablo
   raised): verified that there are **three** rollout model classes but only **two** rollout
   functions — `RolloutMultinomialLSTMModel` and `RolloutValendinLSTMModel` share one. So
   `forecast_lstm` would name a path used by two different models and would be false the day a
   recurrent non-LSTM lands. The tiebreak is ticket 06 decision 5: the registry declares which
   function each model uses, so the function name's job is to say what it does to the model,
   not who calls it. The old pair also had the asymmetry that one was named for its behaviour
   and one for its caller.
   `compute_forecast_metrics` keeps the name `CLAUDE.md` already uses for the scoring authority.
   Cost: 4 notebooks for `mc_forecast`, 4 for `mc_compute_metrics`, 3 for `mc_forecast_transformer`.

6. **`mape_aggregate_style` → `mape_aggregate`.** "Style" is filler, and `CLAUDE.md` **already**
   calls the key `mape_aggregate` — the code is what drifted. Ticket 06 established the metric is
   algebraically identical to the paper's equation 5, so the name is chosen against a known
   definition. Renaming makes an already-written charter line true rather than creating new work.
   `mape_tracking` (from `CONTEXT.md`'s **Tracking** entry) was considered and rejected: it would
   require editing `CLAUDE.md` to fix a drift that the cheaper name closes. The key is a
   `results.csv` column, free to change since floor item 3 was rescinded. Cost: **1** notebook.

7. **`data_info` splits into two `ModelSpec` fields: `search_space` and `training`.** The dict
   carries Optuna search-space overrides *and* training controls (`n_epochs`, `patience`,
   `checkpoint_dir`, `verbose`, `loss_type`, `class_weights`, `focal_gamma`, `grad_clip`,
   `log_wandb`, `seed`), and `validate_data_info` already polices it against
   `_SEARCH_DEFAULTS | _NON_SEARCH_DATA_INFO_KEYS` — the code knows they are two sets, so make
   that the interface instead of a runtime check. A typo then lands in the wrong *field* rather
   than being caught by a hand-maintained allowlist. `validate_data_info` shrinks accordingly.
   Ticket 06 decision 4 moves the search-space *defaults* into the registry, so what remains on
   `ModelSpec` is overrides plus knobs — this decision splits what is left.
   Cost: 4 notebooks, `studies/config.py`, `studies/runner.py` (3 sites), `tuning/optuna_tuning.py`.
   The archived `config.json` records a `data_info` key; floor item 3 is rescinded, so that is free.

8. **`make_loaders` → `split_calibration`, returning a named `CalibrationSplit`;
   `make_refit_loader` → `refit_loader`; both stay put.**

   *Placement, ruled for the first time:* they stay in the subpackage (now `trials/`).
   `data_preparation/` is disqualified — it is deliberately torch-free and this is precisely
   where numpy becomes tensors; `training/` inverts the import direction
   (`tuning → trials → training`). Audit 04 ruled the layering sound.

   *Name:* "split" names the decision the function actually makes. It is the **sole enforcement
   point of ADR-0001** — nothing else in the package decides that training truncates at
   `val_start_idx - 1` while validation keeps the full sequence and scores from `val_score_start`.

   *Return shape:* `CalibrationSplit(train_loader, val_loader, recipe)`. The third element was
   `metadata`, and it is load-bearing — it is what the embedder is constructed from (`seq_cols`,
   `embedded_cols`, `target_col`) plus `seq_len` and `val_score_start`. A dataclass makes
   `recipe` a first-class thing instead of a trailing dict; renaming without fixing the return
   shape would only relabel the problem.

   *And the docstring lie is fixed in the same commit:* `experiments/__init__.py` says the
   subpackage holds "no modeling logic". It holds one, and pretending otherwise is how the
   off-by-one stayed invisible. The new `trials/__init__.py` says so plainly.

   Cost: **no notebook calls or imports either function.** Real sites are `experiment_utils.py`,
   `tests/test_golden_end_to_end.py`, `scripts/trace_golden_reachability.py` and the
   `__init__.py` export. Two notebooks mention `make_loaders` in **code comments only**
   (`Data_integration_LSTM_v2` cell 8, `Data_integration_TRANSFORMER_v2` cell 6) — update them
   for accuracy, but nothing breaks if missed.

9. **The three D25 collisions.** Words meaning several things, not bad words:

   - **`study` ×3.** `StudySuiteConfig.study_name` → **`suite_name`**; `pnbd_grid`'s "study"
     (a folder of generated datasets) → **`dataset_dir`**. `layout.study_dir` keeps the name —
     it really is an Optuna study. The suite one is the misleading case, because it collides
     with a term `CONTEXT.md` defines as something else.
   - **`group` ×2.** `pnbd_grid`'s `(rate, churn)` grid point → **`cell`**, leaving `group` to
     mean customer segment only (`assign_customer_groups`, `group_metrics_table`).
   - **head size ×3.** **`max_trans` is killed** — it is the name that means neither of the other
     two. `num_target_classes` keeps the head size; `clip_target_upper` keeps the config knob
     that sets it. These are genuinely two concepts and rightly keep two names.
     Cost: `max_trans` occurs **13 times across the notebooks** and 22 in `src/` — the single
     most expensive rename in this ticket, and the only one whose cost is worth restating to
     whoever executes it.

10. **Rollout-function naming resolved by mechanism** — folded into decision 5 above; recorded
    as its own decision because it was the one item Pablo pushed back on and it was re-decided
    on new evidence (three classes, two functions), not on the original argument.

11. **`experiment_utils.py` splits into `trials/loaders.py` and `trials/refit.py`.**
    `loaders.py` holds `split_calibration`, `refit_loader` and `CalibrationSplit`;
    `refit.py` holds `refit_best_trial` and `make_data_builder`. `split_calibration` is the
    ADR-0001 enforcement point and deserves a file whose name says where the split lives —
    burying it in a 311-line catch-all is how it stayed invisible. `refit_best_trial` is a
    different altitude (rebuild a winner, reload a checkpoint) and ADR-0008 now makes *refit* a
    named concept. Single-file `trials/assembly.py` rejected. No notebook imports the module path.

    **Accepted, recorded once so no later ticket re-opens it:** `panelclv.trials` sits beside
    Optuna's `trial` object, a local variable throughout `tuning/`. No shadowing occurs
    (`from panelclv.trials import ...` never binds the bare name), and the two *are* the same
    concept — `CONTEXT.md`'s **Trial** is exactly what Optuna's object represents. The visual
    adjacency in `tuning/optuna_tuning.py` is the price.

12. **`CONTEXT.md` gains *Recipe* and *Customer group*; not *Cell*.** Both new terms are
    load-bearing across subpackages, and their absence is what let `metadata` and `group` drift
    in the first place. *Cell* is local to one module and needs no shared definition.
    **Delivery: append as an amendment to ticket 08's file**, not a reopening of its status, so
    all ADR/vocabulary work stays in one place. Draft entries:

    > **Recipe**:
    > The record of how a model must be rebuilt to match a trained one: the feature-axis column
    > names, the embedding cardinalities and the target column. Produced by the calibration
    > split, consumed by every model constructor.
    > _Avoid_: metadata, config, spec

    > **Customer group**:
    > A partition of the cohort by observed behaviour, used to report where a model's error
    > concentrates rather than to fit anything.
    > _Avoid_: segment, cluster, bucket

13. **Delivery: hybrid, weighted toward folding. +1 net issue, ~13 total.** Renames fold into
    the execution issue already touching each module, so nothing rebases against a moving file:

    | rename | host issue |
    |---|---|
    | `Rollout*` (decision 4) | ticket 07's issue — already editing those notebook cells |
    | `mape_aggregate`, `mc_*` deletion, `forecast_*` (5, 6) | the ticket 06 decision-1 / decision-6 issues touching `monte_carlo_forecasting.py` |
    | `data_info` split (7) | the registry issue — decision 4 already rewrites `ModelSpec`'s search-space half |
    | `split_calibration`, `trials/` layout (2, 8, 11) | the issue that creates `trials/` |
    | `CONTEXT.md` amendment (12) | appended to ticket 08 |

    **Orphan set, one issue:** the five file renames of decision 3, `suite_name`, `dataset_dir`,
    `cell`, and killing `max_trans`. Lands **after** the structural work. Budget: ~13 execution
    issues, inside ticket 06's ~15 tripwire.

### Corrections to this ticket's own text

- **The header's cost assumption is wrong as stated.** "Every rename updates the four live
  notebooks in the same commit" holds for some symbols and not others, and the spread is two
  orders of magnitude: `max_trans` costs 13 notebook occurrences, `Inference*` costs 4 bare
  import names with no call sites, `mape_aggregate_style` costs 1, and `make_loaders` costs
  **zero**. Notebook cost is measured per symbol, never assumed.
- **Notebooks import subpackages, not modules** — with four exceptions, which is why only two of
  the five file renames in decision 3 cost anything: `data_preparation.dynamic_panel_dataset`
  (6 sites), `data_preparation.pareto_simulation` (5), `configs.panel_config` (7, untouched
  here) and `evaluation.plot_utils._pareto_from_data` (2, handed to ticket 12).
- **`evaluation_utils` → `metrics` is moot.** One of the six items carried over from the
  superseded ticket names a module that no longer exists — it was retired, and
  `tests/test_imports.py:103` asserts it stays gone. Five carried-over items, not six.
- **The registry's `CONTEXT.md` term is already done.** Ticket 08 added **Registry** before this
  ticket opened; the closing note asking for it is satisfied.
