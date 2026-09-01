# 20 — README and `running-a-model` accuracy batch

**Status:** ready-for-agent

Three small factual errors in the two documents a new reader hits first. Grouped because they
are one editing pass.

---

## 20a — "there are no `sys.path` hacks", contradicted seven paragraphs later

`README.md:17-18`:

> The project uses a **src-layout** (the package lives in `src/panelclv/`), so installing it
> is what puts `panelclv` on the path — **there are no `sys.path` hacks**.

All four live notebooks open with one. Cell 0 of `Data_integration_LSTM_v2.ipynb`,
`Data_integration_TRANSFORMER_v2.ipynb`, `Study.ipynb` and `Pareto_Datasets.ipynb` each does
`sys.path.insert(0, str(_src))` plus an `os.chdir(_root)`.

The README describes this itself at `:94-95`:

> Each opens with a small bootstrap cell that locates the repo root and makes `panelclv`
> importable, **so they run whether or not the package is pip-installed.**

**Fix:** narrow the claim to what is true and is the actual point — the *package* contains no
`sys.path` manipulation, so a packaging bug surfaces at import rather than being papered over.
Then say the notebooks bootstrap deliberately, and cross-reference `:94-95`.

(`pyproject.toml:60-64` makes the same claim, correctly scoped: "the repo root is never on
`sys.path` by accident — you must install the package to import it".)

---

## 20b — The README announces one benchmark; there are two

`README.md:3-4`:

> Modular **LSTM** and **Transformer** models for customer-base transaction-count forecasting,
> **with a Pareto/NBD benchmark.**

`valendin_lstm` is never mentioned anywhere in the README, but it is a full registry entry
that trains, tunes, refits and rolls out
(`src/panelclv/registry/model_registry.py:369-378`), and `CONTEXT.md:76-77` states plainly:

> **The two benchmarks** are the Valendin et al. LSTM and Pareto/NBD.

Related, same paragraph of the README: `:85-86` — "Swap `model_type="lstm"` for
`"transformer"` to run the other family" — presents the model set as two families where
`MODEL_TYPES` has four keys.

**Fix:** name both benchmarks in the opening sentence, and extend `:85-86` to mention that
`"valendin_lstm"` and `"pareto_nbd"` are the other two registered types (the second taking a
different, non-Optuna path in the suite).

---

## 20c — `running-a-model.md` §5.1 omits `emd_weight` from the `training` keys

`docs/running-a-model.md:513-517`:

> `training` holds `n_epochs`, `patience`, `checkpoint_dir`, `verbose`, `loss_type`,
> `class_weights`, `focal_gamma`, `grad_clip`, `log_wandb` and `seed`

`TRAINING_CONTROLS` (`src/panelclv/tuning/optuna_tuning.py:81-86`) is that set **plus
`emd_weight`**, and `_validate_training` (`:89-97`) rejects anything outside it — so the doc's
list reads as exhaustive and is one short. `run_optuna_study`'s own docstring
(`optuna_tuning.py:411`) lists it correctly.

**Fix:** add `emd_weight` to the list. Worth pointing at
`scripts/run_loss_ablation.py:143` as the worked example, since a reader who wants to search λ
will otherwise put it in `search_space` and hit the error in issue `05`.

---

## Verified alongside

Everything else in the README quickstart binds against the current API — checked by importing
each name and comparing signatures: `PanelConfig(...)` kwargs, `prepare_dataset(panel, cfg)`,
`run_optuna_study(model_type=, data_builder=, training=, n_trials=)`,
`refit_best_trial(study, data_full, "lstm", batch_size=512)` returning
`(rollout_model, data_best)`, `rollout_for("lstm")(...)`,
`compute_forecast_metrics(forecast["actual"], forecast["prediction_mean"])`, the `[dev]`
extra, the four notebook filenames, `notebooks/archive/README.md`, all four named `VastAI/`
files, and `Datasets/Dataset_clean/electronics_customer_week_panel.csv`.
