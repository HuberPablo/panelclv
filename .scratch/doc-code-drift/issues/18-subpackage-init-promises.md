# 18 — Subpackage `__init__` files promise more than they deliver

**Status:** ready-for-agent

Five separate gaps between what a subpackage's docstring says it offers and what it actually
exposes. Grouped because the fix is one pass over eleven small files.

## The governing claims

`CLAUDE.md:43-44`:

> A name lives in exactly one of them and there is no umbrella re-export — **import from the
> subpackage that owns it.**

`CLAUDE.md:66-67`:

> **Each subpackage's `__init__.py` documents its own contents.** Read those rather than
> expecting this file to list modules.

---

## 18a — `configs` and `data_preparation` export nothing

Both are docstring-only: no imports, no `__all__`. So the rule at `CLAUDE.md:43-44` cannot be
followed for the two most-used entry points in the package.

```console
$ PYTHONPATH=src ~/Desktop/Thesis/venvs/thesis_rocm/bin/python -c "
import panelclv.configs as C, panelclv.data_preparation as D
print('configs.PanelConfig:', hasattr(C, 'PanelConfig'))
print('data_preparation.prepare_dataset:', hasattr(D, 'prepare_dataset'))"
configs.PanelConfig: False
data_preparation.prepare_dataset: False
```

`README.md:42-43` works around it, reaching past the subpackage to the module:

```python
from panelclv.configs.panel_config import PanelConfig
from panelclv.data_preparation import panel_dataset
```

Both docstrings *do* list their modules (`configs/__init__.py:4-12`,
`data_preparation/__init__.py:3-13`), which is the `CLAUDE.md:66` half — so this is only about
the re-export rule.

**Note before fixing:** `configs/__init__.py:13-15` says "Nothing in this subpackage imports
another `panelclv` subpackage, and that is the point" — re-exporting `PanelConfig` from its own
`__init__` does not violate that (it is an intra-subpackage import), but check
`tests/test_import_graph.py` still passes after.

---

## 18b — `training` and `tuning` document why they exist, not what they hold

`src/panelclv/training/__init__.py` and `src/panelclv/tuning/__init__.py` contain only a
justification of the split ("as opposed to the model definition itself", "hence its own
subpackage"). Neither names a module or a symbol, against `CLAUDE.md:66-67`.

They do export: `training` → `fit_model`, `refit_full_calibration`, `train_one_epoch`,
`validate_one_epoch`, `FitResult`; `tuning` → `run_optuna_study`, `select_features`,
`select_features_for_trial`, `validate_removable_features`. None of the eight is mentioned in
prose. Compare `configs/__init__.py` or `data_preparation/__init__.py`, which do this well.

---

## 18c — `models/__init__.py` lists loss internals as "still importable"; one is not

`src/panelclv/models/__init__.py:56-58`:

> Internals kept OFF this list but still importable: the train-time loss classes/helpers
> (`FocalLoss`, `SquaredEMDLoss`, `compute_class_weights`, `build_criterion`).

`CrossEntropyPlusEMDLoss` (`src/panelclv/models/losses.py:99`) is a fifth public loss class,
built by `build_criterion` for `loss_type="ce_emd"` (`losses.py:272-273`) — and it is *not*
importable from `panelclv.models`, unlike the four that are named:

```console
$ PYTHONPATH=src ~/Desktop/Thesis/venvs/thesis_rocm/bin/python -c "
import panelclv.models as M
print('FocalLoss:', hasattr(M, 'FocalLoss'))
print('build_criterion:', hasattr(M, 'build_criterion'))
print('CrossEntropyPlusEMDLoss:', hasattr(M, 'CrossEntropyPlusEMDLoss'))"
FocalLoss: True
build_criterion: True
CrossEntropyPlusEMDLoss: False
```

---

## 18d — `studies/__init__.py` promises completeness it does not have

`src/panelclv/studies/__init__.py:27-34`:

> **all of their entry points are exported here**, so a caller imports this subpackage, never
> a module … ``suite_metrics`` … **It owns the package's one Student-t interval.**

`t_interval_half_width` (`src/panelclv/studies/suite_metrics.py:53`) is public and is **not**
exported — verified, `hasattr(panelclv.studies, "t_interval_half_width")` → `False`. Both
in-package consumers reach past the subpackage for it
(`studies/suite_plots.py:32`, `studies/pareto_nbd_grid.py:50`). `DEFAULT_METRICS`
(`pareto_nbd_grid.py:63`) and `layout.py`'s public surface (`create_suite_root`, `model_dirs`,
`study_dir`, `prediction_path`, `jsonify`, `write_json`) are likewise unlisted.

The docstring singles out by name the one symbol the promise fails on, which is what makes it
worth fixing rather than softening.

---

## 18e — `models/__init__.py` says "each" sibling and lists 7 of 10

`src/panelclv/models/__init__.py:14-22`:

> The surrounding concerns **each** have their own sibling subpackage.

The list omits `studies`, `data_preparation` and `configs`.

---

## Fix

Per gap, smallest first: 18c and 18e are one-line edits; 18b is two docstrings; 18d is either
exporting the missing names or replacing "all of their entry points" with an accurate
qualifier; 18a is the only one with a real design question — whether `PanelConfig` and
`prepare_dataset` should be re-exported (matching every other subpackage and `CLAUDE.md:43-44`)
or whether `CLAUDE.md` should record that these two are imported by module. Re-exporting is
the smaller change and makes the rule true everywhere; it would also let `README.md:42-43`
read like the rest of the package.

## Verified alongside

The stronger half of `CLAUDE.md:43-44` holds: **no name is exported by more than one
subpackage.** Checked at runtime across all eleven public surfaces — zero collisions. And the
root `src/panelclv/__init__.py` exposes only `__version__`, so there is genuinely no umbrella
re-export.
