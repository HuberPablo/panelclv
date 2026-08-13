"""The live notebooks call APIs that still exist.

Notebooks are JSON, so nothing else in the suite reads them: a rename in `src/`
lands in a notebook as a `TypeError` the next time someone runs a cell, months
later. These tests are the cheap standing check that this has not happened.

Only the notebooks in `notebooks/` are checked. `notebooks/archive/` holds
records of finished experiments that are deliberately frozen against an old API
(see its README), so checking them would mean either migrating them or carrying
a permanent xfail.

The tests are static — no cell is executed, nothing is trained. Two complementary
checks run over every live notebook, and neither subsumes the other:

- **Signature binding** resolves each `panelclv` callable a notebook imports and
  binds the notebook's actual call against `inspect.signature`. This is the check
  that generalises: it catches a renamed keyword, a dropped parameter or a changed
  arity without anyone having to predict the rename in advance.
- **A retired-name blacklist** covers what binding cannot see — commented-out
  lines, explanatory prose, and *stored outputs*. That last one matters in a thesis
  record: a saved traceback quoting an old signature reads to the reader as current
  API and survives every source-level search.
"""

import ast
import importlib
import inspect
import json
import re
from pathlib import Path

import pytest

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"
LIVE_NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))

# Names retired by the benchmark refactor, as (pattern, reason). Each reason says
# what to write instead, so a failure is self-explanatory. Patterns are regexes so
# `variant="paper"` is caught whichever quote style was typed.
RETIRED = [
    (r"INPUT_SPEC", "PanelConfig carries the column roles (ticket 09)"),
    (r"FEATURE_SCHEMA", "PanelConfig carries the column roles (ticket 09)"),
    (r"pareto_nbd_benchmark", "the MLE estimator is gone; use pareto_benchmark=True"),
    (r"pareto_paper_benchmark", "there is one estimator; use pareto_benchmark=True"),
    (r"""variant\s*=\s*['"]paper['"]""",
     "pareto_forecast takes no variant; there is one estimator"),
    (r"penalizer_coef", "an MLE-only knob; the HB sampler has no penalizer"),
    # Retired by the package cleanup, issue 04: rollout-composite trial selection.
    (r"rollout_composite", "trials are selected on validation loss (ADR-0003 retired)"),
    (r"selection_metric", "run_optuna_study has one selection metric, so it has no knob"),
    (r"rollout_(data|horizon|n_simulations|seed|mape_clip|min_actual|weight_)",
     "the rollout selection knobs went with rollout_composite"),
    (r"weekly_aggregate_rollout_metrics",
     "compute_forecast_metrics is the only implementation of rmse / bias / MAPE"),
    (r"mc_compute_metrics", "the authority is compute_forecast_metrics"),
    (r"mape_aggregate_style", "compute_forecast_metrics returns mape_aggregate"),
    # Retired by the package cleanup, issue 05: refit-only forecasting, and the
    # `experiments` -> `trials` rename.
    (r"REFIT_ON_FULL_CALIBRATION",
     "every forecast comes from a refit (ADR-0008), so there is no toggle"),
    (r"prediction_source",
     "StudySuiteConfig has no prediction source; the refit is the only one"),
    (r"panelclv\.experiments", "the subpackage is `panelclv.trials`"),
    (r"make_loaders", "the temporal split is `trials.split_calibration`"),
    (r"make_refit_loader", "it is `trials.refit_loader`"),
    # Retired by the package cleanup, issue 06: one registry entry per model.
    (r"data_info",
     "the search-space overrides and the training controls are two fields: "
     "`search_space` and `training`"),
    (r"VALID_MODEL_TYPES|NEURAL_MODEL_TYPES",
     "the model types are the registry table's keys, and neural is `is_neural`"),
    # Retired by the package cleanup, issue 07: a rollout model comes from its
    # trained model.
    (r"Inference(Multinomial|Valendin)\w*Model",
     "the class is `Rollout...`; CONTEXT.md lists *inference* under _Avoid_"),
    (r"build_inference_from_trial",
     "`refit_best_trial` hands back the rollout model itself (ADR-0007)"),
    # Retired by the package cleanup, issue 08: prediction I/O in its own module,
    # `evaluation/plot_utils.py` split three ways, rollouts named by mechanism.
    (r"mc_forecast(_transformer)?",
     "the rollouts are `models.forecast_recurrent` / `models.forecast_attention`"),
    (r"run_monte_carlo_forecast(_transformer)?",
     "the rollouts are `models.forecast_recurrent` / `models.forecast_attention`"),
    (r"plot_utils",
     "prediction I/O is `panelclv.predictions`; the plots are `evaluation.plots`"),
    (r"_pareto_from_data",
     "the fitter is public: `benchmarks.pareto_from_data`"),
    (r"weekly_actuals|weekly_aggregate_predictions|alignment_check",
     "import-only helpers, deleted with plot_utils"),
    # Retired by the package cleanup, issue 15: the orphan rename sweep.
    (r"dynamic_panel_dataset", "the module is `data_preparation.panel_dataset`"),
    (r"pareto_simulation",
     "the module is `data_preparation.pareto_nbd_simulation`; one spelling of Pareto/NBD"),
    (r"training_utils", "the training loop is `training.loop`"),
    (r"panelclv\.benchmarks\.pareto_benchmark",
     "the module is `benchmarks.pareto_nbd` (the `pareto_benchmark=` plot flag stays)"),
    (r"\bmax_trans\b",
     "the softmax head size is `num_target_classes`; the config knob is `clip_target_upper`"),
    (r"group_summary",
     "a `(rate, churn)` grid point is a cell: `studies.cell_summary`"),
]


def _cells(path: Path, code_only: bool = False) -> list[tuple[int, dict]]:
    """Every cell of a notebook as `(index, cell)`; markdown included by default."""
    cells = json.loads(path.read_text())["cells"]
    return [
        (i, c)
        for i, c in enumerate(cells)
        if not code_only or c["cell_type"] == "code"
    ]


def _output_text(cell: dict) -> str:
    """Everything a cell's stored outputs would show a reader, as one string.

    Covers the four shapes nbformat uses: `stream` text, rich `data` payloads,
    and an error's `evalue` plus its `traceback` — the traceback being the one
    that quotes function signatures back at you.
    """
    chunks: list[str] = []
    for out in cell.get("outputs", []):
        chunks.append("".join(out.get("text", [])))
        chunks.append("\n".join(out.get("traceback", [])))
        chunks.append(str(out.get("evalue", "")))
        chunks.append("".join(out.get("data", {}).get("text/plain", [])))
    return "\n".join(chunks)


def test_live_notebooks_are_discovered():
    """A glob that silently matches nothing would make every test below vacuous."""
    assert LIVE_NOTEBOOKS, f"no notebooks found under {NOTEBOOK_DIR}"


@pytest.mark.parametrize("path", LIVE_NOTEBOOKS, ids=lambda p: p.name)
def test_no_retired_api_names_in_source(path):
    """No live notebook's cell source mentions a name the package has retired.

    Comments count. A commented-out `#pareto_nbd_benchmark=True` is an
    instruction to the reader that raises the moment anyone uncomments it.
    """
    offences = [
        f"cell {i}: {pattern!r} — {reason}"
        for i, cell in _cells(path)
        for pattern, reason in RETIRED
        if re.search(pattern, "".join(cell["source"]))
    ]
    assert not offences, "\n".join([f"{path.name} uses retired API:", *offences])


@pytest.mark.parametrize("path", LIVE_NOTEBOOKS, ids=lambda p: p.name)
def test_no_retired_api_names_in_stored_output(path):
    """No live notebook *displays* a retired name in a saved output.

    Clearing the offending cell's output is the fix — a stored result its own
    migrated source can no longer reproduce is worse than no result at all.
    """
    offences = [
        f"cell {i}: {pattern!r} — {reason}"
        for i, cell in _cells(path)
        for pattern, reason in RETIRED
        if re.search(pattern, _output_text(cell))
    ]
    assert not offences, "\n".join([f"{path.name} displays retired API:", *offences])


@pytest.mark.parametrize("path", LIVE_NOTEBOOKS, ids=lambda p: p.name)
def test_panelclv_imports_resolve(path):
    """Every `from panelclv... import X` in the notebook resolves today.

    Cells that do not parse are skipped rather than failed: notebooks legitimately
    contain IPython magics (`%pip`, `!ls`) that are not Python.
    """
    missing = []
    for index, cell in _cells(path, code_only=True):
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not node.module or not node.module.startswith("panelclv"):
                continue
            try:
                module = importlib.import_module(node.module)
            except ImportError:
                # A retired *module*, not just a retired name — report it in the
                # same list rather than aborting the whole notebook's check.
                missing.append(f"cell {index}: module {node.module}")
                continue
            for alias in node.names:
                if hasattr(module, alias.name):
                    continue
                # `from panelclv.x import y` also resolves if y is a submodule.
                try:
                    importlib.import_module(f"{node.module}.{alias.name}")
                except ImportError:
                    missing.append(f"cell {index}: {node.module}.{alias.name}")
    assert not missing, "\n".join([f"{path.name} imports missing names:", *missing])


def _imported_panelclv_names(path: Path) -> dict[str, object]:
    """Notebook-local name -> the `panelclv` object it refers to.

    Only names the notebook imports directly (`from panelclv.x import f`, and
    submodules imported the same way) are resolved, because those are the ones
    whose calls can be checked against a real signature.

    A name the notebook later assigns to is dropped: `forecast = mc_forecast(...)`
    rebinds it to a dict, and calls through it are no longer calls to the package.
    Without this the check would report failures that are not real.
    """
    imported: dict[str, object] = {}
    assigned: set[str] = set()
    for _, cell in _cells(path, code_only=True):
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                assigned.add(node.id)
            if not isinstance(node, ast.ImportFrom):
                continue
            if not node.module or not node.module.startswith("panelclv"):
                continue
            try:
                module = importlib.import_module(node.module)
            except ImportError:
                continue
            for alias in node.names:
                obj = getattr(module, alias.name, None)
                if obj is None:
                    try:
                        obj = importlib.import_module(f"{node.module}.{alias.name}")
                    except ImportError:
                        continue
                imported[alias.asname or alias.name] = obj
    return {k: v for k, v in imported.items() if k not in assigned}


# Resolvers whose whole job is to return another callable from a literal key, and
# which a notebook therefore calls in the head of another call:
# `rollout_for("lstm")(model, data, ...)`. Without this, the outer call is invisible
# to the binding check below — its `func` is a Call node, not a Name — so migrating
# the notebooks onto the registry (ADR-0006) would have silently dropped the
# forecast's arguments out of coverage. Each entry must be a pure table lookup: the
# check evaluates it, so anything with a side effect does not belong here.
CALLABLE_RESOLVERS = {"rollout_for"}


def _called_panelclv_object(node: ast.Call, env: dict[str, object]):
    """The package callable a `Call` node targets, or None if it targets nothing we know.

    Handles the three shapes notebooks use: a bare `f(...)` for a name imported
    directly, `mod.f(...)` where `mod` is an imported `panelclv` submodule, and
    `resolver("literal")(...)` for the registry lookups in CALLABLE_RESOLVERS.
    """
    func = node.func
    if isinstance(func, ast.Name):
        target = env.get(func.id)
    elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        module = env.get(func.value.id)
        target = getattr(module, func.attr, None) if inspect.ismodule(module) else None
    elif isinstance(func, ast.Call) and isinstance(func.func, ast.Name):
        target = _resolved_callable(func, env)
    else:
        target = None
    return target if callable(target) else None


def _resolved_callable(node: ast.Call, env: dict[str, object]):
    """Evaluate a `resolver("literal")` head to the callable it returns, or None.

    Deliberately narrow: the resolver must be one this file names, every argument
    must be a literal, and a lookup that raises (an unknown model type) resolves to
    None rather than failing here — the resolver's own call is bound separately, and
    an unknown key is that check's business, not this one's.
    """
    if not isinstance(node.func, ast.Name) or node.func.id not in CALLABLE_RESOLVERS:
        return None
    resolver = env.get(node.func.id)
    if not callable(resolver):
        return None
    try:
        args = [ast.literal_eval(a) for a in node.args]
        kwargs = {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg}
    except ValueError:
        return None
    try:
        return resolver(*args, **kwargs)
    except Exception:
        return None


@pytest.mark.parametrize("path", LIVE_NOTEBOOKS, ids=lambda p: p.name)
def test_panelclv_calls_bind_to_current_signatures(path):
    """Every call a notebook makes into `panelclv` still binds.

    `inspect.Signature.bind` is given a placeholder per argument — nothing is
    called and no value is evaluated, so this stays a static check. It reports
    exactly the errors the notebook would raise on its next run: unexpected
    keyword, missing required argument, too many positional arguments.

    Calls using `*args`/`**kwargs` unpacking are skipped: their argument list is
    not knowable without running the cell, so binding them would be guesswork.
    """
    env = _imported_panelclv_names(path)
    failures = []
    for index, cell in _cells(path, code_only=True):
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            target = _called_panelclv_object(node, env)
            if target is None:
                continue
            if any(isinstance(a, ast.Starred) for a in node.args):
                continue
            if any(k.arg is None for k in node.keywords):
                continue
            try:
                signature = inspect.signature(target)
            except (ValueError, TypeError):
                continue  # C-level or otherwise unintrospectable
            placeholders = [object()] * len(node.args)
            keywords = {k.arg: object() for k in node.keywords}
            try:
                signature.bind(*placeholders, **keywords)
            except TypeError as exc:
                name = getattr(target, "__name__", repr(target))
                failures.append(f"cell {index}: {name}(...) — {exc}")
    assert not failures, "\n".join([f"{path.name} calls a dead signature:", *failures])
