# Fix the broken `panelclv.benchmarks` import

Status: done

`pareto_nbd.py` moved to `benchmarks/archive/`, but `benchmarks/__init__.py:13` still
imports it and `archive/` has no `__init__.py`. `import panelclv.benchmarks` raises
`ModuleNotFoundError`, so every caller of the subpackage fails at import.
`evaluation/plot_utils.py:291` imports the moved module too.

Ticket 03 removes the MLE model entirely — this ticket only restores a working import
so the package is usable in the meantime.

Done when: `import panelclv.benchmarks` succeeds and `pytest -q` passes.

## Comments
Done in `e3e290c`. `benchmarks/archive/` got an `__init__.py` and both import sites
(`benchmarks/__init__.py`, `evaluation/plot_utils.py`) were pointed at
`benchmarks.archive.pareto_nbd`. `pytest -q`: 77 passed.
