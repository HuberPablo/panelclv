# Fix the broken `panelclv.benchmarks` import

Status: ready-for-agent

`pareto_nbd.py` moved to `benchmarks/archive/`, but `benchmarks/__init__.py:13` still
imports it and `archive/` has no `__init__.py`. `import panelclv.benchmarks` raises
`ModuleNotFoundError`, so every caller of the subpackage fails at import.
`evaluation/plot_utils.py:291` imports the moved module too.

Ticket 03 removes the MLE model entirely — this ticket only restores a working import
so the package is usable in the meantime.

Done when: `import panelclv.benchmarks` succeeds and `pytest -q` passes.
