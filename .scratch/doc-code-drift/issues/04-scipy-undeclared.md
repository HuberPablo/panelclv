# 04 — `scipy` is a hard import and an undeclared dependency

**Status:** ready-for-agent

## Doc claim

`pyproject.toml:36-45` declares the runtime dependencies, with a comment explaining the
choice:

> ```
> # numpy/pandas/scikit-learn/optuna/matplotlib are all
> # required at import or run time by the model and data-prep modules.
> dependencies = [
>     "torch", "numpy", "pandas", "scikit-learn", "optuna", "matplotlib",
> ]
> ```

`pyproject.toml:48-49` mentions SciPy but does not list it:

> The HB-MCMC Pareto/NBD port runs on pure NumPy/SciPy; R is only needed to *re-validate*
> it (see `scripts/`), never at run time.

`docs/adr/0004-frozen-reference-implementations.md:67-68` goes further and drops SciPy
entirely:

> **Pareto/NBD means the hierarchical-Bayes MCMC estimator** — **a pure NumPy port** of R's
> BTYDplus, the one Valendin et al. actually fit.

## Code reality

SciPy is a top-level import in two modules, so it is needed at import time, not lazily:

- `src/panelclv/benchmarks/pareto_nbd.py:66` — `from scipy.special import gammaln`
- `src/panelclv/studies/suite_metrics.py:28` — `from scipy import stats`

`scipy` appears nowhere in `pyproject.toml`'s `dependencies`. It resolves today only because
`scikit-learn` pulls it in transitively — an implementation detail of another package, not a
guarantee.

The same file also imports pandas (`src/panelclv/benchmarks/pareto_nbd.py:65`), so "pure
NumPy" is inaccurate on two counts.

## Fix

1. Add `"scipy"` to `pyproject.toml`'s `dependencies`, and extend the comment above it to
   name what needs it (the Pareto/NBD log-likelihood's `gammaln`, and the suite's Student-t
   intervals).
2. `docs/adr/0004-frozen-reference-implementations.md:67` — "a pure NumPy port" → a port
   that needs no R at run time. The point that sentence is making is *"no R dependency"*,
   not *"no SciPy"*, so the fix is to say the thing it means.

Low risk: nothing changes at run time in an environment that already works. It matters for a
clean install and for the VastAI provisioning path, which installs the package on a fresh
machine.
