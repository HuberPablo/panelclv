"""Archived benchmark implementations — kept for provenance, not for use.

Nothing in the live pipeline imports from here. The modules are retained so a
result produced by an older estimator can still be traced back to the code that
produced it.

- ``pareto_nbd`` — frequentist-MLE Pareto/NBD via ``lifetimes``. Superseded by the
  hierarchical-Bayes MCMC port in ``panelclv.benchmarks.pareto_paper``, which is
  the estimator Valendin et al. actually use.
"""
