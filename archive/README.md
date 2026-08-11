# archive/

Superseded implementations, kept so a stored result can be traced back to the code
that produced it. Nothing here is imported by `panelclv` and nothing here ships in
the wheel — this is reference material, the same role `Original_paper_model/` plays.

## `pareto_nbd.py`

The frequentist-MLE Pareto/NBD (`lifetimes.ParetoNBDFitter`) that used to be the
package's Pareto benchmark. It was retired in favour of the hierarchical-Bayes MCMC
port in `panelclv.benchmarks.pareto_benchmark`, which is the estimator Valendin et al.
actually use, so the two are no longer a meaningful pair to compare.

Study results produced by this module are the `ParetoNBD_MLE/` folders under
`Studies/`; every other `ParetoNBD` result comes from the hierarchical-Bayes model.
Running it again needs `lifetimes`, which is no longer a project dependency.
