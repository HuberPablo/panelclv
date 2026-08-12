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

### Why the archived results say `ParetoNBD_MLE`

They did not always. Every one of them was written as plain `ParetoNBD`, back when
that name meant this estimator. When the hierarchical-Bayes port took the name over,
those folders became mislabeled — same name on disk, different model behind it — so
they were relabeled in place to record the estimator rather than the family. The
relabel covered every place the name is written: the model folder itself, the
`aggregated_<model>.csv` filename beside it, the `name` field of both the model-level
and suite-level `config.json`, and the `model` column of `metrics.csv` and
`results.csv`. Labels only — the rows are otherwise byte-identical to what the MLE
fitter produced.

This is why `scripts/make_grid_figures.py` and `scripts/recheck_season_churn.py` can
list `ParetoNBD_MLE` as one of three model directories and find it. It is also why a
bare `ParetoNBD` folder under `Studies/` is safe to read as hierarchical-Bayes: no
MLE result carries that label any more.
