# Remove the MLE Pareto/NBD and rename the surviving benchmark

Status: done
Blocked by: 02

Pareto/NBD now means one estimator: the hierarchical-Bayes MCMC port. The MLE variant
is archived, so the code carrying the two-variant distinction is dead weight and the
"paper" naming describes provenance rather than content.

- `benchmarks/pareto_paper.py` becomes `benchmarks/pareto_benchmark.py`
- `compute_pareto_paper_predictions` becomes `compute_pareto_predictions` — the name
  the MLE version vacated
- `_pareto_from_data`'s `variant` parameter goes; there is only one
- `pareto_nbd_benchmark=` and `pareto_paper_benchmark=` collapse to `pareto_benchmark=`
- the display label `"Pareto/NBD (HB)"` becomes `"Pareto/NBD"`
- `lifetimes` comes out of `pyproject.toml` dependencies — nothing else uses it
- `scripts/validate_pareto_paper.py` becomes `validate_pareto_benchmark.py`

Ticket 04 must land alongside this one: the label change collides with archived study
results where `ParetoNBD` meant the MLE model.

Done when: no reference to the MLE estimator remains in `src/` or `scripts/`, and
`pytest -q` passes.

## Comments
Done in `6357431` (landed before ticket 02 — see that ticket's note). Every bullet applied.
Two consequences beyond the listed scope, both forced by dropping `lifetimes`:

- `penalizer_coef` is MLE-only, so it comes out of `scripts/main_plot.py`,
  `scripts/run_studies.py`'s `pareto_kwargs`, and the `param_penalizer_coef` column
  `studies/runner.py` wrote into `results.csv`.
- `studies/runner.py` imports `compute_pareto_predictions` from `panelclv.benchmarks`,
  so the study runner's `pareto_nbd` baseline now fits the hierarchical-Bayes model.
  That is the point of the ticket, but it does mean new suites are not comparable to
  the archived ones — hence ticket 04.

`pytest -q`: 77 passed, plus a smoke run of the renamed benchmark on a synthetic panel.
