# Relabel archived study results that used the MLE Pareto/NBD

Status: done
Blocked by: 03

`Studies/cross_entropy_cfg_2y_Train_1yPred_NoCov_V1_10Studies_100_simulations/` holds
a `ParetoNBD/` folder whose predictions came from the MLE estimator. Once `"Pareto/NBD"`
means the hierarchical-Bayes model, that stored result is mislabeled — same name,
different model, and nothing on disk records which one produced it.

Rename the folder and its `results.csv` rows to `ParetoNBD_MLE`. Check whether any
other suite under `Studies/` has the same problem before finishing.

This is thesis evidence that may need defending later, so the distinction belongs on
disk rather than in memory.

Done when: every archived Pareto result names the estimator that produced it, and
`studies.analysis` still loads those suites.

## Comments
Done in `a22a304`. The named suite was not the only one: **443** suites under `Studies/`
carry a `ParetoNBD` result and every one is MLE (the runner has only ever called that
estimator; all 323 with a `config.json` assert `model_type == "pareto_nbd"`). Renamed all
of them — model folder, its `config.json` name and `metrics.csv` rows, and the suite's
`results.csv` rows and model-spec list.

The migration is checked in at `scripts/migrations/relabel_archived_pareto_mle.py`
(idempotent, dry-runs by default) as the record of what changed, since `Studies/` is
gitignored and the rename therefore leaves no git history.

`scripts/make_grid_figures.py` and `scripts/recheck_season_churn.py` are hard-wired to a
specific archived grid, so they follow the rename. **Worth a look:** the grid figures now
label that series `"Pareto/NBD (MLE)"` — a visible change to thesis figures, made on the
ticket's own logic that a reader must be able to tell the two estimators apart. One string
in `make_grid_figures.py` if you want it back.

The live names stay put (`ModelSpec(name="ParetoNBD")`, `plot_diff_grid`'s `model_b`
default): new suites are the hierarchical-Bayes model. Verified `studies.analysis`
discovery/loading and `collect_grid_results` on the relabeled suites, and re-ran both
grid scripts end to end.
