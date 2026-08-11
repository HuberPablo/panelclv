# Relabel archived study results that used the MLE Pareto/NBD

Status: ready-for-agent
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
