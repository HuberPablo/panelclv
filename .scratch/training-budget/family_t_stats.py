"""Family T, tested: each training recipe against `archive` on electronics (§9).

The per-replication scores live in `results/family_t_scores.csv` (2 models x 4 arms x 20
replications, one row per forecast, scored by `run_training_budget.score`). `--rescore`
rebuilds that file from the suites in `Studies/`; without it the file is read as is.

Every comparison is `docs/statistical-protocol.md`'s: Δ = mean(arm) − mean(archive) with
a 95% percentile-bootstrap interval from 10,000 resamples, via the package's `effect`
(through this folder's shim, which prints electronics' refit noise beside it). Each arm's
20 replications are their own searches or pinned trainings, with unseeded training, so
they share nothing with `archive`'s 20: the independent bootstrap (paired=False).

MAPE and Spearman are the primary metrics; |bias| is printed as the secondary
calibration-accuracy comparison.

    PYTHONPATH=src:scripts python .scratch/training-budget/family_t_stats.py [--rescore]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts"))
from effects import effect, table                                        # noqa: E402

SCORES = HERE / "results" / "family_t_scores.csv"
ARMS = ("archive", "paper", "paper90", "floor50")


def rescore() -> None:
    """Score every family-T forecast from `Studies/` into SCORES."""
    from panelclv.data_preparation.target_channel import holdout_actuals
    from run_training_budget import (MODELS, N_REPLICATIONS, build_data, forecast_path,
                                     score)
    rows = []
    for model in MODELS:
        data = build_data(model)
        actual, ids = holdout_actuals(data), np.asarray(data["ids"])
        for arm in ARMS:
            for r in range(N_REPLICATIONS):
                p = forecast_path(model, arm, r)
                if p.exists():
                    rows.append(dict(model=model, arm=arm, rep=r,
                                     **score(p.parents[1], actual, ids)))
    pd.DataFrame(rows).to_csv(SCORES, index=False)


if "--rescore" in sys.argv:
    rescore()
d = pd.read_csv(SCORES)
d["abs_bias"] = d.bias_percent.abs()

for model in ("ValendinLSTM", "LSTM"):
    g = d[d.model == model]
    base = g[g.arm == "archive"]
    for metric, noise_key in (("mape_aggregate", "mape_aggregate"),
                              ("spearman", "spearman"),
                              ("abs_bias", "bias_percent")):
        rows = []
        for arm in ARMS[1:]:
            # The shim looks the refit noise up under the metric name, so |bias| borrows
            # the bias entry and is relabelled after.
            e = effect(g[g.arm == arm][metric], base[metric], noise_key, "electronics")
            e.metric, e.panel = metric, f"{arm} vs archive"
            rows.append(e)
        print(f"\n### {model}: {metric} (A = archive, B = arm; n = 20 / 20, independent)\n")
        print(table(rows, label="arm"))

# docs/studies-run.md §4.6-4.7: does the search add anything once a model is floored?
# `floor50` keeps the 100-trial search under a 50-epoch floor; `paper90` pins one trial
# under a 90-epoch floor. They differ in the floor length too, so this is the only
# family-T contrast between a searched and a pinned floored arm, not a clean isolation.
for metric in ("mape_aggregate", "spearman"):
    rows = []
    for model in ("ValendinLSTM", "LSTM"):
        g = d[d.model == model]
        e = effect(g[g.arm == "floor50"][metric], g[g.arm == "paper90"][metric],
                   metric, "electronics")
        e.panel = f"{model}: floor50 (searched) vs paper90 (pinned)"
        rows.append(e)
    print(f"\n### Searched against pinned under a floor: {metric} (A = paper90, B = floor50)\n")
    print(table(rows, label="model"))
