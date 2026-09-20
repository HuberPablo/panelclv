"""Family T, tested: each arm against `archive`, 20 vs 20, Mann-Whitney."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from run_training_budget import (MODELS, ARMS, N_REPLICATIONS, build_data,
                                 forecast_path, score)
from panelclv.data_preparation.target_channel import holdout_actuals

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
d = pd.DataFrame(rows)
d.to_csv(Path(__file__).resolve().parent / "results" / "family_t_scores.csv", index=False)

for model, g in d.groupby("model"):
    base = g[g.arm == "archive"]
    print(f"\n{model} — each arm vs `archive`, Mann-Whitney, 20 vs 20")
    print(f"{'arm':9s} {'MAPE':>7s} {'p':>9s}   {'Spearman':>9s} {'p':>9s}   {'|bias|':>7s} {'p':>9s}")
    for arm in ARMS:
        if arm == "archive":
            print(f"{arm:9s} {base.mape_aggregate.mean():7.2f} {'—':>9s}   "
                  f"{base.spearman.mean():9.3f} {'—':>9s}   "
                  f"{base.bias_percent.abs().mean():7.2f} {'—':>9s}")
            continue
        a = g[g.arm == arm]
        pm = mannwhitneyu(a.mape_aggregate, base.mape_aggregate).pvalue
        ps = mannwhitneyu(a.spearman, base.spearman).pvalue
        pb = mannwhitneyu(a.bias_percent.abs(), base.bias_percent.abs()).pvalue
        print(f"{arm:9s} {a.mape_aggregate.mean():7.2f} {pm:9.2e}   "
              f"{a.spearman.mean():9.3f} {ps:9.2e}   "
              f"{a.bias_percent.abs().mean():7.2f} {pb:9.2e}")
