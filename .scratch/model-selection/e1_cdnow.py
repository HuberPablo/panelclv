"""E1 on CDNOW: each training ingredient as the only change against the archive recipe
(`docs/model-selection.md` S5, `docs/training-budget.md` to-do E1).

Arms, 20 replications each, per model: `archive` (patience 7, 100-trial search),
`floor50` (the same with a 50-epoch floor), `paper` (the paper's recipe without a floor),
from `scripts/run_training_budget.py`. Every replication is its own Optuna search, so the
arms are independent samples: `effect(arm, archive, paired=False)`
(`docs/statistical-protocol.md`). Aggregate MAPE is read from each study's `metrics.csv`,
which is also what `.scratch/training-budget/results/e1_cdnow.csv` holds."""
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
import pandas as pd
from panelclv.evaluation.effects import effect, table
rows = [pd.read_csv(f).assign(arm=f.parts[-3].split("__")[3])
        for f in sorted((REPO / "Studies").glob("training_budget__*__cdnow__*__r*/*/metrics.csv"))]
d = pd.concat(rows, ignore_index=True)
for model, dm in d.groupby("model"):
    base = dm[dm.arm == "archive"].mape_aggregate
    effs = [effect(dm[dm.arm == arm].mape_aggregate, base, paired=False,
                   metric="mape_aggregate", panel=f"{model}: {arm} vs archive", refit_noise=5.83)
            for arm in ("floor50", "paper")]
    print(f"\n## cdnow / {model}, MAPE (refit noise 5.83)\n")
    print(table(effs))
