"""What does an unseeded refit move, per panel and metric?

The number every effect in `docs/training-budget.md` is printed beside: refit the SAME
checkpoint a second time, change nothing else, and see how far the forecast moves. That is
the resolution below which a difference between two conditions is small whatever its
p-value.

`Rescored/` holds a second refit of each archived family-N winner (`run_rescore_trials.py`),
and `Studies/` holds the first. Pairing them gives the movement directly. Spearman is not
stored in `metrics.csv` — it is recomputed from the stored predictions — so it is
recomputed here for the archived side, which is why this script exists rather than a
one-line join.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))

from run_real_panel_benchmarks import build_data, spearman            # noqa: E402
from panelclv.data_preparation.target_channel import holdout_actuals  # noqa: E402
from panelclv.models import compute_forecast_metrics                  # noqa: E402
from panelclv.studies import load_model_predictions                   # noqa: E402

rows = []
cache: dict[str, tuple] = {}
for f in sorted((REPO / "Rescored").glob("real_panel_benchmarks__ValendinLSTM__*.csv")):
    suite = re.sub(r"\.csv$", "", f.name)
    panel = suite.split("__")[2]
    model_dir = REPO / "Studies" / suite / "ValendinLSTM"
    if not (model_dir / "Predictions").is_dir():
        continue
    if panel not in cache:
        data = build_data(panel, "2y")
        cache[panel] = (holdout_actuals(data), np.asarray(data["ids"]))
    actual, ids = cache[panel]

    # Refit 1: the forecast the archive actually reported.
    values, pred_ids = load_model_predictions(model_dir, study=1)
    if pred_ids is not None and not np.array_equal(np.asarray(pred_ids), ids):
        continue
    first = {**compute_forecast_metrics(actual, values),
             "spearman": spearman(values.sum(axis=1), actual.sum(axis=1))}

    # Refit 2: the same checkpoint put through the production path again.
    d = pd.read_csv(f)
    second = d.loc[d.val_loss.idxmin()]

    rows.append(dict(panel=panel, suite=suite,
                     d_mape=abs(first["mape_aggregate"] - second.mape_aggregate),
                     d_bias=abs(first["bias_percent"] - second.bias_percent),
                     d_rmse=abs(first["rmse"] - second.rmse),
                     d_spearman=abs(first["spearman"] - second.spearman)))

d = pd.DataFrame(rows)
d.to_csv(RESULTS / "refit_noise.csv", index=False)
print(f"{len(d)} winners refit twice\n")
print("Mean |movement| from an unseeded refit of the same checkpoint:\n")
print("| panel | n | MAPE | bias % | Spearman | RMSE |")
print("| --- | ---: | ---: | ---: | ---: | ---: |")
for panel, g in d.groupby("panel"):
    print(f"| {panel} | {len(g)} | {g.d_mape.mean():.2f} | {g.d_bias.mean():.2f} | "
          f"{g.d_spearman.mean():.4f} | {g.d_rmse.mean():.5f} |")
print(f"\nmedian Spearman movement: {d.d_spearman.median():.4f}, "
      f"90th percentile: {d.d_spearman.quantile(0.9):.4f}")
