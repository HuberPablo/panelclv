# Modular_Models

Modular **LSTM** and **Transformer** models for customer-base transaction-count
forecasting, with **Pareto/NBD** benchmarks. The thesis target is the Valendin et al.
workflow: the models are *classifiers over transaction-count classes* that forecast by
**autoregressive Monte Carlo simulation** (sample a count per period, feed it back,
average many paths) — not point regressors. See `CLAUDE.md` for the modeling details.

## Install

From the repo root (use your PyTorch venv):

```bash
pip install -e .
```

This installs the single top-level `panelclv` package (with the `panelclv.models`,
`panelclv.data_preparation`, and `panelclv.configs` subpackages), so
`from panelclv.models import ...` works anywhere — no `sys.path` hacks.

## Quickstart

The whole flow is: build/load a panel → prepare tensors → tune (Optuna) → rebuild the
winning model → Monte Carlo forecast → report. The three `experiment_utils` helpers
(`make_data_builder`, `build_inference_from_trial`, and `make_loaders`) absorb the
mechanical glue so the notebook stays in control of every modeling choice.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from panelclv.configs.panel_config import PanelConfig
from panelclv.data_preparation import dynamic_panel_dataset
from panelclv.models import make_data_builder, build_inference_from_trial, run_optuna_study, \
    mc_forecast, mc_compute_metrics

# 1. Panel -> model-ready tensors (calibration/holdout/samples/targets/seq_cols/...).
panel = pd.read_csv("Datasets/Dataset_clean/electronics_customer_week_panel.csv")
cfg = PanelConfig(id_col="Id", target_col="Transactions", frequency="weekly",
                  training_start="1999-01-01", training_end="2000-12-31",
                  holdout_start="2001-01-01", holdout_end="2001-12-31",
                  time_cols=("year", "week"), clip_target_upper=6)
data_full = dynamic_panel_dataset.prepare_dataset(panel, cfg)

# 2. Customer-wise split (rows are customers).
train_idx, val_idx = train_test_split(np.arange(data_full["N"]), test_size=0.1,
                                       random_state=42)

# 3. Tune. make_data_builder gives run_optuna_study the per-trial data closure; every
#    other knob (selection_metric, removable_features, loss config, rollout_*) stays
#    yours to set here.
study = run_optuna_study(
    model_type="lstm",
    data_builder=make_data_builder(data_full, train_idx, val_idx),
    data_info={"n_epochs": 150, "patience": 7,
               "checkpoint_dir": "./checkpoints/lstm_optuna", "loss_type": "cross_entropy"},
    n_trials=30,
)

# 4. Rebuild the winning model + load its checkpoint. Returns the model AND data_best
#    (data sliced to the winning feature subset) -- always forecast with data_best.
inference_model, data_best = build_inference_from_trial(study, data_full, "lstm")

# 5. Autoregressive Monte Carlo forecast + metrics.
forecast = mc_forecast(inference_model, data_best, n_simulations=600, seed=42)
print(mc_compute_metrics(forecast["actual"], forecast["prediction_mean"]))
```

Swap `model_type="lstm"` / `"transformer"` (and `mc_forecast` /
`mc_forecast_transformer`) to run the other family on the same contract.

## Notebooks

`Data_integration_LSTM_v2.ipynb` and `Data_integration_TRANSFORMER_v2.ipynb` are the
runnable, annotated walkthroughs of the flow above (built on the helpers). The
original `Data_integration_LSTM.ipynb` / `Data_integration_TRANSFORMER.ipynb` are kept
for reference.
