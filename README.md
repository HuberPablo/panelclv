# panelclv

Modular **LSTM** and **Transformer** models for customer-base transaction-count
forecasting, with **Pareto/NBD** benchmarks. The thesis target is the Valendin et al.
workflow: the models are *classifiers over transaction-count classes* that forecast by
**autoregressive Monte Carlo simulation** (sample a count per period, feed it back,
average many paths) — not point regressors.

## Install

From the repo root (use your PyTorch venv):

```bash
pip install -e .
```

The project uses a **src-layout** (the package lives in `src/panelclv/`), so installing
it is what puts `panelclv` on the path — there are no `sys.path` hacks. It is split by
concern into subpackages: `panelclv.models`, `panelclv.training`, `panelclv.tuning`,
`panelclv.evaluation`, `panelclv.benchmarks`, `panelclv.experiments`,
`panelclv.data_preparation`, `panelclv.configs`. Import from the relevant one, e.g.
`from panelclv.tuning import run_optuna_study`. For the test runner, use
`pip install -e ".[dev]"` and run `pytest`.

## Quickstart

The whole flow is: build/load a panel → prepare tensors → tune (Optuna) → rebuild the
winning model (optionally retrain on full calibration) → Monte Carlo forecast → report.
The `panelclv.experiments` helpers (`make_data_builder`, `make_loaders`,
`build_inference_from_trial`, `refit_best_trial`) absorb the mechanical glue so the
notebook stays in control of every modeling choice.

The train/validation split is **temporal**: set `validation_start` in `PanelConfig` and
the calibration window is cut at that date. Weights train only on
`[training_start, validation_start)`; the tail `[validation_start, training_end]` is the
validation window (all customers), used for early stopping / model selection but never
trained on. There is no customer-wise split.

```python
import pandas as pd

from panelclv.configs.panel_config import PanelConfig
from panelclv.data_preparation import dynamic_panel_dataset
from panelclv.tuning import run_optuna_study
from panelclv.experiments import make_data_builder, build_inference_from_trial, refit_best_trial
from panelclv.models import mc_forecast, mc_compute_metrics

# 1. Panel -> model-ready tensors (calibration/holdout/samples/targets/seq_cols/...).
#    validation_start carves the temporal validation window off the calibration tail.
panel = pd.read_csv("Datasets/Dataset_clean/electronics_customer_week_panel.csv")
cfg = PanelConfig(id_col="Id", target_col="Transactions", frequency="weekly",
                  training_start="1999-01-01", training_end="2000-12-31",
                  validation_start="2000-07-01",
                  holdout_start="2001-01-01", holdout_end="2001-12-31",
                  time_cols=("year", "week"), clip_target_upper=6)
data_full = dynamic_panel_dataset.prepare_dataset(panel, cfg)

# 2. Tune. make_data_builder gives run_optuna_study the per-trial data closure (the
#    temporal split is carried in data_full["val_start_idx"]); every other knob
#    (selection_metric, removable_features, loss config, rollout_*) stays yours to set.
study = run_optuna_study(
    model_type="lstm",
    data_builder=make_data_builder(data_full),
    data_info={"n_epochs": 150, "patience": 7,
               "checkpoint_dir": "./checkpoints/lstm_optuna", "loss_type": "cross_entropy"},
    n_trials=30,
)

# 3. Final model. Either load the tuning checkpoint as-is...
inference_model, data_best = build_inference_from_trial(study, data_full, "lstm")
# ...or warm-start retrain the winner on the FULL calibration (validation tail
#    included) for a few big-batch epochs, the Valendin et al. paper's final step:
inference_model, data_best = refit_best_trial(study, data_full, "lstm", batch_size=512)

# 4. Autoregressive Monte Carlo forecast + metrics (always forecast with data_best).
forecast = mc_forecast(inference_model, data_best, n_simulations=600, seed=42)
print(mc_compute_metrics(forecast["actual"], forecast["prediction_mean"]))
```

Swap `model_type="lstm"` / `"transformer"` (and `mc_forecast` /
`mc_forecast_transformer`) to run the other family on the same contract.

## Notebooks

All notebooks live in `notebooks/`. `notebooks/Data_integration_LSTM_v2.ipynb` and
`notebooks/Data_integration_TRANSFORMER_v2.ipynb` are the runnable, annotated
walkthroughs of the flow above (built on the helpers); the un-suffixed
`Data_integration_{LSTM,TRANSFORMER}.ipynb` are kept for reference, and
`dataset_building.ipynb` builds the clean panels from raw data. Each notebook opens with
a small bootstrap cell that locates the repo root and makes `panelclv` importable, so
they run whether or not the package is pip-installed.
