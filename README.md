# panelclv

Modular **LSTM** and **Transformer** models for customer-base transaction-count
forecasting, with a **Pareto/NBD** benchmark. The thesis target is the Valendin et al.
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
concern into subpackages: `panelclv.models`, `panelclv.registry`, `panelclv.training`,
`panelclv.tuning`, `panelclv.evaluation`, `panelclv.predictions`,
`panelclv.benchmarks`, `panelclv.trials`, `panelclv.studies`,
`panelclv.data_preparation`, `panelclv.configs`. Import from the relevant one, e.g.
`from panelclv.tuning import run_optuna_study`. For the test runner, use
`pip install -e ".[dev]"` and run `pytest`.

## Quickstart

The whole flow is: build/load a panel → prepare tensors → tune (Optuna) → refit the
winning trial on the full calibration window → Monte Carlo forecast → report. The
`panelclv.trials` helpers (`make_data_builder`, `split_calibration`, `refit_best_trial`)
absorb the mechanical glue so the notebook stays in control of every modeling choice.

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
from panelclv.trials import make_data_builder, refit_best_trial
from panelclv.models import forecast_recurrent, compute_forecast_metrics

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
#    (removable_features, loss config, pruning) stays yours to set.
study = run_optuna_study(
    model_type="lstm",
    data_builder=make_data_builder(data_full),
    training={"n_epochs": 150, "patience": 7,
              "checkpoint_dir": "./checkpoints/lstm_optuna", "loss_type": "cross_entropy"},
    n_trials=30,
)

# 3. Final model: warm-start retrain the winner on the FULL calibration window
#    (validation tail included) for several big-batch epochs — the Valendin et al.
#    paper's final step, and the only route to a forecast here (ADR-0008). It hands
#    back the rollout model the refit model itself provides (ADR-0007).
rollout_model, data_best = refit_best_trial(study, data_full, "lstm", batch_size=512)

# 4. Autoregressive Monte Carlo forecast + metrics (always forecast with data_best).
forecast = forecast_recurrent(rollout_model, data_best, n_simulations=600, seed=42)
print(compute_forecast_metrics(forecast["actual"], forecast["prediction_mean"]))
```

Swap `model_type="lstm"` / `"transformer"` (and `forecast_recurrent` /
`forecast_attention`) to run the other family on the same contract.

## Notebooks

The four live notebooks are in `notebooks/`. `Data_integration_LSTM_v2.ipynb` and
`Data_integration_TRANSFORMER_v2.ipynb` are the runnable, annotated walkthroughs of the
flow above (built on the helpers); `Study.ipynb` runs study suites, and
`Pareto_Datasets.ipynb` runs them over synthetic Pareto/NBD panels. Each opens with a
small bootstrap cell that locates the repo root and makes `panelclv` importable, so they
run whether or not the package is pip-installed.

Finished experiments frozen against an older API sit in `notebooks/archive/` — see its
README for which and why. Nothing there is expected to run, and
`tests/test_notebooks_current_api.py` checks only the live four.

## Running on rented hardware

`VastAI/` holds the launch scripts for training on a rented [vast.ai](https://vast.ai)
machine: `vast_search.py` picks an offer (the workload is CPU-bound, so it filters on CPU
generation rather than GPU), `vast_launch.sh` rents it, and `vast_onstart.sh` /
`script_on_start` clone the repo and install the package on the machine once it boots.
