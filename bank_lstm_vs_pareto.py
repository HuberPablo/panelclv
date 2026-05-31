"""Sanity-check the cohort filter end-to-end on the bank panel: LSTM vs Pareto/NBD.

Mirrors the Data_integration_LSTM.ipynb pipeline (prepare_dataset -> train -> MC
forecast -> score) but on the Czech bank panel, with a direct single fit instead of an
Optuna search (this is a wiring/cohort check, not a tuning run). It then runs the
Pareto/NBD benchmark on the SAME cohort and the SAME holdout actuals, so the two are
directly comparable.

The point is to confirm: (1) require_calibration_activity actually selects the paper's
cohort, and (2) the LSTM and the benchmark see identical customers (same N, same ids,
same actuals).

Run: /home/virthian/Desktop/Thesis/venvs/thesis_rocm/bin/python bank_lstm_vs_pareto.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from configs.panel_config import PanelConfig
from Data_preparation import dynamic_panel_dataset
from Models import (
    MultinomialLSTMModel,
    InferenceMultinomialLSTMModel,
    fit_model,
    mc_forecast,
    mc_compute_metrics,
    compute_pareto_predictions,
)

SEED = 42
N_SIMULATIONS = 200          # MC paths (notebook uses 500; 200 is plenty for a check)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# 1) Config — paper's banking window (calibration 1993-1995, holdout 1996-1998).
# ---------------------------------------------------------------------------
cfg = PanelConfig(
    id_col="Id",
    target_col="Transactions",
    frequency="weekly",
    training_start="1993-01-01", training_end="1995-12-31",
    holdout_start="1996-01-01",  holdout_end="1998-12-31",
    time_cols=("year", "week"),
    clip_target_upper=10,                 # bank weekly max is 11 -> cardinality 11
    require_calibration_activity=True,    # the cohort filter under test
    time_features={"add_year_idx": True, "add_week_sin_cos": True},
    known_future=("year_idx",),           # bank panel has no covariates
    embedded_cols={"Transactions": "auto"},
)

panel = pd.read_csv("Datasets/Dataset_clean/bank_customer_week_panel.csv")
print(f"full panel: {panel['Id'].nunique()} customers")

data = dynamic_panel_dataset.prepare_dataset(panel, cfg, verbose=False)
ids = data["ids"]
print(f"cohort after require_calibration_activity=True: N={data['N']} "
      f"(T_CAL={data['T_CAL']}, T_HOLD={data['T_HOLD']}, F={data['F']})")

# ---------------------------------------------------------------------------
# 2) Tensors for fit_model.
# ---------------------------------------------------------------------------
X = data["samples"]
y = data["targets"].squeeze(-1).astype(np.int64)
max_trans = data["input_spec"]["embedded_cols"][data["target_col"]]
assert y.min() >= 0 and y.max() < max_trans

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

train_idx, val_idx = train_test_split(np.arange(data["N"]), test_size=0.2, random_state=SEED)
train_loader = DataLoader(
    TensorDataset(torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx])),
    batch_size=128, shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx])),
    batch_size=128, shuffle=False,
)

# ---------------------------------------------------------------------------
# 3) Train a modest LSTM (single fit, early-stopped).
# ---------------------------------------------------------------------------
ARCH = dict(hidden_dim=64, memory_units=64, dense_units=64, dropout=0.1)
model = MultinomialLSTMModel(
    seq_cols=data["seq_cols"], input_spec=data["input_spec"],
    target_col=data["target_col"], **ARCH,
)
result = fit_model(
    model, train_loader, val_loader, max_trans=max_trans,
    n_epochs=25, patience=5, learning_rate=1e-3, device=device,
    checkpoint_dir="./checkpoints/bank_sanity", model_name="bank_lstm", verbose=False,
)
print(f"LSTM trained: best_val_loss={result.best_val_loss:.4f} @ epoch {result.best_epoch}")

# ---------------------------------------------------------------------------
# 4) MC forecast (Valendin-style autoregressive sampling).
# ---------------------------------------------------------------------------
inference_model = InferenceMultinomialLSTMModel(
    seq_cols=data["seq_cols"], input_spec=data["input_spec"],
    target_col=data["target_col"], mode="sample", **ARCH,
)
inference_model.load_state_dict(torch.load(result.checkpoint_path, map_location="cpu"))
forecast = mc_forecast(inference_model, data, n_simulations=N_SIMULATIONS, device=device, seed=SEED)
lstm_pred = forecast["prediction_mean"]      # (N, T_HOLD)
actual = forecast["actual"]                  # (N, T_HOLD)

# ---------------------------------------------------------------------------
# 5) Pareto/NBD benchmark on the SAME cohort (customer_ids=ids aligns the order).
# ---------------------------------------------------------------------------
pareto_pred, pareto_ids = compute_pareto_predictions(
    data["train_panel"], holdout_length=data["T_HOLD"],
    id_col="Id", target_col="Transactions", customer_ids=ids,
)

# ---------------------------------------------------------------------------
# 6) Cohort-alignment assertions + scores.
# ---------------------------------------------------------------------------
assert pareto_ids == list(ids), "Pareto ids differ from LSTM cohort order"
assert pareto_pred.shape == lstm_pred.shape == actual.shape, "shape mismatch across models"
print(f"\ncohort alignment OK: LSTM and Pareto both on {len(ids)} customers, "
      f"shapes {actual.shape}")

lstm_metrics = mc_compute_metrics(actual, lstm_pred)
pareto_metrics = mc_compute_metrics(actual, pareto_pred)

print("\n=== Holdout metrics (same cohort, same actuals) ===")
print(f"{'metric':<22}{'LSTM':>14}{'Pareto/NBD':>14}")
for k in ("rmse", "bias_percent", "mape_aggregate_style"):
    print(f"{k:<22}{lstm_metrics[k]:>14.4f}{pareto_metrics[k]:>14.4f}")

print(f"\naggregate holdout transactions: actual={actual.sum():.0f} | "
      f"LSTM={lstm_pred.sum():.0f} | Pareto={pareto_pred.sum():.0f}")
