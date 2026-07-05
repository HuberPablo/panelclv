"""One-off check: rebuild Transformer trial 22's config and train it once,
printing the per-epoch loss curve + wall-clock time.

Purpose: prove the ~10s trial was real training (loss descends over many epochs,
then early-stops) rather than a no-op. Reuses the exact tuner code path
(_build_transformer + make_data_builder + fit_model), so the model/loaders match
what run_optuna_study built.
"""

import os
import sys
import time
from pathlib import Path

# --- repo bootstrap (same as the notebook's first cell) ---
_root = Path(__file__).resolve().parent.parent
os.chdir(_root)
sys.path.insert(0, str(_root / "src"))

import torch
import pandas as pd

from panelclv.configs.panel_config import PanelConfig
from panelclv.data_preparation import dynamic_panel_dataset
from panelclv.experiments import make_data_builder
from panelclv.tuning.optuna_tuning import _build_transformer
from panelclv.training import fit_model

# Determinism so the curve is reproducible (same two lines as the notebook).
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# ---------------------------------------------------------------------------
# 1. Rebuild data_full exactly as the active notebook config does (cell 6/10).
# ---------------------------------------------------------------------------
cfg = PanelConfig(
    id_col="Id", target_col="Transactions", frequency="weekly",
    training_start="1999-01-01", training_end="2000-12-31",
    validation_start="2000-01-01",
    holdout_start="2001-01-01", holdout_end="2001-12-31",
    time_cols=("year", "week"), clip_target_upper=6,
    require_calibration_activity=True,
    time_features={"add_year_idx": True, "add_week_sin_cos": True},
    known_future=(), static=(), observed_past=(),
    embedded_cols={"Transactions": "auto"},
)
panel = pd.read_csv("Datasets/Dataset_clean/electronics_customer_week_panel.csv")
data_full = dynamic_panel_dataset.prepare_dataset(panel, cfg)
print(f"data: N={len(data_full['ids'])}  T_CAL={data_full['T_CAL']}  "
      f"F={data_full['F']}  seq_cols={data_full['seq_cols']}")

# ---------------------------------------------------------------------------
# 2. Trial 22's exact hyperparameters (from the Optuna log line).
# ---------------------------------------------------------------------------
params = {
    "d_model": 128, "nhead": 4, "num_encoder_layers": 3, "dropout": 0.3,
    "learning_rate": 0.001775092424066087,
    "weight_decay": 0.00011151871619879718,
    "batch_size": 64,
}
device = "cuda" if torch.cuda.is_available() else "cpu"

# Same loaders the tuner builds: no feature dropping, this trial's batch size.
train_loader, val_loader, metadata = make_data_builder(data_full)(
    feature_config=[], batch_size=params["batch_size"],
)
n_train_batches = len(train_loader)
print(f"device={device}  batch_size={params['batch_size']}  "
      f"train_batches/epoch={n_train_batches}")

model = _build_transformer(params, metadata)
n_params = sum(p.numel() for p in model.parameters())
print(f"transformer params: {n_params:,}")

# ---------------------------------------------------------------------------
# 3. Train once, timed. patience=9 (top of the search set) so we see the full
#    descend-then-early-stop curve; n_epochs=100 matches the study budget.
# ---------------------------------------------------------------------------
t0 = time.time()
result = fit_model(
    model=model, train_loader=train_loader, val_loader=val_loader,
    max_trans=model.num_target_classes,
    n_epochs=100, patience=9,
    learning_rate=params["learning_rate"], weight_decay=params["weight_decay"],
    grad_clip=1.0, device=device,
    checkpoint_dir="./checkpoints/_verify_trial22",
    model_name="transformer_verify_trial22",
    verbose=False, loss_type="cross_entropy",
    val_score_start=metadata.get("val_score_start", 0),
)
elapsed = time.time() - t0

# ---------------------------------------------------------------------------
# 4. Print the per-epoch curve so the descent is visible.
# ---------------------------------------------------------------------------
print("\nepoch | train_loss | val_loss  | val_f1")
print("-" * 44)
for rec in result.history:
    print(f"{rec.get('epoch', '?'):>5} | "
          f"{rec.get('train_loss', float('nan')):>10.5f} | "
          f"{rec.get('val_loss', float('nan')):>8.5f} | "
          f"{rec.get('val_f1', float('nan')):>6.4f}")

print("-" * 44)
print(f"epochs run     : {len(result.history)} (early-stopped before 100)")
print(f"best epoch     : {result.best_epoch}")
print(f"best val_loss  : {result.best_val_loss:.6f}   "
      f"(study logged trial 22 = 0.09076822)")
print(f"wall-clock     : {elapsed:.1f}s")
