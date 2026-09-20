"""Does the paper's stopping rule quit early because of OUR split, or because of the panel?

Family T runs the notebook's recipe under this package's TEMPORAL validation split
(ADR-0001), and it stops at epoch 1 on electronics. That leaves one alternative reading:
the split, not the panel, is what makes the curve flat — the notebook holds out a random
10% of CUSTOMERS and scores every period, which is a different quantity.

This runs the notebook's recipe under the notebook's OWN split on electronics, changing
nothing else. If it stops early here too, the stopping rule does not transfer to this
panel and the split is not the explanation.

    PYTHONPATH=src .../python .scratch/training-budget/paper_split_check.py
"""
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / "scripts"))

from run_real_panel_benchmarks import build_data                    # noqa: E402
from panelclv.registry import build_model                           # noqa: E402
from panelclv.training.loop import fit_model                        # noqa: E402

# The notebook's recipe, cell for cell (banking_transactions_demo.ipynb, cells 10 and 15).
PARAMS = {"learning_rate": 1e-3, "weight_decay": 0.0, "batch_size": 32}
MAX_EPOCHS, PATIENCE, VALIDATION_SPLIT = 150, 5, 0.1
REPS = 3


def customer_wise_split(data: dict, batch_size: int, seed: int):
    """The notebook's split: shuffle customers, hold out 10%, score every period.

    Unlike `trials.split_calibration` this cuts across customers rather than across time,
    so both loaders carry the FULL calibration sequence and no `val_score_start` applies.
    """
    X = data["samples"]                                  # (N, T-1, F) float32
    y = data["targets"].squeeze(-1).astype(np.int64)     # (N, T-1) class indices

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    n_valid = round(len(order) * VALIDATION_SPLIT)
    valid_idx, train_idx = order[:n_valid], order[n_valid:]

    def loader(idx, shuffle):
        return DataLoader(
            TensorDataset(torch.from_numpy(X[idx].copy()),
                          torch.from_numpy(y[idx].copy())),
            batch_size=batch_size, shuffle=shuffle)

    recipe = {"seq_cols": data["seq_cols"], "embedded_cols": data["embedded_cols"],
              "target_col": data["target_col"], "seq_len": X.shape[1]}
    return loader(train_idx, True), loader(valid_idx, False), recipe, len(train_idx), len(valid_idx)


data = build_data("electronics", "2y")
rows = []
for rep in range(REPS):
    train_loader, val_loader, recipe, n_tr, n_va = customer_wise_split(
        data, PARAMS["batch_size"], seed=rep)
    model = build_model("valendin_lstm", {**PARAMS, "n_epochs": MAX_EPOCHS}, recipe)
    res = fit_model(
        model, train_loader, val_loader,
        num_target_classes=model.num_target_classes,
        n_epochs=MAX_EPOCHS, patience=PATIENCE,
        learning_rate=PARAMS["learning_rate"], weight_decay=PARAMS["weight_decay"],
        checkpoint_dir=RESULTS / "split_check_ckpt", model_name=f"split_{rep}",
        verbose=False, val_score_start=0,          # the notebook scores every period
    )
    rows.append(dict(rep=rep, train_customers=n_tr, valid_customers=n_va,
                     best_epoch=res.best_epoch, epochs_run=len(res.history),
                     best_val_loss=res.best_val_loss))
    print(f"rep{rep}: best_epoch={res.best_epoch:3d} of {len(res.history):3d} run, "
          f"val={res.best_val_loss:.5f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "paper_split_check.csv", index=False)
print("\n", df.to_string(index=False))
print(f"\nThe notebook reports ~90 epochs on its own data. Here: "
      f"{df.best_epoch.min()}-{df.best_epoch.max()}.")
