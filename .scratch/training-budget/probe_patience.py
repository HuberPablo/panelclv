"""Was patience 7 the binding constraint on the real panels?

Trains the frozen benchmark on electronics/multichannel/cdnow with early stopping
effectively disabled (patience = n_epochs), then asks of each run's validation curve:
where would patience 7 have stopped, and how much validation cross-entropy was left
on the table after that point?
"""
import json, sys
import numpy as np, pandas as pd, torch

from pathlib import Path

# The repo root, derived so these run unchanged on a rented box, and an
# output folder beside this script.
REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO / "scripts"))
from run_real_panel_benchmarks import build_data                    # noqa: E402
from panelclv.trials import split_calibration                       # noqa: E402
from panelclv.registry import build_model                           # noqa: E402
from panelclv.training.loop import fit_model                        # noqa: E402

OUT = RESULTS
N_EPOCHS, PATIENCE_TEST = 200, 7
# The hyperparameters a real archived study selected on electronics (study_01, r00).
PARAMS = {"learning_rate": 0.0011503383899724748,
          "weight_decay": 0.0005669410925868405, "batch_size": 256}
REPS = 3

def stop_epoch(hist, patience):
    """The epoch index the loop would have broken at, under `patience`."""
    best, counter = np.inf, 0
    for h in hist:
        if h["val_loss"] + 1e-4 < best:
            best, counter = h["val_loss"], 0
        else:
            counter += 1
            if counter >= patience:
                return h["epoch"], best
    return hist[-1]["epoch"], best

rows, curves = [], []
for panel in ("electronics", "multichannel", "cdnow"):
    data = build_data(panel, "2y")
    for rep in range(REPS):
        split = split_calibration(data, PARAMS["batch_size"])
        model = build_model("valendin_lstm", {**PARAMS, "n_epochs": N_EPOCHS}, split.recipe)
        res = fit_model(
            model, split.train_loader, split.val_loader,
            num_target_classes=model.num_target_classes,
            n_epochs=N_EPOCHS, patience=N_EPOCHS,          # never early-stops
            learning_rate=PARAMS["learning_rate"],
            weight_decay=PARAMS["weight_decay"],
            checkpoint_dir=OUT / "ckpt", model_name=f"{panel}_{rep}",
            verbose=False, val_score_start=split.recipe["val_score_start"],
        )
        h = res.history
        e7, best7 = stop_epoch(h, PATIENCE_TEST)
        rows.append(dict(panel=panel, rep=rep,
                         stop_epoch_p7=e7, best_val_p7=best7,
                         best_epoch_200=res.best_epoch, best_val_200=res.best_val_loss,
                         gain=best7 - res.best_val_loss,
                         gain_pct=100 * (best7 - res.best_val_loss) / best7))
        for r in h:
            curves.append(dict(panel=panel, rep=rep, **r))
        print(f"{panel} rep{rep}: p7 stops at epoch {e7+1} (best val {best7:.5f}) | "
              f"200-epoch best at epoch {res.best_epoch+1} (val {res.best_val_loss:.5f}) | "
              f"left on the table {100*(best7-res.best_val_loss)/best7:.2f}%", flush=True)

pd.DataFrame(rows).to_csv(OUT / "patience_probe.csv", index=False)
pd.DataFrame(curves).to_csv(OUT / "patience_curves.csv", index=False)
print("\n", pd.DataFrame(rows).round(5).to_string(index=False))
