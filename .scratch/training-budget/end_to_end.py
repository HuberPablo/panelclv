"""Does a bigger patience change the FORECAST, not just the validation curve?

Same frozen benchmark, same panel, same refit and Monte Carlo as an archived study —
only the early-stopping patience differs. Five unseeded replications per arm.
"""
import sys, time
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr

from pathlib import Path

# The repo root, derived so these run unchanged on a rented box, and an
# output folder beside this script.
REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO / "scripts"))
from run_real_panel_benchmarks import build_data                       # noqa: E402
from panelclv.trials import split_calibration, refit_loader            # noqa: E402
from panelclv.registry import build_model                              # noqa: E402
from panelclv.training.loop import fit_model, refit_full_calibration   # noqa: E402
from panelclv.models import compute_forecast_metrics                   # noqa: E402
from panelclv.models.monte_carlo_forecasting import forecast_recurrent # noqa: E402
from panelclv.data_preparation.target_channel import holdout_actuals   # noqa: E402

OUT = RESULTS
PARAMS = {"learning_rate": 0.0011503383899724748,
          "weight_decay": 0.0005669410925868405, "batch_size": 256}
ARMS = {"patience_7":  dict(n_epochs=100, patience=7),     # the archived setting
        "patience_40": dict(n_epochs=300, patience=40)}
REPS, N_SIM = 5, 200
PANEL = sys.argv[1] if len(sys.argv) > 1 else "electronics"

data = build_data(PANEL, "2y")
actual_tot = holdout_actuals(data).sum(axis=1)
full_recipe = {"seq_cols": data["seq_cols"], "embedded_cols": data["embedded_cols"],
               "target_col": data["target_col"], "seq_len": data["samples"].shape[1]}

rows = []
for arm, cfg in ARMS.items():
    for rep in range(REPS):
        t0 = time.time()
        split = split_calibration(data, PARAMS["batch_size"])
        model = build_model("valendin_lstm", {**PARAMS, **cfg}, split.recipe)
        res = fit_model(model, split.train_loader, split.val_loader,
                        num_target_classes=model.num_target_classes,
                        n_epochs=cfg["n_epochs"], patience=cfg["patience"],
                        learning_rate=PARAMS["learning_rate"],
                        weight_decay=PARAMS["weight_decay"],
                        checkpoint_dir=OUT / "e2e", model_name=f"{arm}_{rep}",
                        verbose=False, val_score_start=split.recipe["val_score_start"])
        # ADR-0008 refit: warm-start the tuned weights, 5 big-batch epochs on the
        # full calibration window — exactly what a study does before forecasting.
        rmodel = build_model("valendin_lstm", {**PARAMS, **cfg}, full_recipe)
        refit_full_calibration(rmodel, refit_loader(data, 512),
                               num_target_classes=rmodel.num_target_classes,
                               n_epochs=5, learning_rate=1e-3, weight_decay=1e-3,
                               checkpoint_dir=OUT / "e2e",
                               model_name=f"{arm}_{rep}_refit",
                               warm_start_state=res.checkpoint_path, verbose=False)
        fc = forecast_recurrent(rmodel.to_rollout(), data, n_simulations=N_SIM,
                                seed=42 + rep, return_simulations=False)
        m = compute_forecast_metrics(fc["actual"], fc["prediction_mean"])
        pred_tot = fc["prediction_mean"].sum(axis=1)
        rows.append(dict(panel=PANEL, arm=arm, rep=rep, best_epoch=res.best_epoch,
                         val_loss=res.best_val_loss,
                         spearman=spearmanr(pred_tot, actual_tot).statistic,
                         pred_sd=float(pred_tot.std()), **m))
        print(f"{arm} rep{rep}: best_epoch={res.best_epoch:3d} val={res.best_val_loss:.5f} "
              f"bias={m['bias_percent']:+7.2f} mape={m['mape_aggregate']:6.2f} "
              f"rho={rows[-1]['spearman']:.3f} pred_sd={rows[-1]['pred_sd']:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)

d = pd.DataFrame(rows)
d.to_csv(OUT / f"end_to_end_{PANEL}.csv", index=False)
print("\n", d.groupby("arm").agg(
    best_epoch=("best_epoch", "mean"), val_loss=("val_loss", "mean"),
    bias=("bias_percent", "mean"), bias_sd=("bias_percent", "std"),
    mape=("mape_aggregate", "mean"), spearman=("spearman", "mean"),
    spearman_sd=("spearman", "std"), pred_sd=("pred_sd", "mean")).round(4).to_string())
