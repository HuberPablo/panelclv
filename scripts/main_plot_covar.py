"""Plot weekly aggregated transactions for three covariate configurations.

Counterpart to the old `main_plot_covar.py`, adapted to the new `Models/`
package. The three Transformer variants share the `Transformer` folder under
`Predictions/`, with one CSV per covariate configuration:

    Predictions/Transformer/predictions_rudimentary.csv
    Predictions/Transformer/predictions_base.csv
    Predictions/Transformer/predictions_extended.csv

`compute_and_save_transformer_config(...)` runs the MC forecast from a
checkpoint and writes the CSV for one config. `main(...)` reads all three
CSVs and plots them against actuals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from panelclv.models import InferenceMultinomialTransformerModel
from panelclv.evaluation import (
    forecast_from_checkpoint,
    load_predictions_from_csv,
    metrics_table,
    plot_weekly_aggregated,
    save_predictions_to_csv,
    weekly_actuals,
)
# `holdout_actuals_NT` returns the per-customer (N, T_HOLD) array that
# `metrics_table` now expects — `weekly_actuals` keeps producing the (T_HOLD,)
# aggregate that `plot_weekly_aggregated` consumes.
from panelclv.evaluation.plot_utils import holdout_actuals_NT


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


PREDICTIONS_DIR = Path(__file__).resolve().parent / "Predictions"
TRANSFORMER_DIR = PREDICTIONS_DIR / "Transformer"

CONFIG_FILENAMES: dict[str, str] = {
    "rudimentary": "predictions_rudimentary.csv",
    "base": "predictions_base.csv",
    "extended": "predictions_extended.csv",
}


def csv_path_for(config: str) -> Path:
    fname = CONFIG_FILENAMES.get(config)
    if fname is None:
        raise KeyError(
            f"Unknown config {config!r}. Known: {list(CONFIG_FILENAMES)}"
        )
    return TRANSFORMER_DIR / fname


# ---------------------------------------------------------------------------
# Compute + save predictions for one covariate configuration
# ---------------------------------------------------------------------------


def compute_and_save_transformer_config(
    config: str,
    checkpoint_path: str | Path,
    *,
    calibration: Sequence[pd.DataFrame],
    holdout_calendar: pd.DataFrame,
    seq_cols: Sequence[str],
    input_spec: dict,
    target_col: str = "Transactions",
    seq_len: int | None = None,
    customer_ids: Sequence | None = None,
    d_model: int = 64,
    nhead: int = 4,
    num_encoder_layers: int = 2,
    dropout: float = 0.0,
    n_simulations: int = 30,
    batch_size: int = 256,
    device: str = "cuda",
    mode: str = "sample",
) -> Path:
    """Run MC from a checkpoint for one covariate configuration and save its CSV."""
    def factory():
        return InferenceMultinomialTransformerModel(
            seq_cols=seq_cols,
            input_spec=input_spec,
            target_col=target_col,
            seq_len=seq_len, d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers, dropout=dropout,
            mode=mode,
        )

    result = forecast_from_checkpoint(
        checkpoint_path=checkpoint_path,
        inference_model_factory=factory,
        calibration=calibration,
        holdout_calendar=holdout_calendar,
        seq_cols=seq_cols,
        target_col=target_col,
        model_type="transformer",
        n_simulations=n_simulations,
        batch_size=batch_size,
        device=device,
        mode=mode,
    )
    return save_predictions_to_csv(
        result["predictions"], csv_path_for(config), customer_ids=customer_ids,
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def load_all_config_predictions(
    holdout_length: int | None = None,
    configs: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    if configs is None:
        configs = list(CONFIG_FILENAMES.keys())

    out: dict[str, np.ndarray] = {}
    for cfg in configs:
        path = csv_path_for(cfg)
        if not path.exists():
            print(f"[warn] {path} not found — skipping config {cfg!r}.")
            continue
        arr, _ = load_predictions_from_csv(path, holdout_length=holdout_length)
        out[f"Transformer ({cfg})"] = arr
    return out


def main(
    holdout: Sequence[pd.DataFrame],
    train: Sequence[pd.DataFrame] | None = None,
    *,
    configs: Sequence[str] | None = None,
    title: str = "Weekly aggregated transactions across covariate configurations",
    target_col: str | None = None,
    count_col: str | int | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[Any, Any, pd.DataFrame]:
    """Load the three covariate-config CSVs, plot, and print metrics.

    Pass `train` (list of per-customer training-window DataFrames) to also
    draw the training period to the left of the holdout.

    Exactly one of `target_col=` (recommended; e.g. `"Transactions"`) or
    `count_col=` (legacy positional fallback) must be supplied — the old
    magic `count_col=3` default was removed because it produced silently
    wrong arrays on any non-default schema.
    """
    holdout_length = holdout[0].shape[0]
    predictions = load_all_config_predictions(
        holdout_length=holdout_length, configs=configs,
    )
    if not predictions:
        raise FileNotFoundError(
            f"No prediction CSVs found under {TRANSFORMER_DIR}. "
            f"Run compute_and_save_transformer_config(...) for each config first."
        )

    # Per-customer (N, T_HOLD) actuals are what the metric helper needs.
    # Sum-across-customers gives the (T_HOLD,) aggregate the plot expects.
    actuals_NT = holdout_actuals_NT(
        holdout, target_col=target_col, count_col=count_col,
    )
    actuals_agg = actuals_NT.sum(axis=0)
    train_actuals = (
        weekly_actuals(train, target_col=target_col, count_col=count_col)
        if train is not None else None
    )
    fig, ax = plot_weekly_aggregated(
        actuals_agg, predictions,
        train_actuals=train_actuals,
        title=title, save_path=save_path,
    )
    # `metrics_table` consumes the per-customer array so RMSE is the
    # individual RMSE the thesis reports (not the cheaper aggregate RMSE).
    table = metrics_table(actuals_NT, predictions)
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return fig, ax, table


# ---------------------------------------------------------------------------
# Script entry point — wire your own data loader
# ---------------------------------------------------------------------------


def _load_dataset(name: str):
    """Stub. Replace with your own data loader."""
    raise NotImplementedError(
        "Wire `_load_dataset` to your data loader before running "
        "main_plot_covar.py as a script, or import `main(holdout=...)` from a "
        "notebook."
    )


if __name__ == "__main__":
    data = _load_dataset("electronic")
    main(holdout=data["holdout"], save_path="./plots/main_plot_covar.png")
