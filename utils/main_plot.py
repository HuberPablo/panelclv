"""Plot weekly aggregated transactions: actual vs Transformer / LSTM / Pareto-NBD.

Counterpart to the old `main_plot.py`, adapted to the new `Models/` package.
Every `compute_and_save_*` call writes a fresh timestamped CSV under each
model's folder, so previous runs are preserved as an archive:

    Predictions/Transformer/transformer_pred_DD_MM_HH_MM.csv
    Predictions/LSTM/lstm_pred_DD_MM_HH_MM.csv
    Predictions/Pareto NBD/pareto_nbd_pred_DD_MM_HH_MM.csv

The plot step reads the most-recent CSV (by mtime) in each model's folder.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from Models import (
    InferenceMultinomialLSTMModel,
    InferenceMultinomialTransformerModel,
    compute_pareto_predictions,
    forecast_from_checkpoint,
    load_predictions_from_csv,
    metrics_table,
    plot_weekly_aggregated,
    save_predictions_to_csv,
    weekly_actuals,
)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


PREDICTIONS_DIR = Path(__file__).resolve().parent / "Predictions"

MODEL_FOLDERS: dict[str, str] = {
    "Transformer": "Transformer",
    "LSTM": "LSTM",
    "Pareto/NBD": "Pareto NBD",
}

MODEL_PREFIXES: dict[str, str] = {
    "Transformer": "transformer",
    "LSTM": "lstm",
    "Pareto/NBD": "pareto_nbd",
}


def _model_dir(model_name: str) -> Path:
    folder = MODEL_FOLDERS.get(model_name)
    if folder is None:
        raise KeyError(
            f"Unknown model {model_name!r}. Known: {list(MODEL_FOLDERS)}"
        )
    return PREDICTIONS_DIR / folder


def new_prediction_path(model_name: str) -> Path:
    """Fresh timestamped CSV path: `<folder>/<prefix>_pred_DD_MM_HH_MM.csv`."""
    prefix = MODEL_PREFIXES[model_name]
    stamp = datetime.now().strftime("%d_%m_%H_%M")
    return _model_dir(model_name) / f"{prefix}_pred_{stamp}.csv"


def latest_prediction_path(model_name: str) -> Path | None:
    """Most-recent (by mtime) prediction CSV for `model_name`, or None."""
    folder = _model_dir(model_name)
    if not folder.exists():
        return None
    prefix = MODEL_PREFIXES[model_name]
    candidates = sorted(
        folder.glob(f"{prefix}_pred_*.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Compute + save predictions from a trained checkpoint
# ---------------------------------------------------------------------------


def compute_and_save_lstm(
    checkpoint_path: str | Path,
    *,
    data: dict[str, Any],
    input_spec: dict[str, Any],
    hidden_dim: int = 128,
    memory_units: int = 64,
    dense_units: int = 64,
    dropout: float = 0.0,
    n_simulations: int = 30,
    device: str | None = None,
) -> Path:
    """Build the inference LSTM, run MC on `data`, save predictions to a CSV."""
    def factory():
        return InferenceMultinomialLSTMModel(
            seq_cols=data["seq_cols"],
            input_spec=input_spec,
            target_col=data["target_col"],
            hidden_dim=hidden_dim,
            memory_units=memory_units,
            dense_units=dense_units,
            dropout=dropout,
            mode="sample",
        )

    forecast = forecast_from_checkpoint(
        checkpoint_path, factory, data,
        n_simulations=n_simulations, device=device,
    )
    return save_predictions_to_csv(
        forecast["prediction_mean"], new_prediction_path("LSTM"),
        customer_ids=data["ids"],
    )


def compute_and_save_transformer(
    checkpoint_path: str | Path,
    *,
    data: dict[str, Any],
    input_spec: dict[str, Any],
    d_model: int = 64,
    nhead: int = 4,
    num_encoder_layers: int = 2,
    dropout: float = 0.0,
    n_simulations: int = 30,
    device: str | None = None,
) -> Path:
    """Build the inference Transformer, run MC on `data`, save predictions to a CSV."""
    def factory():
        return InferenceMultinomialTransformerModel(
            seq_cols=data["seq_cols"],
            input_spec=input_spec,
            target_col=data["target_col"],
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dropout=dropout,
            mode="sample",
        )

    forecast = forecast_from_checkpoint(
        checkpoint_path, factory, data,
        n_simulations=n_simulations, device=device,
    )
    return save_predictions_to_csv(
        forecast["prediction_mean"], new_prediction_path("Transformer"),
        customer_ids=data["ids"],
    )


def compute_and_save_pareto_nbd(
    *,
    data: dict[str, Any],
    id_col: str = "Id",
    time_col: str = "period_start",
    period_in_days: float = 7.0,
    penalizer_coef: float = 0.01,
) -> Path:
    """Fit Pareto/NBD on `data["train_panel"]` and save the per-period forecast."""
    preds, ids = compute_pareto_predictions(
        train_panel=data["train_panel"],
        holdout_length=data["T_HOLD"],
        id_col=id_col,
        target_col=data["target_col"],
        time_col=time_col,
        period_in_days=period_in_days,
        penalizer_coef=penalizer_coef,
        customer_ids=data["ids"],
    )
    return save_predictions_to_csv(
        preds, new_prediction_path("Pareto/NBD"), customer_ids=ids,
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def load_all_predictions(
    holdout_length: int | None = None,
    models: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Read the latest timestamped CSV in each model folder.

    Models with no matching CSV are skipped with a warning rather than
    failing, so the plot still works if (say) the Pareto/NBD file isn't
    present yet. Prints which file was loaded for each model so it's clear
    which run is being plotted.
    """
    if models is None:
        models = list(MODEL_FOLDERS.keys())

    out: dict[str, np.ndarray] = {}
    for name in models:
        path = latest_prediction_path(name)
        if path is None:
            print(f"[warn] no prediction CSV under {_model_dir(name)} — skipping {name}.")
            continue
        print(f"[load] {name} ← {path.name}")
        arr, _ = load_predictions_from_csv(path, holdout_length=holdout_length)
        out[name] = arr
    return out


def main(
    holdout: Sequence[pd.DataFrame],
    train: Sequence[pd.DataFrame] | None = None,
    *,
    models: Sequence[str] | None = None,
    title: str = "Weekly aggregated transactions: actual vs predicted",
    count_col: str | int = 3,
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[Any, Any, pd.DataFrame]:
    """Load the latest CSV per model, plot, and score.

    For each model in `models` (defaults to all three), the most-recent
    timestamped CSV in `Predictions/<folder>/` is loaded. Pass `train`
    (list of per-customer training-window DataFrames) to also draw the
    training period to the left of the holdout, with a dashed boundary line.
    """
    holdout_length = holdout[0].shape[0]
    predictions = load_all_predictions(holdout_length=holdout_length, models=models)
    if not predictions:
        raise FileNotFoundError(
            f"No prediction CSVs found under {PREDICTIONS_DIR}. "
            f"Run compute_and_save_lstm / compute_and_save_transformer / "
            f"compute_and_save_pareto_nbd first."
        )

    actuals = weekly_actuals(holdout, count_col=count_col)
    train_actuals = weekly_actuals(train, count_col=count_col) if train is not None else None
    fig, ax = plot_weekly_aggregated(
        actuals, predictions,
        train_actuals=train_actuals,
        title=title, save_path=save_path,
    )
    table = metrics_table(actuals, predictions)
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return fig, ax, table


# ---------------------------------------------------------------------------
# Script entry point — wire your own data loader
# ---------------------------------------------------------------------------


def _load_dataset(name: str):
    """Stub. Replace with your own data loader, e.g.:

        from scripts.data.data_loader import load_data
        return load_data(name)
    """
    raise NotImplementedError(
        "Wire `_load_dataset` to your data loader before running main_plot.py "
        "as a script, or import `main(holdout=...)` from a notebook."
    )


if __name__ == "__main__":
    data = _load_dataset("apparel")
    main(holdout=data["holdout"], save_path="./plots/main_plot.png")
