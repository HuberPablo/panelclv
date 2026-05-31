"""Multinomial LSTM / Transformer baselines for transaction-count forecasting."""

from .multinomial_lstm import (
    MultinomialLSTMModel,
    InferenceMultinomialLSTMModel,
)
from .multinomial_transformer import (
    MultinomialTransformerModel,
    InferenceMultinomialTransformerModel,
)
from .training_utils import fit_model, train_one_epoch, validate_one_epoch, FitResult
from .losses import (
    FocalLoss,
    SquaredEMDLoss,
    compute_class_weights,
    build_criterion,
)
from .evaluation_utils import (
    compute_metrics,
    rmse,
    mae,
    mape_positive,
    aggregate_bias,
)
from .optuna_tuning import (
    run_optuna_study,
    select_features,
    select_features_for_trial,
    weekly_aggregate_rollout_metrics,
    validate_removable_features,
)
from .plot_utils import (
    weekly_actuals,
    holdout_actuals_NT,
    weekly_aggregate_predictions,
    plot_weekly_aggregated,
    metrics_table,
    alignment_check,
    forecast_from_checkpoint,
    save_predictions_to_csv,
    load_predictions_from_csv,
)
from .pareto_nbd import compute_pareto_predictions
from .pareto_paper import compute_pareto_paper_predictions
from .monte_carlo_forecasting import (
    run_monte_carlo_forecast as mc_forecast,
    run_monte_carlo_forecast_transformer as mc_forecast_transformer,
    simulate_one_path as mc_simulate_one_path,
    simulate_transformer_path as mc_simulate_transformer_path,
    compute_forecast_metrics as mc_compute_metrics,
)

__all__ = [
    "MultinomialLSTMModel",
    "InferenceMultinomialLSTMModel",
    "MultinomialTransformerModel",
    "InferenceMultinomialTransformerModel",
    "fit_model",
    "train_one_epoch",
    "validate_one_epoch",
    "FitResult",
    "FocalLoss",
    "SquaredEMDLoss",
    "compute_class_weights",
    "build_criterion",
    "compute_metrics",
    "rmse",
    "mae",
    "mape_positive",
    "aggregate_bias",
    "run_optuna_study",
    "select_features",
    "select_features_for_trial",
    "weekly_aggregate_rollout_metrics",
    "validate_removable_features",
    "weekly_actuals",
    "holdout_actuals_NT",
    "weekly_aggregate_predictions",
    "plot_weekly_aggregated",
    "metrics_table",
    "alignment_check",
    "forecast_from_checkpoint",
    "save_predictions_to_csv",
    "load_predictions_from_csv",
    "compute_pareto_predictions",
    "compute_pareto_paper_predictions",
    "mc_forecast",
    "mc_forecast_transformer",
    "mc_simulate_one_path",
    "mc_simulate_transformer_path",
    "mc_compute_metrics",
]
