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
from .experiment_utils import (
    make_loaders,
    make_data_builder,
    build_inference_from_trial,
)

# `__all__` is the curated *headline* surface -- the ~16 entry points a user needs to
# run the canonical workflow (prepare data -> build loaders -> tune -> rebuild winner
# -> Monte Carlo forecast -> report). Everything imported above stays importable by
# explicit name; only the advertised set (`from Models import *`, autocompletion, docs)
# is trimmed. Internals deliberately kept OFF this list but still importable:
#   train_one_epoch, validate_one_epoch, FitResult, FocalLoss, SquaredEMDLoss,
#   compute_class_weights, build_criterion, compute_metrics, rmse, mae,
#   mape_positive, aggregate_bias, select_features, select_features_for_trial,
#   weekly_aggregate_rollout_metrics, validate_removable_features, weekly_actuals,
#   holdout_actuals_NT, weekly_aggregate_predictions, alignment_check,
#   forecast_from_checkpoint, save_predictions_to_csv, load_predictions_from_csv,
#   mc_simulate_one_path, mc_simulate_transformer_path.
__all__ = [
    # Models (training + inference wrappers, both families)
    "MultinomialLSTMModel",
    "InferenceMultinomialLSTMModel",
    "MultinomialTransformerModel",
    "InferenceMultinomialTransformerModel",
    # Training + tuning
    "fit_model",
    "run_optuna_study",
    # Orchestration helpers (the thin glue notebooks call)
    "make_loaders",
    "make_data_builder",
    "build_inference_from_trial",
    # Forecasting (autoregressive Monte Carlo simulator + its metrics)
    "mc_forecast",
    "mc_forecast_transformer",
    "mc_compute_metrics",
    # Pareto/NBD benchmarks (MLE + hierarchical-Bayes)
    "compute_pareto_predictions",
    "compute_pareto_paper_predictions",
    # Reporting
    "plot_weekly_aggregated",
    "metrics_table",
]
