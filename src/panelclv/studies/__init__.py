"""Study-suite orchestration: run many Optuna studies across many models.

Specify a set of models and a single shared dataset, run ``n_studies_per_model``
independent Optuna studies per model, keep the best trial of each study, forecast
it, and archive everything under ``Studies/<study_name>/`` in a uniform,
analysis-ready layout. This subpackage holds only orchestration — it reuses
``panelclv.experiments`` / ``tuning`` / ``models`` / ``benchmarks`` and adds no
modeling logic.

Typical use (see ``scripts/run_studies.py`` for a full example)::

    from panelclv.studies import run_study_suite, StudySuiteConfig, ModelSpec

    config = StudySuiteConfig(
        studies_base_path="/path/to/Studies",
        study_name="electronics_2026_06",
        data=data_full,                       # one prepare_dataset dict, shared
        n_studies_per_model=5,
        models=[
            ModelSpec(name="LSTM", model_type="lstm", n_trials=30,
                      data_info={"n_epochs": 150, "patience": 7}),
            ModelSpec(name="ParetoNBD", model_type="pareto_nbd"),
        ],
    )
    run_study_suite(config)
"""

from .config import ModelSpec, StudySuiteConfig
from .runner import run_study_suite
from .analysis import (
    load_model_predictions,
    aggregate_suite_predictions,
    plot_suite_forecast,
    group_metrics_suite_table,
    group_metrics_suite_distribution,
    study_metrics,
    compare_study_metrics,
    describe_dataset,
    describe_suite_dataset,
)
from .pnbd_grid import (
    collect_grid_results,
    seasonality_grid,
    alive_volume_ratio_grid,
    dead_volume_leakage_grid,
    group_summary,
    compare_models_table,
    plot_pattern,
    plot_diff_grid,
)

__all__ = [
    "run_study_suite",
    "StudySuiteConfig",
    "ModelSpec",
    "load_model_predictions",
    "aggregate_suite_predictions",
    "plot_suite_forecast",
    "group_metrics_suite_table",
    "group_metrics_suite_distribution",
    "study_metrics",
    "compare_study_metrics",
    "describe_dataset",
    "describe_suite_dataset",
    "collect_grid_results",
    "seasonality_grid",
    "alive_volume_ratio_grid",
    "dead_volume_leakage_grid",
    "group_summary",
    "compare_models_table",
    "plot_pattern",
    "plot_diff_grid",
]
