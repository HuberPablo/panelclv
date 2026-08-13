"""Study-suite orchestration: run many Optuna studies across many models.

Specify a set of models and a single shared dataset, run ``n_studies_per_model``
independent Optuna studies per model, keep the best trial of each study, forecast
it, and archive everything under ``Studies/<study_name>/`` in a uniform,
analysis-ready layout. This subpackage holds only orchestration — it reuses
``panelclv.trials`` / ``tuning`` / ``models`` / ``benchmarks`` and adds no
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
                      training={"n_epochs": 150, "patience": 7}),
            ModelSpec(name="ParetoNBD", model_type="pareto_nbd"),
        ],
    )
    run_study_suite(config)

Reading a finished suite back is three modules, named for what they do — and all of
their entry points are exported here, so a caller imports this subpackage, never a
module:

- ``suite_reader`` — discovery, prediction loading, the across-studies aggregate, and
  rebuilding (and describing) the dataset a suite ran on from its persisted recipe.
- ``suite_metrics`` — re-scoring stored forecasts: per model, across suites, per
  customer group. It owns the package's one Student-t interval.
- ``suite_plots`` — the headline forecast figure, whose across-studies band is that
  same interval.

The Pareto/NBD grid study — a directory of suites, one per generated dataset — has
its own two readers, split on whether the analysis could ever run on a real panel:

- ``pareto_nbd_grid`` reads only what the suites stored, so it runs on any grid.
- ``synthetic_grid`` scores forecasts against the generation study itself — the panels
  it wrote and the latent truth the models never see (the true death week, the true
  seasonal curve) — so it runs on a generated grid and nothing else. Importing from it
  is the declaration that the analysis is synthetic-only.
"""

from .config import ModelSpec, StudySuiteConfig
from .runner import run_study_suite
from .suite_reader import (
    load_model_predictions,
    aggregate_suite_predictions,
    describe_dataset,
    describe_suite_dataset,
)
from .suite_plots import plot_suite_forecast
from .suite_metrics import (
    group_metrics_suite_table,
    study_metrics,
    compare_study_metrics,
)
from .pareto_nbd_grid import (
    collect_grid_results,
    group_summary,
    compare_models_table,
    plot_pattern,
    plot_diff_grid,
)
from .synthetic_grid import (
    seasonality_grid,
    alive_volume_ratio_grid,
    dead_volume_leakage_grid,
    dead_customer_mass,
    shape_correlation,
)

__all__ = [
    "run_study_suite",
    "StudySuiteConfig",
    "ModelSpec",
    "load_model_predictions",
    "aggregate_suite_predictions",
    "plot_suite_forecast",
    "group_metrics_suite_table",
    "study_metrics",
    "compare_study_metrics",
    "describe_dataset",
    "describe_suite_dataset",
    "collect_grid_results",
    "group_summary",
    "compare_models_table",
    "plot_pattern",
    "plot_diff_grid",
    "seasonality_grid",
    "alive_volume_ratio_grid",
    "dead_volume_leakage_grid",
    "dead_customer_mass",
    "shape_correlation",
]
