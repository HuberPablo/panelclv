"""panelclv — modular customer-base forecasting.

Multinomial LSTM / Transformer classifiers plus Pareto/NBD benchmarks for
per-customer transaction-count prediction over a holdout window.

The package is organised by *altitude* into importable subpackages, so each holds
a single concern:

- ``panelclv.models`` — model definitions only: the LSTM/Transformer architectures,
  their loss functions, and the autoregressive Monte Carlo simulator (the model's
  forecast mechanism).
- ``panelclv.training`` — the training loop (``fit_model`` and friends).
- ``panelclv.tuning`` — Optuna architecture / covariate-subset search.
- ``panelclv.evaluation`` — metrics, plotting, forecast diagnostics, prediction CSV I/O.
- ``panelclv.benchmarks`` — the non-neural Pareto/NBD comparators (MLE + hierarchical Bayes).
- ``panelclv.experiments`` — thin prepare -> tune -> forecast orchestration glue.
- ``panelclv.data_preparation`` — panel/dataset construction and the leak-free
  autoregressive feature engineering used by the simulator.
- ``panelclv.configs`` — schema-driven input/transformation specs and the
  ``PanelConfig`` dataclass.

Submodules are imported explicitly (e.g. ``from panelclv.tuning import
run_optuna_study``) rather than eagerly here, so importing ``panelclv`` itself
stays cheap and does not pull in torch until a model module is actually used.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
