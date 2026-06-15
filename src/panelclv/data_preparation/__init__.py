"""Data-preparation package: customer-period panel -> model-ready tensors.

Modules:
  - ``dynamic_panel_dataset``  ``prepare_dataset(panel, ...)`` -> the model-ready
                               ``data`` dict (calibration/holdout/samples/targets/...).
  - ``ar_features``            autoregressive target-derived feature builders
                               (recency / frequency / tenure / rate).

(Building the raw customer-period panel itself now lives outside the package — see
``notebooks/dataset_building.ipynb``.)

Marked as a real package (rather than relying on a ``sys.path`` hack) so it imports
cleanly after ``pip install -e .``.
"""
