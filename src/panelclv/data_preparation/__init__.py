"""Data-preparation package: customer-period panel -> model-ready tensors.

Modules:
  - ``dynamic_panel_dataset``  ``prepare_dataset(panel, ...)`` -> the model-ready
                               ``data`` dict (calibration/holdout/samples/targets/...).
  - ``ar_features``            autoregressive target-derived feature builders
                               (recency / frequency / tenure / rate).
  - ``period_calendar``        the one calendar-time <-> period-index conversion: the
                               week-numbering convention and the days-per-period table.
  - ``target_channel``         the one statement of where the target sits on the
                               feature axis, and the reads of that channel.
  - ``pareto_simulation``      synthetic Pareto/NBD panels with known ground truth.

(Building the raw customer-period panel itself now lives outside the package — see
``notebooks/archive/dataset_building.ipynb``.)

Marked as a real package (rather than relying on a ``sys.path`` hack) so it imports
cleanly after ``pip install -e .``.
"""
