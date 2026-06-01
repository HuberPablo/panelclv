"""Data-preparation package: raw transactions -> customer-period panel -> model-ready tensors.

Modules:
  - ``dataset_building``        raw records -> a tidy customer-period panel.
  - ``dynamic_panel_dataset``  ``prepare_dataset(panel, ...)`` -> the model-ready
                               ``data`` dict (calibration/holdout/samples/targets/...).
  - ``ar_features`` / ``build_bank_panel``  supporting feature + panel builders.

Marked as a real package (rather than relying on a ``sys.path`` hack) so it imports
cleanly after ``pip install -e .``.
"""
