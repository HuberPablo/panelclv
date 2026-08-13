"""Prediction I/O: the one on-disk layout a forecast is written in and read back from.

``prediction_csv`` holds the wide per-customer CSV (``id_col`` +
``week_0..week_{T-1}``) that every model in the package writes and everything that
scores or plots a stored forecast reads. It also holds ``DEFAULT_ID_COL``, the one
spelling of the customer-key column every writer falls back to when the dataset it
was handed does not name one.

This subpackage is a **leaf**: it imports nothing from ``panelclv``, which is what
lets the model layer write predictions without naming anything above it (ADR-0002).
"""

from .prediction_csv import (
    DEFAULT_ID_COL,
    load_predictions_from_csv,
    reduce_to_customer_period,
    save_predictions_to_csv,
)

__all__ = [
    "DEFAULT_ID_COL",
    "save_predictions_to_csv",
    "load_predictions_from_csv",
    "reduce_to_customer_period",
]
