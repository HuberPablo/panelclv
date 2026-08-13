"""Prediction I/O: the one on-disk layout a forecast is written in and read back from.

``prediction_csv`` holds the wide per-customer CSV (``id_col`` +
``week_0..week_{T-1}``) that every model in the package writes and everything that
scores or plots a stored forecast reads.

This subpackage is a **leaf**: it imports nothing from ``panelclv``, which is what
lets the model layer write predictions without naming anything above it (ADR-0002).
"""

from .prediction_csv import (
    load_predictions_from_csv,
    reduce_to_customer_period,
    save_predictions_to_csv,
)

__all__ = [
    "save_predictions_to_csv",
    "load_predictions_from_csv",
    "reduce_to_customer_period",
]
