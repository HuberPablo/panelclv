"""The customer-group set has one encoding: the predicates that define the groups.

A group exists exactly when a predicate defines it, so `CUSTOMER_GROUPS` is the
predicates' keys by construction and cannot drift from them — the drift that a
second, hand-written tuple in the study-suite reader made possible. The `"Other"`
catch-all is derived in the same place, so no caller re-derives it.

Run:  pytest -q tests/test_customer_groups.py
"""

import numpy as np
import pytest

from panelclv.evaluation import CUSTOMER_GROUPS, assign_customer_groups
from panelclv.evaluation.segment_analysis import _GROUP_PREDICATES

IDS = np.array([1, 2, 3, 4])


def _data():
    """Four customers with hand-chosen calibration / holdout counts.

    Counts are the target channel of the `(N, T, F)` tensors, summed over periods:
    calibration `[4, 0, 1, 2]` (mean 1.75), holdout `[0, 0, 3, 1]`. So customer 1 is
    At Risk (inactive in holdout, above-average calibration frequency), customer 3 is
    an Opportunity (more in holdout than in calibration), and 2 and 4 are neither.
    """
    calib = np.array([[4.0], [0.0], [1.0], [2.0]])[:, :, None]      # (4, 1, 1)
    hold = np.array([[0.0], [0.0], [3.0], [1.0]])[:, :, None]       # (4, 1, 1)
    return {
        "calibration": calib,
        "holdout": hold,
        "ids": IDS,
        "seq_cols": ["transactions"],
        "target_col": "transactions",
    }


def test_group_set_is_the_predicates_keys():
    """The set is not restated anywhere — it *is* what the predicates define."""
    assert CUSTOMER_GROUPS == tuple(_GROUP_PREDICATES)


def test_assign_defaults_to_the_whole_set():
    groups = assign_customer_groups(_data())
    assert list(groups) == list(CUSTOMER_GROUPS)
    assert groups["At Risk"].tolist() == [1]
    assert groups["Opportunity"].tolist() == [3]


def test_other_is_derived_and_covers_the_rest_of_the_cohort():
    """With the catch-all, the groups partition the cohort — nothing is dropped."""
    groups = assign_customer_groups(_data(), with_other=True)
    assert list(groups) == list(CUSTOMER_GROUPS) + ["Other"]
    assert groups["Other"].tolist() == [2, 4]
    assigned = sorted(int(cid) for ids in groups.values() for cid in ids)
    assert assigned == IDS.tolist()


def test_other_absorbs_customers_dropped_by_a_narrowed_request():
    """`groups=` narrows the predicates applied; "Other" is everyone they miss."""
    groups = assign_customer_groups(_data(), groups=("At Risk",), with_other=True)
    assert groups["Other"].tolist() == [2, 3, 4]      # the Opportunity is now "Other"


def test_unknown_group_names_the_available_ones():
    with pytest.raises(ValueError, match="unknown group"):
        assign_customer_groups(_data(), groups=("Churned",))
