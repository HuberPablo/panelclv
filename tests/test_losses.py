"""Tests for compute_class_weights (panelclv.models.losses).

Focus on the data-dict convenience overload added on top of the original
array-of-labels form: passing a ``prepare_dataset`` dict must (a) reproduce the
exact weights the old ``y.squeeze(-1)[train_idx]`` + ``max_trans`` boilerplate
produced, (b) auto-derive ``num_classes`` from the resolved target embedding,
(c) honour ``train_idx`` (train-only weighting, no val leak) and an explicit
``num_classes`` override, while (d) the legacy ``(labels, num_classes)`` calls
keep working and the nonsensical combinations raise clear errors.

These are CPU-only and tiny (no model, no GPU).

Run:  pytest -q tests/test_losses.py
"""

import numpy as np
import pytest
import torch

from panelclv.models.losses import compute_class_weights


def _fake_data(seed=0, n=5, t=4, num_classes=3):
    """A minimal prepare_dataset-shaped dict: (N, T-1, 1) float32 targets."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, num_classes, size=(n, t, 1)).astype(np.float32)
    return {
        "targets": y,
        "target_col": "Transactions",
        "input_spec": {"embedded_cols": {"Transactions": num_classes}},
    }


def test_weights_normalise_to_num_classes_and_are_inverse_frequency():
    """Average weight is 1 (sum == num_classes) and rarer classes weigh more."""
    # Class 0 appears 3x, class 1 once, class 2 twice -> w0 < w2 < w1.
    labels = np.array([0, 0, 0, 1, 2, 2], dtype=np.int64)
    w = compute_class_weights(labels, num_classes=3)
    assert w.shape == (3,)
    assert float(w.sum()) == pytest.approx(3.0)
    assert w[0] < w[2] < w[1]


def test_absent_class_gets_finite_weight():
    """A class with zero observations is clamped to count 1, not inf/nan."""
    labels = np.array([0, 0, 1, 1], dtype=np.int64)  # class 2 never appears
    w = compute_class_weights(labels, num_classes=3)
    assert torch.isfinite(w).all()
    assert w[2] > 0


def test_dict_form_matches_legacy_array_boilerplate():
    """The dict overload reproduces the squeeze + max_trans lookup exactly."""
    data = _fake_data()
    train_idx = [0, 1, 2, 3]  # hold customer 4 out

    w_dict = compute_class_weights(data, train_idx=train_idx)

    y_arr = data["targets"].squeeze(-1).astype(np.int64)
    max_trans = data["input_spec"]["embedded_cols"][data["target_col"]]
    w_arr = compute_class_weights(y_arr[train_idx], num_classes=max_trans)

    torch.testing.assert_close(w_dict, w_arr)


def test_dict_form_infers_num_classes_from_embedding():
    """Without an explicit num_classes, the head size comes from input_spec."""
    data = _fake_data(num_classes=4)
    w = compute_class_weights(data)
    assert w.shape == (4,)  # inferred from embedded_cols[target_col] == 4


def test_train_idx_changes_the_weighting():
    """Restricting to train_idx generally yields different weights than all rows."""
    data = _fake_data(seed=1, n=8)
    w_all = compute_class_weights(data)
    w_train = compute_class_weights(data, train_idx=list(range(6)))
    assert not torch.allclose(w_all, w_train)


def test_explicit_num_classes_overrides_inference():
    """A passed num_classes wins over the embedding-derived default."""
    data = _fake_data(num_classes=3)
    w = compute_class_weights(data, num_classes=5, train_idx=[0, 1, 2])
    assert w.shape == (5,)


def test_legacy_positional_call_still_works():
    """Existing callers passing (labels, num_classes) positionally are unaffected."""
    labels = np.array([0, 1, 2, 2], dtype=np.int64)
    w = compute_class_weights(labels, 3)
    assert w.shape == (3,)
    assert float(w.sum()) == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Error paths
# --------------------------------------------------------------------------- #
def test_array_without_num_classes_raises():
    with pytest.raises(ValueError):
        compute_class_weights(np.array([0, 1, 2]))


def test_train_idx_on_array_raises():
    with pytest.raises(TypeError):
        compute_class_weights(np.array([0, 1, 2]), 3, train_idx=[0])


def test_dict_without_inferable_num_classes_raises():
    """Target not in embedded_cols -> can't infer the head size."""
    data = {"targets": np.zeros((2, 3, 1), dtype=np.float32),
            "target_col": "Transactions",
            "input_spec": {"embedded_cols": {}}}
    with pytest.raises(ValueError):
        compute_class_weights(data)


def test_dict_missing_targets_key_raises():
    with pytest.raises(KeyError):
        compute_class_weights({"target_col": "Transactions"})
