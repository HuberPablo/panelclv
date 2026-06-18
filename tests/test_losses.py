"""Tests for compute_class_weights (panelclv.models.losses).

Focus on the data-dict convenience overload added on top of the original
array-of-labels form: passing a ``prepare_dataset`` dict must (a) reproduce the
exact weights the old ``y.squeeze(-1)[:, :s-1]`` + ``max_trans`` boilerplate
produced, (b) auto-derive ``num_classes`` from the resolved target embedding,
(c) honour ``training_only`` (weight on the training-prefix periods only, so the
temporal validation window never leaks into the loss) and an explicit
``num_classes`` override, while (d) the legacy ``(labels, num_classes)`` calls
keep working and the nonsensical combinations raise clear errors.

The train/val split is temporal (a time window over all customers), so weighting
"train only" means slicing the AR axis to ``[:, :val_start_idx-1]`` — not a customer
subset. That replaced the old ``train_idx`` customer-index argument.

These are CPU-only and tiny (no model, no GPU).

Run:  pytest -q tests/test_losses.py
"""

import numpy as np
import pytest
import torch

from panelclv.models.losses import compute_class_weights


def _fake_data(seed=0, n=5, t=4, num_classes=3, val_start_idx=4):
    """A minimal prepare_dataset-shaped dict: (N, T-1, 1) float32 targets.

    ``val_start_idx`` (= s) is the temporal validation boundary; the training prefix
    is ``targets[:, :s-1]``. Defaulting s = t holds out the final transition so the
    train-only weighting differs from the all-periods weighting.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, num_classes, size=(n, t, 1)).astype(np.float32)
    return {
        "targets": y,
        "target_col": "Transactions",
        "embedded_cols": {"Transactions": num_classes},
        "val_start_idx": val_start_idx,
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


def test_dict_form_matches_training_prefix_boilerplate():
    """The dict overload reproduces the squeeze + prefix-slice + max_trans lookup."""
    data = _fake_data()
    s = data["val_start_idx"]

    w_dict = compute_class_weights(data)  # training_only=True by default

    y_arr = data["targets"].squeeze(-1).astype(np.int64)
    max_trans = data["embedded_cols"][data["target_col"]]
    w_arr = compute_class_weights(y_arr[:, : s - 1], num_classes=max_trans)

    torch.testing.assert_close(w_dict, w_arr)


def test_dict_form_infers_num_classes_from_embedding():
    """Without an explicit num_classes, the head size comes from embedded_cols."""
    data = _fake_data(num_classes=4)
    w = compute_class_weights(data)
    assert w.shape == (4,)  # inferred from embedded_cols[target_col] == 4


def test_training_only_changes_the_weighting():
    """Restricting to the training prefix generally differs from all-periods weights."""
    data = _fake_data(seed=1, n=8)
    w_all = compute_class_weights(data, training_only=False)
    w_train = compute_class_weights(data, training_only=True)
    assert not torch.allclose(w_all, w_train)


def test_explicit_num_classes_overrides_inference():
    """A passed num_classes wins over the embedding-derived default."""
    data = _fake_data(num_classes=3)
    w = compute_class_weights(data, num_classes=5)
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


def test_training_only_on_array_is_ignored():
    """training_only has no meaning for the array form; it is silently ignored."""
    w = compute_class_weights(np.array([0, 1, 2]), 3, training_only=True)
    assert w.shape == (3,)


def test_dict_training_only_without_val_start_idx_raises():
    """training_only=True needs val_start_idx (set by prepare_dataset)."""
    data = {"targets": np.zeros((2, 3, 1), dtype=np.float32),
            "target_col": "Transactions",
            "embedded_cols": {"Transactions": 3}}
    with pytest.raises(KeyError):
        compute_class_weights(data)  # training_only=True by default, no val_start_idx


def test_dict_without_inferable_num_classes_raises():
    """Target not in embedded_cols -> can't infer the head size."""
    data = {"targets": np.zeros((2, 3, 1), dtype=np.float32),
            "target_col": "Transactions",
            "embedded_cols": {},
            "val_start_idx": 3}
    with pytest.raises(ValueError):
        compute_class_weights(data)


def test_dict_missing_targets_key_raises():
    with pytest.raises(KeyError):
        compute_class_weights({"target_col": "Transactions"})
