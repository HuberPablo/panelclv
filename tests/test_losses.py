"""Tests for compute_class_weights (panelclv.models.losses).

Focus on the data-dict convenience overload added on top of the original
array-of-labels form: passing a ``prepare_dataset`` dict must (a) reproduce the
exact weights the old ``y.squeeze(-1)[:, :s-1]`` + ``num_target_classes`` boilerplate
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

from panelclv.models.losses import (
    _PROPER_LOSS_TYPES,
    build_criterion,
    compute_class_weights,
)


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
    """The dict overload reproduces the squeeze + prefix-slice + class-count lookup."""
    data = _fake_data()
    s = data["val_start_idx"]

    w_dict = compute_class_weights(data)  # training_only=True by default

    y_arr = data["targets"].squeeze(-1).astype(np.int64)
    num_target_classes = data["embedded_cols"][data["target_col"]]
    w_arr = compute_class_weights(y_arr[:, : s - 1], num_classes=num_target_classes)

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


# ---------------------------------------------------------------------------
# The criterions themselves. `compute_class_weights` feeds these, but until this
# section nothing in the suite ever called `FocalLoss.forward` or
# `SquaredEMDLoss.forward` — two of the four `loss_type` values a study may be
# configured with went straight from `build_criterion` to production untested.
# ---------------------------------------------------------------------------

LOSS_TYPES = ("cross_entropy", "weighted_ce", "focal", "emd", "ce_emd")


def _logits_and_targets(n=6, num_classes=4, seed=0):
    """A flat (N, K) logit batch and its (N,) class-index targets.

    Flat rather than (B, T, K) because that is the shape `fit_model` reshapes to
    before calling the criterion.
    """
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, num_classes, generator=g)
    targets = torch.randint(0, num_classes, (n,), generator=g)
    return logits, targets


@pytest.mark.parametrize("loss_type", LOSS_TYPES)
def test_every_loss_type_returns_a_finite_scalar_that_backpropagates(loss_type):
    """Each configurable loss runs forward and backward on the training shape.

    Every one of these is reachable from a `ModelSpec`'s `training={"loss_type": ...}`,
    so a criterion that raises, returns a non-scalar, or produces no gradient breaks a
    study only once someone selects it — after the run has started.
    """
    logits, targets = _logits_and_targets()
    logits.requires_grad_(True)
    # Weights go only to the two losses that consume them: `build_criterion`
    # rejects them alongside a strictly proper loss rather than ignoring them.
    weights = (
        torch.ones(logits.shape[-1])
        if loss_type in ("weighted_ce", "focal")
        else None
    )

    criterion = build_criterion(loss_type, class_weights=weights, focal_gamma=2.0)
    loss = criterion(logits, targets)

    assert loss.ndim == 0, f"{loss_type} returned a non-scalar of shape {loss.shape}"
    assert torch.isfinite(loss), f"{loss_type} returned {loss.item()}"
    assert loss.item() >= 0.0

    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_focal_gamma_zero_is_weighted_cross_entropy():
    """`gamma=0` collapses the focal term to 1, so focal reduces to CE.

    The identity the class docstring claims. It pins that the `(1 - p_t)^gamma` factor
    is applied to the CE term rather than replacing it — the mistake that would make
    focal silently a different loss at every gamma.
    """
    logits, targets = _logits_and_targets()

    focal = build_criterion("focal", focal_gamma=0.0)(logits, targets)
    ce = build_criterion("cross_entropy")(logits, targets)

    assert focal.item() == pytest.approx(ce.item(), rel=1e-6)


def test_emd_punishes_a_distant_class_more_than_a_neighbour():
    """The ordinal property EMD exists for, and the reason it is not just CE.

    Cross-entropy scores a confident wrong answer identically wherever the truth sits,
    because it reads only the true class's probability. EMD compares CDFs, so predicting
    class 0 when the answer is 3 must cost more than predicting 0 when the answer is 1.
    """
    logits = torch.tensor([[8.0, 0.0, 0.0, 0.0]])      # confidently class 0
    emd = build_criterion("emd")

    near = emd(logits, torch.tensor([1]))
    far = emd(logits, torch.tensor([3]))
    assert far.item() > near.item()

    # The contrast that makes the point: CE cannot tell these two apart.
    ce = build_criterion("cross_entropy")
    assert ce(logits, torch.tensor([1])).item() == pytest.approx(
        ce(logits, torch.tensor([3])).item(), rel=1e-6
    )


def test_ce_emd_at_zero_weight_is_exactly_cross_entropy():
    """`emd_weight=0` recovers the default loss bit for bit.

    This is what makes 0 a safe left endpoint for a searched lambda: the arm cannot lose
    to the `cross_entropy` baseline except through search noise, and a search that keeps
    choosing 0 has answered the question rather than failed. Exact equality, not
    approximate — the EMD term is multiplied out, never merely made small.
    """
    logits, targets = _logits_and_targets()

    composite = build_criterion("ce_emd", emd_weight=0.0)(logits, targets)
    ce = build_criterion("cross_entropy")(logits, targets)

    assert torch.equal(composite, ce)


def test_ce_emd_is_the_sum_of_its_two_terms():
    """The composite is CE + lambda*EMD, with lambda scaling only the EMD half.

    Pins the arithmetic the properness argument rests on: a non-negative combination of
    two strictly proper rules. A lambda that also scaled the CE term would still look
    sensible in a loss curve while being a different (and rescaled) objective.
    """
    logits, targets = _logits_and_targets()
    ce = build_criterion("cross_entropy")(logits, targets)
    emd = build_criterion("emd")(logits, targets)

    for lam in (0.5, 1.0, 10.0):
        composite = build_criterion("ce_emd", emd_weight=lam)(logits, targets)
        assert composite.item() == pytest.approx(ce.item() + lam * emd.item(), rel=1e-6)


def test_ce_emd_rejects_a_negative_weight():
    """A negative lambda breaks the properness the loss is chosen for, so it raises."""
    with pytest.raises(ValueError, match="emd_weight"):
        build_criterion("ce_emd", emd_weight=-1.0)


@pytest.mark.parametrize("loss_type", sorted(_PROPER_LOSS_TYPES))
def test_a_proper_loss_refuses_class_weights(loss_type):
    """Weighting a strictly proper loss is rejected, not silently dropped.

    Inverse-frequency weighting moves the cross-entropy minimiser to the uniform
    distribution over the K classes, and the rollout samples from that — a mean of
    2.0 against a true 0.0598 on CDNOW (`docs/loss-functions.md` 5.2). Dropping the
    weights instead would leave a study comparing `weighted_ce` against `ce_emd`
    confounded on two axes at once, weighted-vs-unweighted as well as the loss
    itself, with nothing in the results to show it.
    """
    weights = torch.ones(4)

    with pytest.raises(ValueError, match="strictly proper"):
        build_criterion(loss_type, class_weights=weights)


def test_a_proper_loss_is_fine_without_class_weights():
    """The guard triggers on the weights, not on the loss type."""
    for loss_type in sorted(_PROPER_LOSS_TYPES):
        assert build_criterion(loss_type) is not None


def test_an_unknown_loss_type_is_rejected_by_name():
    """A typo in `training={"loss_type": ...}` fails at build, not mid-epoch."""
    with pytest.raises(ValueError, match="loss_type"):
        build_criterion("mse")
