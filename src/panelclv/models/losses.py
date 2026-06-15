"""Alternative losses for the multinomial baselines.

All criterions take (logits, targets) — same call signature as
`nn.CrossEntropyLoss`, with shapes `(B*T, K)` and `(B*T,)` (long indices) —
so they're drop-in replacements inside `training_utils.fit_model`.

Selectable via `loss_type` strings:

    "cross_entropy"  plain CE (default)
    "weighted_ce"    CE with per-class weights (inverse-frequency, etc.)
    "focal"          Focal Loss with optional class weights (`alpha`)
    "emd"            Squared Earth Mover's Distance — ordinal-aware

References
----------
- Lin, Goyal, Girshick, He, Dollár (2017),
  "Focal Loss for Dense Object Detection", ICCV.
- Hou, Yu, Samaras (2016),
  "Squared Earth Mover's Distance-based Loss for Training Deep Neural Networks".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Multi-class focal loss: `(1 - p_t)^gamma * CE`.

    `alpha` (optional) is a per-class weight tensor of shape (num_classes,) —
    typically inverse-frequency weights from `compute_class_weights(...)`.
    `gamma=0` reduces to plain (optionally weighted) CE.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
        pt = log_pt.exp()
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()


# ---------------------------------------------------------------------------
# Squared EMD (ordinal-aware)
# ---------------------------------------------------------------------------


class SquaredEMDLoss(nn.Module):
    """Squared Earth Mover's Distance between predicted and true CDFs.

    For ordinal classes `0 < 1 < ... < K-1`, penalises a wrong prediction
    by the squared L1 distance between cumulative distributions — so
    "predict 0 when actual is 10" is much worse than "predict 0 when
    actual is 1". Same input/output signature as CE.
    """

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        K = logits.shape[-1]
        probs = F.softmax(logits, dim=-1)
        target_onehot = F.one_hot(targets, num_classes=K).float()
        cdf_pred = probs.cumsum(dim=-1)
        cdf_true = target_onehot.cumsum(dim=-1)
        return ((cdf_pred - cdf_true) ** 2).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Class-weight helper
# ---------------------------------------------------------------------------


def compute_class_weights(
    targets,
    num_classes: int | None = None,
    *,
    training_only: bool = True,
) -> torch.Tensor:
    """Inverse-frequency per-class weights, normalised to sum to `num_classes`.

    `targets` may be given two ways:

    * an **array-like of integer class labels** (the original form) — then
      `num_classes` is required; or
    * a **`prepare_dataset` data dict** — then the labels are read from
      `data["targets"]` and, unless `num_classes` is passed, the class count is
      derived from the resolved target embedding
      (`data["input_spec"]["embedded_cols"][data["target_col"]]`, i.e. the same
      `max_trans` the softmax head uses). This folds the old notebook
      boilerplate (squeeze the target axis, look up `max_trans`) into one call.

    The train/val split is **temporal** (a time window over all customers), so with
    the dict form `training_only=True` (default) weights on the **training prefix**
    only — transitions whose target falls before `validation_start`, i.e.
    `targets[:, :val_start_idx-1]` — so the held-out validation periods' class mix
    never leaks into the loss. Pass `training_only=False` to use every period.
    (`training_only` is only meaningful with the dict form.)

    Classes absent from the labels get a count of 1 (so their weight stays
    finite). The normalisation keeps the average weight at 1, so the loss scale
    stays comparable to plain CE.
    """
    # Convenience overload: unpack a prepare_dataset data dict into its labels
    # (+ a default num_classes) using the dict's documented keys.
    if isinstance(targets, dict):
        data = targets
        if "targets" not in data:
            raise KeyError(
                "compute_class_weights got a dict without a 'targets' key; pass "
                "a prepare_dataset data dict, or an array of class labels."
            )
        # (N, T-1, 1) float32 -> (N, T-1); long() happens below with the array path.
        labels = torch.as_tensor(data["targets"]).squeeze(-1)
        if training_only:
            # Keep only the training-prefix transitions (target period < validation
            # window). val_start_idx = s ⇒ training target indices 0..s-2 = [:, :s-1].
            s = data.get("val_start_idx")
            if s is None:
                raise KeyError(
                    "compute_class_weights(training_only=True) needs data['val_start_idx'] "
                    "(set by prepare_dataset from validation_start). Pass training_only=False "
                    "to weight on every period, or rebuild the dataset with a validation_start."
                )
            labels = labels[:, : int(s) - 1]
        if num_classes is None:
            spec = data.get("input_spec") or {}
            embedded = spec.get("embedded_cols", {}) if isinstance(spec, dict) else {}
            target_col = data.get("target_col")
            if target_col not in embedded:
                raise ValueError(
                    "could not infer num_classes from the data dict (its target "
                    "is not an embedded column); pass num_classes explicitly."
                )
            num_classes = int(embedded[target_col])
        targets = labels
    # Array path: `training_only` has no meaning (no temporal axis is known), so it is
    # simply ignored — slice the labels yourself before calling if you need a subset.

    if num_classes is None:
        raise ValueError(
            "num_classes is required when `targets` is an array of class labels."
        )

    t = torch.as_tensor(targets).flatten().long()
    counts = torch.bincount(t, minlength=num_classes).float().clamp(min=1.0)
    weights = 1.0 / counts
    weights = weights * (num_classes / weights.sum())
    return weights


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_criterion(
    loss_type: str = "cross_entropy",
    *,
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Build a loss module from a string name.

    `class_weights` is consumed by `weighted_ce` and (optionally) `focal`.
    `focal_gamma` is consumed by `focal` only. Other args are ignored
    where they don't apply.
    """
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    if loss_type == "weighted_ce":
        if class_weights is None:
            raise ValueError(
                "loss_type='weighted_ce' requires class_weights "
                "(see compute_class_weights)"
            )
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=focal_gamma)
    if loss_type == "emd":
        return SquaredEMDLoss()
    raise ValueError(
        f"Unknown loss_type={loss_type!r}. "
        f"Options: 'cross_entropy', 'weighted_ce', 'focal', 'emd'"
    )
