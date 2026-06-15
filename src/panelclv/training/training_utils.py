"""Shared training utilities for the multinomial LSTM / Transformer baselines.

Both models output logits with shape (batch, seq_len, max_trans), so the same
loop trains both. The targets are integer transaction-count class labels with
shape (batch, seq_len), and the loss is plain CrossEntropyLoss reshaped to a
flat (batch * seq_len, max_trans) prediction vs (batch * seq_len,) target.

Side concerns kept optional:
    - Weights & Biases logging   (wandb)
    - Optuna pruning             (trial.report / trial.should_prune)
Both are imported lazily so the file runs in a plain Python environment.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

# Losses are part of the model definition and stay in `panelclv.models`, so this
# is now a cross-package (absolute) import after the subpackage split.
from panelclv.models.losses import build_criterion


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    best_val_loss: float
    best_val_f1: float
    best_epoch: int
    checkpoint_path: Path
    history: list[dict[str, float]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_targets(targets: torch.Tensor, max_trans: int) -> None:
    """Sanity-check that targets are integer class labels in [0, max_trans)."""
    if targets.dtype not in (torch.int64, torch.long, torch.int32, torch.int16, torch.int8):
        raise TypeError(
            f"Targets must be integer class labels, got dtype={targets.dtype}"
        )
    t_min = int(targets.min().item())
    t_max = int(targets.max().item())
    if t_min < 0 or t_max >= max_trans:
        raise ValueError(
            f"Targets must be in [0, {max_trans - 1}], "
            f"got min={t_min}, max={t_max}"
        )


def _select_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# ---------------------------------------------------------------------------
# Train / validate one epoch
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_trans: int,
    grad_clip: float | None = 1.0,
    validate_targets: bool = True,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    n_batches = 0

    for samples, targets in loader:
        samples = samples.to(device)
        targets = targets.to(device).long()
        if validate_targets:
            _validate_targets(targets, max_trans)

        optimizer.zero_grad(set_to_none=True)
        output = model(samples)
        # Some training wrappers return (logits, _) — be robust to that.
        if isinstance(output, tuple):
            output = output[0]

        loss = criterion(output.reshape(-1, max_trans), targets.reshape(-1))
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        with torch.no_grad():
            preds = output.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_count += targets.numel()

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": total_correct / max(total_count, 1),
    }


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_trans: int,
    compute_f1: bool = True,
    validate_targets: bool = True,
    val_score_start: int = 0,
) -> dict[str, float]:
    """Teacher-forced validation pass.

    `val_score_start` supports the temporal validation split: the loader feeds each
    customer's FULL calibration sequence (so the recurrent/causal state is warmed up
    over the training prefix), but only the time steps at index >= `val_score_start`
    are scored — i.e. the loss/accuracy/F1 are computed on the validation window only,
    not the training prefix the model was already fit on. `val_score_start=0` (default)
    scores every step, preserving the original behaviour.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    n_batches = 0
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.inference_mode():
        for samples, targets in loader:
            samples = samples.to(device)
            targets = targets.to(device).long()
            if validate_targets:
                _validate_targets(targets, max_trans)

            output = model(samples)
            if isinstance(output, tuple):
                output = output[0]

            # Keep only the validation-window steps (the suffix). The prefix steps
            # were warm-up context for the state, not part of the validation score.
            if val_score_start:
                output = output[:, val_score_start:]
                targets = targets[:, val_score_start:]

            loss = criterion(output.reshape(-1, max_trans), targets.reshape(-1))
            total_loss += loss.item()
            n_batches += 1

            preds = output.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_count += targets.numel()
            if compute_f1:
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": total_correct / max(total_count, 1),
    }
    if compute_f1 and all_preds:
        y_pred = torch.cat(all_preds).numpy().flatten()
        y_true = torch.cat(all_targets).numpy().flatten()
        metrics["f1_weighted"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )
    return metrics


# ---------------------------------------------------------------------------
# Fit (train loop + early stopping + optional pruning / wandb)
# ---------------------------------------------------------------------------


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_trans: int,
    n_epochs: int = 50,
    patience: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    grad_clip: float | None = 1.0,
    device: str | torch.device | None = None,
    checkpoint_dir: str | Path = "./checkpoints",
    model_name: str = "model",
    trial: Any = None,           # optuna.trial.Trial — typed loosely to avoid import
    log_wandb: bool = False,
    verbose: bool = True,
    validate_targets: bool = True,
    loss_type: str = "cross_entropy",       # 'cross_entropy' | 'weighted_ce' | 'focal' | 'emd'
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
    val_score_start: int = 0,
) -> FitResult:
    """Train a multinomial model with early stopping on validation loss.

    The optimisation objective is the configured `loss_type` (default
    cross-entropy). Accuracy and weighted F1 are logged for diagnostics only.

    Loss options
    ------------
    "cross_entropy"  plain CE (default).
    "weighted_ce"    `nn.CrossEntropyLoss(weight=class_weights)`. Requires
                     `class_weights` — typically from `compute_class_weights`.
    "focal"          Focal Loss with optional `class_weights` (as `alpha`)
                     and `focal_gamma` (default 2.0).
    "emd"            Squared Earth Mover's Distance — ordinal-aware.

    If `trial` is provided, the validation loss is reported per epoch via
    `trial.report(...)` and `optuna.TrialPruned` is raised on pruning.

    `val_score_start` is the temporal-validation hook: the val_loader feeds the full
    calibration sequence (warm-up), but only steps >= `val_score_start` are scored, so
    early stopping tracks cross-entropy on the validation window alone. Build the loaders
    with `experiments.make_loaders` (which sets `metadata["val_score_start"] = s-1`) and
    pass that value through here. 0 (default) scores every step.
    """
    device = _select_device(device)
    model = model.to(device)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = build_criterion(
        loss_type,
        class_weights=class_weights,
        focal_gamma=focal_gamma,
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_name}.pth"

    # Optional integrations (lazy import).
    wandb = None
    if log_wandb:
        try:
            import wandb as _wandb  # type: ignore
            wandb = _wandb
        except ImportError:
            if verbose:
                print("wandb not installed; continuing without logging.")
            log_wandb = False

    optuna = None
    if trial is not None:
        try:
            import optuna as _optuna  # type: ignore
            optuna = _optuna
        except ImportError as e:
            raise RuntimeError("optuna is required when `trial` is provided") from e

    best_val_loss = math.inf
    best_val_f1 = 0.0
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0
    history: list[dict[str, float]] = []

    for epoch in range(n_epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            max_trans, grad_clip=grad_clip, validate_targets=validate_targets,
        )
        val_metrics = validate_one_epoch(
            model, val_loader, criterion, device,
            max_trans, validate_targets=validate_targets,
            val_score_start=val_score_start,
        )

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics.get("f1_weighted", float("nan")),
        }
        history.append(record)

        if log_wandb and wandb is not None:
            wandb.log({k: v for k, v in record.items() if k != "epoch"}, step=epoch)

        if verbose:
            print(
                f"Epoch {epoch + 1:>3}/{n_epochs} | "
                f"train_loss={record['train_loss']:.4f} "
                f"val_loss={record['val_loss']:.4f} "
                f"val_acc={record['val_accuracy']:.4f} "
                f"val_f1={record['val_f1']:.4f}"
            )

        # Best-by-loss tracking (primary objective).
        improved = (val_metrics["loss"] + 1e-4) < best_val_loss
        if improved:
            best_val_loss = val_metrics["loss"]
            best_val_f1 = val_metrics.get("f1_weighted", best_val_f1)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        # Optuna pruning hook.
        if trial is not None and optuna is not None:
            trial.report(val_metrics["loss"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}.")
            break

    if best_state is None:
        # No improvement seen; persist whatever we ended with.
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = len(history) - 1

    torch.save(best_state, checkpoint_path)

    if log_wandb and wandb is not None:
        try:
            artifact = wandb.Artifact(name=f"model-{model_name}", type="model")
            artifact.add_file(str(checkpoint_path))
            wandb.log_artifact(artifact)
            wandb.log({"best_val_loss": best_val_loss, "best_val_f1": best_val_f1})
        except Exception:
            pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return FitResult(
        best_val_loss=best_val_loss,
        best_val_f1=best_val_f1,
        best_epoch=best_epoch,
        checkpoint_path=checkpoint_path,
        history=history,
    )


# ---------------------------------------------------------------------------
# Final retrain on the full calibration window (Valendin et al. paper step)
# ---------------------------------------------------------------------------


def refit_full_calibration(
    model: nn.Module,
    train_loader: DataLoader,
    max_trans: int,
    *,
    n_epochs: int,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    grad_clip: float | None = 1.0,
    device: str | torch.device | None = None,
    checkpoint_dir: str | Path = "./checkpoints",
    model_name: str = "model_refit",
    warm_start_state: dict[str, torch.Tensor] | str | Path | None = None,
    loss_type: str = "cross_entropy",
    class_weights: torch.Tensor | None = None,
    focal_gamma: float = 2.0,
    validate_targets: bool = True,
    verbose: bool = True,
) -> FitResult:
    """Warm-start fine-tune on the FULL calibration window — no validation, no early stop.

    Valendin et al. (the *paper*, not their GitHub) describe a final step: after the
    architecture / stopping epoch are chosen on the validation window, retrain the
    selected model for a few epochs with a large batch on the full calibration window so
    the weights also LEARN from the most recent periods (the validation tail), not just
    condition on them at forecast time. This is a **warm-start fine-tune**: pass the
    tuned weights via `warm_start_state` and keep optimising them — do NOT start from
    scratch (a "few epochs" from random init would badly underfit).

    Because the validation window is now folded into training there is nothing left to
    early-stop on, so this trains for exactly `n_epochs` (typically the `best_epoch`
    found by `fit_model`) and persists the FINAL-epoch weights — not a best-by-val
    checkpoint. The returned `FitResult` carries the train history; its `best_val_*`
    fields are NaN (no validation set), and `best_epoch` is the last epoch index.

    `train_loader` must yield the full-calibration AR pairs (all T-1 transitions), unlike
    the temporally-truncated training loader used during tuning — build it with
    `experiments.make_refit_loader`.
    """
    device = _select_device(device)
    model = model.to(device)

    # Warm start: load the tuned weights before fine-tuning. Accept a state_dict or a
    # checkpoint path; drop the Transformer's non-persistent cached-mask key if present
    # (the same guard build_inference_from_trial uses) so a strict load succeeds.
    if warm_start_state is not None:
        if isinstance(warm_start_state, (str, Path)):
            state = torch.load(warm_start_state, map_location=device)
        else:
            state = dict(warm_start_state)
        state.pop("_cached_mask", None)
        model.load_state_dict(state)

    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = build_criterion(
        loss_type, class_weights=class_weights, focal_gamma=focal_gamma,
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_name}.pth"

    history: list[dict[str, float]] = []
    for epoch in range(n_epochs):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            max_trans, grad_clip=grad_clip, validate_targets=validate_targets,
        )
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
        }
        history.append(record)
        if verbose:
            print(
                f"[refit] Epoch {epoch + 1:>3}/{n_epochs} | "
                f"train_loss={record['train_loss']:.4f} "
                f"train_acc={record['train_accuracy']:.4f}"
            )

    # Persist the FINAL weights (there is no validation set to pick a "best" epoch).
    torch.save(model.state_dict(), checkpoint_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return FitResult(
        best_val_loss=float("nan"),
        best_val_f1=float("nan"),
        best_epoch=max(n_epochs - 1, 0),
        checkpoint_path=checkpoint_path,
        history=history,
    )
