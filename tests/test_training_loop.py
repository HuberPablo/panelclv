"""`fit_model` returns the weights it selected, not the ones it stopped on.

The loop snapshots the best-by-validation state, then keeps training past that epoch
— that is what `patience` is for. It saves the snapshot to disk; the property tested
here is that it also puts it back into the model object before returning.

This has its own file because the golden end-to-end fixture cannot catch it. Two
epochs improving monotonically make the checkpoint and the object agree by luck, and
a two-epoch run cannot early-stop at all, so the two channels are never forced apart
there. They are forced apart here.

Why it matters: until `to_rollout()` existed, the checkpoint was the only channel out
of training, so nobody could observe the difference. `to_rollout()` reads the object
(ADR-0007), which is the other end — and last-epoch weights would produce a forecast
that is quietly wrong rather than loudly broken.
"""

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from panelclv.models.embedders import ProjectedEmbedder  # noqa: E402
from panelclv.models.multinomial_lstm import MultinomialLSTMModel  # noqa: E402
from panelclv.training import fit_model  # noqa: E402

N_CUSTOMERS = 8
N_STEPS = 6
N_CLASSES = 3
TORCH_SEED = 4242


def _loader(target_class: int) -> DataLoader:
    """A batch whose input and target are one constant count class throughout.

    Shapes mirror the real training pairs: samples (N, T, 1) float — the target's own
    column is the only feature — and targets (N, T) integer class indices.
    """
    samples = torch.full((N_CUSTOMERS, N_STEPS, 1), float(target_class))
    targets = torch.full((N_CUSTOMERS, N_STEPS), target_class, dtype=torch.long)
    return DataLoader(TensorDataset(samples, targets), batch_size=4, shuffle=False)


def _fit_past_the_best_epoch(tmp_path):
    """Train on one class while validating on another, so validation only worsens.

    Every gradient step makes the model more certain of the training class, which is
    the *wrong* answer on the validation set — so epoch 0 is the best epoch and every
    later one is worse. Early stopping therefore fires after `patience` further
    epochs, and the model is left holding weights that are not the ones selected.
    """
    torch.manual_seed(TORCH_SEED)
    model = MultinomialLSTMModel(
        embedder=ProjectedEmbedder(
            seq_cols=["Transactions"],
            embedded_cols={"Transactions": N_CLASSES},
            target_col="Transactions",
            embedding_dim=4,
        ),
        lstm_hidden_size=4,
        dense_units=4,
        dropout=0.0,
    )
    result = fit_model(
        model,
        _loader(target_class=0),
        _loader(target_class=1),
        max_trans=N_CLASSES,
        n_epochs=12,
        patience=2,
        learning_rate=0.1,     # large enough that one epoch moves the weights visibly
        device="cpu",
        checkpoint_dir=str(tmp_path),
        model_name="best_state",
        verbose=False,
    )
    return model, result


def test_the_fixture_really_stops_after_the_best_epoch(tmp_path):
    """Guards the test below from passing vacuously.

    If the run happened to end on its best epoch, the two channels would agree for
    the reason this file exists to rule out, and the state-dict comparison would
    prove nothing.
    """
    _, result = _fit_past_the_best_epoch(tmp_path)

    assert result.best_epoch == 0
    assert len(result.history) > 1, "no epoch trained past the best one"


def test_returned_model_holds_the_best_weights_not_the_last(tmp_path):
    """In-memory weights == on-disk checkpoint, tensor for tensor.

    `to_rollout()` reads the object; every other consumer reads the file. They are
    the same weights or the package has two answers to one question.
    """
    model, result = _fit_past_the_best_epoch(tmp_path)

    checkpoint = torch.load(result.checkpoint_path, map_location="cpu")
    in_memory = model.state_dict()

    assert set(in_memory) == set(checkpoint)
    for name, saved in checkpoint.items():
        assert torch.equal(in_memory[name].cpu(), saved), f"{name} is not the best state"


def test_the_rollout_model_forecasts_with_the_selected_weights(tmp_path):
    """The end-to-end consequence: the handover carries the selected weights out.

    `to_rollout()` shares the trained backbone, so this is the same assertion one
    level up — stated at the level a forecast actually reads.
    """
    model, result = _fit_past_the_best_epoch(tmp_path)

    rollout = model.to_rollout()
    checkpoint = torch.load(result.checkpoint_path, map_location="cpu")

    for name, saved in checkpoint.items():
        # The rollout wrapper nests the shared backbone under the same prefix the
        # trained model does, so the key names line up without translation.
        assert torch.equal(rollout.state_dict()[name].cpu(), saved)
